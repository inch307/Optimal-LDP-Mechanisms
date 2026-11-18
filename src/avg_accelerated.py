"""Prior-specific AVG search. Fixed weights, never a worst-case gamma variable.

Only patterns transfer between changing quadrature/coarsening grids. A previous
dual is NOT lifted as a certificate when weights or bin means change.
"""
from fractions import Fraction as F
import json
from pathlib import Path
import time

import cvxpy as cp
import numpy as np

from .acceleration import (canonical_patterns, close_reflections, manage_patterns,
                           symmetric_dual, symmetric_moments)
from .avg_solver import certify_avg_dual, verify_avg_dual
from .patterns import full_patterns, initial_patterns, pattern_matrix, separating_patterns
from .primal_robust import run_solver_attempt, safe_json
from .channel import from_moments, merge_duplicate_outputs
from .certify_continuous import certify_continuous, verify_continuous
from .exact import save_json


def reflection_allowed(grid, weights):
    x, p = [list(map(lambda v: F(str(v)), seq)) for seq in (grid, weights)]
    return x == [-v for v in x[::-1]] and p == p[::-1]


def transfer_patterns(path, grid):
    """Nearest-neighbour pattern extension is a search heuristic, not a bound."""
    with np.load(path, allow_pickle=False) as d:
        old, bits = d['grid'], d['bits']
        if (old.ndim != 1 or len(old) < 2 or np.any(np.diff(old) <= 0)
                or not np.all(np.isfinite(old)) or bits.ndim != 2
                or bits.shape[1] != len(old) or not np.all((bits == 0) | (bits == 1))):
            raise ValueError('invalid AVG pattern seed')
        # O(M) storage rather than an M_new x M_old distance matrix.
        x = np.asarray(grid, float)
        right = np.clip(np.searchsorted(old, x), 1, len(old)-1)
        near = np.where(abs(x-old[right-1]) <= abs(x-old[right]), right-1, right)
        return np.asarray(bits[:, near], dtype=np.int8)


def solve_avg_dual_accelerated(epsilon, grid, weights, second_moment,
        max_cuts=1000, time_limit=14400., pattern_mode='auto', symmetry=True,
        seed_patterns=None, folder=None, progress=None, checkpoint=None):
    x, pi = np.asarray(grid, float), np.asarray(weights, float)
    if time_limit <= 0 or max_cuts < 1:
        raise ValueError('positive time/cut budgets required')
    symmetry = symmetry and reflection_allowed(grid, weights)
    canonical = canonical_patterns if symmetry else lambda b: np.unique(b, axis=0)
    mode = ('full' if len(x) <= 9 else 'cuts') if pattern_mode == 'auto' else pattern_mode
    if mode not in ('full', 'cuts'):
        raise ValueError('invalid pattern mode')
    core = canonical(full_patterns(len(x)) if mode == 'full' else initial_patterns(len(x)))
    bits = core
    if seed_patterns is not None and mode != 'full':
        bits = canonical(np.vstack([bits, seed_patterns]))
    protected = {r.tobytes() for r in core}
    ages, history = {}, []
    scale = 1/np.tanh(epsilon/2)
    u = x/scale
    start = time.perf_counter()
    best, best_value = None, None
    journal = dict(kind='avg_dual_search', symmetry=symmetry, M=len(x),
                   fixed_weights=True, solver_warm_start=False, history=history)
    folder = Path(folder) if folder is not None else None
    if folder:
        folder.mkdir(parents=True, exist_ok=True)

    def persist():
        journal['elapsed_seconds'] = time.perf_counter()-start
        if folder:
            save_json(folder/'search.json', safe_json(journal))

    def consider(aa, bb, pool, status):
        nonlocal best, best_value
        cert = certify_avg_dual(epsilon, grid, weights, second_moment, aa, bb)
        value = F(cert['L_rational'])
        if best is None or value > best_value:
            best, best_value = dict(certificate=cert, solver_status=status), value
            journal['best_L_cert'] = cert['L_cert']
            if folder:
                save_json(folder/'dual_cert.json', cert)
            if checkpoint:
                checkpoint(cert)
        return cert['L_cert']

    if folder and (folder/'dual_cert.json').exists():
        saved = json.loads((folder/'dual_cert.json').read_text(encoding='utf-8'))
        verify_avg_dual(saved)
        if (F(saved['epsilon']) != F(str(epsilon)) or saved['grid'] != list(map(str, grid))
                or saved['weights'] != list(map(str, weights))
                or F(saved['second_moment']) != F(second_moment)):
            raise ValueError('AVG checkpoint objective mismatch')
        best = dict(certificate=saved, solver_status='certified_checkpoint')
        best_value = F(saved['L_rational'])
        journal['best_L_cert'] = saved['L_cert']
        if checkpoint:
            checkpoint(saved)
    # Sum pi_j*s_j*(y-x_j)^2 >= 0 gives the safe baseline -D for coarsening.
    consider(-pi*x*x, 2*pi*x, core, 'fallback')
    if folder and (folder/'patterns.npz').exists():
        bits = canonical(np.vstack([bits, transfer_patterns(folder/'patterns.npz', x)]))
    # Saved certificates from a prior interrupted run are usable only for the
    # identical objective, including exact weights and original second moment.
    # The caller stores each configuration under its own source/config hash.
    stage, failures = 0, 0
    tolerances = (1e-6, 1e-8, 1e-10)
    stop = 'max_cuts'
    for iteration in range(max_cuts):
        remaining = time_limit-(time.perf_counter()-start)
        if remaining < .05:
            stop = 'time_budget'; break
        S = pattern_matrix(bits, epsilon)
        if symmetry:
            alpha, beta, _ = symmetric_dual(len(x))
        else:
            alpha, beta = cp.Variable(len(x)), cp.Variable(len(x))
        A, B, C = S.T@pi, S.T@beta, S.T@alpha
        problem = cp.Problem(cp.Maximize(cp.sum(alpha)+u@beta),
                             [cp.SOC(A-C, cp.vstack([B, -C-A]), axis=0)])
        record = dict(iteration=iteration+1, patterns=len(bits), tolerance=tolerances[stage])
        history.append(record)
        options = dict(max_iter=1000, tol_gap_abs=tolerances[stage],
                       tol_gap_rel=tolerances[stage], tol_feas=tolerances[stage])
        if failures:
            options.update(static_regularization_constant=1e-6, equilibrate_max_iter=30,
                           iterative_refinement_max_iter=20, max_step_fraction=.95)
        usable = run_solver_attempt(problem, 'CLARABEL', options, remaining/2,
                                    record, persist, False)
        if not usable or any(v.value is None for v in (alpha, beta)):
            failures += 1
            if failures >= 2:
                stop = 'solver_failure_incumbent_retained'; break
            continue
        aa, bb = np.asarray(alpha.value).copy(), np.asarray(beta.value).copy()
        if not all(np.all(np.isfinite(v)) for v in (aa, bb)):
            stop = 'nonfinite_incumbent_retained'; break
        failures = 0
        candidates = canonical(separating_patterns(aa, bb, pi))
        T = pattern_matrix(candidates, epsilon)
        slack = -(T.T@aa)-(T.T@bb)**2/(4*(T.T@pi))
        minimum = float(min(slack))
        try:
            record['candidate_L_cert'] = consider(aa*scale**2, bb*scale, bits, problem.status)
        except ValueError as exc:
            record['certificate_error'] = str(exc)
        added = candidates[slack < -max(1e-10, tolerances[stage]*.1)]
        existing_slack = -(S.T@aa)-(S.T@bb)**2/(4*A)
        next_bits, dropped = manage_patterns(bits, existing_slack, added, ages,
                                             protected, 0 if mode == 'full' else max(1024, 4*len(x)))
        record.update(min_slack=minimum, dropped=dropped, best_L_cert=best['certificate']['L_cert'])
        if progress:
            progress(f'AVG dual cuts={iteration+1} M={len(x)} symmetry={symmetry} '
                     f'patterns={len(bits)} violation={max(0.,-minimum):.3g} L={journal["best_L_cert"]:.10g}')
        persist()
        if folder:
            tmp = folder/'patterns.tmp.npz'
            np.savez_compressed(tmp, grid=x, bits=close_reflections(next_bits) if symmetry else next_bits)
            tmp.replace(folder/'patterns.npz')
        if stage == 2 and minimum >= -1e-9 and problem.status != 'user_limit':
            stop = 'separation_complete'; bits = next_bits; break
        if problem.status == 'user_limit':
            stop = 'solver_limit_incumbent_retained'; bits = next_bits; break
        unchanged = np.array_equal(next_bits, bits)
        if stage < 2 and (minimum >= (-1e-4 if stage == 0 else -1e-6) or unchanged):
            stage += 1
        elif unchanged and stage == 2:
            stop = 'working_set_stagnation'; bits = next_bits; break
        bits = next_bits
    journal.update(stop_reason=stop, search_complete=stop == 'separation_complete')
    persist()
    best.update(bits=close_reflections(bits) if symmetry else bits, symmetry=symmetry,
                search_complete=journal['search_complete'], history=safe_json(history),
                pattern_mode=mode, runtime_seconds=time.perf_counter()-start,
                stop_reason=stop, dual_status=best['solver_status'])
    return best


def build_avg_primal(epsilon, grid, weights, bits, symmetry=True):
    """Weighted second moment, no tau or pointwise variance epigraph."""
    x, pi = np.asarray(grid, float), np.asarray(weights, float)
    symmetry = symmetry and reflection_allowed(grid, weights)
    if symmetry:
        bits = close_reflections(bits)
    S = pattern_matrix(bits, epsilon)
    scale = 1/np.tanh(epsilon/2)
    k = len(bits)
    w, z, r = symmetric_moments(bits) if symmetry else (cp.Variable(k), cp.Variable(k), cp.Variable(k))
    if symmetry:
        even, odd = np.flatnonzero(x >= 0), np.flatnonzero(x > 0)
        reps = [j for j, b in enumerate(bits) if b.tobytes() <= b[::-1].tobytes()]
        paired = [j for j in reps if not np.array_equal(bits[j], bits[j, ::-1])]
        fixed = [j for j in reps if np.array_equal(bits[j], bits[j, ::-1])]
        constraints = [S[even]@w == 1, S[odd]@z == x[odd]/scale]
        if paired:
            constraints.append(cp.SOC(w[paired]+r[paired], cp.vstack([2*z[paired], w[paired]-r[paired]]), axis=0))
        if fixed:
            constraints += [w[fixed] >= 0, r[fixed] >= 0]
    else:
        constraints = [S@w == 1, S@z == x/scale,
                       cp.SOC(w+r, cp.vstack([2*z, w-r]), axis=0)]
    problem = cp.Problem(cp.Minimize((S.T@pi)@r), constraints)
    return problem, dict(w=w, z=z, r=r, bits=bits, scale=scale, symmetry=symmetry)


def solve_avg_primal_robust(epsilon, grid, weights, prior, bits, lower_bound,
        time_limit=14400., symmetry=True, folder=None, progress=None):
    from .avg_experiments import avg_upper
    start = time.perf_counter()
    folder = Path(folder) if folder is not None else None
    problem, model = build_avg_primal(epsilon, grid, weights, bits, symmetry)
    journal = dict(kind='avg_primal_retries', objective='prior_average_variance',
                   symmetry=model['symmetry'], M=len(grid), attempts=[],
                   stop_reason='attempts_exhausted', solver_warm_start=False)
    def persist():
        journal['elapsed_seconds'] = time.perf_counter()-start
        if folder:
            save_json(folder/'primal_attempts.json', safe_json(journal))
    stages = [
        ('clarabel_coarse', 'CLARABEL', dict(max_iter=1000,tol_gap_abs=1e-7,tol_gap_rel=1e-7,tol_feas=1e-7)),
        ('clarabel_fine', 'CLARABEL', dict(max_iter=1000,tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9)),
        ('clarabel_regularized', 'CLARABEL', dict(max_iter=1000,tol_gap_abs=1e-8,tol_gap_rel=1e-8,tol_feas=1e-8,
            equilibrate_max_iter=30,static_regularization_constant=1e-6,iterative_refinement_max_iter=20,max_step_fraction=.95)),
    ]
    best, value = None, None
    if folder and (folder/'channel_cert.json').exists():
        saved = json.loads((folder/'channel_cert.json').read_text(encoding='utf-8'))
        verify_continuous(saved)
        if F(saved['epsilon']) != F(str(epsilon)) or [F(v) for v in saved['grid']] != list(grid):
            raise ValueError('AVG channel checkpoint mismatch')
        best, value = saved, avg_upper(prior, saved)
        journal.update(selected_attempt='certified_checkpoint', best_U_avg=float(value))
    for i, (name, solver, options) in enumerate(stages):
        remaining = time_limit-(time.perf_counter()-start)
        if remaining < .05:
            journal['stop_reason'] = 'time_budget'; break
        record = dict(name=name)
        journal['attempts'].append(record)
        if progress:
            progress(f'AVG primal {name}, M={len(grid)}, symmetry={model["symmetry"]}')
        if not run_solver_attempt(problem, solver, options, remaining/(len(stages)-i), record, persist, False):
            continue
        try:
            if any(model[k].value is None or not np.all(np.isfinite(model[k].value)) for k in ('w','z','r')):
                raise ValueError('missing/nonfinite primal moments')
            candidate = dict(w=model['w'].value.copy(), z=model['z'].value.copy()*model['scale'], bits=model['bits'])
            raw = merge_duplicate_outputs(from_moments(epsilon, np.asarray(grid, float), candidate))
            tick = time.perf_counter()
            cert = certify_continuous(raw, epsilon, symmetry=model['symmetry'])
            U = avg_upper(prior, cert)
            # This lower is for the SAME hat-weight finite-input objective,
            # not the continuous-prior coarsening lower used in the final report.
            if U < F(lower_bound):
                raise ArithmeticError('AVG upper below matching finite-objective lower')
            record.update(state='certified', U_avg=float(U), certificate_seconds=time.perf_counter()-tick)
            if best is None or U < value:
                best, value = cert, U
                journal.update(selected_attempt=name, best_U_avg=float(U))
                if folder:
                    save_json(folder/'channel_cert.json', cert)
            if U-F(lower_bound) <= F('1e-7')*max(F(1), abs(F(lower_bound))):
                journal['stop_reason'] = 'certified_finite_avg_gap'; break
        except ValueError as exc:
            record.update(state='candidate_rejected', error=str(exc))
        finally:
            persist()
    if best is None:
        raise RuntimeError('AVG primal attempts failed; certified dual/checkpoints retained')
    return best, safe_json(journal)
