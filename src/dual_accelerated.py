"""Certified-incumbent cutting planes with reflection reduction and seed reuse.

No solver warm-start claim: CVXPY/Clarabel problems are rebuilt after a working
set change. Reuse here is of certified coefficients and useful pattern rows.
"""
import time
from fractions import Fraction

import cvxpy as cp
import numpy as np

from .acceleration import (canonical_patterns, close_reflections, lift_seed,
                           manage_patterns, require_symmetric_grid, symmetric_dual)
from .certify_dual import certify_dual
from .grid import validate_grid
from .patterns import initial_patterns, pattern_matrix, separating_patterns
from .primal_b import solve_problem


def pattern_slack(bits, epsilon, alpha, beta, gamma):
    S = pattern_matrix(bits, epsilon)
    A, B, C = S.T@gamma, S.T@beta, S.T@alpha
    return -C-B*B/(4*np.maximum(A, 1e-30))


def solve_B_dual_accelerated(epsilon, grid, max_cuts=1000, time_limit=14400.,
                            progress=None, seed_path=None, symmetry=True,
                            pattern_limit=None, staged=True, checkpoint=None):
    x = validate_grid(grid)
    if epsilon <= 0 or time_limit <= 0 or max_cuts < 1:
        raise ValueError("positive epsilon, time and cut budget required")
    if symmetry:
        require_symmetric_grid(x)
    start = time.perf_counter()
    scale = 1/np.tanh(epsilon/2)
    u = x/scale
    canonical = canonical_patterns if symmetry else lambda b: np.unique(b, axis=0)
    core = canonical(initial_patterns(len(x)))
    protected = {row.tobytes() for row in core}
    limit = max(1024, 4*len(x)) if pattern_limit is None else pattern_limit
    if limit < 0:
        raise ValueError("pattern limit must be nonnegative; zero disables pruning")
    bits = core
    history, ages = [], {}
    best = None
    best_value = None
    last_solve = 0.
    stage = 0 if staged else 2
    tolerances = (1e-6, 1e-8, 1e-10)
    stop_reason = "max_cuts"
    seed_info = None

    def consider(aa, bb, gg, candidate_bits, status, complete, raw_value, iteration):
        nonlocal best, best_value
        if not all(np.all(np.isfinite(v)) for v in (aa, bb, gg)):
            return None
        cert = certify_dual(epsilon, x, aa, bb, gg, method="interval")
        value = Fraction(cert["L_rational"])
        if best is None or value > best_value:
            best_value = value
            best = {k: np.array([float(Fraction(v)) for v in cert[k]]) for k in ("alpha", "beta", "gamma")}
            best.update(bits=close_reflections(candidate_bits) if symmetry else candidate_bits.copy(),
                        certificate=cert, solver_status=status, search_complete=complete,
                        L_float=float(raw_value), L_cert=cert["L_cert"], selected_iteration=iteration)
            if checkpoint:
                checkpoint(best)
        return cert["L_cert"]

    # A valid emergency lower bound, also saved before any expensive solve.
    gamma0 = np.zeros(len(x)); gamma0[np.argmin(abs(x))] = 1.
    consider(np.zeros(len(x)), np.zeros(len(x)), gamma0, core, "fallback", False,
             -float((x*x)@gamma0), -2)
    if seed_path:
        seed = lift_seed(seed_path, epsilon, x)
        seed_info = {k: seed[k] for k in ("source", "source_M", "source_L")}
        bits = canonical(seed["bits"])
        consider(seed["alpha"], seed["beta"], seed["gamma"], bits, "certified_seed", False,
                 seed["source_L"], -1)
        seed_slack = pattern_slack(bits, epsilon, seed["alpha"]/scale**2,
                                   seed["beta"]/scale, seed["gamma"])
        ages = {row.tobytes(): 3 for row in bits}
        bits, _ = manage_patterns(bits, seed_slack, core, ages, protected, limit)
        if progress:
            progress(f"reused certified seed M={seed['source_M']}, L={seed['source_L']:.9g}; patterns={len(bits)}")

    for iteration in range(max_cuts):
        remaining = time_limit-(time.perf_counter()-start)
        # Avoid starting another full SOCP with only a few seconds left.
        if remaining < max(.05, .5*last_solve):
            stop_reason = "time_reserve"
            break
        solved_bits = bits.copy()
        S = pattern_matrix(solved_bits, epsilon)
        alpha, beta, gamma = symmetric_dual(len(x)) if symmetry else (cp.Variable(len(x)), cp.Variable(len(x)), cp.Variable(len(x)))
        A, B, C = S.T@gamma, S.T@beta, S.T@alpha
        problem = cp.Problem(cp.Maximize(cp.sum(alpha)+u@beta-(u*u)@gamma),
                             [gamma >= 0, cp.sum(gamma) == 1,
                              cp.SOC(A-C, cp.vstack([B, -C-A]), axis=0)])
        remaining = time_limit-(time.perf_counter()-start)
        if remaining <= 0:
            stop_reason = "time_limit"
            break
        tick = time.perf_counter()
        try:
            solve_problem(problem, remaining, tolerance=tolerances[stage], max_iter=1000)
            if any(v.value is None for v in (alpha, beta, gamma)):
                raise ValueError("missing dual candidate")
            aa, bb, gg = (np.asarray(v.value).copy() for v in (alpha, beta, gamma))
            if not all(np.all(np.isfinite(v)) for v in (aa, bb, gg)):
                raise ValueError("nonfinite dual candidate")
        except (cp.error.SolverError, ValueError, RuntimeError) as exc:
            history.append(dict(iteration=iteration, error=str(exc), patterns=len(solved_bits)))
            stop_reason = "solver_failure_incumbent_retained"
            break
        last_solve = time.perf_counter()-tick
        candidate = canonical(separating_patterns(aa, bb, gg))
        slack = pattern_slack(candidate, epsilon, aa, bb, gg)
        minimum = float(min(slack))
        complete = stage == 2 and minimum >= -1e-9 and problem.status != "user_limit"
        cert_value = None
        try:
            cert_value = consider(aa*scale**2, bb*scale, gg, solved_bits,
                                  problem.status, complete, problem.value*scale**2, iteration)
        except ValueError as exc:
            # Bad candidates cannot replace an independently verified incumbent.
            history.append(dict(iteration=iteration, certificate_error=str(exc)))
        added = candidate[slack < -max(1e-10, tolerances[stage]*.1)]
        existing_slack = pattern_slack(solved_bits, epsilon, aa, bb, gg)
        next_bits, dropped = manage_patterns(solved_bits, existing_slack, added, ages, protected, limit)
        history.append(dict(iteration=iteration, patterns=len(solved_bits), min_slack=minimum,
                            tolerance=tolerances[stage], solver_status=problem.status,
                            solve_seconds=last_solve, candidate_L_cert=cert_value,
                            best_L_cert=best["L_cert"], added=len(added), dropped=dropped))
        if progress:
            progress(f"dual cuts={iteration+1}, patterns={len(solved_bits)}, tol={tolerances[stage]:g}, "
                     f"violation={max(0.,-minimum):.2g}, best L={best['L_cert']:.9g}, dropped={dropped}")
        if complete:
            stop_reason = "separation_complete"
            break
        if problem.status == "user_limit":
            stop_reason = "solver_limit_incumbent_retained"
            break
        if stage < 2 and minimum >= (-1e-4 if stage == 0 else -1e-6):
            stage += 1
        elif np.array_equal(next_bits, solved_bits):
            if stage < 2:
                stage += 1
            else:
                stop_reason = "working_set_stagnation"
                break
        bits = next_bits
    best.update(runtime=time.perf_counter()-start, pattern_mode="cuts", history=history,
                symmetry=symmetry, independent_dual_variables=(3*len(x)+1)//2 if symmetry else 3*len(x),
                stop_reason=stop_reason, seed=seed_info, pattern_limit=limit,
                staged_precision=staged, solver_warm_start=False)
    return best
