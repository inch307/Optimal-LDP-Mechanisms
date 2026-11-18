"""Primal retries selected by EXACT certified upper bounds, not solver status.

Capture native solver diagnostics before CVXPY collapses numerical failures into
SolverError. Each attempt gets part of a single shared time budget. An accepted
channel is checkpointed before trying another CLARABEL configuration.
"""
from fractions import Fraction as F
import time

import cvxpy as cp
import numpy as np

from .primal_b import build_B_primal, extract_primal
from .channel import from_moments, merge_duplicate_outputs, verify_grid_channel
from .certify_continuous import certify_continuous
from .exact import save_json


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def run_solver_attempt(problem, solver, options, seconds, record, persist, verbose=True):
    """Separate native solve from unpacking so a NumericalError is not lost."""
    if solver != 'CLARABEL':
        raise ValueError('Only CLARABEL is enabled; SCS retries have been removed')
    tick = time.perf_counter()
    record.update(state="canonicalizing", solver=solver)
    persist()
    try:
        data, chain, inverse = problem.get_problem_data(solver)
        record.update(canonical_seconds=time.perf_counter()-tick,
                      matrix_shape=list(data['A'].shape), matrix_nnz=int(data['A'].nnz),
                      cone_dimensions=str(data['dims']))
        remaining = seconds-(time.perf_counter()-tick)
        if remaining <= 0:
            record.update(state="budget_exhausted_before_solve")
            persist()
            return False
        actual = dict(options)
        actual['time_limit'] = remaining
        record.update(state="solving", options=actual)
        persist()
        # Avoid stale values from a preceding successful attempt on this model.
        for variable in problem.variables():
            variable.value = None
        raw = chain.solve_via_data(problem, data, warm_start=False, verbose=verbose, solver_opts=actual)
        record['native'] = {k:getattr(raw,k,None) for k in
                            ('iterations','obj_val','obj_val_dual','r_prim','r_dual','solve_time')}
        record['native']['status'] = str(raw.status)
        record['state'] = 'native_result_saved'
        persist()  # save BEFORE unpack_results can raise SolverError
        problem.unpack_results(raw,chain,inverse)
        record.update(state="unpacked", cvxpy_status=problem.status)
        return problem.status in ('optimal','optimal_inaccurate','user_limit')
    except (cp.error.SolverError, ValueError, RuntimeError, TypeError, ImportError) as exc:
        record.update(state="solver_failed", error_type=type(exc).__name__, error=str(exc))
        return False
    finally:
        record['attempt_wall_seconds'] = time.perf_counter()-tick
        persist()


def solve_B_primal_robust(epsilon, grid, bits, time_limit=14400., symmetry=True,
                         lower_bound=None, log_path=None, checkpoint=None,
                         progress=None, verbose=True, grid_gap_tolerance=1e-7):
    if time_limit <= 0 or grid_gap_tolerance < 0:
        raise ValueError('invalid primal retry budget/tolerance')
    start = time.perf_counter()
    journal = dict(kind='primal_retries', epsilon=epsilon, M=len(grid),
                   time_budget_seconds=time_limit, attempts=[], best_U_cont=None,
                   solver_warm_start=False, stop_reason='attempts_exhausted')
    def persist():
        journal['elapsed_seconds'] = time.perf_counter()-start
        if log_path:
            save_json(log_path,safe_json(journal))
    persist()
    problem, model = build_B_primal(epsilon,grid,'cuts',bits,symmetry)
    journal['model_dimensions'] = model['dimensions']
    journal['independent_variables'] = sum(v.size for v in problem.variables())
    persist()
    stages = [
        ('clarabel_coarse','CLARABEL',dict(max_iter=1000,tol_gap_abs=1e-7,tol_gap_rel=1e-7,tol_feas=1e-7)),
        ('clarabel_fine','CLARABEL',dict(max_iter=1000,tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9)),
        ('clarabel_regularized','CLARABEL',dict(max_iter=1000,tol_gap_abs=1e-8,tol_gap_rel=1e-8,tol_feas=1e-8,
            equilibrate_max_iter=30, static_regularization_constant=1e-6,
            iterative_refinement_max_iter=20, max_step_fraction=.95)),
    ]
    best, best_value = None, None
    for i,(name,solver,options) in enumerate(stages):
        remaining = time_limit-(time.perf_counter()-start)
        if remaining < .05:
            journal['stop_reason'] = 'time_budget'
            break
        record = dict(name=name,solver=solver)
        journal['attempts'].append(record)
        if solver not in cp.installed_solvers():
            record.update(state='solver_unavailable'); persist(); continue
        # Share the budget across the remaining CLARABEL configurations.
        allowance = remaining/(len(stages)-i)
        if progress:
            progress(f'primal {name}, allowance={allowance:.0f}s, best U={journal["best_U_cont"]}')
        usable = run_solver_attempt(problem,solver,options,allowance,record,persist,verbose)
        if not usable:
            continue
        try:
            candidate = extract_primal(problem,model,time.perf_counter()-start)
            raw = merge_duplicate_outputs(from_moments(epsilon,grid,candidate))
            record['raw_diagnostics'] = verify_grid_channel(epsilon,grid,raw.outputs,raw.P_grid)
            record['state'] = 'certifying'; persist()
            tick = time.perf_counter()
            cert = certify_continuous(raw,epsilon,symmetry=symmetry)
            record.update(certificate_seconds=time.perf_counter()-tick, state='certified',
                          U_grid=cert['U_grid_cert'],U_cont=cert['U_cont_cert'])
            if lower_bound is not None and F(cert['U_grid_rational']) < F(str(lower_bound)):
                raise ArithmeticError('certified upper below supplied certified lower')
            value = F(cert['U_cont_rational'])
            if best is None or value < best_value:
                best, best_value = candidate, value
                best.update(certificate=cert,selected_attempt=name,solver=solver)
                journal.update(best_U_cont=cert['U_cont_cert'],selected_attempt=name)
                if checkpoint:
                    checkpoint(best)
            persist()
            if lower_bound is not None:
                gap = F(cert['U_grid_rational'])-F(str(lower_bound))
                if gap <= F(str(grid_gap_tolerance))*max(F(1),abs(F(str(lower_bound)))):
                    journal['stop_reason'] = 'certified_grid_gap'
                    break
        except ValueError as exc:
            record.update(state='candidate_rejected',certificate_error=str(exc)); persist()
    persist()
    if best is None:
        raise RuntimeError('all primal attempts failed to produce an exactly certified channel; see primal_attempts.json')
    best.update(runtime=time.perf_counter()-start,attempts=safe_json(journal))
    return best
