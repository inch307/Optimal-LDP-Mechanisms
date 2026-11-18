import time
import cvxpy as cp
import numpy as np
from .grid import validate_grid
from .patterns import full_patterns, initial_patterns, pattern_matrix, separating_patterns
from .primal_b import solve_problem


def solve_B_dual(epsilon, grid, pattern_mode="full", max_cuts=60, time_limit=120., progress=None):
    x = validate_grid(grid)
    if epsilon <= 0 or pattern_mode not in ("full", "cuts"):
        raise ValueError("positive epsilon and full/cuts mode required")
    start = time.perf_counter()
    scale = 1 / np.tanh(epsilon/2)
    u = x / scale
    bits = full_patterns(len(x)) if pattern_mode == "full" else initial_patterns(len(x))
    history = []
    result = None
    for iteration in range(max_cuts if pattern_mode == "cuts" else 1):
        solved_bits = bits.copy()
        S = pattern_matrix(solved_bits, epsilon)
        alpha, beta, gamma = cp.Variable(len(x)), cp.Variable(len(x)), cp.Variable(len(x))
        A, B, C = S.T@gamma, S.T@beta, S.T@alpha
        problem = cp.Problem(cp.Maximize(cp.sum(alpha)+u@beta-(u*u)@gamma),
                             [gamma >= 0, cp.sum(gamma) == 1,
                              cp.SOC(A-C, cp.vstack([B, -C-A]), axis=0)])
        remaining = time_limit-(time.perf_counter()-start)
        if remaining <= 0 and result is not None:
            break
        solve_problem(problem, max(.01, remaining))
        if any(v.value is None for v in (alpha, beta, gamma)):
            raise RuntimeError("dual solver returned no candidate")
        aa, bb, gg = alpha.value, beta.value, gamma.value
        candidate = separating_patterns(aa, bb, gg)
        T = pattern_matrix(candidate, epsilon)
        va, vb, vc = T.T@gg, T.T@bb, T.T@aa
        slack = -vc - vb*vb/(4*np.maximum(va, 1e-30))
        history.append(dict(iteration=iteration, patterns=len(solved_bits), min_slack=float(min(slack))))
        result = dict(alpha=aa*scale**2, beta=bb*scale, gamma=gg,
                      L_float=float(problem.value*scale**2), solver_status=problem.status,
                      bits=solved_bits, search_complete=bool(min(slack) >= -1e-9), history=history)
        if progress:
            progress(f"dual cuts={iteration+1}, patterns={len(solved_bits)}, violation={max(0.,-min(slack)):.2g}")
        if pattern_mode == "full" or min(slack) >= -1e-9:
            break
        old = {tuple(row) for row in solved_bits}
        expanded = old | {tuple(row) for row, s in zip(candidate, slack) if s < -1e-10}
        if expanded == old:
            break
        bits = np.array(sorted(expanded), dtype=np.int8)
    result.update(runtime=time.perf_counter()-start, pattern_mode=pattern_mode)
    return result
