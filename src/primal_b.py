import time
import cvxpy as cp
import numpy as np
from .grid import validate_grid
from .patterns import full_patterns, pattern_matrix


def solve_problem(problem, time_limit=120., tolerance=1e-10, max_iter=300):
    problem.solve(solver="CLARABEL", verbose=False, max_iter=max_iter,
                  tol_gap_abs=tolerance, tol_gap_rel=tolerance, tol_feas=tolerance,
                  time_limit=max(.01, time_limit))
    if problem.status not in ("optimal", "optimal_inaccurate", "user_limit") or problem.value is None:
        raise RuntimeError(f"conic solver status: {problem.status}")


def build_B_primal(epsilon, grid, pattern_mode="full", bits=None, symmetry=False, reduce_symmetry=True):
    x = validate_grid(grid)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if bits is None:
        if pattern_mode != "full":
            raise ValueError("restricted primal requires explicit bits")
        bits = full_patterns(len(x))
    if symmetry:
        from .acceleration import symmetric_moments, close_reflections, require_symmetric_grid
        require_symmetric_grid(x)
        bits = close_reflections(bits)
    S = pattern_matrix(bits, epsilon)
    scale = 1 / np.tanh(epsilon/2)
    u = x / scale
    k = S.shape[1]
    w, z, r = symmetric_moments(bits) if symmetry else (cp.Variable(k), cp.Variable(k), cp.Variable(k))
    tau = cp.Variable()
    if symmetry and reduce_symmetry:
        # Reflection makes normalization/variance even and the mean odd.
        # The centre mean equation is identically zero; negative rows duplicate
        # positive ones. Keep each reflection orbit of SOC constraints once.
        even_rows, odd_rows = np.flatnonzero(x >= 0), np.flatnonzero(x > 0)
        representatives = [j for j, row in enumerate(bits) if row.tobytes() <= row[::-1].tobytes()]
        paired = [j for j in representatives if not np.array_equal(bits[j], bits[j, ::-1])]
        fixed = [j for j in representatives if np.array_equal(bits[j], bits[j, ::-1])]
        constraints = [S[even_rows]@w == 1, S[odd_rows]@z == u[odd_rows],
                       S[even_rows]@r <= tau+u[even_rows]**2]
        if paired:
            constraints.append(cp.SOC(w[paired]+r[paired], cp.vstack([2*z[paired], w[paired]-r[paired]]), axis=0))
        if fixed:
            # z=0 on self-reflected patterns: their cone is just w>=0, r>=0.
            constraints.extend([w[fixed] >= 0, r[fixed] >= 0])
        counts = dict(normalization_rows=len(even_rows), mean_rows=len(odd_rows),
                      variance_rows=len(even_rows), soc_cones=len(paired), fixed_patterns=len(fixed))
    else:
        constraints = [S@w == 1, S@z == u, S@r <= tau+u*u,
                       cp.SOC(w+r, cp.vstack([2*z, w-r]), axis=0)]
        counts = dict(normalization_rows=len(x), mean_rows=len(x), variance_rows=len(x), soc_cones=k, fixed_patterns=0)
    problem = cp.Problem(cp.Minimize(tau), constraints)
    return problem, dict(w=w, z=z, r=r, tau=tau, bits=np.asarray(bits), scale=scale,
                        pattern_mode=pattern_mode, dimensions=counts)


def extract_primal(problem, model, runtime):
    w, z, r, tau = (model[k] for k in ("w", "z", "r", "tau"))
    scale = model["scale"]
    if any(v.value is None for v in (w, z, r, tau)):
        raise RuntimeError("primal solver returned no candidate")
    values = [np.asarray(v.value) for v in (w,z,r,tau)]
    if not all(np.all(np.isfinite(v)) for v in values):
        raise ValueError("nonfinite primal candidate")
    return dict(w=w.value.copy(), z=z.value.copy()*scale, r=r.value.copy()*scale**2,
                tau=float(tau.value*scale**2), bits=model["bits"],
                solver_status=problem.status, pattern_mode=model["pattern_mode"],
                runtime=runtime, dimensions=model["dimensions"])


def solve_B_primal(epsilon, grid, pattern_mode="full", bits=None, time_limit=120., symmetry=False):
    start = time.perf_counter()
    problem, model = build_B_primal(epsilon, grid, pattern_mode, bits, symmetry)
    solve_problem(problem, time_limit)
    return extract_primal(problem, model, time.perf_counter()-start)
