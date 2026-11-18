import time
import numpy as np
from scipy.optimize import linprog
from .channel import GridChannel
from .patterns import pattern_matrix


def recover_from_dual(epsilon, grid, alpha, beta, gamma, bits, time_limit=120.):
    start = time.perf_counter()
    S = pattern_matrix(bits, epsilon)
    A, B, C = S.T@gamma, S.T@beta, S.T@alpha
    if np.any(A <= 0):
        raise ValueError("nonpositive contact curvature")
    y, slack = B/(2*A), -C-B*B/(4*A)
    scale = 1/np.tanh(epsilon/2)
    attempts, best = [], None
    # Widen near-contact candidates, then try all available patterns.
    # All available != all 2^M patterns when dual search used cuts.
    for threshold in (1e-8, 1e-5, 1e-2, float("inf")):
        keep = slack <= threshold*scale**2
        if not np.any(keep):
            continue
        T, v = S[:,keep], y[keep]/scale
        k, u = T.shape[1], np.asarray(grid)/scale
        obj = np.r_[np.zeros(k), 1.]
        eq = np.column_stack([np.vstack([T, T*v]), np.zeros(2*len(u))])
        ub = np.column_stack([T*v*v, -np.ones(len(u))])
        remaining = time_limit-(time.perf_counter()-start)
        if remaining <= 0:
            break
        result = linprog(obj, A_ub=ub, b_ub=u*u, A_eq=eq, b_eq=np.r_[np.ones(len(u)),u],
                         bounds=[(0,None)]*k+[(None,None)], method="highs",
                         options={"time_limit":remaining, "primal_feasibility_tolerance":1e-9,
                                  "dual_feasibility_tolerance":1e-9})
        attempts.append(dict(threshold=None if np.isinf(threshold) else threshold,
                             candidates=k, status=int(result.status), message=result.message))
        if result.success:
            w = result.x[:-1]
            positive = w > 1e-12
            channel = GridChannel(grid, y[keep][positive], T[:,positive]*w[positive])
            if best is None or result.fun*scale**2 < best["tau"]:
                best = dict(channel=channel, tau=float(result.fun*scale**2), status="success")
            # Still expand if the candidate objective is noticeably above dual.
            L = sum(alpha)+np.asarray(grid)@beta-(np.asarray(grid)**2)@gamma
            if result.fun*scale**2-L <= 1e-7*max(1,abs(L)):
                break
    if best is None:
        best = dict(channel=None, status="failed", tau=None)
    best.update(attempts=attempts, runtime=time.perf_counter()-start)
    return best
