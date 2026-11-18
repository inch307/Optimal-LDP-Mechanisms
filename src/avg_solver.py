"""Weighted staircase-moment SOCP; all outputs remain free variables."""
import time
import cvxpy as cp
import numpy as np
from fractions import Fraction as F
from .patterns import full_patterns, initial_patterns, pattern_matrix, separating_patterns
from .primal_b import solve_problem
from .certify_dual import exhaustive_lower, interval_lower
from .exact import exp_enclosure, rational, outward, seal, check_hash


def solve_avg(epsilon, grid, weights, second_moment, pattern_mode="auto",
              max_cuts=100, time_limit=120., progress=None):
    x, pi = np.asarray(grid, float), np.asarray(weights, float)
    if not 0 < epsilon <= 100 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be in (0,100]")
    if (x.ndim != 1 or len(x) < 2 or x[0] != -1 or x[-1] != 1
            or np.any(np.diff(x) <= 0) or not np.all(np.isfinite(x))
            or pi.shape != x.shape or not np.all(np.isfinite(pi))
            or np.min(pi) < 0 or abs(sum(pi)-1) > 1e-12):
        raise ValueError("invalid grid or prior weights")
    if pattern_mode == "auto":
        pattern_mode = "full" if len(x) <= 9 else "cuts"
    if pattern_mode not in ("full", "cuts") or max_cuts < 1 or time_limit <= 0:
        raise ValueError("invalid solve settings")
    started = time.perf_counter()
    scale = 1 / np.tanh(epsilon/2)
    u = x / scale
    bits = full_patterns(len(x)) if pattern_mode == "full" else initial_patterns(len(x))
    history = []
    for iteration in range(max_cuts if pattern_mode == "cuts" else 1):
        S = pattern_matrix(bits, epsilon)
        alpha, beta = cp.Variable(len(x)), cp.Variable(len(x))
        A, B, C = S.T @ pi, S.T @ beta, S.T @ alpha
        dual = cp.Problem(cp.Maximize(cp.sum(alpha) + u @ beta),
                          [cp.SOC(A-C, cp.vstack([B, -C-A]), axis=0)])
        solve_problem(dual, max(.01, time_limit-(time.perf_counter()-started)))
        aa, bb = alpha.value, beta.value
        if aa is None or bb is None:
            raise RuntimeError("AVG dual returned no candidate")
        candidates = separating_patterns(aa, bb, pi)
        T = pattern_matrix(candidates, epsilon)
        slack = -(T.T@aa) - (T.T@bb)**2/(4*(T.T@pi))
        complete = bool(min(slack) >= -1e-9)
        history.append(dict(iteration=iteration+1, patterns=len(bits), min_slack=float(min(slack))))
        if progress:
            progress(f"cuts={iteration+1} patterns={len(bits)} violation={max(0.,-min(slack)):.3g}")
        if pattern_mode == "full" or complete or iteration+1 == max_cuts or time.perf_counter()-started >= time_limit:
            break
        old = {tuple(row) for row in bits}
        expanded = old | {tuple(row) for row, s in zip(candidates, slack) if s < -1e-10}
        if expanded == old:
            break
        bits = np.array(sorted(expanded), dtype=np.int8)
    # A restricted primal is an upper candidate even if cut generation stopped early.
    S = pattern_matrix(bits, epsilon)
    k = S.shape[1]
    w, z, r = cp.Variable(k), cp.Variable(k), cp.Variable(k)
    primal = cp.Problem(cp.Minimize((S.T@pi)@r),
                        [S@w == 1, S@z == u,
                         cp.SOC(w+r, cp.vstack([2*z, w-r]), axis=0)])
    solve_problem(primal, time_limit)
    if any(v.value is None for v in (w, z, r)):
        raise RuntimeError("AVG primal returned no candidate")
    return dict(w=w.value, z=z.value*scale, r=r.value*scale**2, bits=bits,
                alpha=aa*scale**2, beta=bb*scale,
                primal_value=float(primal.value*scale**2-float(second_moment)),
                dual_value=float(dual.value*scale**2-float(second_moment)),
                primal_status=primal.status, dual_status=dual.status,
                search_complete=complete, history=history, pattern_mode=pattern_mode,
                runtime_seconds=time.perf_counter()-started)


def certify_avg_dual(epsilon, grid, weights, second_moment, alpha, beta):
    """Fixed *exact* prior weights. Never optimize or round/renormalize gamma."""
    x, pi = list(grid), list(weights)
    aa, bb = list(map(rational, alpha)), list(map(rational, beta))
    E = exp_enclosure(epsilon)[1]
    method = "exhaustive" if len(x) <= 10 else "interval"
    checker = exhaustive_lower if method == "exhaustive" else interval_lower
    bound, checks = checker(aa, bb, pi, E)
    repair = F(0)
    if bound < 0:
        repair = -bound/len(x) + F(1, 10**14)*max(F(1), max(map(abs, aa)))
        repair = F((repair*10**18).__ceil__(), 10**18)
        aa = [v-repair for v in aa]
    bound, checks = checker(aa, bb, pi, E)
    if bound < 0:
        raise ValueError("AVG dual repair failed")
    L = sum(aa) + sum(t*b for t, b in zip(x, bb)) - second_moment
    data = seal(dict(kind="avg_dual", epsilon=str(epsilon), grid=list(map(str, x)),
                     weights=list(map(str, pi)), second_moment=str(second_moment),
                     alpha=list(map(str, aa)), beta=list(map(str, bb)), Ebar=str(E),
                     method=method, checks=checks, repair_per_alpha=str(repair),
                     L_rational=str(L), L_cert=outward(L, False)))
    verify_avg_dual(data)
    return data


def verify_avg_dual(data):
    check_hash(data)
    if data["kind"] != "avg_dual":
        raise ValueError("wrong AVG certificate type")
    x, pi, aa, bb = [[F(v) for v in data[k]] for k in ("grid", "weights", "alpha", "beta")]
    if (not len(x) == len(pi) == len(aa) == len(bb) or len(x) < 2
            or x[0] != -1 or x[-1] != 1 or any(a >= b for a, b in zip(x, x[1:]))
            or min(pi) < 0 or sum(pi) != 1):
        raise ValueError("invalid AVG dimensions/grid/weights")
    E = F(data["Ebar"])
    if E < exp_enclosure(data["epsilon"])[1]:
        raise ValueError("invalid privacy enclosure")
    checker = {"exhaustive": exhaustive_lower, "interval": interval_lower}[data["method"]]
    if checker(aa, bb, pi, E)[0] < 0:
        raise ValueError("AVG all-pattern dual verification failed")
    L = sum(aa) + sum(t*b for t, b in zip(x, bb)) - F(data["second_moment"])
    if L != F(data["L_rational"]) or F.from_float(data["L_cert"]) > L:
        raise ValueError("invalid AVG lower bound")
    return True
