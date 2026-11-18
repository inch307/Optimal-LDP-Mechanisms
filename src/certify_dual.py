"""No solver dependency: exact exhaustive or root-interval envelope certificate."""
import json
from fractions import Fraction as F
from pathlib import Path
from .exact import rational, exp_enclosure, outward, seal, check_hash, save_json, quadratic_min, sqrt_enclosure


def exhaustive_lower(alpha, beta, gamma, E):
    m = len(alpha)
    if m > 16:
        raise ValueError("exact exhaustive certificate limited to M<=16")
    worst = None
    for mask in range(1 << m):
        weights = [E if (mask >> j) & 1 else F(1) for j in range(m)]
        A = sum(s*g for s, g in zip(weights, gamma))
        B = sum(s*b for s, b in zip(weights, beta))
        C = sum(s*a for s, a in zip(weights, alpha))
        value = -C-B*B/(4*A)
        worst = value if worst is None else min(worst, value)
    return worst, 1 << m


def interval_lower(alpha, beta, gamma, E):
    """Rigorous lower bound on min_{y,s} q_s(y), covering the entire real line.

    Isolate every root of every f_j with rational sqrt enclosures. Between root
    boxes all signs are exact. On root boxes sum conservative individual minima.
    This may lose a tiny amount, but never calls float sampling a certificate.
    """
    boxes = []
    for a, b, g in zip(alpha, beta, gamma):
        if g == 0:
            if b:
                r = -a/b
                boxes.append((r, r))
        else:
            disc = b*b+4*g*a
            if disc >= 0:
                lo, hi = sqrt_enclosure(disc)
                boxes.extend([((b-hi)/(2*g), (b-lo)/(2*g)),
                              ((b+lo)/(2*g), (b+hi)/(2*g))])
    merged = []
    for lo, hi in sorted(boxes):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
        else:
            merged.append((lo, hi))
    pieces, left = [], None
    for lo, hi in merged:
        if left is None or left < lo:
            pieces.append((left, lo, False))
        pieces.append((lo, hi, True))
        left = hi
    pieces.append((left, None, False))
    bounds = []
    for lo, hi, root_box in pieces:
        if root_box:
            minima = [quadratic_min(g, -b, -a, lo, hi) for a, b, g in zip(alpha, beta, gamma)]
            bounds.append(sum(min(v, E*v) for v in minima))
        else:
            sample = F(0) if lo is None and hi is None else (hi-1 if lo is None else lo+1 if hi is None else (lo+hi)/2)
            weights = [E if g*sample*sample-b*sample-a < 0 else F(1) for a, b, g in zip(alpha, beta, gamma)]
            A = sum(s*g for s, g in zip(weights, gamma))
            B = sum(s*b for s, b in zip(weights, beta))
            C = sum(s*a for s, a in zip(weights, alpha))
            bounds.append(quadratic_min(A, -B, -C, lo, hi))
    return min(bounds), len(pieces)


def certify_dual(epsilon, grid, alpha, beta, gamma, method="auto", path=None):
    x = [F(str(v)) for v in grid]
    aa, bb = [rational(v) for v in alpha], [rational(v) for v in beta]
    gg = [max(F(0), rational(v)) for v in gamma]
    total = sum(gg)
    if total <= 0:
        raise ValueError("gamma has no positive mass")
    gg = [g/total for g in gg]
    _, E = exp_enclosure(epsilon)
    method = ("exhaustive" if len(x) <= 10 else "interval") if method == "auto" else method
    checker = {"exhaustive": exhaustive_lower, "interval": interval_lower}[method]
    repair = F(0)
    for _ in range(5):
        bound, checks = checker(aa, bb, gg, E)
        if bound >= 0:
            break
        delta = -bound/len(x) + F(1, 10**14)*max(F(1), max(abs(a) for a in aa))
        # Round repair upwards to avoid denominator growth over repeated audits.
        delta = F((delta*10**18).__ceil__(), 10**18)
        aa = [a-delta for a in aa]
        repair += delta
    if bound < 0:
        raise ValueError("could not certify repaired dual")
    L = sum(aa)+sum(t*b-t*t*g for t, b, g in zip(x, bb, gg))
    data = seal(dict(kind="dual", epsilon=str(epsilon), grid=[str(v) for v in x],
                     Ebar=str(E), alpha=[str(v) for v in aa], beta=[str(v) for v in bb],
                     gamma=[str(v) for v in gg], L_rational=str(L), L_cert=outward(L, False),
                     method=method, checks=checks, repair_per_alpha=str(repair),
                     verify_status="passed", meaning="universal lower bound for continuous-input unrestricted-output unbiased epsilon-LDP"))
    verify_dual(data)
    if path:
        save_json(path, data)
    return data


def verify_dual(certificate):
    data = json.loads(Path(certificate).read_text(encoding="utf-8")) if isinstance(certificate, (str, Path)) else certificate
    check_hash(data)
    if data["kind"] != "dual":
        raise ValueError("wrong certificate type")
    x, aa, bb, gg = [[F(v) for v in data[k]] for k in ("grid", "alpha", "beta", "gamma")]
    if not len(x) == len(aa) == len(bb) == len(gg) or len(x) < 2:
        raise ValueError("invalid dimensions")
    if x[0] != -1 or x[-1] != 1 or any(u >= v for u,v in zip(x, x[1:])):
        raise ValueError("invalid input grid")
    if min(gg) < 0 or sum(gg) != 1:
        raise ValueError("gamma infeasible")
    E = F(data["Ebar"])
    if E < exp_enclosure(data["epsilon"])[1]:
        raise ValueError("exp upper enclosure invalid")
    checker = {"exhaustive": exhaustive_lower, "interval": interval_lower}[data["method"]]
    bound, _ = checker(aa, bb, gg, E)
    if bound < 0:
        raise ValueError("dual fails all-pattern PSD/envelope check")
    L = sum(aa)+sum(t*b-t*t*g for t, b, g in zip(x, bb, gg))
    if L != F(data["L_rational"]) or F.from_float(data["L_cert"]) > L:
        raise ValueError("invalid lower-bound rounding")
    return True
