import numpy as np


def full_patterns(m, max_m=16):
    if m > max_m:
        raise ValueError(f"full enumeration M={m} exceeds guard {max_m}; use cuts")
    return np.array([[(k >> j) & 1 for j in range(m)] for k in range(1 << m)], dtype=np.int8)


def initial_patterns(m):
    patterns = {tuple([0] * m), tuple([1] * m)}
    for j in range(m):
        row = tuple(int(k <= j) for k in range(m))
        patterns.add(row)
        patterns.add(tuple(1 - t for t in row))
        patterns.add(tuple(int(k == j) for k in range(m)))
        patterns.add(tuple(int(k != j) for k in range(m)))
    return np.array(sorted(patterns), dtype=np.int8)


def pattern_matrix(bits, epsilon):
    """Shape (M, number_of_patterns)."""
    return (1 + np.expm1(epsilon) * np.asarray(bits, dtype=float)).T


def separating_patterns(alpha, beta, gamma):
    """Floating search only; rational certification is a separate module."""
    roots = []
    for a, b, g in zip(alpha, beta, gamma):
        if abs(g) < 1e-16:
            if abs(b) > 1e-16:
                roots.append(-a / b)
        else:
            disc = b*b + 4*g*a
            if disc >= 0:
                r = np.sqrt(disc)
                roots.extend([(b-r)/(2*g), (b+r)/(2*g)])
    roots = sorted(set(v for v in roots if np.isfinite(v)))
    if roots:
        points = [roots[0] - max(1, abs(roots[0]))]
        points += [(u+v)/2 for u, v in zip(roots[:-1], roots[1:])]
        points += [roots[-1] + max(1, abs(roots[-1]))]
    else:
        points = [0.]
    rows = {(gamma*y*y-beta*y-alpha < 0).astype(np.int8).tobytes() for y in points}
    return np.frombuffer(b''.join(sorted(rows)), dtype=np.int8).reshape(-1, len(alpha)).copy()
