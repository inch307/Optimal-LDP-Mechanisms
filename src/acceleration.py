"""Symmetry, nested-grid seed transfer and conservative working-set management.

Working sets are search heuristics. Only the independent all-pattern rational
certificate is a lower-bound claim; removing a cut never certifies anything.
"""
from fractions import Fraction
import json
from pathlib import Path

import cvxpy as cp
import numpy as np
from scipy import sparse

from .certify_dual import verify_dual
from .patterns import initial_patterns, separating_patterns


def require_symmetric_grid(x):
    if not np.array_equal(x, -np.asarray(x)[::-1]):
        raise ValueError("symmetry requires an exactly reflection-symmetric grid")


def canonical_patterns(bits):
    bits = np.asarray(bits, dtype=np.int8)
    # Byte keys avoid M Python/NumPy scalar objects per pattern (important at M~1000).
    keys = sorted({min(row.tobytes(), row[::-1].tobytes()) for row in bits})
    return np.frombuffer(b''.join(keys), dtype=np.int8).reshape(-1, bits.shape[1]).copy()


def close_reflections(bits):
    return np.unique(np.concatenate([bits, np.asarray(bits)[:, ::-1]]), axis=0)


def symmetric_dual(m):
    """Independent coordinates only: alpha/gamma even, beta odd."""
    half = (m+1)//2
    index = np.minimum(np.arange(m), np.arange(m)[::-1])
    even = sparse.csr_matrix((np.ones(m), (np.arange(m), index)), shape=(m, half))
    rows = [i for i in range(m) if i != m-1-i]
    odd = sparse.csr_matrix(([1. if i < m//2 else -1. for i in rows],
                             (rows, index[rows])), shape=(m, m//2))
    return even@cp.Variable(half), odd@cp.Variable(m//2), even@cp.Variable(half)


def symmetric_moments(bits):
    """Pair reflected patterns with equal w/r and opposite z; fixed points z=0."""
    keys = [np.asarray(row, dtype=np.int8).tobytes() for row in bits]
    representatives = sorted({min(k, k[::-1]) for k in keys})
    ids = {k: i for i, k in enumerate(representatives)}
    odd_ids = {k: i for i, k in enumerate(k for k in representatives if k != k[::-1])}
    rows, cols, vals = [], [], []
    indices = []
    for j, key in enumerate(keys):
        rep = min(key, key[::-1])
        indices.append(ids[rep])
        if key != key[::-1]:
            rows.append(j); cols.append(odd_ids[rep]); vals.append(1. if key == rep else -1.)
    even = sparse.csr_matrix((np.ones(len(keys)), (np.arange(len(keys)), indices)),
                             shape=(len(keys), len(ids)))
    if odd_ids:
        odd = sparse.csr_matrix((vals, (rows, cols)), shape=(len(keys), len(odd_ids)))
        z = odd@cp.Variable(len(odd_ids))
    else:
        z = cp.Constant(np.zeros(len(keys)))
    return even@cp.Variable(len(ids)), z, even@cp.Variable(len(ids))


def nested_indices(old, new):
    indices = np.searchsorted(new, old)
    if np.any(indices >= len(new)) or not np.array_equal(np.asarray(new)[indices], old):
        raise ValueError("seed grid is not an exact subset of the new grid")
    return indices


def lift_seed(path, epsilon, grid):
    """Verified dual coefficients get zero mass at new input points.

    Old binary patterns are extended by nearest-neighbor interpolation ONLY as
    search candidates. That heuristic is not used as a feasibility proof.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    verify_dual(data)
    if float(data["epsilon"]) != float(epsilon):
        raise ValueError("seed epsilon mismatch")
    old = np.array([float(Fraction(v)) for v in data["grid"]])
    locations = nested_indices(old, grid)
    result = {}
    for key in ("alpha", "beta", "gamma"):
        result[key] = np.zeros(len(grid))
        result[key][locations] = [float(Fraction(v)) for v in data[key]]
    npz_path = path.with_name(path.name.replace("_dual_cert.json", "_dual_float.npz"))
    bits = initial_patterns(len(grid))
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as saved:
            if np.array_equal(saved["grid"], old):
                old_bits = saved["bits"]
                if old_bits.ndim != 2 or old_bits.shape[1] != len(old) or not np.all((old_bits == 0) | (old_bits == 1)):
                    raise ValueError("invalid seed patterns")
                nearest = np.abs(np.asarray(grid)[:, None]-old).argmin(axis=1)
                bits = np.vstack([bits, old_bits[:, nearest]])
    candidates = separating_patterns(result["alpha"], result["beta"], result["gamma"])
    result.update(bits=np.unique(np.vstack([bits, candidates]), axis=0), source=str(path),
                  source_M=len(old), source_L=data["L_cert"])
    return result


def manage_patterns(bits, slack, added, ages, protected, limit):
    """Soft limit: only drop cuts slack for >=3 rounds, always retain new cuts.

    Active, recently violated and protected rows may exceed the limit. All
    omitted rows remain covered by separation and the exact final audit.
    """
    keys = [np.asarray(row, dtype=np.int8).tobytes() for row in bits]
    new_keys = {np.asarray(row, dtype=np.int8).tobytes() for row in added}
    protected = {k if isinstance(k, bytes) else np.asarray(k, dtype=np.int8).tobytes() for k in protected}
    for key, value in zip(keys, slack):
        ages[key] = ages.get(key, 0)+1 if value > 1e-5 else 0
    keep = set(keys) | new_keys
    removable = sorted(((float(s), k) for k, s in zip(keys, slack)
                        if ages[k] >= 3 and k not in protected and k not in new_keys), reverse=True)
    dropped = 0
    if limit:
        for _, key in removable:
            if len(keep) <= limit:
                break
            keep.remove(key); ages.pop(key, None); dropped += 1
    # Do not retain an unbounded historical dictionary for removed cuts.
    for key in list(ages):
        if key not in keep:
            del ages[key]
    result = np.frombuffer(b''.join(sorted(keep)), dtype=np.int8).reshape(-1, bits.shape[1]).copy()
    return result, dropped
