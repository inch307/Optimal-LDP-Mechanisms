"""Strict privacy auditing and repair helpers for symmetric N-output mechanisms.

The optimizers in this project store probabilities only for non-negative input
knots.  The negative half of the channel is obtained by reversing output rows.
Consequently an LDP audit must compare both ``p`` and ``p[::-1]``; checking the
stored half alone can incorrectly certify a mechanism with a zero probability.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Tuple

import numpy as np


@dataclass
class SafetyAudit:
    epsilon: float
    target_ratio: float
    max_ratio: float
    privacy_slack: float
    ldp_safe: bool
    min_probability: float
    max_probability: float
    max_column_sum_error: float
    symmetry_error_at_zero: float
    min_dx: float
    x_last_error: float
    inactive_outputs: int
    finite: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _arrays(a_values, p_matrix) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a_values, dtype=float).reshape(-1)
    p = np.asarray(p_matrix, dtype=float)
    if p.ndim != 2 or p.shape[0] != a.size:
        raise ValueError("p_matrix must have shape (len(a_values), number_of_knots)")
    if p.shape[1] < 1:
        raise ValueError("p_matrix must contain at least one input knot")
    return a, p


def full_row_extrema(p_matrix) -> Tuple[np.ndarray, np.ndarray]:
    """Return extrema over the stored and reflected halves of the channel."""
    p = np.asarray(p_matrix, dtype=float)
    full_min = np.minimum(np.min(p, axis=1), np.min(p[::-1], axis=1))
    full_max = np.maximum(np.max(p, axis=1), np.max(p[::-1], axis=1))
    return full_min, full_max


def strict_max_ratio(p_matrix) -> float:
    row_min, row_max = full_row_extrema(p_matrix)
    active = row_max > 0.0
    if not np.any(active):
        return 1.0
    if np.any((row_min <= 0.0) & active):
        return float("inf")
    return float(np.max(row_max[active] / row_min[active]))


def audit_arrays(
    a_values,
    p_matrix,
    epsilon: float,
    *,
    ratio_rtol: float = 1e-9,
    probability_atol: float = 1e-12,
    equality_atol: float = 2e-6,
    inactive_threshold: float = 1e-9,
) -> SafetyAudit:
    """Audit the exact symmetric piecewise-linear channel, without pseudocounts."""
    a, p = _arrays(a_values, p_matrix)
    finite = bool(np.all(np.isfinite(a)) and np.all(np.isfinite(p)))
    if not finite:
        return SafetyAudit(
            float(epsilon), float(np.exp(epsilon)), float("inf"),
            float("-inf"), False, float("nan"), float("nan"),
            float("inf"), float("inf"), float("-inf"), float("inf"),
            int(a.size), False,
        )

    target = float(np.exp(epsilon))
    ratio = strict_max_ratio(p)
    col_error = float(np.max(np.abs(np.sum(p, axis=0) - 1.0)))
    symmetry_error = float(np.max(np.abs(p[:, 0] - p[::-1, 0])))
    x = a @ p
    min_dx = float(np.min(np.diff(x))) if x.size > 1 else float("inf")
    x_last_error = float(abs(x[-1] - 1.0))
    row_min, row_max = full_row_extrema(p)
    inactive = int(np.sum(row_max <= inactive_threshold))
    nonnegative = bool(np.min(p) >= -probability_atol)
    privacy_ok = bool(ratio <= target * (1.0 + ratio_rtol))
    equalities_ok = bool(
        col_error <= equality_atol
        and symmetry_error <= equality_atol
        and x_last_error <= equality_atol
        and min_dx > 0.0
    )
    return SafetyAudit(
        epsilon=float(epsilon),
        target_ratio=target,
        max_ratio=ratio,
        privacy_slack=float(target - ratio),
        ldp_safe=bool(finite and nonnegative and privacy_ok and equalities_ok),
        min_probability=float(np.min(p)),
        max_probability=float(np.max(p)),
        max_column_sum_error=col_error,
        symmetry_error_at_zero=symmetry_error,
        min_dx=min_dx,
        x_last_error=x_last_error,
        inactive_outputs=inactive,
        finite=finite,
    )


def normalize_columns(p_matrix) -> np.ndarray:
    p = np.maximum(np.asarray(p_matrix, dtype=float), 0.0)
    sums = np.sum(p, axis=0, keepdims=True)
    if np.any(sums <= 0.0):
        raise ValueError("Cannot normalize a probability column with zero mass")
    return p / sums


def enforce_zero_input_symmetry(p_matrix) -> np.ndarray:
    p = np.asarray(p_matrix, dtype=float).copy()
    p[:, 0] = 0.5 * (p[:, 0] + p[::-1, 0])
    return p


def prune_inactive_symmetric_pairs(
    a_values,
    p_matrix,
    threshold: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove only output rows whose row and reflected partner are both inactive."""
    a, p = _arrays(a_values, p_matrix)
    row_activity = np.maximum(np.max(p, axis=1), np.max(p[::-1], axis=1))
    keep = np.ones(a.size, dtype=bool)
    for i in range((a.size + 1) // 2):
        j = a.size - 1 - i
        if max(row_activity[i], row_activity[j]) <= threshold:
            keep[i] = False
            keep[j] = False
    if np.sum(keep) < 2:
        keep[:] = True
    if np.all(keep):
        return a.copy(), p.copy(), keep
    return a[keep], normalize_columns(p[keep]), keep


def privacy_blanket(
    p_matrix,
    epsilon: float,
    *,
    margin: float = 1e-5,
) -> Tuple[np.ndarray, float, float]:
    """Mix minimally with uniform noise to enforce strict full-channel LDP.

    For every output row, the mixture must satisfy
    ``((1-alpha) max + alpha/N) / ((1-alpha) min + alpha/N) <= E``.
    The largest row-wise lower bound on ``alpha`` is used.
    """
    p = normalize_columns(p_matrix)
    if margin < 0.0 or margin >= epsilon:
        if epsilon == 0.0 and margin == 0.0:
            effective_epsilon = 0.0
        else:
            raise ValueError("margin must satisfy 0 <= margin < epsilon")
    else:
        effective_epsilon = float(epsilon - margin)
    target = float(np.exp(effective_epsilon))
    row_min, row_max = full_row_extrema(p)
    violation = row_max - target * row_min
    denominator = violation + (target - 1.0) / p.shape[0]
    required = np.zeros_like(violation)
    mask = violation > 0.0
    required[mask] = violation[mask] / denominator[mask]
    alpha = float(np.clip(np.max(required, initial=0.0), 0.0, 1.0))
    # A few ulps of interior slack avoid equality flipping in serialization.
    if alpha > 0.0:
        alpha = float(min(1.0, np.nextafter(alpha, 1.0)))
    repaired = (1.0 - alpha) * p + alpha / p.shape[0]
    return normalize_columns(repaired), alpha, effective_epsilon


def stabilize_input_knots(
    a_values,
    p_matrix,
    epsilon: float,
    *,
    min_gap: float = 2e-7,
) -> Tuple[np.ndarray, float, int]:
    """Sort nonzero knots and minimally mix a strictly monotone safe anchor.

    SLSQP can return adjacent input means in the wrong order by a few tolerance
    units.  Column permutation preserves LDP.  A small affine tilt of the
    uniform channel then makes every input-mean gap strictly positive while
    itself remaining inside the requested privacy budget.
    """
    a, p = _arrays(a_values, p_matrix)
    x = a @ p
    if p.shape[1] > 1:
        tail_order = np.argsort(x[1:], kind="stable") + 1
        order = np.concatenate([[0], tail_order])
        reordered = int(np.sum(order != np.arange(order.size)))
        p = p[:, order]
    else:
        reordered = 0

    max_abs_a = float(np.max(np.abs(a)))
    if max_abs_a <= 0.0 or p.shape[1] <= 1:
        return p, 0.0, reordered
    direction = a / max_abs_a
    rho = 0.9 * np.tanh(max(float(epsilon), 0.0) / 2.0)
    rho = max(rho, 1e-8)
    t = np.linspace(0.0, 1.0, p.shape[1])
    anchor = (1.0 + rho * direction[:, None] * t[None, :]) / a.size
    anchor = normalize_columns(anchor)
    current_dx = np.diff(a @ p)
    anchor_dx = np.diff(a @ anchor)
    required = np.zeros_like(current_dx)
    mask = current_dx < min_gap
    denominator = anchor_dx - current_dx
    if np.any(mask & (denominator <= 0.0)):
        raise ValueError("Could not construct a monotone LDP anchor")
    required[mask] = (min_gap - current_dx[mask]) / denominator[mask]
    alpha = float(np.clip(np.max(required, initial=0.0), 0.0, 1.0))
    if alpha > 0.0:
        alpha = float(min(1.0, np.nextafter(alpha, 1.0)))
        p = (1.0 - alpha) * p + alpha * anchor
    return normalize_columns(p), alpha, reordered


def repair_arrays(
    a_values,
    p_matrix,
    epsilon: float,
    *,
    prune_threshold: float = 1e-10,
    privacy_margin: float = 1e-5,
    allow_prune: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Make a stored symmetric mechanism feasible and strictly LDP safe.

    Probability repair does not change output values.  Afterwards output values
    are scaled symmetrically so the last non-negative knot has mean one.
    """
    a, p = _arrays(a_values, p_matrix)
    before = audit_arrays(a, p, epsilon).to_dict()
    order = np.argsort(a)
    a, p = a[order], p[order]
    p = normalize_columns(p)
    p = normalize_columns(enforce_zero_input_symmetry(p))
    keep = np.ones(a.size, dtype=bool)
    if allow_prune:
        candidate_a, candidate_p, candidate_keep = prune_inactive_symmetric_pairs(
            a, p, threshold=prune_threshold
        )
        candidate_p = normalize_columns(enforce_zero_input_symmetry(candidate_p))
        # Knot ordering is repaired below; pruning itself preserves symmetry/LDP.
        a, p, keep = candidate_a, candidate_p, candidate_keep

    p, alpha, effective_epsilon = privacy_blanket(
        p, epsilon, margin=privacy_margin
    )
    p = normalize_columns(enforce_zero_input_symmetry(p))
    p, knot_alpha, reordered_columns = stabilize_input_knots(
        a, p, effective_epsilon
    )
    p = normalize_columns(enforce_zero_input_symmetry(p))
    x = a @ p
    if not np.all(np.diff(x) > 0.0):
        raise ValueError("Repaired mechanism has non-increasing input knots")
    if x[-1] <= 0.0:
        raise ValueError("Cannot rescale a mechanism whose endpoint mean is non-positive")
    a = a / float(x[-1])
    after = audit_arrays(a, p, epsilon).to_dict()
    info = {
        "before": before,
        "after": after,
        "privacy_blanket_alpha": alpha,
        "knot_stabilization_alpha": knot_alpha,
        "reordered_input_columns": reordered_columns,
        "effective_epsilon": effective_epsilon,
        "pruned_outputs": int(keep.size - np.sum(keep)),
        "kept_mask": keep.tolist(),
    }
    return a, p, info


def average_squared_error(a_values, p_matrix) -> float:
    a, p = _arrays(a_values, p_matrix)
    x = a @ p
    ey2 = (a * a) @ p
    dx = np.diff(x)
    # E[Y^2|x] is piecewise linear, whereas the unbiased channel contributes x^2.
    return float(np.sum(0.5 * (ey2[1:] + ey2[:-1]) * dx - (x[1:] ** 3 - x[:-1] ** 3) / 3.0))


def worst_squared_error(a_values, p_matrix) -> float:
    a, p = _arrays(a_values, p_matrix)
    x = a @ p
    ey2 = (a * a) @ p
    candidates = list(ey2 - x * x)
    for j in range(1, x.size):
        dx = x[j] - x[j - 1]
        if dx <= 0.0:
            continue
        x_star = (ey2[j] - ey2[j - 1]) / (2.0 * dx)
        if x[j - 1] <= x_star <= x[j]:
            slope = (ey2[j] - ey2[j - 1]) / dx
            ey2_star = ey2[j - 1] + slope * (x_star - x[j - 1])
            candidates.append(ey2_star - x_star * x_star)
    return float(np.max(candidates))


def repair_data(data: Dict[str, Any], epsilon: float | None = None, **kwargs):
    md = data.get("metadata", {})
    eps = float(md.get("epsilon") if epsilon is None else epsilon)
    a, p, info = repair_arrays(data["a_values"], data["p_matrix"], eps, **kwargs)
    repaired = dict(data)
    repaired["metadata"] = dict(md)
    repaired["a_values"] = a
    repaired["p_matrix"] = p
    repaired["metadata"].update({
        "epsilon": eps,
        "N_total_a_points": int(a.size),
        "safety_repair": info,
        "avg_var": average_squared_error(a, p),
        "worst_var": worst_squared_error(a, p),
        "strict_ldp_safe": bool(info["after"]["ldp_safe"]),
    })
    return repaired, info
