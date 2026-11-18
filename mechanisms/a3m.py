"""Distribution-aware AAA (A3M) mechanism for quantized mean estimation.

This module implements the linear program in Section 4 of
"AAA: an Adaptive Mechanism for Locally Differentially Private Mean
Estimation".  In particular, the two geometric tails are part of the
normalization, zero-mean, objective, and privacy constraints.  The official
artifact uses dense constraint matrices; this implementation builds sparse
matrices so that the paper's 0.125 quantization on [-1, 1] is practical.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, lil_matrix


def _validate_distribution(distribution: np.ndarray) -> np.ndarray:
    distribution = np.asarray(distribution, dtype=float).reshape(-1)
    if distribution.size < 2:
        raise ValueError("AAA needs at least two quantized input points")
    if not np.all(np.isfinite(distribution)) or np.any(distribution < 0):
        raise ValueError("distribution must contain finite non-negative values")
    total = float(distribution.sum())
    if total <= 0:
        raise ValueError("distribution must have positive mass")
    return distribution / total


def _tail_moments(half_width: int, tail_ratio: float) -> tuple[float, float, float]:
    """Return geometric-tail mass, first-index, and squared-index factors."""
    one_minus_r = 1.0 - tail_ratio
    mass = 1.0 / one_minus_r
    first = half_width / one_minus_r + tail_ratio / one_minus_r**2
    second = (
        half_width**2 / one_minus_r
        + (2 * half_width - 1) * tail_ratio / one_minus_r**2
        + 2 * tail_ratio / one_minus_r**3
    )
    return mass, first, second


def _noise_term(
    input_index: int,
    report_index: int,
    half_width: int,
    tail_ratio: float,
) -> tuple[int, float]:
    """Map P(Y=report | X=input) to one core LP variable and coefficient."""
    noise_index = report_index - input_index
    if noise_index <= -half_width:
        return 0, tail_ratio ** (-half_width - noise_index)
    if noise_index >= half_width:
        return 2 * half_width, tail_ratio ** (noise_index - half_width)
    return noise_index + half_width, 1.0


def audit_mechanism(mechanism: dict[str, Any]) -> dict[str, float]:
    """Audit normalization, unbiasedness, non-negativity, and epsilon-LDP."""
    probabilities = np.asarray(mechanism["core_probabilities"], dtype=float)
    input_points = np.asarray(mechanism["input_points"], dtype=float)
    half_width = int(mechanism["noise_half_width"])
    tail_ratio = float(mechanism["tail_ratio"])
    epsilon = float(mechanism["epsilon"])
    sigma = float(mechanism["sigma"])
    num_inputs = input_points.size
    mass_factor, first_factor, _ = _tail_moments(half_width, tail_ratio)

    normalization = (
        probabilities[:, 1:-1].sum(axis=1)
        + mass_factor * (probabilities[:, 0] + probabilities[:, -1])
    )
    indices = np.arange(-half_width + 1, half_width, dtype=float)
    means = sigma * (
        probabilities[:, 1:-1] @ indices
        + first_factor * (probabilities[:, -1] - probabilities[:, 0])
    )

    exp_epsilon = math.exp(epsilon)
    max_ldp_violation = 0.0
    # Outside this range all inputs are in the same geometric tail, and the
    # pairwise ratios are constant in the report index.
    for report_index in range(-half_width, half_width + num_inputs):
        report_probabilities = np.empty(num_inputs, dtype=float)
        for input_index in range(num_inputs):
            column, coefficient = _noise_term(
                input_index, report_index, half_width, tail_ratio
            )
            report_probabilities[input_index] = (
                coefficient * probabilities[input_index, column]
            )
        violation = (
            report_probabilities[:, None]
            - exp_epsilon * report_probabilities[None, :]
        )
        max_ldp_violation = max(max_ldp_violation, float(np.max(violation)))

    return {
        "max_normalization_error": float(np.max(np.abs(normalization - 1.0))),
        "max_noise_mean_error": float(np.max(np.abs(means))),
        "min_probability": float(np.min(probabilities)),
        "max_ldp_violation": max(0.0, max_ldp_violation),
    }


def optimize_a3m(
    distribution: np.ndarray,
    epsilon: float,
    *,
    beta: float = 1.0,
    noise_range_ratio: float = 4.0,
    tail_ratio: float = 0.5,
    solver_options: dict[str, Any] | None = None,
    enforce_audit: bool = True,
) -> dict[str, Any]:
    """Solve the quantized AAA linear program.

    ``distribution`` has one mass for every point in an equally spaced grid
    from ``-beta`` to ``beta``.  ``noise_range_ratio`` is |A_core| / |X|, so
    the paper default 4 uses a geometric-tail boundary at twice the number of
    input intervals on either side of zero.
    """
    distribution = _validate_distribution(distribution)
    if epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")
    if beta <= 0 or not np.isfinite(beta):
        raise ValueError("beta must be finite and positive")
    if noise_range_ratio <= 0 or not np.isfinite(noise_range_ratio):
        raise ValueError("noise_range_ratio must be finite and positive")
    if not (0 < tail_ratio < 1):
        raise ValueError("tail_ratio must lie strictly between zero and one")

    num_inputs = distribution.size
    num_intervals = num_inputs - 1
    sigma = 2.0 * beta / num_intervals
    input_points = np.linspace(-beta, beta, num_inputs)
    half_width = max(
        num_intervals,
        int(math.ceil(0.5 * noise_range_ratio * num_intervals)),
    )
    num_core = 2 * half_width + 1
    num_variables = num_inputs * num_core
    mass_factor, first_factor, second_factor = _tail_moments(
        half_width, tail_ratio
    )

    interior_indices = np.arange(-half_width + 1, half_width, dtype=float)
    second_moment_coefficients = np.concatenate((
        [second_factor], interior_indices**2, [second_factor]
    )) * sigma**2
    objective = (
        distribution[:, None] * second_moment_coefficients[None, :]
    ).reshape(-1)

    # Each input row has one normalization and one zero-noise-mean equality.
    equality = lil_matrix((2 * num_inputs, num_variables), dtype=float)
    equality_rhs = np.zeros(2 * num_inputs, dtype=float)
    normalization_coefficients = np.concatenate((
        [mass_factor], np.ones(num_core - 2), [mass_factor]
    ))
    mean_coefficients = sigma * np.concatenate((
        [-first_factor], interior_indices, [first_factor]
    ))
    for input_index in range(num_inputs):
        start = input_index * num_core
        stop = start + num_core
        equality[2 * input_index, start:stop] = normalization_coefficients
        equality_rhs[2 * input_index] = 1.0
        equality[2 * input_index + 1, start:stop] = mean_coefficients

    # For a fixed report lattice point t, privacy compares noise t-i under
    # input i with noise t-j under input j.  Two non-zeros per inequality keep
    # this matrix compact even at the paper's fine quantization.
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    inequality_row = 0
    exp_epsilon = math.exp(epsilon)
    for report_index in range(-half_width, half_width + num_inputs):
        terms = [
            _noise_term(i, report_index, half_width, tail_ratio)
            for i in range(num_inputs)
        ]
        for left_input in range(num_inputs):
            left_column, left_coefficient = terms[left_input]
            for right_input in range(num_inputs):
                if left_input == right_input:
                    continue
                right_column, right_coefficient = terms[right_input]
                row_indices.extend((inequality_row, inequality_row))
                column_indices.extend((
                    left_input * num_core + left_column,
                    right_input * num_core + right_column,
                ))
                values.extend((left_coefficient, -exp_epsilon * right_coefficient))
                inequality_row += 1

    inequality = coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(inequality_row, num_variables),
        dtype=float,
    ).tocsr()
    result = linprog(
        c=objective,
        A_ub=inequality,
        b_ub=np.zeros(inequality_row, dtype=float),
        A_eq=equality.tocsr(),
        b_eq=equality_rhs,
        bounds=(0.0, None),
        method="highs",
        options=solver_options,
    )
    if not result.success or result.x is None:
        raise RuntimeError(
            f"AAA LP failed (status={result.status}): {result.message}"
        )

    mechanism: dict[str, Any] = {
        "mechanism": "a3m",
        "epsilon": float(epsilon),
        "beta": float(beta),
        "sigma": float(sigma),
        "input_points": input_points,
        "design_distribution": distribution,
        "noise_range_ratio": float(noise_range_ratio),
        "noise_half_width": int(half_width),
        "tail_ratio": float(tail_ratio),
        "core_probabilities": result.x.reshape(num_inputs, num_core),
        "objective": float(result.fun),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_iterations": int(result.nit),
        "audit_enforced": bool(enforce_audit),
    }
    audit = audit_mechanism(mechanism)
    mechanism["audit"] = audit
    if enforce_audit and (
        audit["max_normalization_error"] > 2e-7
        or audit["max_noise_mean_error"] > 2e-7
        or audit["min_probability"] < -2e-9
        or audit["max_ldp_violation"] > 2e-7
    ):
        raise RuntimeError(f"AAA LP solution failed its post-solve audit: {audit}")
    return mechanism


def conditional_noise_moments(mechanism: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return E[A|x] and E[A^2|x] for all quantized inputs."""
    probabilities = np.asarray(mechanism["core_probabilities"], dtype=float)
    half_width = int(mechanism["noise_half_width"])
    tail_ratio = float(mechanism["tail_ratio"])
    sigma = float(mechanism["sigma"])
    _, first_factor, second_factor = _tail_moments(half_width, tail_ratio)
    interior = np.arange(-half_width + 1, half_width, dtype=float)
    means = sigma * (
        probabilities[:, 1:-1] @ interior
        + first_factor * (probabilities[:, -1] - probabilities[:, 0])
    )
    seconds = sigma**2 * (
        probabilities[:, 1:-1] @ (interior**2)
        + second_factor * (probabilities[:, 0] + probabilities[:, -1])
    )
    return means, seconds


def sample_noise_sum(
    mechanism: dict[str, Any],
    input_counts: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """Sample the sum of all AAA noises without materializing all reports."""
    input_counts = np.asarray(input_counts, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(mechanism["core_probabilities"], dtype=float)
    if input_counts.size != probabilities.shape[0]:
        raise ValueError("input_counts and AAA input grid have different sizes")
    half_width = int(mechanism["noise_half_width"])
    tail_ratio = float(mechanism["tail_ratio"])
    sigma = float(mechanism["sigma"])
    interior_indices = np.arange(-half_width + 1, half_width, dtype=np.int64)
    total_index_sum = 0
    for input_index, count in enumerate(input_counts):
        if count <= 0:
            continue
        row = probabilities[input_index]
        categories = np.concatenate((
            [row[0] / (1.0 - tail_ratio)],
            row[1:-1],
            [row[-1] / (1.0 - tail_ratio)],
        ))
        categories = np.clip(categories, 0.0, None)
        categories /= categories.sum()
        sampled_counts = rng.multinomial(int(count), categories)
        total_index_sum += int(sampled_counts[1:-1] @ interior_indices)
        left_count = int(sampled_counts[0])
        right_count = int(sampled_counts[-1])
        if left_count:
            distances = rng.geometric(1.0 - tail_ratio, size=left_count) - 1
            total_index_sum += int(np.sum(-half_width - distances))
        if right_count:
            distances = rng.geometric(1.0 - tail_ratio, size=right_count) - 1
            total_index_sum += int(np.sum(half_width + distances))
    return float(sigma * total_index_sum)
