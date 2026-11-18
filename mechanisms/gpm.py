"""Mean-unbiased optimal generalized piecewise mechanism (GPM).

This implements Theorem 5.1 of Zheng, Mishra, and Hu (PoPETs 2025).
The theorem is stated for inputs in ``[0, 1]`` and uses the enlarged output
domain ``[-C, C + 1]``. The implementation maps it affinely to the project's
symmetric input domain ``[-range, range]``. Unlike the classical-domain
mechanism from Theorem 3.7, this mechanism is unbiased and its squared error
is therefore directly comparable to variance baselines.
"""

from __future__ import annotations

import numpy as np


def parameters(eps, range=1.0):
    """Return ``(C, A, high_density, low_density)`` for symmetric inputs."""
    eps = float(eps)
    range = float(range)
    if eps <= 0.0:
        raise ValueError("GPM requires epsilon > 0")
    if range <= 0.0:
        raise ValueError("GPM requires range > 0")
    root_e = np.exp(eps / 2.0)
    C = (root_e + 1.0) / (root_e - 1.0)
    A = range * (2.0 * C + 1.0)
    high_density = root_e / (2.0 * C + 1.0)
    low_density = high_density / np.exp(eps)
    return C, A, high_density, low_density


class GPM:
    """Mean-unbiased GPM with the experiment API used by this repository."""

    def __init__(self, d, eps, rng, range=1.0, k=1):
        self.d = int(d)
        self.eps = float(eps)
        self.k = int(k)
        self.eps_k = self.eps / self.k
        self.rng = rng
        self.range = float(range)
        self.C, self.A, self.high_density_01, self.low_density_01 = parameters(
            self.eps_k, self.range
        )

    def _high_interval_01(self, x01):
        """High-density interval in the theorem's enlarged z-domain."""
        x01 = np.asarray(x01, dtype=float)
        C = self.C
        left = 0.5 * (C + 1.0) * x01 - ((3.0 * C + 1.0) * (C - 1.0)) / (4.0 * C)
        right = 0.5 * (C + 1.0) * x01 + ((C + 1.0) * (C - 1.0)) / (4.0 * C)
        return left, right

    def GPM_batch(self, data):
        values = np.asarray(data, dtype=float).reshape(-1)
        x01 = np.clip(0.5 * (values / self.range + 1.0), 0.0, 1.0)
        left, right = self._high_interval_01(x01)
        lower = -self.C
        upper = self.C + 1.0
        high_width = right - left
        high_mass = self.high_density_01 * high_width

        choose_high = self.rng.random(len(values)) < high_mass
        z = np.empty(len(values), dtype=float)
        if np.any(choose_high):
            idx = np.flatnonzero(choose_high)
            z[idx] = self.rng.uniform(left[idx], right[idx])

        low_idx = np.flatnonzero(~choose_high)
        if low_idx.size:
            left_width = left[low_idx] - lower
            right_width = upper - right[low_idx]
            choose_left = self.rng.random(low_idx.size) < left_width / (left_width + right_width)
            li = low_idx[choose_left]
            if li.size:
                z[li] = self.rng.uniform(lower, left[li])
            ri = low_idx[~choose_left]
            if ri.size:
                z[ri] = self.rng.uniform(right[ri], upper)

        # z in [-C, C+1] and x01=(x/range+1)/2 imply y=2*range*z-range.
        return 2.0 * self.range * z - self.range, None

    def exp_batch(self, name, data):
        if name not in ("gpm", "ogpm"):
            raise ValueError(f"GPM cannot execute mechanism name {name!r}")
        return self.GPM_batch(data)

    def exp_multi_batch(self, name, sampled_values, row_indices, sampled_col_indices, num_sam, d, k):
        perturbed, _ = self.GPM_batch(np.asarray(sampled_values).reshape(-1))
        perturbed = perturbed.reshape(num_sam, k)
        noisy = np.zeros((num_sam, d), dtype=float)
        noisy[row_indices, sampled_col_indices] = perturbed
        return noisy * (d / k)

    def squared_error(self, x):
        """Analytic variance ``E[(Y-x)^2 | x]`` (the mechanism is unbiased)."""
        x = np.asarray(x, dtype=float)
        u = np.clip(0.5 * (x / self.range + 1.0), 0.0, 1.0)
        left, right = self._high_interval_01(u)
        lower = -self.C
        upper = self.C + 1.0
        whole = ((upper - u) ** 3 - (lower - u) ** 3) / 3.0
        inner = ((right - u) ** 3 - (left - u) ** 3) / 3.0
        mse01 = self.low_density_01 * whole + (
            self.high_density_01 - self.low_density_01
        ) * inner
        return 4.0 * self.range**2 * mse01

    def worst_squared_error(self, grid_size=20001):
        grid = np.linspace(-self.range, self.range, int(grid_size))
        return float(np.max(self.squared_error(grid)))


def transition_from_grid(eps, x_grid, y_edges, range=1.0):
    """Exact probabilities for arbitrary bins spanning the enlarged output."""
    C, A, high, low = parameters(eps, range)
    x = np.asarray(x_grid, dtype=float).reshape(-1, 1)
    u = np.clip(0.5 * (x / range + 1.0), 0.0, 1.0)
    left = 0.5 * (C + 1.0) * u - ((3.0 * C + 1.0) * (C - 1.0)) / (4.0 * C)
    right = 0.5 * (C + 1.0) * u + ((C + 1.0) * (C - 1.0)) / (4.0 * C)

    # y = 2*range*z-range.
    edges = np.asarray(y_edges, dtype=float).reshape(1, -1)
    z_edges = (edges + range) / (2.0 * range)
    widths = z_edges[:, 1:] - z_edges[:, :-1]
    overlap = np.clip(
        np.minimum(z_edges[:, 1:], right) - np.maximum(z_edges[:, :-1], left),
        0.0,
        None,
    )
    probs = low * widths + (high - low) * overlap
    probs = np.clip(probs, 0.0, None)
    row_sums = probs.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError(f"GPM output bins must overlap [-{A:g}, {A:g}]")
    return probs / row_sums