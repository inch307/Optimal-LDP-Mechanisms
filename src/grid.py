import numpy as np


def validate_grid(grid):
    x = np.asarray(grid, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.all(np.isfinite(x)):
        raise ValueError("grid must be a finite 1D array with at least two points")
    if x[0] != -1 or x[-1] != 1 or np.any(np.diff(x) <= 0):
        raise ValueError("grid must strictly increase from -1 to 1")
    return x


def make_grid(m, kind="uniform"):
    if m < 3:
        raise ValueError("initial M must be >= 3")
    x = np.linspace(-1, 1, m)
    if kind == "nonuniform":
        x = np.sign(x) * np.abs(x) ** 1.5
    elif kind != "uniform":
        raise ValueError(kind)
    return validate_grid(x)


def refine_grid(grid, maximizers, max_points=3, min_spacing=1e-10):
    """Always spend the first slot on a longest-interval midpoint."""
    x = validate_grid(grid)
    if max_points < 1:
        raise ValueError("max_points must be positive")
    j = int(np.argmax(np.diff(x)))
    candidates = [(x[j] + x[j + 1]) / 2] + list(maximizers)
    added = []
    for point in candidates:
        if -1 < point < 1 and min(abs(point - z) for z in [*x, *added]) > min_spacing:
            added.append(float(point))
        if len(added) >= max_points:
            break
    return validate_grid(sorted([*x, *added]))
