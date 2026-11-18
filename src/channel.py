from dataclasses import dataclass
import numpy as np
from .grid import validate_grid
from .patterns import pattern_matrix


@dataclass
class GridChannel:
    grid: np.ndarray
    outputs: np.ndarray
    P_grid: np.ndarray  # M rows, K columns

    def __post_init__(self):
        self.grid = validate_grid(self.grid)
        self.outputs = np.asarray(self.outputs, dtype=float)
        self.P_grid = np.asarray(self.P_grid, dtype=float)
        if self.outputs.ndim != 1 or self.P_grid.shape != (len(self.grid), len(self.outputs)):
            raise ValueError("expected P_grid shape (M,K)")
        if not len(self.outputs) or not np.all(np.isfinite(self.P_grid)) or not np.all(np.isfinite(self.outputs)):
            raise ValueError("nonfinite/empty channel")

    def save(self, path):
        np.savez_compressed(path, grid=self.grid, outputs=self.outputs, P_grid=self.P_grid)


def from_moments(epsilon, grid, result, drop_tol=1e-12):
    w, z = result["w"], result["z"]
    keep = w > drop_tol
    if not np.any(keep):
        raise ValueError("no positive primal weights")
    return GridChannel(grid, z[keep]/w[keep], pattern_matrix(result["bits"][keep], epsilon)*w[keep])


def merge_duplicate_outputs(channel):
    """Only identical floating values merge; no approximate support clustering."""
    y, inv = np.unique(channel.outputs, return_inverse=True)
    p = np.zeros((len(channel.grid), len(y)))
    for k, j in enumerate(inv):
        p[:, j] += channel.P_grid[:, k]
    return GridChannel(channel.grid, y, p)


def verify_grid_channel(epsilon, grid, outputs, P_grid, tolerance=1e-7):
    channel = GridChannel(grid, outputs, P_grid)
    p, y, x = channel.P_grid, channel.outputs, channel.grid
    normalization = float(np.max(np.abs(p.sum(axis=1)-1)))
    unbiasedness = float(np.max(np.abs(p@y-x)))
    ldp = float(np.max(p.max(axis=0)-np.exp(epsilon)*p.min(axis=0)))
    negative = float(max(0., -p.min()))
    second = p@(y*y)
    return dict(valid=max(normalization, unbiasedness, ldp, negative) <= tolerance,
                normalization=normalization, unbiasedness=unbiasedness, ldp=max(0.,ldp),
                negativity=negative, U_grid=float(max(second-x*x)),
                verification="floating diagnostic, not an exact certificate")
