import numpy as np
from .channel import GridChannel


class PiecewiseLinearChannel(GridChannel):
    def probabilities(self, values):
        values = np.atleast_1d(values).astype(float)
        if not np.all(np.isfinite(values)) or np.any(np.abs(values) > 1):
            raise ValueError("inputs must lie in [-1,1]")
        j = np.clip(np.searchsorted(self.grid, values, side="right")-1, 0, len(self.grid)-2)
        t = (values-self.grid[j])/(self.grid[j+1]-self.grid[j])
        return (1-t[:,None])*self.P_grid[j] + t[:,None]*self.P_grid[j+1]

    def variance(self, values):
        values = np.atleast_1d(values)
        return self.probabilities(values)@(self.outputs**2)-values**2


def interpolate_channel(grid, outputs, P_grid):
    return PiecewiseLinearChannel(grid, outputs, P_grid)


def continuous_worst(channel):
    x, y, p = channel.grid, channel.outputs, channel.P_grid
    second = p@(y*y)
    slopes = np.diff(second)/np.diff(x)
    stars = np.clip(slopes/2, x[:-1], x[1:])
    values = second[:-1]+slopes*(stars-x[:-1])-stars**2
    order = np.argsort(-values)
    return dict(U_cont=float(max(values)), U_grid=float(max(second-x*x)),
                maximizers=stars[order].tolist(), interval_values=values.tolist(),
                Delta=float(max(np.diff(x))))
