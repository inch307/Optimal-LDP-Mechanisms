"""Exact histogram models and integration weights for the AVG experiment."""
from fractions import Fraction as F


def fractions(values):
    return [F(str(v)) for v in values]


class Prior:
    def __init__(self, edges, masses, model="continuous", name="histogram"):
        self.edges, self.masses = fractions(edges), fractions(masses)
        if model not in ("continuous", "discrete"):
            raise ValueError("model must be continuous or discrete")
        if (len(self.edges) != len(self.masses) + 1 or not self.masses
                or self.edges[0] != -1 or self.edges[-1] != 1
                or any(a >= b for a, b in zip(self.edges, self.edges[1:]))
                or min(self.masses) < 0 or sum(self.masses) <= 0):
            raise ValueError("histogram needs increasing edges from -1 to 1 and nonnegative, nonempty masses")
        total = sum(self.masses)
        self.masses = [v / total for v in self.masses]
        self.model, self.name = model, name

    @classmethod
    def uniform(cls):
        return cls([-1, 1], [1], name="uniform")

    def as_dict(self):
        return dict(edges=list(map(str, self.edges)), masses=list(map(str, self.masses)),
                    model=self.model, name=self.name)

    def moments(self, left=F(-1), right=F(1)):
        """Unnormalized moments (mass, first, second) on [left,right]."""
        result = [F(0), F(0), F(0)]
        for a, b, p in zip(self.edges, self.edges[1:], self.masses):
            if self.model == "discrete":
                x = (a + b) / 2
                if left <= x < right or x == right == 1:
                    for k in range(3):
                        result[k] += p * x**k
            else:
                lo, hi = max(left, a), min(right, b)
                if lo < hi and p:
                    for k in range(3):
                        result[k] += p * (hi**(k+1) - lo**(k+1)) / ((k+1)*(b-a))
        return result

    def hat_weights(self, grid):
        """Integrate the linear channel basis, including histogram/grid crossings."""
        x = fractions(grid)
        if x[0] != -1 or x[-1] != 1 or any(a >= b for a, b in zip(x, x[1:])):
            raise ValueError("channel grid must increase from -1 to 1")
        w = [F(0)] * len(x)
        for j, (a, b) in enumerate(zip(x, x[1:])):
            mass, first, _ = self.moments(a, b)
            w[j] += (b*mass-first)/(b-a)
            w[j+1] += (first-a*mass)/(b-a)
        assert min(w) >= 0 and sum(w) == 1
        return w

    def coarsen(self, partition):
        """Bin means + zero-weight endpoints; D is the exact within-bin variance."""
        masses = {F(-1): F(0), F(1): F(0)}
        if self.model == "discrete":
            for a, b, p in zip(self.edges, self.edges[1:], self.masses):
                if p:
                    masses[(a+b)/2] = masses.get((a+b)/2, F(0)) + p
        else:
            partition = fractions(partition)
            if partition[0] != -1 or partition[-1] != 1 or any(a >= b for a, b in zip(partition, partition[1:])):
                raise ValueError("invalid coarsening partition")
            for a, b in zip(partition, partition[1:]):
                p, first, _ = self.moments(a, b)
                if p:
                    center = first/p
                    masses[center] = masses.get(center, F(0)) + p
        x = sorted(masses)
        pi = [masses[v] for v in x]
        D = self.moments()[2] - sum(p*v*v for p, v in zip(pi, x))
        return x, pi, D


def uniform_grid(m):
    if m < 2:
        raise ValueError("M must be >= 2")
    return [F(-1) + F(2*j, m-1) for j in range(m)]
