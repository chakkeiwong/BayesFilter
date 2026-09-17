"""Independent diagnostic legacy probe stream for matched numerical fixtures."""

import numpy as np


class FrozenLegacyGeometryStream:
    """Replay the original seeded clouds without altering any fit arithmetic."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(int(seed[0]) ^ (int(seed[1]) << 16))

    def normal(self, *, size):
        return self.rng.normal(size=size)

    def permutation(self, count):
        return self.rng.permutation(count)

    def ball(self, rows, dimension, *, radius, minimum_uniform=0.0):
        directions = self.rng.normal(size=(rows, dimension))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        while np.any(norms == 0.0):
            zero = np.flatnonzero(norms[:, 0] == 0.0)
            directions[zero] = self.rng.normal(size=(zero.size, dimension))
            norms = np.linalg.norm(directions, axis=1, keepdims=True)
        radial = radius * self.rng.uniform(minimum_uniform, 1.0, size=(rows, 1)) ** (
            1.0 / dimension
        )
        return directions / norms * radial
