from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np

from .tdigest_rs import create_from_array


@dataclass
class TDigest:
    means: np.ndarray
    weights: np.ndarray
    delta: float

    def __repr__(self) -> str:
        return ", ".join(f"(centroid={c}, num_points={n})" for _, c, n in self._iter())

    def _iter(self) -> Iterator[Tuple[int, float, int]]:
        for i, (c, n) in enumerate(zip(self.means, self.weights, strict=True)):
            yield i, c, n

    @property
    def total_weight(self) -> int:
        return int(np.sum(self.weights))

    def __len__(self) -> int:
        return len(self.means)

    @classmethod
    def create(cls, arr: np.ndarray, delta: float, sort_in_place: bool = False) -> "TDigest":
        if sort_in_place:
            arr.sort()
        means, weights = create_from_array(arr, delta)
        return cls(means=means, weights=weights, delta=delta)

    def merge(self, other: "TDigest") -> "TDigest":
        raise NotImplementedError("merge method not implemented")

    def quantile(self, x: float) -> float:
        if len(self.means) < 3:
            return 0.0

        q = x * self.total_weight
        m = len(self)
        cum_weight = 0
        for i, _mean, weight in self._iter():
            if cum_weight + weight > q:
                if i == 0:
                    delta = self.means[i + 1] - _mean
                elif i == m - 1:
                    delta = _mean - self.means[i - 1]
                else:
                    delta = (self.means[i + 1] - self.means[i - 1]) / 2
                return _mean + ((q - cum_weight) / (weight) - 0.5) * delta
            cum_weight += weight

        return _mean
