import numpy as np


class MinMaxNormalization:
    def __init__(self, feature_range: tuple = (0, 1)) -> None:
        self.lower, self.upper = feature_range

    def fit(self, X: np.ndarray):
        self._min = X.min()
        self._max = X.max()
        print("min:", self._min, "max:", self._max)
        return self

    def transform(self, X: np.ndarray):
        X = 1. * (X - self._min) / (self._max - self._min)
        return X * (self.upper - self.lower) + self.lower

    def fit_transform(self, X: np.ndarray):
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray):
        X = (X - self.lower) / (self.upper - self.lower)
        return X * (self._max - self._min) + self._min
