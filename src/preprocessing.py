"""
preprocessing.py
----------------
Data normalization utilities using Min-Max and Z-score (StandardScaler-style).
"""

import numpy as np


class MinMaxScaler:
    """Scale features to the [0, 1] range."""

    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        denom = self.max_ - self.min_
        denom[denom == 0] = 1  # avoid divide-by-zero for constant features
        return (X - self.min_) / denom

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        return X_scaled * (self.max_ - self.min_) + self.min_


class StandardScaler:
    """Standardize features to zero mean and unit variance."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1  # avoid divide-by-zero
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        return X_scaled * self.std_ + self.mean_


def add_polynomial_features(X: np.ndarray, degree: int = 2) -> np.ndarray:
    """
    Add polynomial and interaction features up to `degree`.

    For degree=2 and features [a, b, c], adds: a², b², c², ab, ac, bc.
    Returns the original features plus the new ones (no bias column added).
    """
    if degree < 2:
        return X

    n_samples, n_features = X.shape
    poly_cols = [X]

    # Squared terms
    if degree >= 2:
        poly_cols.append(X ** 2)

    # Interaction terms (degree 2 cross-products)
    if degree >= 2:
        for i in range(n_features):
            for j in range(i + 1, n_features):
                poly_cols.append((X[:, i] * X[:, j]).reshape(-1, 1))

    # Cubic terms
    if degree >= 3:
        poly_cols.append(X ** 3)

    return np.hstack(poly_cols)


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42):
    """Simple train/test split."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y))
    split = int(len(y) * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
