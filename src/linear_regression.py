"""
linear_regression.py
--------------------
Linear Regression implemented from scratch using NumPy and gradient descent.

Supports:
  - Ordinary gradient descent
  - Mini-batch gradient descent
  - L2 regularization (Ridge)
  - Model save / load (via numpy .npz)
  - Feature importance (coefficient magnitudes)
"""

import numpy as np
import json
from pathlib import Path


class LinearRegression:
    """
    Linear Regression via gradient descent.

    Parameters
    ----------
    learning_rate : float
        Step size for gradient descent updates.
    n_iterations : int
        Number of training epochs.
    batch_size : int or None
        Mini-batch size. None = full-batch gradient descent.
    l2_lambda : float
        L2 regularization strength (0 = no regularization).
    verbose : bool
        Print loss every 100 iterations when True.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        batch_size: int | None = None,
        l2_lambda: float = 0.0,
        verbose: bool = True,
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.l2_lambda = l2_lambda
        self.verbose = verbose

        # Learned parameters
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []
        self.feature_names: list[str] | None = None

    # ------------------------------------------------------------------ #
    #  Core gradient descent                                               #
    # ------------------------------------------------------------------ #

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """MSE loss with optional L2 penalty."""
        mse = np.mean((y_true - y_pred) ** 2)
        if self.l2_lambda > 0 and self.weights is not None:
            mse += self.l2_lambda * np.sum(self.weights ** 2)
        return float(mse)

    def _compute_gradients(
        self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Return (dW, db) gradients."""
        n = len(y)
        error = y_pred - y                          # (n,)
        dW = (2 / n) * X.T @ error                 # (n_features,)
        db = (2 / n) * np.sum(error)               # scalar

        # L2 gradient contribution (regularise weights only, not bias)
        if self.l2_lambda > 0:
            dW += 2 * self.l2_lambda * self.weights

        return dW, db

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None) -> "LinearRegression":
        """
        Train the model using gradient descent.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        feature_names : list[str], optional
            Names of input features for interpretability.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []
        self.feature_names = feature_names

        use_mini_batch = self.batch_size is not None and self.batch_size < n_samples

        for iteration in range(self.n_iterations):
            if use_mini_batch:
                # Shuffle and pick a mini-batch
                idx = np.random.permutation(n_samples)[: self.batch_size]
                X_batch, y_batch = X[idx], y[idx]
            else:
                X_batch, y_batch = X, y

            y_pred = self._predict_raw(X_batch)
            dW, db = self._compute_gradients(X_batch, y_batch, y_pred)

            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * db

            # Record full-batch loss every 10 iterations
            if iteration % 10 == 0:
                full_pred = self._predict_raw(X)
                loss = self._compute_loss(y, full_pred)
                self.loss_history.append(loss)

                if self.verbose and iteration % 100 == 0:
                    print(f"  Iteration {iteration:>5} | Loss: {loss:>14,.2f}")

        if self.verbose:
            final_loss = self._compute_loss(y, self._predict_raw(X))
            print(f"  Iteration {self.n_iterations:>5} | Loss: {final_loss:>14,.2f}  ✓ Training complete")

        return self

    # ------------------------------------------------------------------ #
    #  Prediction                                                          #
    # ------------------------------------------------------------------ #

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for input X."""
        if self.weights is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")
        return self._predict_raw(X)

    # ------------------------------------------------------------------ #
    #  Feature importance                                                  #
    # ------------------------------------------------------------------ #

    def feature_importance(self) -> dict[str, float]:
        """
        Return feature importance as the absolute value of each weight,
        normalised so they sum to 1.
        """
        if self.weights is None:
            raise RuntimeError("Model not trained.")
        abs_weights = np.abs(self.weights)
        total = abs_weights.sum()
        importances = abs_weights / total if total > 0 else abs_weights

        names = self.feature_names or [f"feature_{i}" for i in range(len(self.weights))]
        return dict(zip(names, importances.tolist()))

    # ------------------------------------------------------------------ #
    #  Save / Load                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """
        Save model weights and hyper-parameters to a .npz file.

        Usage: model.save("models/my_model.npz")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iterations,
            "l2_lambda": self.l2_lambda,
            "feature_names": self.feature_names or [],
        }

        np.savez(
            path,
            weights=self.weights,
            bias=np.array([self.bias]),
            loss_history=np.array(self.loss_history),
            meta=np.array([json.dumps(meta)]),
        )
        print(f"  💾  Model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "LinearRegression":
        """
        Load a previously saved model.

        Usage: model = LinearRegression.load("models/my_model.npz")
        """
        data = np.load(path, allow_pickle=True)
        meta = json.loads(str(data["meta"][0]))

        model = cls(
            learning_rate=meta["learning_rate"],
            n_iterations=meta["n_iterations"],
            l2_lambda=meta["l2_lambda"],
        )
        model.weights = data["weights"]
        model.bias = float(data["bias"][0])
        model.loss_history = data["loss_history"].tolist()
        model.feature_names = meta["feature_names"] or None

        print(f"  📂  Model loaded ← {path}")
        return model

    # ------------------------------------------------------------------ #
    #  Dunder helpers                                                      #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        trained = self.weights is not None
        return (
            f"LinearRegression(lr={self.learning_rate}, "
            f"iters={self.n_iterations}, "
            f"l2={self.l2_lambda}, "
            f"trained={trained})"
        )
