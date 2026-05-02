"""
metrics.py
----------
Evaluation metrics for regression models.
"""

import numpy as np


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error (MSE)."""
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error (MAE)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of Determination (R²).
    R² = 1 means perfect fit; 0 means the model is as good as predicting the mean.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0
    return float(1 - ss_res / ss_tot)


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (MAPE) as a percentage."""
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def print_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "Model") -> None:
    """Pretty-print all regression metrics."""
    print(f"\n{'─' * 40}")
    print(f"  📊  {label} Evaluation")
    print(f"{'─' * 40}")
    print(f"  MSE   : ${mean_squared_error(y_true, y_pred):>15,.2f}")
    print(f"  RMSE  : ${root_mean_squared_error(y_true, y_pred):>15,.2f}")
    print(f"  MAE   : ${mean_absolute_error(y_true, y_pred):>15,.2f}")
    print(f"  R²    : {r2_score(y_true, y_pred):>16.4f}")
    print(f"  MAPE  : {mean_absolute_percentage_error(y_true, y_pred):>15.2f}%")
    print(f"{'─' * 40}")
