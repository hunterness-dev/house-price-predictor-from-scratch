"""
visualizer.py
-------------
All matplotlib-based plots for the house price predictor.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

PLOTS_DIR = Path(__file__).parent.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

PALETTE = {
    "blue":   "#2563EB",
    "green":  "#16A34A",
    "red":    "#DC2626",
    "orange": "#EA580C",
    "purple": "#7C3AED",
    "gray":   "#6B7280",
    "bg":     "#F9FAFB",
}


def _save(fig: plt.Figure, name: str) -> None:
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  📈  Plot saved → {path}")
    plt.close(fig)


# ------------------------------------------------------------------ #
#  1. Predicted vs Actual                                             #
# ------------------------------------------------------------------ #

def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred_custom: np.ndarray,
    y_pred_sklearn: np.ndarray | None = None,
) -> None:
    """Scatter of predicted vs actual prices."""
    fig, axes = plt.subplots(1, 2 if y_pred_sklearn is not None else 1,
                             figsize=(13 if y_pred_sklearn is not None else 7, 6),
                             facecolor=PALETTE["bg"])

    if y_pred_sklearn is None:
        axes = [axes]

    def _scatter(ax, y_true, y_pred, title, color):
        ax.set_facecolor(PALETTE["bg"])
        ax.scatter(y_true, y_pred, alpha=0.55, color=color, edgecolors="white",
                   linewidths=0.5, s=60, label="Samples")
        mn = min(y_true.min(), y_pred.min())
        mx = max(y_true.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], "--", color=PALETTE["gray"], lw=1.5, label="Perfect fit")
        ax.set_xlabel("Actual Price ($)", fontsize=11)
        ax.set_ylabel("Predicted Price ($)", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.legend(fontsize=10)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
        ax.grid(True, alpha=0.3)

    _scatter(axes[0], y_true, y_pred_custom,
             "Custom Linear Regression\n(Predicted vs Actual)", PALETTE["blue"])
    if y_pred_sklearn is not None:
        _scatter(axes[1], y_true, y_pred_sklearn,
                 "sklearn LinearRegression\n(Predicted vs Actual)", PALETTE["green"])

    fig.suptitle("House Price Prediction — Model Comparison", fontsize=15,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "predicted_vs_actual.png")


# ------------------------------------------------------------------ #
#  2. Training Loss Curve                                             #
# ------------------------------------------------------------------ #

def plot_loss_curve(loss_history: list[float]) -> None:
    """Plot MSE loss over training iterations."""
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    iters = [i * 10 for i in range(len(loss_history))]
    ax.plot(iters, loss_history, color=PALETTE["blue"], lw=2)
    ax.fill_between(iters, loss_history, alpha=0.15, color=PALETTE["blue"])

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("MSE Loss", fontsize=11)
    ax.set_title("Gradient Descent — Training Loss Curve", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M" if x >= 1e6 else f"{x:.0f}"))
    ax.grid(True, alpha=0.3)

    _save(fig, "loss_curve.png")


# ------------------------------------------------------------------ #
#  3. Feature Importance                                              #
# ------------------------------------------------------------------ #

def plot_feature_importance(importances: dict[str, float]) -> None:
    """Horizontal bar chart of normalised feature importances."""
    names = list(importances.keys())
    values = list(importances.values())
    colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["purple"],
              PALETTE["green"], PALETTE["red"]][:len(names)]

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.9 + 1)),
                           facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    bars = ax.barh(names, values, color=colors, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=10, color=PALETTE["gray"])

    ax.set_xlabel("Relative Importance (|weight| normalised)", fontsize=11)
    ax.set_title("Feature Importance", fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(values) * 1.18)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    _save(fig, "feature_importance.png")


# ------------------------------------------------------------------ #
#  4. Residuals plot                                                  #
# ------------------------------------------------------------------ #

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Residual plot and residual distribution."""
    residuals = y_true - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=PALETTE["bg"])
    for ax in (ax1, ax2):
        ax.set_facecolor(PALETTE["bg"])

    # Residuals vs Predicted
    ax1.scatter(y_pred, residuals, alpha=0.5, color=PALETTE["orange"],
                edgecolors="white", linewidths=0.5, s=55)
    ax1.axhline(0, color=PALETTE["gray"], lw=1.5, linestyle="--")
    ax1.set_xlabel("Predicted Price ($)", fontsize=11)
    ax1.set_ylabel("Residual ($)", fontsize=11)
    ax1.set_title("Residuals vs Predicted", fontsize=13, fontweight="bold")
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
    ax1.grid(True, alpha=0.3)

    # Residual distribution
    ax2.hist(residuals, bins=30, color=PALETTE["purple"], edgecolor="white",
             linewidth=0.6, alpha=0.8)
    ax2.axvline(0, color=PALETTE["gray"], lw=1.5, linestyle="--")
    ax2.set_xlabel("Residual ($)", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Residual Distribution", fontsize=13, fontweight="bold")
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, "residuals.png")


# ------------------------------------------------------------------ #
#  5. Metrics comparison bar chart                                    #
# ------------------------------------------------------------------ #

def plot_metrics_comparison(
    custom_metrics: dict[str, float],
    sklearn_metrics: dict[str, float],
) -> None:
    """Side-by-side bar chart comparing model metrics."""
    metric_labels = list(custom_metrics.keys())
    x = np.arange(len(metric_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    bars1 = ax.bar(x - width / 2, list(custom_metrics.values()),
                   width, label="Custom (from scratch)", color=PALETTE["blue"],
                   edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + width / 2, list(sklearn_metrics.values()),
                   width, label="sklearn", color=PALETTE["green"],
                   edgecolor="white", linewidth=0.8)

    def _label_bars(bars):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.01,
                    f"{h:,.0f}", ha="center", va="bottom", fontsize=8,
                    color=PALETTE["gray"])

    _label_bars(bars1)
    _label_bars(bars2)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_title("Custom vs sklearn — Metric Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    _save(fig, "metrics_comparison.png")
