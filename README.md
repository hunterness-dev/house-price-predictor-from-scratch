# 🏠 House Price Predictor — From Scratch

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A complete machine learning application that predicts house prices using **Linear Regression implemented from scratch** with NumPy and gradient descent — no scikit-learn for the model itself.

> **Test result:** Custom model achieves **R² = 0.957 / RMSE ≈ $21,597** on the test set, nearly identical to sklearn's Linear Regression, validating the from-scratch implementation.

---

## 📁 Project Structure

```
house-price-predictor-from-scratch/
├── app_streamlit.py                        # Main entry point (CLI)
├── data/
│   ├── house_prices.csv          # Synthetic dataset (300 rows)
│   └── generate_data.py          # Script that created the dataset
├── src/
│   ├── linear_regression.py      # Custom model (gradient descent)
│   ├── preprocessing.py          # Scalers + polynomial features
│   ├── metrics.py                # MSE, RMSE, MAE, R², MAPE
│   ├── visualizer.py
│   └── requirements.txt
│          # All matplotlib plots
├── models/
│   └── linear_regression.npz     # Saved model weights (auto-created)
└── plots/
    ├── predicted_vs_actual.png
    ├── loss_curve.png
    ├── feature_importance.png
    ├── residuals.png
    └── metrics_comparison.png
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python app.py
```

The script will:
1. Load the dataset
2. Train the custom linear regression model
3. Evaluate on the test set (MSE, RMSE, MAE, R²)
4. Compare results against sklearn's implementation
5. Generate 5 visualisation plots in `plots/`
6. Ask you to enter a house's features and print a predicted price

---

## ⚙️ CLI Options

| Flag | Description |
|------|-------------|
| `--poly 2` | Add degree-2 polynomial + interaction features |
| `--poly 3` | Add degree-3 polynomial features |
| `--load-model` | Load saved model instead of retraining |
| `--no-sklearn` | Skip sklearn comparison |
| `--no-cli` | Skip the interactive prediction prompt |

### Examples

```bash
# Linear regression (default)
python app.py

# Polynomial features (degree 2) — more expressive model
python app.py --poly 2

# Load a previously trained model, skip retraining
python app.py --load-model

# Train, evaluate, skip comparison and interactive prompt
python app.py --no-sklearn --no-cli
```

---

## 🧮 How the Model Works

### Linear Regression

The model learns a linear relationship between house features and price:

```
price = w₁·size + w₂·rooms + w₃·location_score + b
```

Where `w₁, w₂, w₃` are **weights** and `b` is the **bias**, both learned during training.

### Gradient Descent

Training minimises the **Mean Squared Error (MSE)** loss:

```
L = (1/n) Σ (y_pred - y_true)²
```

At each iteration, weights are updated using the gradients:

```
∂L/∂W = (2/n) · Xᵀ · (y_pred - y_true)
∂L/∂b = (2/n) · Σ(y_pred - y_true)

W ← W - α · ∂L/∂W
b ← b - α · ∂L/∂b
```

Where `α` is the **learning rate** (step size).

### L2 Regularisation (Ridge)

An optional L2 penalty discourages large weights, improving generalisation:

```
L_reg = MSE + λ · Σ wᵢ²
```

### Polynomial Features

With `--poly 2`, the model adds squared terms and cross-product (interaction) features:

```
[size, rooms, location_score]
→ [size, rooms, location_score, size², rooms², score², size×rooms, size×score, rooms×score]
```

This allows the model to capture non-linear relationships without changing the underlying linear algebra.

---

## 📊 Dataset

Synthetic dataset with 300 houses generated from:

```
price = 80·size + 15000·rooms + 12000·location_score + N(0, 20000) + 50000
```

| Column | Type | Range | Description |
|--------|------|--------|-------------|
| `size` | int | 500–4000 | House size in sq ft |
| `rooms` | int | 1–7 | Number of rooms |
| `location_score` | float | 1.0–10.0 | Neighbourhood quality |
| `price` | float | ~$130K–$600K | Sale price in USD |

---

## 📈 Visualisations

| Plot | Description |
|------|-------------|
| `predicted_vs_actual.png` | Scatter of predicted vs real prices (custom vs sklearn) |
| `loss_curve.png` | MSE loss decreasing over gradient descent iterations |
| `feature_importance.png` | Normalised absolute weight per feature |
| `residuals.png` | Residual scatter + residual distribution histogram |
| `metrics_comparison.png` | Side-by-side bar chart: custom vs sklearn metrics |

---

## 🔢 Example Output

```
=======================================================
   🏠  House Price Predictor — Training Pipeline
=======================================================

  ✅  Loaded dataset: 300 rows × 4 columns
      Price range: $130,600 – $602,100

  🔧  Training custom Linear Regression (gradient descent) …

  Iteration     0 | Loss:           0.81
  Iteration   100 | Loss:           0.06
  ...
  Iteration  2000 | Loss:           0.06  ✓ Training complete

  ────────────────────────────────────────
    📊  Custom Model (test set) Evaluation
  ────────────────────────────────────────
    MSE   : $ 466,451,393.66
    RMSE  : $      21,597.49
    MAE   : $      17,349.07
    R²    :           0.9573
    MAPE  :            5.31%
  ────────────────────────────────────────

  🏡  Interactive Price Predictor
  Enter house features to get a price estimate.

  House size (sq ft, 100–10000): 2000
  Number of rooms (1–20): 3
  Location score (1.0–10.0): 7.5

  ──────────────────────────────────────
  🏠  Size          :    2,000 sq ft
  🛏   Rooms         :        3
  📍  Location Score:      7.5 / 10
  💰  Predicted Price:   $323,450
  ──────────────────────────────────────
```

---

## 🛠️ Save & Load Model

The model is automatically saved to `models/linear_regression.npz` after training.

To reload it later:

```bash
python app.py --load-model
```

Or in Python:

```python
from src.linear_regression import LinearRegression
model = LinearRegression.load("models/linear_regression.npz")
predictions = model.predict(X_scaled)
```

---

## 📦 Dependencies

```
numpy      — numerical computing (model, preprocessing)
pandas     — CSV loading
matplotlib — visualisations
scikit-learn (optional) — comparison only, not used for training
```

Install with:
```bash
pip install -r requirements.txt
```
