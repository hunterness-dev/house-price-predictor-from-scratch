"""
app_streamlit.py
----------------
House Price Predictor — UI fully integrated with the custom LinearRegression model.

File layout (place this file next to the project folder):

    house-price-predictor-from-scratch/
        src/linear_regression.py
        src/preprocessing.py
        src/metrics.py
        data/house_prices.csv
        models/linear_regression.npz
    app_streamlit.py    ← this file

Run:
    streamlit run app_streamlit.py
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Logging — visible in the terminal running `streamlit run`
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("house_predictor")

# ---------------------------------------------------------------------------
# Path setup — works regardless of where `streamlit run` is called from
# ---------------------------------------------------------------------------
_APP_DIR = Path(__file__).resolve().parent
_PROJECT = _APP_DIR / "house-price-predictor-from-scratch"

# Add the project root to sys.path so `from src.xxx import ...` works
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

# ---------------------------------------------------------------------------
# ML layer — everything model-related lives here, isolated from the UI
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelBundle:
    """Immutable container holding the trained model and fitted scalers."""
    model:     object   # LinearRegression instance
    scaler_X:  object   # StandardScaler fitted on training features
    scaler_y:  object   # StandardScaler fitted on training target
    feature_names: list[str]
    # Metrics computed on the hold-out test set at load/train time
    rmse: float
    mae:  float
    r2:   float
    trained_from: str   # "loaded" | "trained"


def _load_or_train() -> ModelBundle:
    """
    Try to load the pre-trained model from disk.
    If not found, train from scratch on the CSV.
    This function is called exactly once (wrapped in st.cache_resource).
    """
    from src.linear_regression import LinearRegression
    from src.preprocessing   import StandardScaler, train_test_split
    from src.metrics         import root_mean_squared_error, mean_absolute_error, r2_score

    FEATURE_COLS  = ["size", "rooms", "location_score"]
    TARGET_COL    = "price"
    DATA_PATH     = _PROJECT / "data"  / "house_prices.csv"
    MODEL_PATH    = _PROJECT / "models" / "linear_regression.npz"

    log.info("Loading dataset from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    X  = df[FEATURE_COLS].values.astype(float)
    y  = df[TARGET_COL].values.astype(float)

    # Fit scalers on the full dataset (same split seed as training script)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)

    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))

    X_train_s = scaler_X.transform(X_train)
    X_test_s  = scaler_X.transform(X_test)
    y_train_s = scaler_y.transform(y_train.reshape(-1, 1)).ravel()

    if MODEL_PATH.exists():
        log.info("Pre-trained model found — loading from %s", MODEL_PATH)
        model = LinearRegression.load(str(MODEL_PATH))
        source = "loaded"
    else:
        log.warning("No saved model at %s — training from scratch", MODEL_PATH)
        model = LinearRegression(
            learning_rate=0.05,
            n_iterations=2000,
            l2_lambda=0.001,
            verbose=False,          # suppress stdout; use logging instead
        )
        model.fit(X_train_s, y_train_s, feature_names=FEATURE_COLS)
        model.save(str(MODEL_PATH))
        log.info("Model trained and saved to %s", MODEL_PATH)
        source = "trained"

    # Evaluate on held-out test set
    y_pred_s = model.predict(X_test_s)
    y_pred   = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
    rmse = root_mean_squared_error(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    log.info("Model ready | source=%s RMSE=$%,.0f MAE=$%,.0f R²=%.4f",
             source, rmse, mae, r2)

    return ModelBundle(
        model=model,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        feature_names=FEATURE_COLS,
        rmse=rmse,
        mae=mae,
        r2=r2,
        trained_from=source,
    )


@st.cache_resource(show_spinner=False)
def get_model() -> ModelBundle | None:
    """
    Cached model loader — Streamlit calls this once per server session.
    Returns None on failure so the UI can show a graceful error.
    """
    try:
        return _load_or_train()
    except Exception as exc:
        log.exception("Model initialisation failed: %s", exc)
        return None


def run_prediction(bundle: ModelBundle, size: float, rooms: int, loc: float) -> dict:
    """
    Pure ML function: preprocess → predict → postprocess.
    Returns a dict with price and per-feature dollar contributions.

    Contribution formula (in dollar space):
        contrib_i = w_i * ((x_i - mean_i) / std_i) * std_y

    The base is the dataset mean of y plus the (near-zero) bias term.
    """
    X_raw = np.array([[size, rooms, loc]], dtype=float)

    log.debug("Raw input: size=%.0f, rooms=%d, loc=%.1f", size, rooms, loc)

    X_scaled = bundle.scaler_X.transform(X_raw)
    y_scaled  = bundle.model.predict(X_scaled)  # shape (1,)
    price     = float(
        bundle.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()[0]
    )

    log.info("Prediction: $%,.0f (scaled=%.4f)", price, float(y_scaled[0]))

    # Per-feature dollar contributions
    std_y   = float(bundle.scaler_y.std_.ravel()[0])
    mean_y  = float(bundle.scaler_y.mean_.ravel()[0])
    weights = bundle.model.weights          # shape (n_features,)
    x_scaled_row = X_scaled[0]             # shape (n_features,)

    contribs = {}
    for feat, w, xs in zip(bundle.feature_names, weights, x_scaled_row):
        contribs[feat] = float(w * xs * std_y)

    base = mean_y + float(bundle.model.bias) * std_y

    log.debug("Contributions: %s | base=$%,.0f", contribs, base)

    return {
        "price":   price,
        "contribs": contribs,   # {feature_name: dollar_value}  — can be negative
        "base":    base,
    }


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _fmt_price(v: float) -> str:
    return f"${v:,.0f}"


def _location_hint(score: float) -> str:
    if score < 3.5:   return "🔴 Below average"
    if score < 6.5:   return "🟡 Average"
    if score < 8.5:   return "🟢 Good"
    return "⭐ Prime"


def _contrib_bar(label: str, raw_value: str, dollar: float, price: float) -> str:
    """Render one contribution row with a proportional bar."""
    frac_pct = min(100.0, max(0.0, abs(dollar) / price * 100))
    sign_tag = "" if dollar >= 0 else "<span style='color:#e85555;font-size:0.7rem;'> ▼</span>"
    grad = "linear-gradient(90deg,#e8a030,#f0c060)" if dollar >= 0 \
           else "linear-gradient(90deg,#c03030,#e85555)"
    return f"""
    <div style='margin-bottom:1rem;'>
        <div style='display:flex;justify-content:space-between;margin-bottom:0.3rem;'>
            <span style='font-size:0.83rem;color:#9a9890;'>{label}{sign_tag}</span>
            <span style='font-size:0.83rem;font-weight:600;color:#e8e6e1;'>
                {raw_value} &nbsp;<span style='color:#4a4d5a;font-weight:400;'>
                ({_fmt_price(dollar)})</span>
            </span>
        </div>
        <div style='height:4px;background:#0d0f14;border-radius:99px;overflow:hidden;'>
            <div style='height:100%;width:{frac_pct:.1f}%;
                        background:{grad};border-radius:99px;'></div>
        </div>
    </div>"""


# ---------------------------------------------------------------------------
# Page config — MUST be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0d0f14 !important;
    color: #e8e6e1 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #13151c !important;
    border-right: 1px solid #1e2130 !important;
}
[data-testid="stSidebar"] * { color: #c9c7c0 !important; }
[data-testid="stSidebar"] h2 {
    font-family: 'DM Serif Display', serif !important;
    color: #f0c060 !important;
    font-size: 1.4rem !important;
}
[data-testid="stSidebar"] hr { border-color: #1e2130 !important; margin: 1.2rem 0 !important; }

/* ── Main container ── */
.block-container { padding: 0 3rem 3rem !important; max-width: 1100px !important; }

/* ── Hero ── */
.hero { padding: 3.5rem 0 2rem; border-bottom: 1px solid #1e2130; margin-bottom: 2.5rem; }
.hero-eyebrow {
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.15em;
    text-transform: uppercase; color: #f0c060; margin-bottom: 0.6rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2.4rem, 5vw, 3.8rem);
    line-height: 1.1; color: #f5f3ee; margin: 0 0 0.8rem; letter-spacing: -0.02em;
}
.hero-title em { color: #f0c060; font-style: italic; }
.hero-subtitle { font-size: 1.05rem; color: #7a7870; max-width: 520px; line-height: 1.6; }

/* ── Cards ── */
.card {
    background: #13151c; border: 1px solid #1e2130;
    border-radius: 16px; padding: 2rem 2.2rem; margin-bottom: 1.5rem;
    transition: border-color 0.2s;
}
.card:hover { border-color: #2d3148; }
.card-title {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.13em;
    text-transform: uppercase; color: #4a4d5a; margin-bottom: 1.4rem;
}

/* ── Inputs ── */
[data-testid="stNumberInput"] input {
    background: #0d0f14 !important; border: 1.5px solid #1e2130 !important;
    border-radius: 10px !important; color: #e8e6e1 !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 1rem !important;
    padding: 0.6rem 1rem !important; transition: border-color 0.2s !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: #f0c060 !important;
    box-shadow: 0 0 0 3px rgba(240,192,96,0.08) !important; outline: none !important;
}
[data-testid="stNumberInput"] label,
[data-testid="stSlider"] label {
    color: #9a9890 !important; font-size: 0.82rem !important;
    font-weight: 500 !important; letter-spacing: 0.04em !important;
    text-transform: uppercase !important; margin-bottom: 0.3rem !important;
}

/* ── Slider thumb & track ── */
[data-testid="stSlider"] [role="slider"] { background: #f0c060 !important; border-color: #f0c060 !important; }

/* ── Predict button ── */
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #f0c060 0%, #e8a030 100%) !important;
    color: #0d0f14 !important; font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important; font-size: 1rem !important;
    letter-spacing: 0.04em !important; border: none !important;
    border-radius: 12px !important; padding: 0.85rem 2.5rem !important;
    width: 100% !important; cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 24px rgba(240,192,96,0.22) !important;
}
div[data-testid="stButton"] button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(240,192,96,0.35) !important;
}

/* ── Result box ── */
.result-box {
    background: linear-gradient(135deg, #181b24 0%, #13151c 100%);
    border: 1.5px solid #f0c060; border-radius: 20px;
    padding: 2.5rem 2rem; text-align: center; position: relative;
    overflow: hidden; animation: fadeSlideUp 0.5s ease forwards;
}
.result-box::before {
    content: ''; position: absolute; top: -60px; left: 50%;
    transform: translateX(-50%); width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(240,192,96,0.08) 0%, transparent 70%);
    pointer-events: none;
}
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0);    }
}
.result-label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; color: #4a4d5a; margin-bottom: 0.5rem; }
.result-price { font-family: 'DM Serif Display', serif; font-size: clamp(2.8rem, 6vw, 4.5rem); color: #f0c060; line-height: 1; margin: 0.3rem 0 0.8rem; letter-spacing: -0.02em; }
.result-sub   { font-size: 0.85rem; color: #4a4d5a; }
.result-icon  { font-size: 2rem; margin-bottom: 0.8rem; display: block; filter: drop-shadow(0 0 12px rgba(240,192,96,0.4)); }

/* ── Summary chips ── */
.stat-row  { display: flex; gap: 0.8rem; flex-wrap: wrap; margin-top: 0.5rem; }
.stat-chip { flex: 1; min-width: 90px; background: #0d0f14; border: 1px solid #1e2130; border-radius: 12px; padding: 0.9rem 1rem; text-align: center; }
.stat-chip-value { font-family: 'DM Serif Display', serif; font-size: 1.5rem; color: #f5f3ee; display: block; }
.stat-chip-label { font-size: 0.68rem; font-weight: 500; letter-spacing: 0.1em; text-transform: uppercase; color: #4a4d5a; margin-top: 2px; display: block; }

/* ── Range bar ── */
.range-bar-wrap  { margin-top: 1.2rem; }
.range-bar-labels { display: flex; justify-content: space-between; font-size: 0.72rem; color: #4a4d5a; margin-bottom: 0.35rem; }
.range-bar       { height: 6px; background: #1e2130; border-radius: 99px; overflow: hidden; position: relative; }
.range-bar-fill  { height: 100%; background: linear-gradient(90deg,#e8a030,#f0c060); border-radius: 99px; transition: width 0.6s ease; }
.range-bar-marker { width: 12px; height: 12px; background: #f0c060; border: 2px solid #0d0f14; border-radius: 50%; position: absolute; top: -3px; transform: translateX(-50%); box-shadow: 0 0 8px rgba(240,192,96,0.5); }

/* ── Status badges ── */
.badge {
    display: inline-block; border-radius: 20px; padding: 2px 10px;
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.08em;
}
.badge-green  { background: rgba(22,163,74,0.15);  color: #4ade80 !important; border: 1px solid rgba(22,163,74,0.3);  }
.badge-yellow { background: rgba(240,192,96,0.12); color: #f0c060 !important; border: 1px solid rgba(240,192,96,0.3); }
.badge-red    { background: rgba(220,38,38,0.12);  color: #f87171 !important; border: 1px solid rgba(220,38,38,0.3);  }

/* ── Footer ── */
.footer { margin-top: 4rem; padding: 2rem 0 1rem; border-top: 1px solid #1e2130; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem; }
.footer-left { font-size: 0.82rem; color: #3a3d4a; }
.footer-left strong { color: #5a5d6a; }
.footer-right { display: flex; gap: 1.5rem; }
.footer-link  { font-size: 0.78rem; color: #3a3d4a; text-decoration: none; }

/* ── Misc ── */
[data-testid="stAlert"]             { background: #13151c !important; border: 1px solid #1e2130 !important; border-radius: 12px !important; color: #7a7870 !important; }
[data-testid="stSpinner"]           { color: #f0c060 !important; }
hr                                  { border-color: #1e2130 !important; }
small, [data-testid="stCaptionContainer"] { color: #4a4d5a !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load model (cached — runs once per server session)
# ---------------------------------------------------------------------------
with st.spinner("Initialising model…"):
    bundle = get_model()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏠 House Price Predictor")
    st.markdown("---")

    # Model status
    if bundle is None:
        st.markdown("""
        <span class='badge badge-red'>⚠ Model unavailable</span>
        <p style='font-size:0.82rem;color:#7a7870;margin-top:0.6rem;'>
        Could not load the model. Check the terminal for details.
        </p>""", unsafe_allow_html=True)
    else:
        source_badge = "badge-green" if bundle.trained_from == "loaded" else "badge-yellow"
        source_text  = "Pre-trained model" if bundle.trained_from == "loaded" else "Trained on startup"
        st.markdown(f"<span class='badge {source_badge}'>✓ {source_text}</span>",
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    <p style='font-size:0.87rem;line-height:1.7;color:#7a7870;'>
    Estimates residential property prices using a Linear Regression model
    built from scratch with NumPy gradient descent — no sklearn for inference.
    </p>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Model Metrics")

    if bundle:
        metrics_rows = [
            ("Algorithm",    "Linear Regression"),
            ("Optimizer",    "Gradient Descent"),
            ("Regulariz.",   "L2 / Ridge  λ=0.001"),
            ("R² Score",     f"{bundle.r2:.4f}"),
            ("RMSE",         f"${bundle.rmse:,.0f}"),
            ("MAE",          f"${bundle.mae:,.0f}"),
            ("Features",     ", ".join(bundle.feature_names)),
        ]
    else:
        metrics_rows = [("Status", "Unavailable")]

    for label, val in metrics_rows:
        st.markdown(f"""
        <div style='margin-bottom:0.65rem;'>
            <span style='font-size:0.68rem;letter-spacing:0.1em;text-transform:uppercase;color:#4a4d5a;'>{label}</span><br>
            <span style='font-size:0.88rem;color:#c9c7c0;font-weight:500;'>{val}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <p style='font-size:0.72rem;color:#3a3d4a;line-height:1.6;'>
    Built with NumPy · pandas · Streamlit<br>
    Trained on synthetic housing data (300 samples).
    </p>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown("""
<div class='hero'>
    <div class='hero-eyebrow'>AI · Real Estate · Analytics</div>
    <h1 class='hero-title'>Estimate your home's<br><em>market value</em></h1>
    <p class='hero-subtitle'>
        Enter a few property details and our gradient-descent–trained
        linear regression model returns an instant price estimate.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Fatal error — model could not be initialised
# ---------------------------------------------------------------------------
if bundle is None:
    st.error(
        "**Model initialisation failed.** "
        "Make sure the `house-price-predictor-from-scratch/` folder is next to this file "
        "and that `data/house_prices.csv` exists. "
        "Check the terminal for the full traceback."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Two-column layout
# ---------------------------------------------------------------------------
col_form, col_result = st.columns([1.05, 0.95], gap="large")


# ── LEFT: inputs ─────────────────────────────────────────────────────────────
with col_form:
    st.markdown("<div class='card'><div class='card-title'>Property Details</div>",
                unsafe_allow_html=True)

    house_size = st.number_input(
        "House Size (sq ft)",
        min_value=100,
        max_value=10_000,
        value=1_800,
        step=50,
        help="Total liveable area in square feet",
    )

    num_rooms = st.number_input(
        "Number of Rooms",
        min_value=1,
        max_value=20,
        value=3,
        step=1,
        help="Total number of bedrooms and living rooms",
    )

    location_score = st.slider(
        "Location Score",
        min_value=1.0,
        max_value=10.0,
        value=6.5,
        step=0.1,
        help="Neighbourhood quality: 1 = poor, 10 = prime",
    )
    st.markdown(f"<small>{_location_hint(location_score)}</small>",
                unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Live summary chips — update instantly on widget change
    st.markdown(f"""
    <div class='stat-row' style='margin-bottom:1.5rem;'>
        <div class='stat-chip'>
            <span class='stat-chip-value'>{house_size:,}</span>
            <span class='stat-chip-label'>sq ft</span>
        </div>
        <div class='stat-chip'>
            <span class='stat-chip-value'>{num_rooms}</span>
            <span class='stat-chip-label'>rooms</span>
        </div>
        <div class='stat-chip'>
            <span class='stat-chip-value'>{location_score:.1f}</span>
            <span class='stat-chip-label'>loc. score</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    predict_clicked = st.button("🔍  Predict Price", use_container_width=True)


# ── RIGHT: result ─────────────────────────────────────────────────────────────
with col_result:

    if predict_clicked:
        # Input validation
        if house_size < 100:
            st.warning("⚠️  Please enter a house size of at least 100 sq ft.")
        elif num_rooms < 1:
            st.warning("⚠️  Number of rooms must be at least 1.")
        else:
            with st.spinner("Running model…"):
                time.sleep(0.6)   # intentional pause for UX feedback
                result = run_prediction(bundle, float(house_size), int(num_rooms), location_score)

            # Persist result across Streamlit reruns triggered by widget changes
            st.session_state["result"]      = result
            st.session_state["last_inputs"] = (house_size, num_rooms, location_score)

    if "result" in st.session_state:
        result   = st.session_state["result"]
        price    = result["price"]
        contribs = result["contribs"]   # dict: feature → dollar contribution
        base     = result["base"]

        inp_size, inp_rooms, inp_loc = st.session_state["last_inputs"]

        lo_price = price * 0.90
        hi_price = price * 1.10
        pct = max(0.0, min(1.0, (price - 130_000) / (602_000 - 130_000))) * 100

        # Main result card
        st.markdown(f"""
        <div class='result-box'>
            <span class='result-icon'>💰</span>
            <div class='result-label'>Estimated Market Value</div>
            <div class='result-price'>{_fmt_price(price)}</div>
            <div class='result-sub'>
                Confidence range &nbsp;·&nbsp;
                <strong style='color:#c9c7c0;'>{_fmt_price(lo_price)} – {_fmt_price(hi_price)}</strong>
            </div>
            <div class='range-bar-wrap'>
                <div class='range-bar-labels'>
                    <span>$130K<br><span style='font-size:0.65rem;'>Low</span></span>
                    <span style='text-align:center;'>Market Position<br>
                        <strong style='color:#f0c060;font-size:0.78rem;'>{pct:.0f}th percentile</strong>
                    </span>
                    <span style='text-align:right;'>$602K<br><span style='font-size:0.65rem;'>High</span></span>
                </div>
                <div class='range-bar'>
                    <div class='range-bar-fill' style='width:{pct:.1f}%;'></div>
                    <div class='range-bar-marker' style='left:{pct:.1f}%;'></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Price breakdown — derived from actual model weights, not hardcoded coefficients
        st.markdown("<div class='card'><div class='card-title'>Price Breakdown (from model weights)</div>",
                    unsafe_allow_html=True)

        breakdown_rows = [
            ("📐 Size",          f"{inp_size:,} sq ft",   contribs.get("size",           0.0)),
            ("🛏 Rooms",         f"{inp_rooms} rooms",     contribs.get("rooms",          0.0)),
            ("📍 Location",      f"score {inp_loc:.1f}",  contribs.get("location_score", 0.0)),
            ("🏗 Base (intercept)", "dataset mean",       base),
        ]

        for label, raw_val, dollar in breakdown_rows:
            st.markdown(_contrib_bar(label, raw_val, dollar, price), unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        # Empty state placeholder
        st.markdown("""
        <div style='height:340px;display:flex;flex-direction:column;
                    align-items:center;justify-content:center;
                    border:1.5px dashed #1e2130;border-radius:20px;
                    color:#3a3d4a;text-align:center;padding:2rem;'>
            <div style='font-size:3rem;margin-bottom:1rem;filter:grayscale(1);opacity:0.4;'>🏠</div>
            <div style='font-size:0.85rem;line-height:1.7;'>
                Fill in the property details<br>on the left, then click<br>
                <strong style='color:#4a4d5a;'>Predict Price</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# How it works expander
# ---------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📖  How the model works"):
    c1, c2, c3 = st.columns(3)
    for col, (title, body) in zip([c1, c2, c3], [
        ("Gradient Descent",
         "Weights are updated each iteration by following the negative gradient of the MSE loss — minimising prediction error step by step."),
        ("L2 Regularisation",
         "A Ridge penalty term (λ=0.001) discourages large weights, improving generalisation to unseen property inputs."),
        ("Feature Scaling",
         "Inputs are standardised to zero mean and unit variance before training. Predictions are inverse-transformed back to dollar scale."),
    ]):
        col.markdown(f"""
        <div class='card' style='margin-bottom:0;'>
            <div class='card-title'>{title}</div>
            <p style='font-size:0.85rem;color:#7a7870;line-height:1.65;margin:0;'>{body}</p>
        </div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<div class='footer'>
    <div class='footer-left'>
        <strong>House Price Predictor</strong> &nbsp;·&nbsp; Linear Regression from scratch
        &nbsp;·&nbsp; Built with NumPy &amp; Streamlit
    </div>
    <div class='footer-right'>
        <a class='footer-link' href='#'>Documentation</a>
        <a class='footer-link' href='#'>Source Code</a>
        <a class='footer-link' href='#'>Model Card</a>
    </div>
</div>
""", unsafe_allow_html=True)
