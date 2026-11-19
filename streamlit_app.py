
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    import shap
except ImportError:
    shap = None

try:
    from retention_engine import generate_retention_plan
except ImportError:
    generate_retention_plan = None


# ---------------------------------------------------------------------
# Page and project paths
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="ChurnIQ | Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium visual system. This changes presentation only; model logic is unchanged.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#17243b; --muted:#718096; --line:#e7edf5; --navy:#101d35; --blue:#3767f5; --cyan:#38bdf8; --surface:#ffffff; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:linear-gradient(180deg,#f5f8fd 0%,#f8faff 42%,#f4f7fc 100%); color:var(--ink); }
.block-container { max-width:1500px; padding-top:1.55rem; padding-bottom:3rem; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#101d35 0%,#172b4d 100%); border-right:1px solid rgba(255,255,255,.08); }
[data-testid="stSidebar"] * { color:#e9f0ff !important; }
[data-testid="stSidebar"] [data-testid="stMetric"] { background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.12); border-radius:14px; padding:12px 14px; margin:8px 0; }
[data-testid="stSidebar"] [data-testid="stMetricLabel"] p { color:#aebfda !important; font-size:.78rem; }
[data-testid="stSidebar"] [data-testid="stMetricValue"] { color:#fff !important; font-weight:800; }
[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.16); }
.hero { position:relative; overflow:hidden; padding:28px 32px; border-radius:22px; margin:0 0 22px 0; color:#fff; background:radial-gradient(circle at 85% 10%,rgba(56,189,248,.35),transparent 28%),linear-gradient(115deg,#101d35 0%,#1d3764 58%,#3159a8 100%); box-shadow:0 16px 38px rgba(20,43,82,.16); }
.hero:after { content:''; position:absolute; width:230px; height:230px; right:-70px; bottom:-150px; border:1px solid rgba(255,255,255,.18); border-radius:50%; box-shadow:0 0 0 28px rgba(255,255,255,.035),0 0 0 58px rgba(255,255,255,.025); }
.hero-kicker { color:#9fc4ff; font-size:.76rem; letter-spacing:.16em; text-transform:uppercase; font-weight:800; margin-bottom:8px; }
.hero h1 { color:#fff; font-family:'Manrope',sans-serif; font-size:2.05rem; font-weight:800; letter-spacing:-.04em; margin:0 0 8px 0; }
.hero p { color:#d6e4fb; font-size:.98rem; margin:0; max-width:760px; }
.hero-tag { display:inline-block; margin-top:17px; padding:6px 11px; border:1px solid rgba(255,255,255,.2); border-radius:999px; color:#e5efff; background:rgba(255,255,255,.09); font-size:.76rem; }
h1,h2,h3,h4 { font-family:'Manrope',sans-serif; color:#17243b; letter-spacing:-.025em; }
h2 { font-size:1.45rem !important; }
h3 { font-size:1.15rem !important; }
p, label, .stMarkdown { color:#34445c; }
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap:8px; border-bottom:1px solid var(--line); padding:0 0 8px 0; }
[data-testid="stTabs"] button[data-baseweb="tab"] { border:1px solid #e4eaf3; border-radius:11px; background:#fff; padding:10px 15px; color:#53647d; font-weight:700; }
[data-testid="stTabs"] button[aria-selected="true"] { background:#eaf0ff; border-color:#c8d6ff; color:#2854d8; }
[data-testid="stMetric"] { background:#fff; border:1px solid #e7edf5; border-radius:16px; padding:17px 18px; box-shadow:0 5px 18px rgba(27,49,84,.045); }
[data-testid="stMetricLabel"] p { color:#72819a; font-size:.82rem; font-weight:600; }
[data-testid="stMetricValue"] { color:#17243b; font-family:'Manrope',sans-serif; font-weight:800; }
[data-testid="stForm"], [data-testid="stFileUploader"] section, [data-testid="stExpander"] { border:1px solid #e5ebf4; border-radius:15px; background:#fff; }
[data-testid="stForm"] { padding:18px 20px; }
.stButton button, .stDownloadButton button, [data-testid="stFormSubmitButton"] button { border-radius:11px; font-weight:700; min-height:2.7rem; border:0; box-shadow:0 5px 14px rgba(55,103,245,.16); }
.stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] { background:linear-gradient(100deg,#315cf4,#477cfb); color:#fff; }
.stButton button:hover, .stDownloadButton button:hover { transform:translateY(-1px); transition:.15s ease; }
[data-testid="stDataFrame"] { border:1px solid #e5ebf4; border-radius:13px; overflow:hidden; }
[data-testid="stPlotlyChart"] { background:#fff; border:1px solid #e7edf5; border-radius:16px; padding:8px; box-shadow:0 5px 18px rgba(27,49,84,.035); }
[data-testid="stAlert"] { border-radius:13px; }
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input, [data-testid="stSelectbox"] > div { border-radius:10px; }
.footer { text-align:center; color:#8b98aa; font-size:.78rem; padding:26px 0 4px; border-top:1px solid #e7edf5; margin-top:34px; }
@media (max-width: 800px) { .block-container{padding-left:1rem;padding-right:1rem;} .hero{padding:22px;} .hero h1{font-size:1.55rem;} [data-testid="stTabs"] [data-baseweb="tab-list"]{gap:4px;} [data-testid="stTabs"] button[data-baseweb="tab"]{padding:8px 9px;font-size:.78rem;} }
</style>
""", unsafe_allow_html=True)


# CHURNIQ signature art direction: editorial ivory + ink + signal-lime.
st.markdown("""
<style>
:root {--ink:#17231e;--muted:#738078;--line:#e2e8df;--paper:#f3f5ef;--white:#ffffff;--forest:#142b24;--lime:#c8f169;--violet:#7768e8;}
.stApp {background:#f1f3ed;color:#17231e;}
.block-container {max-width:1560px;padding-top:1.1rem;padding-bottom:3.5rem;}
[data-testid="stSidebar"] {background:linear-gradient(160deg,#10231d 0%,#172f27 58%,#203d32 100%);border-right:1px solid #29483b;}
[data-testid="stSidebar"] * {color:#eef4e9 !important;}
[data-testid="stSidebar"] [data-testid="stMetric"] {background:rgba(255,255,255,.055);border:1px solid rgba(220,245,211,.13);border-radius:13px;box-shadow:none;}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] p {color:#a9bbae !important;}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {color:#d2f58a !important;}
.hero {position:relative;overflow:hidden;padding:36px 40px;border-radius:7px;margin:0 0 24px;background:linear-gradient(112deg,#10231d 0%,#19372b 62%,#294b39 100%);box-shadow:0 18px 48px rgba(23,48,36,.16);border:1px solid #2a4939;}
.hero:before {content:'CQ';position:absolute;right:7%;top:-53px;font-family:Manrope,sans-serif;font-size:240px;font-weight:800;letter-spacing:-.12em;color:rgba(226,255,205,.045);pointer-events:none;}
.hero:after {content:'';position:absolute;right:8%;bottom:24px;width:82px;height:5px;background:#c8f169;box-shadow:0 0 32px rgba(200,241,105,.4);border-radius:5px;}
.hero-kicker {color:#c8f169;font-size:.68rem;letter-spacing:.22em;font-weight:800;}
.hero h1 {color:#f5f7ef;font-family:Manrope,sans-serif;font-size:2.55rem;line-height:1.08;letter-spacing:-.065em;margin:8px 0 12px;max-width:760px;}
.hero p {color:#c2d1c5;font-size:1rem;max-width:690px;line-height:1.65;}
.hero-tag {color:#d8f3b2;background:rgba(200,241,105,.09);border:1px solid rgba(200,241,105,.27);border-radius:4px;font-size:.7rem;letter-spacing:.06em;text-transform:uppercase;}
h1,h2,h3,h4 {font-family:Manrope,sans-serif;color:#17231e;letter-spacing:-.045em;}
h2 {font-size:1.55rem !important;}
h3 {font-size:1.13rem !important;}
p,label,.stMarkdown {color:#405047;}
[data-testid="stTabs"] [data-baseweb="tab-list"] {gap:5px;border-bottom:1px solid #dce4d9;padding:0 0 9px;}
[data-testid="stTabs"] button[data-baseweb="tab"] {border:1px solid transparent;border-radius:4px;background:transparent;padding:11px 15px;color:#6b786e;font-weight:700;}
[data-testid="stTabs"] button[aria-selected="true"] {background:#18352a;color:#d5f48d;border-color:#18352a;}
[data-testid="stMetric"] {background:#fff;border:1px solid #e2e8df;border-radius:5px;padding:19px 20px;box-shadow:0 7px 22px rgba(28,47,34,.045);}
[data-testid="stMetricLabel"] p {color:#758177;font-size:.72rem;font-weight:800;letter-spacing:.09em;text-transform:uppercase;}
[data-testid="stMetricValue"] {color:#172b22;font-family:Manrope,sans-serif;font-size:1.8rem;font-weight:800;letter-spacing:-.05em;}
[data-testid="stForm"], [data-testid="stFileUploader"] section, [data-testid="stExpander"] {border:1px solid #e1e8df;border-radius:5px;background:#fff;}
[data-testid="stForm"] {padding:22px 23px;}
.stButton button,.stDownloadButton button,[data-testid="stFormSubmitButton"] button {border-radius:4px;font-weight:800;min-height:2.8rem;border:1px solid #19382c;box-shadow:none;letter-spacing:.015em;}
.stButton button[kind="primary"],[data-testid="stFormSubmitButton"] button[kind="primary"] {background:#c8f169;color:#15271d;border:1px solid #c8f169;}
.stButton button[kind="primary"]:hover,[data-testid="stFormSubmitButton"] button[kind="primary"]:hover {background:#d8ff83;border-color:#d8ff83;color:#15271d;}
.stDownloadButton button {background:#18352a;color:#eaf4e5;border:1px solid #18352a;}
[data-testid="stDataFrame"] {border:1px solid #e1e8df;border-radius:5px;overflow:hidden;}
[data-testid="stPlotlyChart"] {background:#fff;border:1px solid #e2e8df;border-radius:5px;padding:8px;box-shadow:0 7px 22px rgba(28,47,34,.035);}
[data-testid="stAlert"] {border-radius:5px;}
[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input,[data-testid="stSelectbox"] > div {border-radius:4px;}
[data-testid="stProgress"] > div > div {background:#c8f169;}
.footer {text-align:left;color:#829087;font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;padding:22px 0 4px;border-top:1px solid #dce4d9;margin-top:36px;}
@media(max-width:800px){.block-container{padding-left:1rem;padding-right:1rem}.hero{padding:25px 23px}.hero h1{font-size:1.8rem}.hero:before{font-size:130px;right:0;top:0}[data-testid="stTabs"] button[data-baseweb="tab"]{padding:8px 8px;font-size:.72rem}}
</style>
""", unsafe_allow_html=True)

# Final polish overrides: keep interactive controls aligned with the
# ChurnIQ forest / signal-lime identity.
st.markdown("""
<style>
[data-testid="stFormSubmitButton"] button,
.stButton button[kind="primary"] {
  background: #c8f169 !important;
  color: #15271d !important;
  border: 1px solid #c8f169 !important;
  border-radius: 5px !important;
  font-weight: 800 !important;
  box-shadow: none !important;
}
[data-testid="stFormSubmitButton"] button:hover,
.stButton button[kind="primary"]:hover {
  background: #d8ff83 !important;
  border-color: #d8ff83 !important;
  color: #15271d !important;
}
[data-testid="stForm"] {
  border-color: #dce5d9 !important;
  box-shadow: 0 10px 30px rgba(28, 47, 34, .045);
}
.cq-result-card {
  background: #fff; border: 1px solid #e2e8df; border-radius: 7px;
  padding: 24px 26px; min-height: 245px;
  box-shadow: 0 8px 24px rgba(28, 47, 34, .045);
}
.cq-account-card {
  background: #fff; border: 1px solid #e2e8df; border-radius: 5px;
  padding: 15px 16px; min-height: 105px;
}
.cq-account-label {
  color: #758177; font-size: .70rem; font-weight: 800;
  letter-spacing: .09em; text-transform: uppercase; margin-bottom: 8px;
}
.cq-account-value {
  color: #172b22; font-family: Manrope, sans-serif;
  font-size: 1.28rem; line-height: 1.3; font-weight: 800;
  overflow-wrap: anywhere; word-break: normal;
}
</style>
""", unsafe_allow_html=True)

# ChurnIQ signature navigation: editorial, quiet, high-contrast tabs.
st.markdown("""
<style>
/* A restrained navigation rail with consistent spacing and no emoji clutter. */
[data-testid="stTabs"] {
  margin-top: 1.05rem;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  display: flex;
  flex-wrap: nowrap;
  gap: 5px !important;
  align-items: stretch;
  padding: 7px !important;
  background: #e7ebe2 !important;
  border: 1px solid #d9e1d5 !important;
  border-radius: 8px !important;
  overflow-x: auto;
  scrollbar-width: thin;
}
[data-testid="stTabs"] button[data-baseweb="tab"] {
  position: relative;
  flex: 1 0 auto;
  justify-content: center;
  min-height: 46px;
  padding: 0 16px !important;
  border: 1px solid transparent !important;
  border-radius: 5px !important;
  background: transparent !important;
  color: #647168 !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: .70rem !important;
  font-weight: 800 !important;
  letter-spacing: .075em !important;
  white-space: nowrap;
  transition: background .18s ease, color .18s ease, transform .18s ease;
}
[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
  background: rgba(255,255,255,.72) !important;
  color: #18352a !important;
}
[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
  background: #18352a !important;
  color: #d7f58e !important;
  border-color: #18352a !important;
  box-shadow: 0 3px 10px rgba(20,43,36,.13) !important;
}
[data-testid="stTabs"] button[data-baseweb="tab"] p {
  color: inherit !important;
  font-size: inherit !important;
  font-weight: inherit !important;
  letter-spacing: inherit !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
  background: #c8f169 !important;
  height: 3px !important;
}
[data-testid="stTabs"] [data-baseweb="tab-border"] {
  background: transparent !important;
}
@media (max-width: 900px) {
  [data-testid="stTabs"] [data-baseweb="tab-list"] {
    justify-content: flex-start;
  }
  [data-testid="stTabs"] button[data-baseweb="tab"] {
    min-height: 42px;
    padding: 0 12px !important;
    font-size: .65rem !important;
  }
}
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "churn_model_v2.joblib"
DATA_PATH = BASE_DIR / "data" / "cleaned_data.csv"
EXPLANATION_DIR = BASE_DIR / "explanation_results"
MONITORING_DIR = BASE_DIR / "monitoring"
PREDICTION_HISTORY_PATH = MONITORING_DIR / "prediction_history.csv"

st.markdown("""
<div class="hero">
  <div class="hero-kicker">FIELD NOTES / 01 &nbsp; — &nbsp; CUSTOMER INTELLIGENCE</div>
  <h1>See the signal.<br>Keep the customer.</h1>
  <p>A decision cockpit for churn risk — combining predictive scoring, explainable signals and portfolio intelligence in one focused workspace.</p>
  <span class="hero-tag">● V2 MODEL ONLINE &nbsp; / &nbsp; EXPLAINABLE AI &nbsp; / &nbsp; RETENTION INTELLIGENCE</span>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Load the V2 model bundle and dataset
# ---------------------------------------------------------------------
@st.cache_resource
def load_model_bundle(model_path: str, modified_time: float):
    return joblib.load(model_path)


@st.cache_data
def load_dataset(data_path: str, modified_time: float):
    return pd.read_csv(data_path)


if not MODEL_PATH.is_file():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.info(
        "Expected file: models/churn_model_v2.joblib. "
        "Keep the trained V2 model in the project's models folder."
    )
    st.stop()

if not DATA_PATH.is_file():
    st.error(f"Dataset file not found: {DATA_PATH}")
    st.info("Expected file: data/cleaned_data.csv.")
    st.stop()

try:
    bundle = load_model_bundle(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    df = load_dataset(str(DATA_PATH), DATA_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not load the model or dataset: {exc}")
    st.stop()

if not isinstance(bundle, dict) or "model" not in bundle:
    st.error("The saved model file does not contain the expected V2 model bundle.")
    st.stop()

model = bundle["model"]
threshold = float(bundle.get("threshold", 0.5))

feature_columns = bundle.get("feature_columns")
if not feature_columns:
    feature_columns = (
        list(bundle.get("numeric_features", []))
        + list(bundle.get("categorical_features", []))
    )

if not feature_columns:
    st.error("The model bundle does not contain its expected feature columns.")
    st.stop()

missing_dataset_columns = [c for c in feature_columns if c not in df.columns]
if missing_dataset_columns:
    st.error(
        "The cleaned dataset is missing columns required by the V2 model: "
        + ", ".join(missing_dataset_columns)
    )
    st.stop()

X_dataset = df.loc[:, feature_columns].copy()


def churn_as_binary(series: pd.Series) -> pd.Series:
    """Convert common Telco churn labels to 0/1."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)

    labels = series.astype(str).str.strip().str.lower()
    return labels.map(
        {"yes": 1, "no": 0, "1": 1, "0": 0}
    ).fillna(0).astype(int)


def predict_probabilities(input_frame: pd.DataFrame) -> np.ndarray:
    """Return probability of Churn=Yes using the fitted V2 pipeline."""
    clean_frame = input_frame.loc[:, feature_columns].copy()
    probabilities = model.predict_proba(clean_frame)[:, 1]
    return np.asarray(probabilities, dtype=float)


def clean_batch_data(batch_frame: pd.DataFrame) -> pd.DataFrame:
    """Clean uploaded Telco data before model prediction."""
    cleaned = batch_frame.copy()

    # Trim strings and convert empty / whitespace-only cells to NaN.
    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[column] = cleaned[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
        cleaned[column] = cleaned[column].replace(r"^\\s*$", np.nan, regex=True)

    # Explicitly convert Telco numeric fields that may arrive as strings.
    for column in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    # Match numeric feature dtypes from the reference training dataset.
    for column in feature_columns:
        if column in cleaned.columns and column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    missing_columns = [column for column in feature_columns if column not in cleaned.columns]
    if missing_columns:
        raise ValueError(
            "The uploaded CSV is missing required model columns: "
            + ", ".join(missing_columns)
        )

    model_frame = cleaned.loc[:, feature_columns].copy()
    all_missing_numeric = [
        column for column in model_frame.columns
        if column in df.columns
        and pd.api.types.is_numeric_dtype(df[column])
        and model_frame[column].isna().all()
    ]
    if all_missing_numeric:
        raise ValueError(
            "These numeric columns contain no usable values: "
            + ", ".join(all_missing_numeric)
        )

    return model_frame


def risk_label(probability: float) -> str:
    if probability >= threshold:
        return "High"
    if probability >= max(0.0, threshold - 0.10):
        return "Medium"
    return "Low"


def log_prediction(probability: float, source: str, count: int = 1, predicted_churn_count: int = 0) -> None:
    """Append privacy-conscious prediction metadata; raw customer fields are not stored."""
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "customers_scored": int(count),
        "average_churn_probability": float(probability),
        "predicted_churn_count": int(predicted_churn_count),
        "threshold": float(threshold),
        "model_file": MODEL_PATH.name,
    }])
    row.to_csv(
        PREDICTION_HISTORY_PATH,
        mode="a",
        header=not PREDICTION_HISTORY_PATH.exists(),
        index=False,
    )


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI for numeric/categorical distributions, using smoothed proportions."""
    ref = reference.dropna()
    cur = current.dropna()
    if ref.empty or cur.empty:
        return float("nan")
    if pd.api.types.is_numeric_dtype(ref) and pd.api.types.is_numeric_dtype(cur):
        edges = np.unique(np.quantile(ref.astype(float), np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            edges = np.linspace(float(ref.min()), float(ref.max()) + 1e-9, min(bins, 5) + 1)
        edges[0] = -np.inf
        edges[-1] = np.inf
        ref_bins = pd.cut(ref.astype(float), bins=edges, include_lowest=True).value_counts(sort=False)
        cur_bins = pd.cut(cur.astype(float), bins=edges, include_lowest=True).value_counts(sort=False)
        expected = ref_bins.to_numpy(dtype=float)
        actual = cur_bins.to_numpy(dtype=float)
    else:
        ref_values = ref.astype(str).str.strip().replace("", "<missing>")
        cur_values = cur.astype(str).str.strip().replace("", "<missing>")
        categories = sorted(set(ref_values.unique()) | set(cur_values.unique()))
        expected = ref_values.value_counts().reindex(categories, fill_value=0).to_numpy(dtype=float)
        actual = cur_values.value_counts().reindex(categories, fill_value=0).to_numpy(dtype=float)
    expected = (expected + 0.5) / (expected.sum() + 0.5 * len(expected))
    actual = (actual + 0.5) / (actual.sum() + 0.5 * len(actual))
    return float(np.sum((actual - expected) * np.log(actual / expected)))


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 2px 18px;">
      <div style="font-size:1.65rem;font-weight:800;letter-spacing:-.04em;color:#fff;">◈ Churn<span style="color:#79b5ff;">IQ</span></div>
      <div style="font-size:.75rem;color:#aebfda;letter-spacing:.12em;text-transform:uppercase;margin-top:3px;">Customer intelligence</div>
    </div>
    <div style="height:1px;background:rgba(255,255,255,.14);margin-bottom:16px;"></div>
    """, unsafe_allow_html=True)
    st.markdown("**MODEL STATUS**")
    st.markdown("<span style='color:#86efac;font-weight:700;'>● V2 pipeline loaded</span>", unsafe_allow_html=True)
    st.metric("Decision threshold", f"{threshold:.2%}")
    st.metric("Input features", len(feature_columns))
    st.divider()
    st.caption(f"Model artifact: {MODEL_PATH.name}")
    st.caption("Preprocessing is included in the fitted pipeline.")
    st.caption("Predictions are estimates, not guarantees.")


# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------
tab_predict, tab_dashboard, tab_explain, tab_batch, tab_monitor, tab_drift, tab_about = st.tabs(
    [
        "01  INDIVIDUAL",
        "02  INTELLIGENCE",
        "03  EXPLAINABILITY",
        "04  BATCH STUDIO",
        "05  MONITORING",
        "06  DATA HEALTH",
        "07  MODEL LAB",
    ]
)


# ---------------------------------------------------------------------
# Individual prediction
# ---------------------------------------------------------------------
with tab_predict:
    st.subheader("Assess an individual customer")
    st.write(
        "Enter the customer's original Telco account details. "
        "The V2 pipeline performs the required preprocessing automatically."
    )

    with st.form("customer_prediction_form"):
        st.markdown("#### Core customer details")
        st.caption(
            "Enter the key account details. The remaining model inputs are pre-filled "
            "with sensible defaults and can be adjusted under Advanced details."
        )

        core1, core2, core3 = st.columns(3)
        with core1:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
        with core2:
            internet = st.selectbox(
                "Internet service", ["DSL", "Fiber optic", "No"]
            )
            monthly = st.number_input(
                "Monthly charges",
                min_value=0.0,
                max_value=1000.0,
                value=70.0,
                step=1.0,
            )
        with core3:
            total = st.number_input(
                "Total charges",
                min_value=0.0,
                max_value=100000.0,
                value=840.0,
                step=10.0,
            )
            st.markdown(
                "<div style='padding-top:8px;color:#758177;font-size:.82rem;'>"
                "Five key fields shown. All 19 model features are still supplied."
                "</div>",
                unsafe_allow_html=True,
            )

        with st.expander("Advanced customer details · 14 additional model inputs"):
            st.caption(
                "These fields are included in the trained model. Defaults are provided; "
                "change them when you know the customer's actual details."
            )
            adv1, adv2, adv3 = st.columns(3)

            with adv1:
                gender = st.selectbox("Gender", ["Female", "Male"], index=0)
                senior = st.selectbox(
                    "Senior citizen", [0, 1],
                    format_func=lambda x: "Yes" if x else "No",
                )
                partner = st.selectbox("Partner", ["No", "Yes"], index=0)
                dependents = st.selectbox("Dependents", ["No", "Yes"], index=0)
                phone = st.selectbox("Phone service", ["Yes", "No"], index=0)

            with adv2:
                multiple_lines = st.selectbox(
                    "Multiple lines", ["No", "Yes", "No phone service"], index=0
                )
                online_security = st.selectbox(
                    "Online security", ["No", "Yes", "No internet service"], index=0
                )
                online_backup = st.selectbox(
                    "Online backup", ["No", "Yes", "No internet service"], index=0
                )
                device_protection = st.selectbox(
                    "Device protection", ["No", "Yes", "No internet service"], index=0
                )
                tech_support = st.selectbox(
                    "Tech support", ["No", "Yes", "No internet service"], index=0
                )

            with adv3:
                streaming_tv = st.selectbox(
                    "Streaming TV", ["No", "Yes", "No internet service"], index=0
                )
                streaming_movies = st.selectbox(
                    "Streaming movies", ["No", "Yes", "No internet service"], index=0
                )
                paperless = st.selectbox("Paperless billing", ["Yes", "No"], index=0)
                payment = st.selectbox(
                    "Payment method",
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ],
                    index=0,
                )

        submitted = st.form_submit_button(
            "Assess churn risk", type="primary", width="stretch"
        )

    if submitted:
        customer_values = {
            "gender": gender,
            "SeniorCitizen": int(senior),
            "Partner": partner,
            "Dependents": dependents,
            "tenure": int(tenure),
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": float(monthly),
            "TotalCharges": float(total),
        }

        input_row = pd.DataFrame(
            [{column: customer_values[column] for column in feature_columns}]
        )

        try:
            probability = float(predict_probabilities(input_row)[0])
            prediction = int(probability >= threshold)
            log_prediction(probability, "individual", 1, prediction)

            # Premium customer risk result: visual presentation only; prediction is unchanged.
            risk = risk_label(probability)
            risk_colors = {"High": "#dc4655", "Medium": "#d99524", "Low": "#15966b"}
            risk_color = risk_colors[risk]
            st.divider()
            st.markdown("### Customer risk assessment")
            result_left, result_right = st.columns([1.05, 1.55], gap="large")

            with result_left:
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number={"suffix": "%", "font": {"size": 43, "color": "#17243b"}},
                    title={"text": "Estimated churn probability", "font": {"size": 15, "color": "#64748b"}},
                    gauge={
                        "axis": {"range": [0, 100], "ticksuffix": "%", "tickcolor": "#94a3b8"},
                        "bar": {"color": risk_color, "thickness": 0.24},
                        "bgcolor": "white",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, max(0, (threshold - 0.10) * 100)], "color": "#e6f5ef"},
                            {"range": [max(0, (threshold - 0.10) * 100), threshold * 100], "color": "#fff3d9"},
                            {"range": [threshold * 100, 100], "color": "#fde8eb"},
                        ],
                        "threshold": {"line": {"color": "#18352a", "width": 3}, "thickness": 0.8, "value": threshold * 100},
                    },
                ))
                gauge.update_layout(height=365, margin=dict(l=28, r=28, t=72, b=34), paper_bgcolor="rgba(0,0,0,0)", font={"family": "DM Sans, sans-serif"})
                st.plotly_chart(gauge, width="stretch", config={"displayModeBar": False})

            with result_right:
                outcome = "Likely to churn" if prediction else "Likely to stay"
                st.markdown(
                    f"<div class='cq-result-card'>"
                    f"<div style='font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;color:#758177;font-weight:800;'>MODEL ASSESSMENT</div>"
                    f"<div style='display:flex;align-items:center;gap:10px;margin:13px 0 7px;'>"
                    f"<span style='background:{risk_color}18;color:{risk_color};border:1px solid {risk_color}55;border-radius:999px;padding:6px 12px;font-size:.82rem;font-weight:800;'>{risk.upper()} RISK</span>"
                    f"<span style='color:#718096;font-size:.82rem;'>Threshold {threshold:.1%}</span></div>"
                    f"<div style='font-family:Manrope,sans-serif;font-size:1.8rem;font-weight:800;color:#172b22;margin:14px 0 7px;'>{outcome}</div>"
                    f"<div style='color:#758177;font-size:.94rem;line-height:1.7;'>The model estimates a {probability:.1%} probability of churn for the account details entered.</div>"
                    f"</div>", unsafe_allow_html=True)
                st.markdown("#### Account snapshot")
                p1, p2, p3 = st.columns([0.9, 1.0, 1.35], gap="medium")
                with p1:
                    st.markdown(
                        f"<div class='cq-account-card'><div class='cq-account-label'>TENURE</div>"
                        f"<div class='cq-account-value'>{tenure} months</div></div>",
                        unsafe_allow_html=True,
                    )
                with p2:
                    st.markdown(
                        f"<div class='cq-account-card'><div class='cq-account-label'>MONTHLY CHARGE</div>"
                        f"<div class='cq-account-value'>${monthly:,.2f}</div></div>",
                        unsafe_allow_html=True,
                    )
                with p3:
                    st.markdown(
                        f"<div class='cq-account-card'><div class='cq-account-label'>CONTRACT</div>"
                        f"<div class='cq-account-value'>{contract}</div></div>",
                        unsafe_allow_html=True,
                    )

            st.caption("The gauge's dark marker is the saved model decision threshold. Predictions are statistical estimates, not guarantees or proof that a customer will leave.")

            # ---------------------------------------------------------
            # FIXED INDIVIDUAL EXPLANATION
            # Uses permutation impact, not SHAP.
            # This method supports categorical text and numeric values.
            # ---------------------------------------------------------
            feature_impact = None

            if len(X_dataset) < 2:
                st.info(
                    "Not enough customer records to calculate feature impact."
                )
            else:
                try:
                    background = X_dataset.sample(
                        n=min(20, len(X_dataset)),
                        random_state=42,
                    )

                    baseline_probability = float(
                        predict_probabilities(input_row)[0]
                    )

                    impact_rows = []

                    for feature in feature_columns:
                        changed_rows = []

                        for replacement_value in background[feature].tolist():
                            changed = input_row.copy()
                            changed.loc[:, feature] = replacement_value
                            changed_rows.append(changed)

                        if not changed_rows:
                            continue

                        alternatives = pd.concat(
                            changed_rows,
                            ignore_index=True,
                        )

                        alternative_probabilities = predict_probabilities(
                            alternatives
                        )

                        mean_alternative = float(
                            np.mean(alternative_probabilities)
                        )

                        impact = (
                            baseline_probability - mean_alternative
                        )

                        impact_rows.append(
                            {
                                "Feature": feature,
                                "Customer value": input_row.iloc[0][feature],
                                "Permutation impact": impact,
                                "Absolute impact": abs(impact),
                            }
                        )

                    feature_impact = pd.DataFrame(impact_rows)

                    feature_impact = feature_impact.sort_values(
                        "Absolute impact",
                        ascending=False,
                    )

                    st.markdown(
                        "#### What is influencing this prediction?"
                    )

                    st.caption(
                        "Permutation impact compares this customer's prediction "
                        "with predictions where one feature at a time is replaced "
                        "by values sampled from other customers. It is a model "
                        "sensitivity estimate, not SHAP and not proof of causation."
                    )

                    positive = feature_impact[
                        feature_impact["Permutation impact"] > 0
                    ]

                    negative = feature_impact[
                        feature_impact["Permutation impact"] < 0
                    ]

                    chart_impacts = feature_impact.head(10).sort_values("Permutation impact")
                    impact_fig = px.bar(
                        chart_impacts,
                        x="Permutation impact",
                        y="Feature",
                        orientation="h",
                        color="Permutation impact",
                        color_continuous_scale=["#2563eb", "#dbeafe", "#dc4655"],
                        labels={"Permutation impact": "Change in estimated churn probability", "Feature": "Customer attribute"},
                        title="Top customer-specific model sensitivities",
                    )
                    impact_fig.update_layout(
                        height=390,
                        margin=dict(l=12, r=18, t=55, b=35),
                        coloraxis_showscale=False,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    impact_fig.update_xaxes(tickformat=".0%", zeroline=True, zerolinecolor="#94a3b8", gridcolor="#e9eef6")
                    impact_fig.update_yaxes(showgrid=False)
                    st.plotly_chart(impact_fig, width="stretch", config={"displayModeBar": False})

                    left, right = st.columns(2)

                    with left:
                        st.markdown(
                            "**Values associated with higher churn score**"
                        )

                        st.dataframe(
                            positive[
                                [
                                    "Feature",
                                    "Customer value",
                                    "Permutation impact",
                                ]
                            ].head(8),
                            hide_index=True,
                            width="stretch",
                        )

                    with right:
                        st.markdown(
                            "**Values associated with lower churn score**"
                        )

                        st.dataframe(
                            negative[
                                [
                                    "Feature",
                                    "Customer value",
                                    "Permutation impact",
                                ]
                            ].head(8),
                            hide_index=True,
                            width="stretch",
                        )

                    with st.expander("View all feature impacts"):
                        st.dataframe(
                            feature_impact.drop(
                                columns=["Absolute impact"]
                            ),
                            hide_index=True,
                            width="stretch",
                        )

                except Exception as exc:
                    st.warning(
                        f"Feature impact explanation is unavailable: {exc}"
                    )

            # ---------------------------------------------------------
            # RETENTION ACTION PLAN
            # ---------------------------------------------------------
            if (
                generate_retention_plan is not None
                and feature_impact is not None
                and not feature_impact.empty
            ):
                try:
                    retention_input = feature_impact.rename(
                        columns={
                            "Permutation impact": "permutation impact"
                        }
                    )[
                        [
                            "Feature",
                            "Customer value",
                            "permutation impact",
                        ]
                    ].copy()

                    retention_plan = generate_retention_plan(
                        probability,
                        threshold,
                        retention_input,
                    )

                    st.divider()
                    st.markdown("### Retention playbook")

                    st.metric(
                        "Customer risk level",
                        retention_plan["risk_level"],
                    )

                    st.write(retention_plan["summary"])

                    st.markdown("**Recommended retention actions**")

                    st.dataframe(
                        pd.DataFrame(
                            retention_plan["recommendations"]
                        ),
                        hide_index=True,
                        width="stretch",
                    )

                    st.caption(
                        "These are heuristic decision-support suggestions. "
                        "Permutation impact is not SHAP, and recommendations "
                        "are not guaranteed to prevent churn."
                    )

                except Exception as exc:
                    st.info(
                        f"Retention suggestions could not be generated: {exc}"
                    )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# ---------------------------------------------------------------------
# Business dashboard
# ---------------------------------------------------------------------
with tab_dashboard:
    st.markdown("### Portfolio performance")
    st.caption("A business-level view of customer retention, tenure and billing patterns.")

    if "Churn" not in df.columns:
        st.warning(
            "The dataset has no Churn target column, "
            "so churn metrics are unavailable."
        )
    else:
        churn_binary = churn_as_binary(df["Churn"])
        customer_count = len(df)
        churn_count = int(churn_binary.sum())
        stayed_count = customer_count - churn_count
        churn_rate = churn_count / customer_count if customer_count else 0.0
        retention_rate = 1 - churn_rate

        chart_df = df.copy()
        chart_df["Churn status"] = np.where(churn_binary.eq(1), "Churned", "Stayed")
        chart_df["_churn"] = churn_binary

        # KPI row: four compact business metrics.
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total customers", f"{customer_count:,}", help="Customers in the analysed dataset")
        k2.metric("Customers retained", f"{stayed_count:,}", f"{retention_rate:.1%} of portfolio")
        k3.metric("Customers churned", f"{churn_count:,}", f"{churn_rate:.1%} of portfolio")
        k4.metric("Overall churn rate", f"{churn_rate:.1%}", help="Share of customers marked as churned")

        st.markdown("#### Customer movement and engagement")
        left, right = st.columns([0.9, 1.35], gap="large")

        with left:
            status_counts = pd.DataFrame({
                "Customer status": ["Stayed", "Churned"],
                "Customers": [stayed_count, churn_count],
            })
            fig_status = px.pie(
                status_counts,
                names="Customer status",
                values="Customers",
                hole=0.68,
                color="Customer status",
                color_discrete_map={"Stayed": "#2563eb", "Churned": "#7dd3fc"},
            )
            fig_status.update_traces(
                textposition="inside", textinfo="percent",
                hovertemplate="%{label}<br>%{value:,} customers<br>%{percent}<extra></extra>",
                marker=dict(line=dict(color="#ffffff", width=3)),
            )
            fig_status.add_annotation(
                text=f"<b>{customer_count:,}</b><br><span style='font-size:11px'>customers</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=19, color="#17243b"),
            )
            fig_status.update_layout(
                title="Portfolio composition", height=340,
                margin=dict(l=12, r=12, t=55, b=15),
                legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_status, width="stretch", config={"displayModeBar": False})

        with right:
            if "tenure" in chart_df.columns:
                fig_tenure = px.histogram(
                    chart_df,
                    x="tenure",
                    color="Churn status",
                    nbins=24,
                    barmode="overlay",
                    opacity=0.78,
                    color_discrete_map={"Stayed": "#2563eb", "Churned": "#7dd3fc"},
                    labels={"tenure": "Customer tenure (months)", "count": "Customers"},
                )
                fig_tenure.update_layout(
                    title="Customer tenure distribution", height=340,
                    margin=dict(l=18, r=18, t=55, b=45),
                    legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    bargap=0.08,
                )
                fig_tenure.update_xaxes(showgrid=False, zeroline=False)
                fig_tenure.update_yaxes(gridcolor="#e9eef6", zeroline=False)
                st.plotly_chart(fig_tenure, width="stretch", config={"displayModeBar": False})
            else:
                st.info("Tenure data is not available in this dataset.")

        st.markdown("#### Where churn is concentrated")
        left, right = st.columns(2, gap="large")

        contract_summary = None
        if "Contract" in chart_df.columns:
            contract_summary = (
                chart_df.groupby("Contract", dropna=False)['_churn']
                .agg(churn_rate="mean", customers="size")
                .reset_index()
                .sort_values("churn_rate", ascending=True)
            )
            contract_summary["Churn rate label"] = contract_summary["churn_rate"].map(lambda x: f"{x:.1%}")
            with left:
                fig_contract = px.bar(
                    contract_summary,
                    x="churn_rate", y="Contract", orientation="h",
                    text="Churn rate label",
                    color="churn_rate", color_continuous_scale=["#bfdbfe", "#2563eb"],
                    labels={"churn_rate": "Churn rate", "Contract": "Contract type"},
                )
                fig_contract.update_traces(textposition="outside", cliponaxis=False, hovertemplate="%{y}<br>Churn rate: %{x:.1%}<extra></extra>")
                fig_contract.update_layout(
                    title="Churn rate by contract", height=330,
                    margin=dict(l=15, r=55, t=55, b=35),
                    coloraxis_showscale=False,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                )
                fig_contract.update_xaxes(tickformat=".0%", gridcolor="#e9eef6")
                fig_contract.update_yaxes(showgrid=False)
                st.plotly_chart(fig_contract, width="stretch", config={"displayModeBar": False})

        if "MonthlyCharges" in chart_df.columns:
            with right:
                fig_charges = px.box(
                    chart_df,
                    x="Churn status", y="MonthlyCharges", color="Churn status",
                    color_discrete_map={"Stayed": "#2563eb", "Churned": "#7dd3fc"},
                    points=False,
                    labels={"MonthlyCharges": "Monthly charges", "Churn status": "Customer status"},
                )
                fig_charges.update_layout(
                    title="Monthly charges by customer status", height=330,
                    margin=dict(l=18, r=18, t=55, b=40),
                    showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                )
                fig_charges.update_xaxes(showgrid=False)
                fig_charges.update_yaxes(gridcolor="#e9eef6")
                st.plotly_chart(fig_charges, width="stretch", config={"displayModeBar": False})

        # Automatically surface the segment with the highest observed churn.
        if contract_summary is not None and not contract_summary.empty:
            top_segment = contract_summary.iloc[-1]
            st.markdown("#### Business takeaway")
            st.markdown(
                f"<div class='hero-tag' style='margin-top:0;background:#eaf1ff;color:#244a9b;border-color:#d8e4ff;'>"
                f"Highest observed churn: <b>{top_segment['Contract']}</b> contracts · "
                f"<b>{top_segment['churn_rate']:.1%}</b> churn across {int(top_segment['customers']):,} customers"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.caption("This is a descriptive pattern in the historical dataset, not proof that contract type causes churn.")


# ---------------------------------------------------------------------
# Explainability: global SHAP results
# ---------------------------------------------------------------------
with tab_explain:
    st.subheader("Why does the model predict churn?")
    st.write(
        "A simple view of the customer factors the model relies on most. "
        "Use this section to explain the prediction to a non-technical audience."
    )

    result_dirs = [EXPLANATION_DIR, BASE_DIR]
    csv_candidates = [
        "global_shap_importance.csv",
        "shap_feature_importance.csv",
        "global_feature_importance.csv",
    ]
    summary_candidates = ["shap_summary.png", "global_shap_summary.png"]
    bar_candidates = [
        "global_shap_importance.png",
        "shap_feature_importance.png",
        "global_feature_importance.png",
    ]

    def find_result_file(names):
        for folder in result_dirs:
            for filename in names:
                candidate = folder / filename
                if candidate.is_file():
                    return candidate
        return None

    def friendly_feature_name(value):
        name = str(value)
        name = name.replace("numeric__", "").replace("categorical__", "")
        name = name.replace("_", " ")
        replacements = {
            "tenure": "Customer tenure",
            "TotalCharges": "Total charges",
            "MonthlyCharges": "Monthly charges",
            "Contract Month-to-month": "Month-to-month contract",
            "Contract One year": "One-year contract",
            "Contract Two year": "Two-year contract",
            "InternetService Fiber optic": "Fiber-optic internet",
            "InternetService DSL": "DSL internet",
            "PaymentMethod Electronic check": "Electronic-check payment",
            "PaperlessBilling No": "Paperless billing preference",
            "TechSupport No": "No technical support",
            "OnlineSecurity No": "No online security",
            "Dependents Yes": "Has dependents",
            "MultipleLines No": "Single phone line",
            "StreamingMovies Yes": "Streaming movies enabled",
            "StreamingMovies No": "Streaming movies disabled",
        }
        return replacements.get(name, name.title())

    importance_path = find_result_file(csv_candidates)
    summary_plot_path = find_result_file(summary_candidates)
    bar_plot_path = find_result_file(bar_candidates)
    display_importance = pd.DataFrame()

    if importance_path is not None:
        try:
            raw_importance = pd.read_csv(importance_path)
            lower_names = {
                str(column).strip().lower(): column
                for column in raw_importance.columns
            }
            feature_column = next(
                (
                    lower_names[name]
                    for name in ["feature", "feature_name", "features"]
                    if name in lower_names
                ),
                None,
            )
            value_column = next(
                (
                    lower_names[name]
                    for name in [
                        "mean_abs_shap",
                        "mean_absolute_shap",
                        "mean_abs_shap_value",
                        "mean_abs",
                        "importance",
                        "shap_importance",
                    ]
                    if name in lower_names
                ),
                None,
            )

            if feature_column is not None and value_column is not None:
                display_importance = raw_importance[
                    [feature_column, value_column]
                ].copy()
                display_importance.columns = ["Feature", "Importance"]
                display_importance["Importance"] = pd.to_numeric(
                    display_importance["Importance"], errors="coerce"
                )
                display_importance = (
                    display_importance.dropna()
                    .sort_values("Importance", ascending=False)
                    .head(5)
                )
                display_importance["Feature"] = display_importance[
                    "Feature"
                ].map(friendly_feature_name)

        except Exception as exc:
            st.warning(f"Could not read the SHAP importance file: {exc}")

    if not display_importance.empty:
        top_feature = display_importance.iloc[0]["Feature"]
        st.markdown(
            f"<div style='background:#18352a;color:#f4f7ee;padding:18px 22px;"
            f"border-radius:8px;margin:12px 0 18px;'>"
            f"<div style='font-size:.7rem;letter-spacing:.13em;font-weight:800;"
            f"color:#c8f169;'>THE BIGGEST MODEL SIGNAL</div>"
            f"<div style='font-size:1.45rem;font-weight:800;margin-top:5px;'>"
            f"{top_feature}</div>"
            f"<div style='font-size:.88rem;color:#d5dfd5;margin-top:5px;'>"
            f"This feature has the highest overall influence in the saved SHAP analysis."
            f"</div></div>",
            unsafe_allow_html=True,
        )

        chart_data = display_importance.sort_values(
            "Importance", ascending=True
        )
        fig = px.bar(
            chart_data,
            x="Importance",
            y="Feature",
            orientation="h",
            text="Importance",
            title="Top 5 factors influencing model predictions",
        )
        fig.update_traces(
            marker_color="#9BCB55",
            texttemplate="%{x:.3f}",
            textposition="outside",
            cliponaxis=False,
        )
        fig.update_layout(
            height=360,
            margin=dict(l=12, r=48, t=58, b=18),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            xaxis_title="Relative SHAP importance",
            yaxis_title="",
            font=dict(family="DM Sans, sans-serif", color="#26372d"),
        )
        st.plotly_chart(fig, width="stretch")

        st.caption(
            "How to explain this: the longer the bar, the more influence that "
            "feature had on the model's predictions overall. This does not mean "
            "the feature causes churn or always increases an individual customer's risk."
        )

        with st.expander("Technical details · SHAP results and full feature list"):
            st.dataframe(
                display_importance,
                hide_index=True,
                width="stretch",
            )
            if summary_plot_path is not None:
                st.markdown("**SHAP summary plot**")
                st.image(str(summary_plot_path), width="stretch")
            if bar_plot_path is not None:
                st.markdown("**Saved global importance plot**")
                st.image(str(bar_plot_path), width="stretch")
            with st.expander("What SHAP means"):
                st.markdown(
                    "- SHAP estimates how model features contribute to predictions.\n"
                    "- Mean absolute SHAP measures average influence strength.\n"
                    "- Global importance does not show whether a feature raises or lowers "
                    "risk for every individual customer.\n"
                    "- These are explanations of model behaviour, not proof of causation."
                )
    else:
        st.info(
            "The saved SHAP importance CSV was not found or did not contain "
            "recognizable feature and importance columns. Expected "
            "global_shap_importance.csv or shap_feature_importance.csv in "
            "the project folder or explanation_results folder."
        )
        if summary_plot_path is not None:
            with st.expander("View saved SHAP summary"):
                st.image(str(summary_plot_path), width="stretch")


# ---------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------
with tab_batch:
    st.subheader("Score multiple customers")

    st.write(
        "Upload a CSV containing the original Telco customer columns. "
        "The fitted V2 pipeline handles categorical encoding and scaling."
    )

    uploaded_file = st.file_uploader(
        "Upload customer CSV",
        type=["csv"],
        key="v2_batch_upload",
    )

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)

            # Remove target / identifier columns, clean blanks and numeric
            # strings, then score using the exact feature order expected by V2.
            input_df = batch_df.drop(
                columns=["Churn", "customerID", "CustomerID"],
                errors="ignore",
            )

            model_input = clean_batch_data(input_df)
            probabilities = predict_probabilities(model_input)
            log_prediction(
                float(np.mean(probabilities)),
                "batch",
                len(probabilities),
                int(np.sum(probabilities >= threshold)),
            )

            results = batch_df.copy()

            results["Churn Probability"] = probabilities

            results["Predicted Churn"] = (
                probabilities >= threshold
            ).astype(int)

            results["Prediction"] = np.where(
                results["Predicted Churn"].eq(1),
                "Likely to churn",
                "Likely to stay",
            )

            results["Risk Level"] = [
                risk_label(p)
                for p in probabilities
            ]

            a, b, c = st.columns(3)

            a.metric(
                "Customers scored",
                f"{len(results):,}",
            )

            b.metric(
                "Predicted churn",
                f"{int(results['Predicted Churn'].sum()):,}",
            )

            c.metric(
                "Average churn probability",
                f"{float(probabilities.mean()):.1%}",
            )

            st.dataframe(
                results.head(50),
                hide_index=True,
                width="stretch",
            )

            st.download_button(
                "Download scored customers CSV",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="customer_churn_predictions.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Could not score this CSV: {exc}")


# ---------------------------------------------------------------------
# V9.1: prediction monitoring and history
# ---------------------------------------------------------------------
with tab_monitor:
    st.subheader("Is ChurnIQ being used, and what is it seeing?")
    st.write(
        "Monitoring gives you a simple activity report: how often customers were "
        "scored, how many records were processed, and the average churn risk "
        "returned by the model."
    )

    if PREDICTION_HISTORY_PATH.exists():
        try:
            history = pd.read_csv(PREDICTION_HISTORY_PATH)

            if not history.empty:
                history["timestamp_utc"] = pd.to_datetime(
                    history["timestamp_utc"], errors="coerce", utc=True
                )
                history["customers_scored"] = pd.to_numeric(
                    history["customers_scored"], errors="coerce"
                ).fillna(0)
                history["average_churn_probability"] = pd.to_numeric(
                    history["average_churn_probability"], errors="coerce"
                )

                total_events = int(len(history))
                total_customers = int(history["customers_scored"].sum())
                valid_risk = history.dropna(
                    subset=["average_churn_probability"]
                )
                if not valid_risk.empty:
                    average_risk = float(
                        np.average(
                            valid_risk["average_churn_probability"],
                            weights=valid_risk["customers_scored"].clip(lower=1),
                        )
                    )
                else:
                    average_risk = 0.0

                st.markdown(
                    "<div style='background:#18352a;color:#f4f7ee;padding:18px 22px;"
                    "border-radius:8px;margin:10px 0 18px;'>"
                    "<div style='font-size:.7rem;letter-spacing:.13em;font-weight:800;"
                    "color:#c8f169;'>MONITORING AT A GLANCE</div>"
                    "<div style='font-size:1rem;font-weight:700;margin-top:6px;'>"
                    "Your model's recent scoring activity</div>"
                    "<div style='font-size:.86rem;color:#d5dfd5;margin-top:4px;'>"
                    "These numbers describe predictions made by the app, not confirmed "
                    "customer cancellations.</div></div>",
                    unsafe_allow_html=True,
                )

                m1, m2, m3 = st.columns(3)
                m1.metric("Scoring sessions", f"{total_events:,}")
                m2.metric("Customers checked", f"{total_customers:,}")
                m3.metric("Average predicted risk", f"{average_risk:.1%}")

                st.markdown("### What does this tell us?")
                if average_risk >= threshold:
                    st.info(
                        f"The average predicted risk across logged scoring activity is "
                        f"{average_risk:.1%}, which is at or above the model's "
                        f"{threshold:.0%} decision threshold. Review the customers "
                        "flagged as high risk; this is a model signal, not proof they will leave."
                    )
                else:
                    st.success(
                        f"The average predicted risk across logged scoring activity is "
                        f"{average_risk:.1%}, below the model's {threshold:.0%} "
                        "decision threshold. Individual customers can still be high risk."
                    )

                st.markdown("### Is predicted risk changing?")
                chart_data = history.dropna(
                    subset=["timestamp_utc", "average_churn_probability"]
                ).copy()
                if not chart_data.empty:
                    chart_data["Scored at"] = chart_data["timestamp_utc"].dt.strftime(
                        "%d %b %Y, %H:%M UTC"
                    )
                    chart = px.line(
                        chart_data,
                        x="Scored at",
                        y="average_churn_probability",
                        color="source" if "source" in chart_data.columns else None,
                        markers=True,
                        labels={
                            "average_churn_probability": "Average predicted churn risk",
                            "Scored at": "Scoring time",
                            "source": "Scoring type",
                        },
                    )
                    chart.update_traces(line_width=3, marker_size=8)
                    chart.update_layout(
                        height=330,
                        margin=dict(l=12, r=20, t=30, b=12),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        yaxis_tickformat=".0%",
                        yaxis_title="Average predicted risk",
                        xaxis_title="",
                        legend_title_text="Scoring type",
                        font=dict(family="DM Sans, sans-serif", color="#26372d"),
                    )
                    st.plotly_chart(chart, width="stretch")
                else:
                    st.caption("There are no valid risk values to plot yet.")

                with st.expander("View scoring history and download activity log"):
                    st.dataframe(
                        history.sort_values("timestamp_utc", ascending=False),
                        hide_index=True,
                        width="stretch",
                    )
                    st.download_button(
                        "Download monitoring history",
                        history.to_csv(index=False).encode("utf-8"),
                        "prediction_history.csv",
                        "text/csv",
                    )

                with st.expander("What monitoring does — and does not — tell us"):
                    st.markdown(
                        "- **It tells us:** how many scoring sessions ran, how many "
                        "customers were checked, and what risk probabilities the model returned.\n"
                        "- **It does not tell us:** how many customers actually cancelled "
                        "or whether the model was correct.\n"
                        "- **To measure model accuracy over time:** we would need to "
                        "compare these predictions with verified customer outcomes later."
                    )
            else:
                st.info(
                    "No scoring activity has been recorded yet. Run an individual "
                    "prediction or upload a customer CSV in Batch Studio. This page "
                    "will then show the activity."
                )
        except Exception as exc:
            st.warning(f"Could not read the monitoring history: {exc}")
    else:
        st.info(
            "No scoring activity has been recorded yet. Run an individual prediction "
            "or upload a customer CSV in Batch Studio. This page will then show the activity."
        )


with tab_drift:
    st.subheader("Data health check")
    st.write(
        "Check whether the customer information you receive today looks different "
        "from the data used to build ChurnIQ. Large changes are a reason to investigate, "
        "not automatic proof that the model is failing."
    )

    with st.expander("What is data drift?", expanded=False):
        st.markdown(
            "Data drift happens when the pattern of incoming customer information "
            "changes over time. For example, a much larger share of new customers "
            "might be on month-to-month contracts than in the original dataset. "
            "This check compares feature distributions; it does not measure prediction accuracy."
        )

    drift_file = st.file_uploader(
        "Upload recent customer data (CSV)",
        type=["csv"],
        key="v92_drift_upload",
        help="Use the original customer input columns expected by the model.",
    )
    st.caption(
        "Your uploaded file is checked in memory. It is not added to prediction history."
    )

    if drift_file is None:
        st.info(
            "Upload a recent customer CSV to compare it with the project's reference dataset."
        )
    else:
        try:
            current_df = pd.read_csv(drift_file)
            missing_drift = [c for c in feature_columns if c not in current_df.columns]

            if missing_drift:
                st.error(
                    "This file is missing required model columns: "
                    + ", ".join(missing_drift)
                )
            elif current_df.empty:
                st.warning("The uploaded file contains no customer rows.")
            else:
                drift_rows = []
                for feature in feature_columns:
                    psi = population_stability_index(
                        X_dataset[feature], current_df[feature]
                    )
                    if np.isfinite(psi):
                        drift_rows.append({"Feature": feature, "PSI": psi})

                if not drift_rows:
                    st.warning(
                        "There was not enough valid data to compare the uploaded file. "
                        "Check that the feature columns contain usable values."
                    )
                else:
                    drift_results = pd.DataFrame(drift_rows).sort_values(
                        "PSI", ascending=False
                    )
                    drift_results["Status"] = np.select(
                        [
                            drift_results["PSI"] < 0.10,
                            drift_results["PSI"] < 0.25,
                        ],
                        ["Little change", "Some change"],
                        default="Large change",
                    )

                    stable = int((drift_results["PSI"] < 0.10).sum())
                    moderate = int(
                        ((drift_results["PSI"] >= 0.10) &
                         (drift_results["PSI"] < 0.25)).sum()
                    )
                    significant = int((drift_results["PSI"] >= 0.25).sum())
                    total_features = len(drift_results)

                    st.markdown("### Data health summary")
                    d1, d2, d3 = st.columns(3)
                    d1.metric("Little change", f"{stable} / {total_features}")
                    d2.metric("Some change", f"{moderate} / {total_features}")
                    d3.metric("Large change", f"{significant} / {total_features}")

                    if significant:
                        st.error(
                            f"{significant} feature(s) show a large distribution change. "
                            "Check the affected columns and investigate before relying "
                            "on predictions without further evaluation."
                        )
                    elif moderate:
                        st.warning(
                            f"{moderate} feature(s) show some change. Keep an eye on "
                            "these inputs and compare future uploads."
                        )
                    else:
                        st.success(
                            "No large feature distribution changes were detected in "
                            "this upload. Continue checking new data regularly."
                        )

                    st.markdown("### Features that changed the most")
                    top_changes = drift_results.head(5).copy()
                    top_changes["Feature"] = top_changes["Feature"].str.replace(
                        "_", " ", regex=False
                    ).str.replace("([a-z])([A-Z])", r"\1 \2", regex=True).str.title()
                    top_changes["PSI"] = top_changes["PSI"].round(3)
                    st.dataframe(
                        top_changes[["Feature", "Status", "PSI"]],
                        hide_index=True,
                        width="stretch",
                    )

                    with st.expander("View feature comparison chart"):
                        chart_data = drift_results.copy()
                        chart_data["Feature"] = chart_data["Feature"].str.replace(
                            "_", " ", regex=False
                        ).str.title()
                        chart = px.bar(
                            chart_data.sort_values("PSI"),
                            x="PSI",
                            y="Feature",
                            color="Status",
                            orientation="h",
                            color_discrete_map={
                                "Little change": "#16a36a",
                                "Some change": "#e4a12c",
                                "Large change": "#dc4655",
                            },
                        )
                        chart.update_layout(
                            height=max(320, 25 * len(chart_data)),
                            margin=dict(l=10, r=15, t=15, b=10),
                            xaxis_title="Distribution change (PSI)",
                            yaxis_title="",
                            legend_title_text="Change level",
                        )
                        st.plotly_chart(chart, width="stretch")

                    with st.expander("Technical details and full drift report"):
                        st.caption(
                            "PSI (Population Stability Index) compares reference and "
                            "uploaded feature distributions. Common screening guides: "
                            "below 0.10 = little change; 0.10–0.25 = some change; "
                            "0.25 or above = large change. These are guidelines, not guarantees."
                        )
                        st.dataframe(
                            drift_results.round({"PSI": 4}),
                            hide_index=True,
                            width="stretch",
                        )
                        st.download_button(
                            "Download full data health report",
                            drift_results.to_csv(index=False).encode("utf-8"),
                            "data_drift_report.csv",
                            "text/csv",
                        )

        except Exception as exc:
            st.error(f"Could not complete the data health check: {exc}")


# ---------------------------------------------------------------------
# Model details and evaluation metrics
# ---------------------------------------------------------------------
with tab_about:
    st.subheader("Saved model details")

    st.write(
        f"**Model:** {bundle.get('model_name', 'V2 fitted pipeline')}"
    )

    st.write(f"**Decision threshold:** {threshold:.4f}")
    st.write(f"**Input features:** {len(feature_columns)}")

    st.write(
        "The V2 model bundle contains a fitted preprocessing/model pipeline. "
        "Pass the original customer columns; do not manually one-hot encode them."
    )

    metrics = bundle.get("test_metrics")

    if isinstance(metrics, dict) and metrics:
        st.markdown("#### Held-out test performance")

        metric_columns = st.columns(min(4, len(metrics)))

        for index, (name, value) in enumerate(metrics.items()):
            if isinstance(
                value,
                (int, float, np.integer, np.floating),
            ):
                metric_columns[index % len(metric_columns)].metric(
                    str(name).replace("_", " ").title(),
                    f"{float(value):.3f}",
                )

        with st.expander("Full test metrics"):
            st.json(metrics)

    validation = bundle.get("validation_results")

    if validation is not None:
        with st.expander("Validation results"):
            st.write(validation)

    st.warning(
        "Model predictions are estimates. Validate performance on relevant "
        "new data and review recommendations before making customer decisions."
    )


st.markdown(
    "<div class='footer'>CHURNIQ · Customer Churn Intelligence &nbsp; | &nbsp; Built with Python, scikit-learn and Streamlit</div>",
    unsafe_allow_html=True,
)
