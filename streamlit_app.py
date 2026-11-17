
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
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
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "churn_model_v2.joblib"
DATA_PATH = BASE_DIR / "data" / "cleaned_data.csv"
EXPLANATION_DIR = BASE_DIR / "explanation_results"

st.title("📊 Customer Churn Intelligence")
st.caption(
    "Predict customer churn, explore customer segments, and review "
    "model-informed retention opportunities."
)


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


def risk_label(probability: float) -> str:
    if probability >= threshold:
        return "High"
    if probability >= max(0.0, threshold - 0.10):
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Model information")
    st.metric("Decision threshold", f"{threshold:.2%}")
    st.metric("Input features", len(feature_columns))
    st.caption(f"Model artifact: {MODEL_PATH.name}")
    st.caption("Predictions use the fitted V2 preprocessing pipeline.")


# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------
tab_predict, tab_dashboard, tab_explain, tab_batch, tab_about = st.tabs(
    [
        "🔍 Individual prediction",
        "📈 Business dashboard",
        "🧠 Explainability",
        "📂 Batch scoring",
        "ℹ️ Model details",
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
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox(
                "Senior citizen",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No",
            )
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple lines", ["No", "Yes", "No phone service"]
            )

        with col2:
            internet = st.selectbox(
                "Internet service", ["DSL", "Fiber optic", "No"]
            )
            online_security = st.selectbox(
                "Online security", ["No", "Yes", "No internet service"]
            )
            online_backup = st.selectbox(
                "Online backup", ["No", "Yes", "No internet service"]
            )
            device_protection = st.selectbox(
                "Device protection", ["No", "Yes", "No internet service"]
            )
            tech_support = st.selectbox(
                "Tech support", ["No", "Yes", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["No", "Yes", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming movies", ["No", "Yes", "No internet service"]
            )

        with col3:
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly = st.number_input(
                "Monthly charges",
                min_value=0.0,
                max_value=1000.0,
                value=70.0,
                step=1.0,
            )
            total = st.number_input(
                "Total charges",
                min_value=0.0,
                max_value=100000.0,
                value=840.0,
                step=10.0,
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

            st.divider()
            st.subheader("Prediction result")

            metric1, metric2, metric3 = st.columns(3)
            metric1.metric("Churn probability", f"{probability:.1%}")
            metric2.metric("Decision threshold", f"{threshold:.1%}")
            metric3.metric(
                "Predicted outcome",
                "Likely to churn" if prediction else "Likely to stay",
            )

            if prediction:
                st.error(
                    "High churn risk: the model predicts this customer may leave."
                )
            else:
                st.success(
                    "Lower churn risk: the model predicts this customer may stay."
                )

            st.progress(float(np.clip(probability, 0.0, 1.0)))
            st.caption(
                "This is a statistical estimate, not a guarantee or proof "
                "that a customer will leave."
            )

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
                    st.subheader("🎯 Retention action plan")

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
    st.subheader("Customer portfolio overview")

    if "Churn" not in df.columns:
        st.warning(
            "The dataset has no Churn target column, "
            "so churn metrics are unavailable."
        )
    else:
        churn_binary = churn_as_binary(df["Churn"])

        customer_count = len(df)
        churn_count = int(churn_binary.sum())

        churn_rate = (
            churn_count / customer_count
            if customer_count
            else 0.0
        )

        c1, c2, c3 = st.columns(3)

        c1.metric("Total customers", f"{customer_count:,}")
        c2.metric("Customers churned", f"{churn_count:,}")
        c3.metric("Overall churn rate", f"{churn_rate:.1%}")

        chart_df = df.copy()

        chart_df["Churn status"] = np.where(
            churn_binary.eq(1),
            "Churned",
            "Stayed",
        )

        left, right = st.columns(2)

        with left:
            status_counts = (
                chart_df["Churn status"]
                .value_counts()
                .rename_axis("Customer status")
                .reset_index(name="Customers")
            )

            st.plotly_chart(
                px.pie(
                    status_counts,
                    names="Customer status",
                    values="Customers",
                    title="Customer churn distribution",
                    hole=0.45,
                ),
                width="stretch",
            )

        with right:
            if "tenure" in chart_df.columns:
                st.plotly_chart(
                    px.histogram(
                        chart_df,
                        x="tenure",
                        color="Churn status",
                        title="Customer tenure distribution",
                        labels={"tenure": "Tenure (months)"},
                        nbins=24,
                    ),
                    width="stretch",
                )

        if "Contract" in chart_df.columns:
            contract_summary = (
                chart_df.assign(_churn=churn_binary)
                .groupby("Contract", dropna=False)["_churn"]
                .mean()
                .reset_index(name="Churn rate")
            )

            st.plotly_chart(
                px.bar(
                    contract_summary,
                    x="Contract",
                    y="Churn rate",
                    title="Churn rate by contract type",
                    text_auto=".1%",
                ),
                width="stretch",
            )

        if "MonthlyCharges" in chart_df.columns:
            st.plotly_chart(
                px.box(
                    chart_df,
                    x="Churn status",
                    y="MonthlyCharges",
                    color="Churn status",
                    title="Monthly charges by customer status",
                ),
                width="stretch",
            )


# ---------------------------------------------------------------------
# Explainability: global SHAP results
# ---------------------------------------------------------------------
with tab_explain:
    st.subheader("Explainable AI")

    st.write(
        "Global feature importance describes which features influenced "
        "model predictions across the analysed dataset. It does not prove causation."
    )

    importance_path = EXPLANATION_DIR / "shap_feature_importance.csv"
    summary_plot_path = EXPLANATION_DIR / "shap_summary.png"
    bar_plot_path = EXPLANATION_DIR / "shap_feature_importance.png"

    if importance_path.is_file():
        try:
            importance = pd.read_csv(importance_path)

            required = {"feature", "mean_abs_shap"}

            if required.issubset(importance.columns):
                importance = importance.sort_values(
                    "mean_abs_shap",
                    ascending=False,
                ).head(15)

                st.dataframe(
                    importance.rename(
                        columns={
                            "feature": "Feature",
                            "mean_abs_shap": "Mean absolute SHAP value",
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )
            else:
                st.warning(
                    "The global SHAP CSV does not contain "
                    "feature and mean_abs_shap columns."
                )

        except Exception as exc:
            st.warning(f"Could not read global SHAP results: {exc}")

    else:
        st.info(
            "Global SHAP results are not present yet. This section will display "
            "them after compatible global explanations have been generated."
        )

    image_left, image_right = st.columns(2)

    with image_left:
        st.markdown("**SHAP summary plot**")

        if summary_plot_path.is_file():
            st.image(
                str(summary_plot_path),
                width="stretch",
            )
        else:
            st.info("shap_summary.png is not available.")

    with image_right:
        st.markdown("**Global feature importance plot**")

        if bar_plot_path.is_file():
            st.image(
                str(bar_plot_path),
                width="stretch",
            )
        else:
            st.info("shap_feature_importance.png is not available.")


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

            input_df = batch_df.drop(
                columns=[
                    "Churn",
                    "customerID",
                    "CustomerID",
                ],
                errors="ignore",
            )

            missing = [
                column
                for column in feature_columns
                if column not in input_df.columns
            ]

            if missing:
                st.error(
                    "The uploaded CSV is missing required columns: "
                    + ", ".join(missing)
                )

            else:
                model_input = input_df.loc[
                    :,
                    feature_columns,
                ].copy()

                probabilities = predict_probabilities(model_input)

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
