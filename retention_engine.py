
import pandas as pd


def generate_retention_plan(churn_probability, threshold, shap_details):
    """
    Generate customer-specific retention recommendations
    using churn probability and individual SHAP values.
    """

    probability = float(churn_probability)
    threshold = float(threshold)

    # Classify customer risk relative to model threshold
    if probability >= threshold:
        risk_level = "High"
        summary = "The model predicts that this customer may churn."
    elif probability >= threshold - 0.10:
        risk_level = "Medium"
        summary = "The customer is close to the churn decision threshold. Monitor carefully."
    else:
        risk_level = "Low"
        summary = "The customer currently has a relatively low predicted churn risk."

    # Work with SHAP explanation data
    details = shap_details.copy()

    details["SHAP impact"] = pd.to_numeric(
        details["SHAP impact"], errors="coerce"
    )

    increasing = details[
        details["SHAP impact"] > 0
    ].sort_values("SHAP impact", ascending=False)

    reducing = details[
        details["SHAP impact"] < 0
    ].sort_values("SHAP impact")

    # Customer-specific recommendation rules
    recommendations = []

    feature_actions = {
        "tenure": (
            "Strengthen customer engagement through personalized loyalty benefits."
        ),
        "Contract_Two year": (
            "Review contract preferences and offer suitable renewal benefits."
        ),
        "Contract_One year": (
            "Offer a suitable long-term contract with transparent pricing."
        ),
        "PaymentMethod_Electronic check": (
            "Offer convenient automatic payment options and explain their benefits."
        ),
        "PaperlessBilling_Yes": (
            "Maintain digital billing and provide clear billing notifications."
        ),
        "MonthlyCharges": (
            "Review the customer's monthly charges and suggest a suitable plan."
        ),
        "TotalCharges": (
            "Review the customer's billing history and identify possible concerns."
        ),
        "InternetService_Fiber optic": (
            "Check service quality, speed and technical support satisfaction."
        ),
        "TechSupport_Yes": (
            "Continue providing accessible technical support."
        ),
        "OnlineSecurity_Yes": (
            "Highlight the benefits of the customer's existing security service."
        ),
        "StreamingMovies_Yes": (
            "Review streaming service satisfaction and package value."
        ),
        "MultipleLines_Yes": (
            "Check whether the current multi-line plan meets customer needs."
        ),
    }

    for _, row in increasing.iterrows():
        feature = str(row["Feature"])

        if feature in feature_actions:
            recommendations.append({
                "Feature": feature,
                "SHAP impact": round(row["SHAP impact"], 4),
                "Recommended action": feature_actions[feature]
            })

    # Generic fallback if no feature-specific rules match
    if not recommendations:
        recommendations.append({
            "Feature": "Overall customer risk",
            "SHAP impact": None,
            "Recommended action": (
                "Contact the customer, collect feedback and review service satisfaction."
            )
        })

    return {
        "risk_level": risk_level,
        "churn_probability": probability,
        "decision_threshold": threshold,
        "summary": summary,
        "increasing_factors": increasing,
        "reducing_factors": reducing,
        "recommendations": recommendations
    }
