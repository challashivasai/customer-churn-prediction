# Customer Churn Prediction & Intelligence System

An end-to-end machine learning application that predicts customer churn, explains model predictions, and helps businesses prioritize customer retention.

## Project Overview

Customer churn occurs when customers stop using a company's products or services. Retaining existing customers is important for sustainable business growth.

This project uses machine learning to estimate customer churn probability and provides an interactive dashboard for analyzing customer behavior, identifying high-risk customers, and supporting retention decisions.

## Key Features

- Individual customer churn prediction
- Batch prediction using CSV uploads
- Customer portfolio analytics dashboard
- Global feature importance using SHAP
- Individual prediction explanations using SHAP
- Retention review recommendations
- Model performance and validation dashboard
- Downloadable customer risk reports

## Technology Stack

- **Language:** Python
- **Data analysis:** Pandas, NumPy
- **Machine learning:** Scikit-learn
- **Visualization:** Plotly
- **Web application:** Streamlit
- **Model explainability:** SHAP
- **Model storage:** Joblib

## Machine Learning

The project evaluates multiple classification algorithms:

- Logistic Regression
- Random Forest
- Extra Trees Classifier

The selected model is Logistic Regression, based on validation performance.

The preprocessing pipeline includes:
- Missing-value imputation
- Numerical feature scaling
- Categorical feature encoding
- Stratified train, validation, and test splits
- Decision-threshold tuning

## Model Performance

Test-set results:

| Metric | Score |
|---|---:|
| Accuracy | 75.94% |
| Precision | 53.33% |
| Recall | 74.87% |
| F1-score | 62.29% |
| ROC-AUC | 83.21% |
| PR-AUC | 62.34% |

**Decision threshold:** 0.57

These results are from the held-out test dataset and represent model estimates, not guaranteed business outcomes.

## Dataset

The project uses the Telco Customer Churn dataset.

- 7,043 customer records
- 21 original columns
- Customer demographics, services, contract details, and billing information
- Target variable: `Churn`

## Project Structure

```text
customer-churn-prediction/
│
├── data/
│   ├── raw/
│   ├── cleaned_data.csv
│   ├── prepare_data.py
│   └── train_model_v2.py
│
├── models/
│   └── churn_model_v2.joblib
│
├── evaluation_results/
│   └── model_v2_metrics.json
│
├── explanation_results/
│
├── streamlit_app_v8.py
├── retention_engine.py
├── global_shap_analysis.py
├── requirements.txt
└── README.md
```

## Installation and Execution

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd customer-churn-prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run streamlit_app_v8.py
```

Make sure the trained model and required data files are available in their expected folders.

## Explainability

SHAP (SHapley Additive exPlanations) is used to understand which customer characteristics influence model predictions.

The application provides:
- Global feature importance
- Individual customer feature contributions
- Prediction waterfall visualization

## Disclaimer

Predictions are statistical estimates and retention recommendations are decision-support suggestions. They should be reviewed before being used for real business decisions.

## Author

Developed as a machine learning and data analytics portfolio project.