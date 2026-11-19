# Customer Churn Intelligence

An end-to-end machine-learning project that estimates customer churn
risk, explains model predictions, and helps prioritize
customer-retention reviews through an interactive Streamlit dashboard.

## Project overview

Customer churn occurs when a customer stops using a company's product or
service. This project uses the Telco Customer Churn dataset to estimate
the likelihood that a customer will leave.

The application provides: - Individual customer churn-risk assessment -
Portfolio-level business dashboard - Batch scoring for uploaded customer
CSV files - Model and validation metrics - Local and global model
explainability using SHAP - A priority customer review list and
rule-based retention suggestions

> Predictions are statistical estimates of model behaviour. Retention
> suggestions are review prompts, not proven causal interventions.

## Dataset

The project uses the Telco Customer Churn dataset.

-   7,043 customer records
-   21 original columns
-   Target: `Churn`
-   Customer identifier: `customerID`

The preprocessing workflow converts numeric fields such as `tenure`,
`MonthlyCharges`, and `TotalCharges` to numeric types and handles blank
`TotalCharges` values for zero-tenure customers.

Place the source CSV at:

``` text
data/raw/Telco-Customer-Churn.csv
```

## Machine-learning workflow

1.  Load and validate the raw customer data.
2.  Clean numeric fields and check the target labels.
3.  Split data into stratified training, validation, and test sets.
4.  Preprocess numeric and categorical features with a fitted
    scikit-learn pipeline.
5.  Compare Logistic Regression, Random Forest, and Extra Trees models.
6.  Select the model using validation PR-AUC and tune a decision
    threshold on validation data.
7.  Evaluate the selected model on the held-out test set.
8.  Save the trained model bundle and evaluation metrics.
9.  Use the Streamlit app to score customers and explore model
    explanations.

## Model performance

The recorded held-out test metrics for the current V2 model are:

  Metric                  Score
  -------------------- --------
  Accuracy               75.94%
  Precision              53.33%
  Recall                 74.87%
  F1-score               62.29%
  ROC-AUC                83.21%
  PR-AUC                 62.34%
  Decision threshold       0.57

Confusion matrix:

                      Predicted No Churn   Predicted Churn
  ----------------- -------------------- -----------------
  Actual No Churn                    790               245
  Actual Churn                        94               280

These are dataset-specific evaluation results, not a guarantee of
performance on future customers or other businesses.

## Technology stack

-   Python
-   pandas and NumPy
-   scikit-learn
-   Streamlit
-   Plotly
-   SHAP
-   joblib

## Project structure

``` text
customer-churn-prediction/
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/
â”‚   â”‚   â””â”€â”€ Telco-Customer-Churn.csv
â”‚   â”œâ”€â”€ cleaned_data.csv
â”‚   â”œâ”€â”€ prepare_data.py
â”‚   â””â”€â”€ train_model_v2.py
â”œâ”€â”€ models/
â”‚   â””â”€â”€ churn_model_v2.joblib
â”œâ”€â”€ evaluation_results/
â”‚   â””â”€â”€ model_v2_metrics.json
â”œâ”€â”€ streamlit_app.py
â”œâ”€â”€ requirements.txt
â””â”€â”€ README.md
```

Your local folder may contain additional experimentation and backup
files. Keep those if you still need them; this structure highlights the
main project components.

## Setup and run

### 1. Clone the repository

``` bash
git clone https://github.com/challashivasai/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Create and activate a virtual environment

Windows PowerShell:

``` powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare the data and train the model

Run these only if the required raw dataset is present and you want to
rebuild the artifacts:

``` powershell
python .\data\prepare_data.py
python .\data\train_model_v2.py
```

### 5. Launch the dashboard

``` powershell
python -m streamlit run streamlit_app.py
```

Streamlit will display a local URL, usually `http://localhost:8501`.

## Batch scoring input

The batch-scoring page expects a CSV containing the original Telco
customer input columns required by the trained model. Keep the column
names and value formats consistent with the training dataset. Do not
upload a file containing only pre-encoded features.

## Important limitations

-   The model is trained and evaluated on the Telco Customer Churn
    dataset; external validation is needed before real business use.
-   The decision threshold affects the precision/recall trade-off.
-   SHAP explains model output, not cause-and-effect relationships.
-   Retention suggestions are heuristic prompts and should be tested
    with controlled experiments before being treated as effective
    interventions.
-   Do not use the predictions as the sole basis for consequential
    decisions about customers.

## Future improvements

-   Add cross-validation and calibration analysis.
-   Compare models using business-oriented cost/benefit metrics.
-   Add monitoring for data drift and prediction quality.
-   Validate retention actions through A/B testing.
-   Add automated tests and a reproducible training pipeline.

## Author

Add your name, contact details, and portfolio links here.

