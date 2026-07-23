# Fraud Detection Transaction Classifier

A machine learning project that detects fraudulent financial transactions
using a Random Forest classifier, with a full exploratory data analysis
that identifies and corrects a target leakage issue in the source dataset.

## Overview

This project builds a binary classifier to flag fraudulent transactions
from a synthetic financial transactions dataset. Beyond training a model,
the focus of this project is **data integrity**: the EDA notebook
documents a target leakage issue discovered in the raw data (`Risk_Score`)
and shows how excluding it changes the model from a suspicious "perfect"
result to a realistic, defensible one.

**Result:** ROC-AUC of **0.81** and PR-AUC of **0.81** on held-out test data,
using only features that would realistically be available at prediction time.

## Dataset

[Fraud Detection Transactions Dataset](https://www.kaggle.com/datasets/samayashar/fraud-detection-transactions-dataset)
(Kaggle, CC0: Public Domain) — 50,000 synthetic transactions, 21 features,
including transaction amount, device/location metadata, card details, and
behavioral signals (failed transaction counts, average 7-day spend, etc.).

> This dataset is synthetically generated and does not reflect real-world
> transaction data or real users. Results here should be read as a
> demonstration of methodology, not as a claim about real fraud patterns.

The raw CSV is included in `data/` for reproducibility.

## Key finding: target leakage in `Risk_Score`

During EDA, the `Risk_Score` column showed an unusually strong correlation
with the target (0.39, versus <0.01 for most other features) and a
distribution that clusters near 1.0 for fraud cases and 0.4 for non-fraud
cases with almost no overlap. Given the dataset documents this field as a
"fraud risk score computed for the transaction," the most likely
explanation is that it was derived from the label during data generation,
rather than being a signal a system would have *before* knowing the
outcome.

Training with `Risk_Score` included produces a ROC-AUC of **1.00** —
a strong signal of leakage rather than genuine model quality. This
project excludes it, producing the more realistic 0.81 ROC-AUC reported
above. Full analysis is in [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb).

## Approach

### Model
Random Forest Classifier (`scikit-learn`), chosen for strong baseline
performance on tabular data, resistance to overfitting via
`min_samples_leaf`, and interpretable feature importances.

### Class imbalance
~32% fraud / 68% non-fraud — handled with `class_weight="balanced"`
rather than resampling, since the imbalance is moderate rather than severe.

### Feature engineering
- Extracted hour-of-day and day-of-week from the transaction timestamp
- Label-encoded low-cardinality categorical fields (transaction type,
  device, location, merchant category, card type, authentication method)
- Dropped identifier columns (`Transaction_ID`, `User_ID`) and the raw
  `Timestamp` after extracting time features
- Dropped `Risk_Score` due to confirmed target leakage (see above)

### Evaluation
ROC-AUC and PR-AUC (not accuracy alone, since accuracy is a misleading
metric under class imbalance), plus a full classification report,
confusion matrix, ROC curve, and feature importance plot.

## Results

| Metric | Score |
|---|---|
| ROC-AUC | 0.8105 |
| PR-AUC (Average Precision) | 0.8084 |
| Accuracy | 0.877 |
| Precision (Fraud class) | 1.00 |
| Recall (Fraud class) | 0.62 |
| F1-score (Fraud class) | 0.76 |

The model is highly precise when it flags a transaction as fraud (no false
positives in this test set) but only catches ~62% of actual fraud cases.
This precision/recall trade-off is a realistic and common tension in fraud
detection systems, where false positives carry customer-experience costs
and false negatives carry financial ones — the right balance point depends
on business context, not just model accuracy.



![Confusion Matrix](reports/figures/confusion_matrix.png)




![ROC Curve](reports/figures/roc_curve.png)




![Feature Importance](reports/figures/feature_importance.png)



The most predictive feature is `Failed_Transaction_Count_7d`, a plausible
behavioral fraud indicator, followed by transaction amount and account
balance.

## Project structure
fraud-detection-transaction-classifier/
├── data/
│   └── synthetic_fraud_dataset.csv
├── notebooks/
│   └── 01_eda.ipynb              # Exploratory analysis + leakage investigation
├── src/
│   ├── preprocessing.py          # Feature engineering & encoding pipeline
│   └── train_model.py            # Training, evaluation, and plot generation
├── models/
│   └── random_forest_fraud_model.joblib
├── reports/
│   ├── metrics.json
│   ├── feature_importance.csv
│   └── figures/
├── requirements.txt
├── LICENSE
└── 
git clone https://github.com/<your-username>/fraud-detection-transaction-classifier.git
cd fraud-detection-transaction-classifier
pip install -r requirements.txt

# Run the training pipeline
python src/train_model.py

# Or explore the analysis interactively
jupyter notebook notebooks/01_eda.ipynb

## Setup and usage

```bash
git clone https://github.com/<your-username>/fraud-detection-transaction-classifier.git
cd fraud-detection-transaction-classifier
pip install -r requirements.txt

# Run the training pipeline
python src/train_model.py

# Or explore the analysis interactively
jupyter notebook notebooks/01_eda.ipynb

Below that, the two source files — copy each into `src/preprocessing.py` and `src/train_model.py` respectively.

**`src/preprocessing.py`**
```python
"""
preprocessing.py
-----------------
Loads the raw transaction data and prepares it for modeling:
- Parses timestamps into usable time-based features
- Encodes categorical variables
- Drops identifier columns that would leak or add no signal
- Splits features (X) from the target (y)
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

CATEGORICAL_COLUMNS = [
    "Transaction_Type",
    "Device_Type",
    "Location",
    "Merchant_Category",
    "Card_Type",
    "Authentication_Method",
]

DROP_COLUMNS = ["Transaction_ID", "User_ID", "Timestamp", "Risk_Score"]
# NOTE: Risk_Score is dropped deliberately. EDA showed it is almost
# certainly derived from Fraud_Label during synthetic data generation
# (fraud cases cluster near 1.0, non-fraud near 0.4), so including it
# causes severe target leakage (ROC-AUC of 1.00). See notebooks/01_eda.ipynb
# for the analysis that led to this decision.

TARGET_COLUMN = "Fraud_Label"


def load_data(path: str) -> pd.DataFrame:
    """Load the raw CSV into a DataFrame."""
    return pd.read_csv(path)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour-of-day and day-of-week from the Timestamp column."""
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Transaction_Hour"] = df["Timestamp"].dt.hour
    df["Transaction_DayOfWeek"] = df["Timestamp"].dt.dayofweek
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode categorical columns. Simple and effective for tree-based models."""
    df = df.copy()
    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df


def prepare_features(df: pd.DataFrame):
    """
    Full preprocessing pipeline: adds time features, encodes categoricals,
    drops identifier columns, and splits into X (features) and y (target).
    """
    df = add_time_features(df)
    df = encode_categoricals(df)
    df = df.drop(columns=DROP_COLUMNS)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


if __name__ == "__main__":
    df = load_data("data/synthetic_fraud_dataset.csv")
    X, y = prepare_features(df)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts(normalize=True)}")
