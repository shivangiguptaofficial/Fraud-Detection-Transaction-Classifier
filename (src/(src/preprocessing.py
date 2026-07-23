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
