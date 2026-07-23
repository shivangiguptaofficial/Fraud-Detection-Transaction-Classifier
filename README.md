# Fraud Detection Transaction Classifier

A machine learning project that detects fraudulent financial transactions using a Random Forest classifier, complete with full exploratory data analysis (EDA) that identifies and corrects a critical **target leakage** issue in the source dataset.

---

## Overview

This project builds a binary classifier to flag fraudulent transactions from a synthetic financial transactions dataset. Beyond training a model, the primary focus of this project is **data integrity**: the EDA notebook documents a target leakage issue discovered in the raw data (`Risk_Score`) and demonstrates how excluding it changes the model from a suspiciously "perfect" result to a realistic, defensible one.

* **Result:** **ROC-AUC of 0.81** and **PR-AUC of 0.81** on held-out test data, using only features that would realistically be available at prediction time.

---

## Dataset

* **Source:** [Fraud Detection Transactions Dataset](https://www.kaggle.com/datasets/samayashar/fraud-detection-transactions-dataset) (Kaggle, CC0: Public Domain) — 50,000 synthetic transactions, 21 features, including transaction amount, device/location metadata, card details, and behavioral signals.
* **Note:** This dataset is synthetically generated and does not reflect real-world transaction data or real users. 
* The raw CSV is included in `data/` for reproducibility.

---

## Key Finding: Target Leakage in `Risk_Score`

During EDA, the `Risk_Score` column showed an unusually strong correlation with the target (~0.39, versus <0.01 for most other features) and a distribution that clusters near `1.0` for fraud cases and `0.4` for non-fraud cases with almost no overlap. 

* **The Problem:** Given the dataset documents this field as a "fraud risk score computed for the transaction," it was clearly derived from the label during data generation rather than being a signal a system would have *before* knowing the outcome.
* **The Fix:** Training with `Risk_Score` included produces a suspicious ROC-AUC of `1.00`. This project explicitly excludes it, producing the more realistic **0.81 ROC-AUC** reported below. Full analysis is available in `notebooks/01_eda.ipynb`.

---

## Approach

* **Model:** `RandomForestClassifier` (scikit-learn), chosen for strong baseline performance on tabular data, resistance to overfitting via `min_samples_leaf`, and interpretable feature importances.
* **Class Imbalance:** ~32% fraud / 68% non-fraud — handled with `class_weight='balanced'` rather than aggressive resampling, since the imbalance is moderate.
* **Feature Engineering:**
  * Extracted hour-of-day and day-of-week from the transaction timestamp.
  * Label-encoded low-cardinality categorical fields (transaction type, device, location, merchant category, card type, authentication method).
  * Dropped identifier columns (`Transaction_ID`, `User_ID`) and the raw timestamp.
  * Dropped `Risk_Score` due to confirmed target leakage.

---

## Results

| Metric | Score |
| :--- | :--- |
| **ROC-AUC** | 0.8105 |
| **PR-AUC (Average Precision)** | 0.8084 |
| **Accuracy** | 0.8770 |
| **Precision (Fraud class)** | 1.0000 |
| **Recall (Fraud class)** | 0.6200 |
| **F1-score (Fraud class)** | 0.7600 |

* **Key Takeaway:** The model is highly precise when it flags a transaction as fraud (no false positives in this test set) but catches ~62% of actual fraud cases. This trade-off is realistic and standard in fraud systems, where false positives hurt customer experience and false negatives carry financial risk.

---
## Project Structure

- **fraud-detection-transaction-classifier/**
  - **data/**: Contains the raw dataset (synthetic_fraud_dataset.csv)
  - **notebooks/**: Jupyter notebooks for EDA and leakage investigation (01_eda.ipynb)
  - **src/**: Source code modules (preprocessing.py, train_model.py)
  - **models/**: Saved serialized model artifacts (random_forest_fraud_model.joblib)
  - **reports/**: Generated metrics, feature importances, and evaluation figures (metrics.json, confusion_matrix.png, etc.)
  - **requirements.txt**: Project dependencies
  - **LICENSE**: Open-source license
  - **README.md**: Project documentation

### Feature Importance Analysis

The Random Forest model highlights which features contribute most significantly to detecting fraudulent transactions. Below is the complete feature importance ranking based on the trained model:

| Feature | Importance Score |
| :--- | :--- |
| Failed Transactions | 0.862524 |
| Transaction Amount | 0.018370 |
| Avg Transaction Amount | 0.017438 |
| Account Balance | 0.017273 |
| Transaction Count | 0.016956 |
| Card Age | 0.014994 |
| Transaction Frequency | 0.009702 |
| Daily Transaction Limit | 0.008265 |
| Transaction Type | 0.006143 |
| Location | 0.004944 |
| Merchant Category | 0.004817 |
| Authentication Method | 0.004050 |
| Card Type | 0.004039 |
| Transaction Status | 0.003965 |
| Device Type | 0.003210 |
| Is Weekend | 0.001718 |
| Previous Frauds | 0.000940 |
| IP Address | 0.000653 |

You can also view the generated visualization chart directly from the repository:

![Feature Importance Plot](reports/figures/feature_importance.png)
