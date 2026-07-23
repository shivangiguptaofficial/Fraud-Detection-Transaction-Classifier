"""
train_model.py
---------------
Trains a Random Forest classifier to detect fraudulent transactions,
evaluates it on a held-out test set, and saves the trained model
plus evaluation artifacts (metrics, confusion matrix, ROC curve,
feature importance plot) to the reports/ directory.
"""

import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

from preprocessing import load_data, prepare_features

RANDOM_STATE = 42
DATA_PATH = "data/synthetic_fraud_dataset.csv"
MODEL_PATH = "models/random_forest_fraud_model.joblib"
REPORTS_DIR = "reports"


def train():
    df = load_data(DATA_PATH)
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # --- Metrics ---
    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)

    metrics = {
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "classification_report": report,
    }
    with open(f"{REPORTS_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Average Precision (PR-AUC): {avg_precision:.4f}")
    print(classification_report(y_test, y_pred))

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Fraud", "Fraud"],
                yticklabels=["Not Fraud", "Fraud"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Random Forest")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/figures/confusion_matrix.png", dpi=150)
    plt.close()

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Random Forest")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/figures/roc_curve.png", dpi=150)
    plt.close()

    # --- Precision-Recall Curve ---
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR curve (AP = {avg_precision:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Random Forest")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/figures/precision_recall_curve.png", dpi=150)
    plt.close()

    # --- Feature Importance ---
    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances.values, y=importances.index, color="steelblue")
    plt.xlabel("Importance")
    plt.title("Feature Importance - Random Forest")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/figures/feature_importance.png", dpi=150)
    plt.close()

    importances.to_csv(f"{REPORTS_DIR}/feature_importance.csv")

    # --- Save model ---
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Reports and figures saved to {REPORTS_DIR}/")

    return model, metrics


if __name__ == "__main__":
    train()
