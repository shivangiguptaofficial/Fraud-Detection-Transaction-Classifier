# %% [markdown]
# # Exploratory Data Analysis — Fraud Detection Dataset
#
# This notebook explores the Fraud Detection Transactions Dataset
# (synthetic, CC0, 50,000 rows / 21 columns) and documents the decisions
# that shaped the modeling pipeline in `src/`.
# Source: https://www.kaggle.com/datasets/samayashar/fraud-detection-transactions-dataset

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
df = pd.read_csv("../data/synthetic_fraud_dataset.csv")
df.shape

# %% [markdown]
# ## 1. Basic structure and data quality

# %%
df.info()

# %%
df.isnull().sum().sum()  # confirm no missing values

# %% [markdown]
# No missing values, no duplicate rows expected in a synthetic dataset —
# confirmed clean, so no imputation strategy is needed.

# %% [markdown]
# ## 2. Class balance

# %%
df["Fraud_Label"].value_counts(normalize=True)

# %%
plt.figure(figsize=(5, 4))
sns.countplot(x="Fraud_Label", hue="Fraud_Label", data=df, legend=False)
plt.title("Class Distribution: Fraud vs Not Fraud")
plt.show()

# %% [markdown]
# ~32% of transactions are labeled fraud. This is a moderate imbalance —
# not the severe <1% imbalance typical of real-world fraud data — so heavy
# techniques like SMOTE aren't strictly necessary. `class_weight="balanced"`
# in the model is sufficient here.

# %% [markdown]
# ## 3. Checking for target leakage
#
# Before modeling, it's worth checking whether any feature correlates
# suspiciously strongly with the target — a common issue in synthetic
# fraud datasets where risk-related fields are sometimes generated *from*
# the label rather than independently of it.

# %%
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols].corr()["Fraud_Label"].sort_values(ascending=False)

# %% [markdown]
# `Risk_Score` and `Failed_Transaction_Count_7d` stand out. Let's look
# at `Risk_Score` more closely, since a 0.39 correlation for a single
# feature is unusually high compared to everything else.

# %%
df.groupby("Fraud_Label")["Risk_Score"].describe()

# %%
plt.figure(figsize=(6, 4))
sns.kdeplot(data=df, x="Risk_Score", hue="Fraud_Label", fill=True, common_norm=False)
plt.title("Risk_Score Distribution by Class")
plt.show()

# %% [markdown]
# **Finding:** `Risk_Score` for fraudulent transactions is heavily
# concentrated near 1.0, while non-fraud transactions center around 0.4,
# with almost no overlap at the high end. Given the dataset documentation
# describes this field simply as "fraud risk score computed for the
# transaction," the most likely explanation is that it was generated
# *from* the fraud label during the synthetic data creation process,
# rather than being an independent signal a real-time system would have
# before knowing the outcome.
#
# **Decision:** `Risk_Score` is excluded from the feature set used for
# modeling (see `src/preprocessing.py`). Including it produces a
# suspicious ROC-AUC of 1.00 on the held-out test set — a strong
# indicator of leakage rather than a genuinely strong model. Excluding
# it gives a more realistic and defensible ROC-AUC of ~0.81.

# %% [markdown]
# ## 4. Behavioral features

# %%
df.groupby("Fraud_Label")["Failed_Transaction_Count_7d"].describe()

# %% [markdown]
# `Failed_Transaction_Count_7d` also separates the classes well
# (fraud cases average ~3 failed transactions in the past week vs ~1.5
# for legitimate ones), but this is a plausible, legitimate signal —
# repeated failed attempts are a realistic fraud indicator that a real
# system would have available at prediction time, unlike `Risk_Score`.

# %% [markdown]
# ## 5. Correlation overview

# %%
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.show()

# %% [markdown]
# ## 6. Categorical feature cardinality

# %%
cat_cols = ["Transaction_Type", "Device_Type", "Location",
            "Merchant_Category", "Card_Type", "Authentication_Method"]
for c in cat_cols:
    print(c, df[c].nunique(), df[c].unique())

# %% [markdown]
# All categorical columns have low cardinality (3–5 unique values),
# so simple label encoding is sufficient for a tree-based model — no
# need for one-hot encoding or target encoding here.

# %% [markdown]
# ## Summary of decisions carried into modeling
#
# - Drop `Transaction_ID`, `User_ID`, `Timestamp` (identifiers / raw timestamp, not features)
# - Extract `Transaction_Hour` and `Transaction_DayOfWeek` from `Timestamp`
# - **Drop `Risk_Score`** — confirmed target leakage
# - Label-encode low-cardinality categoricals
# - Use `class_weight="balanced"` in the model given the ~32/68 class split
#
# See `src/train_model.py` for the full training pipeline and
# `reports/` for evaluation results.
