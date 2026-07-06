# """
# analysis_and_model.py
# ----------------------
# End-to-end pipeline on the REAL PaySim fraud dataset.

# Steps:
# 1. Load data/financial_data.csv into SQLite.
# 2. Run SQL to explore fraud patterns (fraud only happens in TRANSFER/
#    CASH_OUT, balance-mismatch red flags, mule-account patterns).
# 3. Feature engineer using the well-known PaySim tricks:
#      - errorBalanceOrig = oldbalanceOrg - amount - newbalanceOrig
#      - errorBalanceDest = oldbalanceDest + amount - newbalanceDest
#      - Only TRANSFER/CASH_OUT rows are used for modeling since fraud
#        never occurs in the other 3 types (confirmed by SQL step 2).
# 4. Train Logistic Regression + Random Forest, evaluate with
#    precision/recall/ROC-AUC (fraud is extremely rare in real data).
# 5. Export scored transactions for Power BI / Tableau dashboard.

# Run:
#     python analysis_and_model.py
# """

# import sqlite3
# import pandas as pd
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import (classification_report, roc_auc_score,
#                               confusion_matrix, RocCurveDisplay)

# # ---------------------------------------------------------------------
# # 1. Load into SQLite
# # ---------------------------------------------------------------------
# df_raw = pd.read_csv("data/financial_data.csv")
# print(f"Loaded {len(df_raw)} transactions")

# conn = sqlite3.connect(":memory:")
# df_raw.to_sql("transactions", conn, index=False, if_exists="replace")

# # ---------------------------------------------------------------------
# # 2. SQL exploration: fraud rate by transaction type
# # ---------------------------------------------------------------------
# fraud_by_type = pd.read_sql_query("""
#     SELECT type,
#            COUNT(*) AS total_txns,
#            SUM(isFraud) AS fraud_txns,
#            ROUND(100.0 * SUM(isFraud) / COUNT(*), 4) AS fraud_rate_pct
#     FROM transactions
#     GROUP BY type
#     ORDER BY fraud_rate_pct DESC;
# """, conn)
# print("\nFraud rate by transaction type:")
# print(fraud_by_type)

# # ---------------------------------------------------------------------
# # 3. Feature engineering (SQL derives the raw error terms, pandas keeps
# #    it simple for the rest)
# # ---------------------------------------------------------------------
# feat_query = """
# SELECT
#     type,
#     amount,
#     oldbalanceOrg,
#     newbalanceOrig,
#     oldbalanceDest,
#     newbalanceDest,
#     ROUND(oldbalanceOrg - amount - newbalanceOrig, 2) AS errorBalanceOrig,
#     ROUND(oldbalanceDest + amount - newbalanceDest, 2) AS errorBalanceDest,
#     isFraud
# FROM transactions
# WHERE type IN ('TRANSFER', 'CASH_OUT');
# """
# df = pd.read_sql_query(feat_query, conn)
# print(f"\nModeling on {len(df)} TRANSFER/CASH_OUT rows "
#       f"(fraud never occurs in PAYMENT/CASH_IN/DEBIT in this dataset)")
# print(df["isFraud"].value_counts())

# # one-hot encode type (only 2 categories left: TRANSFER, CASH_OUT)
# df["type_TRANSFER"] = (df["type"] == "TRANSFER").astype(int)

# # ---------------------------------------------------------------------
# # 4. EDA chart: amount and error-balance distributions, fraud vs normal
# # ---------------------------------------------------------------------
# fig, axes = plt.subplots(1, 2, figsize=(11, 5))
# df.boxplot(column="amount", by="isFraud", ax=axes[0])
# axes[0].set_title("Transaction Amount")
# axes[0].set_xlabel("isFraud (0=normal, 1=fraud)")
# axes[0].set_yscale("log")

# df.boxplot(column="errorBalanceOrig", by="isFraud", ax=axes[1])
# axes[1].set_title("Sender Balance Error")
# axes[1].set_xlabel("isFraud (0=normal, 1=fraud)")

# plt.suptitle("")
# plt.tight_layout()
# plt.savefig("charts/eda_comparison.png", dpi=120)
# plt.close()
# print("\nSaved chart -> charts/eda_comparison.png")

# # ---------------------------------------------------------------------
# # 5. Train / evaluate models
# # ---------------------------------------------------------------------
# feature_cols = ["amount", "oldbalanceOrg", "newbalanceOrig",
#                  "oldbalanceDest", "newbalanceDest",
#                  "errorBalanceOrig", "errorBalanceDest", "type_TRANSFER"]
# X = df[feature_cols]
# y = df["isFraud"]

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.25, random_state=42, stratify=y
# )

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# log_reg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
# log_reg.fit(X_train_scaled, y_train)
# y_pred_lr = log_reg.predict(X_test_scaled)
# y_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

# rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
#                              random_state=42, n_jobs=-1)
# rf.fit(X_train, y_train)
# y_pred_rf = rf.predict(X_test)
# y_proba_rf = rf.predict_proba(X_test)[:, 1]

# print("\n" + "=" * 60)
# print("LOGISTIC REGRESSION")
# print("=" * 60)
# print(classification_report(y_test, y_pred_lr, target_names=["Normal", "Fraud"]))
# print("ROC-AUC:", round(roc_auc_score(y_test, y_proba_lr), 4))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

# print("\n" + "=" * 60)
# print("RANDOM FOREST")
# print("=" * 60)
# print(classification_report(y_test, y_pred_rf, target_names=["Normal", "Fraud"]))
# print("ROC-AUC:", round(roc_auc_score(y_test, y_proba_rf), 4))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
# print("\nFeature importance (Random Forest):")
# print(importances)

# fig, ax = plt.subplots(figsize=(6, 6))
# RocCurveDisplay.from_predictions(y_test, y_proba_lr, name="Logistic Regression", ax=ax)
# RocCurveDisplay.from_predictions(y_test, y_proba_rf, name="Random Forest", ax=ax)
# ax.set_title("ROC Curve - PaySim Fraud Detection")
# plt.tight_layout()
# plt.savefig("charts/roc_curve.png", dpi=120)
# plt.close()
# print("\nSaved chart -> charts/roc_curve.png")

# # ---------------------------------------------------------------------
# # 6. Score all TRANSFER/CASH_OUT rows and export for the dashboard
# # ---------------------------------------------------------------------
# df["fraud_probability"] = rf.predict_proba(X)[:, 1]
# df["risk_flag"] = np.where(df["fraud_probability"] >= 0.5, "High Risk",
#                      np.where(df["fraud_probability"] >= 0.2, "Medium Risk", "Low Risk"))

# df_out = df[["type", "amount", "errorBalanceOrig", "errorBalanceDest",
#              "isFraud", "fraud_probability", "risk_flag"]].sort_values(
#              "fraud_probability", ascending=False)

# df_out.to_csv("outputs/scored_transactions.csv", index=False)
# print("\nSaved scored dataset -> outputs/scored_transactions.csv")
# print(df_out.head(10))

# conn.close()










"""
analysis_and_model.py
----------------------
End-to-end pipeline on the REAL PaySim fraud dataset.

Steps:
1. Load data/financial_data.csv into SQLite.
2. Run SQL to explore fraud patterns (fraud only happens in TRANSFER/
   CASH_OUT, balance-mismatch red flags, mule-account patterns).
3. Feature engineer using the well-known PaySim tricks:
     - errorBalanceOrig = oldbalanceOrg - amount - newbalanceOrig
     - errorBalanceDest = oldbalanceDest + amount - newbalanceDest
     - Only TRANSFER/CASH_OUT rows are used for modeling since fraud
       never occurs in the other 3 types (confirmed by SQL step 2).
4. Handle severe class imbalance (fraud is <0.5% of data) using SMOTE
   to oversample the minority (fraud) class in the TRAINING set only.
5. Train Logistic Regression + Random Forest, evaluate with
   precision/recall/ROC-AUC (fraud is extremely rare in real data).
6. Export scored transactions for Power BI / Tableau dashboard.

Run:
    python analysis_and_model.py
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix, RocCurveDisplay, accuracy_score)
from imblearn.over_sampling import SMOTE

# ---------------------------------------------------------------------
# 1. Load into SQLite
# ---------------------------------------------------------------------
df_raw = pd.read_csv("data/financial_data.csv")
print(f"Loaded {len(df_raw)} transactions")

conn = sqlite3.connect(":memory:")
df_raw.to_sql("transactions", conn, index=False, if_exists="replace")

# ---------------------------------------------------------------------
# 2. SQL exploration: fraud rate by transaction type
# ---------------------------------------------------------------------
fraud_by_type = pd.read_sql_query("""
    SELECT type,
           COUNT(*) AS total_txns,
           SUM(isFraud) AS fraud_txns,
           ROUND(100.0 * SUM(isFraud) / COUNT(*), 4) AS fraud_rate_pct
    FROM transactions
    GROUP BY type
    ORDER BY fraud_rate_pct DESC;
""", conn)
print("\nFraud rate by transaction type:")
print(fraud_by_type)

# ---------------------------------------------------------------------
# 3. Feature engineering (SQL derives the raw error terms, pandas keeps
#    it simple for the rest)
# ---------------------------------------------------------------------
feat_query = """
SELECT
    type,
    amount,
    oldbalanceOrg,
    newbalanceOrig,
    oldbalanceDest,
    newbalanceDest,
    ROUND(oldbalanceOrg - amount - newbalanceOrig, 2) AS errorBalanceOrig,
    ROUND(oldbalanceDest + amount - newbalanceDest, 2) AS errorBalanceDest,
    isFraud
FROM transactions
WHERE type IN ('TRANSFER', 'CASH_OUT');
"""
df = pd.read_sql_query(feat_query, conn)
print(f"\nModeling on {len(df)} TRANSFER/CASH_OUT rows "
      f"(fraud never occurs in PAYMENT/CASH_IN/DEBIT in this dataset)")
print(df["isFraud"].value_counts())

# one-hot encode type (only 2 categories left: TRANSFER, CASH_OUT)
df["type_TRANSFER"] = (df["type"] == "TRANSFER").astype(int)

# ---------------------------------------------------------------------
# 4. EDA chart: amount and error-balance distributions, fraud vs normal
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
df.boxplot(column="amount", by="isFraud", ax=axes[0])
axes[0].set_title("Transaction Amount")
axes[0].set_xlabel("isFraud (0=normal, 1=fraud)")
axes[0].set_yscale("log")

df.boxplot(column="errorBalanceOrig", by="isFraud", ax=axes[1])
axes[1].set_title("Sender Balance Error")
axes[1].set_xlabel("isFraud (0=normal, 1=fraud)")

plt.suptitle("")
plt.tight_layout()
plt.savefig("charts/eda_comparison.png", dpi=120)
plt.close()
print("\nSaved chart -> charts/eda_comparison.png")

# ---------------------------------------------------------------------
# 5. Train / evaluate models
# ---------------------------------------------------------------------
feature_cols = ["amount", "oldbalanceOrg", "newbalanceOrig",
                 "oldbalanceDest", "newbalanceDest",
                 "errorBalanceOrig", "errorBalanceDest", "type_TRANSFER"]
X = df[feature_cols]
y = df["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\nBefore SMOTE - training class counts:\n{y_train.value_counts()}")

# ---------------------------------------------------------------------
# SMOTE (Synthetic Minority Oversampling Technique)
# ---------------------------------------------------------------------
# Applied ONLY to the training set, never to the test set - oversampling
# the test set would leak synthetic fraud patterns into evaluation and
# give a falsely inflated score. SMOTE creates new synthetic fraud rows
# by interpolating between real fraud rows' feature values (rather than
# just duplicating existing ones), until both classes are equal size.
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"After SMOTE - training class counts:\n{y_train_smote.value_counts()}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# Note: class_weight="balanced" is no longer needed once SMOTE has
# already balanced the training classes 1:1, so it's left out below.
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train_smote)
y_pred_lr = log_reg.predict(X_test_scaled)
y_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train_smote, y_train_smote)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)
print(classification_report(y_test, y_pred_lr, target_names=["Normal", "Fraud"]))
lr_accuracy = accuracy_score(y_test, y_pred_lr)
print("Accuracy:", round(lr_accuracy, 4))
print("ROC-AUC:", round(roc_auc_score(y_test, y_proba_lr), 4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

print("\n" + "=" * 60)
print("RANDOM FOREST")
print("=" * 60)
print(classification_report(y_test, y_pred_rf, target_names=["Normal", "Fraud"]))
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print("Accuracy:", round(rf_accuracy, 4))
print("ROC-AUC:", round(roc_auc_score(y_test, y_proba_rf), 4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)
print(f"{'Model':<22}{'Accuracy':<12}{'ROC-AUC'}")
print(f"{'Logistic Regression':<22}{round(lr_accuracy, 4):<12}{round(roc_auc_score(y_test, y_proba_lr), 4)}")
print(f"{'Random Forest':<22}{round(rf_accuracy, 4):<12}{round(roc_auc_score(y_test, y_proba_rf), 4)}")
print("\nNote: with rare fraud cases, accuracy alone can look high even if")
print("fraud recall is poor (a model predicting 'no fraud' every time would")
print("still score ~99% accuracy). Always read accuracy together with")
print("precision/recall/ROC-AUC above, not in isolation.")

importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature importance (Random Forest):")
print(importances)

fig, ax = plt.subplots(figsize=(6, 6))
RocCurveDisplay.from_predictions(y_test, y_proba_lr, name="Logistic Regression", ax=ax)
RocCurveDisplay.from_predictions(y_test, y_proba_rf, name="Random Forest", ax=ax)
ax.set_title("ROC Curve - PaySim Fraud Detection")
plt.tight_layout()
plt.savefig("charts/roc_curve.png", dpi=120)
plt.close()
print("\nSaved chart -> charts/roc_curve.png")

# ---------------------------------------------------------------------
# 6. Score all TRANSFER/CASH_OUT rows and export for the dashboard
# ---------------------------------------------------------------------
df["fraud_probability"] = rf.predict_proba(X)[:, 1]
df["risk_flag"] = np.where(df["fraud_probability"] >= 0.5, "High Risk",
                     np.where(df["fraud_probability"] >= 0.2, "Medium Risk", "Low Risk"))

df_out = df[["type", "amount", "errorBalanceOrig", "errorBalanceDest",
             "isFraud", "fraud_probability", "risk_flag"]].sort_values(
             "fraud_probability", ascending=False)

df_out.to_csv("outputs/scored_transactions.csv", index=False)
print("\nSaved scored dataset -> outputs/scored_transactions.csv")
print(df_out.head(10))

conn.close()