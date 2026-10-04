"""
Flood Risk Prediction - Model Development & Evaluation
Loads the preprocessed splits saved by eda_preprocessing.py
"""

import matplotlib
matplotlib.use("Agg")  # non-GUI backend; avoids Tkinter thread conflicts in PyCharm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier

from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# ---------------------------------------------------------
# 1. LOAD PREPROCESSED DATA
# ---------------------------------------------------------
X_train = np.load("X_train_scaled.npy")
X_test = np.load("X_test_scaled.npy")

y_reg_train = pd.read_csv("y_reg_train.csv").squeeze()
y_reg_test = pd.read_csv("y_reg_test.csv").squeeze()

y_clf_train = pd.read_csv("y_clf_train.csv").squeeze()
y_clf_test = pd.read_csv("y_clf_test.csv").squeeze()

print("Data loaded:", X_train.shape, X_test.shape)

# ===========================================================
# PART A: REGRESSION  (predicting FloodProbability directly)
# ===========================================================

reg_results = {}

def evaluate_regressor(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    r2 = r2_score(y_te, preds)
    mae = mean_absolute_error(y_te, preds)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    reg_results[name] = {"R2": r2, "MAE": mae, "RMSE": rmse}
    print(f"\n{name}  ->  R2: {r2:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")
    return model, preds

# Baseline
lr_model, lr_preds = evaluate_regressor(
    "Linear Regression", LinearRegression(), X_train, y_reg_train, X_test, y_reg_test
)

# Random Forest Regressor
rf_reg_model, rf_reg_preds = evaluate_regressor(
    "Random Forest Regressor",
    RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    X_train, y_reg_train, X_test, y_reg_test
)

# XGBoost Regressor
xgb_reg_model, xgb_reg_preds = evaluate_regressor(
    "XGBoost Regressor",
    XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42),
    X_train, y_reg_train, X_test, y_reg_test
)

# Comparison table
reg_df = pd.DataFrame(reg_results).T
print("\n=== Regression Model Comparison ===")
print(reg_df)
reg_df.to_csv("regression_results.csv")

# Actual vs predicted plot (best model: XGBoost)
plt.figure(figsize=(7, 7))
plt.scatter(y_reg_test, xgb_reg_preds, alpha=0.3, s=10)
plt.plot([y_reg_test.min(), y_reg_test.max()], [y_reg_test.min(), y_reg_test.max()], 'r--')
plt.xlabel("Actual Flood Probability")
plt.ylabel("Predicted Flood Probability")
plt.title("XGBoost: Actual vs Predicted")
plt.tight_layout()
plt.savefig("outputs_actual_vs_predicted.png")
plt.close()

# Feature importance (XGBoost)
feature_names = pd.read_csv("y_reg_train.csv").columns  # placeholder, replaced below
import joblib

# Reload original columns from flood.csv header for importance labels
feature_names = pd.read_csv("data/flood.csv").drop(columns=["FloodProbability"]).columns

importances = xgb_reg_model.feature_importances_
imp_df = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x=imp_df.values, y=imp_df.index, hue=imp_df.index, palette="viridis", legend=False)
plt.title("XGBoost Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("outputs_feature_importance.png")
plt.close()

# ===========================================================
# PART B: CLASSIFICATION  (HighRisk vs not, thresholded)
# ===========================================================

clf_results = {}

def evaluate_classifier(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    probs = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, probs)
    clf_results[name] = auc
    print(f"\n--- {name} ---")
    print(classification_report(y_te, preds))
    print(f"ROC-AUC: {auc:.4f}")
    cm = confusion_matrix(y_te, preds)
    return model, preds, probs, cm

# Logistic Regression baseline
logreg_model, logreg_preds, logreg_probs, logreg_cm = evaluate_classifier(
    "Logistic Regression", LogisticRegression(max_iter=1000),
    X_train, y_clf_train, X_test, y_clf_test
)

# Random Forest Classifier
rf_clf_model, rf_clf_preds, rf_clf_probs, rf_clf_cm = evaluate_classifier(
    "Random Forest Classifier",
    RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    X_train, y_clf_train, X_test, y_clf_test
)

# Confusion matrix plot (best model assumed: Random Forest)
plt.figure(figsize=(6, 5))
sns.heatmap(rf_clf_cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Low Risk", "High Risk"], yticklabels=["Low Risk", "High Risk"])
plt.title("Random Forest Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("outputs_confusion_matrix.png")
plt.close()

# ROC curve comparison
plt.figure(figsize=(7, 6))
for name, probs in [("Logistic Regression", logreg_probs), ("Random Forest Classifier", rf_clf_probs)]:
    fpr, tpr, _ = roc_curve(y_clf_test, probs)
    auc = clf_results[name]
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("outputs_roc_curve.png")
plt.close()

print("\n=== Classification AUC Comparison ===")
print(clf_results)

print("\nModeling complete. Results saved: regression_results.csv")
print("Plots saved: outputs_actual_vs_predicted.png, outputs_feature_importance.png,")
print("             outputs_confusion_matrix.png, outputs_roc_curve.png")