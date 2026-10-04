"""
Flood Risk Prediction - EDA & Preprocessing
Mini Project: AI/ML solution for flood risk prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
df = pd.read_csv("data/flood.csv")

print("Shape:", df.shape)
print("\nColumn dtypes:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nSummary statistics:\n", df.describe())

# ---------------------------------------------------------
# 2. EDA VISUALIZATIONS
# ---------------------------------------------------------

# Target distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["FloodProbability"], bins=40, kde=True, color="steelblue")
plt.title("Distribution of Flood Probability")
plt.xlabel("Flood Probability")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("outputs_target_distribution.png")
plt.close()

# Correlation heatmap
plt.figure(figsize=(14, 10))
corr = df.corr()
sns.heatmap(corr, cmap="coolwarm", annot=False, center=0)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs_correlation_heatmap.png")
plt.close()

# Correlation with target, sorted
target_corr = corr["FloodProbability"].drop("FloodProbability").sort_values(ascending=False)
plt.figure(figsize=(10, 8))
sns.barplot(x=target_corr.values, y=target_corr.index, palette="viridis")
plt.title("Feature Correlation with Flood Probability")
plt.xlabel("Correlation")
plt.tight_layout()
plt.savefig("outputs_target_correlation_bar.png")
plt.close()

print("\nTop correlated features with target:\n", target_corr.head(10))

# ---------------------------------------------------------
# 3. CREATE A CLASSIFICATION TARGET (optional, for metrics
#    like confusion matrix / ROC-AUC later)
# ---------------------------------------------------------
threshold = df["FloodProbability"].median()  # or use 0.5
df["HighRisk"] = (df["FloodProbability"] >= threshold).astype(int)
print("\nHighRisk class balance:\n", df["HighRisk"].value_counts(normalize=True))

# ---------------------------------------------------------
# 4. TRAIN/TEST SPLIT
# ---------------------------------------------------------
X = df.drop(columns=["FloodProbability", "HighRisk"])
y_reg = df["FloodProbability"]       # for regression models
y_clf = df["HighRisk"]               # for classification models

X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ---------------------------------------------------------
# 5. FEATURE SCALING
# ---------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 6. SAVE PROCESSED DATA FOR THE NEXT STAGE (modeling)
# ---------------------------------------------------------
np.save("X_train_scaled.npy", X_train_scaled)
np.save("X_test_scaled.npy", X_test_scaled)
y_reg_train.to_csv("y_reg_train.csv", index=False)
y_reg_test.to_csv("y_reg_test.csv", index=False)
y_clf_train.to_csv("y_clf_train.csv", index=False)
y_clf_test.to_csv("y_clf_test.csv", index=False)

print("\nPreprocessing complete. Scaled arrays and splits saved.")
print("Plots saved: outputs_target_distribution.png, outputs_correlation_heatmap.png, outputs_target_correlation_bar.png")