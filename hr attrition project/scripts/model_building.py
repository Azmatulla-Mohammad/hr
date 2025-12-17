# 03_model_building.py
# Author: Azmatulla Mohammad
# Purpose: Train Logistic and Decision Tree models with evaluation and save best one

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pickle
import os
import matplotlib.pyplot as plt

# 📥 Load cleaned dataset
df = pd.read_csv("data/cleaned_hr_data.csv")
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# ⚙️ Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔀 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# ✅ Logistic Regression (with class_weight)
logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)

print("\n🔹 Logistic Regression")
print("Accuracy:", logreg.score(X_test, y_test))
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# ✅ Decision Tree
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

print("\n🔹 Decision Tree")
print("Accuracy:", tree.score(X_test, y_test))
print(confusion_matrix(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))

# 📊 Save Confusion Matrix for Logistic Regression
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lr, display_labels=["Stay", "Leave"], cmap="Blues")
plt.title("Logistic Regression - Confusion Matrix")
os.makedirs("report", exist_ok=True)
plt.savefig("report/logistic_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

# 💾 Save model and scaler
os.makedirs("models", exist_ok=True)
with open("models/model.pkl", "wb") as f:
    pickle.dump(logreg, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# 💾 Save feature columns for input generation
with open("models/feature_columns.json", "w") as f:
    json_cols = list(X.columns)
    import json
    json.dump(json_cols, f)

print("\n✅ Best model saved to models/model.pkl")
print("🖼️ Confusion matrix saved to report/logistic_confusion_matrix.png")
