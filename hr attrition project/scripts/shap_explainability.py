

import pandas as pd
import shap
import matplotlib.pyplot as plt
import pickle
import os

# -----------------------------
# Load cleaned dataset
# -----------------------------
df = pd.read_csv("data/cleaned_hr_data.csv")
X = df.drop("Attrition", axis=1)

# -----------------------------
# Load trained model & scaler
# -----------------------------
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -----------------------------
# Scale data (same as training)
# -----------------------------
X_scaled = pd.DataFrame(
    scaler.transform(X),
    columns=X.columns
)

# -----------------------------
# SHAP Linear Explainer
# -----------------------------
explainer = shap.LinearExplainer(
    model,
    X_scaled,
    feature_perturbation="interventional"
)

# -----------------------------
# Compute SHAP values
# -----------------------------
shap_values = explainer.shap_values(X_scaled)

# -----------------------------
# Output directory
# -----------------------------
os.makedirs("report", exist_ok=True)

# -----------------------------
# SHAP Beeswarm Plot
# -----------------------------
shap.summary_plot(
    shap_values,
    X_scaled,
    show=False
)
plt.title("SHAP Summary Plot – Employee Attrition")
plt.tight_layout()
plt.savefig("report/shap_summary_beeswarm.png", dpi=300)
plt.close()

# -----------------------------
# SHAP Feature Importance (Bar)
# -----------------------------
shap.summary_plot(
    shap_values,
    X_scaled,
    plot_type="bar",
    show=False
)
plt.title("SHAP Feature Importance – Employee Attrition")
plt.tight_layout()
plt.savefig("report/shap_feature_importance.png", dpi=300)
plt.close()

print("✅ SHAP explainability completed successfully")
print("📊 Generated:")
print("   - report/shap_summary_beeswarm.png")
print("   - report/shap_feature_importance.png")
