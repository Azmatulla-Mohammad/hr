import os
import json
import pickle
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # Safe for Flask + Windows

import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request

# =====================================================
# PATH CONFIG
# =====================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(APP_DIR, "static")
TEMPLATES_DIR = os.path.join(APP_DIR, "templates")

# =====================================================
# FLASK APP
# =====================================================
app = Flask(
    __name__,
    static_folder=STATIC_DIR,
    template_folder=TEMPLATES_DIR
)

# =====================================================
# LOAD MODEL & FILES
# =====================================================
with open(os.path.join(MODELS_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(MODELS_DIR, "feature_columns.json"), "r") as f:
    feature_columns = json.load(f)

# =====================================================
# HOME PAGE — MODE SELECTION
# =====================================================
@app.route("/")
def index():
    return render_template("index.html")

# =====================================================
# SINGLE EMPLOYEE FORM
# =====================================================
@app.route("/single")
def single_employee():
    return render_template("single_predict.html")

# =====================================================
# SINGLE EMPLOYEE PREDICTION
# =====================================================
@app.route("/predict", methods=["POST"])
def predict():
    age = int(request.form["age"])
    income = float(request.form["income"])
    hours_day = float(request.form.get("hours_day", 0))
    hours_week = float(request.form.get("hours_week", 0))


    # RULE-BASED OVERTIME
    overtime = 1 if (hours_day > 8 or hours_week > 40) else 0

    # Model input
    input_dict = {
        "Age": age,
        "MonthlyIncome": income,
        "OverTime": overtime
    }

    input_df = pd.DataFrame(
        [[input_dict.get(col, 0) for col in feature_columns]],
        columns=feature_columns
    )

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    confidence = model.predict_proba(input_scaled)[0][prediction] * 100

    # EXPLANATION
    reasons = []

    if overtime == 1:
        reasons.append("Employee works overtime based on working hours")

    if income < 25000:
        reasons.append("Monthly income is comparatively low")

    if age < 40:
        reasons.append("Employee belongs to a younger age group")

    if len(reasons) == 0:
        reasons.append("No major attrition risk factors detected")
        risk_level = "Low"
    elif len(reasons) == 1:
        risk_level = "Medium"
    else:
        risk_level = "High"

    label = "🟢 Likely to Stay" if prediction == 0 else "🟠 Likely to Leave"

    return render_template(
        "result.html",
        result=f"{label} (Confidence: {confidence:.2f}%)",
        risk_level=risk_level,
        reasons=reasons,
        age=age,
        income=income,
        overtime="Yes" if overtime else "No",
        hours_day=hours_day,
        hours_week=hours_week
    )
@app.route("/what_if", methods=["POST"])
def what_if():
    # Original inputs
    age = int(request.form["age"])
    income = float(request.form["income"])
    hours_day = float(request.form["hours_day"])
    hours_week = float(request.form["hours_week"])

    # -----------------------------
    # ORIGINAL PREDICTION
    # -----------------------------
    overtime_original = 1 if (hours_day > 8 or hours_week > 40) else 0

    original_input = {
        "Age": age,
        "MonthlyIncome": income,
        "OverTime": overtime_original
    }

    original_df = pd.DataFrame(
        [[original_input.get(col, 0) for col in feature_columns]],
        columns=feature_columns
    )

    original_scaled = scaler.transform(original_df)
    original_pred = model.predict(original_scaled)[0]
    original_conf = model.predict_proba(original_scaled)[0][original_pred] * 100

    # -----------------------------
    # WHAT-IF SCENARIO (REDUCE HOURS)
    # -----------------------------
    reduced_hours_day = min(hours_day, 8)
    reduced_hours_week = min(hours_week, 40)

    overtime_new = 1 if (reduced_hours_day > 8 or reduced_hours_week > 40) else 0

    new_input = {
        "Age": age,
        "MonthlyIncome": income,
        "OverTime": overtime_new
    }

    new_df = pd.DataFrame(
        [[new_input.get(col, 0) for col in feature_columns]],
        columns=feature_columns
    )

    new_scaled = scaler.transform(new_df)
    new_pred = model.predict(new_scaled)[0]
    new_conf = model.predict_proba(new_scaled)[0][new_pred] * 100

    # Labels
    original_label = "Leave" if original_pred == 1 else "Stay"
    new_label = "Leave" if new_pred == 1 else "Stay"

    return render_template(
        "what_if_result.html",
        age=age,
        income=income,
        original_hours_day=hours_day,
        original_hours_week=hours_week,
        reduced_hours_day=reduced_hours_day,
        reduced_hours_week=reduced_hours_week,
        original_label=original_label,
        original_conf=round(original_conf, 2),
        new_label=new_label,
        new_conf=round(new_conf, 2)
    )


# =====================================================
# DASHBOARD (DATASET-LEVEL INSIGHTS)
# =====================================================
@app.route("/dashboard")
def dashboard():
    df = pd.read_csv(os.path.join(DATA_DIR, "cleaned_hr_data.csv"))

    dashboard_dir = os.path.join(STATIC_DIR, "dashboard")
    os.makedirs(dashboard_dir, exist_ok=True)

    # Attrition Count
    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x="Attrition")
    plt.title("Employee Attrition Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(dashboard_dir, "attrition_count.png"))
    plt.close()

    # Overtime vs Attrition
    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x="OverTime", hue="Attrition")
    plt.title("Overtime vs Attrition")
    plt.tight_layout()
    plt.savefig(os.path.join(dashboard_dir, "overtime_vs_attrition.png"))
    plt.close()

    # Income vs Attrition
    plt.figure(figsize=(5, 4))
    sns.boxplot(data=df, x="Attrition", y="MonthlyIncome")
    plt.title("Monthly Income vs Attrition")
    plt.tight_layout()
    plt.savefig(os.path.join(dashboard_dir, "income_vs_attrition.png"))
    plt.close()

    return render_template(
        "dashboard.html",
        total_employees=len(df),
        attrition_rate=round(df["Attrition"].mean() * 100, 2)
    )

# =====================================================
# BATCH (CSV) PREDICTION
# =====================================================
@app.route("/batch_predict", methods=["GET", "POST"])
def batch_predict():
    if request.method == "GET":
        return render_template("batch_upload.html")

    df = pd.read_csv(request.files["file"])

    # VALIDATION
    required_cols = ["Age", "MonthlyIncome", "HoursPerDay", "HoursPerWeek"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return f"Missing required columns: {missing}"

    # RULE-BASED OVERTIME
    df["OverTime"] = (
        (df["HoursPerDay"] > 8) |
        (df["HoursPerWeek"] > 40)
    ).astype(int)

    # Build model input
    input_df = pd.DataFrame(
        [[row.get(col, 0) for col in feature_columns] for _, row in df.iterrows()],
        columns=feature_columns
    )

    input_scaled = scaler.transform(input_df)

    df["Prediction"] = model.predict(input_scaled)
    df["PredictionLabel"] = df["Prediction"].map({0: "Stay", 1: "Leave"})
    df["Confidence"] = model.predict_proba(input_scaled).max(axis=1) * 100

    total = len(df)
    leave_count = (df["Prediction"] == 1).sum()
    stay_count = (df["Prediction"] == 0).sum()

    

    return render_template(
        "batch_result.html",
        total=total,
        leave_count=leave_count,
        stay_count=stay_count,
        table=df.head(10).to_html(index=False)
    )
@app.route("/what_if_salary", methods=["POST"])
def what_if_salary():
    age = int(request.form["age"])
    income = float(request.form["income"])
    hours_day = float(request.form["hours_day"])
    hours_week = float(request.form["hours_week"])

    overtime = 1 if (hours_day > 8 or hours_week > 40) else 0

    # ORIGINAL
    base_input = {
        "Age": age,
        "MonthlyIncome": income,
        "OverTime": overtime
    }

    base_df = pd.DataFrame(
        [[base_input.get(col, 0) for col in feature_columns]],
        columns=feature_columns
    )

    base_scaled = scaler.transform(base_df)
    base_pred = model.predict(base_scaled)[0]
    base_conf = model.predict_proba(base_scaled)[0][base_pred] * 100

    # WHAT-IF: Increase salary by 20%
    new_income = income * 1.2

    new_input = {
        "Age": age,
        "MonthlyIncome": new_income,
        "OverTime": overtime
    }

    new_df = pd.DataFrame(
        [[new_input.get(col, 0) for col in feature_columns]],
        columns=feature_columns
    )

    new_scaled = scaler.transform(new_df)
    new_pred = model.predict(new_scaled)[0]
    new_conf = model.predict_proba(new_scaled)[0][new_pred] * 100

    return render_template(
        "what_if_salary.html",
        old_income=income,
        new_income=round(new_income, 2),
        before="Leave" if base_pred else "Stay",
        after="Leave" if new_pred else "Stay",
        before_conf=round(base_conf, 2),
        after_conf=round(new_conf, 2)
    )
@app.route("/policy_impact")
def policy_impact():
    df = pd.read_csv(os.path.join(DATA_DIR, "cleaned_hr_data.csv"))

    # Assume hours are available or simulated
    df["HoursPerDay"] = 9
    df["HoursPerWeek"] = 48

    # Base overtime
    df["OverTime"] = ((df["HoursPerDay"] > 8) | (df["HoursPerWeek"] > 40)).astype(int)

    # Base prediction
    base_input = pd.DataFrame(
        [[row.get(col, 0) for col in feature_columns] for _, row in df.iterrows()],
        columns=feature_columns
    )
    base_scaled = scaler.transform(base_input)
    df["BasePred"] = model.predict(base_scaled)

    # Policy: reduce hours
    df["HoursPerDay"] = 8
    df["HoursPerWeek"] = 40
    df["OverTime"] = 0

    new_input = pd.DataFrame(
        [[row.get(col, 0) for col in feature_columns] for _, row in df.iterrows()],
        columns=feature_columns
    )
    new_scaled = scaler.transform(new_input)
    df["NewPred"] = model.predict(new_scaled)

    impacted = df[(df["BasePred"] == 1) & (df["NewPred"] == 0)]

    return render_template(
        "policy_impact.html",
        count=len(impacted),
        table=impacted.head(10).to_html(index=False)
    )

# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    app.run(debug=True)
