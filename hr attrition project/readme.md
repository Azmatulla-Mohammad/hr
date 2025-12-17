# 🧠 HR Analytics – Attrition Prediction Web App

This is an interactive machine learning web application that predicts whether an employee is **likely to leave or stay** in a company. It includes a **live prediction form**, SHAP-based model explanation, batch CSV upload, downloadable report, and beautiful frontend UI.

## 🚀 Features

✅ Predict employee attrition using **Logistic Regression**  
✅ Clean, modern **Flask frontend** (HTML + CSS)  
✅ Interactive **SHAP visual explanations** (Top 3 features)  
✅ 📁 Upload CSV for **bulk predictions**  
✅ 🧾 Generate and download **PDF report**  
✅ 📊 Includes EDA visualizations and Power BI dashboard screenshots

---

## 🖼️ Screenshots

### 🔍 Single Prediction Output  
![Prediction Screenshot](static/top_features.png)

### 📊 Power BI Dashboard  
![Power BI Dashboard](app/static/hr_dashboard.png) <!-- If image available -->

---

## 🛠️ Tech Stack

| Layer      | Technology               |
|------------|---------------------------|
| Backend    | Python, Flask             |
| Frontend   | HTML, CSS (custom UI)     |
| ML Model   | Scikit-learn (Logistic Regression) |
| Visuals    | SHAP, Matplotlib, Power BI |
| Batch Tool | CSV Upload via Flask Form |
| Deployment | *(Optional)* Render / Streamlit Cloud |

---

## 📁 Project Structure

├── app/
│ ├── templates/
│ │ ├── index.html
│ │ └── result.html
│ ├── static/
│ │ ├── top_features.png
│ │ └── image_hr.webp
│ └── app.py
├── data/
│ └── cleaned_hr_data.csv
├── models/
│ ├── model.pkl
│ ├── scaler.pkl
│ └── feature_columns.json
├── report/
│ └── HR_Analytics_Attrition_Report.pdf
├── scripts/
│ ├── model_building.py
│ ├── shap_explainability.py
│ ├── generate_report.py
│ └── eda_visuals.py
├── visuals/
│ └── *.png (charts from analysis)
├── requirements.txt
└── README.md


---

## ⚙️ How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/your-username/hr-analytics-attrition

# 2. Navigate
cd hr-analytics-attrition

# 3. Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the Flask app
python app/app.py


Learnings & Contributions
🧠 Trained ML models (Logistic Regression, Decision Tree)

📊 Created insightful dashboards and EDA charts

💡 Used SHAP for explainable AI

