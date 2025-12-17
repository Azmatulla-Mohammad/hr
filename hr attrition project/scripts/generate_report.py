# scripts/generate_report.py
# Author: Azmatulla Mohammad
# Purpose: Create a PDF report with model evaluation

from fpdf import FPDF
from datetime import datetime
import os

# Create report folder if needed
os.makedirs("report", exist_ok=True)

# ✅ Initialize PDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=14)

# ✅ Title
pdf.set_text_color(0, 102, 204)
pdf.cell(200, 10, txt="HR Analytics - Model Evaluation Report", ln=True, align='C')


# ✅ Date
pdf.set_text_color(0, 0, 0)
pdf.set_font("Arial", size=11)
pdf.cell(200, 10, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')

pdf.ln(10)

# ✅ Section: Description
pdf.set_font("Arial", 'B', 12)
pdf.cell(200, 10, txt="Model Used: Logistic Regression (Balanced)", ln=True)
pdf.set_font("Arial", size=11)
pdf.multi_cell(0, 10,
               txt="The model was trained to predict employee attrition using features such as Age, Monthly Income, and OverTime. "
                   "To address class imbalance, class weights were applied.")

pdf.ln(5)

# ✅ Section: Confusion Matrix
conf_matrix_path = "report/logistic_confusion_matrix.png"
if os.path.exists(conf_matrix_path):
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Confusion Matrix", ln=True)
    pdf.image(conf_matrix_path, x=30, w=150)
else:
    pdf.set_text_color(255, 0, 0)
    pdf.cell(200, 10, txt="Confusion Matrix image not found.", ln=True)

pdf.ln(10)

# ✅ Section: Notes
pdf.set_text_color(0, 0, 0)
pdf.set_font("Arial", 'B', 12)
pdf.cell(200, 10, txt="Interpretation", ln=True)
pdf.set_font("Arial", size=11)
pdf.multi_cell(0, 10,
               txt="High precision for the 'Leave' class means the model accurately identifies those likely to leave. "
                   "With class weighting applied, the model now provides more balanced predictions across both classes.")

# ✅ Save PDF
output_path = "report/HR_Analytics_Attrition_Report.pdf"
pdf.output(output_path)
print(f"✅ PDF report saved to {output_path}")
