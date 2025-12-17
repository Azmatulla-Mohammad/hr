# 02_eda_visuals.py
# Author: Azmatulla Mohammad
# Purpose: Generate EDA plots for HR attrition dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load cleaned dataset
df = pd.read_csv("data/cleaned_hr_data.csv")
os.makedirs("visuals", exist_ok=True)

# Plot 1: Attrition count
plt.figure(figsize=(5, 4))
sns.countplot(x='Attrition', data=df)
plt.title("Attrition Count (0 = Stay, 1 = Leave)")
plt.savefig("visuals/attrition_count.png")
plt.close()

# Plot 2: Age distribution by attrition
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x='Age', hue='Attrition', kde=True, bins=30)
plt.title("Age vs Attrition")
plt.savefig("visuals/age_vs_attrition.png")
plt.close()

# Plot 3: Monthly Income vs Attrition
plt.figure(figsize=(6, 4))
sns.boxplot(x='Attrition', y='MonthlyIncome', data=df)
plt.title("Monthly Income vs Attrition")
plt.savefig("visuals/income_vs_attrition.png")
plt.close()

# Plot 4: Correlation heatmap
plt.figure(figsize=(12, 10))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, cmap='coolwarm', center=0, annot=False)
plt.title("Correlation Heatmap")
plt.savefig("visuals/correlation_heatmap.png")
plt.close()

print("✅ All EDA plots saved to /visuals/")
