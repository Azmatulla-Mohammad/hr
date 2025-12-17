# 01_data_cleaning.py
# Author: Azmatulla Mohammad
# Purpose: Clean IBM HR dataset for model training

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 📥 Load the original dataset
DATA_PATH = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded data from {DATA_PATH} with shape: {df.shape}")

# 🧹 Drop irrelevant or constant columns
columns_to_drop = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]
df.drop(columns=columns_to_drop, inplace=True)
print(f"🧹 Dropped columns: {columns_to_drop}")

# 🔁 Encode binary categorical columns
binary_cols = ["Attrition", "Gender", "OverTime"]
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])
    print(f"🔁 Encoded column: {col}")

# 🔄 One-hot encode multi-class categorical features
df = pd.get_dummies(df, drop_first=True)
print(f"📊 One-hot encoded remaining categorical columns. Final shape: {df.shape}")

# 💾 Save the cleaned dataset
OUTPUT_PATH = "data/cleaned_hr_data.csv"
df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Cleaned data saved to {OUTPUT_PATH}")
