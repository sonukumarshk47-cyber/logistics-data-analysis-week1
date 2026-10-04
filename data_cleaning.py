import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load raw data
df = pd.read_csv("data/raw_logistics_data.csv")

print("Initial shape:", df.shape)

# 1. Remove duplicate records
df = df.drop_duplicates()

# 2. Convert data types
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

# 3. Handle missing numeric values using the median
numeric_cols = [
    "quantity", "distance_km", "shipping_time_hours",
    "inventory_units", "vehicle_utilization"
]
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# 4. Detect outliers using the IQR method
def cap_iqr(data, column):
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    data[column] = data[column].clip(lower, upper)
    return data

for col in ["distance_km", "shipping_time_hours"]:
    df = cap_iqr(df, col)

# 5. Normalize selected numerical variables
scale_cols = ["distance_km", "shipping_time_hours", "inventory_units"]
scaler = MinMaxScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# 6. Final quality checks
print("\nMissing values after cleaning:")
print(df.isnull().sum())
print("\nDuplicates after cleaning:", df.duplicated().sum())
print("\nCleaned shape:", df.shape)

# 7. Save cleaned dataset
df.to_csv("data/cleaned_logistics_data.csv", index=False)
print("\nCleaned dataset saved successfully.")
