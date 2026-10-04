import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("data/sample_logistics_data.csv")
df = df.drop_duplicates()

for col in ["order_date", "promised_date", "delivery_date"]:
    df[col] = pd.to_datetime(df[col])

df["delay_hours"] = (
    (df["delivery_date"] - df["promised_date"]).dt.total_seconds() / 3600
)
df["is_late"] = (df["delay_hours"] > 0).astype(int)

on_time = (1 - df["is_late"].mean()) * 100
late = df["is_late"].mean() * 100
avg_delay = df["delay_hours"].mean()
avg_cost = df["delivery_cost"].mean()

print("LOGISTICS KPI REPORT")
print(f"On-time delivery rate: {on_time:.2f}%")
print(f"Late delivery rate: {late:.2f}%")
print(f"Average delay: {avg_delay:.2f} hours")
print(f"Average cost per delivery: INR {avg_cost:.2f}")

summary = df.groupby("carrier").agg(
    deliveries=("order_id", "count"),
    average_delay=("delay_hours", "mean"),
    late_rate=("is_late", "mean")
)
summary["late_rate"] *= 100
print("\nCarrier performance:")
print(summary.round(2))

features = [
    "distance_km", "traffic_level",
    "order_quantity", "warehouse_processing_hours"
]
X = df[features]
y = df["is_late"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("\nLate-delivery prediction:")
print(classification_report(y_test, pred, zero_division=0))

df.groupby("carrier")["delay_hours"].mean().plot(kind="bar")
plt.title("Average Delivery Delay by Carrier")
plt.ylabel("Delay (hours)")
plt.xlabel("Carrier")
plt.tight_layout()
plt.savefig("carrier_delay.png", dpi=150)
plt.show()
