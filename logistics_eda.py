import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/logistics_week3_dataset.csv")
df["order_date"] = pd.to_datetime(df["order_date"])

print(df.describe(numeric_only=True))
print(df[["distance_km","shipping_time_hours","transport_cost_inr",
          "vehicle_utilization_pct","customer_rating"]].corr().round(2))

# 1. Shipping-time distribution
sns.histplot(df["shipping_time_hours"], kde=True)
plt.title("Distribution of Shipping Time")
plt.xlabel("Shipping time (hours)")
plt.tight_layout()
plt.show()

# 2. Average shipping time by city
city_time = df.groupby("city")["shipping_time_hours"].mean().sort_values()
city_time.plot(kind="bar")
plt.title("Average Shipping Time by City")
plt.ylabel("Hours")
plt.tight_layout()
plt.show()

# 3. Distance vs transport cost
sns.scatterplot(data=df, x="distance_km", y="transport_cost_inr", hue="carrier")
plt.title("Distance vs Transport Cost")
plt.tight_layout()
plt.show()

# 4. Correlation heatmap
corr = df.select_dtypes("number").corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues")
plt.title("Logistics KPI Correlation Matrix")
plt.tight_layout()
plt.show()

# 5. Carrier performance
sns.boxplot(data=df, x="carrier", y="shipping_time_hours")
plt.title("Shipping Time by Carrier")
plt.tight_layout()
plt.show()
