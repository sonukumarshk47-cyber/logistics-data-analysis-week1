import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import linprog

df = pd.read_csv("data/logistics_week4_dataset.csv")
features = ["warehouse_id","city","carrier","quantity","distance_km",
            "traffic_level","warehouse_processing_hours","vehicle_utilization_pct"]
X, y = df[features], df["shipping_time_hours"]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["warehouse_id","city","carrier"]),
    ("num", StandardScaler(), ["quantity","distance_km","traffic_level",
                               "warehouse_processing_hours","vehicle_utilization_pct"])
])
model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=8)
pipeline = Pipeline([("preprocess", preprocess), ("model", model)])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)
pipeline.fit(X_train,y_train)
pred = pipeline.predict(X_test)

print("MAE:", mean_absolute_error(y_test,pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test,pred)))
print("R2:", r2_score(y_test,pred))

cv_mae = -cross_val_score(pipeline,X,y,cv=5,scoring="neg_mean_absolute_error").mean()
print("5-fold CV MAE:", cv_mae)

# Capacity allocation: minimum demand [20,15,18,12], total capacity 80
result = linprog(c=[120,150,135,180], A_ub=[[1,1,1,1]], b_ub=[80],
                 bounds=[(20,None),(15,None),(18,None),(12,None)], method="highs")
print("Optimal allocation:", result.x)
print("Minimum cost:", result.fun)
