import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer

# ----------------------------
# 1. Load Data
# ----------------------------
data_path = "data/karachi_aqi_log.csv"
df = pd.read_csv(data_path)

print(f"✅ Loaded data with {df.shape[0]} rows and {df.shape[1]} columns")

# ----------------------------
# 2. Features & Target
# ----------------------------
features = ["pm2_5", "pm10", "temperature", "humidity"]
target = "aqi"

X = df[features]
y = df[target]

# Drop all-NaN or constant columns
X = X.loc[:, X.notna().any()]  # remove all-NaN
X = X.loc[:, X.nunique() > 1]  # remove constant

print(f"✅ Features after cleaning: {list(X.columns)}")

# ----------------------------
# 3. Handle Missing Values
# ----------------------------
if not X.empty:
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

# ----------------------------
# 4. Train-Test Split
# ----------------------------
if X.shape[1] == 0:
    print("⚠️ No usable features after cleaning. Exiting training.")
    exit(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 5. Models to Train
# ----------------------------
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    "LightGBM": LGBMRegressor(n_estimators=100, random_state=42),
}

results = []

# ----------------------------
# 6. Train & Evaluate
# ----------------------------
for name, model in models.items():
    try:
        print(f"🚀 Training {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append({"Model": name, "MSE": mse, "R2": r2})

        # Save model
        os.makedirs("models/saved", exist_ok=True)
        joblib.dump(model, f"models/saved/{name}.pkl")
        print(f"💾 Saved {name} → models/saved/{name}.pkl")

    except Exception as e:
        print(f"⚠️ Skipping {name} due to error: {e}")

# ----------------------------
# 7. Save Results
# ----------------------------
results_df = pd.DataFrame(results)
os.makedirs("models", exist_ok=True)
results_df.to_csv("models/model_results.csv", index=False)

print("✅ Training complete. Results saved to models/model_results.csv")
print(results_df)
