import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer

# ---------- Load data ----------
DATA_PATH = "data/karachi_aqi_log.csv"
if not os.path.exists(DATA_PATH):
    print(f"❌ Data file not found at {DATA_PATH}")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)

if df.empty:
    print("❌ CSV is empty!")
    sys.exit(1)

print(f"✅ Loaded data with {len(df)} rows and {len(df.columns)} columns")

# ---------- Clean / preprocess ----------
df = df.dropna(subset=["aqi", "pm2_5"])   # must have AQI & PM2.5

# Select feature columns that actually exist & have some data
feature_cols = ["pm2_5", "pm10", "temperature", "humidity"]
feature_cols = [c for c in feature_cols if c in df.columns]

# Drop any feature columns that are fully NaN
valid_features = []
for col in feature_cols:
    if df[col].notna().sum() > 0:   # at least 1 valid value
        valid_features.append(col)
    else:
        print(f"⚠️ Dropping {col} (all values are NaN)")

X = df[valid_features].copy()
y = df["aqi"]

# Impute remaining missing values with median
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=valid_features)

print("✅ After cleaning:")
print(X.head())

# ---------- Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- Models ----------
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "XGBoost": XGBRegressor(objective="reg:squarederror", random_state=42),
    "LightGBM": LGBMRegressor(random_state=42),
    "CatBoost": CatBoostRegressor(verbose=0, random_state=42)
}

results = []
MODEL_DIR = "models/saved"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------- Train & Evaluate ----------
for name, model in models.items():
    print(f"🚀 Training {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    results.append({"Model": name, "MSE": mse, "R2": r2})

    # Save model
    model_path = os.path.join(MODEL_DIR, f"{name}.pkl")
    joblib.dump(model, model_path)
    print(f"💾 Saved {name} → {model_path}")

# ---------- Save results ----------
results_df = pd.DataFrame(results)
os.makedirs("models", exist_ok=True)
results_df.to_csv("models/model_results.csv", index=False)

print("\n✅ Training complete. Results:")
print(results_df)
