import os
import sys
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import joblib

# ---------- Paths ----------
DATA_PATH = "data/karachi_aqi_log.csv"
MODELS_DIR = "models/saved"
RESULTS_CSV = "models/model_results.csv"
BEST_MODEL_PATH = "models/best_model.pkl"
BEST_MODEL_INFO_JSON = "models/best_model_info.json"

os.makedirs(MODELS_DIR, exist_ok=True)

# ---------- Load data ----------
if not os.path.exists(DATA_PATH):
    print(f"❌ CSV not found at {DATA_PATH}")
    sys.exit(0)

df = pd.read_csv(DATA_PATH)

# Quick sanity checks
if df.empty or len(df) < 20:
    print(f"❌ Not enough data to train. Found {len(df)} rows.")
    sys.exit(0)

# Make sure expected columns exist
required_cols = ["aqi", "pm2_5", "pm10", "temperature", "humidity"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    print(f"❌ Missing columns in CSV: {missing}")
    sys.exit(0)

# Clean/minimal preprocessing
df = df.dropna(subset=["aqi", "pm2_5"])  # must have target+pm2.5
df["pm10"] = df["pm10"].fillna(df["pm10"].mean())  # fill missing pm10
df["temperature"] = df["temperature"].fillna(df["temperature"].median())
df["humidity"] = df["humidity"].fillna(df["humidity"].median())

if df.empty or len(df) < 20:
    print(f"❌ Not enough usable rows after cleaning. Rows: {len(df)}")
    sys.exit(0)

X = df[["pm2_5", "pm10", "temperature", "humidity"]]
y = df["aqi"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- Define models ----------
models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="rmse",
        n_jobs=2
    ),
}

# ---------- Train, evaluate, save ----------
results = []
best_name = None
best_score = -1e9  # we’ll use R2 to pick best
best_model = None

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    results.append({
        "model": name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    })

    # save each model
    model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
    joblib.dump(model, model_path)

    # track best by R2
    if r2 > best_score:
        best_score = r2
        best_name = name
        best_model = model

# Save results CSV
results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_CSV, index=False)

# Save best model
if best_model is not None:
    joblib.dump(best_model, BEST_MODEL_PATH)
    with open(BEST_MODEL_INFO_JSON, "w") as f:
        json.dump({
            "best_model": best_name,
            "best_R2": best_score
        }, f, indent=2)

print("✅ Training complete.")
print(f"→ Results: {RESULTS_CSV}")
print(f"→ Best model: {best_name} (R2={best_score:.3f}) saved to {BEST_MODEL_PATH}")
