# pipeline/train_models.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import joblib

# Load dataset
df = pd.read_csv("data/karachi_aqi_log.csv")

# Drop rows with missing values
df = df.dropna()

# Features and target
X = df[["pm2_5", "pm10", "temperature", "humidity"]]
y = df["aqi"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(),
    "RandomForest": RandomForestRegressor(),
    "GradientBoosting": GradientBoostingRegressor(),
    "XGBoost": XGBRegressor(eval_metric="rmse")
}

# Train and save models
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    print(f"{name} → RMSE: {rmse:.2f}, R²: {r2:.2f}")

    # Save trained model
    joblib.dump(model, f"models/{name}.pkl")
