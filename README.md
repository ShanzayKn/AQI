# AQI-Prediction

# 🌍 Karachi AQI Monitoring & Prediction

This project collects **real-time AQI data** for Karachi and trains machine learning models to **predict AQI values** based on PM2.5, PM10, temperature, and humidity.

## 🚀 Features
- Hourly AQI data fetched from [AQICN API](https://aqicn.org/api/) via GitHub Actions CI/CD.
- Data stored in `data/karachi_aqi_log.csv`.
- Daily model training pipeline with:
  - Linear Regression
  - Random Forest
  - XGBoost
  - LightGBM
- Trained models saved in `models/saved/`.
- Model performance metrics stored in `models/model_results.csv`.

## 📂 Project Structure

AQI/
├── data/
│ └── karachi_aqi_log.csv # Hourly AQI dataset
├── pipeline/
│ └── train_models.py # Model training script
├── models/
│ └── saved/ # Trained model .pkl files
│ └── model_results.csv # Model performance results
├── utils/
│ └── fetch_aqicn_live.py # Data fetching script
└── .github/
└── workflows/
├── fetch_aqi.yml # Hourly data fetch workflow
└── train_models.yml # Daily training workflow

## ⚙️ Workflows
- **`fetch_aqi.yml`**  
  Runs every hour to fetch AQI data and append it to `karachi_aqi_log.csv`.

- **`train_models.yml`**  
  Runs daily at midnight to train models, evaluate results, and save them.

## 📊 Results
Latest model results are stored in `models/model_results.csv`, including:
- Mean Squared Error (MSE)
- R² score

## 🔮 Future Plans
- Add visualization dashboard.
- Deploy trained models as an API for real-time AQI prediction.
- Improve feature engineering (wind speed, weather conditions).
- Compare additional models (LSTM for time series).
