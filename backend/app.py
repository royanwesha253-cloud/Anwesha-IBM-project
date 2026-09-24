"""
app.py  —  Flask Backend API
Sri Lanka District Flood Risk Prediction Service
Endpoints:
  GET  /health            — liveness check
  GET  /metrics           — model performance metrics
  GET  /districts         — list of all district names
  GET  /dataset_summary   — summary statistics of the training dataset
  POST /predict           — predict flood risk level for given climate inputs
  POST /predict_batch     — batch prediction (list of records)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "model", "flood_risk_model.pkl")
METRICS_PATH  = os.path.join(BASE_DIR, "model", "model_metrics.json")
ENCODER_PATH  = os.path.join(BASE_DIR, "model", "label_encoder.pkl")
FEATURE_PATH  = os.path.join(BASE_DIR, "model", "feature_names.json")
DATA_PATH     = os.path.join(BASE_DIR, "data",  "sl_flood_risk_dataset.csv")

# ─────────────────────────────────────────────────────────────
# Load artefacts at startup
# ─────────────────────────────────────────────────────────────
model   = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
with open(METRICS_PATH) as f:
    metrics = json.load(f)
with open(FEATURE_PATH) as f:
    FEATURES = json.load(f)

df_raw = pd.read_csv(DATA_PATH)

# District-level baseline mean rainfall (needed for anomaly features)
district_mean_rain = df_raw.groupby("District")["Annual_Rainfall_mm"].mean().to_dict()
# Rolling-3 baseline per district per year (for real-time estimation we use last 3 years available)
district_rain_history = (
    df_raw.sort_values(["District", "Year"])
          .groupby("District")["Annual_Rainfall_mm"]
          .apply(list).to_dict()
)

# ─────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)


def build_feature_vector(data: dict) -> np.ndarray:
    """
    Converts a raw input dict into the model's feature vector.
    Required raw keys: District, Annual_Rainfall_mm, Max_Monthly_Rainfall_mm,
      Avg_Temperature_C, Max_Temperature_C, Min_Temperature_C, Humidity_pct,
      River_Level_m, Drought_Index, Rainy_Days, Wind_Speed_kmh,
      Soil_Moisture_pct, Historical_Flood_Events
    Optional: Year (defaults to 2024)
    """
    district = data.get("District", "Colombo")
    rainfall  = float(data.get("Annual_Rainfall_mm", 2000))

    # Encode district
    if district in encoder.classes_:
        district_enc = int(encoder.transform([district])[0])
    else:
        district_enc = 0  # unknown district → default

    # Derived features
    mean_rain = district_mean_rain.get(district, rainfall)
    anomaly   = rainfall - mean_rain
    anomaly_pct = (anomaly / mean_rain * 100) if mean_rain else 0

    history = district_rain_history.get(district, [rainfall])
    rolling3 = float(np.mean(history[-3:]))

    max_temp = float(data.get("Max_Temperature_C", float(data.get("Avg_Temperature_C", 28)) + 5))
    min_temp = float(data.get("Min_Temperature_C", float(data.get("Avg_Temperature_C", 28)) - 3))
    temp_range = max_temp - min_temp

    humidity      = float(data.get("Humidity_pct", 70))
    soil_moisture = float(data.get("Soil_Moisture_pct", 50))
    moisture_idx  = (rainfall / 1000 * 0.4 + humidity / 100 * 0.3 + soil_moisture / 100 * 0.3)

    hist_floods   = int(data.get("Historical_Flood_Events", 0))
    high_hist     = 1 if hist_floods >= 3 else 0

    row = {
        "District_Enc":           district_enc,
        "Annual_Rainfall_mm":     rainfall,
        "Max_Monthly_Rainfall_mm": float(data.get("Max_Monthly_Rainfall_mm", rainfall * 0.22)),
        "Avg_Temperature_C":      float(data.get("Avg_Temperature_C", 28)),
        "Humidity_pct":           humidity,
        "River_Level_m":          float(data.get("River_Level_m", 3.0)),
        "Drought_Index":          float(data.get("Drought_Index", 5.0)),
        "Rainy_Days":             int(data.get("Rainy_Days", 120)),
        "Wind_Speed_kmh":         float(data.get("Wind_Speed_kmh", 18)),
        "Soil_Moisture_pct":      soil_moisture,
        "Historical_Flood_Events": hist_floods,
        "Rainfall_Anomaly":       anomaly,
        "Rainfall_Anomaly_pct":   anomaly_pct,
        "Rainfall_Rolling3":      rolling3,
        "Temp_Range_C":           temp_range,
        "Moisture_Index":         moisture_idx,
        "High_Historical_Flood":  high_hist,
    }
    return np.array([[row[f] for f in FEATURES]])


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": metrics.get("model_name", "unknown")})


@app.route("/metrics", methods=["GET"])
def get_metrics():
    return jsonify(metrics)


@app.route("/districts", methods=["GET"])
def get_districts():
    return jsonify({"districts": sorted(encoder.classes_.tolist())})


@app.route("/dataset_summary", methods=["GET"])
def dataset_summary():
    summary = {
        "total_records": len(df_raw),
        "districts":     int(df_raw["District"].nunique()),
        "years":         sorted(df_raw["Year"].unique().tolist()),
        "class_distribution": df_raw["Flood_Risk_Level"].value_counts().to_dict(),
        "feature_stats": df_raw[[
            "Annual_Rainfall_mm", "Avg_Temperature_C", "Humidity_pct",
            "River_Level_m", "Historical_Flood_Events"
        ]].describe().round(2).to_dict(),
    }
    return jsonify(summary)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    try:
        X = build_feature_vector(data)
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        classes = model.classes_.tolist()
        confidence = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}

        risk_colors = {"Low": "green", "Medium": "orange", "High": "red"}
        return jsonify({
            "district":          data.get("District", "Unknown"),
            "flood_risk_level":  prediction,
            "confidence":        confidence,
            "risk_color":        risk_colors.get(prediction, "grey"),
            "input_summary": {
                "Annual_Rainfall_mm": data.get("Annual_Rainfall_mm"),
                "Avg_Temperature_C":  data.get("Avg_Temperature_C"),
                "Humidity_pct":       data.get("Humidity_pct"),
                "River_Level_m":      data.get("River_Level_m"),
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    data = request.get_json(force=True)
    if not isinstance(data, list):
        return jsonify({"error": "Expected a JSON array of records"}), 400

    results = []
    for record in data:
        try:
            X = build_feature_vector(record)
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            classes = model.classes_.tolist()
            results.append({
                "district":         record.get("District"),
                "flood_risk_level": pred,
                "confidence":       {cls: round(float(p), 4) for cls, p in zip(classes, proba)},
            })
        except Exception as e:
            results.append({"district": record.get("District"), "error": str(e)})
    return jsonify(results)


@app.route("/district_risk_summary", methods=["GET"])
def district_risk_summary():
    summary = (
        df_raw.groupby("District")
              .agg(
                  mean_rainfall=("Annual_Rainfall_mm", "mean"),
                  mean_river_level=("River_Level_m", "mean"),
                  total_flood_events=("Historical_Flood_Events", "sum"),
                  high_risk_years=(
                      "Flood_Risk_Level",
                      lambda x: (x == "High").sum()
                  ),
              )
              .reset_index()
              .round(2)
              .to_dict(orient="records")
    )
    return jsonify(summary)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
