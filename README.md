# 🌊 Sri Lanka District Flood Risk Prediction (2015–2024)

**Author:** Anwesha Roy  
**Domain:** Climate Analytics · Machine Learning · Web APIs

---

## 📋 Project Description

This project builds an end-to-end flood risk prediction system for Sri Lanka's 25 administrative districts using climate and hydrological data spanning **2015 to 2024**.

Given district-level inputs such as annual rainfall, temperature, humidity, river level, and historical flood events, the system classifies the **flood risk level** as **Low**, **Medium**, or **High**.

The project includes:
- A fully synthetic but realistic dataset generated from known Sri Lanka climate patterns
- A machine learning pipeline (EDA → feature engineering → model training → evaluation)
- A REST API backend (Flask)
- An interactive dashboard frontend (Streamlit)
- A Jupyter Notebook covering the full data science workflow
- A Word (.docx) project report

---

## 📂 Dataset

| Property | Value |
|----------|-------|
| Source | Synthetic — generated from Sri Lanka DMC climate patterns |
| Reference | [Sri Lanka Dept. of Meteorology](https://www.meteo.gov.lk/) |
| Kaggle equivalent | [Sri Lanka District Climate Dataset](https://www.kaggle.com/datasets/) |
| Records | 250 (25 districts × 10 years) |
| Features | 15 climate + 1 target (Flood_Risk_Level) |
| Target classes | Low / Medium / High |

### Feature Descriptions

| Feature | Description |
|---------|-------------|
| District | Sri Lankan administrative district name |
| Year | Observation year (2015–2024) |
| Annual_Rainfall_mm | Total annual rainfall in millimetres |
| Max_Monthly_Rainfall_mm | Highest single-month rainfall |
| Avg_Temperature_C | Mean annual temperature (°C) |
| Max_Temperature_C | Annual maximum temperature (°C) |
| Min_Temperature_C | Annual minimum temperature (°C) |
| Humidity_pct | Average relative humidity (%) |
| River_Level_m | Mean river level (metres) |
| Drought_Index | Composite drought severity (0–10) |
| Rainy_Days | Number of days with measurable rainfall |
| Wind_Speed_kmh | Mean wind speed (km/h) |
| Soil_Moisture_pct | Estimated soil moisture content (%) |
| Historical_Flood_Events | Count of recorded flood events |
| Flood_Risk_Score | Continuous risk score (0–100) |
| **Flood_Risk_Level** | **Target — Low / Medium / High** |

---

## 🧰 Technologies Used

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Data | pandas, numpy |
| ML | scikit-learn (RandomForestClassifier, GradientBoostingClassifier) |
| Visualisation | matplotlib, seaborn |
| Backend API | Flask 3.0 |
| Frontend | Streamlit 1.31 |
| Model persistence | joblib |
| Report generation | python-docx |
| Notebook | Jupyter / nbformat |

---

## 🗂️ Project Structure

```
flood_risk_project/
├── data/
│   └── sl_flood_risk_dataset.csv       ← generated dataset
├── model/
│   ├── flood_risk_model.pkl            ← trained model
│   ├── label_encoder.pkl               ← district encoder
│   ├── feature_names.json              ← ordered feature list
│   └── model_metrics.json              ← evaluation metrics
├── backend/
│   └── app.py                          ← Flask API
├── frontend/
│   └── ui.py                           ← Streamlit dashboard
├── report_images/                      ← EDA + evaluation charts
├── generate_dataset.py                 ← synthetic data generator
├── train_model.py                      ← ML training pipeline
├── requirements.txt
├── README.md
├── AnweshaRoy_FloodRiskPrediction.ipynb
└── AnweshaRoy_ProjectReport.docx
```

---

## ⚙️ Setup & Run Instructions

### 1. Prerequisites
- Python 3.10 or later
- pip

### 2. Install dependencies
```bash
cd flood_risk_project
pip install -r requirements.txt
```

### 3. Generate the dataset
```bash
python generate_dataset.py
```
This creates `data/sl_flood_risk_dataset.csv`.

### 4. Train the model
```bash
python train_model.py
```
Outputs:
- `model/flood_risk_model.pkl`
- `model/model_metrics.json`
- `report_images/fig1_*.png` … `fig8_*.png`

### 5. Start the Flask backend
```bash
python backend/app.py
```
API runs on `http://localhost:5000`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check |
| `/metrics` | GET | Model performance metrics |
| `/districts` | GET | List of all 25 districts |
| `/dataset_summary` | GET | Dataset statistics |
| `/predict` | POST | Single flood risk prediction |
| `/predict_batch` | POST | Batch predictions |
| `/district_risk_summary` | GET | Per-district risk summary |

**Example `/predict` request:**
```json
POST http://localhost:5000/predict
{
  "District": "Ratnapura",
  "Annual_Rainfall_mm": 3800,
  "Avg_Temperature_C": 27.0,
  "Humidity_pct": 88,
  "River_Level_m": 6.5,
  "Historical_Flood_Events": 4
}
```

### 6. Launch the Streamlit frontend
```bash
streamlit run frontend/ui.py
```
Opens at `http://localhost:8501`

---

## 🔑 Key Findings

1. **High-risk districts** are concentrated in the **Wet Zone** — Ratnapura, Kegalle, Colombo, Kalutara — due to high rainfall and low-lying terrain.
2. **Annual rainfall** is the single strongest predictor of flood risk, followed by **river level** and **historical flood events**.
3. **La Niña years** (2017, 2020, 2022) showed elevated flood occurrences across western and southern districts.
4. **El Niño years** (2015–16, 2023) caused below-normal rainfall, reducing flood risk in the dry zone but intensifying drought.
5. The **Random Forest classifier** achieved >90% accuracy and F1 score on the test set.
6. Engineered features — **Rainfall Anomaly** and **Moisture Index** — contributed significantly to model performance.
7. Districts like **Mannar**, **Jaffna**, and **Hambantota** consistently showed **Low** flood risk due to semi-arid conditions.

---

## 📄 License

This project is for academic and demonstration purposes. The dataset is synthetically generated.
