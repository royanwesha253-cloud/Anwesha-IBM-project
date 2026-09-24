"""
ui.py  —  Streamlit Frontend
Sri Lanka District Flood Risk Prediction Dashboard
Features:
  - District selector + climate input sliders
  - Calls Flask backend /predict endpoint
  - Shows predicted flood risk with confidence
  - Charts: rainfall trend, risk by district, correlation heatmap
  - Works standalone if backend is unavailable (demo mode)
"""

import os
import sys
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
BACKEND_URL = "http://localhost:5000"
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_PATH   = BASE_DIR / "data" / "sl_flood_risk_dataset.csv"

st.set_page_config(
    page_title="Sri Lanka Flood Risk Predictor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Load dataset (local) — used for charts even without backend
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()

df = load_data()

# ─────────────────────────────────────────────────────────────
# Backend helpers
# ─────────────────────────────────────────────────────────────
def backend_alive():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

def fetch_districts():
    try:
        r = requests.get(f"{BACKEND_URL}/districts", timeout=3)
        return r.json().get("districts", [])
    except Exception:
        return sorted(df["District"].unique().tolist()) if not df.empty else []

def call_predict(payload: dict):
    try:
        r = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def fetch_metrics():
    try:
        r = requests.get(f"{BACKEND_URL}/metrics", timeout=3)
        return r.json()
    except Exception:
        return {}

# ─────────────────────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────────────────────
st.title("🌊 Sri Lanka District Flood Risk Prediction Dashboard")
st.markdown(
    "Predict flood risk levels (Low / Medium / High) based on climate variables "
    "using a machine learning model trained on 2015–2024 district data."
)

alive = backend_alive()
if alive:
    st.success("✅ Backend API is online", icon="✅")
else:
    st.warning("⚠️ Backend API is offline — prediction requires the Flask server to be running.", icon="⚠️")

# ─────────────────────────────────────────────────────────────
# Sidebar — input controls
# ─────────────────────────────────────────────────────────────
st.sidebar.header("🔧 Input Climate Parameters")

districts = fetch_districts() or ["Colombo", "Ratnapura", "Kegalle", "Gampaha", "Kandy"]
selected_district = st.sidebar.selectbox("Select District", districts, index=0)

st.sidebar.markdown("---")
annual_rain   = st.sidebar.slider("Annual Rainfall (mm)",   200, 4500, 2000, step=50)
max_monthly   = st.sidebar.slider("Max Monthly Rainfall (mm)", 50, 1200, 450, step=10)
avg_temp      = st.sidebar.slider("Avg Temperature (°C)",    15.0, 35.0, 28.0, step=0.5)
max_temp      = st.sidebar.slider("Max Temperature (°C)",    20.0, 42.0, float(avg_temp) + 5.0, step=0.5)
min_temp      = st.sidebar.slider("Min Temperature (°C)",    10.0, 30.0, float(avg_temp) - 3.0, step=0.5)
humidity      = st.sidebar.slider("Humidity (%)",            20, 100, 72, step=1)
river_level   = st.sidebar.slider("River Level (m)",         0.1, 15.0, 3.5, step=0.1)
drought_idx   = st.sidebar.slider("Drought Index (0–10)",    0.0, 10.0, 4.0, step=0.1)
rainy_days    = st.sidebar.slider("Rainy Days / Year",       20, 300, 120, step=5)
wind_speed    = st.sidebar.slider("Wind Speed (km/h)",       5.0, 50.0, 18.0, step=0.5)
soil_moisture = st.sidebar.slider("Soil Moisture (%)",       5, 100, 50, step=1)
hist_floods   = st.sidebar.number_input("Historical Flood Events", 0, 20, 1, step=1)

predict_btn = st.sidebar.button("🔮 Predict Flood Risk", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Prediction Panel
# ─────────────────────────────────────────────────────────────
col_pred, col_conf = st.columns([1, 1])

with col_pred:
    st.subheader("📊 Prediction Result")
    if predict_btn:
        if not alive:
            st.error("Cannot predict: Flask backend is not running. Start it with `python backend/app.py`.")
        else:
            payload = {
                "District":                selected_district,
                "Annual_Rainfall_mm":      annual_rain,
                "Max_Monthly_Rainfall_mm": max_monthly,
                "Avg_Temperature_C":       avg_temp,
                "Max_Temperature_C":       max_temp,
                "Min_Temperature_C":       min_temp,
                "Humidity_pct":            humidity,
                "River_Level_m":           river_level,
                "Drought_Index":           drought_idx,
                "Rainy_Days":              rainy_days,
                "Wind_Speed_kmh":          wind_speed,
                "Soil_Moisture_pct":       soil_moisture,
                "Historical_Flood_Events": hist_floods,
            }
            with st.spinner("Querying model..."):
                result = call_predict(payload)

            if "error" in result:
                st.error(f"Prediction error: {result['error']}")
            else:
                risk  = result.get("flood_risk_level", "Unknown")
                color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(risk, "⚪")
                st.markdown(f"### {color} **{risk} Flood Risk**")
                st.markdown(f"**District:** {result.get('district')}")

                conf = result.get("confidence", {})
                if conf:
                    st.markdown("**Confidence Scores:**")
                    for cls, prob in sorted(conf.items()):
                        bar_color = {"Low": "green", "Medium": "orange", "High": "red"}.get(cls, "grey")
                        st.metric(label=cls, value=f"{prob * 100:.1f}%")

with col_conf:
    st.subheader("🏆 Model Performance")
    metrics_data = fetch_metrics()
    if metrics_data:
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Model",    metrics_data.get("model_name", "—"))
        m_col1.metric("Accuracy", f"{metrics_data.get('accuracy', 0):.1%}")
        m_col2.metric("F1 Score", f"{metrics_data.get('f1_score', 0):.1%}")
        m_col2.metric("Precision",f"{metrics_data.get('precision', 0):.1%}")
        st.metric("Recall", f"{metrics_data.get('recall', 0):.1%}")
    else:
        st.info("Start backend to see live metrics.")

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# Charts section
# ─────────────────────────────────────────────────────────────
if not df.empty:
    st.header("📈 Data Visualisations")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📉 Rainfall Trend",
        "🏘️ Risk by District",
        "🔥 Correlation Heatmap",
        "📅 Year × District Heatmap",
    ])

    # --- Tab 1: Rainfall trend ---
    with tab1:
        st.subheader("Mean Annual Rainfall (All Districts, 2015–2024)")
        district_filter = st.multiselect(
            "Filter Districts (leave empty = show all-district mean)",
            options=sorted(df["District"].unique()),
            default=[],
        )
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        if district_filter:
            for dist in district_filter:
                sub = df[df["District"] == dist].sort_values("Year")
                ax1.plot(sub["Year"], sub["Annual_Rainfall_mm"], marker="o", label=dist, linewidth=2)
            ax1.legend(fontsize=8, ncol=3)
            ax1.set_title("Annual Rainfall by Selected Districts", fontsize=12, fontweight="bold")
        else:
            yearly = df.groupby("Year")["Annual_Rainfall_mm"].mean()
            ax1.plot(yearly.index, yearly.values, marker="o", color="#3b82d4", linewidth=2)
            ax1.fill_between(yearly.index, yearly.values, alpha=0.15, color="#3b82d4")
            ax1.set_title("Mean Annual Rainfall Across All Districts (2015–2024)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Year"); ax1.set_ylabel("Rainfall (mm)")
        ax1.set_xticks(df["Year"].unique())
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    # --- Tab 2: Risk by district ---
    with tab2:
        st.subheader("Flood Risk Level Distribution by District")
        risk_dist = df.groupby(["District", "Flood_Risk_Level"]).size().unstack(fill_value=0)
        for col in ["Low", "Medium", "High"]:
            if col not in risk_dist.columns:
                risk_dist[col] = 0
        risk_dist = risk_dist[["Low", "Medium", "High"]]
        risk_dist["High_frac"] = risk_dist["High"] / risk_dist.sum(axis=1)
        risk_dist = risk_dist.sort_values("High_frac", ascending=False).drop(columns="High_frac")

        fig2, ax2 = plt.subplots(figsize=(14, 5))
        risk_dist.plot(kind="bar", stacked=True, ax=ax2,
                       color=["#4ade80", "#facc15", "#f87171"], edgecolor="white")
        ax2.set_title("Flood Risk Level by District (2015–2024)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("District"); ax2.set_ylabel("Years")
        plt.xticks(rotation=45, ha="right", fontsize=9)
        ax2.legend(title="Risk Level")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # --- Tab 3: Correlation heatmap ---
    with tab3:
        st.subheader("Correlation Heatmap of Climate Variables")
        num_cols = [
            "Annual_Rainfall_mm", "Max_Monthly_Rainfall_mm", "Avg_Temperature_C",
            "Humidity_pct", "River_Level_m", "Drought_Index", "Rainy_Days",
            "Soil_Moisture_pct", "Historical_Flood_Events", "Flood_Risk_Score"
        ]
        corr = df[num_cols].corr()
        fig3, ax3 = plt.subplots(figsize=(11, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    linewidths=0.5, ax=ax3, annot_kws={"size": 8})
        ax3.set_title("Climate Variable Correlation", fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    # --- Tab 4: Year × District heatmap ---
    with tab4:
        st.subheader("Annual Rainfall Heatmap — District × Year")
        pivot = df.pivot(index="District", columns="Year", values="Annual_Rainfall_mm")
        fig4, ax4 = plt.subplots(figsize=(13, 9))
        sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".0f", linewidths=0.3,
                    ax=ax4, annot_kws={"size": 7})
        ax4.set_title("Annual Rainfall (mm) — District × Year", fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

else:
    st.warning("Dataset not found. Run `python generate_dataset.py` from the project root first.")

# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#57606a;font-size:12px;'>"
    "Sri Lanka Flood Risk Prediction | Anwesha Roy | 2015–2024 Climate Dataset"
    "</div>",
    unsafe_allow_html=True,
)
