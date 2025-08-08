# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score
import joblib
from io import StringIO

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="Occupancy Detection Dashboard",
    layout="wide",
    page_icon="🏠"
)

# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    return joblib.load("models/occupancy_final_model.pkl")  # replace with your model file

try:
    model = load_model()
except:
    model = None
    st.error("⚠️ Model file not found. Please make sure `models/occupancy_final_model.pkl` exists.")

# -------------------- Sample Data --------------------
SAMPLE_DATA = """Date,Temperature,Light,CO2,HumidityRatio,Occupancy
2025-08-01,21.5,350.5,420,0.0045,1
2025-08-02,22.0,340.2,410,0.0044,0
2025-08-03,23.1,355.0,430,0.0046,1
2025-08-04,22.8,345.0,425,0.0045,0
"""

# -------------------- Sidebar --------------------
st.sidebar.title("📂 Data Input")

data_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
if st.sidebar.button("Use Sample Data"):
    data_file = StringIO(SAMPLE_DATA)

# -------------------- Main Header --------------------
st.title("🏠 Occupancy Detection Dashboard")
st.markdown(
    "Welcome to the **Occupancy Detection Dashboard**. "
    "Upload your sensor data or use the sample dataset to analyze occupancy trends, "
    "view visualizations, and make predictions."
)
st.markdown("---")

# -------------------- If no data yet --------------------
if not data_file:
    st.info("👈 Please upload a CSV file or click **Use Sample Data** in the sidebar to get started.")
else:
    # Read data
    df = pd.read_csv(data_file)

    # Data Preview
    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(df.head())

    # Time Series
    if "Date" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["Date"])
            fig_ts = px.line(df, x="Date", y="Occupancy", title="Occupancy Over Time")
            st.plotly_chart(fig_ts, use_container_width=True)
        except:
            st.warning("⚠️ Could not parse dates in `Date` column.")

    # Correlation Heatmap
st.subheader("🔍 Sensor Data Correlation")
numeric_df = df.select_dtypes(include=[np.number])  # keep only numeric columns
if numeric_df.shape[1] > 1:
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    st.pyplot(plt.gcf())
else:
    st.info("Not enough numeric data to calculate correlations.")

    # Feature Importance
    if model and hasattr(model, "feature_importances_"):
        st.subheader("📌 Feature Importance")
        importances = model.feature_importances_
        features = df.drop(columns=["Occupancy", "Date"], errors="ignore").columns[:len(importances)]
        fig_imp = px.bar(x=features, y=importances, title="Feature Importance")
        st.plotly_chart(fig_imp, use_container_width=True)

    # Batch Prediction
    st.subheader("📦 Batch Prediction")
    input_df = df.drop(columns=["Occupancy", "Date"], errors="ignore")
    if model:
        predictions = model.predict(input_df)
        df["Predicted Occupancy"] = predictions
        st.dataframe(df)
        st.download_button("📥 Download Predictions", df.to_csv(index=False), file_name="predictions.csv")

    # Evaluation Metrics
    if "Occupancy" in df.columns and model:
        y_true = df["Occupancy"]
        y_pred = df["Predicted Occupancy"]

        st.subheader("✅ Model Performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2f}")
        col2.metric("Precision", f"{precision_score(y_true, y_pred):.2f}")
        col3.metric("Recall", f"{recall_score(y_true, y_pred):.2f}")

        # Confusion Matrix
        st.subheader("📉 Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        st.pyplot(plt.gcf())

        # ROC Curve
        if hasattr(model, "predict_proba"):
            st.subheader("📈 ROC Curve")
            y_proba = model.predict_proba(input_df)[:, 1]
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            plt.figure()
            plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')
            plt.plot([0,1], [0,1], linestyle='--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            st.pyplot(plt.gcf())

# -------------------- Single Prediction --------------------
st.sidebar.subheader("📍 Single Prediction")

temp = st.sidebar.slider("Temperature (°C)", min_value=0.0, max_value=50.0, step=0.1, help="Indoor temperature in Celsius")
light = st.sidebar.slider("Light (lux)", min_value=0.0, max_value=2000.0, step=0.1, help="Light level in lux")
co2 = st.sidebar.slider("CO2 (ppm)", min_value=0.0, max_value=5000.0, step=1.0, help="CO₂ concentration in ppm")
humidity_ratio = st.sidebar.slider("Humidity Ratio", min_value=0.0, max_value=0.02, step=0.0001, help="Ratio of water vapor mass to dry air mass")

if st.sidebar.button("Predict"):
    if model:
        single_input = pd.DataFrame([[temp, light, co2, humidity_ratio]],
                                    columns=["Temperature", "Light", "CO2", "HumidityRatio"])
        pred = model.predict(single_input)[0]
        if pred == 1:
            st.sidebar.success("✅ Predicted Occupancy: **YES**")
        else:
            st.sidebar.error("❌ Predicted Occupancy: **NO**")
    else:
        st.sidebar.warning("⚠️ Model not loaded. Cannot make predictions.")
