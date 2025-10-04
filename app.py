import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.express as px

# ===== CUSTOM BACKGROUND (online image, no sidebar) =====
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.pexels.com/photos/957010/pexels-photo-957010.jpeg");
        background-size: cover;
        background-attachment: fixed;
        color: white;
    }
    .css-1d391kg, .css-18e3th9, .css-10trblm, .stMarkdown, .stText, .stMetric, .stTitle {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===== LOAD MODEL =====
MODEL_PATH = "models/final_model.h5"
model = load_model(MODEL_PATH)

# ===== TITLE =====
st.title("🚀 Exoplanet Detector by ASID Robotics")
st.write("Upload a light curve file (CSV with time and flux columns) to detect possible exoplanet transits.")

# ===== FILE UPLOAD (main area, no sidebar) =====
uploaded_file = st.file_uploader("Upload light curve CSV", type=["csv"])

if uploaded_file is not None:
    # Load uploaded file
    df = pd.read_csv(uploaded_file)

    # Time slider (if time column exists)
    if "time" in df.columns:
        min_time, max_time = float(df["time"].min()), float(df["time"].max())
        start, end = st.slider(
            "Select time range", 
            min_value=min_time, max_value=max_time, 
            value=(min_time, max_time)
        )
        df = df[(df["time"] >= start) & (df["time"] <= end)]
        fig = px.line(df, x="time", y="flux", title="Light Curve", template="plotly_dark")
    else:
        fig = px.line(df, y=df.columns[0], title="Light Curve", template="plotly_dark")

    st.plotly_chart(fig, use_container_width=True)

    # Preprocess flux
    flux = df.iloc[:, -1].values
    flux = flux[:2048] if len(flux) > 2048 else np.pad(flux, (0, 2048 - len(flux)))
    flux = flux[np.newaxis, :, np.newaxis].astype(np.float32)
    flux = (flux - flux.mean()) / flux.std()

    # Prediction
    pred = model.predict(flux)[0][0]
    label = "🌍 Possible Exoplanet Detected!" if pred > 0.5 else "❌ No Exoplanet Transit Detected"

    st.subheader("Prediction")
    st.metric(label="Confidence", value=f"{pred:.3f}")
    st.success(label if pred > 0.5 else label)

    # Download button
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Cleaned Light Curve", data=csv, file_name="cleaned_lightcurve.csv", mime="text/csv")



















