import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.express as px
import base64

# ==== BACKGROUND SETUP ====
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(jpg_file):
    bin_str = get_base64(jpg_file)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{bin_str}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call background function with your image
set_background("pexels-umkreisel-app-957010.jpg")

# ==== APP TITLE ====
MODEL_PATH = "models/final_model.h5"
model = load_model(MODEL_PATH)

st.title("🚀 Exoplanet Detector by ASID Robotics")
st.write("Upload a light curve file (CSV with a single flux column) to detect possible exoplanet transits.")

# ==== FILE UPLOAD IN SIDEBAR ====
uploaded_file = st.sidebar.file_uploader("Upload light curve CSV", type=["csv"])

if uploaded_file is not None:
    # Load uploaded file
    df = pd.read_csv(uploaded_file)

    # Time slider
    if "time" in df.columns:
        min_time, max_time = float(df["time"].min()), float(df["time"].max())
        start, end = st.slider("Select time range", min_value=min_time, max_value=max_time, value=(min_time, max_time))
        df = df[(df["time"] >= start) & (df["time"] <= end)]
        fig = px.line(df, x="time", y="flux", title="Light Curve", template="plotly_dark")
    else:
        fig = px.line(df, y=df.columns[0], title="Light Curve", template="plotly_dark")

    st.plotly_chart(fig, use_container_width=True)

    # Assume flux column is first column
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

    # Download button for cleaned CSV
    csv = df.to_c_

















