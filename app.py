import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from exoplanet import normalize, pad_or_trim  # reuse functions from exoplanet.py
import base64

# Function to convert image to base64
def get_base64(bin_file):
    with open(bin_file, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Function to set custom background
def set_background(image_file):
    bin_str = get_base64(image_file)
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bin_str}");
        background-size: cover;
        color: white;
    }}
    h1, h2, h3, h4, h5, h6, p, div, span {{
        color: white !important;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set page background
set_background("Untitled design (5).jpg")

# Load model
MODEL_PATH = "models/final_model.h5"
model = load_model(MODEL_PATH)

# Title and description
st.title("🌌 Exovision")
st.write("Upload a light curve file (CSV with time + flux columns) to detect possible exoplanet transits.")

# ---------- CUSTOM STYLED Uploader (keeps st.file_uploader backend) ----------

uploaded_file = st.file_uploader("Upload light curve CSV", type=["csv"])
# -------------------------------------------------------------------------


# --- File processing logic ---
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Handle time + flux columns
    if "time" in df.columns and "flux" in df.columns:
        time = df["time"].values
        flux = df["flux"].values
    else:
        time = np.arange(len(df))
        flux = df.iloc[:, 0].values

    flux_norm = flux / np.nanmedian(flux)

    # Time window slider
    min_time, max_time = float(np.nanmin(time)), float(np.nanmax(time))
    start, end = st.slider(
        "Select time window (days)",
        min_value=int(min_time),
        max_value=int(max_time),
        value=(int(min_time), int(min_time) + 10),
        step=1
    )

    # Auto-zoom button
    auto_zoom = st.button("🔎 Auto Zoom to Deepest Transit")
    if auto_zoom:
        dip_index = np.nanargmin(flux_norm)
        dip_time = time[dip_index]
        start, end = int(dip_time) - 5, int(dip_time) + 5
        if start < min_time:
            start = int(min_time)
        if end > max_time:
            end = int(max_time)

    # Apply time mask
    mask = (time >= start) & (time <= end)
    time_zoom = time[mask]
    flux_zoom = flux_norm[mask]

    # Plot light curve
    st.subheader("Light Curve (Zoomable)")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(time_zoom, flux_zoom, s=2, color="blue")
    ax.set_xlabel("Time (BJD - 2457000)")
    ax.set_ylabel("Normalized Flux")
    ax.set_title(f"Light Curve (Time {start} - {end} days)")
    st.pyplot(fig)

    # Prediction
    flux_input = pad_or_trim(flux, 2048)
    flux_input = flux_input[np.newaxis, :, np.newaxis].astype(np.float32)
    flux_input = normalize(flux_input)

    pred = model.predict(flux_input)[0][0]
    label = "🌍 Possible Exoplanet Detected!" if pred > 0.5 else "❌ No Exoplanet Transit Detected"

    st.subheader("Results")
    st.write(f"Confidence: {pred:.3f}")
    if pred > 0.5:
        st.success(label)
    else:
        st.error(label)














































































