import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from exoplanet import normalize, pad_or_trim  # reuse functions from exoplanet.py

# Load pre-trained model
MODEL_PATH = "models/final_model.h5"
model = load_model(MODEL_PATH)

st.title("🚀 Exoplanet Detector by ASID Robotics")
st.write("Upload a light curve file (CSV with time + flux columns) to detect possible exoplanet transits.")

uploaded_file = st.file_uploader("Upload light curve CSV", type=["csv"])

if uploaded_file is not None:
    # Load uploaded file
    df = pd.read_csv(uploaded_file)

    # Check if file has time + flux or just flux
    if "time" in df.columns and "flux" in df.columns:
        time = df["time"].values
        flux = df["flux"].values
    else:
        # fallback if only flux column exists
        time = np.arange(len(df))
        flux = df.iloc[:, 0].values

    # Normalize flux for clearer dips
    flux_norm = flux / np.nanmedian(flux)

    # --- Interactive slider for zooming ---
    min_time, max_time = float(np.nanmin(time)), float(np.nanmax(time))
    start, end = st.slider(
        "Select time window to zoom",
        min_value=min_time,
        max_value=max_time,
        value=(min_time, min_time + (max_time - min_time) * 0.1)  # default first 10% of data
    )

    # Apply time mask based on slider
    mask = (time >= start) & (time <= end)
    time_zoom = time[mask]
    flux_zoom = flux_norm[mask]

    # Plot with matplotlib for better control
    st.subheader("Light Curve (Zoomable)")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(time_zoom, flux_zoom, s=2, color="blue")
    ax.set_xlabel("Time (BJD - 2457000)")
    ax.set_ylabel("Normalized Flux")
    ax.set_title(f"Light Curve (Time {start:.1f} - {end:.1f})")
    st.pyplot(fig)

    # Prediction (using original flux only, not zoomed)
    flux_input = pad_or_trim(flux, 2048)
    flux_input = flux_input[np.newaxis, :, np.newaxis].astype(np.float32)
    flux_input = normalize(flux_input)

    pred = model.predict(flux_input)[0][0]
    label = "🌍 Possible Exoplanet Detected!" if pred > 0.5 else "❌ No Exoplanet Transit Detected"

    st.subheader("Prediction")
    st.write(f"Confidence: {pred:.3f}")
    if pred > 0.5:
        st.success(label)
    else:
        st.error(label)







