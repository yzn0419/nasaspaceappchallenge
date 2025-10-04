import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from exoplanet import normalize, pad_or_trim  # reuse functions from exoplanet.py

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("Untitled design.jpg");
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

# Load pre-trained model
MODEL_PATH = "models/final_model.h5"
model = load_model(MODEL_PATH)

st.title("🌌Exovision")
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

    # --- Slider to select time window in DAYS ---
    min_time, max_time = float(np.nanmin(time)), float(np.nanmax(time))
    start, end = st.slider(
        "Select time window (days)",
        min_value=int(min_time),
        max_value=int(max_time),
        value=(int(min_time), int(min_time) + 10),  # default = 10-day window
        step=1
    )

    # --- Button to auto-zoom into deepest flux dip ---
    auto_zoom = st.button("🔎 Auto Zoom to Deepest Transit")
    if auto_zoom:
        dip_index = np.nanargmin(flux_norm)  # index of deepest dip
        dip_time = time[dip_index]
        # center window ~5 days around dip
        start, end = int(dip_time) - 5, int(dip_time) + 5
        if start < min_time:
            start = int(min_time)
        if end > max_time:
            end = int(max_time)

    # Apply time mask
    mask = (time >= start) & (time <= end)
    time_zoom = time[mask]
    flux_zoom = flux_norm[mask]

    # Plot with matplotlib
    st.subheader("Light Curve (Zoomable)")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(time_zoom, flux_zoom, s=2, color="blue")
    ax.set_xlabel("Time (BJD - 2457000)")
    ax.set_ylabel("Normalized Flux")
    ax.set_title(f"Light Curve (Time {start} - {end} days)")
    st.pyplot(fig)

    # Prediction (using original flux only, not zoomed)
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










































