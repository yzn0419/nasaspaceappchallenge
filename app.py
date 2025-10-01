import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import exoplanet
normalize = exoplanet.normalize
pad_or_trim = exoplanet.pad_or_trim

# Load pre-trained model
MODEL_PATH = "models/final_model.h5"
model = load_model(MODEL_PATH)

st.title("🚀 Exoplanet Detector (ASID Robotics)")
st.write("Upload a light curve file (CSV with a single flux column) to detect possible exoplanet transits.")

uploaded_file = st.file_uploader("Upload light curve CSV", type=["csv"])

if uploaded_file is not None:
    # Load uploaded file
    df = pd.read_csv(uploaded_file)
    st.line_chart(df)  # Show the raw curve

    # Assume flux column is first column
    flux = df.iloc[:, 0].values
    flux = pad_or_trim(flux, 2048)
    flux = flux[np.newaxis, :, np.newaxis].astype(np.float32)
    flux = normalize(flux)

    # Prediction
    pred = model.predict(flux)[0][0]
    label = "🌍 Possible Exoplanet Detected!" if pred > 0.5 else "❌ No Exoplanet Transit Detected"

    st.subheader("Prediction")
    st.write(f"Confidence: {pred:.3f}")
    st.success(label if pred > 0.5 else label)


