import streamlit as st
import base64
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from exoplanet import normalize, pad_or_trim
import plotly.express as px

# ========== BACKGROUND SETUP ==========
def set_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: white;
        }}
        .stSidebar {{
            background-color: rgba(0,0,0,0.7);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("pexels-umkreisel-app-957010.jpg")  # use your uploaded image

# ========== MODEL ==========
MODEL_PATH = "models/final_model.h5"
model = load_model(MODEL_PATH)

# ========== UI ==========
st.title("🚀 Exoplanet Detector by ASID Robotics")
st.write("Upload a light curve file (CSV with a single flux column) to detect possible exoplanet transits.")

uploaded_file = st.sidebar.file_uploader("📂 Upload light curve CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Slider for selecting time range
    min_day, max_day = int(df.index.min()), int(df.index.max())
    day_range = st.slider("Select time range (days)", min_day, max_day, (min_day, max_day))
    df_range = df.iloc[day_range[0]:day_range[1]]

    # Interactive dark plot
    fig = px.line(df_range, y=df_range.columns[0], title="Light Curve", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # Preprocess for prediction
    flux = df.iloc[:, 0].values
    flux = pad_or_trim(flux, 2048)
    flux = flux[np.newaxis, :, np.newaxis].astype(np.float32)
    flux = normalize(flux)

    # Prediction
    pred = model.predict(flux)[0][0]
    label = "🌍 Possible Exoplanet Detected!" if pred > 0.5 else "❌ No Exoplanet Transit Detected"

    # Display result
    st.subheader("Prediction Result")
    if pred > 0.5:
        st.success(f"{label} (Confidence: {pred:.3f})")
    else:
        st.error(f"{label} (Confidence: {pred:.3f})")

    # Download cleaned light curve
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Cleaned Light Curve", csv, "cleaned_lightcurve.csv", "text/csv")












