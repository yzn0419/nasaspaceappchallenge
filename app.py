import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from exoplanet import normalize, pad_or_trim

# Load pre-trained model
MODEL_PATH = "models/final_model.h5"
model = load_model(MODEL_PATH)

# --- CUSTOM CSS FOR SPACE BACKGROUND ---
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://i.ibb.co/sF7z0sk/starfield-bg.jpg");
    background-size: cover;
    background-attachment: fixed;
    color: white;
}
[data-testid="stSidebar"] {
    background-color: rgba(0, 0, 0, 0.7);
}
h1, h2, h3, h4, h5, h6, p, div, label {
    color: white !important;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("Upload Light Curve CSV", type=["csv"])
st.sidebar.write("Upload a light curve file (with time + flux columns).")
st.sidebar.divider()
st.sidebar.info("Model: Exoplanet Transit Detector")

# Main UI
st.title("🚀 Exoplanet Detector")
st.write("Analyze TESS light curves and detect possible exoplanet transits.")

if uploaded_file is not None:
    # Load uploaded file
    df = pd.read_csv(uploaded_file)

    # Detect time + flux
    if "time" in df.columns and "flux" in df.columns:
        time = df["time"].values
        flux = df["flux"].values
    else:
        time = np.arange(len(df))
        flux = df.iloc[:, 0].values

    # Slider
    min_time, max_time = int(time.min()), int(time.max())
    time_range = st.slider("Select time range (days)", min_time, max_time, (min_time, max_time), step=1)
    mask = (time >= time_range[0]) & (time <= time_range[1])
    st.line_chart(pd.DataFrame({"time": time[mask], "flux": flux[mask]}).set_index("time"))

    # Model preprocess
    flux_input = pad_or_trim(flux, 2048)
    flux_input = flux_input[np.newaxis, :, np.newaxis].astype(np.float32)
    flux_input = normalize(flux_input)

    pred = model.predict(flux_input)[0][0]
    confidence = f"{pred:.3f}"

    st.subheader("Prediction Result")
    if pred > 0.5:
        st.success(f"🌍 Possible Exoplanet Transit Detected! (Confidence: {confidence})")
    else:
        st.error(f"❌ No Exoplanet Transit Detected (Confidence: {confidence})")

    st.download_button(
        label="⬇️ Download Processed CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="processed_lightcurve.csv",
        mime="text/csv"
    )
else:
    st.info("👆 Upload a CSV file in the sidebar to begin.")











