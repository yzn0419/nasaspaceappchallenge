import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from exoplanet import normalize, pad_or_trim  # reuse functions from exoplanet.py
import base64
import streamlit as st

# Custom CSS for text + file uploader
st.markdown(
    """
    <style>
    /* Force all text to white */
    .stApp, .stMarkdown, .stHeader,
    .stCaption, div, span, p {
        color: black !important;
    }

    /* File uploader box styling */
    .stFileUploader > div {
        background-color: black !important;
        border: 2px dashed black !important;
        color: black !important;
    }

    /* File uploader caption text */
    .stFileUploader label {
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌌Exovision")
st.write("Upload a light curve file (CSV with a single flux column) to detect possible exoplanet transits.")
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
set_background("exovisionbackground.jpg")

MODEL_PATH = "models/final_model.h5"
model = load_model(MODEL_PATH)



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






































