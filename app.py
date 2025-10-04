import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from exoplanet import normalize, pad_or_trim  # reuse functions from exoplanet.py
import base64
import streamlit as st

# Force all text to white without changing background
st.markdown(
    """
    <style>
    .stApp, .stMarkdown, .stText, .stTitle, .stHeader, .stSubheader,
    .stCaption, .stDataFrame, div, span, p {
        color: white !important;
    }

  /* File uploader box */
    section[data-testid="st.file_uploader"] div[role="button"] {
        background-color: black !important;
        border: 2px dashed white !important;
        color: white !important;
    }
    input, textarea {
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
set_background("pexels-umkreisel-app-957010.jpg")

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




























