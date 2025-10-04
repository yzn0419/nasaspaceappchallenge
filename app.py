import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from exoplanet import normalize, pad_or_trim  # reuse functions from exoplanet.py
import base64
import streamlit as st

# Robust CSS + JS to style the drag & drop uploader box black
st.markdown(
    """
    <style>
    /* Try many selectors so different Streamlit versions are covered */
    section[data-testid="stFileUploader"] div[role="button"],
    section[data-testid="stFileUploader"] div[role="button"] > div,
    div[data-testid="stFileUploader"] div[role="button"],
    .stFileUploader div[role="button"],
    .stFileUploader {
        background-color: rgba(0,0,0,0.95) !important;
        border: 2px dashed rgba(255,255,255,0.9) !important;
        color: #ffffff !important;
        padding: 18px !important;
        border-radius: 8px !important;
    }

    /* Uploader label / caption white */
    section[data-testid="stFileUploader"] label,
    div[data-testid="stFileUploader"] label,
    .stFileUploader label {
        color: #ffffff !important;
    }

    /* Make sure the "browse files" text inside the uploader is white */
    section[data-testid="stFileUploader"] span,
    .stFileUploader span,
    section[data-testid="stFileUploader"] p,
    .stFileUploader p {
        color: #ffffff !important;
    }
    </style>

    <script>
    // Retry loop: Streamlit renders asynchronously, so keep retrying until element exists
    (function applyUploaderStyle(retries=0){
        try {
            const sel = document.querySelector('section[data-testid="stFileUploader"]') ||
                        document.querySelector('div[data-testid="stFileUploader"]') ||
                        document.querySelector('.stFileUploader');
            if (!sel && retries < 20) {
                setTimeout(()=> applyUploaderStyle(retries+1), 150);
                return;
            }
            if (sel) {
                const btn = sel.querySelector('div[role="button"]') || sel.querySelector('div');
                if (btn) {
                    btn.style.backgroundColor = 'rgba(0,0,0,0.95)';
                    btn.style.border = '2px dashed rgba(255,255,255,0.9)';
                    btn.style.color = '#fff';
                    btn.style.padding = '18px';
                    btn.style.borderRadius = '8px';
                }
                const label = sel.querySelector('label');
                if (label) label.style.color = '#fff';
                // ensure inner text nodes are white
                sel.querySelectorAll('span, p, div').forEach(n => {
                    n.style.color = '#fff';
                });
            }
        } catch (e) {
            if (retries < 20) setTimeout(()=> applyUploaderStyle(retries+1), 150);
        }
    })();
    </script>
    """,
    unsafe_allow_html=True,
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





























