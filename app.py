import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import exoplanet  # reuse preprocessing helpers

# --------------------------
# Config
# --------------------------
st.set_page_config(
    page_title="🌌 Exoplanet Detector",
    page_icon="✨",
    layout="centered"
)

MODEL_PATH = "models/final_model.h5"
model = load_model(MODEL_PATH)

# --------------------------
# UI Layout
# --------------------------
st.title("🌍🔭 Exoplanet Transit Detector")
st.markdown(
    """
    Upload a **light curve CSV file** and this AI model will analyze it to predict 
    whether an **exoplanet transit** is present.  
    - Format: `time, flux` (optionally `label`)  
    - Example: [Download Sample Light Curve](https://raw.githubusercontent.com/username/repo/main/sample.csv)  
    """
)

uploaded_file = st.file_uploader("📂 Upload your light curve CSV", type=["csv"])

if uploaded_file is not None:
    with st.spinner("Processing light curve..."):
        # Load CSV
        df = pd.read_csv(uploaded_file)
        if "flux" not in df.columns:
            st.error("CSV must contain a 'flux' column.")
        else:
            flux = df["flux"].values.astype(np.float32)

            # Preprocess with exoplanet helpers
            flux = exoplanet.pad_or_trim(flux, 2048)
            X = flux[np.newaxis, :, np.newaxis]
            X = exoplanet.normalize(X)

            # Predict
            prediction = model.predict(X)[0][0]
            prob = float(prediction)
            label = "✅ Transit Likely (Exoplanet)" if prob > 0.5 else "❌ No Transit Detected"

            # Show results
            st.subheader("🔎 Prediction Result")
            st.metric(label=label, value=f"{prob*100:.2f}% confidence")

            # Show plot
            st.line_chart(df["flux"], height=300, use_container_width=True)

            st.success("Analysis complete!")

else:
    st.info("👆 Upload a CSV file to begin analysis.")




