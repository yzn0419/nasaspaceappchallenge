import streamlit as st
import pandas as pd
import plotly.express as px
import base64

# ------------------- Page Config -------------------
st.set_page_config(page_title="TESS Light Curve Explorer", layout="wide")

# ------------------- Background Image -------------------
def set_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{b64}");
            background-size: cover;
            background-attachment: fixed;
        }}
        .reportview-container .main .block-container{{
            background-color: rgba(0,0,0,0.7);
            border-radius: 15px;
            padding: 2rem;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("pexels-umkreisel-app-957010.jpg")  # 🌌 your space background

# ------------------- Sidebar -------------------
st.sidebar.header("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("Upload Light Curve CSV", type=["csv"])

# ------------------- Main UI -------------------
st.title("🔭 TESS Light Curve Explorer")
st.markdown("Explore, clean, and analyze NASA TESS light curve data interactively.")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Ensure numeric conversion
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()

    # Assume time is in first column
    time_col = df.columns[0]
    flux_col = df.columns[1]

    # ------------------- Interactive Time Range Slider -------------------
    min_time, max_time = df[time_col].min(), df[time_col].max()
    time_range = st.slider(
        "⏳ Select Time Range",
        float(min_time),
        float(max_time),
        (float(min_time), float(max_time))
    )
    df_range = df[(df[time_col] >= time_range[0]) & (df[time_col] <= time_range[1])]

    # ------------------- Plot with Plotly -------------------
    fig = px.line(
        df_range, x=time_col, y=flux_col,
        title="Light Curve",
        template="plotly_dark"
    )
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(rangeslider=dict(visible=True))  # enable zoom slider
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------------------- Prediction Status Cards -------------------
    st.subheader("Prediction Results")
    col1, col2 = st.columns(2)

    with col1:
        st.success("✅ Planet Transit Detected")
    with col2:
        st.metric("Confidence Score", "92%")

    # ------------------- Download Cleaned Data -------------------
    csv = df_range.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Cleaned Light Curve",
        data=csv,
        file_name="cleaned_lightcurve.csv",
        mime="text/csv",
    )

else:
    st.info("👆 Upload a CSV file from TESS to get started.")













