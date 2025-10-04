import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from exoplanet import normalize, pad_or_trim  # reuse functions from exoplanet.py
import base64

# Function to convert image to base64
def get_base64(bin_file):
    with open(bin_file, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Function to set custom background
def set_background(image_file):
    bin_str = get_base64(image_file)
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bin_str}");
        background-size: cover;
        color: white;
    }}
    h1, h2, h3, h4, h5, h6, p, div, span {{
        color: white !important;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set page background
set_background("Untitled design (5).jpg")

# Load model
MODEL_PATH = "models/final_model.h5"
model = load_model(MODEL_PATH)

# Title and description
st.title("🌌 Exovision")
st.write("Upload a light curve file (CSV with time + flux columns) to detect possible exoplanet transits.")

# ---------- CUSTOM STYLED Uploader (keeps st.file_uploader backend) ----------
st.markdown(
    """
    <style>
    /* Visual custom uploader box */
    #custom-uploader {
        background-color: #0b274a;
        border: 3px dashed #4fc3f7;
        border-radius: 12px;
        padding: 36px;
        text-align: center;
        color: white;
        font-family: "Poppins", sans-serif;
        margin-bottom: 12px;
        cursor: pointer;
        user-select: none;
    }
    #custom-uploader h3 { margin: 0 0 6px 0; }
    #custom-uploader p { margin: 0; color: white; font-size: 14px; }
    /* Small visual highlight on dragover */
    #custom-uploader.dragover {
        background-color: #133b73;
        transform: scale(1.01);
    }
    /* Hide the default uploader visuals (we keep the input element in DOM) */
    div[data-testid="stFileUploaderDropzone"] {
        opacity: 0;
        height: 1px;  /* keep it in DOM but visually tiny */
        overflow: hidden;
        position: relative;
    }
    </style>

    <div id="custom-uploader" aria-hidden="true">
      <h3>📤 Drag & Drop CSV here</h3>
      <p>or click to browse — (CSV with time + flux columns)</p>
    </div>

    <script>
    (function() {
        console.log("[custom-uploader] init script running");

        // Attempts to find the Streamlit file input element that accepts .csv
        function findStreamlitFileInput() {
            const inputs = Array.from(document.querySelectorAll('input[type="file"]'));
            if (!inputs.length) return null;
            // prefer the input that accepts .csv (some Streamlit inputs might accept other types)
            for (const inp of inputs) {
                try {
                    const accept = inp.getAttribute('accept') || "";
                    if (accept.includes('.csv') || accept.includes('text/csv')) return inp;
                } catch(e){}
            }
            // fallback: return first file input
            return inputs[0] || null;
        }

        function attachForwarding() {
            const box = document.getElementById('custom-uploader');
            if (!box) { console.log("[custom-uploader] box not found"); return false; }

            const input = findStreamlitFileInput();
            if (!input) { console.log("[custom-uploader] file input not found yet"); return false; }

            console.log("[custom-uploader] found file input:", input);

            // Make sure input is not display:none; keep it tiny and invisible (we already set CSS)
            input.style.opacity = 0;
            input.style.position = 'relative';
            input.style.zIndex = 1;

            // Click forwards to native input
            box.addEventListener('click', function() {
                console.log("[custom-uploader] box clicked -> opening native file dialog");
                try { input.click(); } catch (e) { console.warn("[custom-uploader] input.click() failed", e); }
            });

            // Drag events: visual feedback
            box.addEventListener('dragover', function(e) {
                e.preventDefault();
                e.stopPropagation();
                box.classList.add('dragover');
            });
            box.addEventListener('dragleave', function(e) {
                e.preventDefault();
                e.stopPropagation();
                box.classList.remove('dragover');
            });

            // Drop forwarding: set the native input.files via DataTransfer, then dispatch change
            box.addEventListener('drop', function(e) {
                e.preventDefault();
                e.stopPropagation();
                box.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (!files || files.length === 0) {
                    console.log("[custom-uploader] no files dropped");
                    return;
                }
                try {
                    const dt = new DataTransfer();
                    for (let i = 0; i < files.length; i++) {
                        dt.items.add(files[i]);
                    }
                    input.files = dt.files;

                    // dispatch change event on input so Streamlit picks it up
                    const evt = new Event('change', { bubbles: true });
                    input.dispatchEvent(evt);
                    console.log("[custom-uploader] forwarded drop to native input (files set, change dispatched)");
                } catch (err) {
                    console.warn("[custom-uploader] error forwarding dropped files:", err);
                }
            });

            // Also forward native input changes to give visual feedback in console
            input.addEventListener('change', () => {
                console.log("[custom-uploader] native input change event fired, files:", input.files);
            });

            return true;
        }

        // Retry loop (Streamlit renders components asynchronously)
        let tries = 0;
        const maxTries = 40;
        const interval = setInterval(() => {
            tries++;
            const ok = attachForwarding();
            if (ok || tries > maxTries) {
                clearInterval(interval);
                if (!ok) console.warn("[custom-uploader] Giving up after", tries, "attempts");
            }
        }, 250);

        // Also observe DOM mutations to re-run attach when new nodes appear
        const observer = new MutationObserver((mutations) => {
            // tiny debounce
            setTimeout(() => attachForwarding(), 120);
        });
        observer.observe(document.body, { childList: true, subtree: true });

    })();
    </script>
    """,
    unsafe_allow_html=True
)

# now call the real uploader (kept for Streamlit backend). It will be hidden visually but present in DOM:
uploaded_file = st.file_uploader("Upload light curve CSV", type=["csv"])
# -------------------------------------------------------------------------


# --- File processing logic ---
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Handle time + flux columns
    if "time" in df.columns and "flux" in df.columns:
        time = df["time"].values
        flux = df["flux"].values
    else:
        time = np.arange(len(df))
        flux = df.iloc[:, 0].values

    flux_norm = flux / np.nanmedian(flux)

    # Time window slider
    min_time, max_time = float(np.nanmin(time)), float(np.nanmax(time))
    start, end = st.slider(
        "Select time window (days)",
        min_value=int(min_time),
        max_value=int(max_time),
        value=(int(min_time), int(min_time) + 10),
        step=1
    )

    # Auto-zoom button
    auto_zoom = st.button("🔎 Auto Zoom to Deepest Transit")
    if auto_zoom:
        dip_index = np.nanargmin(flux_norm)
        dip_time = time[dip_index]
        start, end = int(dip_time) - 5, int(dip_time) + 5
        if start < min_time:
            start = int(min_time)
        if end > max_time:
            end = int(max_time)

    # Apply time mask
    mask = (time >= start) & (time <= end)
    time_zoom = time[mask]
    flux_zoom = flux_norm[mask]

    # Plot light curve
    st.subheader("Light Curve (Zoomable)")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(time_zoom, flux_zoom, s=2, color="blue")
    ax.set_xlabel("Time (BJD - 2457000)")
    ax.set_ylabel("Normalized Flux")
    ax.set_title(f"Light Curve (Time {start} - {end} days)")
    st.pyplot(fig)

    # Prediction
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













































































