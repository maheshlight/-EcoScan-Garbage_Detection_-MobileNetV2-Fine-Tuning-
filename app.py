import streamlit as st
import numpy as np
import os
from PIL import Image

# ── Try importing tensorflow ──────────────────────────────────────────────────
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EcoScan — Garbage Detection",
    page_icon="♻️",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f172a; }
    h1, h2, h3, p, label, .stMarkdown { color: white !important; }
    .stButton > button {
        width: 100%; border-radius: 8px; font-weight: bold;
        font-size: 16px; background-color: #1e293b;
        color: white !important; border: 1px solid #334155;
    }
    .stButton > button:hover { background-color: #22c55e; color: white !important; }
    .result-box { border-radius: 12px; padding: 20px; margin-top: 20px; text-align: center; }
    .footer { text-align: center; color: #475569; font-size: 13px; margin-top: 30px;
               padding: 10px; border-top: 1px solid #1e293b; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASSES  = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMG_SIZE = (224, 224)

CLASS_INFO = {
    "cardboard": {"danger": "LOW",    "color": "#2ECC71",
                  "tip": "Flatten and put in paper recycling bin",
                  "category": "Biodegradable / Recyclable"},
    "glass":     {"danger": "MEDIUM", "color": "#F39C12",
                  "tip": "Rinse and drop in glass recycling bin",
                  "category": "Non-Biodegradable / Recyclable"},
    "metal":     {"danger": "MEDIUM", "color": "#F39C12",
                  "tip": "Clean and place in metal recycling bin",
                  "category": "Non-Biodegradable / Recyclable"},
    "paper":     {"danger": "LOW",    "color": "#2ECC71",
                  "tip": "Dry paper goes in paper recycling bin",
                  "category": "Biodegradable / Recyclable"},
    "plastic":   {"danger": "HIGH",   "color": "#E74C3C",
                  "tip": "Check recycling code, use plastic recycle bin",
                  "category": "Non-Biodegradable / Harmful"},
    "trash":     {"danger": "HIGH",   "color": "#C0392B",
                  "tip": "Dispose in general waste / landfill bin",
                  "category": "Non-Biodegradable / Mixed Waste"},
}

DANGER_BADGE = {
    "LOW":    ("🟢", "#2ECC71"),
    "MEDIUM": ("🟡", "#F39C12"),
    "HIGH":   ("🔴", "#E74C3C"),
}

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_ecoscan_model():
    if not TF_AVAILABLE:
        return None
    model_path = "ecoscan_best_model.h5"
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_ecoscan_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<h1 style='text-align:center;'>♻️ EcoScan</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#94a3b8; font-size:16px;'>AI-Powered Garbage Detection using MobileNetV2</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#64748b; font-size:13px;'>NSTI Mumbai-G &nbsp;|&nbsp; Teacher: Mala Mishra (Edunet Foundation)</p>", unsafe_allow_html=True)
st.markdown("---")

# ── Upload Image ──────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload a garbage image", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if not TF_AVAILABLE:
        st.warning("⚠️ TensorFlow is not installed. Model prediction is unavailable. Please add tensorflow to requirements.txt.")
    elif model is None:
        st.warning("⚠️ Model file `ecoscan_best_model.h5` not found. Please upload your trained model.")
    else:
        with st.spinner("🔍 Analyzing image..."):
            img_resized = image.resize(IMG_SIZE)
            arr = np.array(img_resized) / 255.0
            probs = model.predict(np.expand_dims(arr, 0), verbose=0)[0]
            idx = int(np.argmax(probs))
            label = CLASSES[idx]
            confidence = float(probs[idx]) * 100
            info = CLASS_INFO[label]
            emoji, color = DANGER_BADGE[info["danger"]]

        st.markdown(f"""
        <div class="result-box" style="border: 2px solid {color}; background-color: #1e293b;">
            <h2 style="color:{color};">🗑️ {label.upper()}</h2>
            <p style="color:white; font-size:18px;">Confidence: <b>{confidence:.1f}%</b></p>
            <p style="color:{color}; font-size:16px;">{emoji} Danger Level: <b>{info['danger']}</b></p>
            <p style="color:#94a3b8;">{info['category']}</p>
            <hr style="border-color:#334155;">
            <p style="color:#ccffcc;">💡 <b>How to Dispose:</b> {info['tip']}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 All Class Probabilities")
        for i, cls in enumerate(CLASSES):
            st.progress(float(probs[i]), text=f"{cls.capitalize()}: {probs[i]*100:.1f}%")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="footer">
    ♻️ EcoScan — Garbage Detection System<br>
    NSTI Mumbai-G &nbsp;|&nbsp; Faculty: Mala Mishra (Edunet Foundation)<br>
    Team Lead: Samir | Team: Ankit | Sushant | Mahesh | Pooja
</div>
""", unsafe_allow_html=True)
