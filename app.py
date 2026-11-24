"""
app.py - Plant Disease Detection Web Application
A modern, professional Streamlit interface for real-time agricultural disease diagnosis.
"""

import os
import json
from PIL import Image
import pandas as pd
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="AgriShield AI | Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "plant_model.keras")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
CLASSES_PATH = os.path.join(MODELS_DIR, "classes.json")
HISTORY_PLOT_PATH = os.path.join(MODELS_DIR, "training_history.png")
CONFUSION_MATRIX_PATH = os.path.join(MODELS_DIR, "confusion_matrix.png")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")

# Import inference engine
from predict import predict_disease, DISEASE_INFO, load_class_names

# Custom CSS for Modern Green Agricultural Aesthetic
st.markdown("""
<style>
    /* Theme color variables */
    :root {
        --primary-green: #2d6a4f;
        --dark-green: #1b4332;
        --light-green: #d8f3dc;
        --accent-green: #52b788;
        --bg-subtle: #f7faf7;
        --card-border: #e2e8f0;
    }

    /* Top banner header */
    .app-header {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 50%, #40916c 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(45, 106, 79, 0.25);
    }
    .app-header h1 {
        color: #ffffff !important;
        font-size: 2.3rem !important;
        font-weight: 800 !important;
        margin: 0 !important;
        letter-spacing: -0.5px;
    }
    .app-header p {
        color: #d8f3dc !important;
        font-size: 1.05rem !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0 !important;
    }

    /* Diagnosis status cards */
    .status-card-healthy {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border: 2px solid #4caf50;
        border-radius: 14px;
        padding: 1.5rem;
        color: #1b5e20;
        margin-bottom: 1.2rem;
    }
    .status-card-diseased {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border: 2px solid #ff9800;
        border-radius: 14px;
        padding: 1.5rem;
        color: #e65100;
        margin-bottom: 1.2rem;
    }
    .badge-tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-healthy {
        background-color: #2e7d32;
        color: white;
    }
    .badge-diseased {
        background-color: #c62828;
        color: white;
    }

    /* Stat KPI Box */
    .kpi-container {
        display: flex;
        gap: 1rem;
        margin: 1.5rem 0;
    }
    .kpi-card {
        flex: 1;
        background: #ffffff;
        border: 1px solid #e0ebd9;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #2d6a4f;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* Disease info card */
    .info-section-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem;
        margin-top: 1rem;
    }

    /* Footer styling */
    .app-footer {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        padding: 2.5rem 0 1rem 0;
        border-top: 1px solid #e2e8f0;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("## 🌿 AgriShield AI")
    st.caption("College Assignment Project | Machine Learning")
    st.markdown("---")
    
    app_mode = st.radio(
        "Navigation Menu",
        [
            "🔍 Leaf Disease Diagnosis",
            "📊 Model Performance & Metrics",
            "📚 Disease Knowledge Library",
            "ℹ️ About Project & Architecture"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    
    model_exists = os.path.exists(MODEL_PATH)
    if model_exists:
        st.success("● CNN Model: Active & Loaded", icon="✅")
    else:
        st.warning("● CNN Model: Not Found (Run train.py)", icon="⚠️")
        
    st.info("""
    **Crop Coverage:**
    - 🍅 Tomato
    - 🥔 Potato
    - 🫑 Bell Pepper
    - 🌽 Maize / Corn
    
    **Architecture:**
    - Deep ConvNet (4 Blocks)
    - Keras 3 (PyTorch Engine)
    """)
    st.markdown("---")
    st.caption("Made with Streamlit & TensorFlow/Keras")

# ----------------- MAIN HEADER -----------------
st.markdown("""
<div class="app-header">
    <h1>🌿 Plant Disease Detection System</h1>
    <p>Automated Deep Learning Leaf Pathology Classification & Treatment Recommendation Engine</p>
</div>
""", unsafe_allow_html=True)

# ----------------- TAB 1: DIAGNOSIS -----------------
if app_mode == "🔍 Leaf Disease Diagnosis":
    st.markdown("### 📸 Upload or Select Leaf Image")
    st.write("Upload a clear photo of an infected or healthy plant leaf to identify the condition and view recommended agricultural remedies.")

    # Two input methods: Upload or Preset Samples
    col_upload, col_sample = st.columns([2, 1])
    
    with col_upload:
        uploaded_file = st.file_uploader(
            "Choose a leaf photo (JPG, JPEG, or PNG)",
            type=["jpg", "jpeg", "png"],
            help="High-contrast leaf images on neutral backgrounds produce the most reliable predictions."
        )

    # Check for sample test images
    sample_images = []
    if os.path.exists(SAMPLES_DIR):
        sample_images = [f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
    selected_sample = None
    with col_sample:
        st.markdown("**Or test with a sample image:**")
        if sample_images:
            sample_choice = st.selectbox(
                "Select a benchmark test leaf:",
                ["-- None --"] + sample_images,
                format_func=lambda x: x.replace("sample_", "").replace(".jpg", "").replace("___", " - ") if x != "-- None --" else "-- None --"
            )
            if sample_choice != "-- None --":
                selected_sample = os.path.join(SAMPLES_DIR, sample_choice)
        else:
            st.info("No sample images found in samples/ directory.")

    # Determine which image source to use
    active_image = None
    source_name = ""
    if uploaded_file is not None:
        active_image = uploaded_file
        source_name = uploaded_file.name
    elif selected_sample is not None:
        active_image = selected_sample
        source_name = os.path.basename(selected_sample)

    if active_image is not None:
        st.markdown("---")
        col_img, col_diag = st.columns([1, 1.3], gap="large")

        # Left Column: Image Preview
        with col_img:
            st.markdown("#### 🖼️ Leaf Specimen")
            img_display = Image.open(active_image)
            st.image(img_display, caption=f"Specimen: {source_name}", use_container_width=True)
            
            # Image attributes
            width, height = img_display.size
            st.markdown(f"""
            <div style="background:#f1f5f9; padding: 0.8rem; border-radius: 8px; font-size: 0.85rem; color:#475569;">
                <b>Resolution:</b> {width} × {height} px &nbsp;|&nbsp; 
                <b>Format:</b> {img_display.format or 'JPEG'} &nbsp;|&nbsp; 
                <b>Channels:</b> {img_display.mode}
            </div>
            """, unsafe_allow_html=True)

        # Right Column: AI Diagnosis
        with col_diag:
            st.markdown("#### 🔬 AI Pathology Analysis")
            
            if not os.path.exists(MODEL_PATH):
                st.error("⚠️ Trained model not found! Please run `py train.py` first to generate `models/plant_model.keras`.")
            else:
                with st.spinner("Analyzing leaf pathology with CNN model..."):
                    try:
                        result = predict_disease(active_image)
                    except Exception as e:
                        st.error(f"Inference error: {e}")
                        result = None

                if result:
                    # Status header card
                    if result["is_healthy"]:
                        st.markdown(f"""
                        <div class="status-card-healthy">
                            <span class="badge-tag badge-healthy">✓ HEALTHY SPECIMEN</span>
                            <h2 style="margin: 0.5rem 0 0.2rem 0; font-size: 1.8rem; color: #1b5e20;">
                                {result['display_name']}
                            </h2>
                            <p style="margin:0; font-size: 0.95rem;">
                                No pathological lesions or active biotic infections detected.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="status-card-diseased">
                            <span class="badge-tag badge-diseased">⚠️ DISEASE DETECTED</span>
                            <h2 style="margin: 0.5rem 0 0.2rem 0; font-size: 1.8rem; color: #b71c1c;">
                                {result['display_name']}
                            </h2>
                            <p style="margin:0; font-size: 0.95rem;">
                                Active infection identified. Prompt agricultural treatment recommended.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Diagnostic Confidence Meter
                    conf = result["confidence"]
                    st.markdown(f"**Diagnostic Confidence:** `{conf:.2f}%`")
                    st.progress(conf / 100.0)

                    # Crop & Pathogen Metrics
                    m_col1, m_col2, m_col3 = st.columns(3)
                    with m_col1:
                        st.metric("Target Crop", result["crop"].split("(")[0].strip())
                    with m_col2:
                        st.metric("Pathogen Type", result["pathogen"].split("(")[0].strip())
                    with m_col3:
                        st.metric("Severity", result["severity"])

                    # Tabs for Actionable Agricultural Advice
                    diag_tab1, diag_tab2, diag_tab3 = st.tabs([
                        "📋 Symptoms & Identification",
                        "💊 Curative Treatment",
                        "🛡️ Prevention Protocols"
                    ])

                    with diag_tab1:
                        st.markdown(f"**Identified Symptoms:**")
                        st.write(result["symptoms"])

                    with diag_tab2:
                        st.markdown(f"**Recommended Action Plan:**")
                        treatment_lines = result["treatment"].split("\n")
                        for line in treatment_lines:
                            if line.strip():
                                st.markdown(f"- {line.strip()}")

                    with diag_tab3:
                        st.markdown(f"**Preventative Cultural Practices:**")
                        st.write(result["prevention"])

                    # Top-3 Alternative Probability Distribution
                    with st.expander("📊 View Probability Distribution (Top Classes)", expanded=False):
                        top_preds = result["top_predictions"]
                        df_preds = pd.DataFrame(top_preds)
                        st.dataframe(
                            df_preds[["display_name", "confidence"]].rename(
                                columns={"display_name": "Class Name", "confidence": "Confidence (%)"}
                            ),
                            use_container_width=True,
                            hide_index=True
                        )
                        st.bar_chart(df_preds.set_index("display_name")["confidence"])

    else:
        st.info("👆 Please upload an image or choose a benchmark sample from the selector above to start.")

# ----------------- TAB 2: MODEL PERFORMANCE -----------------
elif app_mode == "📊 Model Performance & Metrics":
    st.markdown("### 📊 Model Evaluation & Validation Performance")
    st.write("Real-time benchmark evaluation calculated on an independent, unseen test dataset partition.")

    metrics_data = None
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

    if metrics_data:
        # KPI Cards
        test_acc = metrics_data.get("test_accuracy", 0.0) * 100.0
        test_loss = metrics_data.get("test_loss", 0.0)
        epochs_cnt = metrics_data.get("epochs_trained", 12)
        classes_cnt = metrics_data.get("num_classes", 10)

        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        with kpi_col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{test_acc:.2f}%</div>
                <div class="kpi-label">Test Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_col2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{test_loss:.4f}</div>
                <div class="kpi-label">Test Loss</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_col3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{epochs_cnt}</div>
                <div class="kpi-label">Epochs Completed</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_col4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{classes_cnt}</div>
                <div class="kpi-label">Total Classes</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Charts Section
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("#### 📈 Learning Curves (Accuracy & Loss)")
            if os.path.exists(HISTORY_PLOT_PATH):
                st.image(HISTORY_PLOT_PATH, use_container_width=True)
            else:
                st.info("Training history plot not found.")

        with chart_col2:
            st.markdown("#### 🎯 Test Dataset Confusion Matrix")
            if os.path.exists(CONFUSION_MATRIX_PATH):
                st.image(CONFUSION_MATRIX_PATH, use_container_width=True)
            else:
                st.info("Confusion matrix plot not found.")

        # Classification Report Table
        st.markdown("#### 📋 Detailed Per-Class Classification Report")
        clf_report = metrics_data.get("classification_report", {})
        if clf_report:
            rows = []
            for k, v in clf_report.items():
                if isinstance(v, dict):
                    clean_k = k.replace("___", " - ").replace("_", " ")
                    rows.append({
                        "Class": clean_k,
                        "Precision": round(v.get("precision", 0) * 100, 2),
                        "Recall": round(v.get("recall", 0) * 100, 2),
                        "F1-Score": round(v.get("f1-score", 0) * 100, 2),
                        "Test Support": int(v.get("support", 0))
                    })
            if rows:
                df_rep = pd.DataFrame(rows)
                st.dataframe(df_rep, use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ Metrics file `models/metrics.json` not found. Please train the model by running `py train.py`.")

# ----------------- TAB 3: DISEASE LIBRARY -----------------
elif app_mode == "📚 Disease Knowledge Library":
    st.markdown("### 📚 Plant Pathology & Disease Knowledge Base")
    st.write("Comprehensive scientific profiles and treatment guidelines for all diagnosed crop conditions.")

    for class_key, info in DISEASE_INFO.items():
        with st.expander(f"🌿 {info['display_name']} ({info['crop']})", expanded=False):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(f"**Status:** `{info['status'].upper()}`")
                st.markdown(f"**Pathogen:** `{info['pathogen']}`")
                st.markdown(f"**Severity:** `{info['severity']}`")
            with c2:
                st.markdown("**Diagnostic Symptoms:**")
                st.write(info["symptoms"])
                st.markdown("**Remediation & Treatment:**")
                st.write(info["treatment"])
                st.markdown("**Prevention:**")
                st.write(info["prevention"])

# ----------------- TAB 4: ABOUT PROJECT -----------------
elif app_mode == "ℹ️ About Project & Architecture":
    st.markdown("### ℹ️ About This Project")
    st.markdown("""
    **Project Title:** Plant Disease Detection Using Machine Learning  
    **Course:** Machine Learning / Artificial Intelligence Capstone Project  
    **Objective:** Build an end-to-end computer vision system to detect crop leaf diseases from digital imagery and provide actionable agronomic guidance.
    """)

    st.markdown("---")
    st.markdown("#### 🏗️ Technical Architecture")
    
    st.markdown("""
    ```
    ┌───────────────────────┐
    │ Input Leaf Image      │ (JPG, JPEG, PNG)
    └──────────┬────────────┘
               ▼
    ┌───────────────────────┐
    │ Preprocessing Pipeline│ Resize to 128x128, RGB Normalization (1/255)
    └──────────┬────────────┘
               ▼
    ┌───────────────────────┐
    │ Data Augmentation     │ Random Flip, 10% Rotation, 10% Zoom
    └──────────┬────────────┘
               ▼
    ┌───────────────────────┐
    │ Convolutional Blocks  │ 4x [Conv2D (3x3) + BatchNorm + MaxPool (2x2)]
    │                       │ Filters: 32 -> 64 -> 128 -> 128
    └──────────┬────────────┘
               ▼
    ┌───────────────────────┐
    │ Classification Head   │ Flatten -> Dense(128, ReLU) -> Dropout(0.4)
    └──────────┬────────────┘
               ▼
    ┌───────────────────────┐
    │ Output Layer          │ Dense(10, Softmax) -> Disease & Confidence %
    └──────────┬────────────┘
               ▼
    ┌───────────────────────┐
    │ Knowledge Engine      │ Agronomic Treatment, Symptoms & Remediation
    └───────────────────────┘
    ```
    """)

    st.markdown("#### 📦 Technology Stack")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown("""
        **Core ML Frameworks**
        - Python 3.13
        - Keras 3 (Deep Learning)
        - PyTorch Tensor Engine
        """)
    with t2:
        st.markdown("""
        **Data & Visualization**
        - NumPy & Pandas
        - Matplotlib & Seaborn
        - Scikit-Learn
        """)
    with t3:
        st.markdown("""
        **Frontend & Deployment**
        - Streamlit Web App
        - Pillow (PIL)
        - High-performance Async Downloads
        """)

    st.markdown("#### 🌐 Dataset Information")
    st.markdown("""
    - **Dataset Benchmark:** PlantVillage (Public Open Research Dataset curated by Penn State & EPFL).
    - **Sample Diversity:** 10 discrete classes across 4 staple agricultural crops (Tomato, Potato, Bell Pepper, and Maize/Corn).
    - **Split Strategy:** 70% Training, 15% Validation, 15% Unseen Test Set.
    """)

# ----------------- FOOTER -----------------
st.markdown("""
<div class="app-footer">
    Plant Disease Detection Using Machine Learning • College Assignment Project • Built with Streamlit & Keras
</div>
""", unsafe_allow_html=True)
