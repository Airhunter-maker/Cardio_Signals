import streamlit as st
import matplotlib.pyplot as plt

from utils.ecg_utils import load_ecg_image

try:
    # Keep backend import paths unchanged; handle missing models gracefully
    from utils.model_utils import load_ecg_model, predict_ecg_risk
    _MODEL_AVAILABLE = True
    _MODEL_IMPORT_ERROR = None
except Exception as e:  # ImportError / ModuleNotFoundError etc.
    _MODEL_AVAILABLE = False
    _MODEL_IMPORT_ERROR = e

# Neo-clinical visual style for this page (UI only)
st.markdown(
    """
    <style>
    /* Inherit global theme from app.py, plus page-specific tweaks */
    
    .cs-card {
        background: rgba(30, 41, 59, 0.4);
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(148, 163, 184, 0.1);
        backdrop-filter: blur(8px);
        margin-bottom: 1.5rem;
    }

    .cs-section-title {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #f8fafc;
    }

    .cs-subtle {
        color: #94a3b8;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    .cs-risk-box {
        padding: 1rem 1.5rem;
        border-radius: 0.75rem;
        border-left: 4px solid #38bdf8;
        background: linear-gradient(130deg, rgba(15, 23, 42, 0.6), rgba(8, 47, 73, 0.4));
        margin-top: 0.5rem;
        font-size: 0.95rem;
        color: #e2e8f0;
    }
    
    /* File uploader styling override */
    [data-testid='stFileUploader'] {
        width: 100%;
        max-width: 600px;
        margin: 0 auto;
    }
    
    .cs-alert-error {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("#### ECG Risk Prediction")
    st.caption(
        "React-style card layout for ECG-based risk.\n\n"
        "v1.0 Research Build"
    )
    st.markdown("---")

st.title("📈 ECG-Based Heart Risk Prediction")

# Overview Section
st.markdown(
    """
    <div class="cs-card">
        <div class="cs-section-title">Overview</div>
        <div class="cs-subtle">
            This page demonstrates <strong>early cardiovascular risk detection using ECG signals only</strong>.
            The model analyzes the raw signal waveform to estimate the probability of future cardiac events.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Upload Section
st.markdown(
    """
    <div class="cs-card">
        <div class="cs-section-title">1. Upload ECG</div>
        <div class="cs-subtle">
            Accepted formats: <strong>PNG, JPG, JPEG</strong>. Use a single-lead or 12-lead ECG image export.
            <br/>The uploaded file is processed into a model-ready representation; the original image
            is shown only for visual confirmation.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded = st.file_uploader("Upload an ECG image", type=["png", "jpg", "jpeg"])

if not _MODEL_AVAILABLE:
    st.markdown(
        """
        <div class="cs-alert-error">
            <strong>Model Status: Unavailable</strong><br/>
            The ECG risk model could not be loaded. Ensure that the <code>models</code> package and <code>ecg_model</code> definition are available in your environment.
            The rest of the dashboard remains usable for exploration.
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    model = load_ecg_model()

    if uploaded:
        # Core processing & prediction logic (unchanged)
        ecg_signal = load_ecg_image(uploaded)
        risk = predict_ecg_risk(model, ecg_signal)

        st.success("ECG uploaded successfully. Risk probability has been estimated using the research model.")

        # Visual-only threshold slider (does not affect prediction logic)
        vis_threshold = st.slider(
            "Visual risk threshold (for interpretation only)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help="Adjust to explore how different cut-offs might be interpreted. "
                 "This slider does not change the model output.",
        )

        st.markdown("<br/>", unsafe_allow_html=True)

        # Layout for metric and status
        # Using a styled container for the results
        st.markdown('<div class="cs-card">', unsafe_allow_html=True)
        
        metric_col, status_col = st.columns([1, 1])
        with metric_col:
            st.metric("Predicted Heart Risk Probability", f"{risk:.1%}")

        # Visual-only status based on risk value (no backend logic changes)
        if risk < 0.33:
            status_label = "Low ECG-derived risk pattern"
            status_color = "#22c55e"  # green
            bg_color = "rgba(34, 197, 94, 0.15)"
            border_color = "#22c55e"
        elif risk < 0.66:
            status_label = "Moderate ECG-derived risk pattern"
            status_color = "#f97316"  # orange
            bg_color = "rgba(249, 115, 22, 0.15)"
            border_color = "#f97316"
        else:
            status_label = "High ECG-derived risk pattern"
            status_color = "#ef4444"  # red
            bg_color = "rgba(239, 68, 68, 0.15)"
            border_color = "#ef4444"

        with status_col:
            st.markdown(
                f"""
                <div class="cs-risk-box" style="border-left-color: {border_color}; background-color: {bg_color};">
                    <strong style="color:{status_color}; text-transform: uppercase; font-size: 0.8rem; letter-spacing: 0.05em;">Status:</strong><br/>
                    <span style="font-weight: 600; font-size: 1.05rem; color: #f1f5f9;">{status_label}</span><br/>
                    <div style="margin-top: 0.4rem; font-size: 0.85rem; opacity: 0.8;">
                        Interpretation is for research visualisation only and does not represent a clinical diagnosis.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # Centered preview of uploaded ECG image (visual only)
        prev_col1, prev_col2, prev_col3 = st.columns([1, 4, 1])
        with prev_col2:
            st.markdown(
                """
                <div class="cs-card" style="text-align: center;">
                    <div class="cs-section-title">ECG Preview</div>
                    <div class="cs-subtle" style="margin-bottom: 1rem;">
                        Raw ECG image as uploaded (for visual confirmation only).
                    </div>
                """,
                unsafe_allow_html=True,
            )
            st.image(
                uploaded,
                caption="Uploaded ECG trace (visual check)",
                use_column_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # Processed ECG signal plot (existing logic, wrapped for layout only)
        st.markdown(
            """
            <div class="cs-card">
                <div class="cs-section-title">2. Processed ECG Signal</div>
                <div class="cs-subtle">
                    The model operates on a processed representation of the ECG signal. The plot below
                    shows the transformed signal used downstream by the model.
                </div>
            """,
            unsafe_allow_html=True,
        )

        with plt.style.context('dark_background'):
            fig, ax = plt.subplots()
            fig.patch.set_alpha(0.0) # Transparent figure background
            ax.patch.set_alpha(0.0)  # Transparent axes background
            ax.plot(ecg_signal, color='#38bdf8', linewidth=1.5)
            ax.set_title("Processed ECG Signal", color='#e2e8f0')
            ax.tick_params(colors='#94a3b8')
            for spine in ax.spines.values():
                spine.set_color('#475569')
            st.pyplot(fig)
        
        st.markdown("</div>", unsafe_allow_html=True)

        # Interpretation text preserved, moved into an expander for readability
        with st.expander("3. Interpretation of the risk score", expanded=True):
            st.markdown(
                """
                ### Interpretation
                Higher scores indicate ECG patterns associated with increased cardiovascular risk.
                """,
                unsafe_allow_html=False,
            )