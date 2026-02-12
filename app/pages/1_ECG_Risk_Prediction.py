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
    .cs-page-shell {
        position: relative;
        padding-top: 0.4rem;
    }

    .cs-page-shell::before {
        content: "";
        position: absolute;
        inset: -10%;
        opacity: 0.12;
        background-image: radial-gradient(circle at top left, #22d3ee 0, transparent 55%);
        pointer-events: none;
    }

    .cs-card {
        background: radial-gradient(circle at 0 0, rgba(15, 23, 42, 0.9), rgba(15, 23, 42, 0.96));
        padding: 1.3rem 1.55rem;
        border-radius: 0.9rem;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(51, 65, 85, 0.9);
        margin-bottom: 1rem;
    }

    .cs-section-title {
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
        color: #e5e7eb;
    }

    .cs-subtle {
        color: #9ca3af;
        font-size: 0.9rem;
    }

    .cs-risk-box {
        padding: 0.75rem 1rem;
        border-radius: 0.75rem;
        border-left: 4px solid #38bdf8;
        background: linear-gradient(130deg, rgba(15, 23, 42, 0.95), rgba(8, 47, 73, 0.98));
        margin-top: 0.5rem;
        font-size: 0.9rem;
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

st.markdown('<div class="cs-page-shell">', unsafe_allow_html=True)

st.markdown(
    """
    <div class="cs-card">
        <div class="cs-section-title">Overview</div>
        <div class="cs-subtle">
            This page demonstrates <strong>early cardiovascular risk detection using ECG signals only</strong>.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="cs-card">
        <div class="cs-section-title">1. Upload ECG</div>
        <div class="cs-subtle">
            Accepted formats: PNG, JPG, JPEG. Use a single-lead or 12-lead ECG image export.
            The uploaded file is processed into a model-ready representation; the original image
            is shown only for visual confirmation.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded = st.file_uploader("Upload an ECG image", type=["png", "jpg", "jpeg"])

if not _MODEL_AVAILABLE:
    st.error(
        "The ECG risk model could not be loaded.\n\n"
        "Ensure that the `models` package and `ecg_model` definition are available in your environment. "
        "The rest of the dashboard remains usable for exploration."
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

        # Layout for metric and status
        metric_col, status_col = st.columns([1, 1])
        with metric_col:
            st.metric("Predicted Heart Risk Probability", f"{risk:.2f}")

        # Visual-only status based on risk value (no backend logic changes)
        if risk < 0.33:
            status_label = "Low ECG-derived risk pattern"
            status_color = "#22c55e"  # green
            bg_color = "rgba(22, 163, 74, 0.24)"
        elif risk < 0.66:
            status_label = "Moderate ECG-derived risk pattern"
            status_color = "#f97316"  # orange
            bg_color = "rgba(249, 115, 22, 0.24)"
        else:
            status_label = "High ECG-derived risk pattern"
            status_color = "#ef4444"  # red
            bg_color = "rgba(239, 68, 68, 0.24)"

        with status_col:
            st.markdown(
                f"""
                <div class="cs-risk-box" style="border-left-color: {status_color}; background-color: {bg_color};">
                    <strong style="color:{status_color};">Status:</strong><br/>
                    {status_label}<br/>
                    <span class="cs-subtle">
                        Interpretation is for research visualisation only and does not represent a clinical diagnosis.
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Centered preview of uploaded ECG image (visual only)
        prev_col1, prev_col2, prev_col3 = st.columns([1, 2, 1])
        with prev_col2:
            st.markdown(
                """
                <div class="cs-card">
                    <div class="cs-section-title">ECG preview</div>
                    <div class="cs-subtle">
                        Raw ECG image as uploaded (for visual confirmation only).
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.image(
                uploaded,
                caption="Uploaded ECG trace (visual check)",
                use_column_width=True,
            )

        # Processed ECG signal plot (existing logic, wrapped for layout only)
        st.markdown(
            """
            <div class="cs-card">
                <div class="cs-section-title">2. Processed ECG signal</div>
                <div class="cs-subtle">
                    The model operates on a processed representation of the ECG signal. The plot below
                    shows the transformed signal used downstream by the model.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        fig, ax = plt.subplots()
        ax.plot(ecg_signal)
        ax.set_title("Processed ECG Signal")
        st.pyplot(fig)

        # Interpretation text preserved, moved into an expander for readability
        with st.expander("3. Interpretation of the risk score", expanded=True):
            st.markdown(
                """
                ### Interpretation
                Higher scores indicate ECG patterns associated with increased cardiovascular risk.
                """,
                unsafe_allow_html=False,
            )

st.markdown("</div>", unsafe_allow_html=True)