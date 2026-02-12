import streamlit as st
import matplotlib.pyplot as plt
from utils.ecg_utils import load_ecg_image
from utils.model_utils import load_ecg_model, predict_ecg_risk

# Lightweight UI styling for this page (visual only)
st.markdown(
    """
    <style>
    .cs-card {
        background-color: #ffffff;
        padding: 1.25rem 1.5rem;
        border-radius: 0.85rem;
        box-shadow: 0 1px 8px rgba(15, 23, 42, 0.06);
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }

    .cs-section-title {
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
        color: #111827;
    }

    .cs-subtle {
        color: #6b7280;
        font-size: 0.9rem;
    }

    .cs-risk-box {
        padding: 0.75rem 1rem;
        border-radius: 0.75rem;
        border-left: 4px solid #9ca3af;
        background-color: #f9fafb;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("#### ECG Risk Prediction")
    st.caption("Upload an ECG trace to obtain a research-grade risk estimate.\n\nv1.0 Research Build")
    st.markdown("---")

st.title("📈 ECG-Based Heart Risk Prediction")

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
        status_color = "#16a34a"  # green
        bg_color = "rgba(22, 163, 74, 0.06)"
    elif risk < 0.66:
        status_label = "Moderate ECG-derived risk pattern"
        status_color = "#ea580c"  # orange
        bg_color = "rgba(234, 88, 12, 0.06)"
    else:
        status_label = "High ECG-derived risk pattern"
        status_color = "#b91c1c"  # red
        bg_color = "rgba(185, 28, 28, 0.06)"

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