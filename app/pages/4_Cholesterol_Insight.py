import streamlit as st

st.markdown(
    """
    <style>
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
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
        color: #f8fafc;
    }

    .cs-subtle {
        color: #94a3b8;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("#### Cholesterol & ECG")
    st.caption("How lipid profiles relate to ECG-derived risk patterns.")
    st.markdown("---")

st.title("🧪 Cholesterol & ECG")

st.markdown(
    """
    <div class="cs-card">
        <div class="cs-section-title">Physiological Context</div>
        <div class="cs-subtle">
            High cholesterol contributes to ischemia (reduced blood flow) and arterial stiffness, which can alter electrical conduction.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("ECG correlates of dyslipidaemia", expanded=True):
    st.markdown(
        """
        ### ECG Correlates:
        - **ST Depression**: Often a sign of ischemia.
        - **T-Wave Abnormalities**: Repolarization changes linked to metabolic stress.
        - **Reduced Variability**: Changes in heart rate variability patterns.
        """,
        unsafe_allow_html=False,
    )

st.markdown(
    """
    <div class="cs-card">
        <div class="cs-section-title">Model Insight</div>
        <div class="cs-subtle">
            ECG-derived risk reflects these latent physiological effects, aggregating them into a unified view of cardiovascular health.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)