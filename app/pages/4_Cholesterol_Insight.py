import streamlit as st

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
        <div class="cs-section-title">Physiological context</div>
        <div class="cs-subtle">
            High cholesterol contributes to ischemia and arterial stiffness.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("ECG correlates of dyslipidaemia", expanded=True):
    st.markdown(
        """
        ### ECG correlates:
        - ST depression  
        - T-wave abnormalities  
        - Reduced variability  
        """,
        unsafe_allow_html=False,
    )

st.markdown(
    """
    <div class="cs-card">
        <div class="cs-section-title">Model insight</div>
        <div class="cs-subtle">
            ECG-derived risk reflects these latent physiological effects.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)