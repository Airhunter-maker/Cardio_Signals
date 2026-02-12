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
    st.markdown("#### BP & ECG Insight")
    st.caption("Conceptual link between blood pressure and ECG-derived features.")
    st.markdown("---")

st.title("🩺 Blood Pressure & ECG")

st.markdown(
    """
    <div class="cs-card">
        <div class="cs-section-title">Physiological context</div>
        <div class="cs-subtle">
            Hypertension affects cardiac electrical activity.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("ECG correlates of high BP", expanded=True):
    st.markdown(
        """
        ### ECG correlates of high BP:
        - Increased QRS amplitude  
        - Ventricular hypertrophy patterns  
        - ST-segment changes  
        """,
        unsafe_allow_html=False,
    )

st.markdown(
    """
    <div class="cs-card">
        <div class="cs-section-title">Model insight</div>
        <div class="cs-subtle">
            Our ECG model captures these indirect patterns without measuring BP explicitly.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)