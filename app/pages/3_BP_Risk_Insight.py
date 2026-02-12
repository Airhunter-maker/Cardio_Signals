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
    
    .cs-list li {
        color: #cbd5e1;
        margin-bottom: 0.25rem;
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
        <div class="cs-section-title">Physiological Context</div>
        <div class="cs-subtle">
            Hypertension affects cardiac electrical activity in measurable ways, even before overt damage occurs.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("ECG correlates of high BP", expanded=True):
    st.markdown(
        """
        ### ECG Correlates of High BP:
        - **Increased QRS Amplitude**: Suggesting higher voltage from thicker muscle.
        - **Ventricular Hypertrophy Patterns**: Structural adaptation to pressure overload.
        - **ST-Segment Changes**: Indicating strain or potential ischemia.
        """,
        unsafe_allow_html=False,
    )

st.markdown(
    """
    <div class="cs-card">
        <div class="cs-section-title">Model Insight</div>
        <div class="cs-subtle">
            Our ECG model captures these indirect patterns without measuring BP explicitly, allowing it to factor hypertension-related risk into its overall cardiovascular assessment.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)