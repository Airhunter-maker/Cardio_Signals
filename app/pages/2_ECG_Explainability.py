import streamlit as st
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SAL_PATH = ROOT_DIR / "results" / "plots" / "ecg_saliency_example.png"

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
    st.markdown("#### ECG Explainability")
    st.caption("Inspect which regions of the ECG most influence model predictions.")
    st.markdown("---")

st.title("🔍 ECG Explainability")

st.markdown(
    """
    <div class="cs-card">
        <div class="cs-section-title">Why explainability matters</div>
        <div class="cs-subtle">
            We use <strong>saliency maps</strong> to understand which ECG regions influence predictions.
            This helps assess whether the model is focusing on clinically plausible segments of the trace.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("How to read the saliency map", expanded=True):
    st.markdown(
        """
        - **Bright / highlighted regions** indicate segments of the ECG that contribute most strongly to the risk score.  
        - **Dim regions** have lower influence on the model output.  
        - Use this view alongside domain expertise to judge whether attention is placed on physiologically meaningful areas.
        """,
        unsafe_allow_html=False,
    )

if SAL_PATH.exists():
    st.markdown(
        """
        <div class="cs-card">
            <div class="cs-section-title">Saliency visualisation</div>
            <div class="cs-subtle">
                The figure below highlights ECG segments that most strongly drive the model&apos;s prediction.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.image(str(SAL_PATH), caption="ECG Saliency Map", use_column_width=True)
else:
    st.warning("Saliency image not found. Run explainability notebook.")

st.info(
    "Explainability views are provided for transparency and research only and are not a substitute for expert ECG review."
)