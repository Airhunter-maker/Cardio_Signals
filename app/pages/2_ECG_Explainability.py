import streamlit as st
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SAL_PATH = ROOT_DIR / "results" / "plots" / "ecg_saliency_example.png"

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
    
    .cs-info-box {
        background: rgba(56, 189, 248, 0.1);
        border-left: 4px solid #38bdf8;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #e2e8f0;
        margin-top: 1rem;
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
            <div class="cs-section-title">Saliency Visualisation</div>
            <div class="cs-subtle" style="margin-bottom: 1rem;">
                The figure below highlights ECG segments that most strongly drive the model&apos;s prediction.
            </div>
        """,
        unsafe_allow_html=True,
    )
    st.image(str(SAL_PATH), caption="ECG Saliency Map", use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.warning("Saliency image not found. Run explainability notebook.")

st.markdown(
    """
    <div class="cs-info-box">
        <strong>Note:</strong> Explainability views are provided for transparency and research only and are not a substitute for expert ECG review.
    </div>
    """,
    unsafe_allow_html=True
)