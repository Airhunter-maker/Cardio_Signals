import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT_DIR / "results" / "metrics" / "outcome_validation_results.csv"

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
    st.markdown("#### Outcome Relevance")
    st.caption("How ECG-derived risk compares against traditional clinical models.")
    st.markdown("---")

st.title("🧠 Clinical Outcome Relevance")

st.markdown(
    """
    <div class="cs-card">
        <div class="cs-section-title">Why outcome validation?</div>
        <div class="cs-subtle">
            This page compares <strong>ROC-AUC</strong> performance across models, showing how ECG-derived
            risk relates to prediction of real cardiac failure outcomes.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if CSV_PATH.exists():
    df = pd.read_csv(CSV_PATH)

    st.markdown(
        """
        <div class="cs-card">
            <div class="cs-section-title">Model comparison</div>
            <div class="cs-subtle">
                Each bar represents a different model configuration. Higher ROC-AUC indicates better
                discrimination between patients who do and do not experience the outcome.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    container = st.container()
    with container:
        fig, ax = plt.subplots()
        ax.bar(df["Model"], df["ROC_AUC"])
        ax.set_ylabel("ROC-AUC")
        ax.set_xlabel("Model")
        ax.set_title("Outcome prediction performance")
        st.pyplot(fig)

    with st.expander("How to interpret this chart", expanded=True):
        st.markdown(
            """
            - **Higher bars** indicate stronger ability to separate outcome vs non-outcome cases.  
            - **ECG-enhanced models** can complement traditional clinical baselines.  
            - ROC-AUC values here are fixed outputs from offline evaluation and are not recomputed in the app.
            """,
            unsafe_allow_html=False,
        )

    st.markdown(
        """
        <div class="cs-card">
            <div class="cs-section-title">Takeaway</div>
            <div class="cs-subtle">
                ECG-derived risk predicts real cardiac failure outcomes
                and complements traditional clinical features.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.warning("Outcome metrics not found.")