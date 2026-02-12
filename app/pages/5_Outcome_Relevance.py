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
    st.markdown("#### Outcome Relevance")
    st.caption("How ECG-derived risk compares against traditional clinical models.")
    st.markdown("---")

st.title("🧠 Clinical Outcome Relevance")

st.markdown(
    """
    <div class="cs-card">
        <div class="cs-section-title">Why Outcome Validation?</div>
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
            <div class="cs-section-title">Model Comparison</div>
            <div class="cs-subtle">
                Each bar represents a different model configuration. Higher ROC-AUC indicates better
                discrimination between patients who do and do not experience the outcome.
            </div>
        """,
        unsafe_allow_html=True,
    )

    container = st.container()
    with container:
        with plt.style.context('dark_background'):
            fig, ax = plt.subplots()
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            
            # Using custom colors for bars if possible, or standard logic
            bars = ax.bar(df["Model"], df["ROC_AUC"], color='#38bdf8', alpha=0.8)
            
            ax.set_ylabel("ROC-AUC", color='#e2e8f0')
            ax.set_xlabel("Model", color='#e2e8f0')
            ax.set_title("Outcome Prediction Performance", color='#f1f5f9')
            ax.tick_params(colors='#94a3b8')
            for spine in ax.spines.values():
                spine.set_color('#475569')
                
            st.pyplot(fig)
            
    st.markdown("</div>", unsafe_allow_html=True)

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
        <div class="cs-info-box">
            <strong>Takeaway:</strong> ECG-derived risk predicts real cardiac failure outcomes
            and complements traditional clinical features.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.warning("Outcome metrics not found.")