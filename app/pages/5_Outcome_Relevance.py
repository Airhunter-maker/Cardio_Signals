import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT_DIR / "results" / "metrics" / "outcome_validation_results.csv"

st.title("🧠 Clinical Outcome Relevance")

if CSV_PATH.exists():
    df = pd.read_csv(CSV_PATH)

    fig, ax = plt.subplots()
    ax.bar(df["Model"], df["ROC_AUC"])
    ax.set_ylabel("ROC-AUC")
    st.pyplot(fig)

    st.success("""
ECG-derived risk predicts real cardiac failure outcomes
and complements traditional clinical features.
""")
else:
    st.warning("Outcome metrics not found.")