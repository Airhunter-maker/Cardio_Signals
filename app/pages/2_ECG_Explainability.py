import streamlit as st
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SAL_PATH = ROOT_DIR / "results" / "plots" / "ecg_saliency_example.png"

st.title("🔍 ECG Explainability")

st.markdown("""
We use **saliency maps** to understand which ECG regions influence predictions.
""")

if SAL_PATH.exists():
    st.image(str(SAL_PATH), caption="ECG Saliency Map")
else:
    st.warning("Saliency image not found. Run explainability notebook.")