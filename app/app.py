import streamlit as st
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="CardioSignals",
    layout="wide"
)

st.title("🫀 CardioSignals")
st.subheader("Inferring Cardiovascular Risk Directly from ECG Signals")

st.markdown("""
### What does this app demonstrate?
This system shows how **ECG signals alone** can be used to:
- Detect cardiovascular risk early
- Approximate clinical measurements (BP, Cholesterol)
- Predict downstream cardiac failure outcomes
""")

st.info("""
⚠️ Research & educational use only.  
Not a medical diagnostic tool.
""")

st.markdown("### Navigate using the sidebar to explore each component.")