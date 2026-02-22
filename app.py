"""
app.py — CardioSignals Entry Point
Streamlit app: sidebar navigation, session state, CSS injection, routing.
"""

import os
import sys
import streamlit as st

# ── Repo root on path ───────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from components.ui_components import inject_css, status_chip, GLOBAL_CSS
from core.model_loader import load_all_models


# ── PAGE CONFIG ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioSignals",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    defaults = {
        "risk_score":            None,
        "shap_values":           None,
        "model_choice":          "Random Forest",
        "ecg_model_loaded":      False,
        "clinical_model_loaded": False,
        "errors":                [],
        "last_features":         None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── MAIN ────────────────────────────────────────────────────────────────
def main():
    init_session_state()
    inject_css()

    # Pre-load models (cached — runs once per session)
    with st.spinner("Initialising CardioSignals models..."):
        models = load_all_models()

    clinical_ok = st.session_state.get("clinical_model_loaded", False)
    ecg_ok      = st.session_state.get("ecg_model_loaded",      False)

    # ── SIDEBAR ─────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<div style="color:#00D4FF;font-size:1.4rem;font-weight:700;'
            'letter-spacing:-0.01em;margin-bottom:2px;">🫀 CardioSignals</div>',
            unsafe_allow_html=True)
        st.markdown(
            '<div style="color:#64748B;font-size:0.78rem;margin-bottom:1rem;">'
            'Cardiovascular Risk Intelligence</div>',
            unsafe_allow_html=True)
        st.markdown(
            '<hr style="border:none;border-top:1px solid #1E2A3A;margin:0.5rem 0;">',
            unsafe_allow_html=True)

        page = st.radio(
            "",
            ["🏠  Dashboard", "🔬  Risk Analyser", "📡  ECG Explorer"],
            label_visibility="collapsed",
        )

        st.markdown(
            '<hr style="border:none;border-top:1px solid #1E2A3A;margin:0.75rem 0;">',
            unsafe_allow_html=True)

        st.markdown(
            '<div style="font-size:0.72rem;color:#64748B;text-transform:uppercase;'
            'letter-spacing:0.08em;margin-bottom:6px;">Active Model</div>',
            unsafe_allow_html=True)
        model_choice = st.selectbox(
            "Active Model",
            ["Random Forest", "Logistic Regression"],
            key="model_choice",
            label_visibility="collapsed",
        )

        st.markdown(
            '<hr style="border:none;border-top:1px solid #1E2A3A;margin:0.75rem 0;">',
            unsafe_allow_html=True)

        st.markdown(
            '<div style="font-size:0.72rem;color:#64748B;text-transform:uppercase;'
            'letter-spacing:0.08em;margin-bottom:8px;">System Status</div>',
            unsafe_allow_html=True)
        status_chip("Clinical Model", "online" if clinical_ok else "error")
        status_chip("ECG CNN",        "online" if ecg_ok else "demo")
        status_chip("Dataset",        "online")

        st.markdown(
            '<hr style="border:none;border-top:1px solid #1E2A3A;margin:0.75rem 0;">',
            unsafe_allow_html=True)

        st.markdown("""
<div style="font-size:0.7rem;color:#64748B;line-height:1.8;">
  Hack4Health 2025<br>
  CardioSignals v1.0<br>
  <span style="font-size:0.65rem;">70,000 de-identified patient records</span>
</div>
""", unsafe_allow_html=True)

    # ── PAGE ROUTING ────────────────────────────────────────────────────
    if "Dashboard" in page:
        from views.dashboard import render
        render()
    elif "Risk Analyser" in page:
        from views.risk_analyser import render
        render()
    elif "ECG Explorer" in page:
        from views.ecg_explorer import render
        render()


if __name__ == "__main__":
    main()
