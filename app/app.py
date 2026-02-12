import streamlit as st
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="CardioSignals – Cardiovascular Risk from ECG",
    page_icon="🫀",
    layout="wide"
)

# Global clinical-style theming (UI only)
st.markdown(
    """
    <style>
    /* Global page background */
    .main {
        background-color: #f5f7fb;
    }

    /* Generic card container */
    .cs-card {
        background-color: #ffffff;
        padding: 1.5rem 1.8rem;
        border-radius: 0.85rem;
        box-shadow: 0 2px 12px rgba(15, 23, 42, 0.06);
        border: 1px solid #e5e7eb;
        margin-bottom: 1.25rem;
    }

    .cs-card-header {
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 0.35rem;
        color: #0f172a;
    }

    .cs-subtle {
        color: #6b7280;
        font-size: 0.9rem;
    }

    .cs-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.20rem 0.60rem;
        border-radius: 999px;
        background-color: #e0edff;
        color: #1d4ed8;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    .cs-section-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
        color: #111827;
    }

    .cs-divider {
        border-bottom: 1px solid #e5e7eb;
        margin: 0.75rem 0 1.2rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar framing (navigation structure untouched)
with st.sidebar:
    st.markdown("### CardioSignals Dashboard")
    st.markdown("v1.0 **Research Build**")
    st.markdown("---")
    st.markdown(
        "ECG-first **cardiovascular risk exploration**:\n"
        "- ECG-based risk prediction\n"
        "- Links to BP & cholesterol\n"
        "- Downstream outcome validation"
    )

# Header layout
title_col, meta_col = st.columns([3, 1.4])
with title_col:
    st.markdown("## 🫀 CardioSignals")
    st.markdown(
        "<div class='cs-subtle'>Inferring cardiovascular risk directly from ECG signals.</div>",
        unsafe_allow_html=True,
    )

with meta_col:
    st.markdown("<span class='cs-badge'>Research prototype</span>", unsafe_allow_html=True)
    st.caption("Not for clinical decision-making.")

st.markdown("<div class='cs-divider'></div>", unsafe_allow_html=True)

# Overview cards
top_col1, top_col2 = st.columns(2)

with top_col1:
    st.markdown(
        """
        <div class="cs-card">
            <div class="cs-card-header">What does this app demonstrate?</div>
            <div class="cs-subtle">
                This system shows how <strong>ECG signals alone</strong> can be used to:
            </div>
            <ul>
                <li>Detect cardiovascular risk early</li>
                <li>Approximate clinical measurements (BP, Cholesterol)</li>
                <li>Predict downstream cardiac failure outcomes</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_col2:
    st.markdown(
        """
        <div class="cs-card">
            <div class="cs-card-header">Intended use</div>
            <div class="cs-subtle">
                ⚠️ <strong>Research & educational use only.</strong><br/>
                Not a medical diagnostic tool.
            </div>
            <br/>
            <div class="cs-subtle">
                Use this dashboard to explore how ECG-derived features relate to:
                <ul>
                    <li>Individual risk probabilities</li>
                    <li>Blood pressure and cholesterol patterns</li>
                    <li>Downstream cardiac failure outcomes</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="cs-card">
        <div class="cs-section-title">How to use this dashboard</div>
        <div class="cs-subtle">
            Use the navigation in the left sidebar to move between:
        </div>
        <ul>
            <li><strong>ECG Risk Prediction</strong> – upload an ECG and view model-estimated risk.</li>
            <li><strong>Explainability</strong> – inspect saliency maps for model decisions.</li>
            <li><strong>Physiological links</strong> – see how ECG relates to BP and cholesterol.</li>
            <li><strong>Outcome relevance</strong> – review ROC-AUC validation against real outcomes.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)