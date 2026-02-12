import streamlit as st
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="CardioSignals – Cardiovascular Risk from ECG",
    page_icon="🫀",
    layout="wide"
)

# Global neo-clinical theming (UI only)
st.markdown(
    """
    <style>
    /* Full-page gradient medical background */
    .main {
        background: radial-gradient(circle at top left, #0f172a 0, #020617 55%, #020617 100%);
        color: #e5e7eb;
    }

    /* Hero section */
    .cs-hero {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.14), rgba(59, 130, 246, 0.06));
        border-radius: 1.25rem;
        padding: 1.8rem 2rem;
        border: 1px solid rgba(148, 163, 184, 0.35);
        box-shadow: 0 20px 45px rgba(15, 23, 42, 0.65);
        position: relative;
        overflow: hidden;
    }

    .cs-hero::before {
        content: "";
        position: absolute;
        inset: -40%;
        background-image:
            radial-gradient(circle at 10% 20%, rgba(56, 189, 248, 0.2) 0, transparent 55%),
            radial-gradient(circle at 80% 0%, rgba(59, 130, 246, 0.25) 0, transparent 50%);
        opacity: 0.7;
        z-index: -2;
    }

    .cs-hero-ecg {
        position: absolute;
        inset: 0;
        background-image: linear-gradient(
            120deg,
            transparent 0,
            transparent 18%,
            rgba(248, 250, 252, 0.7) 19%,
            transparent 20%,
            transparent 40%,
            rgba(248, 250, 252, 0.7) 41%,
            transparent 42%,
            transparent 70%,
            rgba(248, 250, 252, 0.6) 71%,
            transparent 72%
        );
        opacity: 0.18;
        animation: cs-ecg-move 14s linear infinite;
        z-index: -1;
    }

    @keyframes cs-ecg-move {
        0% { transform: translateX(0); }
        100% { transform: translateX(-25%); }
    }

    .cs-hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: 0.03em;
        color: #e5e7eb;
    }

    .cs-hero-tagline {
        color: #cbd5f5;
        font-size: 0.98rem;
        margin-top: 0.35rem;
    }

    /* Generic card container */
    .cs-card {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.9), rgba(15, 23, 42, 0.94));
        padding: 1.25rem 1.5rem;
        border-radius: 0.9rem;
        box-shadow: 0 18px 35px rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(51, 65, 85, 0.95);
        margin-bottom: 1.1rem;
    }

    .cs-card-header {
        font-weight: 700;
        font-size: 1.02rem;
        margin-bottom: 0.45rem;
        color: #f9fafb;
    }

    .cs-subtle {
        color: #9ca3af;
        font-size: 0.9rem;
    }

    .cs-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.20rem 0.80rem;
        border-radius: 999px;
        background: radial-gradient(circle at 0 0, #22d3ee, #1d4ed8);
        color: #e5f2ff;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.09em;
        text-transform: uppercase;
    }

    .cs-section-title {
        font-size: 1.02rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
        color: #e5e7eb;
    }

    .cs-divider {
        border-bottom: 1px solid rgba(51, 65, 85, 0.9);
        margin: 0.75rem 0 1.2rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar framing (navigation structure untouched)
with st.sidebar:
    st.markdown("### 🫀 CardioSignals")
    st.markdown("v1.0 **Research Build**")
    st.markdown("---")
    st.markdown(
        "A modern ECG-first **cardiovascular risk** explorer:\n"
        "- ECG risk estimation\n"
        "- BP & cholesterol links\n"
        "- Outcome relevance metrics"
    )

# Hero-style header
hero_left, hero_right = st.columns([2.4, 1.6])
with hero_left:
    st.markdown(
        """
        <div class="cs-hero">
            <div class="cs-hero-ecg"></div>
            <div class="cs-hero-title">CardioSignals</div>
            <div class="cs-hero-tagline">
                React-style research dashboard for <strong>ECG-driven cardiovascular risk</strong>.
            </div>
            <br/>
            <div class="cs-subtle">
                Upload ECGs, explore explainability, and relate signal-derived risk to
                blood pressure, cholesterol and downstream failure outcomes &mdash;
                all in a single, clinician-friendly interface.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with hero_right:
    st.markdown("<span class='cs-badge'>Research prototype · Not for clinical use</span>", unsafe_allow_html=True)
    st.caption("Outputs are for exploration only and must not be used for diagnosis or treatment decisions.")

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
            <div class="cs-card-header">Navigation</div>
            <div class="cs-subtle">
                Use the sidebar to move between:
            </div>
            <ul>
                <li><strong>ECG Risk Prediction</strong> – upload an ECG and view model-estimated risk.</li>
                <li><strong>Explainability</strong> – inspect saliency maps for model decisions.</li>
                <li><strong>BP / Cholesterol Insights</strong> – physiological context for ECG-derived signals.</li>
                <li><strong>Outcome relevance</strong> – ROC-AUC validation against real outcomes.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )