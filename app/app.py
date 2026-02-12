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
    .stApp {
        background: radial-gradient(circle at top left, #0f172a 0, #020617 55%, #020617 100%);
        color: #e5e7eb;
    }

    /* Hero section */
    .cs-hero {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.14), rgba(59, 130, 246, 0.06));
        border-radius: 1.25rem;
        padding: 2.2rem 2.5rem;
        border: 1px solid rgba(148, 163, 184, 0.25);
        box-shadow: 0 20px 45px rgba(0, 0, 0, 0.4);
        position: relative;
        overflow: hidden;
        margin-bottom: 2rem;
    }

    .cs-hero::before {
        content: "";
        position: absolute;
        inset: -40%;
        background-image:
            radial-gradient(circle at 10% 20%, rgba(56, 189, 248, 0.1) 0, transparent 55%),
            radial-gradient(circle at 80% 0%, rgba(59, 130, 246, 0.15) 0, transparent 50%);
        opacity: 0.7;
        z-index: 0;
        pointer-events: none;
    }

    .cs-hero-ecg {
        position: absolute;
        inset: 0;
        background-image: linear-gradient(
            120deg,
            transparent 0,
            transparent 18%,
            rgba(248, 250, 252, 0.3) 19%,
            transparent 20%,
            transparent 40%,
            rgba(248, 250, 252, 0.3) 41%,
            transparent 42%,
            transparent 70%,
            rgba(248, 250, 252, 0.2) 71%,
            transparent 72%
        );
        opacity: 0.12;
        animation: cs-ecg-move 14s linear infinite;
        z-index: 0;
        pointer-events: none;
    }

    @keyframes cs-ecg-move {
        0% { transform: translateX(0); }
        100% { transform: translateX(-25%); }
    }

    .cs-hero-content {
        position: relative;
        z-index: 1;
    }

    .cs-hero-title {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(to right, #e5e7eb, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .cs-hero-tagline {
        color: #94a3b8;
        font-size: 1.15rem;
        line-height: 1.6;
        font-weight: 300;
        max-width: 800px;
    }

    /* Generic card container */
    .cs-card {
        background: rgba(30, 41, 59, 0.4);
        padding: 1.75rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(148, 163, 184, 0.1);
        backdrop-filter: blur(8px);
        margin-bottom: 1.5rem;
        height: 100%;
        transition: transform 0.2s ease-in-out, border-color 0.2s;
    }
    
    .cs-card:hover {
        border-color: rgba(56, 189, 248, 0.3);
    }

    .cs-card-header {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.75rem;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .cs-subtle {
        color: #94a3b8;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    .cs-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.2);
        color: #38bdf8;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .cs-divider {
        height: 1px;
        background: linear-gradient(to right, transparent, rgba(148, 163, 184, 0.2), transparent);
        margin: 2.5rem 0;
    }
    
    /* List styling */
    .cs-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .cs-list li {
        position: relative;
        padding-left: 1.5rem;
        margin-bottom: 0.5rem;
        color: #cbd5e1;
    }
    
    .cs-list li::before {
        content: "•";
        color: #38bdf8;
        font-weight: bold;
        position: absolute;
        left: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar framing (navigation structure untouched)
with st.sidebar:
    st.markdown("### 🫀 CardioSignals")
    st.markdown('<span class="cs-badge">v1.0 Research Build</span>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        """
        <div class="cs-subtle" style="font-size: 0.9rem;">
        A modern ECG-first <strong>cardiovascular risk</strong> explorer:
        <br/><br/>
        <ul class="cs-list" style="padding-left: 0;">
            <li>ECG risk estimation</li>
            <li>BP & cholesterol links</li>
            <li>Outcome relevance metrics</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

# Hero-style header
st.markdown(
    """
    <div class="cs-hero">
        <div class="cs-hero-ecg"></div>
        <div class="cs-hero-content">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <div class="cs-hero-title">CardioSignals</div>
                    <div class="cs-hero-tagline">
                        React-style research dashboard for <strong>ECG-driven cardiovascular risk</strong>.
                    </div>
                </div>
                <div style="text-align: right;">
                    <span class='cs-badge'>Research Prototype</span>
                </div>
            </div>
            <br/>
            <div class="cs-subtle" style="max-width: 750px; font-size: 1.05rem; color: #cbd5e1;">
                Upload ECGs, explore explainability, and relate signal-derived risk to
                blood pressure, cholesterol and downstream failure outcomes &mdash;
                all in a single, clinician-friendly interface.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("⚠️ Outputs are for exploration only and must not be used for diagnosis or treatment decisions.")

st.markdown("<div class='cs-divider'></div>", unsafe_allow_html=True)

# Overview cards
top_col1, top_col2 = st.columns(2)

with top_col1:
    st.markdown(
        """
        <div class="cs-card">
            <div class="cs-card-header">
                Experiment Overview
            </div>
            <div class="cs-subtle" style="margin-bottom: 1rem;">
                This system shows how <strong>ECG signals alone</strong> can be used to:
            </div>
            <ul class="cs-list">
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
            <div class="cs-card-header">
                System Navigation
            </div>
            <div class="cs-subtle" style="margin-bottom: 1rem;">
                Use the sidebar to accesses:
            </div>
            <ul class="cs-list">
                <li><strong>ECG Risk Prediction</strong> – Upload an ECG for model-estimated risk.</li>
                <li><strong>Explainability</strong> – Inspect saliency maps for interactions.</li>
                <li><strong>BP / Cholesterol</strong> – Physiological context overlays.</li>
                <li><strong>Outcome Relevance</strong> – ROC-AUC validation metrics.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )