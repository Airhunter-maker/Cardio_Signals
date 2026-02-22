"""
pages/dashboard.py
Page 1 — Dashboard with Hero, Metric Cards, Architecture, and Dataset Snapshot.
"""

import os
import sys
import streamlit as st
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from components.ui_components import (
    metric_card, section_header, insight_card, flow_diagram, empty_state_card
)
from components.charts import (
    age_distribution_chart, gender_donut_chart,
    cholesterol_bar_chart, target_donut_chart
)
from core.data_loader import load_cardio_data


def render():
    # ── HERO ──────────────────────────────────────────────────────────────
    st.markdown("""
<div style="position:relative;background:linear-gradient(135deg,#080C16 0%,#0D1830 50%,#080C16 100%);
     border-radius:16px;padding:3rem 2.5rem;margin-bottom:2rem;
     border:1px solid #1E2A3A;overflow:hidden;">

  <!-- Animated ECG SVG background -->
  <svg style="position:absolute;top:50%;left:0;transform:translateY(-50%);
       width:100%;height:80px;opacity:0.12;" viewBox="0 0 1200 80"
       preserveAspectRatio="none">
    <polyline id="ecgPath"
      points="0,40 60,40 80,40 90,10 100,70 110,20 120,60 130,40
              200,40 300,40 310,40 320,10 330,70 340,20 350,60 360,40
              430,40 500,40 510,40 520,10 530,70 540,20 550,60 560,40
              630,40 700,40 710,40 720,10 730,70 740,20 750,60 760,40
              830,40 900,40 910,40 920,10 930,70 940,20 950,60 960,40
              1030,40 1100,40 1110,40 1120,10 1130,70 1140,20 1150,60 1200,40"
      fill="none" stroke="#00D4FF" stroke-width="2.5"/>
  </svg>
  <style>
    @keyframes ecgScroll {
      0%   { stroke-dashoffset: 0; }
      100% { stroke-dashoffset: -1200; }
    }
    #ecgPath {
      stroke-dasharray: 1200;
      animation: ecgScroll 4s linear infinite;
    }
  </style>

  <!-- Content -->
  <div style="position:relative;z-index:2;">
    <h1 style="font-size:3.2rem;font-weight:800;color:#F1F5F9;
         margin:0 0 0.5rem 0;letter-spacing:-0.02em;">
      🫀 CardioSignals
    </h1>
    <p style="font-size:1rem;color:#94A3B8;max-width:600px;margin:0 0 1.5rem 0;
         line-height:1.6;">
      Explainable cardiovascular risk intelligence trained on
      70,000 de-identified patient records
    </p>
    <div style="display:flex;gap:0.75rem;flex-wrap:wrap;">
      <span style="background:rgba(0,212,255,0.08);color:#00D4FF;
           border:1px solid rgba(0,212,255,0.3);border-radius:20px;
           padding:5px 14px;font-size:0.78rem;font-weight:600;">
        🏥 Early Detection
      </span>
      <span style="background:rgba(124,58,237,0.08);color:#7C3AED;
           border:1px solid rgba(124,58,237,0.3);border-radius:20px;
           padding:5px 14px;font-size:0.78rem;font-weight:600;">
        🔍 Interpretability
      </span>
      <span style="background:rgba(16,185,129,0.08);color:#10B981;
           border:1px solid rgba(16,185,129,0.3);border-radius:20px;
           padding:5px 14px;font-size:0.78rem;font-weight:600;">
        📡 Multi-Modal Analysis
      </span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── FOUR METRIC CARDS ──────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Patients Analysed", "70,000",
                    "de-identified records", "#00D4FF")
    with c2:
        metric_card("Best Model AUC", "0.784",
                    "Random Forest", "#10B981")
    with c3:
        metric_card("Clinical Features", "11",
                    "incl. BP, cholesterol, BMI", "#7C3AED")
    with c4:
        metric_card("Dataset Balance", "50 / 50",
                    "positive vs negative cases", "#F59E0B")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── THREE COLUMNS ──────────────────────────────────────────────────────
    left, mid, right = st.columns([1.2, 1, 0.8])

    # LEFT — The Problem
    with left:
        section_header("The Access Gap", badge="Context")
        insight_card(
            title="Why ECG + Clinical Data?",
            body=(
                "Cardiovascular disease causes 18M deaths annually. "
                "Traditional risk scoring requires lab visits, blood "
                "draws, and specialist access — unavailable to billions. "
                "CardioSignals uses clinical biomarkers and ECG signal "
                "analysis already available in most clinics to deliver "
                "explainable, immediate risk stratification."
            ),
            accent="#00D4FF",
            icon="🫀",
        )
        st.markdown("""
<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:0.75rem;">
  <span style="background:#111827;border:1px solid #1E3A4A;border-radius:20px;
       padding:5px 12px;font-size:0.75rem;color:#F1F5F9;">
    18M deaths/year
  </span>
  <span style="background:#111827;border:1px solid #1E3A4A;border-radius:20px;
       padding:5px 12px;font-size:0.75rem;color:#F1F5F9;">
    80% preventable
  </span>
  <span style="background:#111827;border:1px solid #1E3A4A;border-radius:20px;
       padding:5px 12px;font-size:0.75rem;color:#F1F5F9;">
    50% lack lab access
  </span>
</div>
""", unsafe_allow_html=True)

    # MIDDLE — Architecture
    with mid:
        section_header("System Architecture", badge="Technical")
        flow_diagram([
            {"icon": "📊", "title": "Clinical Data",
             "subtitle": "70,000 patients · 11 features"},
            {"icon": "⚙️", "title": "Preprocessing",
             "subtitle": "BP filtering · age conversion · BMI"},
            {"icon": "🤖", "title": "ML Models",
             "subtitle": "Logistic Regression + Random Forest"},
            {"icon": "🔍", "title": "SHAP Explainer",
             "subtitle": "Feature-level attribution"},
            {"icon": "📡", "title": "ECG CNN",
             "subtitle": "1D Conv · Saliency Maps"},
            {"icon": "📋", "title": "Risk Score",
             "subtitle": "Explainable cardiovascular risk output"},
        ])

    # RIGHT — Evaluation Criteria
    with right:
        section_header("Criteria Coverage", badge="Hackathon")
        criteria = [
            ("#00D4FF", "✅", "Model Performance", "RF AUC 0.784"),
            ("#7C3AED", "✅", "Innovation",         "Multi-modal pipeline"),
            ("#10B981", "✅", "Explainability",     "SHAP + Saliency Maps"),
            ("#F59E0B", "✅", "Storytelling",        "End-to-end clinical demo"),
        ]
        for color, check, label, detail in criteria:
            st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;
     background:#111827;border-radius:8px;padding:0.65rem 0.8rem;
     margin-bottom:8px;border-left:3px solid {color};">
  <span style="font-size:1rem;">{check}</span>
  <div>
    <div style="font-weight:700;color:#F1F5F9;font-size:0.82rem;">{label}</div>
    <div style="font-size:0.7rem;color:#64748B;">{detail}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── DATASET SNAPSHOT ───────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section_header(
        "Dataset at a Glance",
        subtitle="cardio_base.csv — 70,000 de-identified patients",
        badge="Real Data",
    )

    df = load_cardio_data()
    if df is not None:
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.plotly_chart(age_distribution_chart(df, 220),
                            use_container_width=True)
        with d2:
            st.plotly_chart(gender_donut_chart(df, 220),
                            use_container_width=True)
        with d3:
            st.plotly_chart(cholesterol_bar_chart(df, 220),
                            use_container_width=True)
        with d4:
            st.plotly_chart(target_donut_chart(df, 220),
                            use_container_width=True)
    else:
        insight_card(
            "Data Unavailable",
            "Could not load <code>cardio_base.csv</code>. "
            "Expected at <code>data/raw/cardio_base.csv</code>.",
            accent="#F59E0B",
            icon="⚠️",
        )
