"""Dashboard — Page 1."""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def render_dashboard():
    from components.ui_components import (
        metric_card, section_header, insight_card, flow_diagram)
    from core.data_loader import load_cardio_data

    # ── HERO ─────────────────────────────────────────────────────
    st.markdown("""
    <div style='
        background: linear-gradient(135deg, #0A0E1A 0%, #0D1829 100%);
        border: 1px solid #1E2A3A;
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    '>
        <svg style='position:absolute;top:0;left:0;width:100%;
                    height:100%;opacity:0.08;' viewBox='0 0 800 120'>
            <polyline points='0,60 60,60 80,20 100,100 120,60
                              160,60 180,40 200,80 220,60 260,60
                              280,10 300,110 320,60 360,60
                              380,45 400,75 420,60 460,60
                              480,25 500,95 520,60 560,60
                              580,50 600,70 620,60 660,60
                              680,15 700,105 720,60 800,60'
                  fill='none' stroke='#00D4FF' stroke-width='2'/>
        </svg>
        <h1 style='font-size:2.8rem; font-weight:800; color:#F1F5F9;
                   margin:0; letter-spacing:-0.02em; position:relative;'>
            🫀 CardioSignals
        </h1>
        <p style='font-size:1rem; color:#94A3B8; margin:0.5rem 0 1rem 0;
                   position:relative;'>
            Explainable cardiovascular risk intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── DATASET SNAPSHOT ─────────────────────────────────────────
    section_header(
        "Dataset at a Glance",
        subtitle="cardio_base.csv — 68,754 de-identified patients",
        badge="Real Data"
    )

    df = load_cardio_data()

    if df.empty:
        insight_card("Data Unavailable",
                     "cardio_base.csv could not be loaded.",
                     accent="#F59E0B", icon="⚠️")
        return

    c1, c2, c3, c4 = st.columns(4)
    chart_layout = dict(
        paper_bgcolor="#111827", plot_bgcolor="#111827",
        font=dict(color="#94A3B8", size=11),
        margin=dict(l=10, r=10, t=30, b=10),
        height=210,
    )

    with c1:
        fig = go.Figure(go.Histogram(
            x=df["age_years"], nbinsx=20,
            marker_color="#00D4FF", opacity=0.85,
            name="Patients"
        ))
        fig.update_layout(**chart_layout, title="Age Distribution",
                          showlegend=False)
        fig.update_xaxes(showgrid=False, title_text="Age (years)",
                         color="#64748B")
        fig.update_yaxes(showgrid=False, color="#64748B")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        gender_counts = df["gender"].value_counts()
        fig = go.Figure(go.Pie(
            labels=["Female", "Male"],
            values=[gender_counts.get(1, 0), gender_counts.get(2, 0)],
            marker_colors=["#7C3AED", "#00D4FF"],
            hole=0.55,
            textfont_size=11
        ))
        fig.update_layout(**chart_layout, title="Gender Split", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        chol = df["cholesterol"].value_counts().sort_index()
        fig = go.Figure(go.Bar(
            x=["Normal", "Above\nNormal", "Well Above\nNormal"],
            y=[chol.get(1, 0), chol.get(2, 0), chol.get(3, 0)],
            marker_color=["#10B981", "#F59E0B", "#EF4444"],
        ))
        fig.update_layout(**chart_layout, title="Cholesterol Levels",
                          showlegend=False)
        fig.update_xaxes(showgrid=False, color="#64748B")
        fig.update_yaxes(showgrid=False, color="#64748B")
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        target = df["cardio"].value_counts()
        fig = go.Figure(go.Pie(
            labels=["No CVD", "CVD"],
            values=[target.get(0, 0), target.get(1, 0)],
            marker_colors=["#10B981", "#EF4444"],
            hole=0.55,
            textfont_size=11
        ))
        fig.update_layout(**chart_layout, title="Target Distribution", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)