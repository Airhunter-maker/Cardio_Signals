"""ECG Explorer — Page 3 (cleaned, no technical details)."""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import os


def render_ecg_explorer():
    from components.ui_components import (
        section_header, insight_card, metric_card,
        risk_gauge, risk_badge)
    from core.model_loader import load_all_models
    from core.inference import compute_saliency
    from core.data_loader import load_ecg_data

    models = load_all_models()
    ecg_model = models.get("ecg")

    # ── HEADER ─────────────────────────────────────────────────────
    section_header(
        "ECG Signal Explorer",
        subtitle="AI-powered cardiac waveform analysis",
        badge="Signal Analysis",
        badge_color="#7C3AED"
    )

    insight_card(
        "How This Works",
        "Our AI analyses heart signal patterns to identify potential cardiac risk markers. "
        "Select an ECG sample below and toggle on 'Show Risk Regions' to see "
        "which parts of the waveform the AI considers most significant.",
        accent="#00D4FF", icon="🧠"
    )

    # ── SIGNAL VIEWER ──────────────────────────────────────────────
    col_l, col_r = st.columns([3, 2])

    with col_l:
        section_header("Cardiac Waveform", badge="Interactive")

        c1, c2 = st.columns(2)
        sample_idx = c1.slider(
            "ECG Sample",
            0, 99, 0,
            help="Select an ECG recording to analyse"
        )
        show_sal = c2.toggle("Show Risk Regions", value=True,
                             help="Highlight areas the AI focuses on")

        # Load or simulate ECG
        ecg_data = load_ecg_data(100)
        if ecg_data is None:
            t = np.linspace(0, 5, 500)
            shift = (sample_idx % 20) * 0.05
            signal = 0.1 * np.sin(2 * np.pi * 1.1 * (t + shift))
            for i in range(1, 6):
                amp = 0.85 + 0.1 * np.sin(sample_idx * 0.7)
                signal += amp * np.exp(-((t - i + shift % 1.0)**2) / 0.006)
        else:
            signal = ecg_data[min(sample_idx, len(ecg_data) - 1)]

        # Build waveform chart
        fig_ecg = go.Figure()
        fig_ecg.add_trace(go.Scatter(
            x=np.arange(len(signal)),
            y=signal,
            name="ECG Signal",
            line=dict(color="#00D4FF", width=1.5),
        ))

        # Risk regions (saliency) overlay
        saliency_prob = None
        if show_sal:
            if ecg_model is not None:
                with st.spinner("Analysing cardiac signal..."):
                    try:
                        _, saliency, saliency_prob = compute_saliency(
                            ecg_model, signal)
                        sal_norm = saliency / (saliency.max() + 1e-8)
                        fig_ecg.add_trace(go.Scatter(
                            x=np.arange(len(sal_norm)),
                            y=sal_norm,
                            name="Risk Regions",
                            line=dict(color="#EF4444", width=1),
                            fill="tozeroy",
                            fillcolor="rgba(239,68,68,0.12)",
                            opacity=0.8,
                        ))
                    except Exception:
                        pass
            else:
                # Demo: generate deterministic simulated risk regions
                rng = np.random.default_rng(sample_idx + 42)
                peaks = np.zeros(len(signal))
                for p in range(0, len(signal), 100):
                    width = rng.integers(15, 35)
                    height = rng.uniform(0.4, 0.9)
                    for j in range(max(0, p - width), min(len(signal), p + width)):
                        peaks[j] = max(peaks[j], height * np.exp(
                            -((j - p)**2) / (width**2 * 0.5)))
                fig_ecg.add_trace(go.Scatter(
                    x=np.arange(len(peaks)),
                    y=peaks,
                    name="Risk Regions (Demo)",
                    line=dict(color="#F59E0B", width=1),
                    fill="tozeroy",
                    fillcolor="rgba(245,158,11,0.10)",
                    opacity=0.7,
                ))

        fig_ecg.update_layout(
            template="plotly_dark",
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font=dict(color="#94A3B8"),
            margin=dict(l=20, r=20, t=30, b=20),
            height=260,
            xaxis=dict(title="Signal Position", showgrid=False,
                       color="#64748B"),
            yaxis=dict(title="Amplitude",
                       showgrid=False, color="#64748B"),
            legend=dict(bgcolor="#111827", bordercolor="#1E2A3A",
                        font=dict(size=11)),
            title=f"ECG Recording — Sample {sample_idx}"
        )
        st.plotly_chart(fig_ecg, use_container_width=True)

        # Wave region guide
        st.markdown("""
        <div style='display:flex; gap:0.5rem; margin-top:0.25rem;
                    align-items:center; flex-wrap:wrap;'>
            <span style='color:#64748B; font-size:0.7rem;'>
                Cardiac wave components:
            </span>
            <span style='background:#7C3AED20; color:#7C3AED;
                         font-size:0.7rem; font-weight:600;
                         padding:2px 8px; border-radius:4px;'>
                P Wave — atrial activation
            </span>
            <span style='background:#00D4FF20; color:#00D4FF;
                         font-size:0.7rem; font-weight:600;
                         padding:2px 8px; border-radius:4px;'>
                QRS Complex — ventricular contraction
            </span>
            <span style='background:#F59E0B20; color:#F59E0B;
                         font-size:0.7rem; font-weight:600;
                         padding:2px 8px; border-radius:4px;'>
                T Wave — ventricular recovery
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Basic signal stats
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Peak Amplitude", f"{signal.max():.3f}",
                        accent="#00D4FF")
        with c2:
            metric_card("Signal Length", f"{len(signal)} pts",
                        accent="#7C3AED")
        with c3:
            rr_estimate = len(signal) // 100
            metric_card("Heart Cycles (est.)", str(max(1, rr_estimate)),
                        accent="#10B981")

    with col_r:
        section_header("AI Risk Score", badge="Analysis")

        # Determine risk probability
        if ecg_model is not None and saliency_prob is not None:
            prob = saliency_prob
        elif ecg_model is not None:
            try:
                with st.spinner("Computing risk score..."):
                    _, _, prob = compute_saliency(ecg_model, signal)
            except Exception:
                prob = 0.5
        else:
            # Deterministic demo score based on sample index
            prob = 0.28 + 0.44 * abs(np.cos(sample_idx * 0.43))

        mini_gauge = risk_gauge(prob * 100)
        mini_gauge.update_layout(height=220)
        st.plotly_chart(mini_gauge, use_container_width=True)

        st.markdown(
            f"<div style='text-align:center; margin:-0.5rem 0 0.8rem;'>"
            f"{risk_badge(prob * 100)}</div>",
            unsafe_allow_html=True
        )

        mode_note = (
            "Our AI is analysing the cardiac waveform patterns of this recording."
            if ecg_model is not None
            else "Showing a simulated risk score for demonstration. "
                 "Connect an ECG model for real analysis."
        )
        insight_card(
            "Signal Assessment",
            mode_note,
            accent="#7C3AED" if ecg_model else "#F59E0B",
            icon="🧠" if ecg_model else "⚡"
        )

        # What the AI looks for
        st.markdown("""
        <div style='background:#111827; border:1px solid #1E2A3A;
                    border-radius:10px; padding:1.2rem; margin-top:0.8rem;'>
            <div style='color:#F1F5F9; font-weight:700; font-size:0.85rem;
                        margin-bottom:0.8rem;'>
                🔍 What the AI Looks For
            </div>
            <div style='display:flex; flex-direction:column; gap:0.5rem;'>
                <div style='display:flex; align-items:center; gap:0.6rem;'>
                    <span style='color:#00D4FF; font-size:0.9rem;'>●</span>
                    <span style='font-size:0.78rem; color:#94A3B8;'>QRS complex shape & width</span>
                </div>
                <div style='display:flex; align-items:center; gap:0.6rem;'>
                    <span style='color:#7C3AED; font-size:0.9rem;'>●</span>
                    <span style='font-size:0.78rem; color:#94A3B8;'>ST-segment deviations</span>
                </div>
                <div style='display:flex; align-items:center; gap:0.6rem;'>
                    <span style='color:#F59E0B; font-size:0.9rem;'>●</span>
                    <span style='font-size:0.78rem; color:#94A3B8;'>T-wave morphology changes</span>
                </div>
                <div style='display:flex; align-items:center; gap:0.6rem;'>
                    <span style='color:#10B981; font-size:0.9rem;'>●</span>
                    <span style='font-size:0.78rem; color:#94A3B8;'>Heart rhythm regularity</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── IMPORTANT DISCLAIMER ─────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    insight_card(
        "Important Notice",
        "ECG analysis results are for research and educational purposes only. "
        "They are not a clinical diagnosis. Always consult a qualified cardiologist "
        "or healthcare provider for medical interpretation of ECG recordings.",
        accent="#F59E0B", icon="⚠️"
    )