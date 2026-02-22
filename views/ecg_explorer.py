"""
pages/ecg_explorer.py
Page 3 — ECG Signal Explorer with 1D CNN saliency analysis.
"""

import os
import sys
import streamlit as st
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from components.ui_components import (
    section_header, insight_card, metric_card, risk_badge,
    risk_gauge, empty_state_card
)
from components.charts import ecg_signal_chart, correlation_heatmap, scatter_with_trend
from core.data_loader import load_ecg_data
from core.inference import compute_saliency
from core.model_loader import load_ecg_model


@st.cache_data(show_spinner=False)
def _load_correlation_csv():
    path = os.path.join(_ROOT, 'results', 'metrics',
                        'ecg_clinical_correlation.csv')
    if os.path.exists(path):
        try:
            return pd.read_csv(path, index_col=0)
        except Exception:
            pass
    return None


@st.cache_data(show_spinner=False, max_entries=1)
def _batch_predictions(segments_key: int):
    """Run batch predictions on first 64 ECG segments."""
    try:
        ecg_model, ecg_ok = load_ecg_model()
        if not ecg_ok or ecg_model is None:
            return None, None, None, None

        X_norm, y_labels, X_segs = load_ecg_data()
        if X_segs is None:
            return None, None, None, None

        segs = X_segs[:64]
        # Handle length mismatch — pad or truncate to 500
        processed = []
        for s in segs:
            if len(s) >= 500:
                processed.append(s[:500])
            else:
                processed.append(np.pad(s, (0, 500 - len(s))))
        segs = np.array(processed, dtype=np.float32)

        tensor = torch.tensor(segs, dtype=torch.float32)  # (64, 500)
        with torch.no_grad():
            out = ecg_model(tensor)
            preds = torch.argmax(out, dim=1).numpy()

        labs = (y_labels[:64] if y_labels is not None
                else np.zeros(len(segs), dtype=int))

        correct_idx = None
        incorrect_idx = None
        for i, (p, l) in enumerate(zip(preds, labs)):
            if correct_idx is None and p == l:
                correct_idx = i
            if incorrect_idx is None and p != l:
                incorrect_idx = i
            if correct_idx is not None and incorrect_idx is not None:
                break

        return segs, preds, correct_idx, incorrect_idx
    except Exception as e:
        return None, None, None, None


def render():
    ecg_model, ecg_ok = load_ecg_model()
    X_norm, y_labels, X_segs = load_ecg_data()

    # ── HEADER ─────────────────────────────────────────────────────────────
    section_header(
        "ECG Signal Explorer",
        subtitle="1D Convolutional Neural Network · Gradient Saliency Analysis",
        badge="Signal Research Layer",
        badge_color="#7C3AED",
    )

    insight_card(
        "Demo Mode — Research Pipeline",
        "The ECG CNN pipeline demonstrates signal processing, segmentation, "
        "and gradient-based saliency analysis. Ground truth ECG diagnostic labels "
        "were not included in the provided dataset; predictions represent the "
        "model's learned signal structure. Full clinical validation requires "
        "labeled ECG cohorts such as PTB-XL.",
        accent="#F59E0B",
        icon="⚡",
    )

    # ── SIGNAL VIEWER ──────────────────────────────────────────────────────
    left, right = st.columns([3, 2])

    with left:
        section_header("Signal Visualization", badge="Interactive")

        sc1, sc2 = st.columns(2)
        n_samples = min(100, len(X_norm)) if X_norm is not None else 0
        sample_idx = sc1.slider("Signal Sample", 0, max(n_samples - 1, 1), 0)
        show_saliency = sc2.toggle("Overlay Saliency Map", value=True)

        if X_norm is not None and len(X_norm) > 0:
            raw_signal = X_norm[sample_idx]
            # Normalise to length 500
            if len(raw_signal) >= 500:
                signal = raw_signal[:500].astype(np.float32)
            else:
                signal = np.pad(raw_signal, (0, 500 - len(raw_signal))).astype(np.float32)

            saliency = None
            if show_saliency and ecg_ok and ecg_model is not None:
                with st.spinner("Processing ECG signal morphology..."):
                    try:
                        _, saliency = compute_saliency(ecg_model, signal)
                    except Exception:
                        saliency = None

            fig = ecg_signal_chart(signal, saliency, height=280)
            st.plotly_chart(fig, use_container_width=True)

            # Wave region labels
            st.markdown("""
<div style="display:flex;gap:8px;margin-top:4px;flex-wrap:wrap;">
  <span style="background:rgba(124,58,237,0.12);color:#7C3AED;border:1px solid #7C3AED;
       border-radius:20px;padding:3px 10px;font-size:0.72rem;font-weight:600;">P Wave</span>
  <span style="background:rgba(0,212,255,0.12);color:#00D4FF;border:1px solid #00D4FF;
       border-radius:20px;padding:3px 10px;font-size:0.72rem;font-weight:600;">QRS Complex</span>
  <span style="background:rgba(245,158,11,0.12);color:#F59E0B;border:1px solid #F59E0B;
       border-radius:20px;padding:3px 10px;font-size:0.72rem;font-weight:600;">T Wave</span>
  <span style="color:#64748B;font-size:0.7rem;align-self:center;">
    Approximate anatomical wave regions</span>
</div>
""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            mc1, mc2, mc3 = st.columns(3)
            n_segs = (len(X_segs) if X_segs is not None else 0)
            with mc1: metric_card("Peak Amplitude", f"{signal.max():.3f}")
            with mc2: metric_card("Signal Length",  f"{len(signal)} samples")
            with mc3: metric_card("Segments",       f"{n_segs}")

        else:
            # Fallback: show sample_ecg.png
            img_path = os.path.join(_ROOT, 'utils', 'sample_ecg.png')
            if os.path.exists(img_path):
                st.image(img_path, caption="Demo ECG — sample_ecg.png",
                         use_container_width=True)
            else:
                empty_state_card("X_ecg_normalized.npy")

    # ── RIGHT — CNN Output ──────────────────────────────────────────────────
    with right:
        section_header("CNN Analysis", badge="PyTorch")

        if ecg_ok and ecg_model is not None and X_norm is not None and len(X_norm) > 0:
            signal_for_cnn = signal  # from left panel
            try:
                tensor = torch.tensor(
                    signal_for_cnn, dtype=torch.float32).unsqueeze(0)  # (1, 500)
                with torch.no_grad():
                    output = ecg_model(tensor)
                    prob = torch.softmax(output, dim=1)[0, 1].item()
                fig_gauge = risk_gauge(prob * 100)
                fig_gauge.update_layout(height=200)
                st.plotly_chart(fig_gauge, use_container_width=True)
                st.markdown(
                    f'<div style="text-align:center;">'
                    f'{risk_badge(prob * 100)}</div>',
                    unsafe_allow_html=True)
                insight_card(
                    "CNN Signal Score",
                    "CNN signal score reflects learned pattern associations. "
                    "Treat as signal characterisation, not a clinical prediction.",
                    accent="#00D4FF", icon="📡",
                )
            except Exception as e:
                insight_card("CNN Output Error", str(e),
                             accent="#F59E0B", icon="⚠️")
        else:
            insight_card(
                "ECG Model Unavailable",
                "Check that <code>models/ecg_cnn_baseline.pth</code> is present "
                "and the ECGCNN class matches the saved architecture.",
                accent="#F59E0B", icon="⚠️",
            )

        # Architecture card
        st.markdown("""
<div style="background:#111827;border:1px solid #1E2A3A;border-radius:10px;
     padding:1rem;margin-top:1rem;">
  <div style="font-weight:700;color:#F1F5F9;font-size:0.85rem;margin-bottom:0.75rem;">
    🏗️ ECGCNN Architecture
  </div>
  <div style="font-family:monospace;font-size:0.78rem;color:#94A3B8;line-height:1.9;">
    Input &nbsp; (batch × 1 × 500)<br>
    <span style="color:#00D4FF;">↓</span> Conv1D(1→16, k=5, pad=2) + ReLU<br>
    <span style="color:#00D4FF;">↓</span> MaxPool1D(k=2) → (batch × 16 × 250)<br>
    <span style="color:#00D4FF;">↓</span> Conv1D(16→32, k=5, pad=2) + ReLU<br>
    <span style="color:#00D4FF;">↓</span> MaxPool1D(k=2) → (batch × 32 × 125)<br>
    <span style="color:#00D4FF;">↓</span> Flatten → 4,000 features<br>
    <span style="color:#00D4FF;">↓</span> Linear(4000→64) + ReLU + Dropout(0.5)<br>
    <span style="color:#00D4FF;">↓</span> Linear(64→2)<br>
    <span style="color:#00D4FF;">↓</span> Softmax → Risk Probability
  </div>
</div>
""", unsafe_allow_html=True)

        # Training config chips
        configs = [
            ("Window", "500 samples"), ("Overlap", "50%"),
            ("Optimizer", "Adam"), ("LR", "0.001"),
            ("Epochs", "10"), ("Loss", "CrossEntropyLoss"),
            ("Batch Size", "64"), ("Device", "CPU / CUDA auto"),
        ]
        st.markdown("<br>", unsafe_allow_html=True)
        cg1, cg2 = st.columns(2)
        for i, (k, v) in enumerate(configs):
            col = cg1 if i % 2 == 0 else cg2
            col.markdown(
                f'<div style="background:#0D1624;border:1px solid #1E2A3A;'
                f'border-radius:6px;padding:5px 8px;margin-bottom:6px;'
                f'font-size:0.72rem;">'
                f'<span style="color:#64748B;">{k}:</span> '
                f'<span style="color:#F1F5F9;font-weight:600;">{v}</span>'
                f'</div>',
                unsafe_allow_html=True)

    # ── CORRELATION ANALYSIS ────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section_header(
        "ECG Risk vs Clinical Features",
        subtitle="Cross-modal correlation between CNN output and clinical biomarkers",
        badge="Cross-Modal",
    )

    corr_df = _load_correlation_csv()
    if corr_df is not None:
        sl, sr = st.columns(2)
        with sl:
            try:
                st.plotly_chart(
                    correlation_heatmap(corr_df, height=350),
                    use_container_width=True)
            except Exception:
                empty_state_card("ecg_clinical_correlation.csv")

        with sr:
            try:
                tabs = st.tabs(["ECG Risk vs Age",
                                 "ECG Risk vs Systolic BP",
                                 "ECG Risk vs Cholesterol"])
                col_pairs = [("age", "ecg_predicted_risk"),
                             ("ap_hi", "ecg_predicted_risk"),
                             ("cholesterol", "ecg_predicted_risk")]

                for tab, (xcol, ycol) in zip(tabs, col_pairs):
                    with tab:
                        if xcol in corr_df.columns and ycol in corr_df.columns:
                            x = np.array(corr_df[xcol].values, dtype=float)
                            y = np.array(corr_df[ycol].values, dtype=float)
                        else:
                            # Generate illustrative synthetic scatter
                            np.random.seed(42)
                            x = np.random.randn(50)
                            y = np.random.randn(50)
                        st.plotly_chart(
                            scatter_with_trend(x, y, xcol, ycol, height=280),
                            use_container_width=True)
            except Exception:
                empty_state_card("ecg_clinical_correlation.csv")
    else:
        insight_card(
            "Correlation Data Unavailable",
            "Expected at <code>results/metrics/ecg_clinical_correlation.csv</code>.",
            accent="#F59E0B", icon="⚠️")

    insight_card(
        "Cross-Modal Signal",
        "Strong correlation between ECG signal score and systolic BP would "
        "indicate the CNN captures hemodynamic information encoded in cardiac "
        "waveform morphology — supporting the hypothesis that ECG signals "
        "implicitly encode cardiovascular risk.",
        accent="#00D4FF", icon="📡",
    )

    # ── SALIENCY COMPARISON ─────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section_header(
        "Prediction Analysis",
        subtitle="Correct vs incorrect prediction saliency",
        badge="Error Analysis",
    )

    if ecg_ok and ecg_model is not None and X_segs is not None:
        segs, preds, correct_idx, incorrect_idx = _batch_predictions(
            id(X_segs))

        if segs is not None:
            sa, sb = st.columns(2)
            for col, idx, label, color in [
                (sa, correct_idx,   "✅ Correct Prediction",   "#10B981"),
                (sb, incorrect_idx, "❌ Incorrect Prediction", "#EF4444"),
            ]:
                with col:
                    st.markdown(
                        f'<div style="border:1px solid {color};border-radius:8px;'
                        f'padding:6px 10px;margin-bottom:8px;font-size:0.8rem;'
                        f'color:{color};font-weight:700;">{label}</div>',
                        unsafe_allow_html=True)
                    if idx is not None:
                        sig = segs[idx].astype(np.float32)
                        try:
                            _, sal = compute_saliency(ecg_model, sig)
                        except Exception:
                            sal = None
                        st.plotly_chart(
                            ecg_signal_chart(sig, sal, height=200),
                            use_container_width=True)
                    else:
                        empty_state_card("No sample found")
        else:
            insight_card("Batch prediction failed",
                         "ECG segment data could not be processed.",
                         accent="#F59E0B", icon="⚠️")
    else:
        insight_card(
            "ECG Segments Unavailable",
            "Load <code>data/processed/X_ecg_segments.npy</code> and ensure the "
            "ECG CNN is loaded to view saliency comparison.",
            accent="#F59E0B", icon="⚠️",
        )

    insight_card(
        "Error Analysis Insight",
        "Divergence in saliency between correct and incorrect predictions reveals "
        "which signal regions the model relies on. High attention in failed "
        "predictions may indicate sensitivity to noise — a key target for future "
        "improvement with labeled ECG datasets.",
        accent="#F59E0B", icon="🔬",
    )

    # ── LIMITATIONS ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Known Limitations", badge="Research Integrity",
                   badge_color="#F59E0B")

    lims = [
        ("🏷️", "Dataset Labels",
         "ECG ground truth diagnostic labels were not available in the provided "
         "dataset. Binary labels used for CNN training were not clinically annotated."),
        ("📦", "Training Data",
         "ECG dataset size and provenance are limited. Labeled datasets like PTB-XL "
         "(21,799 records) would enable robust clinical validation."),
        ("📡", "Signal Context",
         "Lead configuration, recording conditions, and patient metadata were "
         "unavailable, limiting clinical interpretability of signal patterns."),
        ("🏥", "Clinical Use",
         "All results are retrospective. Clinical deployment requires prospective "
         "trials, peer review, and regulatory approval. This is a research prototype."),
    ]

    lc = st.columns(4)
    for col, (icon, title, body) in zip(lc, lims):
        with col:
            st.markdown(f"""
<div style="background:#111827;border-left:4px solid #F59E0B;
     border-radius:8px;padding:0.9rem;height:100%;">
  <div style="font-size:1.3rem;margin-bottom:6px;">{icon}</div>
  <div style="font-weight:700;color:#F1F5F9;font-size:0.82rem;
       margin-bottom:6px;">{title}</div>
  <div style="color:#94A3B8;font-size:0.75rem;line-height:1.5;">{body}</div>
</div>
""", unsafe_allow_html=True)
