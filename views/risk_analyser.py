"""
pages/risk_analyser.py
Page 2 — Live patient risk inference with SHAP explanation.
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
    section_header, metric_card, insight_card, risk_badge,
    bp_badge, bmi_display, compute_bmi, risk_gauge
)
from components.charts import shap_bar_chart, roc_curve_chart, feature_importance_chart
from core.data_loader import preprocess_input, FEATURES, load_cardio_data
from core.inference import run_inference, compute_shap
from core.model_loader import load_all_models


@st.cache_data(show_spinner=False)
def _get_test_split():
    """Build a test split for ROC curve rendering."""
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        import os
        paths = [
            os.path.join(_ROOT, 'data', 'raw', 'cardio_base.csv'),
            os.path.join(_ROOT, 'cardio_base.csv'),
        ]
        for p in paths:
            if os.path.exists(p):
                df = pd.read_csv(p, sep=';')
                df = df[(df['ap_hi'] >= 70) & (df['ap_hi'] <= 250)]
                df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 150)]
                X = df[FEATURES]
                y = df['cardio']
                _, X_test, _, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y)
                return X_test, y_test
    except Exception:
        pass
    return None, None


def render():
    models = load_all_models()
    rf  = models["rf"]
    lr  = models["lr"]
    sc  = models["scaler"]
    ok  = models["clinical_ok"]

    section_header(
        "Risk Analyser",
        subtitle="Configure patient profile → real-time explainable cardiovascular risk assessment",
        badge="Live Inference",
    )

    if not ok:
        insight_card(
            "Models Unavailable",
            "Clinical models could not be loaded or trained. "
            "Ensure <code>data/raw/cardio_base.csv</code> is accessible.",
            accent="#F59E0B", icon="⚠️",
        )
        return

    # ── TWO COLUMNS ────────────────────────────────────────────────────────
    left, right = st.columns([55, 45])

    # ── LEFT — Patient Input ────────────────────────────────────────────────
    with left:
        section_header("Patient Profile", badge="Interactive")

        with st.expander("👤 Demographics", expanded=True):
            age_years = st.slider("Age (years)", 29, 64, 45)
            gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
            c1, c2 = st.columns(2)
            height = c1.slider("Height (cm)", 140, 200, 168)
            weight = c2.slider("Weight (kg)", 40, 150, 70)
            bmi = compute_bmi(height, weight)
            st.markdown(
                f"**BMI:** {bmi:.1f} &nbsp; {bmi_display(height, weight)}",
                unsafe_allow_html=True)

        with st.expander("🩺 Blood Pressure & Labs", expanded=True):
            ap_hi = st.slider("Systolic BP (mmHg)", 70, 250, 120,
                              help="The upper number in your BP reading")
            ap_lo = st.slider("Diastolic BP (mmHg)", 40, 150, 80)
            st.markdown(
                f"**BP Category:** {bp_badge(ap_hi)}",
                unsafe_allow_html=True)
            cholesterol = st.select_slider(
                "Cholesterol Level",
                options=["Normal", "Above Normal", "Well Above Normal"])
            gluc = st.select_slider(
                "Glucose Level",
                options=["Normal", "Above Normal", "Well Above Normal"])

        with st.expander("🏃 Lifestyle Factors", expanded=True):
            lc1, lc2, lc3 = st.columns(3)
            smoke  = lc1.toggle("Smoker",            value=False)
            alco   = lc2.toggle("Alcohol Use",       value=False)
            active = lc3.toggle("Physically Active", value=True)

            risk_count = sum([smoke, alco, not active])
            if risk_count == 0:
                chip_bg, chip_text = "#10B981", "No lifestyle risk factors"
            elif risk_count == 1:
                chip_bg, chip_text = "#F59E0B", "1 lifestyle risk factor"
            else:
                chip_bg, chip_text = "#EF4444", f"{risk_count} lifestyle risk factors"
            st.markdown(
                f'<span style="background:{chip_bg};color:#0A0E1A;font-weight:700;'
                f'font-size:0.72rem;border-radius:20px;padding:3px 12px;">'
                f'{chip_text}</span>',
                unsafe_allow_html=True)

        run_clicked = st.button("🔍 Analyse Risk", key="run_inference")

        if st.button("↺ Reset to defaults", key="reset_btn"):
            for k in ["risk_score", "shap_values", "last_features"]:
                st.session_state[k] = None
            st.rerun()

    # ── RIGHT — Results Panel ──────────────────────────────────────────────
    with right:
        if run_clicked or st.session_state.get("risk_score") is not None:
            if run_clicked:
                with st.spinner("Analysing cardiac risk profile..."):
                    feat_vec = preprocess_input(
                        age_years, gender, height, weight,
                        ap_hi, ap_lo, cholesterol, gluc,
                        smoke, alco, active)
                    model_choice = st.session_state.get("model_choice",
                                                        "Random Forest")
                    active_model = rf if model_choice == "Random Forest" else lr
                    score = run_inference(active_model, sc, feat_vec, model_choice)
                    st.session_state["risk_score"] = score
                    st.session_state["last_features"] = feat_vec

            score = st.session_state["risk_score"]
            model_choice = st.session_state.get("model_choice", "Random Forest")
            feat_vec = st.session_state.get("last_features") or preprocess_input(
                age_years, gender, height, weight,
                ap_hi, ap_lo, cholesterol, gluc,
                smoke, alco, active)

            # Gauge
            st.plotly_chart(risk_gauge(score * 100), use_container_width=True)
            st.markdown(
                f'<div style="text-align:center;margin:-8px 0 12px 0;">'
                f'{risk_badge(score * 100)}</div>',
                unsafe_allow_html=True)

            # 3 Metric cards
            conf_label = "High" if (score > 0.7 or score < 0.3) else "Moderate"
            conf_color = "#10B981" if conf_label == "High" else "#F59E0B"
            mc1, mc2, mc3 = st.columns(3)
            with mc1: metric_card("Risk Score", f"{score*100:.1f}%", accent="#00D4FF")
            with mc2: metric_card("Model", model_choice, accent="#7C3AED")
            with mc3: metric_card("Confidence", conf_label, accent=conf_color)

            # SHAP Section
            section_header("Why This Score?", badge="SHAP")
            with st.spinner("Computing feature attributions..."):
                active_model = rf if model_choice == "Random Forest" else lr
                shap_vals, feat_names = compute_shap(
                    active_model, sc, feat_vec, model_choice)

            st.plotly_chart(
                shap_bar_chart(shap_vals, feat_names, height=300),
                use_container_width=True)

            # Auto-generated interpretation
            sorted_idx = np.argsort(shap_vals)[::-1]
            pos_feats = [feat_names[i] for i in sorted_idx if shap_vals[i] >= 0][:2]
            neg_feats = [feat_names[i] for i in sorted_idx[::-1] if shap_vals[i] < 0][:1]
            risk_level = ("LOW" if score < 0.33 else
                          "MODERATE" if score < 0.66 else "HIGH")

            pos_str = (f"{pos_feats[0]} and {pos_feats[1]}"
                       if len(pos_feats) >= 2 else
                       (pos_feats[0] if pos_feats else "multiple factors"))
            neg_str = neg_feats[0] if neg_feats else "no factor"

            insight_card(
                "Clinical Interpretation",
                f"The strongest risk drivers for this patient are "
                f"<b>{pos_str}</b>. "
                f"<b>{neg_str}</b> is acting as a protective factor. "
                f"The <b>{model_choice}</b> model assessed this patient as "
                f"<b>{risk_level} RISK</b> with <b>{score*100:.1f}%</b> probability.",
                accent="#00D4FF",
                icon="🩺",
            )

            # Clinical Flag Chips
            flags = []
            if ap_hi >= 140:
                flags.append(("🔴", "#EF4444", "Hypertensive Range — Stage 2"))
            elif 130 <= ap_hi < 140:
                flags.append(("🟡", "#F59E0B", "Elevated Blood Pressure"))
            if cholesterol == "Well Above Normal":
                flags.append(("🔴", "#EF4444", "High Cholesterol"))
            if smoke and alco:
                flags.append(("🟡", "#F59E0B", "Combined Lifestyle Risk"))
            if compute_bmi(height, weight) >= 30:
                flags.append(("🟡", "#F59E0B", "Obese BMI Range"))

            if flags:
                html = '<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;">'
                for emoji, color, label in flags:
                    html += (
                        f'<span style="background:rgba({_hex_to_rgb(color)},0.15);'
                        f'color:{color};border:1px solid {color};border-radius:20px;'
                        f'padding:3px 10px;font-size:0.72rem;font-weight:600;">'
                        f'{emoji} {label}</span>')
                html += '</div>'
                st.markdown(html, unsafe_allow_html=True)

        else:
            # Placeholder
            st.markdown("""
<div style="background:#111827;border:2px dashed #1E3A4A;border-radius:12px;
     padding:3rem 2rem;text-align:center;margin-top:1rem;">
  <div style="font-size:3.5rem;margin-bottom:1rem;">🫀</div>
  <div style="color:#94A3B8;font-size:0.95rem;line-height:1.6;max-width:280px;margin:0 auto;">
    Configure the patient profile and click<br>
    <strong style="color:#F1F5F9;">Analyse Risk</strong> to generate a real-time<br>
    cardiovascular risk assessment.
  </div>
  <div style="color:#64748B;font-size:0.75rem;margin-top:1rem;">
    Powered by Random Forest · SHAP Explainability
  </div>
</div>
""", unsafe_allow_html=True)

    # ── MODEL BENCHMARKS ───────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section_header(
        "Model Benchmarks",
        subtitle="Trained and validated on 70,000 de-identified patient records",
        badge="Validated",
    )

    bl, br = st.columns(2)
    X_test, y_test = _get_test_split()

    with bl:
        scaler_safe = sc if ok else None
        fig_roc = roc_curve_chart(rf, lr, scaler_safe, X_test, y_test, height=350)
        st.plotly_chart(fig_roc, use_container_width=True)

    with br:
        # Custom HTML performance table
        st.markdown("""
<table style="width:100%;border-collapse:collapse;font-size:0.85rem;">
  <thead>
    <tr>
      <th style="background:#0A0E1A;color:#94A3B8;padding:0.6rem 0.8rem;
           text-align:left;border-bottom:1px solid #1E2A3A;">Metric</th>
      <th style="background:#0A0E1A;color:#94A3B8;padding:0.6rem 0.8rem;
           text-align:center;border-bottom:1px solid #1E2A3A;">Logistic Regression</th>
      <th style="background:#003A52;color:#00D4FF;padding:0.6rem 0.8rem;
           text-align:center;border-bottom:1px solid #1E2A3A;">Random Forest</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:#111827;">
      <td style="padding:0.55rem 0.8rem;color:#94A3B8;border-bottom:1px solid #1E2A3A;">AUC</td>
      <td style="padding:0.55rem 0.8rem;color:#F1F5F9;text-align:center;border-bottom:1px solid #1E2A3A;">0.778</td>
      <td style="padding:0.55rem 0.8rem;color:#F1F5F9;text-align:center;font-weight:700;border-bottom:1px solid #1E2A3A;">
        ✓ 0.784 <span style="background:#00D4FF;color:#0A0E1A;font-size:0.6rem;border-radius:10px;padding:1px 6px;margin-left:4px;">Best</span>
      </td>
    </tr>
    <tr style="background:#0F1724;">
      <td style="padding:0.55rem 0.8rem;color:#94A3B8;border-bottom:1px solid #1E2A3A;">Accuracy</td>
      <td style="padding:0.55rem 0.8rem;color:#F1F5F9;text-align:center;border-bottom:1px solid #1E2A3A;">71.4%</td>
      <td style="padding:0.55rem 0.8rem;color:#F1F5F9;text-align:center;font-weight:700;border-bottom:1px solid #1E2A3A;">
        ✓ 72.2% <span style="background:#00D4FF;color:#0A0E1A;font-size:0.6rem;border-radius:10px;padding:1px 6px;margin-left:4px;">Best</span>
      </td>
    </tr>
    <tr style="background:#111827;">
      <td style="padding:0.55rem 0.8rem;color:#94A3B8;">Precision</td>
      <td style="padding:0.55rem 0.8rem;color:#F1F5F9;text-align:center;">71.2%</td>
      <td style="padding:0.55rem 0.8rem;color:#F1F5F9;text-align:center;font-weight:700;">✓ 73.1%</td>
    </tr>
  </tbody>
</table>
""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        insight_card(
            "Key Finding",
            "Systolic blood pressure and age are the dominant predictors — "
            "consistent with established clinical literature on cardiovascular "
            "risk stratification. Both models exceed AUC 0.77, indicating "
            "clinically meaningful discrimination.",
            accent="#00D4FF", icon="📊",
        )

    # Feature Importance
    st.markdown("<br>", unsafe_allow_html=True)
    fi_path = os.path.join(_ROOT, 'results', 'metrics',
                           'clinical_feature_importance.csv')
    if os.path.exists(fi_path):
        try:
            fi_df = pd.read_csv(fi_path)
            # Exclude 'id' column from importance display
            fi_df = fi_df[fi_df['feature'] != 'id'] if 'feature' in fi_df.columns else fi_df
            st.plotly_chart(
                feature_importance_chart(fi_df, 300),
                use_container_width=True)
        except Exception:
            from components.ui_components import empty_state_card
            empty_state_card("clinical_feature_importance.csv")
    else:
        from components.ui_components import empty_state_card
        empty_state_card("results/metrics/clinical_feature_importance.csv")


def _hex_to_rgb(hex_color: str) -> str:
    """Convert #RRGGBB to 'R,G,B' string for CSS rgba()."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"
