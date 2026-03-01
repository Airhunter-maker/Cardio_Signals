"""Risk Analyser — Page 2."""
import streamlit as st
import numpy as np


def render_risk_analyser():
    # All UI component imports at the top of the function scope
    from components.ui_components import (
        metric_card, section_header, insight_card,
        risk_gauge, risk_badge, bmi_display)
    from core.model_loader import load_all_models
    from core.inference import run_general_logic
    from core.validators import (
        categorize_bp, calculate_bmi, validate_inputs, level_to_int)

    models = load_all_models()

    section_header(
        "Risk Analyser",
        subtitle="Enter patient clinical data for AI-powered cardiovascular risk assessment",
        badge="Live Analysis"
    )

    if not st.session_state.get("clinical_model_loaded"):
        insight_card(
            "Models Initialising",
            "Clinical models are loading. If this persists, "
            "ensure model files or cardio_base.csv are present in the project root.",
            accent="#F59E0B", icon="⚠️"
        )

    # ── TOOLTIP STYLE ─────────────────────────────────────────────────
    st.markdown("""
    <style>
    .field-hint {
        font-size: 0.72rem;
        color: #475569;
        margin-top: -0.3rem;
        margin-bottom: 0.4rem;
        line-height: 1.5;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── PATIENT DATA INPUT ────────────────────────────────────────────
    st.markdown("### Patient Profile")

    r1c1, r1c2 = st.columns(2)

    with r1c1:
        with st.expander("👤 Demographics", expanded=True):
            age_years = st.slider("Age (years)", 18, 80, 45)
            st.markdown("<p class='field-hint'>ℹ️ Age is one of the strongest cardiovascular risk factors</p>",
                        unsafe_allow_html=True)
            gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
            c1, c2 = st.columns(2)
            height = c1.slider("Height (cm)", 140, 210, 168)
            weight = c2.slider("Weight (kg)", 30, 200, 70)
            bmi_val = calculate_bmi(height, weight)
            st.markdown(
                f"<div style='font-size:0.8rem; color:#64748B; "
                f"margin-top:0.3rem;'>BMI: "
                f"{bmi_display(height, weight)}</div>",
                unsafe_allow_html=True
            )

    with r1c2:
        with st.expander("🩺 Vitals & Lab Results", expanded=True):
            ap_hi = st.slider("Systolic Blood Pressure (mmHg)", 70, 250, 120)
            st.markdown("<p class='field-hint'>ℹ️ Systolic BP: the pressure when your heart beats (upper number)</p>",
                        unsafe_allow_html=True)
            ap_lo = st.slider("Diastolic Blood Pressure (mmHg)", 40, 150, 80)
            st.markdown("<p class='field-hint'>ℹ️ Diastolic BP: the pressure when your heart rests (lower number)</p>",
                        unsafe_allow_html=True)
            bp_label, bp_color = categorize_bp(ap_hi, ap_lo)
            st.markdown(
                f"<div style='font-size:0.8rem; color:#64748B; "
                f"margin-bottom:0.6rem;'>Blood Pressure Category: "
                f"<span style='color:{bp_color}; font-weight:700;'>"
                f"{bp_label}</span></div>",
                unsafe_allow_html=True
            )
            cholesterol = st.select_slider(
                "Cholesterol Level",
                options=["Low", "Normal", "High"],
                value="Normal",
            )
            st.markdown("<p class='field-hint'>ℹ️ Total cholesterol level from your blood test</p>",
                        unsafe_allow_html=True)
            gluc = st.select_slider(
                "Glucose Level",
                options=["Low", "Normal", "High"],
                value="Normal",
            )
            st.markdown("<p class='field-hint'>ℹ️ Fasting blood glucose level from blood test</p>",
                        unsafe_allow_html=True)

    r2c1, r2c2 = st.columns(2)

    with r2c1:
        with st.expander("🏃 Lifestyle Factors", expanded=True):
            c1, c2, c3 = st.columns(3)
            smoke  = c1.toggle("Smoker", value=False)
            alco   = c2.toggle("Alcohol Use", value=False)
            active = c3.toggle("Physically Active", value=True)
            st.markdown(
                "<p class='field-hint'>ℹ️ Smoker: current or recent tobacco use &nbsp;|&nbsp; "
                "Active: 30+ min of activity most days</p>",
                unsafe_allow_html=True
            )

    with r2c2:
        st.write("")
        st.write("")
        btn = st.button(
            "🔍  Analyse Risk",
            key="run_inference",
            use_container_width=True,
            type="primary"
        )
        if st.button("↺  Clear Form", key="reset_btn", use_container_width=True):
            for k in ["risk_score", "shap_values", "last_features", "analyse_clicked"]:
                st.session_state[k] = None
            st.session_state["analyse_clicked"] = False
            st.rerun()

    # ── VALIDATION & RESULTS ──────────────────────────────────────────
    if btn or st.session_state.get("analyse_clicked"):
        st.session_state["analyse_clicked"] = True

        is_valid, err_msg = validate_inputs(age_years, height, weight, ap_hi, ap_lo)

        if not is_valid:
            st.error(f"⚠️ Please check your inputs: {err_msg}")
        else:
            st.markdown("---")
            res_col1, res_col2 = st.columns([1, 1.3])

            with res_col1:
                with st.spinner("🔬 Analysing risk factors..."):
                    try:
                        score = run_general_logic(
                            age_years, gender, height, weight,
                            ap_hi, ap_lo, cholesterol, gluc,
                            smoke, alco, active
                        )
                        st.session_state["risk_score"] = score
                    except Exception:
                        score = 0.5
                        st.session_state["risk_score"] = score

                fig = risk_gauge(score * 100)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(
                    f"<div style='text-align:center; margin:-0.5rem 0 1rem;'>"
                    f"{risk_badge(score * 100)}</div>",
                    unsafe_allow_html=True
                )

                # Save Results
                _save_results_btn(
                    age_years, gender, height, weight, ap_hi, ap_lo,
                    cholesterol, gluc, smoke, alco, active, bmi_val, score, bp_label
                )

            with res_col2:
                risk_level = (
                    "LOW" if score < 0.33
                    else "MODERATE" if score < 0.66
                    else "HIGH"
                )
                risk_color = {"LOW": "#10B981", "MODERATE": "#F59E0B", "HIGH": "#EF4444"}[risk_level]

                c1, c2 = st.columns(2)
                with c1:
                    metric_card("Risk Score", f"{score*100:.1f}%", accent="#00D4FF")
                with c2:
                    metric_card("Risk Level", risk_level, accent=risk_color)

                # Pass insight_card into helpers to avoid NameError
                _render_clinical_interpretation(
                    insight_card, risk_level, score, ap_hi, ap_lo,
                    bp_label, bp_color, cholesterol, smoke, alco, bmi_val
                )
                _render_contributing_factors(
                    age_years, bmi_val, ap_hi, ap_lo,
                    cholesterol, gluc, smoke, alco, active
                )


def _save_results_btn(age_years, gender, height, weight, ap_hi, ap_lo,
                      cholesterol, gluc, smoke, alco, active, bmi_val,
                      score, bp_label):
    """Render a download button with formatted text summary."""
    risk_level = "LOW" if score < 0.33 else "MODERATE" if score < 0.66 else "HIGH"
    summary = (
        "CardioSignals — Risk Assessment Report\n"
        "=========================================\n\n"
        "PATIENT PROFILE\n"
        f"  Age:               {age_years} years\n"
        f"  Gender:            {gender}\n"
        f"  Height:            {height} cm\n"
        f"  Weight:            {weight} kg\n"
        f"  BMI:               {bmi_val:.1f}\n\n"
        "CLINICAL VITALS\n"
        f"  Systolic BP:       {ap_hi} mmHg\n"
        f"  Diastolic BP:      {ap_lo} mmHg\n"
        f"  BP Category:       {bp_label}\n"
        f"  Cholesterol:       {cholesterol}\n"
        f"  Glucose:           {gluc}\n\n"
        "LIFESTYLE\n"
        f"  Smoker:            {'Yes' if smoke else 'No'}\n"
        f"  Alcohol Use:       {'Yes' if alco else 'No'}\n"
        f"  Physically Active: {'Yes' if active else 'No'}\n\n"
        "RISK ASSESSMENT\n"
        f"  Risk Score:        {score*100:.1f}%\n"
        f"  Risk Level:        {risk_level}\n\n"
        "=========================================\n"
        "For research and educational purposes only.\n"
        "Not a substitute for professional medical advice.\n"
    )
    st.download_button(
        label="💾  Save Results (.txt)",
        data=summary,
        file_name="cardiosignals_risk_report.txt",
        mime="text/plain",
        use_container_width=True,
        key="save_results_btn"
    )


def _render_clinical_interpretation(insight_card_fn, risk_level, score,
                                    ap_hi, ap_lo, bp_label, bp_color,
                                    cholesterol, smoke, alco, bmi_val):
    """Render the clinical interpretation card."""
    desc_map = {
        "LOW": (
            "Your clinical profile suggests a <strong>lower likelihood</strong> "
            "of cardiovascular disease based on the provided data. "
            "Maintaining a healthy lifestyle is key to keeping risk low."
        ),
        "MODERATE": (
            "Your profile indicates <strong>moderate cardiovascular risk</strong>. "
            "Some clinical factors warrant attention. "
            "Lifestyle modifications can significantly reduce risk."
        ),
        "HIGH": (
            "Your profile indicates <strong>elevated cardiovascular risk</strong>. "
            "Several risk factors are present. Clinical review and intervention "
            "are strongly recommended."
        ),
    }
    insight_card_fn(
        "Clinical Interpretation",
        desc_map[risk_level],
        accent="#00D4FF"
    )

    # Clinical flags
    flags = []
    if ap_hi > 180 or ap_lo > 120:
        flags.append(("🔴", "#991B1B", "Hypertensive Crisis Range"))
    elif ap_hi >= 140 or ap_lo >= 90:
        flags.append(("🔴", "#EF4444", "Hypertensive — Stage 2"))
    elif ap_hi >= 130 or ap_lo >= 80:
        flags.append(("🟡", "#F59E0B", "Elevated Blood Pressure"))
    if cholesterol == "Well Above Normal":
        flags.append(("🔴", "#EF4444", "High Cholesterol"))
    elif cholesterol == "Above Normal":
        flags.append(("🟡", "#F59E0B", "Above-Normal Cholesterol"))
    if smoke and alco:
        flags.append(("🟡", "#F59E0B", "Combined Lifestyle Risk"))
    elif smoke:
        flags.append(("🟡", "#F59E0B", "Active Smoker"))
    if bmi_val >= 30:
        flags.append(("🟡", "#F59E0B", "Obese BMI"))
    elif bmi_val >= 25:
        flags.append(("🔵", "#64748B", "Overweight BMI"))

    if flags:
        chips = " ".join([
            f"<span style='background:{c}20; color:{c}; "
            f"border:1px solid {c}40; font-size:0.72rem; "
            f"font-weight:600; padding:3px 10px; "
            f"border-radius:6px; margin:2px; "
            f"display:inline-block;'>{e} {t}</span>"
            for e, c, t in flags
        ])
        st.markdown(
            f"<div style='margin-top:0.5rem;'>{chips}</div>",
            unsafe_allow_html=True
        )


def _render_contributing_factors(age_years, bmi_val, ap_hi, ap_lo,
                                  cholesterol, gluc, smoke, alco, active):
    """Render top 3 contributing factors with simple bar indicators."""
    from core.validators import level_to_int

    factors = []

    # Age
    if age_years > 60:
        factors.append(("Age", f"{age_years} years", 0.9))
    elif age_years > 50:
        factors.append(("Age", f"{age_years} years", 0.6))
    elif age_years > 40:
        factors.append(("Age", f"{age_years} years", 0.35))

    # BP
    if ap_hi > 180 or ap_lo > 120:
        factors.append(("Blood Pressure", f"{ap_hi}/{ap_lo} mmHg", 0.95))
    elif ap_hi >= 140 or ap_lo >= 90:
        factors.append(("Blood Pressure", f"{ap_hi}/{ap_lo} mmHg", 0.80))
    elif ap_hi >= 130 or ap_lo >= 80:
        factors.append(("Blood Pressure", f"{ap_hi}/{ap_lo} mmHg", 0.50))

    # Cholesterol
    chol_int = level_to_int(cholesterol, 1)
    if chol_int >= 3:
        factors.append(("Cholesterol", cholesterol, 0.75))
    elif chol_int == 2:
        factors.append(("Cholesterol", cholesterol, 0.40))

    # BMI
    if bmi_val >= 30:
        factors.append(("BMI", f"{bmi_val:.1f} — Obese", 0.65))
    elif bmi_val >= 25:
        factors.append(("BMI", f"{bmi_val:.1f} — Overweight", 0.30))

    # Lifestyle
    if smoke:
        factors.append(("Smoking", "Active Smoker", 0.60))
    if not active:
        factors.append(("Physical Inactivity", "Not Active", 0.35))

    # Top 3 by contribution weight
    factors.sort(key=lambda x: x[2], reverse=True)
    top3 = factors[:3]

    if top3:
        st.markdown("""
        <div style='margin-top:0.8rem; font-size:0.82rem; font-weight:700;
                    color:#F1F5F9; margin-bottom:0.5rem;'>
            🔑 Key Contributing Factors
        </div>
        """, unsafe_allow_html=True)

        for name, display_val, contrib in top3:
            bar_color = (
                "#EF4444" if contrib > 0.7 else
                "#F59E0B" if contrib > 0.4 else
                "#64748B"
            )
            pct = int(contrib * 100)
            st.markdown(f"""
            <div style='margin-bottom:0.65rem;'>
                <div style='display:flex; justify-content:space-between;
                            margin-bottom:4px;'>
                    <span style='font-size:0.78rem; color:#94A3B8; font-weight:600;'>{name}</span>
                    <span style='font-size:0.73rem; color:{bar_color}; font-weight:700;'>{display_val}</span>
                </div>
                <div style='background:#1E2A3A; border-radius:4px; height:6px;'>
                    <div style='background:{bar_color}; border-radius:4px;
                                height:6px; width:{pct}%;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
