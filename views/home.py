"""Home — Landing page for CardioSignals."""
import streamlit as st


def render_home():
    # Inject home-specific styles + animations
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&display=swap');

    /* ── HERO SECTION ─────────────────────────────────── */
    .hero-section {
        min-height: 92vh;
        display: flex;
        align-items: center;
        padding: 2rem 0 1rem 0;
    }
    .hero-left {
        padding-right: 3rem;
    }
    .hero-tag {
        display: inline-block;
        background: linear-gradient(135deg, #00D4FF20, #7C3AED20);
        border: 1px solid #00D4FF40;
        color: #00D4FF;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        padding: 5px 16px;
        border-radius: 20px;
        margin-bottom: 1.5rem;
    }
    .hero-title {
        font-size: clamp(2.8rem, 5vw, 4.5rem);
        font-weight: 900;
        line-height: 1.05;
        color: #F1F5F9;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.03em;
        font-family: 'Inter', sans-serif;
    }
    .hero-title span {
        background: linear-gradient(135deg, #00D4FF, #7C3AED);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        font-weight: 600;
        color: #94A3B8;
        margin: 0 0 1.2rem 0;
        letter-spacing: -0.01em;
    }
    .hero-desc {
        font-size: 0.95rem;
        color: #64748B;
        line-height: 1.8;
        max-width: 520px;
        margin-bottom: 2rem;
    }
    .feature-chips {
        display: flex;
        flex-direction: column;
        gap: 0.7rem;
        margin-bottom: 2.2rem;
    }
    .feature-chip {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        background: #111827;
        border: 1px solid #1E2A3A;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        transition: all 0.2s;
    }
    .feature-chip:hover {
        border-color: #00D4FF40;
        background: #0D1F30;
        transform: translateX(4px);
    }
    .chip-icon {
        font-size: 1.25rem;
        flex-shrink: 0;
    }
    .chip-text strong {
        display: block;
        color: #F1F5F9;
        font-size: 0.85rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
    }
    .chip-text span {
        color: #64748B;
        font-size: 0.75rem;
        line-height: 1.4;
    }

    /* ── CTA BUTTONS ──────────────────────────────────── */
    .cta-primary {
        display: inline-block;
        background: linear-gradient(135deg, #00D4FF, #0090B8);
        color: #0A0E1A !important;
        font-size: 1.05rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        padding: 0.85rem 2.2rem;
        border-radius: 12px;
        text-decoration: none !important;
        box-shadow: 0 4px 24px rgba(0, 212, 255, 0.35);
        transition: all 0.25s;
        cursor: pointer;
        border: none;
        margin-right: 1rem;
    }
    .cta-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.5);
        background: linear-gradient(135deg, #22E5FF, #00BAE0);
    }
    .cta-secondary {
        display: inline-block;
        color: #94A3B8 !important;
        font-size: 0.92rem;
        font-weight: 600;
        text-decoration: none !important;
        padding: 0.85rem 0;
        border-bottom: 1px solid transparent;
        cursor: pointer;
        transition: all 0.2s;
    }
    .cta-secondary:hover {
        color: #00D4FF !important;
        border-bottom-color: #00D4FF;
    }

    /* ── ANIMATED HEART + ECG VISUAL ─────────────────── */
    .heart-visual {
        position: relative;
        width: 100%;
        min-height: 480px;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
    }
    .glow-orb {
        position: absolute;
        width: 340px;
        height: 340px;
        border-radius: 50%;
        background: radial-gradient(circle, #00D4FF08, #7C3AED06, transparent 70%);
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        animation: pulse-orb 3s ease-in-out infinite;
    }
    @keyframes pulse-orb {
        0%, 100% { transform: translate(-50%, -50%) scale(1); opacity: 0.6; }
        50% { transform: translate(-50%, -50%) scale(1.12); opacity: 1; }
    }
    .heart-svg-wrap {
        position: relative;
        z-index: 2;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 1.2rem;
    }
    .heart-icon {
        font-size: 7rem;
        animation: heartbeat 1.2s ease-in-out infinite;
        filter: drop-shadow(0 0 24px rgba(239,68,68,0.5));
        line-height: 1;
    }
    @keyframes heartbeat {
        0%   { transform: scale(1); }
        14%  { transform: scale(1.12); }
        28%  { transform: scale(1); }
        42%  { transform: scale(1.06); }
        70%  { transform: scale(1); }
        100% { transform: scale(1); }
    }
    .ecg-container {
        width: 360px;
        height: 70px;
        position: relative;
        overflow: hidden;
    }
    .ecg-line {
        stroke: #00D4FF;
        stroke-width: 2.5;
        fill: none;
        animation: ecg-scroll 2.2s linear infinite;
    }
    .ecg-line-glow {
        stroke: #00D4FF;
        stroke-width: 6;
        fill: none;
        opacity: 0.18;
        animation: ecg-scroll 2.2s linear infinite;
    }
    @keyframes ecg-scroll {
        0%   { stroke-dashoffset: 0; }
        100% { stroke-dashoffset: -440; }
    }
    .ecg-badge {
        background: linear-gradient(135deg, #00D4FF15, #7C3AED15);
        border: 1px solid #00D4FF30;
        border-radius: 40px;
        padding: 0.4rem 1.2rem;
        display: flex;
        gap: 0.8rem;
        align-items: center;
    }
    .ecg-badge span {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .dot-live {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10B981;
        box-shadow: 0 0 8px #10B981;
        animation: blink 1.4s ease-in-out infinite;
        flex-shrink: 0;
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
    /* Ring decoration */
    .ring-deco {
        position: absolute;
        border-radius: 50%;
        border: 1px solid;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        animation: ring-spin 18s linear infinite;
    }
    .ring-1 {
        width: 300px; height: 300px;
        border-color: #00D4FF15;
    }
    .ring-2 {
        width: 400px; height: 400px;
        border-color: #7C3AED10;
        animation-direction: reverse;
        animation-duration: 25s;
    }

    /* ── FEATURES GRID ────────────────────────────────── */
    .features-section {
        padding: 3rem 0;
        border-top: 1px solid #1E2A3A;
    }
    .section-eyebrow {
        text-align: center;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #00D4FF;
        margin-bottom: 0.6rem;
    }
    .section-title {
        text-align: center;
        font-size: 1.8rem;
        font-weight: 800;
        color: #F1F5F9;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
    }
    .section-sub {
        text-align: center;
        color: #64748B;
        font-size: 0.9rem;
        margin-bottom: 2rem;
    }
    .feat-card {
        background: #111827;
        border: 1px solid #1E2A3A;
        border-radius: 16px;
        padding: 1.8rem;
        height: 100%;
        transition: all 0.25s;
    }
    .feat-card:hover {
        border-color: #00D4FF30;
        transform: translateY(-3px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 0 1px #00D4FF10;
    }
    .feat-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
        display: block;
    }
    .feat-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #F1F5F9;
        margin-bottom: 0.5rem;
    }
    .feat-desc {
        font-size: 0.82rem;
        color: #64748B;
        line-height: 1.65;
    }

    /* ── HOW IT WORKS ─────────────────────────────────── */
    .how-section {
        padding: 3rem 0;
        border-top: 1px solid #1E2A3A;
    }
    .step-card {
        background: #111827;
        border: 1px solid #1E2A3A;
        border-radius: 14px;
        padding: 2rem 1.5rem;
        text-align: center;
        position: relative;
        height: 100%;
    }
    .step-number {
        width: 40px; height: 40px;
        border-radius: 50%;
        background: linear-gradient(135deg, #00D4FF, #7C3AED);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.9rem;
        font-weight: 800;
        color: #0A0E1A;
        margin: 0 auto 1.2rem;
    }
    .step-icon-big {
        font-size: 2.2rem;
        margin-bottom: 0.8rem;
        display: block;
    }
    .step-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #F1F5F9;
        margin-bottom: 0.4rem;
    }
    .step-desc {
        font-size: 0.8rem;
        color: #64748B;
        line-height: 1.6;
    }
    .step-arrow {
        text-align: center;
        font-size: 1.5rem;
        color: #1E2A3A;
        padding-top: 3rem;
    }

    /* ── FOOTER CTA ───────────────────────────────────── */
    .footer-cta {
        padding: 3rem 0 1.5rem;
        border-top: 1px solid #1E2A3A;
        text-align: center;
    }
    .footer-cta-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: #F1F5F9;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .footer-cta-sub {
        color: #64748B;
        font-size: 0.88rem;
        margin-bottom: 1.8rem;
    }
    .disclaimer {
        font-size: 0.7rem;
        color: #475569;
        margin-top: 1.2rem;
        line-height: 1.6;
    }

    /* ── FADE IN ANIMATION ────────────────────────────── */
    .fade-in {
        animation: fadeInUp 0.6s ease both;
    }
    .fade-in-2 { animation-delay: 0.15s; }
    .fade-in-3 { animation-delay: 0.3s; }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)

    # ── HERO SECTION ──────────────────────────────────────────────────────
    col_left, col_right = st.columns([6, 4], gap="large")

    with col_left:
        st.markdown("""
        <div class="hero-left fade-in">
            <div class="hero-tag">🫀 Cardiovascular AI Platform</div>
            <h1 class="hero-title">Cardio<span>Signals</span></h1>
            <p class="hero-subtitle">AI-Powered Cardiovascular Risk Intelligence</p>
            <p class="hero-desc">
                Predict cardiovascular disease risk using advanced machine learning.
                Get instant risk assessments with explainable AI insights and
                deep learning–based ECG signal analysis — designed for clinicians
                and researchers who need reliable, actionable cardiovascular insights.
            </p>
            <div class="feature-chips">
                <div class="feature-chip">
                    <span class="chip-icon">🎯</span>
                    <div class="chip-text">
                        <strong>Real-Time Risk Assessment</strong>
                        <span>Instant cardiovascular risk prediction from clinical data</span>
                    </div>
                </div>
                <div class="feature-chip">
                    <span class="chip-icon">🧠</span>
                    <div class="chip-text">
                        <strong>Explainable AI</strong>
                        <span>Understand which factors drive your individual risk score</span>
                    </div>
                </div>
                <div class="feature-chip">
                    <span class="chip-icon">📊</span>
                    <div class="chip-text">
                        <strong>ECG Signal Analysis</strong>
                        <span>Deep learning-powered cardiac waveform interpretation</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Navigation buttons — use session state to trigger page switch
        btn_risk = st.button(
            "🔍  Start Risk Analysis →",
            key="home_cta_risk",
            use_container_width=False,
            type="primary"
        )
        if btn_risk:
            st.session_state["current_page"] = "🔬  Risk Analyser"
            st.rerun()

        st.markdown("<div style='margin-top:0.6rem;'></div>",
                    unsafe_allow_html=True)

        btn_ecg = st.button(
            "📡  View ECG Signals →",
            key="home_cta_ecg",
            use_container_width=False
        )
        if btn_ecg:
            st.session_state["current_page"] = "📡  ECG Explorer"
            st.rerun()

    with col_right:
        # Animated heart + ECG waveform visual
        st.markdown("""
        <div class="heart-visual fade-in fade-in-2">
            <div class="ring-deco ring-1"></div>
            <div class="ring-deco ring-2"></div>
            <div class="glow-orb"></div>
            <div class="heart-svg-wrap">
                <div class="heart-icon">🫀</div>
                <!-- ECG Waveform SVG -->
                <div class="ecg-container">
                    <svg viewBox="0 0 440 70" xmlns="http://www.w3.org/2000/svg"
                         width="100%" height="100%">
                        <defs>
                            <filter id="glow">
                                <feGaussianBlur stdDeviation="3" result="blur"/>
                                <feMerge>
                                    <feMergeNode in="blur"/>
                                    <feMergeNode in="SourceGraphic"/>
                                </feMerge>
                            </filter>
                        </defs>
                        <!-- Glow layer -->
                        <path class="ecg-line-glow"
                              stroke-dasharray="440"
                              stroke-dashoffset="0"
                              filter="url(#glow)"
                              d="M0,35 L30,35 L38,35 L42,20 L46,55 L50,10 L54,60 L58,35 L66,35
                                 L96,35 L104,35 L108,20 L112,55 L116,10 L120,60 L124,35 L132,35
                                 L162,35 L170,35 L174,20 L178,55 L182,10 L186,60 L190,35 L198,35
                                 L228,35 L236,35 L240,20 L244,55 L248,10 L252,60 L256,35 L264,35
                                 L294,35 L302,35 L306,20 L310,55 L314,10 L318,60 L322,35 L330,35
                                 L360,35 L368,35 L372,20 L376,55 L380,10 L384,60 L388,35 L440,35"
                        />
                        <!-- Sharp ECG line -->
                        <path class="ecg-line"
                              stroke-dasharray="440"
                              stroke-dashoffset="0"
                              d="M0,35 L30,35 L38,35 L42,20 L46,55 L50,10 L54,60 L58,35 L66,35
                                 L96,35 L104,35 L108,20 L112,55 L116,10 L120,60 L124,35 L132,35
                                 L162,35 L170,35 L174,20 L178,55 L182,10 L186,60 L190,35 L198,35
                                 L228,35 L236,35 L240,20 L244,55 L248,10 L252,60 L256,35 L264,35
                                 L294,35 L302,35 L306,20 L310,55 L314,10 L318,60 L322,35 L330,35
                                 L360,35 L368,35 L372,20 L376,55 L380,10 L384,60 L388,35 L440,35"
                        />
                    </svg>
                </div>
                <div class="ecg-badge">
                    <div class="dot-live"></div>
                    <span style="color:#10B981;">AI Analysis Active</span>
                    <span style="color:#64748B;">•</span>
                    <span style="color:#00D4FF;">CardioSignals v1.0</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── FEATURES GRID ────────────────────────────────────────────────────
    st.markdown("""
    <div class="features-section fade-in fade-in-3">
        <p class="section-eyebrow">What We Offer</p>
        <h2 class="section-title">Everything You Need for Cardiac Risk Intelligence</h2>
        <p class="section-sub">
            A unified platform combining clinical data analysis and cardiac signal processing.
        </p>
    </div>
    """, unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4)
    features = [
        ("🩺", "Clinical Data Analysis",
         "Analyze multiple clinical features including blood pressure, "
         "cholesterol, BMI, and lifestyle factors for comprehensive risk assessment."),
        ("🤖", "Machine Learning Powered",
         "Advanced AI algorithms trained on large patient cohorts deliver "
         "accurate cardiovascular risk predictions in real time."),
        ("💡", "Transparent Insights",
         "See exactly which factors contribute to your risk score with "
         "clear, jargon-free explanations designed for clinicians and patients."),
        ("📈", "ECG Signal Processing",
         "Neural network analysis of cardiac waveforms provides an additional "
         "layer of intelligence beyond traditional clinical data."),
    ]
    for col, (icon, title, desc) in zip([f1, f2, f3, f4], features):
        with col:
            st.markdown(f"""
            <div class="feat-card">
                <span class="feat-icon">{icon}</span>
                <div class="feat-title">{title}</div>
                <div class="feat-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── HOW IT WORKS ─────────────────────────────────────────────────────
    st.markdown("""
    <div class="how-section">
        <p class="section-eyebrow">Simple, 3-Step Process</p>
        <h2 class="section-title">How It Works</h2>
        <p class="section-sub">From clinical inputs to actionable insights in seconds.</p>
    </div>
    """, unsafe_allow_html=True)

    s1, arr1, s2, arr2, s3 = st.columns([3, 1, 3, 1, 3])
    steps = [
        ("📋", "Input Data",
         "Enter patient demographics, vital signs, and lifestyle "
         "factors using our intuitive clinical form."),
        ("⚡", "AI Analysis",
         "Our models process your data in real time and identify "
         "patterns associated with cardiovascular risk."),
        ("📊", "Get Insights",
         "Receive a clear risk score, the key contributing factors, "
         "and evidence-based clinical interpretation."),
    ]
    for col, i, (icon, title, desc) in zip([s1, s2, s3], [1, 2, 3], steps):
        with col:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-number">{i}</div>
                <span class="step-icon-big">{icon}</span>
                <div class="step-title">{title}</div>
                <div class="step-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    for col in [arr1, arr2]:
        with col:
            st.markdown("""
            <div class="step-arrow">→</div>
            """, unsafe_allow_html=True)

    # ── FOOTER CTA ───────────────────────────────────────────────────────
    st.markdown("""
    <div class="footer-cta">
        <h2 class="footer-cta-title">Ready to Assess Cardiovascular Risk?</h2>
        <p class="footer-cta-sub">
            Start your analysis now — no account required.
        </p>
    </div>
    """, unsafe_allow_html=True)

    cta_col, _ = st.columns([2, 3])
    with cta_col:
        if st.button(
            "🔍  Start Analysis →",
            key="footer_cta",
            use_container_width=True,
            type="primary"
        ):
            st.session_state["current_page"] = "🔬  Risk Analyser"
            st.rerun()

    st.markdown("""
    <p class="disclaimer">
        ⚠️ For research and educational purposes only. Not a substitute for professional
        medical advice, diagnosis, or treatment. Always consult a qualified healthcare
        provider for medical decisions.
    </p>
    """, unsafe_allow_html=True)
