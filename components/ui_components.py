"""
components/ui_components.py
All reusable HTML/CSS helper functions for the CardioSignals UI.
"""

import streamlit as st
import plotly.graph_objects as go


# ── GLOBAL CSS ───────────────────────────────────────────────────────────────

GLOBAL_CSS = """
<style>
/* Hide Streamlit chrome and native nav */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebarNav"] { display: none !important; }
.block-container { padding: 1.5rem 2rem; max-width: 100% !important; }
.stApp { background-color: #0A0E1A; }

[data-testid="stSidebar"] {
    background-color: #080C16 !important;
    border-right: 1px solid #1E2A3A;
}
.stButton > button {
    background: linear-gradient(135deg, #00D4FF, #0099CC);
    color: #0A0E1A; font-weight: 700; border: none;
    border-radius: 8px; padding: 0.6rem 1.5rem;
    letter-spacing: 0.05em; width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }
[data-testid="stSlider"] > div > div > div {
    background: #00D4FF !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; color: #64748B;
    border-bottom: 2px solid transparent; font-weight: 600;
}
.stTabs [aria-selected="true"] {
    color: #00D4FF !important;
    border-bottom-color: #00D4FF !important;
}
[data-testid="metric-container"] { display: none; }
[data-testid="stFileUploadDropzone"] {
    background: #111827 !important;
    border: 2px dashed #1E3A4A !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #00D4FF !important;
}
div[data-testid="stExpander"] {
    background: #111827;
    border: 1px solid #1E2A3A;
    border-radius: 12px;
}

/* Radio nav pills */
[data-testid="stSidebar"] .stRadio label {
    display: block; padding: 0.45rem 1rem;
    border-radius: 8px; cursor: pointer;
    color: #64748B; font-weight: 500;
    transition: all 0.15s;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: #111827; color: #F1F5F9;
}
[data-testid="stSidebar"] .stRadio [aria-checked="true"] + label,
[data-testid="stSidebar"] .stRadio input:checked + label,
[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] {
    background: #0F2438 !important; color: #00D4FF !important;
}
</style>
"""

PLOTLY_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor="#111827",
    plot_bgcolor="#111827",
    font=dict(color="#94A3B8"),
    margin=dict(l=20, r=20, t=40, b=20),
)


def inject_css():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ── METRIC CARD ──────────────────────────────────────────────────────────────

def metric_card(label: str, value: str, sublabel: str = None,
                accent: str = "#00D4FF"):
    sub_html = (f'<div style="font-size:0.75rem;color:#64748B;'
                f'font-style:italic;margin-top:4px;">{sublabel}</div>'
                if sublabel else "")
    st.markdown(f"""
<div style="background:#111827;border-radius:12px;padding:1.2rem;
     border-top:3px solid {accent};height:100%;">
  <div style="font-size:0.65rem;text-transform:uppercase;
       color:#64748B;letter-spacing:0.1em;margin-bottom:6px;">{label}</div>
  <div style="font-size:2.2rem;font-weight:700;color:#F1F5F9;
       line-height:1.1;">{value}</div>
  {sub_html}
</div>
""", unsafe_allow_html=True)


# ── RISK GAUGE ───────────────────────────────────────────────────────────────

def risk_gauge(score: float) -> go.Figure:
    """Plotly gauge 0-100. Green / Amber / Red zones."""
    if score < 33:
        bar_color = "#10B981"
    elif score < 66:
        bar_color = "#F59E0B"
    else:
        bar_color = "#EF4444"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "%", "font": {"size": 28, "color": "#F1F5F9"}},
        title={"text": "Cardiovascular Risk Score",
               "font": {"size": 14, "color": "#94A3B8"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#64748B",
                     "tickfont": {"color": "#64748B"}},
            "bar": {"color": bar_color},
            "bgcolor": "#1E2A3A",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 33],  "color": "rgba(16,185,129,0.08)"},
                {"range": [33, 66], "color": "rgba(245,158,11,0.08)"},
                {"range": [66, 100],"color": "rgba(239,68,68,0.08)"},
            ],
            "threshold": {
                "line": {"color": bar_color, "width": 3},
                "thickness": 0.75,
                "value": score,
            },
        }
    ))
    fig.update_layout(
        paper_bgcolor="#111827",
        font=dict(color="#94A3B8"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=220,
    )
    return fig


# ── SECTION HEADER ───────────────────────────────────────────────────────────

def section_header(title: str, subtitle: str = None,
                   badge: str = None, badge_color: str = "#00D4FF"):
    sub_html = (f'<div style="font-size:0.85rem;color:#64748B;'
                f'margin-top:4px;">{subtitle}</div>'
                if subtitle else "")
    badge_html = ""
    if badge:
        badge_html = (
            f'<span style="background:{badge_color};color:#0A0E1A;'
            f'font-size:0.6rem;text-transform:uppercase;'
            f'letter-spacing:0.08em;border-radius:20px;'
            f'padding:2px 10px;font-weight:700;'
            f'float:right;margin-top:2px;">{badge}</span>')
    st.markdown(f"""
<div style="border-left:4px solid #00D4FF;padding-left:12px;
     margin:1.5rem 0 1rem 0;">
  {badge_html}
  <div style="font-size:1.3rem;font-weight:700;color:#F1F5F9;">{title}</div>
  {sub_html}
  <div style="clear:both;"></div>
</div>
""", unsafe_allow_html=True)


# ── INSIGHT CARD ─────────────────────────────────────────────────────────────

def insight_card(title: str, body: str,
                 accent: str = "#00D4FF", icon: str = "💡"):
    st.markdown(f"""
<div style="background:#111827;border-left:4px solid {accent};
     border-radius:8px;padding:1rem;margin:0.75rem 0;">
  <div style="font-weight:700;color:#F1F5F9;margin-bottom:6px;">
    {icon} {title}</div>
  <div style="color:#94A3B8;font-size:0.9rem;line-height:1.55;">
    {body}</div>
</div>
""", unsafe_allow_html=True)


# ── RISK BADGE ───────────────────────────────────────────────────────────────

def risk_badge(score: float) -> str:
    if score < 33:
        bg, label = "#10B981", "LOW RISK"
    elif score < 66:
        bg, label = "#F59E0B", "MODERATE RISK"
    else:
        bg, label = "#EF4444", "HIGH RISK"
    return (f'<span style="background:{bg};color:#0A0E1A;font-weight:700;'
            f'font-size:0.7rem;text-transform:uppercase;border-radius:20px;'
            f'padding:4px 14px;letter-spacing:0.06em;">{label}</span>')


# ── BP BADGE ─────────────────────────────────────────────────────────────────

def bp_badge(ap_hi: int) -> str:
    if ap_hi < 90:
        bg, label = "#64748B", "LOW"
    elif ap_hi < 120:
        bg, label = "#10B981", "NORMAL"
    elif ap_hi < 130:
        bg, label = "#F59E0B", "ELEVATED"
    elif ap_hi < 140:
        bg, label = "#F97316", "STAGE 1"
    else:
        bg, label = "#EF4444", "STAGE 2 — HYPERTENSIVE"
    return (f'<span style="background:{bg};color:#0A0E1A;font-weight:700;'
            f'font-size:0.7rem;text-transform:uppercase;border-radius:20px;'
            f'padding:3px 12px;">{label}</span>')


# ── BMI DISPLAY ──────────────────────────────────────────────────────────────

def bmi_display(height_cm: float, weight_kg: float) -> str:
    bmi = weight_kg / (height_cm / 100) ** 2
    if bmi < 18.5:
        color, label = "#7C3AED", "UNDERWEIGHT"
    elif bmi < 25:
        color, label = "#10B981", "NORMAL"
    elif bmi < 30:
        color, label = "#F59E0B", "OVERWEIGHT"
    else:
        color, label = "#EF4444", "OBESE"
    return (f'<span style="color:{color};font-weight:700;'
            f'font-size:0.7rem;text-transform:uppercase;">{label}</span>')


def compute_bmi(height_cm: float, weight_kg: float) -> float:
    return weight_kg / (height_cm / 100) ** 2


# ── STATUS CHIP ──────────────────────────────────────────────────────────────

def status_chip(label: str, status: str):
    if status == "online":
        dot, suffix = "🟢", ""
    elif status == "demo":
        dot, suffix = "🟡", " (Demo)"
    else:
        dot, suffix = "🔴", " (Unavailable)"
    st.markdown(
        f'<div style="font-size:0.78rem;color:#94A3B8;'
        f'padding:2px 0;">{dot} {label}{suffix}</div>',
        unsafe_allow_html=True)


# ── FLOW DIAGRAM ─────────────────────────────────────────────────────────────

def flow_diagram(steps: list):
    """Vertical architecture flow. steps = [{icon, title, subtitle}]."""
    html = '<div style="display:flex;flex-direction:column;gap:0;">'
    for i, step in enumerate(steps):
        connector = ""
        if i < len(steps) - 1:
            connector = """
<div style="width:2px;height:18px;background:linear-gradient(#00D4FF,#7C3AED);
     margin-left:22px;border-left:2px dashed #00D4FF;opacity:0.5;">
</div>"""
        html += f"""
<div style="background:#111827;border:1px solid #1E2A3A;border-left:3px solid #00D4FF;
     border-radius:8px;padding:0.7rem 0.9rem;display:flex;align-items:center;gap:10px;">
  <span style="font-size:1.2rem;">{step.get('icon','•')}</span>
  <div>
    <div style="font-weight:700;color:#F1F5F9;font-size:0.85rem;">
      {step['title']}</div>
    <div style="font-size:0.72rem;color:#64748B;">
      {step.get('subtitle','')}</div>
  </div>
</div>
{connector}
"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ── EMPTY STATE CARD ─────────────────────────────────────────────────────────

def empty_state_card(filename: str = ""):
    st.markdown(f"""
<div style="background:#111827;border:2px solid #1E2A3A;border-radius:12px;
     padding:2rem;text-align:center;color:#64748B;font-size:0.875rem;
     margin:0.5rem 0;">
  Data unavailable{' — ' + filename + ' not found' if filename else ''}
</div>
""", unsafe_allow_html=True)
