"""Reusable UI components for CardioSignals."""
import streamlit as st
import plotly.graph_objects as go


def metric_card(label: str, value: str, sublabel: str = None,
                accent: str = "#00D4FF") -> None:
    """Render a dark metric card with colored top border."""
    sub_html = (f"<div style='font-size:0.72rem; color:#64748B; "
                f"font-style:italic; margin-top:0.25rem;'>{sublabel}</div>"
                if sublabel else "")
    st.markdown(f"""
    <div style='
        background:#111827;
        border-top: 3px solid {accent};
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        margin-bottom: 0.5rem;
        border: 1px solid #1E2A3A;
        border-top: 3px solid {accent};
    '>
        <div style='font-size:0.62rem; color:#64748B;
                    text-transform:uppercase;
                    letter-spacing:0.1em; margin-bottom:0.4rem;'>
            {label}
        </div>
        <div style='font-size:2rem; font-weight:800;
                    color:#F1F5F9; line-height:1.1;'>
            {value}
        </div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def risk_gauge(score: float) -> go.Figure:
    """Plotly gauge 0-100. Green/amber/red zones."""
    if score <= 33:
        color = "#10B981"
    elif score <= 66:
        color = "#F59E0B"
    else:
        color = "#EF4444"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={
            "suffix": "%",
            "font": {"size": 36, "color": "#F1F5F9", "family": "Inter"}
        },
        title={
            "text": "Cardiovascular Risk Score",
            "font": {"size": 13, "color": "#94A3B8"}
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#1E2A3A",
                "tickfont": {"color": "#64748B", "size": 10}
            },
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#1E2A3A",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 33],  "color": "#0D2E20"},
                {"range": [33, 66], "color": "#2D2008"},
                {"range": [66, 100],"color": "#2D0A0A"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": score
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font={"color": "#94A3B8", "family": "Inter"},
        margin=dict(l=20, r=20, t=40, b=10),
        height=260
    )
    return fig


def section_header(title: str, subtitle: str = None,
                   badge: str = None,
                   badge_color: str = "#00D4FF") -> None:
    """Section divider with optional badge and subtitle."""
    badge_html = ""
    if badge:
        text_color = "#0A0E1A" if badge_color in [
            "#00D4FF", "#10B981", "#F59E0B"] else "#F1F5F9"
        badge_html = (
            f"<span style='background:{badge_color}; color:{text_color}; "
            f"font-size:0.58rem; font-weight:700; text-transform:uppercase; "
            f"letter-spacing:0.1em; padding:3px 10px; border-radius:20px; "
            f"float:right; margin-top:4px;'>{badge}</span>"
        )
    sub_html = (
        f"<div style='font-size:0.8rem; color:#64748B; "
        f"margin-top:0.2rem;'>{subtitle}</div>"
        if subtitle else ""
    )
    st.markdown(f"""
    <div style='
        border-left: 4px solid #00D4FF;
        padding-left: 0.8rem;
        margin: 1.2rem 0 0.8rem 0;
        overflow: hidden;
    '>
        {badge_html}
        <span style='font-size:1.1rem; font-weight:700;
                     color:#F1F5F9;'>{title}</span>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def insight_card(title: str, body: str,
                 accent: str = "#00D4FF",
                 icon: str = "💡") -> None:
    """Interpretation card with colored left border."""
    st.markdown(f"""
    <div style='
        background: #111827;
        border-left: 4px solid {accent};
        border-radius: 8px;
        padding: 1rem 1.1rem;
        margin: 0.5rem 0;
        border: 1px solid #1E2A3A;
        border-left: 4px solid {accent};
    '>
        <div style='font-size:0.85rem; font-weight:700;
                    color:#F1F5F9; margin-bottom:0.4rem;'>
            {icon} {title}
        </div>
        <div style='font-size:0.82rem; color:#94A3B8;
                    line-height:1.6;'>
            {body}
        </div>
    </div>
    """, unsafe_allow_html=True)


def risk_badge(score: float) -> str:
    """Returns HTML span with risk level label."""
    if score < 33:
        bg, label = "#10B981", "LOW RISK"
    elif score < 66:
        bg, label = "#F59E0B", "MODERATE RISK"
    else:
        bg, label = "#EF4444", "HIGH RISK"
    return (
        f"<span style='background:{bg}; color:#0A0E1A; "
        f"font-size:0.72rem; font-weight:800; text-transform:uppercase; "
        f"letter-spacing:0.08em; padding:5px 14px; border-radius:20px; "
        f"display:inline-block;'>{label}</span>"
    )


def bp_badge_v2(systolic: int, diastolic: int) -> str:
    """Returns BP category as colored HTML span using both systolic and diastolic."""
    try:
        from core.validators import categorize_bp
        label, color = categorize_bp(systolic, diastolic)
    except Exception:
        label, color = "Unknown", "#64748B"
    return (
        f"<span style='background:{color}20; color:{color}; "
        f"border:1px solid {color}40; font-size:0.72rem; font-weight:700; "
        f"text-transform:uppercase; letter-spacing:0.06em; "
        f"padding:3px 10px; border-radius:6px;'>{label}</span>"
    )


def bp_badge(ap_hi: int) -> str:
    """Returns BP category as colored HTML span."""
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
    return (
        f"<span style='background:{bg}20; color:{bg}; "
        f"border:1px solid {bg}40; font-size:0.72rem; font-weight:700; "
        f"text-transform:uppercase; letter-spacing:0.06em; "
        f"padding:3px 10px; border-radius:6px;'>{label}</span>"
    )


def bmi_display(height_cm: float, weight_kg: float) -> str:
    """Returns BMI value + category as colored span."""
    bmi = weight_kg / ((height_cm / 100) ** 2)
    if bmi < 18.5:
        color, label = "#7C3AED", "UNDERWEIGHT"
    elif bmi < 25:
        color, label = "#10B981", "NORMAL"
    elif bmi < 30:
        color, label = "#F59E0B", "OVERWEIGHT"
    else:
        color, label = "#EF4444", "OBESE"
    return (
        f"<span style='color:{color}; font-weight:700; "
        f"font-size:0.8rem;'>{bmi:.1f} — {label}</span>"
    )


def status_chip(label: str, status: str) -> str:
    """Inline status indicator HTML."""
    if status == "online":
        dot_color, suffix = "#10B981", ""
    elif status == "demo":
        dot_color, suffix = "#F59E0B", " <span style='color:#F59E0B;font-size:0.65rem;'>(Demo)</span>"
    else:
        dot_color, suffix = "#EF4444", " <span style='color:#EF4444;font-size:0.65rem;'>(Unavailable)</span>"
    return (
        f"<div style='display:flex; align-items:center; gap:8px; "
        f"padding:4px 0; font-size:0.78rem; color:#94A3B8;'>"
        f"<span style='width:8px; height:8px; border-radius:50%; "
        f"background:{dot_color}; display:inline-block; "
        f"box-shadow:0 0 6px {dot_color};'></span>"
        f"{label}{suffix}</div>"
    )


def limitation_card(icon: str, title: str, body: str) -> str:
    """Amber-bordered limitation card HTML."""
    return f"""
    <div style='
        background:#111827;
        border-left:4px solid #F59E0B;
        border:1px solid #1E2A3A;
        border-left:4px solid #F59E0B;
        border-radius:8px;
        padding:1rem;
        height:100%;
    '>
        <div style='font-size:1.2rem; margin-bottom:0.4rem;'>{icon}</div>
        <div style='font-size:0.85rem; font-weight:700;
                    color:#F1F5F9; margin-bottom:0.4rem;'>{title}</div>
        <div style='font-size:0.78rem; color:#94A3B8;
                    line-height:1.6;'>{body}</div>
    </div>
    """


def flow_diagram(steps: list) -> None:
    """Vertical architecture flow diagram."""
    html = "<div style='display:flex; flex-direction:column; gap:0;'>"
    for i, step in enumerate(steps):
        html += f"""
        <div style='
            background:#111827;
            border:1px solid #1E2A3A;
            border-left:4px solid #00D4FF;
            border-radius:8px;
            padding:0.7rem 1rem;
            display:flex;
            align-items:center;
            gap:0.8rem;
        '>
            <span style='font-size:1.1rem;'>{step['icon']}</span>
            <div>
                <div style='font-size:0.82rem; font-weight:700;
                            color:#F1F5F9;'>{step['title']}</div>
                <div style='font-size:0.72rem;
                            color:#64748B;'>{step['subtitle']}</div>
            </div>
        </div>
        """
        if i < len(steps) - 1:
            html += """
            <div style='
                width:2px; height:16px; background:#00D4FF40;
                margin:0 0 0 1.5rem;
                border-left:2px dashed #00D4FF40;
            '></div>
            """
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)