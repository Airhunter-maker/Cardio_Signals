import streamlit as st
import sys
import os

# Add root to path so models/ and utils/ are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="CardioSignals",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .block-container { padding: 1.5rem 2rem; max-width: 100% !important; }
    .stApp { background-color: #0A0E1A; }

    [data-testid="stSidebar"] {
        background-color: #080C16 !important;
        border-right: 1px solid #1E2A3A;
    }
    [data-testid="stSidebar"] * { color: #94A3B8; }

    /* Primary buttons (type="primary") */
    .stButton > button[kind="primary"],
    .stButton > button[kind="primary"] * {
        background: linear-gradient(135deg, #00D4FF, #0090B8) !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 800 !important;
        padding: 0.65rem 1.4rem !important;
        box-shadow: 0 4px 16px rgba(0,212,255,0.25);
        transition: all 0.2s;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 24px rgba(0,212,255,0.45) !important;
        transform: translateY(-1px);
    }

    /* Secondary buttons */
    .stButton > button[kind="secondary"] {
        background: #1E2A3A !important;
        color: #94A3B8 !important;
        border: 1px solid #2D3F55 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s;
    }
    .stButton > button[kind="secondary"]:hover {
        background: #2D3F55 !important;
        color: #F1F5F9 !important;
    }

    /* General buttons fallback */
    .stButton > button {
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        transition: all 0.2s;
        font-weight: 600;
        width: 100%;
        border: none;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #64748B;
        border-bottom: 2px solid transparent;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        color: #00D4FF !important;
        border-bottom-color: #00D4FF !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background: transparent;
    }

    [data-testid="metric-container"] { display: none; }

    div[data-testid="stExpander"] {
        background: #111827;
        border: 1px solid #1E2A3A;
        border-radius: 12px;
        margin-bottom: 0.5rem;
    }
    div[data-testid="stExpander"] summary {
        color: #F1F5F9 !important;
        font-weight: 600;
    }

    /* Numeric slider — thumb only */
    [data-testid="stSlider"] [role="slider"] {
        background: #00D4FF !important;
    }
    /* Numeric slider track fill */
    [data-testid="stSlider"] > div > div > div > div > div {
        background: #00D4FF !important;
    }
    .stSlider label { color: #94A3B8 !important; }
    /* Select slider label */
    .stSelectSlider label { color: #94A3B8 !important; }
    /* Make the whole select-slider track clickable and visible */
    [data-testid="stSelectSlider"] > div > div {
        height: 8px !important;
        background: #1E2A3A !important;
        border-radius: 8px !important;
        cursor: pointer;
    }
    [data-testid="stSelectSlider"] > div {
        padding: 12px 0 !important;
        cursor: pointer;
    }
    /* Slider handle for select-slider */
    [data-testid="stSelectSlider"] [role="slider"] {
        background: #00D4FF !important;
        width: 18px !important;
        height: 18px !important;
        border-radius: 50% !important;
        cursor: grab !important;
        box-shadow: 0 0 8px rgba(0,212,255,0.5) !important;
    }
    /* Filled portion of select-slider */
    [data-testid="stSelectSlider"] > div > div > div:first-child {
        background: #00D4FF !important;
        border-radius: 8px !important;
    }
    .stRadio label { color: #94A3B8 !important; }
    .stRadio [data-testid="stMarkdownContainer"] p { color: #94A3B8; }
    .stToggle label { color: #94A3B8 !important; }

    div[data-baseweb="select"] {
        background: #111827 !important;
        border: 1px solid #1E2A3A !important;
        border-radius: 8px;
    }

    .stAlert { background: #111827; border: 1px solid #1E2A3A; }

    h1, h2, h3 { color: #F1F5F9 !important; }
    p { color: #94A3B8; }

    .js-plotly-plot { background: transparent !important; }
    .stSpinner > div { border-top-color: #00D4FF !important; }

    [data-testid="stSelectSlider"] > div > div {
        background: #1E2A3A;
        border-radius: 8px;
    }

    [data-testid="stHorizontalBlock"] { gap: 0.5rem; }

    .sidebar-nav .stRadio > div { gap: 0.25rem; }

    /* Download button style */
    .stDownloadButton > button {
        background: #1E2A3A !important;
        color: #94A3B8 !important;
        border: 1px solid #2D3F55 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s;
    }
    .stDownloadButton > button:hover {
        background: #2D3F55 !important;
        color: #F1F5F9 !important;
    }
    </style>
    """, unsafe_allow_html=True)


def init_session_state():
    defaults = {
        "risk_score": None,
        "shap_values": None,
        "ecg_model_loaded": False,
        "clinical_model_loaded": False,
        "errors": [],
        "last_features": None,
        "analyse_clicked": False,
        "current_page": "🏠  Home",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='padding: 1rem 0 0.5rem 0;'>
            <span style='font-size:1.5rem; font-weight:800;
                         color:#00D4FF; letter-spacing:-0.02em;'>
                🫀 CardioSignals
            </span><br>
            <span style='font-size:0.75rem; color:#64748B;
                         letter-spacing:0.05em;'>
                CARDIOVASCULAR RISK INTELLIGENCE
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#1E2A3A; margin:0.5rem 0;'>",
                    unsafe_allow_html=True)

        # Navigation — sync with session state so home page CTAs work
        nav_options = ["🏠  Home", "🔬  Risk Analyser", "📡  ECG Explorer"]
        current = st.session_state.get("current_page", "🏠  Home")
        
        # Find index, default to 0
        try:
            current_idx = nav_options.index(current)
        except ValueError:
            current_idx = 0

        page = st.radio(
            "Navigation",
            nav_options,
            index=current_idx,
            label_visibility="collapsed"
        )

        # Keep session state in sync when user clicks sidebar nav
        if page != st.session_state.get("current_page"):
            st.session_state["current_page"] = page

        st.markdown("<hr style='border-color:#1E2A3A; margin:0.5rem 0;'>",
                    unsafe_allow_html=True)

        from components.ui_components import status_chip
        ecg_status = "online" if st.session_state.get(
            "ecg_model_loaded") else "demo"
        st.markdown(status_chip("ECG Analysis", ecg_status),
                    unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#1E2A3A; margin:0.75rem 0;'>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:0.65rem; color:#475569; line-height:1.8;'>
            CardioSignals v1.0<br>
            <em>Research &amp; Educational Tool</em>
        </div>
        """, unsafe_allow_html=True)

    return page


def main():
    inject_css()
    init_session_state()

    # Pre-load clinical models silently
    try:
        from core.model_loader import load_all_models
        models = load_all_models()
        if models.get("rf") is not None:
            st.session_state["clinical_model_loaded"] = True
        if models.get("ecg") is not None:
            st.session_state["ecg_model_loaded"] = True
    except Exception as e:
        st.session_state["errors"].append(str(e))

    page = render_sidebar()

    if "Home" in page:
        from views.home import render_home
        render_home()
    elif "Risk Analyser" in page:
        from views.risk_analyser import render_risk_analyser
        render_risk_analyser()
    elif "ECG Explorer" in page:
        from views.ecg_explorer import render_ecg_explorer
        render_ecg_explorer()


if __name__ == "__main__":
    main()