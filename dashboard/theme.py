"""Shared visual system for the AntaDrishti monitoring console."""

import streamlit as st


def apply_theme(mode: str = "Dark") -> None:
    """Apply the selected app-surface palette without changing Streamlit chrome."""
    palettes = {
        "Dark": {
            "scheme": "dark", "bg": "#08121d", "surface": "#0c1926", "raised": "#10202e",
            "border": "#203344", "text": "#e5edf4", "body": "#c1ced8", "muted": "#9eafbd",
            "accent": "#42bfd2", "input": "#0d1b29", "metric": "linear-gradient(145deg,#102131,#0c1926)",
            "header": "rgba(8,18,29,.96)", "mark": "#102d3a", "mark_text": "#69d2df",
            "button": "#122333", "button_hover": "#172d3e", "severity_high": "#ff9292",
            "severity_medium": "#f2c36d", "severity_low": "#7bd5b4",
        },
        "Light": {
            "scheme": "light", "bg": "#f3f6f8", "surface": "#ffffff", "raised": "#eaf1f4",
            "border": "#cfdae2", "text": "#162632", "body": "#344a58", "muted": "#536a79",
            "accent": "#087e95", "input": "#ffffff", "metric": "linear-gradient(145deg,#ffffff,#f4f8fa)",
            "header": "rgba(243,246,248,.96)", "mark": "#e5f3f5", "mark_text": "#087e95",
            "button": "#eaf1f4", "button_hover": "#dcebef", "severity_high": "#b42318",
            "severity_medium": "#8a5a00", "severity_low": "#087443",
        },
    }
    palette = palettes.get(mode, palettes["Dark"])
    css = """
    <style>
    :root { color-scheme: __SCHEME__; --ad-bg:__BG__; --ad-surface:__SURFACE__; --ad-raised:__RAISED__; --ad-input:__INPUT__;
      --ad-border:__BORDER__; --ad-text:__TEXT__; --ad-body:__BODY__; --ad-muted:__MUTED__; --ad-accent:__ACCENT__; }
    .stApp { background:var(--ad-bg); color:var(--ad-text); }
    [data-testid="stHeader"] { background:__HEADER__; }
    [data-testid="stSidebar"] { background:var(--ad-surface); border-right:1px solid var(--ad-border); }
    [data-testid="stSidebar"] > div:first-child { padding-top:1rem; }
    .block-container { max-width:1480px; padding:1.25rem 2rem 3rem; }
    h1,h2,h3,h4 { color:var(--ad-text); letter-spacing:-.025em; }
    h1 { font-size:clamp(1.8rem,3vw,2.4rem); }
    p,label,[data-testid="stMarkdownContainer"] { color:var(--ad-body); }
    a { color:var(--ad-accent); }
    [data-testid="stMetric"] { background:__METRIC__; border:1px solid var(--ad-border); border-radius:12px; padding:14px 16px; min-height:96px; }
    [data-testid="stMetricLabel"] { color:var(--ad-muted); font-size:.78rem; }
    [data-testid="stMetricValue"] { color:var(--ad-text); font-weight:700; }
    [data-testid="stMetricDelta"] { font-size:.75rem; }
    [data-testid="stVerticalBlockBorderWrapper"] { background:var(--ad-surface); border-color:var(--ad-border); border-radius:13px; }
    [data-testid="stAlert"] { border-radius:10px; }
    [data-testid="stDataFrame"] { border:1px solid var(--ad-border); border-radius:10px; overflow:hidden; }
    [data-testid="stCustomComponentV1"] { border-radius:12px; overflow:hidden; }
    [data-testid="stExpander"] { border-color:var(--ad-border); border-radius:10px; }
    div.stButton > button { border-radius:9px; min-height:2.65rem; border-color:var(--ad-border); color:var(--ad-text); background:__BUTTON__; font-weight:600; transition:border-color .15s ease,background .15s ease; }
    div.stButton > button:hover { border-color:var(--ad-accent); color:var(--ad-text); background:__BUTTON_HOVER__; }
    div.stButton > button:focus-visible { outline:3px solid var(--ad-accent); outline-offset:2px; }
    div.stButton > button[kind="primary"] { background:var(--ad-accent); border-color:var(--ad-accent); color:__PRIMARY_TEXT__; font-weight:750; }
    div.stButton > button[kind="primary"]:hover { filter:brightness(1.08); }
    [data-testid="stProgressBar"] > div > div { background:var(--ad-accent); }
    input,textarea,[data-baseweb="select"] > div { background-color:var(--ad-input) !important; border-color:var(--ad-border) !important; color:var(--ad-text) !important; }
    [data-testid="stRadio"] label,[data-testid="stCheckbox"] label,[data-testid="stToggle"] label { color:var(--ad-body); }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { padding:.22rem 0; }
    .ad-hero { display:flex; justify-content:space-between; align-items:center; gap:1rem; background:var(--ad-surface); border:1px solid var(--ad-border); border-left:3px solid var(--ad-accent); border-radius:13px; padding:15px 20px; margin:0 0 1.4rem; }
    .ad-brand { display:flex; align-items:center; gap:13px; }
    .ad-mark { display:grid; place-items:center; width:42px; height:42px; border:1px solid var(--ad-border); border-radius:11px; background:__MARK__; color:__MARK_TEXT__; font-size:14px; font-weight:800; letter-spacing:.04em; }
    .ad-title { color:var(--ad-text); font-size:19px; font-weight:800; letter-spacing:.04em; }
    .ad-subtitle,.ad-astra { color:var(--ad-muted); font-size:12px; margin-top:3px; }
    .ad-astra { font-size:11px; text-align:right; letter-spacing:.08em; }
    .ad-status { display:inline-flex; align-items:center; gap:8px; border:1px solid var(--ad-border); color:var(--ad-text); background:var(--ad-raised); padding:6px 10px; border-radius:999px; font-size:12px; font-weight:700; }
    .ad-status-dot { width:7px; height:7px; border-radius:50%; background:currentColor; }
    .ad-page-kicker,.ad-kicker { color:var(--ad-accent); text-transform:uppercase; font-size:10px; letter-spacing:.15em; font-weight:750; margin-bottom:.3rem; }
    .ad-page-description,.ad-muted { color:var(--ad-muted); font-size:13px; }
    .ad-section-label { color:var(--ad-muted); text-transform:uppercase; letter-spacing:.11em; font-size:10px; font-weight:750; }
    .ad-severity-high { color:__HIGH__; font-weight:750; }
    .ad-severity-medium { color:__MEDIUM__; font-weight:750; }
    .ad-severity-low { color:__LOW__; font-weight:750; }
    .ad-inline-status { font-size:13px; font-weight:700; }
    [data-testid="stSidebar"] [role="radiogroup"] > label { border-radius:8px; }
    [data-testid="stSidebar"] [role="radiogroup"] > label:hover { background:var(--ad-raised); }
    @media (max-width:800px) { .block-container { padding:1rem 1rem 2rem; } .ad-hero { align-items:flex-start; padding:13px; } .ad-title { font-size:16px; } .ad-mark { width:36px; height:36px; } .ad-astra { font-size:9px; } }
    @media (prefers-reduced-motion:reduce) { *,*::before,*::after { scroll-behavior:auto !important; transition-duration:.01ms !important; } }
    </style>
    """
    replacements = {
        "SCHEME": palette["scheme"], "BG": palette["bg"], "SURFACE": palette["surface"],
        "RAISED": palette["raised"], "BORDER": palette["border"], "TEXT": palette["text"],
        "BODY": palette["body"], "MUTED": palette["muted"], "ACCENT": palette["accent"],
        "HEADER": palette["header"], "METRIC": palette["metric"], "BUTTON": palette["button"],
        "BUTTON_HOVER": palette["button_hover"], "INPUT": palette["input"], "MARK": palette["mark"],
        "MARK_TEXT": palette["mark_text"], "HIGH": palette["severity_high"],
        "MEDIUM": palette["severity_medium"], "LOW": palette["severity_low"],
        "PRIMARY_TEXT": "#ffffff" if mode == "Light" else "#07151e",
    }
    for key, value in replacements.items():
        css = css.replace(f"__{key}__", value)
    st.markdown(css, unsafe_allow_html=True)

def render_header(status: str) -> None:
    status_colors = {
        "Idle": "#aab8c3",
        "Processing": "#5bc5d5",
        "Camera Streaming": "#5bc5d5",
        "Detection Active": "#5bc5d5",
        "Completed": "#71d0af",
        "Error": "#ff9292",
    }
    color = status_colors.get(status, "#bdcbd5")
    safe_status = str(status).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(
        f"""<header class="ad-hero" aria-label="ANTA DRISHTI command center">
          <div class="ad-brand"><div class="ad-mark" aria-hidden="true">AD</div><div>
            <div class="ad-title">ANTA DRISHTI</div>
            <div class="ad-subtitle">AI-Powered Public Safety Intelligence System</div>
          </div></div>
          <div><div class="ad-status" style="border-color:{color};color:{color}">
            <span class="ad-status-dot" aria-hidden="true"></span>{safe_status}
          </div><div class="ad-astra" style="margin-top:7px">PROJECT ASTRA</div></div>
        </header>""",
        unsafe_allow_html=True,
    )


def render_page_heading(kicker: str, title: str, description: str) -> None:
    st.markdown(f'<div class="ad-page-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="ad-page-description">{description}</div>', unsafe_allow_html=True)
