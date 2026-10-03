"""Small local stylesheet for the AntaDrishti command-center UI."""

import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        .stApp { background: #07111f; color: #e6f2ff; }
        [data-testid="stHeader"] { background: rgba(7,17,31,.92); }
        [data-testid="stSidebar"] { background: #0b192a; border-right: 1px solid #16314a; }
        [data-testid="stSidebar"] > div:first-child { padding-top: 1.1rem; }
        .block-container { max-width: 1500px; padding-top: 1.6rem; padding-bottom: 2.5rem; }
        h1, h2, h3 { color: #f1f8ff; letter-spacing: -.02em; }
        p, label, .stMarkdown { color: #bdd0e2; }
        [data-testid="stMetric"] { background: #0c1b2d; border: 1px solid #183653;
          border-radius: 14px; padding: 16px 18px; }
        [data-testid="stMetricLabel"] { color: #9fb6ca; }
        [data-testid="stMetricValue"] { color: #ecf8ff; }
        [data-testid="stVerticalBlockBorderWrapper"] { background: #0b192a; border-color: #183653; border-radius: 14px; }
        div.stButton > button[kind="primary"] { background: #08a6c5; border: 0; color: #04111a; font-weight: 700; }
        div.stButton > button { border-radius: 9px; }
        [data-testid="stProgressBar"] > div > div { background: #08b7d6; }
        [data-testid="stDataFrame"] { border: 1px solid #183653; border-radius: 10px; }
        .ad-hero { display:flex; justify-content:space-between; align-items:center; gap:20px;
          background:linear-gradient(110deg,#0c2035,#0b192a 65%,#0a2638); border:1px solid #1a3e5b;
          border-radius:18px; padding:22px 26px; margin-bottom:22px; }
        .ad-brand { display:flex; align-items:center; gap:15px; }
        .ad-mark { display:grid; place-items:center; width:48px; height:48px; border-radius:14px;
          background:#0a9fba; color:#03111a; font-weight:900; font-size:18px; }
        .ad-title { color:#f1f8ff; font-size:24px; font-weight:800; line-height:1.1; }
        .ad-subtitle { color:#8faec5; font-size:13px; margin-top:5px; }
        .ad-astra { color:#54d9ed; font-size:13px; font-weight:700; text-align:right; }
        .ad-status { display:inline-flex; align-items:center; gap:7px; border:1px solid #1c5362;
          color:#93edf4; background:#0a2935; padding:7px 11px; border-radius:999px; font-size:12px; font-weight:700; }
        .ad-kicker { color:#53d2e5; text-transform:uppercase; font-size:11px; letter-spacing:.14em; font-weight:700; }
        .ad-muted { color:#8fa7bc; font-size:13px; }
        .ad-severity-high { color:#ff9d9d; font-weight:800; }
        .ad-severity-medium { color:#ffd27b; font-weight:800; }
        .ad-severity-low { color:#8ee0bb; font-weight:800; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(status: str) -> None:
    status_colors = {
        "System Ready": "#45d4ad",
        "Processing": "#54d9ed",
        "Detection Active": "#54d9ed",
        "Completed": "#45d4ad",
        "Error": "#ff8585",
    }
    color = status_colors.get(status, "#9fb6ca")
    st.markdown(
        f"""<div class="ad-hero">
          <div class="ad-brand"><div class="ad-mark">AD</div><div>
            <div class="ad-title">AntaDrishti</div>
            <div class="ad-subtitle">Public Safety Command Center</div>
          </div></div>
          <div><div class="ad-status" style="border-color:{color};color:{color}">● {status}</div>
            <div class="ad-astra" style="margin-top:8px">PROJECT ASTRA</div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )
