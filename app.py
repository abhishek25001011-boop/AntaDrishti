"""Streamlit entry point for the AntaDrishti command center."""

import streamlit as st

from config import APP_NAME
from dashboard.dashboard import render_dashboard
from database.database import init_db


def main() -> None:
    init_db()
    st.set_page_config(page_title=APP_NAME, page_icon="🛡️", layout="wide")
    render_dashboard()


if __name__ == "__main__":
    main()
