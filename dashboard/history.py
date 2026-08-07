"""Recent alert history viewer."""

import pandas as pd
import streamlit as st

from database.database import fetch_alerts


def render_recent_alerts() -> None:
    """Render a table of recent alerts."""
    st.subheader("Recent Alerts")
    alerts = fetch_alerts(limit=10)
    if alerts:
        df = pd.DataFrame([dict(item) for item in alerts])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No alerts yet. Monitoring will populate this table.")

    # TODO: Add filtering by severity or alert type.
