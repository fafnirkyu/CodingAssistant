import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
import time
from dotenv import load_dotenv
import os

# Load environment if needed
load_dotenv("backend/.enviorment")

DB_PATH = os.getenv("DATABASE_URL", "sqlite:///./data/tickets.db").replace("sqlite:///", "")

st.set_page_config(
    page_title="Helpdesk AI Dashboard",
    layout="wide",
    page_icon=":robot_face:",
)

st.title("📊 AI Helpdesk Dashboard")
st.caption("Real-time monitoring of Zendesk ticket classification and AI responses")

REFRESH_INTERVAL = st.sidebar.slider("Refresh interval (seconds)", 5, 120, 15)

# Connect to database
def load_tickets():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM tickets", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database load failed: {e}")
        return pd.DataFrame()

# Auto-refresh loop
placeholder = st.empty()

while True:
    with placeholder.container():
        df = load_tickets()

        if df.empty:
            st.warning("No tickets found in database yet.")
        else:
            # Filters
            categories = df["category"].unique().tolist()
            selected_cat = st.sidebar.multiselect(
                "Filter by category",
                categories,
                default=categories,
                key="category_filter"
            )

            filtered_df = df[df["category"].isin(selected_cat)]

            # Sorting
            sort_col = st.sidebar.selectbox("Sort by", df.columns.tolist(), index=0)
            ascending = st.sidebar.checkbox("Ascending", value=False)
            filtered_df = filtered_df.sort_values(by=sort_col, ascending=ascending)

            # KPIs
            total_tickets = len(filtered_df)
            analyzed = filtered_df["analyzed"].sum() if "analyzed" in filtered_df.columns else total_tickets

            col1, col2 = st.columns(2)
            col1.metric("Total Tickets", total_tickets)
            col2.metric("Analyzed", analyzed)

            # Category distribution
            fig1 = px.bar(
                filtered_df.groupby("category").size().reset_index(name="count"),
                x="category",
                y="count",
                title="Ticket Distribution by Category",
                color="category"
            )
            st.plotly_chart(fig1, use_container_width=True)

            # Ticket table
            st.subheader("Recent Tickets")
            st.dataframe(
                filtered_df[["id", "category", "summary", "response"]].tail(20),
                use_container_width=True,
                height=500
            )

    # Refresh automatically
    time.sleep(REFRESH_INTERVAL)
