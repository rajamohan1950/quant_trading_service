import streamlit as st
from database import get_db_connection

def render_archive_ui():
    st.sidebar.header("Fetched Data Archive")
    con = get_db_connection()
    fetch_log_df = con.execute("SELECT * FROM fetch_log ORDER BY fetched_at DESC").fetchdf()
    con.close()
    st.sidebar.dataframe(fetch_log_df) 