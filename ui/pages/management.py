import streamlit as st
from core.database import clear_all_data

def render_management_ui():
    st.sidebar.header("Database Management")
    if st.sidebar.button("Clear All Stored Data"):
        clear_all_data()
        st.sidebar.success("All stored data has been cleared.") 