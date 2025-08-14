import streamlit as st
st.set_page_config(page_title="Quant Trading Service", layout="wide")
from core.database import setup_database
from ui.pages.login import render_login_ui
from ui.pages.ingestion import render_ingestion_ui
from ui.pages.archive import render_archive_ui
from ui.pages.management import render_management_ui
from ui.pages.view import render_view_ui
from ui.pages.backtest import render_backtest_ui
from ui.pages.admin import render_admin_ui
from ui.pages.strategies import render_strategies_ui
from ui.pages.ml_pipeline import render_ml_pipeline_ui
import os

def main():
    # --- Database Setup ---
    if 'db_setup_done' not in st.session_state:
        setup_database()
        st.session_state['db_setup_done'] = True

    # --- UI Sections ---
    render_login_ui()
    render_admin_ui()
    render_ingestion_ui()
    render_archive_ui()
    render_management_ui()

    # Main content area
    st.title("📊 Quant Trading Service")

    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Strategies", "📊 Data View", "🔄 Legacy Backtest", "🤖 ML Pipeline", "📋 Coverage Report"])

    with tab1:
        render_strategies_ui()

    with tab2:
        render_view_ui()

    with tab3:
        render_backtest_ui()

    with tab4:
        render_ml_pipeline_ui()

    with tab5:
        def show_coverage_report():
            html_path = "coverage_html/index.html"
            if os.path.exists(html_path):
                with open(html_path, "r") as f:
                    html = f.read()
                st.header("Test Coverage Report")
                st.components.v1.html(html, height=800, scrolling=True)
            else:
                st.info("No coverage report found. Run `pytest --cov=app --cov-report=html:coverage_html` to generate it.")

        show_coverage_report()

if __name__ == "__main__":
    main()
