import streamlit as st
st.set_page_config(page_title="Quant Trading Service", layout="wide")
# from core.database import setup_database  # Removed - using PostgreSQL instead
# from ui.pages.login import render_login_ui  # Removed - not needed for B2C
# from ui.pages.ingestion import render_ingestion_ui  # Removed - not needed for B2C
# from ui.pages.archive import render_archive_ui  # Removed - not needed for B2C
# from ui.pages.management import render_management_ui  # Removed - not needed for B2C
# from ui.pages.view import render_view_ui  # Removed - not needed for B2C
# from ui.pages.backtest import render_backtest_ui  # Removed - not needed for B2C
# from ui.pages.admin import render_admin_ui  # Removed - not needed for B2C
# from ui.pages.strategies import render_strategies_ui  # Removed - not needed for B2C
# from ui.pages.production_ml_pipeline import render_production_ml_pipeline_ui  # Removed - not needed for B2C
from ui.pages.b2c_investor_simple import B2CInvestorPlatform
import os

def main():
    # --- Database Setup ---
    # Database initialized via PostgreSQL connection
    # if 'db_setup_done' not in st.session_state:
    #     setup_database()  # Removed - using PostgreSQL instead
    #     st.session_state['db_setup_done'] = True

    # --- UI Sections ---
    # render_login_ui()  # Removed - not needed for B2C
    # render_admin_ui()  # Removed - not needed for B2C
    # render_ingestion_ui()  # Removed - not needed for B2C
    # render_archive_ui()  # Removed - not needed for B2C
    # render_management_ui()  # Removed - not needed for B2C

    # Main content area
    st.title("💰 B2C Investment Platform")

    # B2C Investment Platform - Clean and Simple
    platform = B2CInvestorPlatform()
    platform.main()

if __name__ == "__main__":
    main()
