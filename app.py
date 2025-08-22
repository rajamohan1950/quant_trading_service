import streamlit as st

# Page configuration
st.set_page_config(
    page_title="B2C Investment Platform",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import and run the B2C investment platform
import sys
sys.path.append('.')

try:
    from ui.pages.b2c_investment import main
    # Run the main B2C interface
    main()
except Exception as e:
    st.error(f"Failed to load B2C Investment Platform: {str(e)}")
    st.write("Error details:", e)
