import os
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from .env
load_dotenv()

# Prefer Streamlit secrets if available (for Streamlit Cloud), else .env, else manual
try:
    KITE_API_KEY = st.secrets["KITE_API_KEY"]
    KITE_API_SECRET = st.secrets["KITE_API_SECRET"]
    KITE_ACCESS_TOKEN = st.secrets.get("KITE_ACCESS_TOKEN", "")
except (FileNotFoundError, KeyError):
    KITE_API_KEY = os.getenv("KITE_API_KEY", "")
    KITE_API_SECRET = os.getenv("KITE_API_SECRET", "")
    KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN", "")

# Global config
TARGET_SYMBOLS = ["ETERNAL", "SWIGGY", "ZOMATO"]
INTERVAL_OPTIONS = {
    '5minute': '5minute',
    '15minute': '15minute',
    '30minute': '30minute',
    '60minute': '60minute',
    'day': 'day'
}
DB_FILE = "stock_data.duckdb"
