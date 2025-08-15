import streamlit as st
from kiteconnect import KiteConnect

from core.settings import KITE_API_KEY, KITE_API_SECRET, KITE_ACCESS_TOKEN

def render_login_ui():
    st.sidebar.header("Kite Connect Login")
    api_key = st.sidebar.text_input("API Key", value=KITE_API_KEY)
    api_secret = st.sidebar.text_input("API Secret", value=KITE_API_SECRET)
    if 'kite' not in st.session_state:
        st.session_state.kite = None
    if 'access_token' not in st.session_state:
        st.session_state.access_token = KITE_ACCESS_TOKEN or None
    if api_key and api_secret and st.session_state.access_token:
        if st.session_state.kite is None:
            st.session_state.kite = KiteConnect(api_key=api_key)
            st.session_state.kite.set_access_token(st.session_state.access_token)
        st.sidebar.success("Auto-login successful!")
    else:
        if api_key and api_secret:
            if st.session_state.kite is None:
                st.session_state.kite = KiteConnect(api_key=api_key)
            if st.session_state.access_token is None:
                st.sidebar.markdown(f"[Generate Access Token]({st.session_state.kite.login_url()})")
                request_token = st.sidebar.text_input("Enter Request Token")
                if st.sidebar.button("Get Access Token"):
                    try:
                        data = st.session_state.kite.generate_session(request_token, api_secret=api_secret)
                        st.session_state.access_token = data["access_token"]
                        st.session_state.kite.set_access_token(st.session_state.access_token)
                        st.sidebar.success("Login successful!")
                    except Exception as e:
                        st.sidebar.error(f"Authentication failed: {e}")
            else:
                st.sidebar.success("Already logged in.")
    return api_key, api_secret 