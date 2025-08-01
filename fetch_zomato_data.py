#!/usr/bin/env python3
"""
Script to fetch ZOMATO data for the last 6 months with 1-hour candles
"""

import streamlit as st
import pandas as pd
import datetime
from kiteconnect import KiteConnect
from settings import KITE_API_KEY, KITE_API_SECRET, KITE_ACCESS_TOKEN
from database import get_db_connection, log_fetch
from ingestion import fetch_and_store_data

def fetch_zomato_data():
    """Fetch ZOMATO data for the last 6 months with 1-hour candles"""
    
    # Setup Kite Connect
    kite = KiteConnect(api_key=KITE_API_KEY)
    if KITE_ACCESS_TOKEN:
        kite.set_access_token(KITE_ACCESS_TOKEN)
    else:
        print("Please set KITE_ACCESS_TOKEN in your environment variables")
        return
    
    try:
        # Get all NSE instruments
        print("Fetching instruments...")
        all_instruments = pd.DataFrame(kite.instruments("NSE"))
        
        # Filter for ZOMATO
        zomato_instruments = all_instruments[all_instruments['tradingsymbol'] == 'ZOMATO'].reset_index(drop=True)
        
        if zomato_instruments.empty:
            print("ZOMATO not found in NSE instruments")
            return
        
        # Get instrument token
        instrument_token = zomato_instruments.iloc[0]['instrument_token']
        tradingsymbol = zomato_instruments.iloc[0]['tradingsymbol']
        
        print(f"Found ZOMATO with token: {instrument_token}")
        
        # Calculate date range (6 months)
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=180)
        
        print(f"Fetching data from {start_date} to {end_date}")
        
        # Fetch and store data
        fetch_and_store_data(kite, instrument_token, tradingsymbol, start_date, end_date, '60minute')
        
        print("ZOMATO data fetch completed!")
        
    except Exception as e:
        print(f"Error fetching ZOMATO data: {e}")

if __name__ == "__main__":
    fetch_zomato_data() 