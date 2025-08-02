import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_ema(prices, period):
    """Calculate Exponential Moving Average"""
    return prices.ewm(span=period).mean()

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def format_currency(amount):
    """Format amount as Indian Rupees"""
    return f"₹{amount:.2f}"

def format_percentage(value):
    """Format value as percentage"""
    return f"{value:.2f}%"

def validate_date_range(start_date, end_date):
    """Validate date range"""
    if start_date >= end_date:
        return False, "Start date must be before end date"
    if end_date > datetime.now().date():
        return False, "End date cannot be in the future"
    return True, ""

def get_default_date_range(days_back=180):
    """Get default date range for data fetching"""
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)
    return start_date, end_date 