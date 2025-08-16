#!/usr/bin/env python3
"""
Simplified Data Synthesizer Container for Testing
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Streamlit app configuration
st.set_page_config(
    page_title="Data Synthesizer - Simple",
    page_icon="🔢",
    layout="wide"
)

def generate_simple_tick_data(symbol: str, rows: int) -> pd.DataFrame:
    """Generate simple synthetic tick data"""
    try:
        # Generate timestamps
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=rows)
        timestamps = pd.date_range(start=start_time, end=end_time, freq='1min')
        
        # Generate synthetic data
        base_price = np.random.uniform(50, 200)
        data = []
        current_price = base_price
        
        for i, timestamp in enumerate(timestamps[:rows]):
            # Simple price movement
            price_change = np.random.normal(0, 0.001) * current_price
            current_price = max(0.01, current_price + price_change)
            
            # Simple volume
            volume = np.random.randint(1000, 10000)
            
            data.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'price': round(current_price, 4),
                'volume': volume,
                'bid_price': round(current_price * 0.999, 4),
                'ask_price': round(current_price * 1.001, 4)
            })
        
        return pd.DataFrame(data)
        
    except Exception as e:
        st.error(f"Error generating data: {e}")
        return pd.DataFrame()

def main():
    """Main Streamlit application"""
    
    st.title("🔢 Data Synthesizer - Simple Test")
    st.markdown("Generate synthetic tick data for testing")
    
    # Simple form
    with st.form("data_generation"):
        st.subheader("Generate Synthetic Data")
        
        symbol = st.text_input("Symbol", value="RELIANCE")
        rows = st.number_input("Number of Rows", min_value=100, value=1000, step=100)
        
        if st.form_submit_button("🚀 Generate Data"):
            try:
                with st.spinner(f"Generating {rows} rows of data..."):
                    start_time = time.time()
                    df = generate_simple_tick_data(symbol, rows)
                    generation_time = time.time() - start_time
                
                if not df.empty:
                    st.success(f"✅ Generated {len(df)} rows in {generation_time:.2f} seconds")
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows Generated", len(df))
                        st.metric("Generation Time", f"{generation_time:.2f}s")
                    with col2:
                        st.metric("Symbol", symbol)
                        st.metric("Price Range", f"₹{df['price'].min():.2f} - ₹{df['price'].max():.2f}")
                    with col3:
                        st.metric("Volume Range", f"{df['volume'].min():,} - {df['volume'].max():,}")
                        st.metric("Data Points", f"{len(df):,}")
                    
                    # Show sample data
                    st.subheader("📊 Sample Data")
                    st.dataframe(df.head(10))
                    
                    # Download option
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"{symbol}_tick_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                else:
                    st.error("❌ Failed to generate data")
                    
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    # Container info
    st.markdown("---")
    st.info("🔧 **Data Synthesizer Container v2.3.0** - Simple Test Version")
    st.info("📊 This container will generate synthetic tick data for model training")
    st.info("🚀 Replace with real Kite data in production")

if __name__ == "__main__":
    main()
