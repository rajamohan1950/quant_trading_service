import streamlit as st
import altair as alt
import pandas as pd

def render_equity_curve(equity_data, title="Equity Curve"):
    """Render equity curve chart"""
    if not equity_data:
        st.info("No equity curve data to display.")
        return
    
    equity_df = pd.DataFrame({
        'trade': list(range(1, len(equity_data)+1)),
        'equity': equity_data
    })
    
    chart = alt.Chart(equity_df).mark_line().encode(
        x='trade:Q',
        y='equity:Q',
        tooltip=['trade', 'equity']
    ).properties(
        title=title,
        width=600,
        height=300
    )
    
    st.altair_chart(chart)

def render_price_chart(df, title="Price Chart"):
    """Render price chart with indicators"""
    if df.empty:
        st.info("No data to display.")
        return
    
    # Base price chart
    chart = alt.Chart(df).mark_line().encode(
        x='datetime:T',
        y='close:Q',
        color=alt.value('blue'),
        tooltip=['datetime', 'close']
    ).properties(title=title)
    
    # Add moving averages if they exist
    layers = [chart]
    
    if 'ma20' in df.columns:
        ma20_line = alt.Chart(df).mark_line(color='orange').encode(
            x='datetime:T', 
            y='ma20:Q'
        )
        layers.append(ma20_line)
    
    if 'ma50' in df.columns:
        ma50_line = alt.Chart(df).mark_line(color='green').encode(
            x='datetime:T', 
            y='ma50:Q'
        )
        layers.append(ma50_line)
    
    if 'ema20' in df.columns:
        ema20_line = alt.Chart(df).mark_line(color='red').encode(
            x='datetime:T', 
            y='ema20:Q'
        )
        layers.append(ema20_line)
    
    # Combine all layers
    final_chart = layers[0]
    for layer in layers[1:]:
        final_chart = final_chart + layer
    
    st.altair_chart(final_chart) 