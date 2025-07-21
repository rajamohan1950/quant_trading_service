import yfinance as yf
import logging
import datetime
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)

def fetch_historical_data(ticker, start_date, end_date, interval='1d'):
    """
    Fetches historical stock data from Yahoo Finance.
    """
    try:
        data = yf.download(
            tickers=ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False
        )
        if data.empty:
            logging.warning(f"No data found for ticker {ticker} in the given date range.")
            return None
        return data
    except Exception as e:
        logging.error(f"Error fetching historical data for {ticker}: {e}")
        return None

def main():
    # Example: Fetch historical data for Eternal (NSE:ETERNAL)
    ticker = "ETERNAL.NS"  # Yahoo Finance ticker for ETERNAL on NSE
    
    # Define date range
    to_date = datetime.datetime.now()
    from_date = to_date - datetime.timedelta(days=365) # Fetch for the last year
    
    # yfinance intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    interval = "1d"
    
    # Fetch data
    data = fetch_historical_data(
        ticker, 
        from_date.strftime('%Y-%m-%d'), 
        to_date.strftime('%Y-%m-%d'), 
        interval
    )
    
    if data is not None:
        logging.info(f"Historical data for {ticker} fetched successfully!")
        # yfinance returns a pandas DataFrame
        print(data.head()) # Print first 5 rows
        # for index, row in data.iterrows():
        #     print(f"Date: {index.strftime('%Y-%m-%d')}, Open: {row['Open']:.2f}, High: {row['High']:.2f}, "
        #           f"Low: {row['Low']:.2f}, Close: {row['Close']:.2f}, Volume: {int(row['Volume'])}")
    else:
        logging.error(f"Failed to fetch historical data for {ticker}.")

if __name__ == "__main__":
    main()