import os
import time
import pandas as pd
import numpy as np
import yfinance as yf

# Cache folder to save downloaded datasets
CACHE_DIR = "/Users/azimsaidov/.gemini/antigravity/scratch/quant_engine/data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def load_multi_asset_data(tickers, start_date="2020-01-01", end_date="2026-05-28", force_download=False):
    """
    Downloads daily closing prices for a list of tickers, aligns them chronologically,
    handles stock-crypto calendar gaps (forward-filling weekends), and caches the data locally.
    
    Parameters:
    -----------
    tickers : list
        List of ticker symbols (e.g. ['BTC-USD', 'ETH-USD', 'SPY', 'GLD', 'TLT']).
    start_date : str
        Start date of backtest (YYYY-MM-DD).
    end_date : str
        End date of backtest (YYYY-MM-DD).
    force_download : bool
        If True, ignores local cache and re-downloads from Yahoo Finance.
        
    Returns:
    --------
    pd.DataFrame
        Aligned and cleaned daily closing price DataFrame where columns are ticker names.
    """
    # Create a unique filename for cache based on tickers and dates
    sorted_tickers = sorted(tickers)
    cache_name = f"portfolio_{'_'.join(sorted_tickers)}_{start_date}_{end_date}.csv".replace("-", "_")
    cache_path = os.path.join(CACHE_DIR, cache_name)
    
    # 1. Try to load from local cache
    if os.path.exists(cache_path) and not force_download:
        print(f"[DataLoader] Loading aligned portfolio from local cache: {cache_path}")
        df_cached = pd.read_csv(cache_path, parse_dates=True, index_col=0)
        return df_cached
        
    # 2. Programmatic API Download
    print(f"[DataLoader] Local cache miss. Downloading {len(tickers)} tickers via yfinance API...")
    start_time = time.time()
    
    portfolio_data = {}
    
    for ticker in tickers:
        print(f"   Downloading {ticker}...")
        try:
            # Download individual ticker history
            ticker_df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not ticker_df.empty:
                # Ensure we handle 1D series or multi-index anomalies
                if isinstance(ticker_df.columns, pd.MultiIndex):
                    # If yfinance returns multi-index due to single item downloads
                    close_col = ('Close', ticker)
                    if close_col in ticker_df.columns:
                        portfolio_data[ticker] = ticker_df[close_col]
                else:
                    if 'Close' in ticker_df.columns:
                        # Extract the Close column. In modern pandas, sometimes it's 2D, let's squeeze it to 1D
                        portfolio_data[ticker] = ticker_df['Close'].squeeze()
            else:
                print(f"   Warning: No data downloaded for {ticker}")
        except Exception as e:
            print(f"   Failed to download {ticker}: {e}")
            
    if not portfolio_data:
        raise ValueError("Error: No tickers were successfully downloaded!")
        
    # 3. Time Series Calendar Alignment
    print("[DataLoader] Aligning calendars and handling stock-crypto gaps...")
    # Outer join of all downloaded ticker series
    portfolio_df = pd.DataFrame(portfolio_data)
    
    # Clean DatetimeIndex name
    portfolio_df.index.name = 'DateTime'
    
    # Handle Calendar Gaps (Crypto trades 24/7, stocks trade only on weekdays)
    # 1. Forward-fill: stocks take their Friday closing price over Saturday & Sunday
    # 2. Backward-fill: handles any initial NaN periods at the very beginning of the dataset
    portfolio_df = portfolio_df.ffill().bfill()
    
    # Drop rows that are completely NaNs (just in case)
    portfolio_df = portfolio_df.dropna(how='all')
    
    duration = time.time() - start_time
    print(f"[DataLoader] Aligned portfolio of shape {portfolio_df.shape} in {duration:.2f} seconds.")
    
    # 4. Save to local cache
    portfolio_df.to_csv(cache_path)
    print(f"[DataLoader] Aligned portfolio saved to cache: {cache_path}")
    
    return portfolio_df

def load_data(csv_path=None, interval='D'):
    """
    Loads historical daily closing price data for BTC-USD. If csv_path exists, loads it; 
    otherwise, downloads it programmatically from Yahoo Finance.
    """
    if csv_path and os.path.exists(csv_path):
        print(f"[DataLoader] Loading data from local CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
            df.set_index('Timestamp', inplace=True)
        else:
            df.index = pd.to_datetime(df.index)
            
        if interval == 'D':
            # Resample 1-minute data to daily
            df = df.resample('D').last().ffill().dropna()
        return df
        
    # Default fallback: download BTC-USD dynamically from web
    ticker = 'BTC-USD'
    start_date = "2020-01-01"
    end_date = "2026-05-28"
    cache_path = os.path.join(CACHE_DIR, f"{ticker.replace('-', '_')}_daily.csv")
    
    if os.path.exists(cache_path):
        print(f"[DataLoader] Loading single asset {ticker} from local cache: {cache_path}")
        df_cached = pd.read_csv(cache_path, parse_dates=True, index_col=0)
        return df_cached
        
    print(f"[DataLoader] Single asset cache miss. Downloading {ticker} via yfinance...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        raise ValueError(f"Failed to download single asset data for {ticker}")
        
    # Handle modern pandas / yfinance multi-index Close column if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df.index.name = 'DateTime'
    df = df.ffill().bfill().dropna()
    df.to_csv(cache_path)
    print(f"[DataLoader] Saved {ticker} to local cache: {cache_path}")
    return df

if __name__ == "__main__":
    # Self-test download
    test_tickers = ['BTC-USD', 'SPY', 'GLD']
    try:
        df = load_multi_asset_data(test_tickers, start_date="2024-01-01", end_date="2024-12-31")
        print("\nAligned Data Sample (First 5 Rows):")
        print(df.head())
        print("\nChecking for missing values:")
        print(df.isna().sum())
    except Exception as e:
        print(f"DataLoader Self-Test Error: {e}")
