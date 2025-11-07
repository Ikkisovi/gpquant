# data_loaders.py
import pandas as pd
import numpy as np
import os
import zipfile
from datetime import datetime, timedelta

def parse_date(s: str) -> datetime:
    return datetime.strptime(s, '%Y-%m-%d')

def load_real_lean_data(symbol, symbol_path, start_date, end_date):
    """
    Load real Lean data with correct timestamp parsing.
    Timestamps are milliseconds from midnight.
    """
    all_data = []
    current_date = start_date

    while current_date <= end_date:
        date_str = current_date.strftime('%Y%m%d')
        zip_file = os.path.join(symbol_path, f"{date_str}_trade.zip")
        csv_file_in_zip = f"{date_str}_{symbol.lower()}_minute_trade.csv"

        if os.path.exists(zip_file):
            try:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    with zip_ref.open(csv_file_in_zip) as csv_file:
                        df = pd.read_csv(csv_file, header=None,
                                       names=['time', 'open', 'high', 'low', 'close', 'volume'])

                        date = pd.to_datetime(date_str, format='%Y%m%d')
                        df['datetime'] = date + pd.to_timedelta(df['time'], unit='ms')

                        for col in ['open', 'high', 'low', 'close']:
                            df[col] = df[col] / 10000.0

                        df['symbol'] = symbol
                        all_data.append(df)
            except Exception as e:
                # It's common for some dates or symbols to be missing, so we just note it.
                pass

        current_date += timedelta(days=1)

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()

def load_all_symbols_data(symbols, data_path, start_date, end_date):
    """
    Load data for all symbols and combine into a single DataFrame.
    Returns DataFrame with columns: ['symbol','datetime','open','high','low','close','volume']
    """
    all_symbol_data = []

    for symbol in symbols:
        print(f"Loading data for {symbol}...")
        symbol_path = os.path.join(data_path, symbol.lower())

        if not os.path.exists(symbol_path):
            print(f"Path not found for {symbol}: {symbol_path}")
            continue

        df = load_real_lean_data(symbol, symbol_path, start_date, end_date)

        if not df.empty:
            all_symbol_data.append(df)
            print(f"  Loaded {len(df)} records for {symbol}")

    if all_symbol_data:
        combined_data = pd.concat(all_symbol_data, ignore_index=True)
        print(f"\nTotal records loaded: {len(combined_data)}")
        return combined_data
    else:
        print("No data loaded for any symbols")
        return pd.DataFrame()

def aggregate_to_am_pm(
    data_path: str,
    shares_file: str,
    start_date_str: str,
    end_date_str: str,
) -> pd.DataFrame:
    """
    Reads minute-level stock data, aggregates it into two sessions (AM and PM),
    and calculates vwap, mktcap, and turnover.
    Returns DataFrame with columns: ['symbol','date','session','open','high','low','close','volume','vwap','mktcap','turnover',...]
    session ∈ {'AM','PM'}
    """
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

    # --- 1. Load Minute Data ---
    print("--- Step 1: Loading Minute Data ---")
    symbols = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    if not symbols:
        print(f"Error: No symbol directories found in '{data_path}'")
        return pd.DataFrame()
    print(f"Found {len(symbols)} symbols.")

    df = load_all_symbols_data(symbols, data_path, start_date, end_date)
    if df.empty:
        print("Stopping: No minute data was loaded.")
        return pd.DataFrame()

    # --- 2. Load and Merge Shares Outstanding Data ---
    print("\n--- Step 2: Loading and Merging Shares Outstanding ---")
    try:
        shares_df = pd.read_csv(shares_file)
        shares_df = shares_df[['symbol', 'shares_outstanding']].drop_duplicates()
    except FileNotFoundError:
        print(f"Error: Shares outstanding file not found at '{shares_file}'")
        return pd.DataFrame()

    df = pd.merge(df, shares_df, on='symbol', how='left')
    df.dropna(subset=['shares_outstanding'], inplace=True)
    if df.empty:
        print("Stopping: No data left after merging with shares outstanding. Check symbol match.")
        return pd.DataFrame()
    print("Shares outstanding data merged successfully.")

    # --- 3. Prepare Data and Define Sessions ---
    print("\n--- Step 3: Defining AM/PM Sessions ---")
    df['datetime'] = df['datetime'].dt.tz_localize("America/New_York", ambiguous='infer')
    df['date'] = df['datetime'].dt.date
    df['session'] = np.where(df['datetime'].dt.hour < 12, 'AM', 'PM')
    df = df.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    print("AM/PM sessions defined (Split at 12:00 ET).")

    # --- 4. Aggregate Data ---
    print("\n--- Step 4: Aggregating Data into Sessions ---")
    def calculate_vwap(group):
        if group['volume'].sum() > 0:
            return np.average(group['close'], weights=group['volume'])
        return np.nan

    grouped = df.groupby(['symbol', 'date', 'session'])
    vwap = grouped.apply(calculate_vwap).rename('vwap').reset_index()

    session_df = grouped.agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum'),
        shares_outstanding=('shares_outstanding', 'first')
    ).reset_index()

    session_df = pd.merge(session_df, vwap, on=['symbol', 'date', 'session'])
    print("Aggregation to sessions complete.")

    # --- 5. Calculate Financial Metrics ---
    print("\n--- Step 5: Calculating Financial Metrics ---")
    session_df['mktcap'] = session_df['close'] * session_df['shares_outstanding']
    session_df['turnover'] = session_df.apply(
        lambda row: row['volume'] / row['shares_outstanding'] if row['shares_outstanding'] > 0 else 0,
        axis=1
    )
    print("Calculated 'mktcap' and 'turnover'.")

    # --- 6. Finalize Data ---
    print("\n--- Step 6: Finalizing Data ---")
    session_df['timestamp'] = (
        pd.to_datetime(session_df['date']).dt.tz_localize("America/New_York")
        + pd.to_timedelta(session_df['session'].map({'AM': '12H', 'PM': '16H'}))
    )

    final_cols = [
        'symbol', 'timestamp', 'date', 'session', 'open', 'high', 'low', 'close',
        'volume', 'vwap', 'mktcap', 'turnover'
    ]
    session_df = session_df[final_cols]
    session_df = session_df.sort_values(by=['symbol', 'timestamp']).reset_index(drop=True)

    return session_df
