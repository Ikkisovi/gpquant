"""
Data Loader: Panel data loading and walk-forward splitting
Handles MultiIndex (timestamp, symbol) data with embargo
"""
import numpy as np
import pandas as pd
from datetime import timedelta


def load_panel(csv_path, required_columns=None):
    """
    Load panel data from CSV

    Expected format:
        - Columns: symbol, timestamp, open, high, low, close, volume, vwap, turnover, mktcap
        - Half-daily frequency (AM/PM sessions)

    Args:
        csv_path: Path to CSV file
        required_columns: List of required column names

    Returns:
        DataFrame with MultiIndex (timestamp, symbol)
    """
    if required_columns is None:
        required_columns = ['symbol', 'timestamp', 'open', 'high', 'low', 'close',
                            'volume', 'vwap', 'turnover', 'mktcap']

    print(f"Loading data from {csv_path}...")

    # Load CSV
    df = pd.read_csv(csv_path)

    # Check required columns
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Set MultiIndex
    df = df.set_index(['timestamp', 'symbol']).sort_index()

    # Basic validation
    if df.index.duplicated().any():
        print(f"⚠ Warning: Found {df.index.duplicated().sum()} duplicate (timestamp, symbol) pairs")
        df = df[~df.index.duplicated(keep='first')]

    print(f"✓ Loaded {len(df)} observations")
    print(f"  Date range: {df.index.get_level_values(0).min()} to {df.index.get_level_values(0).max()}")
    print(f"  Symbols: {df.index.get_level_values(1).nunique()}")
    print(f"  Columns: {list(df.columns)}")

    return df


def walkforward_splits(
    timestamps,
    train_months=12,
    val_months=1,
    embargo_periods=10,
    min_train_obs=100
):
    """
    Generate walk-forward train/validation splits with embargo

    Args:
        timestamps: DatetimeIndex or array of timestamps
        train_months: Number of months for training window
        val_months: Number of months for validation window
        embargo_periods: Number of periods to embargo between train/val
        min_train_obs: Minimum observations required for training

    Yields:
        (train_idx, val_idx): Tuples of boolean arrays for train and validation
    """
    timestamps = pd.DatetimeIndex(timestamps).unique().sort_values()

    # Convert months to approximate periods (assuming ~21 trading days/month, 2 sessions/day)
    train_periods = train_months * 21 * 2
    val_periods = val_months * 21 * 2

    print(f"\n{'='*80}")
    print(f"Walk-Forward Configuration:")
    print(f"  Train: {train_months} months (~{train_periods} periods)")
    print(f"  Validation: {val_months} months (~{val_periods} periods)")
    print(f"  Embargo: {embargo_periods} periods")
    print(f"{'='*80}\n")

    fold = 0

    # Start from train_periods + embargo_periods
    start_idx = train_periods + embargo_periods

    while start_idx + val_periods <= len(timestamps):
        # Train window
        train_start = start_idx - train_periods - embargo_periods
        train_end = start_idx - embargo_periods

        # Validation window
        val_start = start_idx
        val_end = start_idx + val_periods

        # Create boolean masks
        train_timestamps = timestamps[train_start:train_end]
        val_timestamps = timestamps[val_start:val_end]

        if len(train_timestamps) < min_train_obs:
            print(f"⚠ Skipping fold {fold}: insufficient train data ({len(train_timestamps)} < {min_train_obs})")
            start_idx += val_periods
            continue

        fold += 1

        print(f"\nFold {fold}:")
        print(f"  Train: {train_timestamps[0]} to {train_timestamps[-1]} ({len(train_timestamps)} periods)")
        print(f"  Val:   {val_timestamps[0]} to {val_timestamps[-1]} ({len(val_timestamps)} periods)")

        yield train_timestamps, val_timestamps

        # Move forward by validation window
        start_idx += val_periods


def split_by_date_range(df, start_date, end_date):
    """
    Extract data for a specific date range

    Args:
        df: DataFrame with MultiIndex (timestamp, symbol)
        start_date: Start date (inclusive)
        end_date: End date (inclusive)

    Returns:
        DataFrame: Filtered data
    """
    timestamps = df.index.get_level_values(0)
    mask = (timestamps >= start_date) & (timestamps <= end_date)
    return df[mask]


def get_common_symbols(df, min_obs_per_symbol=50):
    """
    Get symbols that have sufficient observations

    Args:
        df: DataFrame with MultiIndex (timestamp, symbol)
        min_obs_per_symbol: Minimum required observations

    Returns:
        list: Symbols meeting criteria
    """
    symbol_counts = df.groupby(level='symbol').size()
    valid_symbols = symbol_counts[symbol_counts >= min_obs_per_symbol].index.tolist()

    print(f"✓ Found {len(valid_symbols)} symbols with >= {min_obs_per_symbol} observations")

    return valid_symbols


def filter_symbols(df, symbols):
    """
    Filter DataFrame to include only specified symbols

    Args:
        df: DataFrame with MultiIndex (timestamp, symbol)
        symbols: List of symbols to keep

    Returns:
        DataFrame: Filtered data
    """
    df_filtered = df[df.index.get_level_values(1).isin(symbols)]

    print(f"✓ Filtered to {len(df_filtered)} observations across {len(symbols)} symbols")

    return df_filtered


def create_embargo_mask(timestamps, embargo_start, embargo_end):
    """
    Create boolean mask for embargo period

    Args:
        timestamps: DatetimeIndex
        embargo_start: Start of embargo
        embargo_end: End of embargo

    Returns:
        Boolean array: True for non-embargoed periods
    """
    return ~((timestamps >= embargo_start) & (timestamps <= embargo_end))


class DataSplitter:
    """
    Helper class for managing data splits with embargo
    """

    def __init__(self, panel, train_months=12, val_months=1, embargo_periods=10):
        """
        Initialize data splitter

        Args:
            panel: DataFrame with MultiIndex (timestamp, symbol)
            train_months: Training window in months
            val_months: Validation window in months
            embargo_periods: Embargo periods between train/val
        """
        self.panel = panel
        self.train_months = train_months
        self.val_months = val_months
        self.embargo_periods = embargo_periods

        self.timestamps = panel.index.get_level_values(0).unique().sort_values()

    def get_splits(self):
        """
        Get all walk-forward splits

        Yields:
            (train_df, val_df): Tuples of train and validation DataFrames
        """
        for train_ts, val_ts in walkforward_splits(
            self.timestamps,
            self.train_months,
            self.val_months,
            self.embargo_periods
        ):
            # Extract train data
            train_mask = self.panel.index.get_level_values(0).isin(train_ts)
            train_df = self.panel[train_mask]

            # Extract validation data
            val_mask = self.panel.index.get_level_values(0).isin(val_ts)
            val_df = self.panel[val_mask]

            yield train_df, val_df

    def get_fold(self, fold_idx):
        """
        Get specific fold

        Args:
            fold_idx: Fold index (0-based)

        Returns:
            (train_df, val_df): Train and validation DataFrames
        """
        for i, (train_df, val_df) in enumerate(self.get_splits()):
            if i == fold_idx:
                return train_df, val_df

        raise IndexError(f"Fold {fold_idx} not found")

    def count_folds(self):
        """Count total number of folds"""
        return sum(1 for _ in self.get_splits())
