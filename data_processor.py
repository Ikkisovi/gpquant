"""
Data processing utilities for daily_am_pm_data.csv
Handles multi-stock intraday data with AM/PM sessions
"""
import pandas as pd
import numpy as np


def load_daily_am_pm_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess the daily AM/PM data

    Args:
        file_path: Path to daily_am_pm_data.csv

    Returns:
        DataFrame with processed data
    """
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    df = df.sort_values(['symbol', 'timestamp'])
    return df


def prepare_market_data(df: pd.DataFrame, symbol: str, slippage: float = 0.001) -> pd.DataFrame:
    """
    Prepare market data for a single symbol in the format required by Backtester

    Args:
        df: DataFrame with all symbols
        symbol: Stock symbol to extract
        slippage: Slippage ratio for A (ask) and B (bid) calculation

    Returns:
        DataFrame with columns: dt, O, H, L, C, V, A, B, vwap, mktcap, turnover
    """
    symbol_df = df[df['symbol'] == symbol].copy()
    symbol_df = symbol_df.sort_values('timestamp')

    # Rename columns to match framework convention
    symbol_df['dt'] = symbol_df['timestamp']
    symbol_df['O'] = symbol_df['open']
    symbol_df['H'] = symbol_df['high']
    symbol_df['L'] = symbol_df['low']
    symbol_df['C'] = symbol_df['close']
    symbol_df['V'] = symbol_df['volume']

    # Calculate ask (A) and bid (B) prices with slippage
    symbol_df['A'] = symbol_df['C'] * (1 + slippage)
    symbol_df['B'] = symbol_df['C'] * (1 - slippage)

    # Select required columns
    result_df = symbol_df[['dt', 'O', 'H', 'L', 'C', 'V', 'A', 'B',
                            'vwap', 'mktcap', 'turnover']].copy()
    result_df = result_df.reset_index(drop=True)

    return result_df


def get_all_symbols(df: pd.DataFrame) -> list:
    """Get list of all unique symbols in the dataset"""
    return sorted(df['symbol'].unique().tolist())


def calculate_equal_weight_benchmark(df: pd.DataFrame, symbols: list = None) -> pd.Series:
    """
    Calculate equal-weight portfolio returns as benchmark

    Args:
        df: DataFrame with all symbols
        symbols: List of symbols to include (if None, use all)

    Returns:
        Series with equal-weight portfolio close prices indexed by timestamp
    """
    if symbols is None:
        symbols = get_all_symbols(df)

    # Pivot to get close prices for all symbols
    pivot_df = df[df['symbol'].isin(symbols)].pivot_table(
        index='timestamp',
        columns='symbol',
        values='close'
    )

    # Calculate equal-weight returns
    # Fill missing values forward then backward
    pivot_df = pivot_df.fillna(method='ffill').fillna(method='bfill')

    # Equal weight portfolio: average of all stocks
    ew_portfolio = pivot_df.mean(axis=1)

    return ew_portfolio


def calculate_max_drawdown(asset_series: pd.Series) -> float:
    """
    Calculate maximum drawdown of an asset series

    Args:
        asset_series: Time series of asset values

    Returns:
        Maximum drawdown as a decimal (e.g., 0.15 for 15%)
    """
    cummax = asset_series.cummax()
    drawdown = (asset_series - cummax) / cummax
    max_dd = drawdown.min()
    return abs(max_dd)


def calculate_total_return(asset_series: pd.Series) -> float:
    """
    Calculate total return of an asset series

    Args:
        asset_series: Time series of asset values

    Returns:
        Total return as a decimal
    """
    return (asset_series.iloc[-1] - asset_series.iloc[0]) / asset_series.iloc[0]


def check_tracking_constraints(
    strategy_asset: pd.Series,
    benchmark_asset: pd.Series,
    max_dd_excess: float = 0.05
) -> dict:
    """
    Check if strategy meets tracking constraints

    Args:
        strategy_asset: Strategy asset series
        benchmark_asset: Benchmark asset series
        max_dd_excess: Maximum allowed excess drawdown (default 5%)

    Returns:
        Dictionary with constraint check results
    """
    strategy_dd = calculate_max_drawdown(strategy_asset)
    benchmark_dd = calculate_max_drawdown(benchmark_asset)
    strategy_return = calculate_total_return(strategy_asset)
    benchmark_return = calculate_total_return(benchmark_asset)

    dd_excess = strategy_dd - benchmark_dd
    meets_dd_constraint = dd_excess <= max_dd_excess
    meets_return_constraint = strategy_return > benchmark_return

    return {
        'strategy_max_drawdown': strategy_dd,
        'benchmark_max_drawdown': benchmark_dd,
        'drawdown_excess': dd_excess,
        'meets_drawdown_constraint': meets_dd_constraint,
        'strategy_return': strategy_return,
        'benchmark_return': benchmark_return,
        'meets_return_constraint': meets_return_constraint,
        'meets_all_constraints': meets_dd_constraint and meets_return_constraint
    }
