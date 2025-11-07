"""
Benchmark: Equal-Weight Buy-and-Hold
Pool-wide equal-weight portfolio held throughout the window
"""
import numpy as np
import pandas as pd


def equal_weight_buy_hold(returns_df, rebalance=False, rebalance_freq=None):
    """
    Calculate equal-weight buy-and-hold benchmark

    Args:
        returns_df: DataFrame with MultiIndex (timestamp, symbol), column 'ret1'
                    OR Series with same MultiIndex
        rebalance: If True, rebalance at regular intervals
        rebalance_freq: Rebalancing frequency (e.g., 20 for every 20 periods)

    Returns:
        Series: Benchmark returns (indexed by timestamp)
    """
    # Handle both DataFrame and Series input
    if isinstance(returns_df, pd.DataFrame):
        if 'ret1' in returns_df.columns:
            rets = returns_df['ret1']
        else:
            # Assume single column
            rets = returns_df.iloc[:, 0]
    else:
        rets = returns_df

    # Convert to wide format (timestamps x symbols)
    rets_wide = rets.unstack('symbol')

    if rebalance and rebalance_freq is not None:
        # Rebalancing strategy
        port_returns = []
        for i in range(0, len(rets_wide), rebalance_freq):
            window = rets_wide.iloc[i:i+rebalance_freq]
            # Equal weight at start of window
            n_assets = (~window.iloc[0].isna()).sum()
            if n_assets > 0:
                weight = 1.0 / n_assets
                window_port = window.fillna(0).sum(axis=1) * weight
                port_returns.append(window_port)

        benchmark = pd.concat(port_returns)
    else:
        # Pure buy-and-hold: equal weight at t=0, no rebalancing
        # Available assets at first timestamp
        first_row = rets_wide.iloc[0]
        available = ~first_row.isna()
        n_assets = available.sum()

        if n_assets == 0:
            raise ValueError("No assets available at first timestamp")

        # Equal weight
        weight = 1.0 / n_assets

        # Simple average of all assets (equal weight maintained through drift)
        benchmark = rets_wide.mean(axis=1, skipna=True)

    return benchmark


def pool_equal_weight_with_drift(returns_df):
    """
    Equal-weight buy-and-hold with natural drift (most realistic)

    Start with equal dollar amounts, let weights drift with returns
    This is the true "buy and hold" benchmark

    Args:
        returns_df: Returns DataFrame

    Returns:
        Series: Benchmark returns
    """
    # Handle input
    if isinstance(returns_df, pd.DataFrame):
        if 'ret1' in returns_df.columns:
            rets = returns_df['ret1']
        else:
            rets = returns_df.iloc[:, 0]
    else:
        rets = returns_df

    rets_wide = rets.unstack('symbol').fillna(0)

    # Start with equal weights
    n_assets = rets_wide.shape[1]
    weights = np.ones(n_assets) / n_assets

    portfolio_returns = []

    for t in range(len(rets_wide)):
        # Current period returns
        period_rets = rets_wide.iloc[t].values

        # Portfolio return
        port_ret = np.sum(weights * period_rets)
        portfolio_returns.append(port_ret)

        # Update weights (drift with returns)
        weights = weights * (1 + period_rets)
        weights = weights / weights.sum()  # Renormalize

    benchmark = pd.Series(portfolio_returns, index=rets_wide.index)
    return benchmark


def calculate_tracking_error(strategy_returns, benchmark_returns, ann_factor=504):
    """
    Calculate tracking error (volatility of excess returns)

    Args:
        strategy_returns: Series
        benchmark_returns: Series
        ann_factor: Annualization factor

    Returns:
        float: Annualized tracking error
    """
    excess = (strategy_returns - benchmark_returns).dropna()
    te = excess.std(ddof=1) * np.sqrt(ann_factor)
    return te


def calculate_information_ratio(strategy_returns, benchmark_returns, ann_factor=504):
    """
    Calculate Information Ratio = excess_return / tracking_error

    Args:
        strategy_returns: Series
        benchmark_returns: Series
        ann_factor: Annualization factor

    Returns:
        float: Information Ratio
    """
    excess = (strategy_returns - benchmark_returns).dropna()

    if len(excess) < 2:
        return 0.0

    ann_excess = excess.mean() * ann_factor
    te = excess.std(ddof=1) * np.sqrt(ann_factor)

    if te < 1e-12:
        return 0.0

    IR = ann_excess / te
    return IR
