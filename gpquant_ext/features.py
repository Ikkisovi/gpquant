"""
Feature Engineering for Time-Series Strategy
All features are computed per-asset independently (no cross-sectional leakage)
Returns features suitable for GPQuant to search over
"""
import numpy as np
import pandas as pd
from .ops_ts import (
    _ts_ema, _ts_sma, _ts_std_dev, _ts_zscore,
    _ts_sortino, _ts_drawdown_log, _ts_rank_pct,
    _winsorize, _huber_transform, _price_to_vwap, _vwap_slope
)


def zscore_by_symbol(series, window):
    """
    Compute rolling z-score per symbol (time-series, no cross-sectional)
    series: MultiIndex (timestamp, symbol)
    """
    def zscore_ts(x):
        return _ts_zscore(x.values, window)

    return series.groupby(level='symbol', group_keys=False).apply(zscore_ts)


def rolling_sortino_by_symbol(returns, window):
    """Compute Sortino ratio per symbol"""
    def sortino_ts(x):
        return _ts_sortino(x.values, window)

    return returns.groupby(level='symbol', group_keys=False).apply(sortino_ts)


def rolling_dd_log_by_symbol(prices, window):
    """Compute log drawdown per symbol"""
    def dd_log_ts(x):
        return _ts_drawdown_log(x.values, window)

    return prices.groupby(level='symbol', group_keys=False).apply(dd_log_ts)


def rank_pct_by_symbol(series, window):
    """
    Compute rolling percentile rank per symbol (0-1)
    Current value's position in its own history
    """
    def rank_ts(x):
        return _ts_rank_pct(x.values, window)

    return series.groupby(level='symbol', group_keys=False).apply(rank_ts)


def ema_by_symbol(series, window):
    """EMA per symbol"""
    def ema_ts(x):
        return _ts_ema(x.values, window)

    return series.groupby(level='symbol', group_keys=False).apply(ema_ts)


def make_features(panel, lookbacks=None):
    """
    Generate feature matrix from panel data

    Args:
        panel: DataFrame with MultiIndex (timestamp, symbol)
               Columns: open, high, low, close, volume, vwap, turnover, mktcap
        lookbacks: dict of window sizes (default provided)

    Returns:
        X: Feature DataFrame (MultiIndex: timestamp, symbol)
        returns: ret1 DataFrame
        prices: close prices DataFrame
        metadata: dict with additional info
    """
    if lookbacks is None:
        lookbacks = {
            'short': 5,    # ~2.5 trading days (AM/PM)
            'medium': 20,  # ~10 trading days
            'long': 60,    # ~30 trading days
            'very_long': 120  # ~60 trading days
        }

    df = panel.sort_index()
    required_cols = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'turnover', 'mktcap']

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    print("Generating features...")

    # ========================================================================
    # Basic Price/Return Features
    # ========================================================================

    close = df['close']
    ret1 = close.groupby(level='symbol').pct_change()

    # Returns at different horizons
    ret_5 = close.groupby(level='symbol').pct_change(periods=lookbacks['short'])
    ret_20 = close.groupby(level='symbol').pct_change(periods=lookbacks['medium'])
    ret_60 = close.groupby(level='symbol').pct_change(periods=lookbacks['long'])

    # ========================================================================
    # Robust Momentum (Time-Series Version)
    # ========================================================================

    # Winsorize and Huber transform ret_20 per symbol
    def robust_mom_transform(x):
        wins = _winsorize(x.values, 0.01, 0.99)
        return pd.Series(_huber_transform(wins, threshold=1.5), index=x.index)

    ret_20_robust = ret_20.groupby(level='symbol', group_keys=False).apply(robust_mom_transform)

    # Combine recent return with smoothed EMA
    ret_20_ema = ema_by_symbol(ret_20, int(lookbacks['short']*2))
    mom_robust = 0.7 * ret_20_robust + 0.3 * ret_20_ema

    # Rank within own history (0-1 percentile)
    mom_rank_ts = rank_pct_by_symbol(mom_robust, lookbacks['very_long'])

    # ========================================================================
    # VWAP Features
    # ========================================================================

    vwap = df['vwap']
    price_to_vwap_feat = (close / vwap - 1.0).rename('price_to_vwap')

    # VWAP slope (standardized momentum of VWAP itself)
    def vwap_slope_ts(x):
        return _vwap_slope(x.values, 10)

    vwap_slope_feat = vwap.groupby(level='symbol', group_keys=False).apply(vwap_slope_ts)
    vwap_slope_feat = vwap_slope_feat.rename('vwap_slope')

    # ========================================================================
    # Turnover Features (Time-Series Standardized)
    # ========================================================================

    turnover = df['turnover']
    turnover_z = zscore_by_symbol(turnover, lookbacks['long'])
    turnover_z = turnover_z.rename('turnover_z')

    # ========================================================================
    # Market Cap Features (Time-Series Log-Standardized)
    # ========================================================================

    mktcap = df['mktcap']
    mcap_log = np.log(mktcap.clip(lower=1))
    mcap_z = zscore_by_symbol(mcap_log, lookbacks['very_long'])
    mcap_z = mcap_z.rename('mcap_z')

    # ========================================================================
    # Risk Metrics: Sortino Ratios at Different Horizons
    # ========================================================================

    sortino_5 = rolling_sortino_by_symbol(ret1, lookbacks['short'])
    sortino_20 = rolling_sortino_by_symbol(ret1, lookbacks['medium'])
    sortino_60 = rolling_sortino_by_symbol(ret1, lookbacks['long'])

    sortino_5 = sortino_5.rename('sortino_5')
    sortino_20 = sortino_20.rename('sortino_20')
    sortino_60 = sortino_60.rename('sortino_60')

    # ========================================================================
    # Risk Metrics: Log Drawdown at Different Horizons
    # ========================================================================

    dd_log_5 = rolling_dd_log_by_symbol(close, lookbacks['short'])
    dd_log_20 = rolling_dd_log_by_symbol(close, lookbacks['medium'])
    dd_log_60 = rolling_dd_log_by_symbol(close, lookbacks['long'])

    dd_log_5 = dd_log_5.rename('dd_log_5')
    dd_log_20 = dd_log_20.rename('dd_log_20')
    dd_log_60 = dd_log_60.rename('dd_log_60')

    # ========================================================================
    # Volatility Features
    # ========================================================================

    def vol_ts(x, window):
        return _ts_std_dev(x.values, window) * np.sqrt(504)  # Annualized (half-daily)

    vol_20 = ret1.groupby(level='symbol', group_keys=False).apply(lambda x: vol_ts(x, lookbacks['medium']))
    vol_60 = ret1.groupby(level='symbol', group_keys=False).apply(lambda x: vol_ts(x, lookbacks['long']))

    vol_20 = vol_20.rename('vol_20')
    vol_60 = vol_60.rename('vol_60')

    # ========================================================================
    # Assemble Feature Matrix
    # ========================================================================

    features = pd.concat([
        # Price-based
        price_to_vwap_feat,
        vwap_slope_feat,

        # Returns
        ret_5.rename('ret_5'),
        ret_20.rename('ret_20'),
        ret_60.rename('ret_60'),

        # Robust momentum
        mom_rank_ts.rename('mom_rank_ts'),

        # Volume/Liquidity
        turnover_z,

        # Market cap
        mcap_z,

        # Risk: Sortino
        sortino_5,
        sortino_20,
        sortino_60,

        # Risk: Drawdown
        dd_log_5,
        dd_log_20,
        dd_log_60,

        # Volatility
        vol_20,
        vol_60,

    ], axis=1)

    # ========================================================================
    # Also include raw price/volume for GP to use if needed
    # ========================================================================

    raw_features = pd.concat([
        close.rename('close'),
        df['volume'].rename('volume'),
        vwap.rename('vwap'),
    ], axis=1)

    X = pd.concat([features, raw_features], axis=1)

    # Drop NaNs (from rolling windows)
    X = X.dropna()

    # Align returns and prices
    returns_df = ret1.reindex(X.index).to_frame('ret1')
    prices_df = close.reindex(X.index).to_frame('close')

    # Final alignment
    common_idx = X.index.intersection(returns_df.index).intersection(prices_df.index)
    X = X.loc[common_idx]
    returns_df = returns_df.loc[common_idx]
    prices_df = prices_df.loc[common_idx]

    print(f"✓ Generated {len(X.columns)} features for {len(X)} observations")
    print(f"  Features: {list(X.columns)}")
    print(f"  Date range: {X.index.get_level_values(0).min()} to {X.index.get_level_values(0).max()}")
    print(f"  Symbols: {X.index.get_level_values(1).nunique()}")

    metadata = {
        'lookbacks': lookbacks,
        'feature_names': list(X.columns),
        'n_features': len(X.columns),
        'n_obs': len(X),
        'symbols': X.index.get_level_values(1).unique().tolist(),
        'date_range': (X.index.get_level_values(0).min(), X.index.get_level_values(0).max()),
    }

    return X, returns_df, prices_df, metadata


def get_feature_descriptions():
    """Get human-readable descriptions of all features"""
    return {
        'price_to_vwap': 'Price deviation from VWAP',
        'vwap_slope': 'Standardized VWAP momentum',
        'ret_5': '5-period return (~2.5 days)',
        'ret_20': '20-period return (~10 days)',
        'ret_60': '60-period return (~30 days)',
        'mom_rank_ts': 'Robust momentum percentile in own history (0-1)',
        'turnover_z': 'Turnover z-score (time-series)',
        'mcap_z': 'Log market cap z-score (time-series)',
        'sortino_5': 'Sortino ratio (5-period rolling)',
        'sortino_20': 'Sortino ratio (20-period rolling)',
        'sortino_60': 'Sortino ratio (60-period rolling)',
        'dd_log_5': 'Log drawdown (5-period rolling)',
        'dd_log_20': 'Log drawdown (20-period rolling)',
        'dd_log_60': 'Log drawdown (60-period rolling)',
        'vol_20': 'Annualized volatility (20-period)',
        'vol_60': 'Annualized volatility (60-period)',
        'close': 'Raw close price',
        'volume': 'Raw volume',
        'vwap': 'Raw VWAP',
    }
