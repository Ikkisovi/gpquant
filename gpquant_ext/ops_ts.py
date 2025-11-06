"""
Time Series Operators for GPQuant
Pure time-series operations (no cross-sectional leakage)
All operations work on single-asset time series independently
"""
import numpy as np
import pandas as pd
from gpquant.Function import Function


# ============================================================================
# Basic Time Series Operators
# ============================================================================

def _ts_lag(x, n: int):
    """Lag by n periods (time-series safe)"""
    return pd.Series(x).shift(n).values


def _ts_ema(x, n: int):
    """Exponential moving average"""
    alpha = 2.0 / (n + 1)
    return pd.Series(x).ewm(alpha=alpha, adjust=False).mean().values


def _ts_sma(x, n: int):
    """Simple moving average"""
    return pd.Series(x).rolling(n, min_periods=max(1, n//2)).mean().values


def _ts_std_dev(x, n: int):
    """Rolling standard deviation"""
    return pd.Series(x).rolling(n, min_periods=max(1, n//2)).std().values


def _ts_zscore(x, n: int):
    """Rolling z-score normalization"""
    s = pd.Series(x)
    mu = s.rolling(n, min_periods=max(1, n//2)).mean()
    sigma = s.rolling(n, min_periods=max(1, n//2)).std()
    return ((s - mu) / (sigma + 1e-12)).values


def _ts_decay(x, n: int):
    """Linear decay weighted moving average"""
    weights = np.arange(1, n+1)
    weights = weights / weights.sum()

    def apply_decay(window):
        if len(window) < n:
            return np.nan
        return np.sum(window[-n:] * weights)

    return pd.Series(x).rolling(n).apply(apply_decay, raw=True).values


def _ts_pct_change(x, n: int):
    """Percentage change over n periods"""
    s = pd.Series(x)
    return (s / s.shift(n) - 1).values


# ============================================================================
# Risk Metrics (Time Series)
# ============================================================================

def _ts_sortino(x, n: int):
    """
    Rolling Sortino ratio (time-series version)
    Only considers downside deviation
    """
    s = pd.Series(x)
    returns = s.pct_change()

    def sortino_window(window):
        if len(window) < 2:
            return np.nan
        mean_ret = np.mean(window)
        downside = window[window < 0]
        if len(downside) == 0:
            return np.nan
        downside_std = np.std(downside, ddof=1)
        if downside_std < 1e-12:
            return np.nan
        return mean_ret / downside_std

    return returns.rolling(n).apply(sortino_window, raw=True).values


def _ts_drawdown_log(x, n: int):
    """
    Rolling logarithmic drawdown (time-series version)
    Returns log(1 - drawdown) where drawdown is relative to rolling max
    More stable than raw drawdown
    """
    s = pd.Series(x)

    def dd_log_window(window):
        if len(window) < 2:
            return 0.0
        cummax = np.maximum.accumulate(window)
        dd = 1.0 - window[-1] / cummax[-1] if cummax[-1] > 0 else 0.0
        dd = np.clip(dd, 0.0, 0.9999)  # Prevent log(0)
        return np.log(1.0 - dd + 1e-8)

    return s.rolling(n).apply(dd_log_window, raw=True).values


def _ts_max_drawdown(x, n: int):
    """Rolling maximum drawdown"""
    s = pd.Series(x)

    def max_dd_window(window):
        if len(window) < 2:
            return 0.0
        cummax = np.maximum.accumulate(window)
        dd = 1.0 - window / cummax
        return np.max(dd)

    return s.rolling(n).apply(max_dd_window, raw=True).values


# ============================================================================
# Rank and Percentile (Time Series Version - No Cross-Sectional)
# ============================================================================

def _ts_rank_pct(x, n: int):
    """
    Rolling percentile rank (0-1)
    Current value's position in rolling window
    Pure time-series, no cross-sectional comparison
    """
    s = pd.Series(x)

    def rank_window(window):
        if len(window) < 2:
            return 0.5
        current = window[-1]
        rank = np.sum(window <= current) / len(window)
        return rank

    return s.rolling(n).apply(rank_window, raw=True).values


# ============================================================================
# Price-Volume Derivatives
# ============================================================================

def _price_to_vwap(price, vwap):
    """Price deviation from VWAP"""
    return (price / vwap - 1.0)


def _vwap_slope(vwap, n: int = 10):
    """VWAP slope (standardized)"""
    s = pd.Series(vwap)
    ma = s.rolling(n, min_periods=max(1, n//2)).mean()
    std = s.rolling(n, min_periods=max(1, n//2)).std()
    return ((s - ma) / (std + 1e-12)).values


# ============================================================================
# Robust Operators (Winsorization + Huber)
# ============================================================================

def _winsorize(x, lower=0.01, upper=0.99):
    """Winsorize to [lower, upper] percentiles"""
    s = pd.Series(x)
    lower_val = s.quantile(lower)
    upper_val = s.quantile(upper)
    return s.clip(lower_val, upper_val).values


def _huber_transform(x, threshold=1.5):
    """Huber transformation (reduce extreme values)"""
    s = pd.Series(x)
    median = s.median()
    mad = (s - median).abs().median()
    z = (s - median) / (mad + 1e-12)

    # Huber: linear beyond threshold
    huber = np.where(
        np.abs(z) <= threshold,
        z,
        threshold * np.sign(z)
    )
    return huber


# ============================================================================
# Register to GPQuant Function Map
# ============================================================================

# Time series functions with period parameter
ts_lag_func = Function(function=_ts_lag, name="ts_lag", arity=2, is_ts=1)
ts_ema_func = Function(function=_ts_ema, name="ts_ema", arity=2, is_ts=1)
ts_sma_func = Function(function=_ts_sma, name="ts_sma", arity=2, is_ts=1)
ts_std_func = Function(function=_ts_std_dev, name="ts_std", arity=2, is_ts=1)
ts_zscore_func = Function(function=_ts_zscore, name="ts_zscore", arity=2, is_ts=1)
ts_decay_func = Function(function=_ts_decay, name="ts_decay", arity=2, is_ts=1)
ts_pct_change_func = Function(function=_ts_pct_change, name="ts_pct_change", arity=2, is_ts=1)

# Risk metrics
ts_sortino_func = Function(function=_ts_sortino, name="ts_sortino", arity=2, is_ts=1)
ts_drawdown_log_func = Function(function=_ts_drawdown_log, name="ts_drawdown_log", arity=2, is_ts=1)
ts_max_dd_func = Function(function=_ts_max_drawdown, name="ts_max_dd", arity=2, is_ts=1)

# Rank
ts_rank_pct_func = Function(function=_ts_rank_pct, name="ts_rank_pct", arity=2, is_ts=1)

# Price-volume (binary)
price_to_vwap_func = Function(function=_price_to_vwap, name="price_to_vwap", arity=2)
vwap_slope_func = Function(function=_vwap_slope, name="vwap_slope", arity=2, is_ts=1)

# Robust transforms
winsorize_func = Function(function=_winsorize, name="winsorize", arity=1)
huber_func = Function(function=_huber_transform, name="huber", arity=1)


# Complete operator map for registration
TS_OPERATOR_MAP = {
    "ts_lag": ts_lag_func,
    "ts_ema": ts_ema_func,
    "ts_sma": ts_sma_func,
    "ts_std": ts_std_func,
    "ts_zscore": ts_zscore_func,
    "ts_decay": ts_decay_func,
    "ts_pct_change": ts_pct_change_func,
    "ts_sortino": ts_sortino_func,
    "ts_drawdown_log": ts_drawdown_log_func,
    "ts_max_dd": ts_max_dd_func,
    "ts_rank_pct": ts_rank_pct_func,
    "price_to_vwap": price_to_vwap_func,
    "vwap_slope": vwap_slope_func,
    "winsorize": winsorize_func,
    "huber": huber_func,
}


def register_ts_ops():
    """
    Register time-series operators to GPQuant function_map
    Call this before creating SymbolicRegressor
    """
    from gpquant.Function import function_map

    # Add our operators
    function_map.update(TS_OPERATOR_MAP)

    print(f"✓ Registered {len(TS_OPERATOR_MAP)} time-series operators to GPQuant")
    return list(TS_OPERATOR_MAP.keys())


def get_ts_whitelist():
    """
    Get whitelist of allowed functions for time-series strategy
    This restricts GP to only use time-series safe operations
    """
    # Basic math (from gpquant original)
    basic_math = ["add", "sub", "mul", "div", "abs", "sqrt", "log", "sign"]

    # Our time-series operators
    ts_ops = list(TS_OPERATOR_MAP.keys())

    # Combine
    whitelist = basic_math + ts_ops

    return whitelist
