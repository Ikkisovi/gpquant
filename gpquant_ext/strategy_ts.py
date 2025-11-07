"""
Fixed Strategy Template: Line-Deviation to Position Mapping
This is NOT searched by GP - it's a fixed monotonic mapping
GP only searches for the adaptive line f(t)
"""
import numpy as np
import pandas as pd


def map_f_to_positions(f, prices, returns, k=1.0, z_L=40, cost_bps=0.0005, vol_window=60):
    """
    Map factor f (the "adaptive line") to trading positions using fixed template

    Logic:
        1. g = (price - f) / price     # Deviation from line
        2. z = zscore(g, lookback=z_L) # Standardize
        3. w_raw = tanh(k * z)          # Monotonic mapping to [-1, 1]
        4. w = w_raw / vol              # Risk-adjust
        5. w = w.shift(1)               # t-1 execution (no look-ahead)
        6. net_return = port_return - transaction_cost

    Args:
        f: DataFrame, factor values (MultiIndex: timestamp, symbol)
        prices: DataFrame, close prices (MultiIndex: timestamp, symbol)
        returns: DataFrame, ret1 (MultiIndex: timestamp, symbol)
        k: tanh strength parameter
        z_L: zscore lookback window
        cost_bps: transaction cost in bps (one-side)
        vol_window: rolling vol window for risk-adjustment

    Returns:
        w: position weights DataFrame
        net: net returns Series (aggregated across symbols)
        metrics: dict with turnover, gross returns, costs
    """
    # Unstack to wide format (timestamps x symbols)
    px = prices['close'].unstack('symbol')
    ff = f.unstack('symbol').reindex_like(px)
    ret_wide = returns['ret1'].unstack('symbol').reindex_like(px)

    # Step 1: Deviation from line
    g = (px - ff) / (px + 1e-12)  # Prevent division by zero

    # Step 2: Rolling z-score (per symbol independently)
    z = (g - g.rolling(z_L, min_periods=max(1, z_L//2)).mean()) / \
        (g.rolling(z_L, min_periods=max(1, z_L//2)).std() + 1e-12)

    # Step 3: Monotonic mapping to positions
    w_raw = np.tanh(k * z)

    # Step 4: Risk-adjustment (optional but recommended)
    # Volatility per symbol
    vol = ret_wide.rolling(vol_window, min_periods=max(1, vol_window//2)).std() * np.sqrt(504)
    vol = vol.replace(0, np.nan)  # Avoid zero vol
    w_risk_adj = w_raw / vol

    # Step 5: Normalize total weight to 1 (or keep as is for leveraged)
    # Here we normalize to sum of absolute weights = 1
    w_sum_abs = w_risk_adj.abs().sum(axis=1)
    w = w_risk_adj.div(w_sum_abs.replace(0, np.nan), axis=0).fillna(0)

    # Step 6: t-1 execution (shift positions)
    w = w.shift(1).fillna(0)

    # ========================================================================
    # Calculate Returns
    # ========================================================================

    # Gross return (before costs)
    gross_ret = (w * ret_wide).sum(axis=1)

    # Turnover (sum of absolute weight changes)
    turnover = (w - w.shift(1)).abs().sum(axis=1).fillna(0)

    # Transaction cost
    transaction_cost = cost_bps * turnover

    # Net return
    net = gross_ret - transaction_cost

    # ========================================================================
    # Return metrics
    # ========================================================================

    metrics = {
        'turnover': turnover,
        'gross_returns': gross_ret,
        'transaction_costs': transaction_cost,
        'mean_turnover': turnover.mean(),
        'mean_gross': gross_ret.mean(),
        'mean_cost': transaction_cost.mean(),
    }

    return w, net, metrics


def backtest_factor(factor, prices, returns, **strategy_kwargs):
    """
    Convenience wrapper for backtesting a factor

    Args:
        factor: Factor values (can be Series or DataFrame)
        prices: Close prices DataFrame
        returns: Returns DataFrame
        **strategy_kwargs: Arguments for map_f_to_positions

    Returns:
        net: Net return series
        metrics: Performance metrics dict
    """
    # Ensure factor is DataFrame with same structure as prices
    if isinstance(factor, pd.Series):
        factor = factor.to_frame('factor')

    w, net, trade_metrics = map_f_to_positions(
        factor,
        prices,
        returns,
        **strategy_kwargs
    )

    # Calculate performance metrics
    equity = (1 + net).cumprod()
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax

    metrics = {
        **trade_metrics,
        'total_return': equity.iloc[-1] - 1,
        'max_drawdown': drawdown.min(),
        'sharpe': net.mean() / (net.std() + 1e-12) * np.sqrt(504),  # Annualized
        'mean_return': net.mean(),
        'std_return': net.std(),
    }

    return net, metrics


def validate_strategy_output(net, returns):
    """
    Validate strategy output for sanity checks

    Args:
        net: Net return series
        returns: Original returns DataFrame

    Returns:
        bool: True if valid, raises ValueError otherwise
    """
    # Check for extreme values
    if net.abs().max() > 0.5:  # 50% single-period return is suspicious
        raise ValueError(f"Extreme return detected: {net.abs().max():.2%}")

    # Check for too many NaNs
    nan_pct = net.isna().sum() / len(net)
    if nan_pct > 0.5:
        raise ValueError(f"Too many NaN values: {nan_pct:.1%}")

    # Check alignment
    if len(net) != len(returns['ret1'].unstack('symbol')):
        raise ValueError("Length mismatch between net returns and input returns")

    return True
