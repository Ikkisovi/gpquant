import numba as nb
import numpy as np
import pandas as pd


def _signal_to_asset(
    df: pd.DataFrame, signal: np.ndarray, init_cash: float, charge_ratio: float
) -> pd.Series:
    """
    @param df: market information including 'dt', 'C', 'A' and 'B'
    @param signal: trading decision at the end of the datetime (>0: long, <0: short, 0: hold)
    @param init_cash: initial cash
    @param charge_ratio: transaction cost = amount * charge ratio
    @return: asset: asset series with DatetimeIndex
    """
    if len(signal) != len(df):
        raise ValueError("signal must be the same length as df")
    sr_signal = pd.Series(signal, index=df.index)
    sr_long = sr_signal[sr_signal > 0]
    sr_short = sr_signal[sr_signal < 0]
    impact_cost = (sr_long * (df["A"] - df["C"])).fillna(0) + (
        sr_short * (df["B"] - df["C"])
    ).fillna(0)
    transaction_cost = (
        (sr_long * df["A"]).fillna(0) - (sr_short * df["B"]).fillna(0)
    ) * charge_ratio
    raw_position = sr_signal.cumsum() - sr_signal
    change = (df["C"] - df["C"].shift()).fillna(0).values
    raw_return = raw_position * change
    asset = pd.Series(
        np.array(init_cash + (raw_return - impact_cost - transaction_cost).cumsum()),
        index=df["dt"],
    )
    return asset


class Backtester:
    def __init__(self, factor_to_signal, signal_to_asset=_signal_to_asset) -> None:
        """
        Vectorized factor backtesting (factor -> signal -> asset)
        [factor] outcome of SyntaxTree.execute(X)
        [signal] trading decision at the end of the datetime (>0: long, <0: short, =0: hold)
        [asset] backtesting result of an account applying the strategy
        """
        self.f2s = factor_to_signal  # function
        self.s2a = signal_to_asset  # function

    def __call__(
        self, df_market, factor, init_cash, charge_ratio, **kwargs
    ) -> pd.Series:
        """
        @param df_market: market information including 'datetime', 'C', 'A' and 'B'
        @param factor: time series of factor with the same length as df_market
        @param init_cash: initial cash
        @param charge_ratio: transaction cost = amount * charge ratio
        @param kwargs: arguments except factor in factor_to_signal()
        @return: asset: time series of asset with DatetimeIndex
        """
        return self.s2a(df_market, self.f2s(factor, **kwargs), init_cash, charge_ratio)


@nb.jit(nopython=True)
def __limit_max_position(signal: np.ndarray, limit: int = 1) -> np.ndarray:
    # Process the signal so that each position is not greater than 0
    """auxiliary function, such that absolute value of each element
    in signal.cumsum() is not greater than limit"""
    sum_flag = 0
    for i, num in enumerate(signal):
        if abs(sum_flag + num) > limit:
            signal[i] = 0
            continue
        sum_flag += num
    return signal


# strategy (factor_to_signal)
def _strategy_quantile(
    factor: np.ndarray,
    d: int,
    o_upper: float,
    o_lower: float,
    c_upper: float,
    c_lower: float,
) -> np.ndarray:
    sr_factor = pd.Series(factor)
    sr_factor.fillna(method="ffill", inplace=True)
    sr_o_upper = sr_factor.rolling(d).quantile(o_upper)
    sr_o_lower = sr_factor.rolling(d).quantile(o_lower)
    sr_c_upper = sr_factor.rolling(d).quantile(c_upper)
    sr_c_lower = sr_factor.rolling(d).quantile(c_lower)
    signal = np.zeros((len(factor),))
    signal[sr_factor > sr_o_upper] = 1
    signal[sr_factor < sr_c_upper] = -1
    signal[sr_factor > sr_o_lower] = -1
    signal[sr_factor < sr_c_lower] = 1
    return __limit_max_position(signal)


# strategy (three-line with confirmation)
def _strategy_three_line_confirmation(
    factor: np.ndarray,
    d_center: int = 20,
    d_band: int = 20,
    band_width: float = 2.0,
    factor_threshold: float = 0.5,
    price: np.ndarray = None,
) -> np.ndarray:
    """
    Three-line strategy with confirmation signals

    Similar to Bollinger Bands but with a twist:
    - Centerline (MA): Main signal line
    - Upper/Lower bands: Act as filters to detect trending vs ranging markets
    - Factor: GP-evolved confirmation signal

    Logic:
    - In trending markets: price crossing centerline gives direction
    - In ranging markets: price oscillates around centerline (signal fails)
    - Solution: Use factor as confirmation + bands as regime filter
    - Final signal: Need both centerline cross AND factor confirmation

    Args:
        factor: Time series factor from GP (will be normalized to -1, 0, 1)
        price: Price series (typically close price)
        d_center: Period for centerline moving average
        d_band: Period for band calculation
        band_width: Width multiplier for bands (in std devs)
        factor_threshold: Threshold for factor confirmation

    Returns:
        Trading signal array
    """
    if price is None:
        raise ValueError("price must be provided as a keyword argument")

    sr_price = pd.Series(price)
    sr_factor = pd.Series(factor)

    # Centerline: Moving average
    sr_center = sr_price.rolling(d_center).mean()

    # Bands for regime detection
    sr_ma = sr_price.rolling(d_band).mean()
    sr_std = sr_price.rolling(d_band).std()
    sr_upper = sr_ma + band_width * sr_std
    sr_lower = sr_ma - band_width * sr_std

    # Normalize factor to -1, 0, 1 signals
    sr_factor_norm = pd.Series(np.zeros(len(factor)))
    sr_factor_norm[sr_factor > factor_threshold] = 1
    sr_factor_norm[sr_factor < -factor_threshold] = -1

    # Price position relative to centerline
    price_above_center = sr_price > sr_center
    price_below_center = sr_price < sr_center

    # Regime detection: price near bands indicates strong trend
    price_near_upper = sr_price > (sr_upper * 0.95)  # Within 5% of upper band
    price_near_lower = sr_price < (sr_lower * 1.05)  # Within 5% of lower band
    in_trending = price_near_upper | price_near_lower

    signal = np.zeros((len(factor),))

    # Generate trading signals based on three-line logic
    for i in range(len(signal)):
        if i < max(d_center, d_band):
            continue

        # Bullish signal: price above center AND (factor confirms OR in strong uptrend)
        if price_above_center.iloc[i]:
            if sr_factor_norm.iloc[i] > 0 or price_near_upper.iloc[i]:
                if signal[i-1] == 0:  # Not already in position
                    signal[i] = 1  # Enter long

        # Bearish signal: price below center AND (factor confirms OR in strong downtrend)
        elif price_below_center.iloc[i]:
            if sr_factor_norm.iloc[i] < 0 or price_near_lower.iloc[i]:
                if signal[i-1] == 0:  # Not already in position
                    signal[i] = -1  # Enter short

        # Exit conditions
        else:
            if signal[i-1] > 0:  # Was long
                if price_below_center.iloc[i] or sr_factor_norm.iloc[i] < 0:
                    signal[i] = -1  # Close long
            elif signal[i-1] < 0:  # Was short
                if price_above_center.iloc[i] or sr_factor_norm.iloc[i] > 0:
                    signal[i] = 1  # Close short

    return __limit_max_position(signal)


def _strategy_three_line_dual_confirmation(
    factor: np.ndarray,
    d_center: int = 20,
    d_fast: int = 10,
    d_slow: int = 40,
    require_both: bool = True,
    price: np.ndarray = None,
) -> np.ndarray:
    """
    Three-line strategy with dual confirmation (stricter version)

    Three lines:
    1. Centerline (main signal): Medium-term MA
    2. Fast line (filter 1): Short-term MA
    3. Slow line (filter 2): Long-term MA

    Entry signals:
    - Long: Price crosses above all three lines + factor confirms
    - Short: Price crosses below all three lines + factor confirms

    This creates a more conservative strategy that only trades when
    all signals align.

    Args:
        factor: Time series factor from GP
        price: Price series (typically close price)
        d_center: Period for center line
        d_fast: Period for fast line (upper filter)
        d_slow: Period for slow line (lower filter)
        require_both: If True, require all three lines + factor; if False, center + factor

    Returns:
        Trading signal array
    """
    if price is None:
        raise ValueError("price must be provided as a keyword argument")

    sr_price = pd.Series(price)
    sr_factor = pd.Series(factor)

    # Three moving average lines
    sr_fast = sr_price.rolling(d_fast).mean()
    sr_center = sr_price.rolling(d_center).mean()
    sr_slow = sr_price.rolling(d_slow).mean()

    # Normalize factor
    factor_mean = sr_factor.rolling(20).mean()
    factor_std = sr_factor.rolling(20).std()
    sr_factor_norm = (sr_factor - factor_mean) / (factor_std + 1e-8)

    signal = np.zeros((len(factor),))

    for i in range(max(d_slow, 20), len(signal)):
        price_val = sr_price.iloc[i]
        fast_val = sr_fast.iloc[i]
        center_val = sr_center.iloc[i]
        slow_val = sr_slow.iloc[i]
        factor_val = sr_factor_norm.iloc[i]

        if pd.isna(fast_val) or pd.isna(center_val) or pd.isna(slow_val):
            continue

        if require_both:
            # Strict: Need to cross all three lines
            # Long: price > all three MAs and factor positive
            if price_val > center_val and factor_val > 0:
                if price_val > fast_val and price_val > slow_val:
                    if signal[i-1] <= 0:
                        signal[i] = 1

            # Short: price < all three MAs and factor negative
            elif price_val < center_val and factor_val < 0:
                if price_val < fast_val and price_val < slow_val:
                    if signal[i-1] >= 0:
                        signal[i] = -1
        else:
            # Relaxed: Just center + factor
            if price_val > center_val and factor_val > 0.5:
                if signal[i-1] <= 0:
                    signal[i] = 1
            elif price_val < center_val and factor_val < -0.5:
                if signal[i-1] >= 0:
                    signal[i] = -1

    return __limit_max_position(signal)


# backtester
bt_quantile = Backtester(factor_to_signal=_strategy_quantile)
bt_three_line = Backtester(factor_to_signal=_strategy_three_line_confirmation)
bt_three_line_dual = Backtester(factor_to_signal=_strategy_three_line_dual_confirmation)


backtester_map = {
    "quantile": bt_quantile,
    "three_line": bt_three_line,
    "three_line_dual": bt_three_line_dual,
}
