# Three-Line Strategy for Daily AM/PM Data

## Overview

This implementation provides a sophisticated trading strategy based on a three-line system with confirmation signals, specifically designed for the `daily_am_pm_data.csv` dataset.

## Strategy Concept

The strategy is inspired by Bollinger Bands but with a key innovation:

### Core Components

1. **Centerline** (Main Signal Line)
   - Moving average that serves as the primary directional indicator
   - Price crossing above = potential bullish signal
   - Price crossing below = potential bearish signal

2. **Filter Lines** (Regime Detection)
   - Upper and lower bands (based on standard deviation)
   - Detect whether market is in trending or ranging mode
   - In trending markets: price approaches bands
   - In ranging markets: price oscillates around centerline

3. **Confirmation Signal** (GP-Evolved Factor)
   - Machine learning-generated factor using genetic programming
   - Acts as confirmation filter to prevent false signals
   - Trained on historical data to identify true trend changes

### Trading Logic

The strategy addresses a critical problem: **simple centerline crossover strategies fail in ranging markets**.

**Solution:** Require BOTH:
- Price position relative to centerline (directional signal)
- Factor confirmation OR proximity to bands (trend strength filter)

**Entry Signals:**
- **Long**: Price above centerline AND (factor confirms OR near upper band)
- **Short**: Price below centerline AND (factor confirms OR near lower band)

**Exit Signals:**
- Exit when centerline is crossed in opposite direction
- Exit when factor flips to opposite signal

## Two Strategy Variants

### 1. Standard Three-Line (`three_line`)
- Uses centerline with bands for regime detection
- More responsive to market changes
- Parameters:
  - `d_center`: Period for centerline MA (default: 20)
  - `d_band`: Period for band calculation (default: 20)
  - `band_width`: Band width in std devs (default: 2.0)
  - `factor_threshold`: Threshold for factor confirmation (default: 0.5)

### 2. Dual Confirmation (`three_line_dual`)
- Uses three moving averages (fast, center, slow)
- Stricter entry requirements (all three lines + factor)
- More conservative, fewer but higher quality trades
- Parameters:
  - `d_center`: Period for center line (default: 20)
  - `d_fast`: Period for fast line (default: 10)
  - `d_slow`: Period for slow line (default: 40)
  - `require_both`: Require all confirmations (default: True)

## Tracking Constraints

The strategy implements two critical constraints:

1. **Maximum Drawdown Constraint**
   - Strategy max drawdown cannot exceed benchmark by more than 5%
   - Benchmark = equal-weight portfolio of all stocks in the pool
   - Ensures risk management relative to passive holding

2. **Return Constraint**
   - Absolute return must be higher than buy-and-hold benchmark
   - Ensures the active strategy adds value over passive approach

## Dataset Structure

The `daily_am_pm_data.csv` contains:
- **symbol**: Stock ticker
- **timestamp**: Date and time
- **date**: Trading date
- **session**: AM or PM (intraday sessions)
- **open, high, low, close**: OHLC prices
- **volume**: Trading volume
- **vwap**: Volume-weighted average price
- **mktcap**: Market capitalization
- **turnover**: Turnover ratio

## Usage

### Quick Demo

Run a quick test with small population and few generations:

```bash
python demo_three_line.py
```

This will:
1. Load the data
2. Train on a single symbol with minimal GP parameters
3. Evaluate on test set
4. Check tracking constraints
5. Display results

### Full Training

Train on multiple symbols with proper parameters:

```bash
python train_three_line_strategy.py
```

This will:
1. Train on multiple symbols (configurable)
2. Use both strategy variants
3. Split data into train/validation/test
4. Evaluate all sets
5. Check constraints on all sets
6. Provide comprehensive summary

### Custom Training

```python
from gpquant.SymbolicRegressor import SymbolicRegressor
from data_processor import load_daily_am_pm_data, prepare_market_data

# Load data
df = load_daily_am_pm_data("daily_am_pm_data.csv")
market_df = prepare_market_data(df, "agx", slippage=0.001)

# Configure strategy
transformer_kwargs = {
    "init_cash": 10000,
    "charge_ratio": 0.0002,
    "d_center": 20,
    "d_band": 20,
    "band_width": 2.0,
    "factor_threshold": 0.5,
    "price": market_df["C"].values,
}

# Initialize GP
sr = SymbolicRegressor(
    population_size=500,
    tournament_size=20,
    generations=30,
    stopping_criteria=2.5,
    p_crossover=0.7,
    p_subtree_mutate=0.15,
    p_hoist_mutate=0.1,
    p_point_mutate=0.05,
    init_depth=(4, 7),
    init_method="half and half",
    function_set=[],
    variable_set=["O", "H", "L", "C", "V", "vwap", "mktcap", "turnover"],
    const_range=(1, 20),
    ts_const_range=(1, 30),
    build_preference=[0.75, 0.75],
    metric="sharpe ratio",
    transformer="three_line",
    transformer_kwargs=transformer_kwargs,
    parsimony_coefficient=0.001,
)

# Train
sr.fit(market_df, market_df["C"])

# Evaluate
test_score = sr.score(test_df, test_df["C"])
```

## Files

- `data_processor.py`: Data loading and preprocessing utilities
- `demo_three_line.py`: Quick demo script
- `train_three_line_strategy.py`: Full training script
- `gpquant/Backtester.py`: Contains strategy implementations
  - `_strategy_three_line_confirmation`: Standard three-line strategy
  - `_strategy_three_line_dual_confirmation`: Dual confirmation strategy
- `gpquant/Fitness.py`: Contains fitness functions including tracking constraints
  - `_tracking_constrained_sharpe`: Sharpe with drawdown/return constraints
  - `_max_drawdown`: Maximum drawdown calculation
  - `_calmar_ratio`: Return/drawdown ratio

## Key Modifications to Framework

### Added to `Backtester.py`:
1. `_strategy_three_line_confirmation()`: Standard three-line strategy
2. `_strategy_three_line_dual_confirmation()`: Stricter dual confirmation
3. Registered in `backtester_map` as `"three_line"` and `"three_line_dual"`

### Added to `Fitness.py`:
1. `_max_drawdown()`: Calculate maximum drawdown
2. `_calmar_ratio()`: Calmar ratio (return/drawdown)
3. `_tracking_constrained_sharpe()`: Sharpe with tracking constraints
4. Registered in `fitness_map`

### New Utilities:
1. `data_processor.py`: Complete data handling for AM/PM dataset
2. Functions for benchmark calculation and constraint checking

## Performance Metrics

The strategy is evaluated on:
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Total Return**: Absolute performance
- **Constraint Compliance**: Whether tracking rules are met

## Tips for Best Results

1. **Data Split**: Use proper train/val/test split (60/20/20)
2. **Population Size**: Larger populations (500-2000) find better factors
3. **Generations**: Run 30-50 generations for convergence
4. **Variable Set**: Include relevant features (OHLC, volume, vwap, mktcap)
5. **Strategy Choice**:
   - Use `three_line` for more responsive trading
   - Use `three_line_dual` for conservative, high-conviction trades
6. **Parameters**:
   - Shorter periods (10-20) for intraday AM/PM data
   - Adjust `band_width` and `factor_threshold` based on volatility

## Example Output

```
============================================================
RESULTS
============================================================

Strategy Performance:
  Sharpe Ratio:     1.85
  Max Drawdown:     12.5%
  Total Return:     23.4%

Benchmark Performance:
  Max Drawdown:     15.2%
  Total Return:     18.7%

Constraint Check:
  Drawdown Excess:  -2.7% (limit: 5%)
  Return Excess:    4.7%
  Status:           ✓ PASS
```

## Future Enhancements

Potential improvements:
1. Adaptive parameter tuning based on volatility regime
2. Multi-symbol portfolio optimization
3. Intraday session-specific parameters (AM vs PM)
4. Risk-based position sizing
5. Dynamic stop-loss and take-profit levels
6. Machine learning for regime classification

## Contact

For questions or issues, please refer to the gpquant documentation or create an issue in the repository.
