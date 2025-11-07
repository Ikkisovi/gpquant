# Alphagen Data Pipeline - Usage Guide

## Overview

The alphagen data pipeline now supports flexible data resolution options, allowing you to aggregate minute-level data into various time periods including daily, hourly, 30-minute, 2-hour, and AM/PM sessions.

## Supported Resolutions

### Predefined Resolutions

1. **AM/PM Sessions** (`"AM/PM"` or `"ampm"`)
   - Splits trading day at 12:00 PM ET
   - Morning session: 09:30 - 12:00
   - Afternoon session: 12:00 - 16:00
   - Default resolution for backward compatibility

2. **Daily** (`"1D"` or `"daily"`)
   - One bar per trading day
   - Aggregates entire trading session

3. **30 Minutes** (`"30T"` or `"30min"`)
   - 30-minute bars throughout the trading day
   - ~13 bars per trading day

4. **1 Hour** (`"1H"` or `"1h"`)
   - 1-hour bars throughout the trading day
   - ~6.5 bars per trading day

5. **2 Hours** (`"2H"` or `"2h"`)
   - 2-hour bars throughout the trading day
   - ~3.25 bars per trading day

### Custom Resolutions

You can also use any pandas frequency string:
- `"15T"` for 15-minute bars
- `"4H"` for 4-hour bars
- `"45T"` for 45-minute bars
- etc.

## Usage

### Command Line

```bash
# Use AM/PM resolution (default)
python alphagen_data_pipeline/main_build.py

# Use daily resolution
python alphagen_data_pipeline/main_build.py --resolution 1D

# Use 30-minute resolution
python alphagen_data_pipeline/main_build.py --resolution 30T

# Use 2-hour resolution
python alphagen_data_pipeline/main_build.py --resolution 2H

# Use custom 15-minute resolution
python alphagen_data_pipeline/main_build.py --resolution 15T
```

### Python API

```python
from alphagen_data_pipeline.data_loaders import aggregate_data_with_resolution
from alphagen_data_pipeline.config import LEAN_DATA_PATH

# Aggregate to daily resolution
daily_data = aggregate_data_with_resolution(
    data_path=LEAN_DATA_PATH,
    shares_file="shares.csv",
    start_date_str="2022-01-01",
    end_date_str="2025-11-10",
    resolution="1D"
)

# Aggregate to 30-minute resolution
min30_data = aggregate_data_with_resolution(
    data_path=LEAN_DATA_PATH,
    shares_file="shares.csv",
    start_date_str="2022-01-01",
    end_date_str="2025-11-10",
    resolution="30T"
)

# Aggregate to 2-hour resolution
hour2_data = aggregate_data_with_resolution(
    data_path=LEAN_DATA_PATH,
    shares_file="shares.csv",
    start_date_str="2022-01-01",
    end_date_str="2025-11-10",
    resolution="2H"
)

# Aggregate to AM/PM sessions
ampm_data = aggregate_data_with_resolution(
    data_path=LEAN_DATA_PATH,
    shares_file="shares.csv",
    start_date_str="2022-01-01",
    end_date_str="2025-11-10",
    resolution="AM/PM"
)
```

## Configuration

You can configure the default resolution in `config.py`:

```python
# Set default resolution
DATA_RESOLUTION = "30T"  # or "1D", "2H", "AM/PM", etc.
```

Each resolution has its own configuration:

```python
RESOLUTION_CONFIG = {
    "AM/PM": {
        "base_file": "am_pm_base_data.csv",
        "feature_store_dir": "e:/factor/feature_store/am_pm_features",
        "partition_by": ('date', 'session'),
        "annualization": 504  # 252 trading days * 2 sessions
    },
    "1D": {
        "base_file": "daily_base_data.csv",
        "feature_store_dir": "e:/factor/feature_store/daily_features",
        "partition_by": ('date',),
        "annualization": 252
    },
    # ... more resolutions
}
```

## Output Format

All resolutions produce a DataFrame with the following columns:

- `symbol`: Stock ticker symbol
- `timestamp`: Timestamp for the bar (timezone-aware, America/New_York)
- `date`: Trading date
- `session`: Session identifier
  - For AM/PM: `'AM'` or `'PM'`
  - For time-based: time of the bar (e.g., `'09:30'`, `'10:00'`)
  - For daily: `'FULL'`
- `open`: Opening price for the period
- `high`: Highest price during the period
- `low`: Lowest price during the period
- `close`: Closing price for the period
- `volume`: Total volume during the period
- `vwap`: Volume-weighted average price
- `mktcap`: Market capitalization (close * shares_outstanding)
- `turnover`: Volume / shares_outstanding

## Feature Computation

**Note**: Style feature computation (TE, CORR, IDIOVOL, etc.) is currently only available for AM/PM resolution. For other resolutions, the pipeline will save the base aggregated data without computing style features.

To extend feature computation to other resolutions, you would need to adapt the logic in `feature_style.py`.

## Examples

### Example 1: Generate Daily Data

```bash
# Generate daily aggregated data
python alphagen_data_pipeline/main_build.py --resolution 1D
```

Output:
- Base data: `daily_base_data.csv`
- Feature store: `e:/factor/feature_store/daily_features/`

### Example 2: Generate 30-Minute Data

```bash
# Generate 30-minute bars
python alphagen_data_pipeline/main_build.py --resolution 30T
```

Output:
- Base data: `30min_base_data.csv`
- Feature store: `e:/factor/feature_store/30min_features/`

### Example 3: Generate 2-Hour Data

```bash
# Generate 2-hour bars
python alphagen_data_pipeline/main_build.py --resolution 2H
```

Output:
- Base data: `2h_base_data.csv`
- Feature store: `e:/factor/feature_store/2h_features/`

## Loading Data

Use the `load_features` function to read the partitioned Parquet data:

```python
from alphagen_data_pipeline.storage import load_features

# Load daily data for a date range
daily_features = load_features(
    out_dir="e:/factor/feature_store/daily_features",
    start="2023-01-01",
    end="2023-12-31",
    symbols=["AAPL", "MSFT", "GOOGL"]  # optional
)

# Load 30-minute data
min30_features = load_features(
    out_dir="e:/factor/feature_store/30min_features",
    start="2023-01-01",
    end="2023-12-31"
)
```

## Technical Details

### Time Aggregation

For time-based resolutions (30T, 1H, 2H, etc.):
1. Minute data is loaded with timezone-aware timestamps (America/New_York)
2. Data is aggregated using pandas floor operation for the specified frequency
3. OHLC aggregation: first open, max high, min low, last close
4. Volume: sum of all minute volumes in the period
5. VWAP: volume-weighted average of close prices

### Annualization Factors

Different resolutions have different annualization factors for volatility calculations:
- Daily (1D): 252 (trading days per year)
- AM/PM: 504 (252 * 2 sessions per day)
- 30T: 3276 (252 * ~13 bars per day)
- 1H: 1638 (252 * 6.5 bars per day)
- 2H: 819 (252 * 3.25 bars per day)

These factors are used to annualize volatility and other metrics.

## Troubleshooting

### Missing Data

If you see "No data loaded for any symbols":
- Check that `LEAN_DATA_PATH` in `config.py` points to the correct directory
- Verify that symbol directories exist in the data path
- Check that zip files exist for the date range

### Memory Issues

For high-frequency resolutions (e.g., 1-minute bars):
- Process data in smaller date ranges
- Use incremental processing
- Consider using Dask for larger datasets

### Timestamp Issues

All timestamps are localized to America/New_York timezone to handle:
- Market hours (9:30 AM - 4:00 PM ET)
- Daylight saving time transitions
- Holiday schedules
