"""
Example usage of the alphagen data pipeline with different resolutions.

This script demonstrates how to use the data pipeline to generate
aggregated data at different time resolutions.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphagen_data_pipeline.data_loaders import aggregate_data_with_resolution
from alphagen_data_pipeline.config import LEAN_DATA_PATH

def example_daily_aggregation():
    """Example: Aggregate minute data to daily bars"""
    print("=" * 80)
    print("Example 1: Daily Aggregation")
    print("=" * 80)

    daily_data = aggregate_data_with_resolution(
        data_path=LEAN_DATA_PATH,
        shares_file="shares.csv",
        start_date_str="2024-01-01",
        end_date_str="2024-01-31",
        resolution="1D"
    )

    print("\nDaily Data Summary:")
    print(f"Total rows: {len(daily_data)}")
    print(f"Symbols: {daily_data['symbol'].nunique()}")
    print(f"Date range: {daily_data['date'].min()} to {daily_data['date'].max()}")
    print("\nSample data:")
    print(daily_data.head())

    return daily_data


def example_30min_aggregation():
    """Example: Aggregate minute data to 30-minute bars"""
    print("\n" + "=" * 80)
    print("Example 2: 30-Minute Aggregation")
    print("=" * 80)

    min30_data = aggregate_data_with_resolution(
        data_path=LEAN_DATA_PATH,
        shares_file="shares.csv",
        start_date_str="2024-01-01",
        end_date_str="2024-01-31",
        resolution="30T"
    )

    print("\n30-Minute Data Summary:")
    print(f"Total rows: {len(min30_data)}")
    print(f"Symbols: {min30_data['symbol'].nunique()}")
    print(f"Date range: {min30_data['date'].min()} to {min30_data['date'].max()}")
    print(f"Sessions per day: {min30_data.groupby('date')['session'].nunique().mean():.1f}")
    print("\nSample data:")
    print(min30_data.head(15))

    return min30_data


def example_2hour_aggregation():
    """Example: Aggregate minute data to 2-hour bars"""
    print("\n" + "=" * 80)
    print("Example 3: 2-Hour Aggregation")
    print("=" * 80)

    hour2_data = aggregate_data_with_resolution(
        data_path=LEAN_DATA_PATH,
        shares_file="shares.csv",
        start_date_str="2024-01-01",
        end_date_str="2024-01-31",
        resolution="2H"
    )

    print("\n2-Hour Data Summary:")
    print(f"Total rows: {len(hour2_data)}")
    print(f"Symbols: {hour2_data['symbol'].nunique()}")
    print(f"Date range: {hour2_data['date'].min()} to {hour2_data['date'].max()}")
    print(f"Sessions per day: {hour2_data.groupby('date')['session'].nunique().mean():.1f}")
    print("\nSample data:")
    print(hour2_data.head(10))

    return hour2_data


def example_ampm_aggregation():
    """Example: Aggregate minute data to AM/PM sessions"""
    print("\n" + "=" * 80)
    print("Example 4: AM/PM Session Aggregation")
    print("=" * 80)

    ampm_data = aggregate_data_with_resolution(
        data_path=LEAN_DATA_PATH,
        shares_file="shares.csv",
        start_date_str="2024-01-01",
        end_date_str="2024-01-31",
        resolution="AM/PM"
    )

    print("\nAM/PM Data Summary:")
    print(f"Total rows: {len(ampm_data)}")
    print(f"Symbols: {ampm_data['symbol'].nunique()}")
    print(f"Date range: {ampm_data['date'].min()} to {ampm_data['date'].max()}")
    print(f"Sessions: {ampm_data['session'].unique()}")
    print("\nSample data:")
    print(ampm_data.head(10))

    return ampm_data


def compare_resolutions():
    """Compare OHLC values across different resolutions for a single symbol"""
    print("\n" + "=" * 80)
    print("Example 5: Comparing Resolutions for Single Symbol")
    print("=" * 80)

    # Get data for a single day and symbol at different resolutions
    symbol = "AAPL"
    date = "2024-01-15"

    resolutions = ["1D", "AM/PM", "2H", "30T"]
    results = {}

    for resolution in resolutions:
        print(f"\nAggregating {symbol} data at {resolution} resolution...")
        data = aggregate_data_with_resolution(
            data_path=LEAN_DATA_PATH,
            shares_file="shares.csv",
            start_date_str=date,
            end_date_str=date,
            resolution=resolution
        )

        if not data.empty:
            symbol_data = data[data['symbol'] == symbol]
            if not symbol_data.empty:
                results[resolution] = symbol_data
                print(f"{resolution}: {len(symbol_data)} bars")
            else:
                print(f"{resolution}: No data for {symbol}")
        else:
            print(f"{resolution}: No data loaded")

    # Display comparison
    print(f"\n{'Resolution':<10} {'Bars':<6} {'Day Open':<12} {'Day High':<12} {'Day Low':<12} {'Day Close':<12} {'Volume':<15}")
    print("-" * 95)

    for resolution, data in results.items():
        bars = len(data)
        day_open = data['open'].iloc[0]
        day_high = data['high'].max()
        day_low = data['low'].min()
        day_close = data['close'].iloc[-1]
        volume = data['volume'].sum()

        print(f"{resolution:<10} {bars:<6} {day_open:<12.2f} {day_high:<12.2f} {day_low:<12.2f} {day_close:<12.2f} {volume:<15,.0f}")


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("ALPHAGEN DATA PIPELINE - RESOLUTION EXAMPLES")
    print("=" * 80)
    print("\nThis script demonstrates different data resolution options.")
    print("Make sure you have:")
    print("1. Minute-level data in the LEAN_DATA_PATH")
    print("2. shares.csv file with shares outstanding data")
    print("=" * 80)

    try:
        # Run examples
        # Uncomment the examples you want to run:

        # example_daily_aggregation()
        # example_30min_aggregation()
        # example_2hour_aggregation()
        # example_ampm_aggregation()
        # compare_resolutions()

        print("\n" + "=" * 80)
        print("Examples completed successfully!")
        print("=" * 80)
        print("\nTo run specific examples, uncomment them in the main() function.")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
