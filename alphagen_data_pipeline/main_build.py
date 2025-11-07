# main_build.py
import os
import sys
import pandas as pd
import argparse

# Support both direct run and module import
if __name__ == '__main__':
    # Add parent directory for direct run
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from alphagen_data_pipeline.config import *
    from alphagen_data_pipeline.data_loaders import aggregate_data_with_resolution
    from alphagen_data_pipeline.feature_style import add_style_features
    from alphagen_data_pipeline.storage import wide_to_long, save_feature_store
else:
    # Module import
    from .config import *
    from .data_loaders import aggregate_data_with_resolution
    from .feature_style import add_style_features
    from .storage import wide_to_long, save_feature_store

def main(resolution=None):
    """
    Main pipeline execution

    Args:
        resolution: Data resolution to use. If None, uses DATA_RESOLUTION from config.
                    Options: "AM/PM", "1D", "30T", "1H", "2H", or any pandas frequency string
    """
    # Use provided resolution or fall back to config
    if resolution is None:
        resolution = DATA_RESOLUTION

    # Get resolution-specific configuration
    res_config = get_resolution_config(resolution)
    base_file = res_config['base_file']
    feature_store_dir = res_config['feature_store_dir']
    partition_by = res_config['partition_by']

    print("=" * 80)
    print("ALPHAGEN DATA PIPELINE - FULL BUILD")
    print("=" * 80)
    print(f"Resolution: {resolution}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Output: {feature_store_dir}")
    print(f"Partition by: {partition_by}")
    print("=" * 80)

    # 1) Load or aggregate base data
    if os.path.exists(base_file):
        print(f"\n--- Loading base data from {base_file} ---")
        base_df = pd.read_csv(base_file)
        base_df['date'] = pd.to_datetime(base_df['date']).dt.date
        print(f"Loaded {len(base_df)} records")
    else:
        print(f"\n--- {base_file} not found, aggregating from minute data ---")
        base_df = aggregate_data_with_resolution(
            data_path=LEAN_DATA_PATH,
            shares_file="shares.csv",
            start_date_str=START_DATE,
            end_date_str=END_DATE,
            resolution=resolution
        )
        if not base_df.empty:
            print(f"\n--- Saving base data to {base_file} ---")
            base_df.to_csv(base_file, index=False)
            print(f"Saved {len(base_df)} records")

    if base_df.empty:
        print("\n[ERROR] No base data available. Stopping.")
        return 1

    # 2) Compute style features (returns wide table)
    # Note: Feature computation is currently designed for AM/PM sessions
    # For other resolutions, you may need to adapt the feature computation logic
    if resolution.upper() in ['AM/PM', 'AMPM']:
        print(f"\n--- Computing style features ---")
        final_wide = add_style_features(
            base_df,
            data_path=LEAN_DATA_PATH,
            start_date_str=START_DATE,
            end_date_str=END_DATE
        )
        print(f"Created {len(final_wide.columns)} feature columns")

        # 3) Wide -> Long + Partition to Parquet
        print(f"\n--- Converting to long format and saving ---")
        final_long = wide_to_long(final_wide)
        print(f"Converted to long format: {len(final_long)} rows, {len(final_long.columns)} columns")

        save_feature_store(final_long, feature_store_dir, partition_by=partition_by, compression='zstd')
    else:
        # For non-AM/PM resolutions, skip feature computation and save base data directly
        print(f"\n--- Saving base data to feature store (no feature computation for {resolution}) ---")

        # Convert to appropriate format
        final_long = base_df.copy()

        # Ensure proper data types
        for col in ['open', 'high', 'low', 'close', 'volume', 'vwap', 'mktcap', 'turnover']:
            if col in final_long.columns:
                final_long[col] = final_long[col].astype('float32')

        if 'session' in final_long.columns:
            final_long['session'] = final_long['session'].astype('category')

        save_feature_store(final_long, feature_store_dir, partition_by=partition_by, compression='zstd')

    print("\n" + "=" * 80)
    print("[SUCCESS] PIPELINE COMPLETED!")
    print("=" * 80)
    print(f"\nFeature store saved to: {feature_store_dir}")
    print(f"Total rows: {len(final_long) if 'final_long' in locals() else len(base_df)}")
    print(f"Date range: {base_df['date'].min()} to {base_df['date'].max()}")
    print(f"Symbols: {base_df['symbol'].nunique()}")
    print(f"\nSample data:")
    print(base_df.head(3))

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build alphagen data pipeline with configurable resolution')
    parser.add_argument('--resolution', type=str, default=None,
                        help='Data resolution: AM/PM (default), 1D, 30T, 1H, 2H, etc.')
    args = parser.parse_args()

    exit_code = main(resolution=args.resolution)
    sys.exit(exit_code)
