# main_build.py
import os
import sys
import pandas as pd

# Support both direct run and module import
if __name__ == '__main__':
    # Add parent directory for direct run
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from alphagen_data_pipeline.config import *
    from alphagen_data_pipeline.data_loaders import aggregate_to_am_pm
    from alphagen_data_pipeline.feature_style import add_style_features
    from alphagen_data_pipeline.storage import wide_to_long, save_feature_store
else:
    # Module import
    from .config import *
    from .data_loaders import aggregate_to_am_pm
    from .feature_style import add_style_features
    from .storage import wide_to_long, save_feature_store

def main():
    """Main pipeline execution"""
    print("=" * 80)
    print("ALPHAGEN DATA PIPELINE - FULL BUILD")
    print("=" * 80)
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Output: {FEATURE_STORE_DIR}")
    print("=" * 80)

    # 1) 裸数据（AM/PM）
    if os.path.exists(BASE_DATA_FILE):
        print(f"\n--- Loading base data from {BASE_DATA_FILE} ---")
        base_df = pd.read_csv(BASE_DATA_FILE)
        base_df['date'] = pd.to_datetime(base_df['date']).dt.date
        print(f"Loaded {len(base_df)} session records")
    else:
        print(f"\n--- {BASE_DATA_FILE} not found, aggregating from minute data ---")
        base_df = aggregate_to_am_pm(
            data_path=LEAN_DATA_PATH,
            shares_file="shares.csv",
            start_date_str=START_DATE,
            end_date_str=END_DATE,
        )
        if not base_df.empty:
            print(f"\n--- Saving base data to {BASE_DATA_FILE} ---")
            base_df.to_csv(BASE_DATA_FILE, index=False)
            print(f"Saved {len(base_df)} session records")

    if base_df.empty:
        print("\n[ERROR] No base data available. Stopping.")
        return 1

    # 2) 计算风格特征（返回宽表）
    print(f"\n--- Computing style features ---")
    final_wide = add_style_features(
        base_df,
        data_path=LEAN_DATA_PATH,
        start_date_str=START_DATE,
        end_date_str=END_DATE
    )
    print(f"Created {len(final_wide.columns)} feature columns")

    # 3) 宽->长 + 分区落盘为 Parquet（别再写巨大的 CSV）
    print(f"\n--- Converting to long format and saving ---")
    final_long = wide_to_long(final_wide)
    print(f"Converted to long format: {len(final_long)} rows, {len(final_long.columns)} columns")

    save_feature_store(final_long, FEATURE_STORE_DIR, partition_by=PARTITION_BY, compression='zstd')

    print("\n" + "=" * 80)
    print("[SUCCESS] PIPELINE COMPLETED!")
    print("=" * 80)
    print(f"\nFeature store saved to: {FEATURE_STORE_DIR}")
    print(f"Total rows: {len(final_long)}")
    print(f"Date range: {final_long['date'].min()} to {final_long['date'].max()}")
    print(f"Symbols: {final_long['symbol'].nunique()}")
    print(f"\nSample data:")
    print(final_long.head(3))

    return 0

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
