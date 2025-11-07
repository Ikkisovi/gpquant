# storage.py
import os
import pandas as pd
from .config import PARTITION_BY

def wide_to_long(final_df: pd.DataFrame) -> pd.DataFrame:
    # 宽(_am/_pm) -> 长(session维度)
    am_map = {c: c.replace('_am','') for c in final_df.columns if c.endswith('_am')}
    pm_map = {c: c.replace('_pm','') for c in final_df.columns if c.endswith('_pm')}
    keep_am = ['symbol','date'] + [c for c in final_df.columns if c.endswith('_am')]
    keep_pm = ['symbol','date'] + [c for c in final_df.columns if c.endswith('_pm')]

    am_long = (final_df[keep_am].rename(columns=am_map).assign(session='AM'))
    pm_long = (final_df[keep_pm].rename(columns=pm_map).assign(session='PM'))
    long_df = pd.concat([am_long, pm_long], ignore_index=True)

    # 降精度
    for c in long_df.columns:
        if long_df[c].dtype.kind == 'f':
            long_df[c] = long_df[c].astype('float32')
    long_df['session'] = long_df['session'].astype('category')
    if 'ARGMIN_TE_126' in long_df.columns:
        long_df['ARGMIN_TE_126'] = long_df['ARGMIN_TE_126'].astype('category')
    return long_df

def save_feature_store(long_df: pd.DataFrame, out_dir: str,
                       partition_by=PARTITION_BY, compression='zstd',
                       max_partitions=3000):
    """
    Save features to partitioned Parquet with columnar storage.

    Args:
        long_df: DataFrame in long format (session dimension)
        out_dir: Output directory
        partition_by: Columns to partition by (e.g., ('date', 'session'))
        compression: Compression codec ('zstd', 'snappy', 'gzip', None)
        max_partitions: Maximum number of partitions (default 3000 for ~6 years of daily data)
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(out_dir, exist_ok=True)

    # Convert to Arrow table
    table = pa.Table.from_pandas(long_df)

    # Write with increased max_partitions
    pq.write_to_dataset(
        table,
        root_path=out_dir,
        partition_cols=list(partition_by),
        compression=compression,
        max_partitions=max_partitions,
        existing_data_behavior='overwrite_or_ignore'
    )
    print(f"Saved to {out_dir}, partitioned by {partition_by}, compression={compression}")

# 读取器（alphagen 用）
import fnmatch
import pyarrow.dataset as ds

def load_features(out_dir, start, end, feature_patterns=None, symbols=None, sessions=None):
    dataset = ds.dataset(out_dir, format='parquet', partitioning='hive')
    # 过滤：日期区间 (partition columns are strings in hive partitioning)
    start_date = pd.to_datetime(start).date()
    end_date = pd.to_datetime(end).date()
    start_str = str(start_date)
    end_str = str(end_date)
    filt = (ds.field('date') >= start_str) & (ds.field('date') <= end_str)
    if sessions:
        filt = filt & ds.field('session').isin(list(sessions))
    table = dataset.to_table(filter=filt)
    df = table.to_pandas()

    # Convert date back to date type for consistency
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.date

    if symbols is not None:
        df = df[df['symbol'].isin(symbols)]

    if feature_patterns:
        keep = {'symbol','date','session'}
        for pat in feature_patterns:
            keep |= {c for c in df.columns if fnmatch.fnmatch(c, pat)}
        df = df[list(keep)]
    return df.sort_values(['symbol','date','session'])
