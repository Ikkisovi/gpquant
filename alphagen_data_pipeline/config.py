# config.py
ANNUALIZATION = 504
WINDOWS = [42, 126, 252]
BENCHMARKS = ['VBR', 'QQQ', 'SPY', 'SPMO', 'SPHQ', 'SPYG']

LEAN_DATA_PATH = r"e:/factor/lean_project/data/equity/usa/minute"
BASE_DATA_FILE = "am_pm_base_data.csv"

# Feature store 输出（分区 Parquet）
FEATURE_STORE_DIR = r"e:/factor/feature_store/am_pm_features"  # 目录，不是文件
PARTITION_BY = ('date', 'session')  # 或 ('symbol',) 看你的读取习惯

START_DATE = "2022-01-01"
END_DATE   = "2025-11-10"
