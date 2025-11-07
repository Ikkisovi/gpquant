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

# ============ DATA RESOLUTION OPTIONS ============
# Supported resolutions:
#   - "1D" or "daily": Daily aggregation (one bar per day)
#   - "30T" or "30min": 30-minute bars
#   - "1H" or "1h": 1-hour bars
#   - "2H" or "2h": 2-hour bars
#   - "AM/PM" or "ampm": Morning/Afternoon sessions (split at 12:00 ET)
#   - Any pandas frequency string (e.g., "15T", "4H", etc.)
DATA_RESOLUTION = "AM/PM"  # Default to AM/PM for backward compatibility

# Resolution-specific settings
RESOLUTION_CONFIG = {
    "AM/PM": {
        "base_file": "am_pm_base_data.csv",
        "feature_store_dir": r"e:/factor/feature_store/am_pm_features",
        "partition_by": ('date', 'session'),
        "annualization": 504  # 252 trading days * 2 sessions
    },
    "1D": {
        "base_file": "daily_base_data.csv",
        "feature_store_dir": r"e:/factor/feature_store/daily_features",
        "partition_by": ('date',),
        "annualization": 252  # 252 trading days
    },
    "30T": {
        "base_file": "30min_base_data.csv",
        "feature_store_dir": r"e:/factor/feature_store/30min_features",
        "partition_by": ('date',),
        "annualization": 252 * 13  # ~13 30-min bars per day
    },
    "1H": {
        "base_file": "1h_base_data.csv",
        "feature_store_dir": r"e:/factor/feature_store/1h_features",
        "partition_by": ('date',),
        "annualization": 252 * 6.5  # 6.5 hours per trading day
    },
    "2H": {
        "base_file": "2h_base_data.csv",
        "feature_store_dir": r"e:/factor/feature_store/2h_features",
        "partition_by": ('date',),
        "annualization": 252 * 3.25  # ~3.25 2-hour bars per day
    }
}

def get_resolution_config(resolution=None):
    """Get configuration for a specific resolution"""
    if resolution is None:
        resolution = DATA_RESOLUTION

    # Normalize resolution string
    resolution_normalized = resolution.upper() if resolution.upper() in ["AM/PM", "AMPM"] else resolution

    if resolution_normalized in RESOLUTION_CONFIG:
        return RESOLUTION_CONFIG[resolution_normalized]
    else:
        # Default configuration for custom resolutions
        return {
            "base_file": f"{resolution}_base_data.csv",
            "feature_store_dir": f"e:/factor/feature_store/{resolution}_features",
            "partition_by": ('date',),
            "annualization": 252  # Default to daily annualization
        }
