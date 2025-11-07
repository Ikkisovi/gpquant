# alphagen_data_pipeline/__init__.py
"""
Flexible Resolution Data Pipeline with Columnar Storage

This package provides a modular pipeline for processing minute-level stock data
with configurable time resolutions (daily, hourly, AM/PM sessions, etc.) and
stores features in efficient columnar format (Parquet).

Main components:
- config: Configuration settings (dates, paths, windows, benchmarks, resolutions)
- data_loaders: Data loading and flexible resolution aggregation
- feature_style: Style-based feature engineering (AM/PM optimized)
- storage: Columnar storage utilities (Parquet with partitioning)
- main_build: Main pipeline orchestration

Supported resolutions:
- "AM/PM" or "ampm": Morning/Afternoon sessions (split at 12:00 ET)
- "1D" or "daily": Daily bars
- "30T" or "30min": 30-minute bars
- "1H" or "1h": 1-hour bars
- "2H" or "2h": 2-hour bars
- Any pandas frequency string (e.g., "15T", "4H")

Usage:
    # Command line
    python alphagen_data_pipeline/main_build.py --resolution 30T

    # Python API
    from alphagen_data_pipeline.data_loaders import aggregate_data_with_resolution
    data = aggregate_data_with_resolution(
        data_path="path/to/data",
        shares_file="shares.csv",
        start_date_str="2022-01-01",
        end_date_str="2025-11-10",
        resolution="30T"
    )

See USAGE.md for detailed documentation and examples.
"""

from . import config
from . import data_loaders
from . import feature_style
from . import storage

# Export key functions for convenience
from .data_loaders import (
    aggregate_data_with_resolution,
    aggregate_to_am_pm,
    load_all_symbols_data
)
from .config import get_resolution_config
from .storage import load_features, save_feature_store

__version__ = '2.0.0'
__all__ = [
    'config',
    'data_loaders',
    'feature_style',
    'storage',
    'aggregate_data_with_resolution',
    'aggregate_to_am_pm',
    'load_all_symbols_data',
    'get_resolution_config',
    'load_features',
    'save_feature_store'
]
