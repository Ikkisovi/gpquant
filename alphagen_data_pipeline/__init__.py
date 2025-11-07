# alphagen_data_pipeline/__init__.py
"""
AM/PM Data Pipeline with Columnar Storage

This package provides a modular pipeline for processing minute-level stock data
into AM/PM sessions with style features, stored in efficient columnar format (Parquet).

Main components:
- config: Configuration settings (dates, paths, windows, benchmarks)
- data_loaders: Data loading and AM/PM aggregation
- feature_style: Style-based feature engineering
- storage: Columnar storage utilities (Parquet with partitioning)
- main_build: Main pipeline orchestration

Usage:
    from alphagen_data_pipeline import main_build
    # Run: python -m alphagen_data_pipeline.main_build
"""

from . import config
from . import data_loaders
from . import feature_style
from . import storage

__version__ = '1.0.0'
__all__ = ['config', 'data_loaders', 'feature_style', 'storage']
