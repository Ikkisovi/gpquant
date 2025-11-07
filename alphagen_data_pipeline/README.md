# AlphaGen Data Pipeline (列式数据储存)

A modular pipeline for processing minute-level stock data into AM/PM sessions with style features, stored in efficient columnar Parquet format.

## 📁 Pipeline Structure

```
alphagen_data_pipeline/
├── __init__.py          # Package initialization
├── config.py            # Configuration settings
├── data_loaders.py      # Data loading and AM/PM aggregation
├── feature_style.py     # Style-based feature engineering
├── storage.py           # Columnar storage (Parquet + partitioning)
├── main_build.py        # Main pipeline orchestration
└── README.md            # This file
```

## 🚀 Quick Start

### 1. Run the Complete Pipeline

```bash
cd e:/factor/alphagen
python alphagen_data_pipeline/main_build.py
```

### 2. Use as a Python Module

```python
from alphagen_data_pipeline import config, storage
from alphagen_data_pipeline.data_loaders import aggregate_to_am_pm
from alphagen_data_pipeline.feature_style import add_style_features

# Load data with the pipeline
```

## 📊 Pipeline Flow

### Step 1: Data Loading & Aggregation (`data_loaders.py`)

**Functions:**
- `load_real_lean_data()`: Load minute-level data from zip files
- `load_all_symbols_data()`: Load data for multiple symbols
- `aggregate_to_am_pm()`: Aggregate minute data into AM/PM sessions

**Output:** Session-level data with columns:
```
['symbol', 'date', 'session', 'open', 'high', 'low', 'close',
 'volume', 'vwap', 'mktcap', 'turnover', 'timestamp']
```

### Step 2: Feature Engineering (`feature_style.py`)

**Features Calculated:**

1. **Tracking Error (TE)**: Volatility of returns relative to benchmarks
   - `TE_{benchmark}_{window}_AM/PM`

2. **Correlation (CORR)**: Rolling correlation with benchmarks
   - `CORR_{benchmark}_{window}_AM/PM`

3. **TE Share**: Tracking error as fraction of total volatility
   - `TEshare_{benchmark}_{window}_AM/PM` (cross-sectional winsorized)

4. **Differential TE (DTE)**: AM tracking error minus PM tracking error
   - `DTE_{benchmark}_{window}`

5. **Relative TE (RTE)**: Ratio of AM to PM tracking error
   - `RTE_{benchmark}_{window}`

6. **Cross-Session Correlation**: AM returns vs PM benchmark (and vice versa)
   - `CROSS_CORR_{benchmark}_{window}_AM_PM`
   - `CROSS_CORR_{benchmark}_{window}_PM_AM`

7. **Delta TE (dTE)**: 10-period change in tracking error
   - `dTE_{benchmark}_{window}_AM/PM`

8. **ARGMIN TE**: Best-fitting benchmark (lowest TE)
   - `ARGMIN_TE_{window}_AM/PM`

9. **Idiosyncratic Volatility (IDIOVOL)**: Residual volatility after regressing on benchmarks
   - `IDIOVOL_{window}_AM/PM`

**Benchmarks:**
- VBR (Vanguard Small-Cap Value)
- QQQ (Nasdaq-100)
- SPY (S&P 500)
- SPMO (S&P 500 Momentum)
- SPHQ (S&P 500 Quality)
- SPYG (S&P 500 Growth)

**Windows:** 42, 126, 252 half-days (~1, 3, 6 months)

### Step 3: Columnar Storage (`storage.py`)

**Functions:**
- `wide_to_long()`: Convert wide format (_am/_pm columns) to long format (session dimension)
- `save_feature_store()`: Save to partitioned Parquet with compression
- `load_features()`: Load features with date/symbol/session filtering

**Storage Format:**
- **Format:** Parquet (PyArrow)
- **Partitioning:** By `(date, session)` - enables efficient date-range queries
- **Compression:** ZSTD (optimal for time-series data)
- **Precision:** float32 for features, category for categorical columns

**Advantages:**
✅ **10-100x smaller** than CSV
✅ **Faster read/write** with columnar format
✅ **Efficient filtering** with partitioning
✅ **Schema preservation** (data types preserved)
✅ **Compatible** with Pandas, Polars, DuckDB, Spark

## ⚙️ Configuration (`config.py`)

```python
# Data paths
LEAN_DATA_PATH = "e:/factor/lean_project/data/equity/usa/minute"
BASE_DATA_FILE = "am_pm_base_data.csv"
FEATURE_STORE_DIR = "e:/factor/feature_store/am_pm_features"

# Date range
START_DATE = "2022-01-01"
END_DATE = "2025-11-10"

# Feature parameters
ANNUALIZATION = 504  # Half-day periods per year
WINDOWS = [42, 126, 252]
BENCHMARKS = ['VBR', 'QQQ', 'SPY', 'SPMO', 'SPHQ', 'SPYG']

# Storage
PARTITION_BY = ('date', 'session')  # Partition strategy
```

## 📖 Usage Examples

### Example 1: Load Features for Specific Date Range

```python
from alphagen_data_pipeline.storage import load_features

# Load all features for Q1 2024
df = load_features(
    out_dir="e:/factor/feature_store/am_pm_features",
    start="2024-01-01",
    end="2024-03-31"
)

# Load specific feature patterns
df = load_features(
    out_dir="e:/factor/feature_store/am_pm_features",
    start="2024-01-01",
    end="2024-03-31",
    feature_patterns=['TE_SPY_*', 'CORR_*', 'IDIOVOL_*'],
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    sessions=['AM']
)
```

### Example 2: Incremental Updates

```python
from alphagen_data_pipeline.data_loaders import aggregate_to_am_pm
from alphagen_data_pipeline.feature_style import add_style_features
from alphagen_data_pipeline.storage import wide_to_long, save_feature_store

# Process only new data
new_base = aggregate_to_am_pm(
    data_path=LEAN_DATA_PATH,
    shares_file="shares.csv",
    start_date_str="2025-11-01",
    end_date_str="2025-11-10"
)

new_features = add_style_features(new_base, ...)
new_long = wide_to_long(new_features)

# Append to feature store
save_feature_store(new_long, FEATURE_STORE_DIR)
```

### Example 3: Query with DuckDB (Ultra-Fast)

```python
import duckdb

# SQL query on Parquet files
result = duckdb.query("""
    SELECT symbol, date, session,
           TE_SPY_126_AM, IDIOVOL_126_AM
    FROM 'e:/factor/feature_store/am_pm_features/**/*.parquet'
    WHERE date BETWEEN '2024-01-01' AND '2024-12-31'
      AND session = 'AM'
      AND symbol IN ('AAPL', 'MSFT')
    ORDER BY date, symbol
""").to_df()
```

## 🔄 Migration from `create_am_pm_dataset.py`

The original monolithic script has been refactored into a modular pipeline:

| Original                      | New Pipeline                  | Benefits                          |
|-------------------------------|-------------------------------|-----------------------------------|
| Single 518-line script        | 5 focused modules             | Better maintainability            |
| CSV output (10+ GB)           | Partitioned Parquet (1-2 GB)  | 10x smaller, faster I/O           |
| Full reload each time         | Incremental updates possible  | Save time on reruns               |
| Hard-coded paths              | Centralized config            | Easy environment switching        |
| All-in-one function           | Modular components            | Reusable for other pipelines      |

## 🛠️ Requirements

```bash
pip install pandas numpy scipy scikit-learn pyarrow
```

Optional (for advanced queries):
```bash
pip install duckdb polars
```

## 📝 Notes

1. **Memory Optimization**: The pipeline uses float32 precision and categorical dtypes to reduce memory usage
2. **Partitioning Strategy**: Partitioned by `(date, session)` enables fast date-range queries
3. **Caching**: Base data is cached in `am_pm_base_data.csv` to avoid re-aggregating from minute data
4. **Error Handling**: Missing data for specific dates/symbols is gracefully handled
5. **Timezone**: All times are in America/New_York (ET)

## 🐛 Troubleshooting

**Issue:** "No symbol directories found"
- Check that `LEAN_DATA_PATH` points to the correct directory
- Ensure symbol directories exist (e.g., `aapl/`, `msft/`)

**Issue:** "Shares outstanding file not found"
- Create `shares.csv` with columns: `symbol`, `shares_outstanding`

**Issue:** Memory error
- Process in smaller date ranges
- Reduce the number of symbols
- Use `chunks` parameter if implementing batch processing

## 📈 Performance Tips

1. **Partitioning**: Current `(date, session)` partitioning is optimal for time-series queries
2. **Compression**: ZSTD provides best compression for time-series data
3. **Batch Size**: Process 1-2 years at a time for optimal memory usage
4. **Parallel Loading**: Modify `load_all_symbols_data()` to use multiprocessing for faster loads

## 📧 Contact

For issues or questions, please refer to the main AlphaGen project documentation.

---
*Pipeline Version: 1.0.0*
*Last Updated: 2025-11-06*
