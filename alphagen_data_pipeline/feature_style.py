# feature_style.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from .config import ANNUALIZATION, WINDOWS, BENCHMARKS
from .data_loaders import load_all_symbols_data
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def cs_winsorize(g: pd.Series, p: float = 0.01) -> pd.Series:
    x = g.copy()
    lo, hi = x.quantile(p), x.quantile(1 - p)
    return x.clip(lo, hi)

def rolling_ols_residuals_array(y: np.ndarray, X: np.ndarray, window: int) -> np.ndarray:
    n = len(y); resid = np.full(n, np.nan)
    for i in range(window, n + 1):
        yw = y[i-window:i]; Xw = X[i-window:i]
        valid = (~np.isnan(yw)) & (~np.isnan(Xw).any(axis=1))
        if valid.sum() < max(5, Xw.shape[1] + 1):  # 自由度保护
            continue
        try:
            model = LinearRegression()
            model.fit(Xw[valid], yw[valid])
            yhat = model.predict(X[i-1:i])[0]
            resid[i-1] = y[i-1] - yhat
        except Exception:
            continue
    return resid

def add_style_features(session_df: pd.DataFrame,
                       data_path: str,
                       start_date_str: str,
                       end_date_str: str) -> pd.DataFrame:
    """
    输入：半日长表（AM/PM 各一行） -> 计算风格距离特征 -> 宽表（AM/PM 合到同一行，_am/_pm）
    落盘与分区不在此函数做。
    """
    print("\n--- Step 7: Adding Style Features ---")

    start_date = pd.to_datetime(start_date_str)
    end_date   = pd.to_datetime(end_date_str)

    # === 7.1 基准数据 ===
    print("--- Loading & aggregating benchmark AM/PM returns ---")
    bench_df = load_all_symbols_data(BENCHMARKS, data_path, start_date, end_date)
    if bench_df.empty:
        print("Warning: No benchmark data loaded. Skipping feature creation.")
        return session_df

    bench_df['datetime'] = bench_df['datetime'].dt.tz_localize("America/New_York", ambiguous='infer')
    bench_df['date'] = bench_df['datetime'].dt.date
    bench_df['session'] = np.where(bench_df['datetime'].dt.hour < 12, 'AM', 'PM')
    g = bench_df.groupby(['symbol','date','session'])
    bench_agg = (g.agg(open=('open','first'), close=('close','last'))
                   .reset_index()
                   .rename(columns={'symbol':'benchmark'}))
    bench_agg['benchmark_return'] = bench_agg['close']/bench_agg['open'] - 1.0

    bench_pivot = (bench_agg.pivot_table(index=['date','session'],
                                         columns='benchmark',
                                         values='benchmark_return')
                            .reset_index())
    bench_pivot.columns = [f"ret_{c}" if c not in ['date','session'] else c for c in bench_pivot.columns]
    benchmarks = [c.replace('ret_','') for c in bench_pivot.columns if c.startswith('ret_')]

    # === 7.2 股票半日收益 ===
    print("--- Preparing stock session returns ---")
    if 'date' not in session_df.columns:
        session_df['date'] = pd.to_datetime(session_df['datetime']).dt.date
    session_df['ret_session'] = session_df['close']/session_df['open'] - 1.0

    df_am = (session_df[session_df['session'].eq('AM')].copy()
             .rename(columns={'ret_session':'return_am'}))
    df_pm = (session_df[session_df['session'].eq('PM')].copy()
             .rename(columns={'ret_session':'return_pm'}))

    bench_am = bench_pivot[bench_pivot['session'].eq('AM')].drop(columns='session')
    bench_pm = bench_pivot[bench_pivot['session'].eq('PM')].drop(columns='session')

    df_am = df_am.merge(bench_am, on='date', how='left').sort_values(['symbol','date'])
    df_pm = df_pm.merge(bench_pm, on='date', how='left').sort_values(['symbol','date'])

    # === 7.3 TE / CORR ===
    print("--- Calculating TE/CORR per benchmark & window ---")
    for b in benchmarks:
        df_am[f'rel_ret_{b}'] = df_am['return_am'] - df_am[f'ret_{b}']
        df_pm[f'rel_ret_{b}'] = df_pm['return_pm'] - df_pm[f'ret_{b}']
        corr_cols = {}
        te_cols = {}
        for w in WINDOWS:
            te_cols[f'TE_{b}_{w}_AM'] = (
                df_am.groupby('symbol')[f'rel_ret_{b}']
                .transform(lambda s: s.rolling(w, min_periods=w//2).std() * np.sqrt(ANNUALIZATION))
            )
            te_cols[f'TE_{b}_{w}_PM'] = (
                df_pm.groupby('symbol')[f'rel_ret_{b}']
                .transform(lambda s: s.rolling(w, min_periods=w//2).std() * np.sqrt(ANNUALIZATION))
            )
            corr_cols[f'CORR_{b}_{w}_AM'] = (
                df_am.groupby('symbol', group_keys=False)
                .apply(lambda g: g['return_am'].rolling(w, min_periods=w//2).corr(g[f'ret_{b}']))
                .reset_index(level=0, drop=True)
            )
            corr_cols[f'CORR_{b}_{w}_PM'] = (
                df_pm.groupby('symbol', group_keys=False)
                .apply(lambda g: g['return_pm'].rolling(w, min_periods=w//2).corr(g[f'ret_{b}']))
                .reset_index(level=0, drop=True)
            )
        df_am = df_am.join(pd.DataFrame({k:v for k,v in te_cols.items() if k.endswith('_AM')}, index=df_am.index))
        df_pm = df_pm.join(pd.DataFrame({k:v for k,v in te_cols.items() if k.endswith('_PM')}, index=df_pm.index))
        df_am = df_am.join(pd.DataFrame({k:v for k,v in corr_cols.items() if k.endswith('_AM')}, index=df_am.index))
        df_pm = df_pm.join(pd.DataFrame({k:v for k,v in corr_cols.items() if k.endswith('_PM')}, index=df_pm.index))
        df_am = df_am.copy(); df_pm = df_pm.copy()

    # === 7.4 合并成宽表（_am/_pm） ===
    final_df = (pd.merge(df_am, df_pm, on=['symbol','date'], suffixes=('_am','_pm'), how='outer')
                  .sort_values(['symbol','date']).reset_index(drop=True))

    # === 7.5 VOL / TEshare / DTE / RTE / CROSS / dTE ===
    print("--- Building TEshare/DTE/RTE/CROSS_CORR/dTE ---")
    for w in WINDOWS:
        final_df[f'VOL_{w}_AM'] = (
            final_df.groupby('symbol')['return_am']
            .transform(lambda s: s.rolling(w, min_periods=w//2).std() * np.sqrt(ANNUALIZATION))
        )
        final_df[f'VOL_{w}_PM'] = (
            final_df.groupby('symbol')['return_pm']
            .transform(lambda s: s.rolling(w, min_periods=w//2).std() * np.sqrt(ANNUALIZATION))
        )

    for b in benchmarks:
        for w in WINDOWS:
            newcols = {}
            te_am = f'TE_{b}_{w}_AM'; te_pm = f'TE_{b}_{w}_PM'
            newcols[f'TEshare_{b}_{w}_AM'] = final_df[te_am] / (final_df[f'VOL_{w}_AM'] + 1e-8)
            newcols[f'TEshare_{b}_{w}_PM'] = final_df[te_pm] / (final_df[f'VOL_{w}_PM'] + 1e-8)

            # 截面 winsorize
            tmp_am = pd.Series(newcols[f'TEshare_{b}_{w}_AM'], index=final_df.index)
            tmp_pm = pd.Series(newcols[f'TEshare_{b}_{w}_PM'], index=final_df.index)
            newcols[f'TEshare_{b}_{w}_AM'] = (
                tmp_am.groupby(final_df['date']).apply(cs_winsorize).reset_index(level=0, drop=True)
            )
            newcols[f'TEshare_{b}_{w}_PM'] = (
                tmp_pm.groupby(final_df['date']).apply(cs_winsorize).reset_index(level=0, drop=True)
            )

            newcols[f'DTE_{b}_{w}'] = final_df[te_am] - final_df[te_pm]
            newcols[f'RTE_{b}_{w}'] = final_df[te_am] / (final_df[te_pm] + 1e-8)

            # 跨段相关
            cross_am_pm = (
                final_df.groupby('symbol', group_keys=False)
                .apply(lambda g: g['return_am'].rolling(w).corr(g[f'ret_{b}_pm']))
                .reset_index(level=0, drop=True)
            )
            cross_pm_am = (
                final_df.groupby('symbol', group_keys=False)
                .apply(lambda g: g['return_pm'].rolling(w).corr(g[f'ret_{b}_am']))
                .reset_index(level=0, drop=True)
            )
            newcols[f'CROSS_CORR_{b}_{w}_AM_PM'] = cross_am_pm
            newcols[f'CROSS_CORR_{b}_{w}_PM_AM'] = cross_pm_am

            newcols[f'dTE_{b}_{w}_AM'] = final_df.groupby('symbol')[te_am].transform(lambda s: s.diff(10))
            newcols[f'dTE_{b}_{w}_PM'] = final_df.groupby('symbol')[te_pm].transform(lambda s: s.diff(10))

            final_df = final_df.join(pd.DataFrame(newcols, index=final_df.index))
        final_df = final_df.copy()

    # === 7.6 ARGMIN（安全版） ===
    print("--- ARGMIN_TE ---")
    bench_arr = np.array(benchmarks, dtype=object)
    for w in WINDOWS:
        te_cols_am = [f'TE_{b}_{w}_AM' for b in benchmarks]
        te_cols_pm = [f'TE_{b}_{w}_PM' for b in benchmarks]

        vals_am = final_df[te_cols_am].to_numpy(float)
        mask_am = ~np.isnan(vals_am)
        row_any_am = mask_am.any(axis=1)
        safe_am = np.where(np.isnan(vals_am), np.inf, vals_am)
        idx_am = safe_am.argmin(axis=1)
        lab_am = np.full(len(final_df), np.nan, dtype=object)
        lab_am[row_any_am] = bench_arr[idx_am[row_any_am]]
        final_df[f'ARGMIN_TE_{w}_AM'] = lab_am

        vals_pm = final_df[te_cols_pm].to_numpy(float)
        mask_pm = ~np.isnan(vals_pm)
        row_any_pm = mask_pm.any(axis=1)
        safe_pm = np.where(np.isnan(vals_pm), np.inf, vals_pm)
        idx_pm = safe_pm.argmin(axis=1)
        lab_pm = np.full(len(final_df), np.nan, dtype=object)
        lab_pm[row_any_pm] = bench_arr[idx_pm[row_any_pm]]
        final_df[f'ARGMIN_TE_{w}_PM'] = lab_pm

    # === 7.7 IDIOVOL ===
    print("--- IDIOVOL ---")
    Xcols_am = [f'ret_{b}_am' for b in benchmarks]
    Xcols_pm = [f'ret_{b}_pm' for b in benchmarks]
    for w in WINDOWS:
        maskA = final_df['return_am'].notna()
        maskP = final_df['return_pm'].notna()
        def _idiovol(g, ycol, Xcols):
            y = g[ycol].values; X = g[Xcols].values
            resid = rolling_ols_residuals_array(y, X, w)
            return (pd.Series(resid, index=g.index)
                      .rolling(w, min_periods=w//2).std() * np.sqrt(ANNUALIZATION))
        final_df.loc[maskA, f'IDIOVOL_{w}_AM'] = (
            final_df[maskA].sort_values(['symbol','date'])
            .groupby('symbol', group_keys=False)
            .apply(lambda g: _idiovol(g, 'return_am', Xcols_am))
        )
        final_df.loc[maskP, f'IDIOVOL_{w}_PM'] = (
            final_df[maskP].sort_values(['symbol','date'])
            .groupby('symbol', group_keys=False)
            .apply(lambda g: _idiovol(g, 'return_pm', Xcols_pm))
        )

    final_df = final_df.replace([np.inf,-np.inf], np.nan)
    return final_df
