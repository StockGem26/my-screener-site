import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .config import HORIZONS
from .cache_io import _cache_path_close, _safe_read_cached_df, _safe_write_cached_df
from .time_utils import _is_cache_fresh, _now_ny, _today_ny_str
from .yf_batch import _normalize_yf_columns


# -----------------------------
# Formatting helpers
# -----------------------------
def _pct_str(x: float | None) -> str:
    """
    Clean +x.x% formatting and kills "-0.0%".
    Treat tiny values as 0.0 to avoid negative zero.
    """
    if x is None:
        return "—"
    try:
        if np.isnan(x) or np.isinf(x):
            return "—"
    except Exception:
        pass

    # kill -0.0 and tiny noise
    if abs(x) < 0.05:
        x = 0.0

    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.1f}%"


# -----------------------------
# Performance / history helpers (your existing ledger)
# -----------------------------
def _get_close_series_cached(sym: str, years: int = 8) -> pd.Series | None:
    """
    Cached daily close series, sorted by date.
    Uses a dedicated close cache because picks history can get older than 2y.
    """
    path = _cache_path_close(sym)
    cached_df = _safe_read_cached_df(path)
    if cached_df is not None and "Close" in cached_df.columns and not cached_df.empty:
        last_dt = cached_df.index.max()
        if _is_cache_fresh(last_dt):
            s = cached_df["Close"].dropna()
            return s.sort_index() if not s.empty else None

    try:
        data = yf.download(
            sym,
            period=f"{years}y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if data is None or data.empty:
            if cached_df is not None and "Close" in cached_df.columns:
                s = cached_df["Close"].dropna()
                return s.sort_index() if not s.empty else None
            return None

        data = _normalize_yf_columns(data)
        if "Close" not in data.columns:
            return None

        # Save close-only cache (as Close column, indexed by date)
        close_df = pd.DataFrame({"Close": data["Close"].dropna()})
        close_df.index = pd.to_datetime(close_df.index)
        _safe_write_cached_df(close_df, path)

        s = close_df["Close"].dropna()
        return s.sort_index() if not s.empty else None
    except Exception:
        if cached_df is not None and "Close" in cached_df.columns:
            s = cached_df["Close"].dropna()
            return s.sort_index() if not s.empty else None
        return None


def _forward_return_trading_days(close: pd.Series, entry_date: pd.Timestamp, entry_close: float, n: int) -> float | None:
    if close is None or close.empty:
        return None
    idx = close.index
    pos = idx.searchsorted(entry_date)
    if pos >= len(idx):
        return None
    target_pos = pos + n
    if target_pos >= len(idx):
        return None
    target_close = float(close.iloc[target_pos])
    return (target_close / entry_close - 1.0) * 100.0


def _compute_summary_stats(ret_series: pd.Series) -> dict:
    """
    Win-rate summary bar stats.
    Uses ret_now numeric series (percent).
    """
    s = pd.to_numeric(ret_series, errors="coerce").dropna()
    if s.empty:
        return {
            "count": 0,
            "win_rate": None,
            "avg": None,
            "median": None,
            "best": None,
            "worst": None,
        }

    win_rate = float((s > 0).mean() * 100.0)
    return {
        "count": int(len(s)),
        "win_rate": win_rate,
        "avg": float(s.mean()),
        "median": float(s.median()),
        "best": float(s.max()),
        "worst": float(s.min()),
    }


def update_history_and_build_perf_table(today_df: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, dict]:
    """
    Maintains docs/history/picks.csv (ledger of picks)
    Returns:
      (df_perf, summary_stats)

    df_perf columns:
      scan_date, days_since_scan, symbol, entry_close, Now, 15d, 30d, 60d, 100d, 200d
    """
    hist_dir = out_dir / "history"
    hist_dir.mkdir(parents=True, exist_ok=True)

    picks_path = hist_dir / "picks.csv"
    today = _today_ny_str()

    if today_df is None:
        today_df = pd.DataFrame()

    # Build today's new rows
    new_rows = []
    if not today_df.empty and "symbol" in today_df.columns and "close" in today_df.columns:
        for _, row in today_df.iterrows():
            sym = str(row["symbol"])
            entry_close = float(row["close"])
            new_rows.append({"scan_date": today, "symbol": sym, "entry_close": entry_close})
    df_new = pd.DataFrame(new_rows)

    # Load existing picks
    if picks_path.exists():
        df_picks = pd.read_csv(picks_path)
    else:
        df_picks = pd.DataFrame(columns=["scan_date", "symbol", "entry_close"])

    # Append only new (scan_date, symbol)
    if not df_new.empty:
        df_picks["scan_date"] = df_picks["scan_date"].astype(str)
        df_picks["symbol"] = df_picks["symbol"].astype(str)
        existing = set(zip(df_picks["scan_date"], df_picks["symbol"]))
        df_new = df_new[~df_new.apply(lambda r: (r["scan_date"], r["symbol"]) in existing, axis=1)]
        if not df_new.empty:
            df_picks = pd.concat([df_picks, df_new], ignore_index=True)

    # Sort newest first
    if not df_picks.empty:
        df_picks["scan_date"] = df_picks["scan_date"].astype(str)
        df_picks = df_picks.sort_values(["scan_date", "symbol"], ascending=[False, True]).reset_index(drop=True)

    df_picks.to_csv(picks_path, index=False)

    if df_picks.empty:
        empty_cols = ["scan_date", "days_since_scan", "symbol", "entry_close", "Now"] + [f"{n}d" for n in HORIZONS]
        return pd.DataFrame(columns=empty_cols), _compute_summary_stats(pd.Series(dtype=float))

    # Fetch close series for symbols in picks (cached)
    symbols = sorted(set(df_picks["symbol"].astype(str).tolist()))
    close_cache: dict[str, pd.Series | None] = {}

    def fetch_close(sym: str):
        close_cache[sym] = _get_close_series_cached(sym, years=8)

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(fetch_close, s) for s in symbols]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Updating performance", leave=False):
            pass

    perf_rows = []
    today_ny_date = _now_ny().date()

    for _, r in df_picks.iterrows():
        scan_date = str(r["scan_date"])
        sym = str(r["symbol"])
        entry_close = float(r["entry_close"])

        entry_date = pd.Timestamp(scan_date)
        close = close_cache.get(sym)

        # days_since_scan
        try:
            scan_dt = pd.to_datetime(scan_date).date()
            days_since = int((today_ny_date - scan_dt).days)
            if days_since < 0:
                days_since = 0
        except Exception:
            days_since = None

        row_out = {
            "scan_date": scan_date,
            "days_since_scan": days_since,
            "symbol": sym,
            "entry_close": entry_close,
        }

        latest_ret = None
        if close is not None and not close.empty:
            try:
                latest_close = float(close.iloc[-1])
                latest_ret = (latest_close / entry_close - 1.0) * 100.0
            except Exception:
                latest_ret = None
        row_out["ret_now"] = latest_ret

        for n in HORIZONS:
            row_out[f"ret_{n}d"] = _forward_return_trading_days(close, entry_date, entry_close, n)

        perf_rows.append(row_out)

    df_perf_raw = pd.DataFrame(perf_rows)
    if df_perf_raw.empty:
        empty_cols = ["scan_date", "days_since_scan", "symbol", "entry_close", "Now"] + [f"{n}d" for n in HORIZONS]
        return pd.DataFrame(columns=empty_cols), _compute_summary_stats(pd.Series(dtype=float))

    summary = _compute_summary_stats(df_perf_raw["ret_now"])

    # Convert to display strings
    df_perf_raw["Now"] = df_perf_raw["ret_now"].apply(_pct_str)
    for n in HORIZONS:
        df_perf_raw[f"{n}d"] = df_perf_raw[f"ret_{n}d"].apply(_pct_str)

    col_order = ["scan_date", "days_since_scan", "symbol", "entry_close", "Now"] + [f"{n}d" for n in HORIZONS]
    df_perf = df_perf_raw[col_order].copy()

    return df_perf, summary
