import pandas as pd
import yfinance as yf

from .cache_io import _cache_path_ohlcv, _safe_read_cached_df, _safe_write_cached_df
from .time_utils import _is_cache_fresh

def _normalize_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

def _extract_symbol_frame(batch: pd.DataFrame, sym: str) -> pd.DataFrame | None:
    if batch is None or batch.empty:
        return None

    if not isinstance(batch.columns, pd.MultiIndex):
        out = batch.copy()
        out = _normalize_yf_columns(out)
        return out

    cols = batch.columns

    if sym in cols.get_level_values(0):
        try:
            out = batch[sym].copy()
            out = _normalize_yf_columns(out)
            return out
        except Exception:
            pass

    if sym in cols.get_level_values(1):
        try:
            out = batch.xs(sym, level=1, axis=1).copy()
            out = _normalize_yf_columns(out)
            return out
        except Exception:
            pass

    return None

def _download_ohlcv_batch(symbols: list[str], period: str = "2y") -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if not symbols:
        return out

    tickers = " ".join(symbols)
    try:
        batch = yf.download(
            tickers=tickers,
            period=period,
            interval="1d",
            auto_adjust=False,
            group_by="ticker",
            progress=False,
            threads=True,
        )
    except Exception:
        return out

    for sym in symbols:
        df = _extract_symbol_frame(batch, sym)
        if df is None or df.empty:
            continue
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        needed = {"open", "high", "low", "close", "volume"}
        if not needed.issubset(set(df.columns)):
            continue
        df = df.dropna(subset=["close", "volume"]).copy()
        df.index = pd.to_datetime(df.index)
        out[sym] = df
    return out

def _get_ohlcv_cached_or_download(sym: str, period: str = "2y") -> pd.DataFrame | None:
    path = _cache_path_ohlcv(sym)
    cached = _safe_read_cached_df(path)
    if cached is not None and not cached.empty:
        last_dt = cached.index.max()
        if _is_cache_fresh(last_dt) and len(cached) >= 260:
            return cached

    try:
        df = yf.download(
            sym,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            return cached
        df = _normalize_yf_columns(df)
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        needed = {"open", "high", "low", "close", "volume"}
        if not needed.issubset(set(df.columns)):
            return cached
        df = df.dropna(subset=["close", "volume"]).copy()
        df.index = pd.to_datetime(df.index)
        _safe_write_cached_df(df, path)
        return df
    except Exception:
        return cached
