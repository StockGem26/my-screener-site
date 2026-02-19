import pandas as pd
from pathlib import Path
from .config import CACHE_DIR

def _safe_read_cached_df(path: Path) -> pd.DataFrame | None:
    try:
        if not path.exists():
            return None
        df = pd.read_csv(path, compression="gzip", parse_dates=["Date"])
        if df.empty:
            return None
        df = df.set_index("Date").sort_index()
        return df
    except Exception:
        return None

def _safe_write_cached_df(df: pd.DataFrame, path: Path) -> None:
    try:
        out = df.copy()
        out = out.reset_index().rename(columns={"index": "Date"})
        if "Date" not in out.columns:
            out = out.rename(columns={out.columns[0]: "Date"})
        out.to_csv(path, index=False, compression="gzip")
    except Exception:
        pass

def _cache_path_ohlcv(sym: str) -> Path:
    return CACHE_DIR / f"ohlcv_{sym}.csv.gz"

def _cache_path_close(sym: str) -> Path:
    return CACHE_DIR / f"close_{sym}.csv.gz"
