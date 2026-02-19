import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import YF_CHUNK_SIZE
from .cache_io import _cache_path_ohlcv, _safe_read_cached_df, _safe_write_cached_df
from .time_utils import _is_cache_fresh
from .yf_batch import _download_ohlcv_batch, _get_ohlcv_cached_or_download
from .universe import fetch_us_ticker_universe_ex_etf
from .stage2 import stage2_check

def scan_all_stage2(max_workers: int = 6, period: str = "2y") -> pd.DataFrame:
    symbols = fetch_us_ticker_universe_ex_etf()
    print(f"Universe size (ex-ETF): {len(symbols):,}")

    cached_frames: dict[str, pd.DataFrame] = {}
    need_fetch: list[str] = []

    for sym in symbols:
        path = _cache_path_ohlcv(sym)
        cached = _safe_read_cached_df(path)
        if cached is not None and not cached.empty:
            last_dt = cached.index.max()
            if _is_cache_fresh(last_dt) and len(cached) >= 260:
                cached_frames[sym] = cached
                continue
        need_fetch.append(sym)

    print(f"Using cached OHLCV: {len(cached_frames):,} | Need download: {len(need_fetch):,}")

    downloaded_frames: dict[str, pd.DataFrame] = {}
    for i in tqdm(range(0, len(need_fetch), YF_CHUNK_SIZE), desc="Downloading batches"):
        chunk = need_fetch[i:i + YF_CHUNK_SIZE]
        batch_out = _download_ohlcv_batch(chunk, period=period)

        for sym, df in batch_out.items():
            _safe_write_cached_df(df, _cache_path_ohlcv(sym))
        downloaded_frames.update(batch_out)

    all_frames = {**cached_frames, **downloaded_frames}
    results = []
    failures = 0

    def worker(sym: str):
        df = all_frames.get(sym)
        if df is None:
            df = _get_ohlcv_cached_or_download(sym, period=period)
        passed, metrics = stage2_check(df)
        if passed:
            metrics["symbol"] = sym
            return metrics
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(worker, s): s for s in symbols}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Scanning"):
            try:
                out = fut.result()
                if out is not None:
                    results.append(out)
            except Exception:
                failures += 1

    df_out = pd.DataFrame(results)
    if df_out.empty:
        print("No Stage 2 matches found.")
        return df_out

    df_out["pivot_distance_pct"] = (df_out["close"] / df_out["prior_65d_high_close"] - 1) * 100.0
    df_out = df_out.sort_values(["pivot_distance_pct", "extended_pct_vs_50sma"], ascending=[True, True])

    print(f"Stage 2 matches: {len(df_out):,} | Failures: {failures:,}")
    return df_out
