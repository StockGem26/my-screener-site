import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from .config import HORIZONS, STOP_PCTS_DEFAULT, YF_CHUNK_SIZE
from .universe import fetch_us_ticker_universe_ex_etf
from .cache_io import _cache_path_ohlcv, _safe_read_cached_df, _safe_write_cached_df
from .yf_batch import _download_ohlcv_batch
from .stage2 import _stage2_trigger_dates
from .time_utils import _generated_at_ny_str


def _trade_stats(returns: pd.Series, stop_hit: pd.Series | None = None) -> dict:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return {
            "count": 0,
            "win_rate": None,
            "avg_gain": None,
            "avg_loss": None,
            "expectancy": None,
            "profit_factor": None,
            "stop_hit_rate": None if stop_hit is None else None,
        }

    winners = r[r > 0]
    losers = r[r < 0]

    win_rate = float((r > 0).mean() * 100.0)
    avg_gain = float(winners.mean()) if not winners.empty else 0.0
    avg_loss = float(losers.mean()) if not losers.empty else 0.0  # negative
    expectancy = float(r.mean())

    sum_gains = float(winners.sum()) if not winners.empty else 0.0
    sum_losses = float(losers.sum()) if not losers.empty else 0.0  # negative
    profit_factor = None
    if sum_losses != 0:
        profit_factor = float(sum_gains / abs(sum_losses))
    elif sum_gains > 0:
        profit_factor = float("inf")

    stop_hit_rate = None
    if stop_hit is not None:
        sh = pd.to_numeric(stop_hit, errors="coerce").dropna()
        if not sh.empty:
            stop_hit_rate = float((sh.astype(bool)).mean() * 100.0)

    return {
        "count": int(len(r)),
        "win_rate": win_rate,
        "avg_gain": avg_gain,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "stop_hit_rate": stop_hit_rate,
    }


def build_year_replay(year: int, period: str = "2y", stop_pcts: list[int] | None = None) -> None:
    """
    Builds:
      docs/history/<year>/signals.csv
      docs/history/<year>/outcomes.csv
      docs/history/<year>/summary.json

    Entry: signal-day close
    Stop: if future low <= stop price, exit at stop price (conservative)
    """
    if stop_pcts is None:
        stop_pcts = STOP_PCTS_DEFAULT

    out_dir = Path("docs") / "history" / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = fetch_us_ticker_universe_ex_etf()
    print(f"[Replay {year}] Universe size (ex-ETF): {len(symbols):,}")

    cached_frames: dict[str, pd.DataFrame] = {}
    need_fetch: list[str] = []

    for sym in symbols:
        path = _cache_path_ohlcv(sym)
        cached = _safe_read_cached_df(path)
        if cached is not None and not cached.empty and len(cached) >= 260:
            cached_frames[sym] = cached
        else:
            need_fetch.append(sym)

    if need_fetch:
        print(f"[Replay {year}] Downloading missing OHLCV for {len(need_fetch):,} symbols...")
        for i in tqdm(range(0, len(need_fetch), YF_CHUNK_SIZE), desc=f"[Replay {year}] Download batches"):
            chunk = need_fetch[i:i + YF_CHUNK_SIZE]
            batch_out = _download_ohlcv_batch(chunk, period=period)
            for sym, df in batch_out.items():
                _safe_write_cached_df(df, _cache_path_ohlcv(sym))
            cached_frames.update(batch_out)

    signals_rows = []
    outcome_rows = []

    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31")

    for sym in tqdm(symbols, desc=f"[Replay {year}] Processing symbols"):
        df = cached_frames.get(sym)
        if df is None or df.empty:
            continue

        df = df.sort_index()
        df.index = pd.to_datetime(df.index).tz_localize(None)

        trig = _stage2_trigger_dates(df)
        if trig.empty:
            continue

        in_year = trig[(trig.index >= start) & (trig.index <= end) & (trig["trigger"] == True)]
        if in_year.empty:
            continue

        closes = df["close"].astype(float).to_numpy()
        lows = df["low"].astype(float).to_numpy()
        idx = df.index

        for dt_sig in in_year.index:
            pos = idx.searchsorted(dt_sig)
            if pos >= len(idx):
                continue

            entry_close = float(closes[pos])

            signals_rows.append({
                "signal_date": dt_sig.strftime("%Y-%m-%d"),
                "symbol": sym,
                "entry_close": entry_close,
            })

            row = {
                "signal_date": dt_sig.strftime("%Y-%m-%d"),
                "symbol": sym,
                "entry_close": entry_close,
            }

            for n in HORIZONS:
                if pos + n >= len(idx):
                    row[f"ret_{n}d"] = np.nan
                    for sp in stop_pcts:
                        row[f"ret_{n}d_stop{sp}"] = np.nan
                        row[f"stop_hit_{n}d_stop{sp}"] = np.nan
                    continue

                exit_close = float(closes[pos + n])
                base_ret = (exit_close / entry_close - 1.0) * 100.0
                row[f"ret_{n}d"] = base_ret

                window_lows = lows[pos + 1: pos + n + 1]
                for sp in stop_pcts:
                    stop_price = entry_close * (1.0 - sp / 100.0)
                    hit = bool(np.any(window_lows <= stop_price)) if len(window_lows) > 0 else False
                    if hit:
                        ret_stop = (stop_price / entry_close - 1.0) * 100.0
                        row[f"ret_{n}d_stop{sp}"] = ret_stop
                        row[f"stop_hit_{n}d_stop{sp}"] = 1
                    else:
                        row[f"ret_{n}d_stop{sp}"] = base_ret
                        row[f"stop_hit_{n}d_stop{sp}"] = 0

            outcome_rows.append(row)

    df_signals = pd.DataFrame(signals_rows)
    df_outcomes = pd.DataFrame(outcome_rows)

    signals_path = out_dir / "signals.csv"
    outcomes_path = out_dir / "outcomes.csv"
    summary_path = out_dir / "summary.json"

    df_signals.to_csv(signals_path, index=False)
    df_outcomes.to_csv(outcomes_path, index=False)

    summary = {
        "year": year,
        "generated_at_et": _generated_at_ny_str(),
        "entry": "signal-day close",
        "stop_rule": "if future low <= stop price, exit at stop price",
        "stop_pcts": stop_pcts,
        "horizons": HORIZONS,
        "stats": {},
    }

    if df_outcomes.empty:
        summary["stats"] = {}
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[Replay {year}] No signals found. Wrote empty outputs to {out_dir}")
        return

    summary["stats"]["off"] = {}
    for n in HORIZONS:
        s = df_outcomes[f"ret_{n}d"] if f"ret_{n}d" in df_outcomes.columns else pd.Series(dtype=float)
        summary["stats"]["off"][str(n)] = _trade_stats(s)

    for sp in stop_pcts:
        key = f"stop{sp}"
        summary["stats"][key] = {}
        for n in HORIZONS:
            rcol = f"ret_{n}d_stop{sp}"
            hcol = f"stop_hit_{n}d_stop{sp}"
            r = df_outcomes[rcol] if rcol in df_outcomes.columns else pd.Series(dtype=float)
            h = df_outcomes[hcol] if hcol in df_outcomes.columns else None
            summary["stats"][key][str(n)] = _trade_stats(r, h)

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[Replay {year}] Wrote:")
    print(f"  - {signals_path}")
    print(f"  - {outcomes_path}")
    print(f"  - {summary_path}")
