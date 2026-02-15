import io
import re
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

HORIZONS = [15, 30, 60, 100, 200]  # trading days

CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Batch download tuning
YF_CHUNK_SIZE = 250        # 200–400 usually best
CACHE_STALE_DAYS = 5       # allow weekends/holidays

NY_TZ = ZoneInfo("America/New_York")


# -----------------------------
# Time helpers (NY / UTC)
# -----------------------------
def _now_ny() -> datetime:
    return datetime.now(NY_TZ)


def _today_ny_str() -> str:
    # scan_date should match NY market day, not UTC rollover
    return _now_ny().strftime("%Y-%m-%d")


def _generated_at_ny_str() -> str:
    # Cosmetic #1: nicer timestamp format
    return _now_ny().strftime("%b %d, %Y · %I:%M %p ET")


def _today_utc_date() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc).date())


def _is_cache_fresh(last_dt: pd.Timestamp | None) -> bool:
    """Treat cache as fresh if last bar is within CACHE_STALE_DAYS calendar days."""
    if last_dt is None:
        return False
    try:
        return (_today_utc_date() - pd.Timestamp(last_dt).normalize()) <= pd.Timedelta(days=CACHE_STALE_DAYS)
    except Exception:
        return False


# -----------------------------
# Formatting helpers
# -----------------------------
def _pct_str(x: float | None) -> str:
    """
    Cosmetic #2: clean +x.x% formatting and kills "-0.0%".
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


# -----------------------------
# yfinance batch extraction
# -----------------------------
def _normalize_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """If MultiIndex, try to reduce to single-level for one symbol frames."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def _extract_symbol_frame(batch: pd.DataFrame, sym: str) -> pd.DataFrame | None:
    """
    yfinance can return:
    - Single ticker: columns = Open/High/Low/Close/Adj Close/Volume
    - Multi ticker: MultiIndex columns, either:
        level0=tickers, level1=fields   (group_by="ticker")
        OR level0=fields, level1=tickers
    We handle both.
    """
    if batch is None or batch.empty:
        return None

    # Single-ticker case
    if not isinstance(batch.columns, pd.MultiIndex):
        out = batch.copy()
        out = _normalize_yf_columns(out)
        return out

    cols = batch.columns

    # Case A: tickers on level 0
    if sym in cols.get_level_values(0):
        try:
            out = batch[sym].copy()
            out = _normalize_yf_columns(out)
            return out
        except Exception:
            pass

    # Case B: tickers on level 1
    if sym in cols.get_level_values(1):
        try:
            out = batch.xs(sym, level=1, axis=1).copy()
            out = _normalize_yf_columns(out)
            return out
        except Exception:
            pass

    return None


def _download_ohlcv_batch(symbols: list[str], period: str = "2y") -> dict[str, pd.DataFrame]:
    """
    Batch download OHLCV for many symbols, return dict {sym: df}.
    Uses group_by="ticker" to make extraction easier.
    """
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
    """
    Load cached ohlcv if fresh enough; otherwise download single symbol (fallback).
    Note: main scan uses batch downloads; this is just safety net.
    """
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


# -----------------------------
# Performance / history helpers
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
    Cosmetic #4: win-rate summary bar stats.
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


def _summary_html(summary: dict) -> str:
    """
    Render a compact stats bar (Cosmetic #4).
    """
    if not summary or summary.get("count", 0) == 0:
        return """<div class="stats">No performance stats yet.</div>"""

    win = _pct_str(summary["win_rate"]) if summary.get("win_rate") is not None else "—"
    avg = _pct_str(summary["avg"]) if summary.get("avg") is not None else "—"
    med = _pct_str(summary["median"]) if summary.get("median") is not None else "—"
    best = _pct_str(summary["best"]) if summary.get("best") is not None else "—"
    worst = _pct_str(summary["worst"]) if summary.get("worst") is not None else "—"

    # _pct_str expects "percent points" already; win_rate is already %
    # but _pct_str adds % and + sign; that's fine (e.g. +58.2%)
    return f"""
    <div class="stats">
      <span class="stat"><b>Rows with Now:</b> {summary["count"]}</span>
      <span class="stat"><b>Win rate (Now &gt; 0):</b> {win}</span>
      <span class="stat"><b>Avg Now:</b> {avg}</span>
      <span class="stat"><b>Median Now:</b> {med}</span>
      <span class="stat"><b>Best / Worst Now:</b> {best} / {worst}</span>
    </div>
    """


# -----------------------------
# Website output writer (GitHub Pages reads /docs/index.html)
# -----------------------------
def write_site(today_df: pd.DataFrame) -> None:
    out_dir = Path("docs")
    out_dir.mkdir(parents=True, exist_ok=True)

    if today_df is None:
        today_df = pd.DataFrame()

    # Save today's CSV (raw)
    today_df.to_csv(out_dir / "stage2_candidates.csv", index=False)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Build performance table (history page)
    df_perf = update_history_and_build_perf_table(today_df, out_dir)

    # -----------------------------
    # MAIN PAGE: simplify + format
    # -----------------------------
    today_view = today_df.copy()

    if not today_view.empty:
        # Keep only selected columns (you asked to remove sma50/sma150/sma200/prior_65d_high_close/sma200_slope20)
        keep_cols = [
            "symbol",
            "close",
            "volume",
            "vol50",
            "extended_pct_vs_50sma",
            "pivot_distance_pct",
        ]
        today_view = today_view[[c for c in keep_cols if c in today_view.columns]]

        # Nicely formatted numbers
        def fmt_price(x):
            try:
                return f"{float(x):,.2f}"
            except Exception:
                return str(x)

        def fmt_pct(x):
            try:
                return f"{float(x):.2f}%"
            except Exception:
                return str(x)

        def fmt_vol(x):
            try:
                x = float(x)
            except Exception:
                return str(x)
            if x >= 1_000_000_000:
                return f"{x/1_000_000_000:.2f}B"
            if x >= 1_000_000:
                return f"{x/1_000_000:.2f}M"
            if x >= 1_000:
                return f"{x/1_000:.2f}K"
            return f"{x:.0f}"

        if "close" in today_view.columns:
            today_view["close"] = today_view["close"].apply(fmt_price)
        if "volume" in today_view.columns:
            today_view["volume"] = today_view["volume"].apply(fmt_vol)
        if "vol50" in today_view.columns:
            today_view["vol50"] = today_view["vol50"].apply(fmt_vol)
        if "extended_pct_vs_50sma" in today_view.columns:
            today_view["extended_pct_vs_50sma"] = today_view["extended_pct_vs_50sma"].apply(fmt_pct)
        if "pivot_distance_pct" in today_view.columns:
            today_view["pivot_distance_pct"] = today_view["pivot_distance_pct"].apply(fmt_pct)

        # Friendlier column names
        rename_map = {
            "symbol": "Symbol",
            "close": "Close",
            "volume": "Volume",
            "vol50": "Avg Vol (50D)",
            "extended_pct_vs_50sma": "Ext vs 50D",
            "pivot_distance_pct": "From Pivot",
        }
        today_view = today_view.rename(columns=rename_map)

    # Today's table HTML
    if today_view.empty:
        today_table_html = """
        <div class="empty-state">
          <div class="empty-title">No picks today</div>
          <div class="empty-sub">No Stage 2 breakouts met your criteria on this scan.</div>
        </div>
        """
    else:
        today_table_html = today_view.head(500).to_html(index=False, escape=True)
        today_table_html = today_table_html.replace("<table", '<table id="todayTable" class="table"', 1)

    # History table HTML
    if df_perf.empty:
        hist_table_html = """
        <div class="empty-state">
          <div class="empty-title">No history yet</div>
          <div class="empty-sub">Once picks are recorded, performance will appear here automatically.</div>
        </div>
        """
    else:
        # Make history table nicer headers too
        dfh = df_perf.copy()
        dfh = dfh.rename(columns={
            "scan_date": "Scan Date",
            "symbol": "Symbol",
            "entry_close": "Entry",
            "Now": "Now",
            **{f"{n}d": f"{n}D" for n in HORIZONS},
        })
        hist_table_html = dfh.head(5000).to_html(index=False, escape=True)
        hist_table_html = hist_table_html.replace("<table", '<table id="histTable" class="table"', 1)

    # -----------------------------
    # Shared luxury light theme
    # -----------------------------
    shared_head = """
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

  <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css"/>
  <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>

  <style>
    :root{
      --bg0:#f8fafc;
      --bg1:#ffffff;
      --text:#0f172a;
      --muted:#64748b;
      --border: rgba(15,23,42,.08);
      --shadow: 0 18px 60px rgba(2,6,23,.10);
      --shadow2: 0 8px 24px rgba(2,6,23,.08);
      --radius: 22px;
      --accent: #2563eb;
      --accent2:#7c3aed;
      --pos:#16a34a;
      --neg:#dc2626;
    }

    *{ box-sizing:border-box; }
    body{
      margin:0;
      font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      color:var(--text);
      background:
        radial-gradient(1200px 600px at 15% 0%, rgba(37,99,235,.12), transparent 55%),
        radial-gradient(900px 500px at 85% 10%, rgba(124,58,237,.12), transparent 55%),
        linear-gradient(to bottom, var(--bg0), var(--bg0));
    }

    .container{
      max-width: 1120px;
      margin: 0 auto;
      padding: 28px 18px 60px;
    }

    .nav{
      display:flex;
      align-items:center;
      justify-content:space-between;
      padding: 14px 16px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,.72);
      backdrop-filter: blur(12px);
      border-radius: 999px;
      box-shadow: var(--shadow2);
      position: sticky;
      top: 14px;
      z-index: 50;
    }

    .brand{
      display:flex;
      align-items:center;
      gap:10px;
      font-weight:700;
      letter-spacing:-0.02em;
    }
    .dot{
      width:10px;height:10px;border-radius:99px;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      box-shadow: 0 0 0 6px rgba(37,99,235,.10);
    }

    .nav a{
      color: var(--muted);
      text-decoration:none;
      font-weight:600;
      margin-left: 14px;
      transition: 200ms ease;
    }
    .nav a:hover{ color: var(--text); }

    .btn{
      display:inline-flex;
      align-items:center;
      gap:8px;
      padding: 10px 14px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,.75);
      color: var(--text);
      font-weight:700;
      text-decoration:none;
      box-shadow: var(--shadow2);
      transition: 220ms ease;
    }
    .btn:hover{ transform: translateY(-1px); box-shadow: var(--shadow); }
    .btn.primary{
      border: none;
      color: white;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
    }

    .hero{
      margin-top: 22px;
      padding: 42px 28px;
      border-radius: var(--radius);
      border: 1px solid var(--border);
      background: rgba(255,255,255,.80);
      backdrop-filter: blur(14px);
      box-shadow: var(--shadow);
      overflow:hidden;
      position:relative;
    }
    .hero h1{
      margin:0 0 10px 0;
      font-size: clamp(32px, 4vw, 46px);
      line-height: 1.05;
      letter-spacing: -0.03em;
    }
    .hero p{
      margin:0;
      color: var(--muted);
      font-size: 16px;
      line-height: 1.6;
      max-width: 70ch;
    }

    .meta{
      margin-top: 14px;
      color: var(--muted);
      font-weight:600;
      font-size: 13px;
    }

    .chips{
      margin-top: 18px;
      display:flex;
      gap:10px;
      flex-wrap:wrap;
    }
    .chip{
      padding: 8px 10px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,.70);
      color: var(--muted);
      font-weight:700;
      font-size: 12px;
    }

    .grid{
      display:grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 14px;
      margin-top: 16px;
    }
    .card{
      grid-column: span 12;
      padding: 18px;
      border-radius: var(--radius);
      border: 1px solid var(--border);
      background: rgba(255,255,255,.85);
      box-shadow: var(--shadow2);
    }
    @media (min-width: 900px){
      .card.half{ grid-column: span 6; }
    }

    .card-title{
      font-weight:800;
      letter-spacing:-0.02em;
      margin:0 0 8px 0;
    }
    .card-sub{
      margin:0 0 12px 0;
      color: var(--muted);
      font-weight:600;
      font-size: 13px;
    }

    /* DataTables restyle */
    table.table{
      width:100%;
      border-collapse: separate !important;
      border-spacing: 0;
      overflow:hidden;
      border-radius: 16px;
      border: 1px solid var(--border);
      background: white;
    }
    table.table thead th{
      background: rgba(248,250,252,.9) !important;
      color: var(--muted) !important;
      font-weight:800 !important;
      border-bottom: 1px solid var(--border) !important;
      padding: 12px 12px !important;
    }
    table.table tbody td{
      padding: 12px 12px !important;
      border-bottom: 1px solid rgba(15,23,42,.06) !important;
      color: var(--text) !important; /* MAIN PAGE neutral */
      font-weight:600;
    }
    table.table tbody tr:hover td{
      background: rgba(37,99,235,.06) !important;
      transition: 180ms ease;
    }

    .dataTables_wrapper .dataTables_filter input,
    .dataTables_wrapper .dataTables_length select{
      border-radius: 999px;
      border: 1px solid var(--border);
      padding: 8px 10px;
      background: white;
      outline: none;
    }
    .dataTables_wrapper .dataTables_filter label,
    .dataTables_wrapper .dataTables_length label,
    .dataTables_wrapper .dataTables_info,
    .dataTables_wrapper .dataTables_paginate{
      color: var(--muted) !important;
      font-weight:600;
    }

    .toolbar{
      display:flex;
      gap:10px;
      align-items:center;
      flex-wrap:wrap;
      margin: 10px 0 16px 0;
    }
    .toolbar select{
      border-radius: 999px;
      border: 1px solid var(--border);
      padding: 8px 10px;
      background: white;
      outline:none;
      font-weight:700;
      color: var(--text);
    }

    /* Empty states */
    .empty-state{
      padding: 22px;
      border: 1px dashed rgba(15,23,42,.18);
      border-radius: 16px;
      background: rgba(248,250,252,.8);
    }
    .empty-title{ font-weight:900; letter-spacing:-0.02em; margin-bottom: 4px; }
    .empty-sub{ color: var(--muted); font-weight:600; }
  </style>
    """

    # JS: date dropdown + color ONLY % columns (history)
    shared_js = r"""
  <script>
    function isPercentLike(txt){
      if(!txt) return false;
      const s = String(txt).trim();
      if(s === "—") return false;
      return s.endsWith("%") && !isNaN(Number(s.replace("%","").replace("+","")));
    }
    function signOfPercent(txt){
      const s = String(txt).trim().replace("%","").replace("+","");
      const n = Number(s);
      if(!Number.isFinite(n)) return null;
      if(n > 0) return 1;
      if(n < 0) return -1;
      return 0;
    }
  </script>
    """

    # -----------------------------
    # HISTORY PAGE
    # -----------------------------
    history_html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Performance Ledger</title>
  {shared_head}
</head>
<body>
  <div class="container">

    <div class="nav">
      <div class="brand"><span class="dot"></span>StockGem</div>
      <div>
        <a href="../index.html">Today</a>
        <a href="index.html">Performance</a>
        <a href="picks.csv">Download CSV</a>
      </div>
      <a class="btn primary" href="../index.html">View today</a>
    </div>

    <div class="hero">
      <h1>Performance Ledger</h1>
      <p>
        Every pick is timestamped and tracked forward in trading days. This page is the proof archive.
      </p>
      <div class="meta">Updated: <b>{generated_at}</b></div>
      <div class="chips">
        <div class="chip">Daily after close</div>
        <div class="chip">Timestamped picks</div>
        <div class="chip">Forward returns</div>
        <div class="chip">Downloadable CSV</div>
      </div>
    </div>

    <div class="grid">
      <div class="card">
        <div class="card-title">Browse by date</div>
        <div class="card-sub">Select a scan date to filter the ledger to that day.</div>

        <div class="toolbar">
          <label for="dateFilter" style="font-weight:800; color: var(--muted);">Pick date</label>
          <select id="dateFilter">
            <option value="">All dates</option>
          </select>
        </div>

        {hist_table_html}
      </div>
    </div>

  </div>

  {shared_js}

  <script>
    window.addEventListener("load", function () {{
      try {{
        if (window.jQuery && $.fn && $.fn.DataTable && document.getElementById("histTable")) {{
          const table = $('#histTable').DataTable({{
            pageLength: 50,
            order: [[0, 'desc']]
          }});

          // Dropdown from unique Scan Date (column 0)
          const dateIdx = 0;
          const seen = new Set();
          table.column(dateIdx).data().each(function (d) {{
            if (d) seen.add(String(d).trim());
          }});
          const dates = Array.from(seen).sort().reverse();
          const sel = document.getElementById("dateFilter");
          dates.forEach(function (d) {{
            const opt = document.createElement("option");
            opt.value = d;
            opt.textContent = d;
            sel.appendChild(opt);
          }});
          sel.addEventListener("change", function () {{
            const v = this.value;
            if (!v) table.column(dateIdx).search("").draw();
            else table.column(dateIdx).search("^" + v + "$", true, false).draw();
          }});

        // Color ONLY return columns: Now, 15D, 30D, 60D, 100D, 200D
        function colorizeReturnColumns() {{
        const headers = [];
        $("#histTable thead th").each(function () {{
            headers.push($(this).text().trim().toUpperCase());
        }});

        const wanted = new Set(["NOW", "15D", "30D", "60D", "100D", "200D"]);
        const idxs = [];

        headers.forEach((h, i) => {{
            const key = h.replace(/\s+/g, "");
            if (wanted.has(key)) idxs.push(i);
        }});

        if (!idxs.length) return;

        $("#histTable tbody tr").each(function () {{
            const tds = $(this).find("td");

            idxs.forEach((i) => {{
            const cell = $(tds[i]);
            const txt = cell.text().trim();

            if (!txt || txt === "—") {{
                cell.css("color", "var(--muted)");
                return;
            }}

            const n = Number(txt.replace("%", "").replace("+", ""));
            if (!Number.isFinite(n)) return;

            cell.css("font-weight", "800");

            if (n > 0) cell.css("color", "var(--pos)");
            else if (n < 0) cell.css("color", "var(--neg)");
            else cell.css("color", "var(--muted)");
            }});
        }});
        }}

        colorizeReturnColumns();
        table.on("draw", function () {{ colorizeReturnColumns(); }});
        }}
      }} catch (e) {{
        console.warn("History init failed.", e);
      }}
    }});
  </script>
</body>
</html>
"""
    (out_dir / "history" / "index.html").write_text(history_html, encoding="utf-8")

    # -----------------------------
    # MAIN PAGE (luxury product landing)
    # -----------------------------
    main_html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>StockGem</title>
  {shared_head}
</head>
<body>
  <div class="container">

    <div class="nav">
      <div class="brand"><span class="dot"></span>StockGem</div>
      <div>
        <a href="index.html">Today</a>
        <a href="history/index.html">Performance</a>
        <a href="stage2_candidates.csv">Download</a>
      </div>
      <a class="btn primary" href="history/index.html">View ledger</a>
    </div>

    <div class="hero">
      <h1>Proof, not promises.</h1>
      <p>
        A beautiful, public performance ledger that tracks breakout picks forward in trading days.
        Every scan is timestamped. Every result updates automatically.
      </p>
      <div class="meta">Last updated: <b>{generated_at}</b></div>

      <div class="chips">
        <div class="chip">After-close updates</div>
        <div class="chip">Full historical ledger</div>
        <div class="chip">Forward returns</div>
        <div class="chip">Downloadable CSV</div>
      </div>

      <div style="margin-top:16px; display:flex; gap:10px; flex-wrap:wrap;">
        <a class="btn primary" href="#today">View today’s picks</a>
        <a class="btn" href="history/index.html">See performance ledger</a>
      </div>
    </div>

    <div class="grid" id="today">
      <div class="card">
        <div class="card-title">Today’s picks</div>
        <div class="card-sub">
          Curated list for today. Clean metrics only. Full proof lives in the ledger.
        </div>
        {today_table_html}
      </div>

      <div class="card half">
        <div class="card-title">What you’re seeing</div>
        <div class="card-sub">
          This page is not a screener. It’s a presentation layer for tracked picks.
        </div>
        <div style="color:var(--muted); font-weight:600; line-height:1.7;">
          <b>From Pivot</b> = percent above the prior 65D high (closer is tighter).<br/>
          <b>Ext vs 50D</b> = percent above the 50-day average (filters “too extended”).<br/>
          <b>Volume</b> = today’s volume, formatted (K/M/B).<br/>
          <b>Avg Vol (50D)</b> = typical volume baseline.
        </div>
      </div>

      <div class="card half">
        <div class="card-title">Want the proof?</div>
        <div class="card-sub">The ledger shows Now + 15D/30D/60D/100D/200D forward returns.</div>
        <a class="btn primary" href="history/index.html">Open performance ledger</a>
        <div style="height:10px;"></div>
        <a class="btn" href="history/picks.csv">Download picks.csv</a>
      </div>
    </div>

  </div>

  <script>
    window.addEventListener("load", function () {{
      try {{
        if (window.jQuery && $.fn && $.fn.DataTable && document.getElementById("todayTable")) {{
          $('#todayTable').DataTable({{
            pageLength: 25,
            order: []
          }});
        }}
      }} catch (e) {{
        console.warn("Today init failed.", e);
      }}
    }});
  </script>
</body>
</html>
"""
    (out_dir / "index.html").write_text(main_html, encoding="utf-8")


# -----------------------------
# Universe: all US tickers (NYSE/NASDAQ/AMEX) excluding ETFs
# -----------------------------
def fetch_us_ticker_universe_ex_etf() -> list[str]:
    urls = {
        "nasdaqlisted": "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "otherlisted": "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    }

    def load_table(url: str) -> pd.DataFrame:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        text = r.text.strip().splitlines()
        text = "\n".join(text[:-1])  # drop footer
        return pd.read_csv(io.StringIO(text), sep="|")

    nasdaq = load_table(urls["nasdaqlisted"])
    other = load_table(urls["otherlisted"]).rename(columns={"ACT Symbol": "Symbol"})

    def clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Symbol"] = df["Symbol"].astype(str).str.strip()
        if "Test Issue" in df.columns:
            df = df[df["Test Issue"].astype(str).str.upper() != "Y"]
        if "ETF" in df.columns:
            df = df[df["ETF"].astype(str).str.upper() != "Y"]
        df = df[df["Symbol"].str.len() > 0]
        return df

    nasdaq = clean(nasdaq)
    other = clean(other)

    symbols = sorted(set(nasdaq["Symbol"].tolist()) | set(other["Symbol"].tolist()))
    return [s.replace(".", "-") for s in symbols]  # Yahoo format BRK.B -> BRK-B


# -----------------------------
# Stage 2 rules (daily-based)
# -----------------------------
def _slope(series: pd.Series, window: int = 20) -> float:
    s = series.dropna()
    if len(s) < window:
        return float("nan")
    y = s.iloc[-window:].to_numpy(dtype=float)
    x = np.arange(window, dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def stage2_check(df: pd.DataFrame) -> tuple[bool, dict]:
    if df is None or df.empty or len(df) < 260:
        return False, {}

    close = df["close"]
    vol = df["volume"]

    sma50 = close.rolling(50).mean()
    sma150 = close.rolling(150).mean()
    sma200 = close.rolling(200).mean()
    vol50 = vol.rolling(50).mean()

    last = df.index[-1]

    cond_ma = sma50.loc[last] > sma150.loc[last] > sma200.loc[last]
    cond_slope = _slope(sma200, 20) > 0
    cond_price = close.loc[last] > sma50.loc[last] and close.loc[last] > sma150.loc[last]

    lookback = 65
    prior_high = close.iloc[-(lookback + 1):-1].max()
    cond_breakout = close.loc[last] > prior_high

    cond_vol = vol.loc[last] >= 1.4 * vol50.loc[last]
    cond_not_extended = close.loc[last] <= 1.25 * sma50.loc[last]

    passed = all([cond_ma, cond_slope, cond_price, cond_breakout, cond_vol, cond_not_extended])

    metrics = {
        "symbol": None,
        "close": float(close.loc[last]),
        "sma50": float(sma50.loc[last]),
        "sma150": float(sma150.loc[last]),
        "sma200": float(sma200.loc[last]),
        "prior_65d_high_close": float(prior_high),
        "volume": float(vol.loc[last]),
        "vol50": float(vol50.loc[last]),
        "sma200_slope20": float(_slope(sma200, 20)),
        "extended_pct_vs_50sma": float((close.loc[last] / sma50.loc[last] - 1) * 100.0),
    }
    return passed, metrics


# -----------------------------
# Faster scan (cache + batch yf)
# -----------------------------
def scan_all_stage2(max_workers: int = 6, period: str = "2y") -> pd.DataFrame:
    symbols = fetch_us_ticker_universe_ex_etf()
    print(f"Universe size (ex-ETF): {len(symbols):,}")

    # 1) Load fresh cached OHLCV where possible; build list to batch-download
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

    # 2) Batch download missing/stale symbols in chunks and write to cache
    downloaded_frames: dict[str, pd.DataFrame] = {}
    for i in tqdm(range(0, len(need_fetch), YF_CHUNK_SIZE), desc="Downloading batches"):
        chunk = need_fetch[i:i + YF_CHUNK_SIZE]
        batch_out = _download_ohlcv_batch(chunk, period=period)

        # Save batch results to cache
        for sym, df in batch_out.items():
            _safe_write_cached_df(df, _cache_path_ohlcv(sym))
        downloaded_frames.update(batch_out)

    # 3) Run Stage 2 checks (parallelized)
    all_frames = {**cached_frames, **downloaded_frames}
    results = []
    failures = 0

    def worker(sym: str):
        df = all_frames.get(sym)
        if df is None:
            # fallback: try single download or cached
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


if __name__ == "__main__":
    out = scan_all_stage2(max_workers=6, period="2y")
    write_site(out)
    print("Saved site files: docs/index.html, docs/stage2_candidates.csv, docs/history/picks.csv, docs/history/index.html")
