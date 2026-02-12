import io
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

HORIZONS = [15, 30, 60, 100, 200]  # trading days


# -----------------------------
# Performance / history helpers
# -----------------------------
def _pct_str(x: float | None) -> str:
    if x is None:
        return "—"
    try:
        if np.isnan(x) or np.isinf(x):
            return "—"
    except Exception:
        pass
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.1f}%"


def _get_close_series(symbol: str, years: int = 6) -> pd.Series | None:
    """Daily close series, sorted by date."""
    try:
        data = yf.download(
            symbol,
            period=f"{years}y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if data is None or data.empty:
            return None

        # Handle MultiIndex columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]

        if "Close" not in data.columns:
            return None

        s = data["Close"].dropna()
        if s.empty:
            return None

        return s.sort_index()
    except Exception:
        return None


def _forward_return_trading_days(close: pd.Series, entry_date: pd.Timestamp, entry_close: float, n: int) -> float | None:
    """
    Forward return after n trading sessions.
    Find first close date >= entry_date, then jump +n rows.
    """
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


def update_history_and_build_perf_table(today_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    Maintains docs/history/picks.csv (ledger of picks)
    Returns df_perf with columns:
      scan_date, symbol, entry_close, Now, 15d, 30d, ...
    """
    hist_dir = out_dir / "history"
    hist_dir.mkdir(parents=True, exist_ok=True)

    picks_path = hist_dir / "picks.csv"
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

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

    # Save ledger
    df_picks.to_csv(picks_path, index=False)

    # --- Compute performance for every pick ---
    close_cache: dict[str, pd.Series | None] = {}

    def get_close(sym: str) -> pd.Series | None:
        if sym not in close_cache:
            close_cache[sym] = _get_close_series(sym, years=6)
        return close_cache[sym]

    perf_rows = []
    for _, r in df_picks.iterrows():
        scan_date = str(r["scan_date"])
        sym = str(r["symbol"])
        entry_close = float(r["entry_close"])

        entry_date = pd.Timestamp(scan_date)
        close = get_close(sym)

        row_out = {"scan_date": scan_date, "symbol": sym, "entry_close": entry_close}

        # NOW return (latest close vs entry close)
        latest_ret = None
        if close is not None and not close.empty:
            try:
                latest_close = float(close.iloc[-1])
                latest_ret = (latest_close / entry_close - 1.0) * 100.0
            except Exception:
                latest_ret = None
        row_out["ret_now"] = latest_ret

        # Forward returns
        for n in HORIZONS:
            row_out[f"ret_{n}d"] = _forward_return_trading_days(close, entry_date, entry_close, n)

        perf_rows.append(row_out)

    df_perf = pd.DataFrame(perf_rows)

    # If empty, return empty in expected shape
    if df_perf.empty:
        return pd.DataFrame(columns=["scan_date", "symbol", "entry_close", "Now"] + [f"{n}d" for n in HORIZONS])

    # Pretty strings
    df_perf["ret_now_str"] = df_perf["ret_now"].apply(_pct_str)
    for n in HORIZONS:
        df_perf[f"ret_{n}d_str"] = df_perf[f"ret_{n}d"].apply(_pct_str)

    # Final column order + names (THIS is what makes "Now" appear)
    col_order = ["scan_date", "symbol", "entry_close", "ret_now_str"] + [f"ret_{n}d_str" for n in HORIZONS]
    df_perf = df_perf[col_order].rename(
        columns={"ret_now_str": "Now", **{f"ret_{n}d_str": f"{n}d" for n in HORIZONS}}
    )

    return df_perf


# -----------------------------
# Website output writer (GitHub Pages reads /docs/index.html)
# -----------------------------
def write_site(today_df: pd.DataFrame) -> None:
    out_dir = Path("docs")
    out_dir.mkdir(parents=True, exist_ok=True)

    if today_df is None:
        today_df = pd.DataFrame()

    # Save today's CSV
    today_df.to_csv(out_dir / "stage2_candidates.csv", index=False)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Build performance table (history page)
    df_perf = update_history_and_build_perf_table(today_df, out_dir)

    # Today's table HTML
    if today_df.empty:
        today_table_html = "<p>No Stage 2 matches found today.</p>"
    else:
        if "symbol" in today_df.columns:
            cols = ["symbol"] + [c for c in today_df.columns if c != "symbol"]
            today_df = today_df[cols]

        today_table_html = today_df.head(500).to_html(index=False, escape=True)
        today_table_html = today_table_html.replace("<table", '<table id="todayTable" class="display"', 1)

    # History table HTML (this WILL include the "Now" column)
    if df_perf.empty:
        hist_table_html = "<p>No historical picks yet.</p>"
    else:
        hist_table_html = df_perf.head(2000).to_html(index=False, escape=True)
        hist_table_html = hist_table_html.replace("<table", '<table id="histTable" class="display"', 1)

    # History page
    history_html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Stage 2 History & Performance</title>

  <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css"/>
  <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>

  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 30px;
      background-color: #0f172a;
      color: #e2e8f0;
    }}
    a {{ color: #38bdf8; }}
    .muted {{ color: #94a3b8; margin-bottom: 18px; }}
    table.dataTable {{ background-color: #1e293b; color: #e2e8f0; }}
    table.dataTable thead {{ background-color: #334155; }}
    .dataTables_wrapper .dataTables_length,
    .dataTables_wrapper .dataTables_filter,
    .dataTables_wrapper .dataTables_info,
    .dataTables_wrapper .dataTables_processing,
    .dataTables_wrapper .dataTables_paginate {{
      color: #e2e8f0 !important;
    }}
    .dataTables_wrapper .dataTables_filter input,
    .dataTables_wrapper .dataTables_length select {{
      background: #0b1220; color: #e2e8f0; border: 1px solid #334155;
      border-radius: 6px; padding: 4px 6px; outline: none;
    }}
  </style>
</head>
<body>
  <h1>📈 Stage 2 History & Performance</h1>
  <div class="muted">Updated: <b>{generated_at}</b> · Returns are <b>trading days</b>.</div>

  <p><a href="../index.html">← Back to today</a> · <a href="picks.csv">Download picks.csv</a></p>

  {hist_table_html}

  <script>
    window.addEventListener("load", function () {{
      try {{
        if (window.jQuery && $.fn && $.fn.DataTable && document.getElementById("histTable")) {{
          $('#histTable').DataTable({{
            pageLength: 50,
            order: [[0, 'desc']]
          }});
        }}
      }} catch (e) {{
        console.warn("DataTables init failed.", e);
      }}
    }});
  </script>
</body>
</html>
"""
    (out_dir / "history" / "index.html").write_text(history_html, encoding="utf-8")

    # Main index page
    main_html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Stage 2 Screener</title>

  <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css"/>
  <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>

  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 30px;
      background-color: #0f172a;
      color: #e2e8f0;
    }}
    a {{ color: #38bdf8; }}
    .muted {{ color: #94a3b8; margin-bottom: 18px; }}
    table.dataTable {{ background-color: #1e293b; color: #e2e8f0; }}
    table.dataTable thead {{ background-color: #334155; }}
    .dataTables_wrapper .dataTables_length,
    .dataTables_wrapper .dataTables_filter,
    .dataTables_wrapper .dataTables_info,
    .dataTables_wrapper .dataTables_processing,
    .dataTables_wrapper .dataTables_paginate {{
      color: #e2e8f0 !important;
    }}
    .dataTables_wrapper .dataTables_filter input,
    .dataTables_wrapper .dataTables_length select {{
      background: #0b1220; color: #e2e8f0; border: 1px solid #334155;
      border-radius: 6px; padding: 4px 6px; outline: none;
    }}
  </style>
</head>
<body>

  <h1>📊 Stage 2 Screener</h1>
  <div class="muted">Last updated: <b>{generated_at}</b></div>

  <p>
    <a href="stage2_candidates.csv">Download today’s CSV</a>
    · <a href="history/index.html">History & Performance</a>
  </p>

  {today_table_html}

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
        console.warn("DataTables init failed.", e);
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


def fetch_ohlcv_yahoo(symbol: str, period: str = "2y") -> pd.DataFrame | None:
    try:
        data = yf.download(
            symbol,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if data is None or data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]

        data = data.rename(
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
        if not needed.issubset(set(data.columns)):
            return None

        return data.dropna(subset=["close", "volume"])
    except Exception:
        return None


def scan_all_stage2(max_workers: int = 6, period: str = "2y") -> pd.DataFrame:
    symbols = fetch_us_ticker_universe_ex_etf()
    print(f"Universe size (ex-ETF): {len(symbols):,}")

    results = []
    failures = 0

    def worker(sym: str):
        df = fetch_ohlcv_yahoo(sym, period=period)
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
