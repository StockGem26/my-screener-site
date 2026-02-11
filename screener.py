import io
import time
import math
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from datetime import datetime, timezone
from pathlib import Path


# -----------------------------
# Website output writer (GitHub Pages reads /web/index.html)
# -----------------------------
def write_site(df: pd.DataFrame) -> None:
    out_dir = Path("web")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Put the CSV INSIDE the web folder so the site can link to it
    df.to_csv(out_dir / "stage2_candidates.csv", index=False)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if df is None or df.empty:
        table_html = "<p>No Stage 2 matches found today.</p>"
    else:
        table_html = df.head(200).to_html(index=False, escape=True)

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Stage 2 Screener</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    .muted {{ color: #666; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; font-size: 14px; }}
    th {{ background: #f5f5f5; text-align: left; }}
  </style>
</head>
<body>
  <h1>Stage 2 Screener</h1>
  <p class="muted">Last updated: <b>{generated_at}</b></p>

  <p><a href="stage2_candidates.csv">Download stage2_candidates.csv</a></p>

  <h2>Top 200 Results</h2>
  {table_html}
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


# -----------------------------
# Universe: all US tickers (NYSE/NASDAQ/AMEX) excluding ETFs
# Source: NASDAQ Trader symbol directories (have an ETF flag)
# -----------------------------
def fetch_us_ticker_universe_ex_etf() -> list[str]:
    urls = {
        "nasdaqlisted": "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "otherlisted": "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    }

    def load_table(url: str) -> pd.DataFrame:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        # Files are pipe-delimited with a header and a trailing footer line like "File Creation Time: ..."
        text = r.text.strip().splitlines()
        text = "\n".join(text[:-1])  # drop footer
        return pd.read_csv(io.StringIO(text), sep="|")

    nasdaq = load_table(urls["nasdaqlisted"])
    other = load_table(urls["otherlisted"])

    other = other.rename(columns={"ACT Symbol": "Symbol"})

    # Keep only real tickers, exclude test issues, exclude ETFs using ETF flag == 'Y'
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Symbol"] = df["Symbol"].astype(str).str.strip()
        if "Test Issue" in df.columns:
            df = df[df["Test Issue"].astype(str).str.upper() != "Y"]
        if "ETF" in df.columns:
            df = df[df["ETF"].astype(str).str.upper() != "Y"]
        # Remove weird/empty symbols
        df = df[df["Symbol"].str.len() > 0]
        return df

    nasdaq = clean(nasdaq)
    other = clean(other)

    symbols = sorted(set(nasdaq["Symbol"].tolist()) | set(other["Symbol"].tolist()))
    symbols = [s.replace(".", "-") for s in symbols]  # Yahoo format for BRK.B -> BRK-B
    return symbols


# -----------------------------
# Stage 2 rules (daily-based)
# -----------------------------
def _slope(series: pd.Series, window: int = 20) -> float:
    s = series.dropna()
    if len(s) < window:
        return float("nan")
    y = s.iloc[-window:].to_numpy(dtype=float)
    x = np.arange(window, dtype=float)
    m = np.polyfit(x, y, 1)[0]
    return float(m)


def stage2_check(df: pd.DataFrame) -> tuple[bool, dict]:
    """
    Returns (passed, metrics)
    Requires columns: open, high, low, close, volume with a DatetimeIndex.
    """
    if df is None or df.empty or len(df) < 260:
        return False, {}

    close = df["close"]
    vol = df["volume"]

    sma50 = close.rolling(50).mean()
    sma150 = close.rolling(150).mean()
    sma200 = close.rolling(200).mean()
    vol50 = vol.rolling(50).mean()

    last = df.index[-1]

    # Rule 1: MA alignment
    cond_ma = sma50.loc[last] > sma150.loc[last] > sma200.loc[last]

    # Rule 2: 200 SMA rising (20 trading days)
    cond_slope = _slope(sma200, 20) > 0

    # Rule 3: Price above 50 & 150
    cond_price = close.loc[last] > sma50.loc[last] and close.loc[last] > sma150.loc[last]

    # Rule 4: Breakout (close above highest close of prior 65 days)
    lookback = 65
    if len(close) < lookback + 2:
        return False, {}
    prior_high = close.iloc[-(lookback + 1):-1].max()
    cond_breakout = close.loc[last] > prior_high

    # Rule 5: Volume confirmation (today vs 50-day avg)
    cond_vol = vol.loc[last] >= 1.4 * vol50.loc[last]

    # Rule 6: Not extended (>25% above 50dma = too extended)
    cond_not_extended = close.loc[last] <= 1.25 * sma50.loc[last]

    passed = all([cond_ma, cond_slope, cond_price, cond_breakout, cond_vol, cond_not_extended])

    metrics = {
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
# Yahoo download wrapper
# -----------------------------
def fetch_ohlcv_yahoo(symbol: str, period: str = "2y") -> pd.DataFrame | None:
    try:
        data = yf.download(
            symbol,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )
        if data is None or data.empty:
            return None

        data = data.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Adj Close": "adj_close", "Volume": "volume"
        })

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0].lower() for c in data.columns]

        needed = {"open", "high", "low", "close", "volume"}
        if not needed.issubset(set(data.columns)):
            return None

        data = data.dropna(subset=["close", "volume"])
        return data

    except Exception:
        return None


# -----------------------------
# Main scan
# -----------------------------
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

    # Your local file (optional)
    out.to_csv("stage2_candidates.csv", index=False)
    print("Saved: stage2_candidates.csv")

    # Website files for GitHub Pages
    write_site(out)
    print("Saved site files: web/index.html and web/stage2_candidates.csv")
