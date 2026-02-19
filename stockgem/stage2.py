import numpy as np
import pandas as pd

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

def _stage2_trigger_dates(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or len(df) < 260:
        return pd.DataFrame()

    close = df["close"].astype(float)
    low = df["low"].astype(float)
    vol = df["volume"].astype(float)

    sma50 = close.rolling(50).mean()
    sma150 = close.rolling(150).mean()
    sma200 = close.rolling(200).mean()
    vol50 = vol.rolling(50).mean()

    cond_ma = (sma50 > sma150) & (sma150 > sma200)
    sma200_slope20 = sma200.diff(20)
    cond_slope = sma200_slope20 > 0
    cond_price = (close > sma50) & (close > sma150)

    prior_high_65 = close.shift(1).rolling(65).max()
    breakout = close > prior_high_65
    trigger = breakout & (~breakout.shift(1).fillna(False))

    cond_vol = vol >= 1.4 * vol50
    cond_not_extended = close <= 1.25 * sma50

    passed = cond_ma & cond_slope & cond_price & trigger & cond_vol & cond_not_extended

    out = pd.DataFrame({
        "close": close,
        "low": low,
        "trigger": passed.fillna(False),
    }, index=df.index)

    return out.dropna(subset=["close", "low"])
