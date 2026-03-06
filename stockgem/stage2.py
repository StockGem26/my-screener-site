import numpy as np
import pandas as pd


# -----------------------------
# Core helpers
# -----------------------------
def _slope(series: pd.Series, window: int = 20) -> float:
    s = series.dropna()
    if len(s) < window:
        return float("nan")
    y = s.iloc[-window:].to_numpy(dtype=float)
    x = np.arange(window, dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    True Range ATR (simple rolling mean).
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(window).mean()


def _base_breakout_metrics(df: pd.DataFrame) -> tuple[bool, dict]:
    """
    Detects a Minervini-like base setup (contraction + dry volume) AND breakout.
    We do NOT use a fixed 65-day high.
    Instead we:
      - define a base window (40 trading days prior to today)
      - require volatility contraction + volume contraction into the right side
      - define a pivot as the max HIGH of the last 10 days within the base ("handle area")
      - breakout day = close above that pivot + volume expansion
    """
    if df is None or df.empty:
        return False, {}

    # Need enough history for MAs (200), vol50 (50), base windows (40), ATR (14)
    if len(df) < 260:
        return False, {}

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = df["volume"].astype(float)

    # MAs + volume baseline
    sma50 = close.rolling(50).mean()
    sma150 = close.rolling(150).mean()
    sma200 = close.rolling(200).mean()
    vol50 = vol.rolling(50).mean()

    # ATR% (volatility)
    atr14 = _atr(df, 14)
    atr_pct = (atr14 / close) * 100.0

    last = df.index[-1]

    # -----------------------------
    # Trend context (Stage 2-ish)
    # -----------------------------
    cond_ma = sma50.loc[last] > sma150.loc[last] > sma200.loc[last]
    cond_slope = _slope(sma200, 20) > 0
    cond_price = close.loc[last] > sma50.loc[last] and close.loc[last] > sma150.loc[last]

    # -----------------------------
    # Base window definition
    # -----------------------------
    base_len = 40        # last ~8 weeks
    half = 20            # split early/late
    handle_len = 10      # last 10 days of base define pivot

    if len(df) < (200 + base_len + 5):
        # extra safety
        return False, {}

    # Base is the 40 days BEFORE today
    base = df.iloc[-(base_len + 1):-1]  # excludes today
    if len(base) < base_len:
        return False, {}

    base_close = base["close"].astype(float)
    base_high = base["high"].astype(float)
    base_vol = base["volume"].astype(float)

    early = base.iloc[:half]
    late = base.iloc[half:]

    # -----------------------------
    # Base quality metrics
    # -----------------------------
    # Base depth (keep it reasonably tight)
    base_max = float(base_close.max())
    base_min = float(base_close.min())
    if base_max <= 0:
        return False, {}
    base_depth_pct = ((base_max - base_min) / base_max) * 100.0  # percent depth

    # Volatility contraction: ATR% late vs early
    # Use ATR% series aligned to df index, slice within early/late
    early_atr = atr_pct.loc[early.index].dropna()
    late_atr = atr_pct.loc[late.index].dropna()

    early_atr_mean = float(early_atr.mean()) if not early_atr.empty else float("nan")
    late_atr_mean = float(late_atr.mean()) if not late_atr.empty else float("nan")

    # Volume contraction: avg volume late vs early
    early_vol_mean = float(base_vol.iloc[:half].mean())
    late_vol_mean = float(base_vol.iloc[half:].mean())

    # Tightness: count big daily moves in the late half
    # (abs close-to-close > 3% in late 20)
    late_rets = base_close.pct_change().abs().loc[late.index]
    big_move_count = int((late_rets > 0.03).sum())

    # -----------------------------
    # Rules (tunable thresholds)
    # -----------------------------
    # Base depth <= 30%
    cond_depth = base_depth_pct <= 30.0

    # Volatility contraction: late ATR% <= 0.70 * early ATR%
    cond_vol_contraction = False
    if np.isfinite(early_atr_mean) and np.isfinite(late_atr_mean) and early_atr_mean > 0:
        cond_vol_contraction = (late_atr_mean <= 0.70 * early_atr_mean)

    # Volume drying: late volume <= 0.70 * early volume
    cond_volume_dry = False
    if early_vol_mean > 0:
        cond_volume_dry = (late_vol_mean <= 0.70 * early_vol_mean)

    # Tightness: no more than 3 big-move days in late 20
    cond_tight = big_move_count <= 3

    # Not too extended vs 50SMA (same as your old filter)
    cond_not_extended = close.loc[last] <= 1.25 * sma50.loc[last]

    # -----------------------------
    # Pivot derived from the base (NOT 65-day high)
    # -----------------------------
    handle = base.iloc[-handle_len:]
    pivot_level = float(handle["high"].max()) if not handle.empty else float("nan")

    # Breakout day: close > pivot + volume expansion
    cond_breakout = np.isfinite(pivot_level) and (close.loc[last] > pivot_level)
    cond_vol_expand = vol.loc[last] >= 1.4 * vol50.loc[last]

    passed = all([
        cond_ma,
        cond_slope,
        cond_price,
        cond_depth,
        cond_vol_contraction,
        cond_volume_dry,
        cond_tight,
        cond_breakout,
        cond_vol_expand,
        cond_not_extended,
    ])

    metrics = {
        "symbol": None,
        "close": float(close.loc[last]),
        "sma50": float(sma50.loc[last]),
        "sma150": float(sma150.loc[last]),
        "sma200": float(sma200.loc[last]),
        "volume": float(vol.loc[last]),
        "vol50": float(vol50.loc[last]),
        "sma200_slope20": float(_slope(sma200, 20)),
        "extended_pct_vs_50sma": float((close.loc[last] / sma50.loc[last] - 1) * 100.0),

        # new base/pivot fields
        "pivot_level": pivot_level,
        "base_depth_pct": float(base_depth_pct),
        "atr_pct_early": float(early_atr_mean) if np.isfinite(early_atr_mean) else None,
        "atr_pct_late": float(late_atr_mean) if np.isfinite(late_atr_mean) else None,
        "early_vol_mean": float(early_vol_mean),
        "late_vol_mean": float(late_vol_mean),
        "big_move_count_late20": int(big_move_count),
    }

    return passed, metrics


# -----------------------------
# Public API used by scan.py
# -----------------------------
def stage2_check(df: pd.DataFrame) -> tuple[bool, dict]:
    return _base_breakout_metrics(df)


def _stage2_trigger_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized(ish) trigger detection for replay.
    Produces trigger=True on breakout days only.

    Definitions:
      Base window: prior 40 trading days (excluding today)
        - Base depth <= 30%
        - Volatility contraction: ATR%(late20) <= 0.70 * ATR%(early20)
        - Volume drying: AvgVol(late20) <= 0.70 * AvgVol(early20)
        - Tightness: big moves (>3% abs return) in late20 <= 3
      Pivot: max HIGH of last 10 days of base (prior 10 days)
      Breakout: close > pivot AND volume >= 1.4 * vol50
      Plus trend context: 50>150>200, 200 rising, price above 50 and 150, not extended >25% above 50
    """
    if df is None or df.empty or len(df) < 260:
        return pd.DataFrame()

    df = df.copy()
    df = df.sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    vol = df["volume"].astype(float)

    sma50 = close.rolling(50).mean()
    sma150 = close.rolling(150).mean()
    sma200 = close.rolling(200).mean()
    vol50 = vol.rolling(50).mean()

    # 200SMA slope approximation like your old fast version
    sma200_slope20 = sma200.diff(20)

    # ATR% series
    atr14 = _atr(df, 14)
    atr_pct = (atr14 / close) * 100.0

    base_len = 40
    half = 20
    handle_len = 10

    # For a breakout day t, base is t-40..t-1 (40 days), split:
    # early: t-40..t-21 (20 days)
    # late:  t-20..t-1  (20 days)
    # We'll compute rolling means with shifts to line up with day t.

    # Volatility contraction ratios
    late_atr_mean = atr_pct.rolling(half).mean().shift(1)
    early_atr_mean = atr_pct.shift(half).rolling(half).mean().shift(1)

    # Volume drying ratios
    late_vol_mean = vol.rolling(half).mean().shift(1)
    early_vol_mean = vol.shift(half).rolling(half).mean().shift(1)

    # Base depth over prior 40 closes (excluding today)
    base_max = close.rolling(base_len).max().shift(1)
    base_min = close.rolling(base_len).min().shift(1)
    base_depth_pct = ((base_max - base_min) / base_max) * 100.0

    # Tightness: count abs daily return > 3% in late 20 (excluding today)
    abs_ret = close.pct_change().abs()
    big_move_count_late20 = (abs_ret > 0.03).rolling(half).sum().shift(1)

    # Pivot = max HIGH of last 10 days before today
    pivot_level = high.rolling(handle_len).max().shift(1)

    # Conditions
    cond_ma = (sma50 > sma150) & (sma150 > sma200)
    cond_slope = sma200_slope20 > 0
    cond_price = (close > sma50) & (close > sma150)

    cond_depth = base_depth_pct <= 30.0

    cond_vol_contraction = (late_atr_mean <= 0.70 * early_atr_mean) & (early_atr_mean > 0)
    cond_volume_dry = (late_vol_mean <= 0.70 * early_vol_mean) & (early_vol_mean > 0)
    cond_tight = big_move_count_late20 <= 3

    cond_not_extended = close <= 1.25 * sma50

    cond_breakout = close > pivot_level
    cond_vol_expand = vol >= 1.4 * vol50

    passed = (
        cond_ma
        & cond_slope
        & cond_price
        & cond_depth
        & cond_vol_contraction
        & cond_volume_dry
        & cond_tight
        & cond_breakout
        & cond_vol_expand
        & cond_not_extended
    ).fillna(False)

    # Trigger day ONLY (first day it becomes true)
    trigger = passed & (~passed.shift(1).fillna(False))

    out = pd.DataFrame(
        {
            "close": close,
            "low": df["low"].astype(float),
            "trigger": trigger,
            "pivot_level": pivot_level,
            "base_depth_pct": base_depth_pct,
        },
        index=df.index,
    )

    return out.dropna(subset=["close", "low"])
