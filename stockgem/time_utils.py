import pandas as pd
from datetime import datetime, timezone
from .config import NY_TZ, CACHE_STALE_DAYS

def _now_ny() -> datetime:
    return datetime.now(NY_TZ)

def _today_ny_str() -> str:
    return _now_ny().strftime("%Y-%m-%d")

def _generated_at_ny_str() -> str:
    return _now_ny().strftime("%b %d, %Y · %I:%M %p ET")

def _today_utc_date() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc).date())

def _is_cache_fresh(last_dt: pd.Timestamp | None) -> bool:
    if last_dt is None:
        return False
    try:
        return (_today_utc_date() - pd.Timestamp(last_dt).normalize()) <= pd.Timedelta(days=CACHE_STALE_DAYS)
    except Exception:
        return False
