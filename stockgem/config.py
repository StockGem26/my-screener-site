from zoneinfo import ZoneInfo
from pathlib import Path

HORIZONS = [15, 30, 60, 100, 200]  # trading days

CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

YF_CHUNK_SIZE = 250
CACHE_STALE_DAYS = 5

NY_TZ = ZoneInfo("America/New_York")

STOP_PCTS_DEFAULT = [5, 6, 7, 8]
