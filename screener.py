import argparse

from stockgem.config import STOP_PCTS_DEFAULT
from stockgem.scan import scan_all_stage2
from stockgem.site import write_site
from stockgem.replay import build_year_replay


def _parse_args():
    p = argparse.ArgumentParser(description="StockGem Stage 2 scanner + yearly replay generator")
    p.add_argument("--replay-year", type=int, default=None, help="Build a strategy replay for YEAR into docs/history/YEAR/")
    p.add_argument("--replay-period", type=str, default="2y", help="yfinance period for replay downloads (default 2y)")
    p.add_argument("--max-workers", type=int, default=6, help="Thread workers for daily scan")
    p.add_argument("--period", type=str, default="2y", help="yfinance period for daily scan")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Replay mode
    if args.replay_year is not None:
        build_year_replay(year=args.replay_year, period=args.replay_period, stop_pcts=STOP_PCTS_DEFAULT)
    else:
        # Normal daily scan + site update
        out = scan_all_stage2(max_workers=args.max_workers, period=args.period)
        write_site(out)
        print("Saved site files: docs/index.html, docs/stage2_candidates.csv, docs/history/picks.csv, docs/history/index.html")
