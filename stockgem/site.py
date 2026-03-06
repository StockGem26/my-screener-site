from pathlib import Path
import pandas as pd
import json

from .config import HORIZONS
from .time_utils import _generated_at_ny_str
from .history import update_history_and_build_perf_table


# ─────────────────────────────────────────────
#  Shared CSS / head
# ─────────────────────────────────────────────

SHARED_HEAD = """
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,800;1,700&family=DM+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css"/>
  <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>

  <style>
    :root {
      --navy:    #1b2b5e;
      --green:   #2d5a3d;
      --green2:  #3a7050;
      --bg:      #f9f8f6;
      --surface: #ffffff;
      --border:  rgba(27,43,94,.10);
      --border2: rgba(27,43,94,.06);
      --text:    #111827;
      --muted:   #6b7280;
      --muted2:  #9ca3af;
      --pos:     #16803c;
      --neg:     #b91c1c;
      --pos-bg:  rgba(22,128,60,.08);
      --neg-bg:  rgba(185,28,28,.07);
      --radius:  16px;
      --radius-sm: 10px;
      --shadow-sm: 0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04);
      --shadow:    0 4px 16px rgba(27,43,94,.10);
      --shadow-lg: 0 12px 40px rgba(27,43,94,.14);
    }

    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    html { scroll-behavior: smooth; }
    body {
      font-family: 'DM Sans', system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
    }

    /* Layout */
    .page-wrap { max-width: 1200px; margin: 0 auto; padding: 0 20px 80px; }

    /* Nav */
    .nav-outer {
      position: sticky; top: 0; z-index: 100;
      background: rgba(249,248,246,.92);
      backdrop-filter: blur(14px);
      border-bottom: 1px solid var(--border);
    }
    .nav {
      max-width: 1200px; margin: 0 auto; padding: 0 20px;
      height: 60px; display: flex; align-items: center;
      justify-content: space-between; gap: 16px;
    }
    .nav-logo img { height: 30px; display: block; }
    .nav-links { display: flex; align-items: center; gap: 2px; }
    .nav-links a {
      text-decoration: none; color: var(--muted); font-size: 14px;
      font-weight: 500; padding: 6px 12px; border-radius: 8px;
      transition: background 150ms, color 150ms;
    }
    .nav-links a:hover { background: rgba(27,43,94,.07); color: var(--navy); }
    .nav-links a.active { color: var(--navy); font-weight: 600; }
    .nav-cta {
      text-decoration: none;
      background: linear-gradient(135deg, var(--navy), var(--green));
      color: #fff; font-size: 13px; font-weight: 600;
      padding: 8px 18px; border-radius: 999px; white-space: nowrap;
      box-shadow: 0 2px 8px rgba(27,43,94,.25);
      transition: opacity 150ms, transform 150ms;
    }
    .nav-cta:hover { opacity: .88; transform: translateY(-1px); }
    .nav-toggle { display: none; background: none; border: none; cursor: pointer; padding: 4px; }
    .nav-toggle span { display: block; width: 22px; height: 2px; background: var(--text); margin: 5px 0; border-radius: 2px; }
    @media (max-width: 680px) {
      .nav-links {
        display: none; position: absolute; top: 60px; left: 0; right: 0;
        background: var(--surface); border-bottom: 1px solid var(--border);
        flex-direction: column; align-items: flex-start;
        padding: 12px 20px 16px; gap: 2px;
      }
      .nav-links.open { display: flex; }
      .nav-links a { width: 100%; }
      .nav-toggle { display: block; }
    }

    /* Hero */
    .hero {
      margin-top: 48px;
      display: grid; grid-template-columns: 1fr auto;
      gap: 40px; align-items: start;
    }
    @media (max-width: 700px) { .hero { grid-template-columns: 1fr; } }
    .hero-eyebrow {
      font-size: 11px; font-weight: 700; letter-spacing: .12em;
      text-transform: uppercase; color: var(--green2); margin-bottom: 12px;
    }
    .hero-title {
      font-family: 'Playfair Display', Georgia, serif;
      font-size: clamp(36px, 5vw, 56px); font-weight: 800;
      line-height: 1.07; letter-spacing: -.02em; color: var(--navy);
    }
    .hero-title em { font-style: italic; color: var(--green); }
    .hero-sub { margin-top: 16px; font-size: 16px; color: var(--muted); max-width: 48ch; line-height: 1.65; }
    .hero-meta { margin-top: 18px; display: flex; align-items: center; gap: 8px; font-size: 12px; font-weight: 600; color: var(--muted2); }
    .pulse {
      width: 7px; height: 7px; border-radius: 50%; background: var(--pos);
      box-shadow: 0 0 0 0 rgba(22,128,60,.5); animation: pulse 2s infinite;
    }
    @keyframes pulse {
      0%   { box-shadow: 0 0 0 0 rgba(22,128,60,.5); }
      70%  { box-shadow: 0 0 0 7px rgba(22,128,60,0); }
      100% { box-shadow: 0 0 0 0 rgba(22,128,60,0); }
    }
    .hero-actions { margin-top: 26px; display: flex; gap: 10px; flex-wrap: wrap; }
    .chips { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 18px; }
    .chip {
      font-size: 12px; font-weight: 600; color: var(--navy);
      background: rgba(27,43,94,.06); border: 1px solid rgba(27,43,94,.10);
      padding: 5px 12px; border-radius: 999px;
    }

    /* Stat sidebar */
    .hero-stats { display: flex; flex-direction: column; gap: 10px; min-width: 190px; }
    @media (max-width: 700px) { .hero-stats { flex-direction: row; flex-wrap: wrap; } }
    .stat-card {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: var(--radius-sm); padding: 14px 18px; box-shadow: var(--shadow-sm);
    }
    .stat-label { font-size: 11px; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; color: var(--muted2); margin-bottom: 4px; }
    .stat-value { font-family: 'Playfair Display', Georgia, serif; font-size: 22px; font-weight: 700; color: var(--navy); }
    .stat-value.pos { color: var(--pos); }
    .stat-value.neg { color: var(--neg); }

    /* Buttons */
    .btn {
      display: inline-flex; align-items: center; gap: 6px; text-decoration: none;
      font-size: 14px; font-weight: 600; padding: 10px 20px; border-radius: 999px;
      transition: all 150ms; border: none; cursor: pointer;
    }
    .btn-primary {
      background: linear-gradient(135deg, var(--navy), var(--green));
      color: #fff; box-shadow: 0 3px 12px rgba(27,43,94,.25);
    }
    .btn-primary:hover { opacity: .88; transform: translateY(-1px); }
    .btn-outline {
      background: var(--surface); color: var(--navy);
      border: 1.5px solid var(--border); box-shadow: var(--shadow-sm);
    }
    .btn-outline:hover { background: rgba(27,43,94,.04); border-color: rgba(27,43,94,.22); }

    /* Divider */
    .divider { height: 1px; background: var(--border); margin: 40px 0; }

    /* Section */
    .section-header { display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap; gap: 10px; margin-bottom: 16px; }
    .section-title { font-family: 'Playfair Display', Georgia, serif; font-size: 22px; font-weight: 700; color: var(--navy); letter-spacing: -.01em; }
    .section-sub { font-size: 13px; color: var(--muted); }

    /* Cards */
    .card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); box-shadow: var(--shadow-sm); overflow: hidden; }
    .card-body { padding: 24px; }
    .card-header { padding: 18px 24px 16px; border-bottom: 1px solid var(--border2); }
    .card-header h2 { font-family: 'Playfair Display', Georgia, serif; font-size: 18px; font-weight: 700; color: var(--navy); }
    .card-header p { font-size: 13px; color: var(--muted); margin-top: 3px; }

    .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    @media (max-width: 720px) { .two-col { grid-template-columns: 1fr; } }

    /* Explainer */
    .explainer-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
    @media (max-width: 520px) { .explainer-grid { grid-template-columns: 1fr; } }
    .explainer-item { padding: 14px 16px; background: rgba(27,43,94,.03); border: 1px solid var(--border2); border-radius: var(--radius-sm); }
    .explainer-item strong { display: block; font-size: 13px; font-weight: 700; color: var(--navy); margin-bottom: 3px; }
    .explainer-item span { font-size: 13px; color: var(--muted); line-height: 1.5; }

    /* Tables */
    .table-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; }
    table.sg-table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
    table.sg-table thead th {
      background: rgba(27,43,94,.04); color: var(--muted);
      font-size: 11px; font-weight: 700; letter-spacing: .07em; text-transform: uppercase;
      padding: 10px 14px; border-bottom: 1px solid var(--border); white-space: nowrap; text-align: left;
    }
    table.sg-table tbody td {
      padding: 11px 14px; border-bottom: 1px solid var(--border2);
      color: var(--text); font-weight: 500; white-space: nowrap;
    }
    table.sg-table tbody tr:last-child td { border-bottom: none; }
    table.sg-table tbody tr:hover td { background: rgba(27,43,94,.025); }
    .sym-cell { font-weight: 700; color: var(--navy); font-size: 14px; letter-spacing: .02em; }
    td.ret-pos { color: var(--pos); font-weight: 600; background: var(--pos-bg); }
    td.ret-neg { color: var(--neg); font-weight: 600; background: var(--neg-bg); }
    td.ret-now { font-weight: 700 !important; }

    /* DataTables overrides */
    .dataTables_wrapper .dataTables_filter input,
    .dataTables_wrapper .dataTables_length select {
      border: 1px solid var(--border); border-radius: 8px;
      padding: 6px 10px; background: var(--surface); color: var(--text); outline: none;
    }
    .dataTables_wrapper .dataTables_filter,
    .dataTables_wrapper .dataTables_length,
    .dataTables_wrapper .dataTables_info,
    .dataTables_wrapper .dataTables_paginate {
      color: var(--muted) !important; font-weight: 500; font-size: 13px; padding: 10px 0;
    }
    .dataTables_wrapper .dataTables_paginate .paginate_button { border-radius: 6px !important; border: none !important; }
    .dataTables_wrapper .dataTables_paginate .paginate_button.current {
      background: linear-gradient(135deg, var(--navy), var(--green)) !important;
      color: #fff !important; border: none !important;
    }

    /* Chart */
    .chart-wrap { position: relative; height: 240px; padding: 20px 24px; }

    /* Filter row */
    .filter-row { display: flex; align-items: center; gap: 10px; padding: 14px 24px 0; flex-wrap: wrap; }
    .filter-row label { font-size: 13px; font-weight: 600; color: var(--muted); }
    .filter-row select {
      font-size: 13px; font-weight: 600; border: 1px solid var(--border);
      border-radius: 8px; padding: 7px 12px; background: var(--surface); color: var(--navy); outline: none; cursor: pointer;
    }

    /* Empty */
    .empty { padding: 56px 24px; text-align: center; }
    .empty-icon { font-size: 38px; margin-bottom: 12px; }
    .empty-title { font-family: 'Playfair Display', Georgia, serif; font-size: 20px; font-weight: 700; color: var(--navy); }
    .empty-sub { font-size: 14px; color: var(--muted); margin-top: 8px; }

    /* Footer */
    .footer {
      margin-top: 60px; padding-top: 24px; border-top: 1px solid var(--border);
      display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;
    }
    .footer img { height: 22px; opacity: .45; }
    .footer p { font-size: 12px; color: var(--muted2); }

    /* Animations */
    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(16px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    .fade-up   { animation: fadeUp .45s ease both; }
    .delay-1   { animation-delay: .08s; }
    .delay-2   { animation-delay: .16s; }
    .delay-3   { animation-delay: .24s; }
    .delay-4   { animation-delay: .32s; }
  </style>
"""


# ─────────────────────────────────────────────
#  Nav + Footer templates
# ─────────────────────────────────────────────

def _nav(root: str = "", active: str = "today") -> str:
    ta = ' class="active"' if active == "today" else ""
    ha = ' class="active"' if active == "history" else ""
    return f"""
  <nav class="nav-outer">
    <div class="nav">
      <a class="nav-logo" href="{root}index.html">
        <img src="{root}assets/logo.png" alt="StockGems"/>
      </a>
      <div class="nav-links" id="navLinks">
        <a href="{root}index.html"{ta}>Today</a>
        <a href="{root}history/index.html"{ha}>Performance</a>
        <a href="{root}history/2025/">2025 Results</a>
        <a href="{root}stage2_candidates.csv">CSV</a>
      </div>
      <a class="nav-cta" href="{root}history/index.html">View Ledger →</a>
      <button class="nav-toggle" onclick="document.getElementById('navLinks').classList.toggle('open')" aria-label="Menu">
        <span></span><span></span><span></span>
      </button>
    </div>
  </nav>"""


def _footer(root: str = "") -> str:
    return f"""
  <footer class="footer">
    <img src="{root}assets/logo.png" alt="StockGems"/>
    <p>Updated {_generated_at_ny_str()} &nbsp;·&nbsp; Stage 2 breakout scanner &nbsp;·&nbsp; Not financial advice</p>
  </footer>"""


# ─────────────────────────────────────────────
#  Formatters
# ─────────────────────────────────────────────

def _fmt_price(x) -> str:
    try: return f"{float(x):,.2f}"
    except: return str(x)

def _fmt_pct(x) -> str:
    try: return f"{float(x):.2f}%"
    except: return str(x)

def _fmt_vol(x) -> str:
    try: x = float(x)
    except: return str(x)
    if x >= 1_000_000_000: return f"{x/1_000_000_000:.2f}B"
    if x >= 1_000_000:     return f"{x/1_000_000:.2f}M"
    if x >= 1_000:         return f"{x/1_000:.0f}K"
    return f"{x:.0f}"

def _ret_val(txt) -> float | None:
    try: return float(str(txt).replace("%", "").replace("+", "").replace("—", "").strip())
    except: return None


# ─────────────────────────────────────────────
#  Today table
# ─────────────────────────────────────────────

def _today_table_html(today_df: pd.DataFrame) -> str:
    if today_df is None or today_df.empty:
        return """<div class="empty">
          <div class="empty-icon">🔍</div>
          <div class="empty-title">No picks today</div>
          <div class="empty-sub">No breakouts met the criteria. Check back after next market close.</div>
        </div>"""

    rows = ""
    for _, r in today_df.head(500).iterrows():
        sym   = str(r.get("symbol", ""))
        close = _fmt_price(r.get("close"))
        vol   = _fmt_vol(r.get("volume"))
        vol50 = _fmt_vol(r.get("vol50"))
        ext   = _fmt_pct(r.get("extended_pct_vs_50sma"))
        pivot = _fmt_pct(r.get("pivot_distance_pct"))
        rows += f"""<tr>
          <td class="sym-cell">{sym}</td>
          <td>{close}</td><td>{vol}</td><td>{vol50}</td><td>{ext}</td><td>{pivot}</td>
        </tr>"""

    return f"""<div class="table-wrap">
      <table id="todayTable" class="sg-table" style="width:100%">
        <thead><tr>
          <th>Symbol</th><th>Close</th><th>Volume</th>
          <th>Avg Vol (50D)</th><th>Ext vs 50D</th><th>From Pivot</th>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""


# ─────────────────────────────────────────────
#  History table
# ─────────────────────────────────────────────

def _history_table_html(df_perf: pd.DataFrame) -> str:
    if df_perf is None or df_perf.empty:
        return """<div class="empty">
          <div class="empty-icon">📋</div>
          <div class="empty-title">No history yet</div>
          <div class="empty-sub">Once picks are recorded, forward performance will appear here automatically.</div>
        </div>"""

    dfh = df_perf.copy()
    dt = pd.to_datetime(dfh["scan_date"], errors="coerce")
    dfh["scan_date"] = dt.apply(lambda d: f"{d.strftime('%b')} {d.day}" if pd.notna(d) else "")
    dfh = dfh.rename(columns={
        "scan_date": "Date", "days_since_scan": "Days",
        "symbol": "Symbol", "entry_close": "Entry", "Now": "Now",
        **{f"{n}d": f"{n}D" for n in HORIZONS},
    })

    ret_cols = {"Now"} | {f"{n}D" for n in HORIZONS}
    header = "".join(f"<th>{c}</th>" for c in dfh.columns)
    rows = ""
    for _, r in dfh.head(5000).iterrows():
        cells = ""
        for col in dfh.columns:
            val = str(r[col]) if pd.notna(r[col]) else "—"
            if col in ret_cols:
                rv = _ret_val(val)
                is_now = col == "Now"
                css = "ret-now " if is_now else ""
                if rv is not None and rv > 0:   css += "ret-pos"
                elif rv is not None and rv < 0: css += "ret-neg"
                css = css.strip()
                cells += f'<td class="{css}">{val}</td>' if css else f"<td>{val}</td>"
            elif col == "Symbol":
                cells += f'<td class="sym-cell">{val}</td>'
            else:
                cells += f"<td>{val}</td>"
        rows += f"<tr>{cells}</tr>"

    return f"""<div class="table-wrap">
      <table id="histTable" class="sg-table" style="width:100%">
        <thead><tr>{header}</tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""


# ─────────────────────────────────────────────
#  Chart data
# ─────────────────────────────────────────────

def _chart_json(df_perf: pd.DataFrame) -> str:
    if df_perf is None or df_perf.empty:
        return "null"
    labels = ["Now"] + [f"{n}D" for n in HORIZONS]
    avgs = []
    for col in labels:
        if col not in df_perf.columns:
            avgs.append(None)
            continue
        vals = df_perf[col].apply(_ret_val).dropna()
        avgs.append(round(float(vals.mean()), 2) if not vals.empty else None)
    return json.dumps({"labels": labels, "avgs": avgs})


# ─────────────────────────────────────────────
#  Main page
# ─────────────────────────────────────────────

def _write_main(today_df: pd.DataFrame, out_dir: Path) -> None:
    n = len(today_df) if today_df is not None and not today_df.empty else 0
    tbl = _today_table_html(today_df)

    html = f"""<!doctype html>
<html lang="en">
<head><title>StockGems · Stage 2 Scanner</title>{SHARED_HEAD}</head>
<body>
{_nav(root="", active="today")}
<div class="page-wrap">

  <section class="hero fade-up">
    <div>
      <div class="hero-eyebrow">Stage 2 Breakout Scanner</div>
      <h1 class="hero-title">Proof,<br/><em>not promises.</em></h1>
      <p class="hero-sub">Every pick is timestamped and tracked forward in trading days.
        A fully public performance ledger — updated automatically after each close.</p>
      <div class="hero-meta"><span class="pulse"></span>Updated {_generated_at_ny_str()}</div>
      <div class="hero-actions">
        <a class="btn btn-primary" href="#picks">Today's Picks</a>
        <a class="btn btn-outline" href="history/index.html">Performance Ledger</a>
        <a class="btn btn-outline" href="stage2_candidates.csv">↓ CSV</a>
      </div>
      <div class="chips">
        <span class="chip">After-close updates</span>
        <span class="chip">Forward returns tracked</span>
        <span class="chip">Full history ledger</span>
        <span class="chip">Open data</span>
      </div>
    </div>
    <div class="hero-stats delay-1 fade-up">
      <div class="stat-card">
        <div class="stat-label">Picks Today</div>
        <div class="stat-value">{n}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Scan Type</div>
        <div class="stat-value" style="font-size:16px;">Stage 2</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Horizons</div>
        <div class="stat-value" style="font-size:15px;">15–200D</div>
      </div>
    </div>
  </section>

  <div class="divider"></div>

  <section id="picks" class="fade-up delay-2">
    <div class="section-header">
      <div>
        <div class="section-title">Today's Picks</div>
        <div class="section-sub">Sorted by distance from pivot — tightest setups first</div>
      </div>
    </div>
    <div class="card">{tbl}</div>
  </section>

  <div class="divider"></div>

  <section class="two-col fade-up delay-3">
    <div class="card">
      <div class="card-header">
        <h2>What you're seeing</h2>
        <p>Column definitions for today's scan output</p>
      </div>
      <div class="card-body">
        <div class="explainer-grid">
          <div class="explainer-item">
            <strong>From Pivot</strong>
            <span>% above the handle-derived pivot. Closer to 0% = tighter, lower-risk entry.</span>
          </div>
          <div class="explainer-item">
            <strong>Ext vs 50D</strong>
            <span>% above the 50-day moving average. Filters out over-extended breakouts (&gt;25% fails).</span>
          </div>
          <div class="explainer-item">
            <strong>Volume</strong>
            <span>Today's volume. Must be ≥ 1.4× the 50D average to qualify as a breakout.</span>
          </div>
          <div class="explainer-item">
            <strong>Avg Vol (50D)</strong>
            <span>Rolling 50-day volume baseline used for the volume expansion check.</span>
          </div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <h2>The methodology</h2>
        <p>Minervini-style Stage 2 base + breakout detection</p>
      </div>
      <div class="card-body">
        <div class="explainer-grid">
          <div class="explainer-item">
            <strong>Trend context</strong>
            <span>50 SMA &gt; 150 SMA &gt; 200 SMA, rising 200, price above 50 &amp; 150.</span>
          </div>
          <div class="explainer-item">
            <strong>Base quality</strong>
            <span>40-day base ≤ 30% depth. Volatility and volume contract into the right side.</span>
          </div>
          <div class="explainer-item">
            <strong>Pivot</strong>
            <span>Max HIGH of the final 10 days of the base defines the handle pivot point.</span>
          </div>
          <div class="explainer-item">
            <strong>Breakout day</strong>
            <span>Close &gt; pivot on volume ≥ 1.4× 50D. Not extended &gt; 25% above 50 SMA.</span>
          </div>
        </div>
      </div>
    </div>
  </section>

  {_footer(root="")}
</div>

<script>
  window.addEventListener("load", function () {{
    if (window.jQuery && $.fn.DataTable && document.getElementById("todayTable")) {{
      $('#todayTable').DataTable({{ pageLength: 25, order: [] }});
    }}
  }});
</script>
</body>
</html>"""

    (out_dir / "index.html").write_text(html, encoding="utf-8")


# ─────────────────────────────────────────────
#  History page
# ─────────────────────────────────────────────

def _write_history(df_perf: pd.DataFrame, summary: dict, out_dir: Path) -> None:
    hist_dir = out_dir / "history"
    hist_dir.mkdir(parents=True, exist_ok=True)

    tbl          = _history_table_html(df_perf)
    chart_data   = _chart_json(df_perf)
    total        = summary.get("count", 0)
    win_rate     = summary.get("win_rate")
    avg_ret      = summary.get("avg")

    wr_str  = f"{win_rate:.1f}%" if win_rate is not None else "—"
    avg_str = (f"+{avg_ret:.1f}%" if avg_ret >= 0 else f"{avg_ret:.1f}%") if avg_ret is not None else "—"
    wr_cls  = "pos" if (win_rate is not None and win_rate >= 50) else ""
    avg_cls = "pos" if (avg_ret is not None and avg_ret > 0) else ("neg" if (avg_ret is not None and avg_ret < 0) else "")

    html = f"""<!doctype html>
<html lang="en">
<head><title>StockGems · Performance Ledger</title>{SHARED_HEAD}</head>
<body>
{_nav(root="../", active="history")}
<div class="page-wrap">

  <section class="hero fade-up">
    <div>
      <div class="hero-eyebrow">Performance Ledger</div>
      <h1 class="hero-title">Every pick.<br/><em>Every result.</em></h1>
      <p class="hero-sub">Timestamped picks tracked forward in trading days.
        No cherry-picking — the complete public record, automatically updated.</p>
      <div class="hero-meta"><span class="pulse"></span>Updated {_generated_at_ny_str()}</div>
      <div class="hero-actions">
        <a class="btn btn-primary" href="#ledger">Browse Ledger</a>
        <a class="btn btn-outline" href="picks.csv">↓ Download CSV</a>
        <a class="btn btn-outline" href="../index.html">← Today's Picks</a>
      </div>
    </div>
    <div class="hero-stats delay-1 fade-up">
      <div class="stat-card">
        <div class="stat-label">Total Picks</div>
        <div class="stat-value">{total}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Win Rate</div>
        <div class="stat-value {wr_cls}">{wr_str}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Avg Return</div>
        <div class="stat-value {avg_cls}">{avg_str}</div>
      </div>
    </div>
  </section>

  <div class="divider"></div>

  <section class="fade-up delay-2">
    <div class="section-header">
      <div>
        <div class="section-title">Average Return by Horizon</div>
        <div class="section-sub">All tracked picks · Entry at signal-day close</div>
      </div>
    </div>
    <div class="card">
      <div class="chart-wrap">
        <canvas id="returnChart"></canvas>
      </div>
    </div>
  </section>

  <div class="divider"></div>

  <section id="ledger" class="fade-up delay-3">
    <div class="section-header">
      <div>
        <div class="section-title">Full Ledger</div>
        <div class="section-sub">Filter by date · Green = positive return · Red = negative</div>
      </div>
    </div>
    <div class="card">
      <div class="filter-row">
        <label for="dateFilter">Date:</label>
        <select id="dateFilter"><option value="">All dates</option></select>
      </div>
      {tbl}
    </div>
  </section>

  {_footer(root="../")}
</div>

<script>
const chartData = {chart_data};

window.addEventListener("load", function () {{

  // Chart
  if (chartData && window.Chart) {{
    const ctx = document.getElementById("returnChart");
    if (ctx) {{
      const colors = chartData.avgs.map(v =>
        v === null ? '#e5e7eb' : v >= 0 ? 'rgba(22,128,60,.72)' : 'rgba(185,28,28,.68)'
      );
      new Chart(ctx, {{
        type: 'bar',
        data: {{
          labels: chartData.labels,
          datasets: [{{ label: 'Avg Return %', data: chartData.avgs, backgroundColor: colors, borderRadius: 7, borderSkipped: false }}]
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{
            legend: {{ display: false }},
            tooltip: {{ callbacks: {{ label: c => c.raw === null ? 'No data' : (c.raw >= 0 ? '+' : '') + c.raw.toFixed(1) + '%' }} }}
          }},
          scales: {{
            y: {{
              grid: {{ color: 'rgba(27,43,94,.06)' }},
              ticks: {{ callback: v => (v >= 0 ? '+' : '') + v + '%', font: {{ size: 12 }}, color: '#9ca3af' }}
            }},
            x: {{ grid: {{ display: false }}, ticks: {{ font: {{ size: 12, weight: '600' }}, color: '#6b7280' }} }}
          }}
        }}
      }});
    }}
  }}

  // DataTable
  if (window.jQuery && $.fn.DataTable && document.getElementById("histTable")) {{
    const table = $('#histTable').DataTable({{ pageLength: 50, order: [[0, 'desc']] }});
    const seen = new Set();
    table.column(0).data().each(d => {{ if (d) seen.add(String(d).trim()); }});
    Array.from(seen).sort().reverse().forEach(d => {{
      const o = document.createElement("option");
      o.value = d; o.textContent = d;
      document.getElementById("dateFilter").appendChild(o);
    }});
    document.getElementById("dateFilter").addEventListener("change", function () {{
      const v = this.value;
      table.column(0).search(v ? "^" + v + "$" : "", v ? true : false, false).draw();
    }});
  }}

}});
</script>
</body>
</html>"""

    (hist_dir / "index.html").write_text(html, encoding="utf-8")


# ─────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────

def write_site(today_df: pd.DataFrame) -> None:
    out_dir = Path("docs")
    out_dir.mkdir(parents=True, exist_ok=True)

    if today_df is None:
        today_df = pd.DataFrame()

    today_df.to_csv(out_dir / "stage2_candidates.csv", index=False)

    df_perf, summary = update_history_and_build_perf_table(today_df, out_dir)

    _write_main(today_df, out_dir)
    _write_history(df_perf, summary, out_dir)

    print("Site written → docs/index.html · docs/history/index.html")