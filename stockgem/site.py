from pathlib import Path
import pandas as pd

from .config import HORIZONS
from .time_utils import _generated_at_ny_str, _now_ny
from .history import update_history_and_build_perf_table


def write_site(today_df: pd.DataFrame) -> None:
    out_dir = Path("docs")
    out_dir.mkdir(parents=True, exist_ok=True)

    if today_df is None:
        today_df = pd.DataFrame()

    # Save today's CSV (raw)
    today_df.to_csv(out_dir / "stage2_candidates.csv", index=False)

    # Build performance table (history page)
    df_perf, summary = update_history_and_build_perf_table(today_df, out_dir)

    # -----------------------------
    # MAIN PAGE: simplify + format
    # -----------------------------
    today_view = today_df.copy()

    if not today_view.empty:
        # Keep only selected columns
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

        # ✅ Scan Date as "Feb 4" (no year)
        dt = pd.to_datetime(dfh["scan_date"], errors="coerce")
        dfh["scan_date"] = dt.apply(lambda d: f"{d.strftime('%b')} {d.day}" if pd.notna(d) else "")

        dfh = dfh.rename(columns={
            "scan_date": "Scan Date",
            "days_since_scan": "Days Since",
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

    .hero-seal{
      display:block;
      margin: 0 auto 14px auto;
      width: min(320px, 70%);
      height: auto;
      opacity: 0.92;
      filter: drop-shadow(0 10px 28px rgba(2,6,23,.18));
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
      background: rgba(248,250c,252,.9) !important;
      color: var(--muted) !important;
      font-weight:800 !important;
      border-bottom: 1px solid var(--border) !important;
      padding: 12px 12px !important;
    }
    #todayTable tbody td{
      padding: 12px 12px !important;
      border-bottom: 1px solid rgba(15,23,42,.06) !important;
      color: var(--text) !important;
      font-weight:600;
    }
    #histTable tbody td{
      padding: 12px 12px !important;
      border-bottom: 1px solid rgba(15,23,42,.06) !important;
      color: var(--text);
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

    shared_js = r"""
  <script>
    function isPercentLike(txt){
      if(!txt) return false;
      const s = String(txt).trim();
      if(s === "—") return false;
      return s.endsWith("%") && !isNaN(Number(s.replace("%","").replace("+","")));
    }
  </script>
    """

    # History page HTML (unchanged concept)
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
        <a href="2025/">2025 Results</a>
        <a href="picks.csv">Download CSV</a>
      </div>
      <a class="btn primary" href="../index.html">View today</a>
    </div>

    <div class="hero">
      <img src="../assets/logo.png" alt="StockGems" class="hero-seal">
      <h1>Performance Ledger</h1>
      <p>
        Every pick is timestamped and tracked forward in trading days. This page is the proof archive.
      </p>
      <div class="meta">Updated: <b>{_generated_at_ny_str()}</b></div>
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

          // Color ONLY the "Now" column
          function colorizeNowOnly() {{
            let nowIdx = null;
            $("#histTable thead th").each(function (i) {{
              const t = $(this).text().trim().toLowerCase();
              if (t === "now") nowIdx = i;
            }});
            if (nowIdx === null) return;

            $("#histTable tbody tr").each(function () {{
              const tds = $(this).find("td");
              const cell = $(tds[nowIdx]);
              if (!cell || cell.length === 0) return;

              const txt = cell.text().trim();
              cell.css("font-weight", "800");

              if (!txt || txt === "—") {{
                cell.css("color", "var(--muted)");
                return;
              }}

              const n = Number(txt.replace("%", "").replace("+", ""));
              if (!Number.isFinite(n)) return;

              if (n > 0) cell.css("color", "var(--pos)");
              else if (n < 0) cell.css("color", "var(--neg)");
              else cell.css("color", "var(--muted)");
            }});
          }}

          colorizeNowOnly();
          table.on("draw", function () {{ colorizeNowOnly(); }});
        }}
      }} catch (e) {{
        console.warn("History init failed.", e);
      }}
    }});
  </script>
</body>
</html>
"""
    (Path("docs") / "history" / "index.html").write_text(history_html, encoding="utf-8")

    # Main page HTML (unchanged concept)
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
        <a href="history/2025/">2025 Results</a>
        <a href="stage2_candidates.csv">Download</a>
      </div>
      <a class="btn primary" href="history/index.html">View ledger</a>
    </div>

    <div class="hero">
      <img src="assets/logo.png" alt="StockGems" class="hero-seal">
      <h1>Proof, not promises.</h1>
      <p>
        A beautiful, public performance ledger that tracks breakout picks forward in trading days.
        Every scan is timestamped. Every result updates automatically.
      </p>
      <div class="meta">Last updated: <b>{_generated_at_ny_str()}</b></div>

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
    (Path("docs") / "index.html").write_text(main_html, encoding="utf-8")
