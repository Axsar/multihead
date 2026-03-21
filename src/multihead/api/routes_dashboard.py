"""Simple HTML dashboard for run monitoring."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()

DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MultiHead Dashboard</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: system-ui, sans-serif; background: #0f1117; color: #e0e0e0; padding: 2rem; }
  h1 { margin-bottom: 1.5rem; font-size: 1.5rem; color: #7eb8ff; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; }
  .card { background: #1a1d27; border-radius: 8px; padding: 1.2rem; border: 1px solid #2a2d37; }
  .card h2 { font-size: 0.9rem; color: #888; text-transform: uppercase; margin-bottom: 0.5rem; }
  .card .value { font-size: 2rem; font-weight: 700; color: #7eb8ff; }
  .card .sub { font-size: 0.8rem; color: #666; margin-top: 0.25rem; }
  .status-ok { color: #4ade80; }
  .status-warn { color: #fbbf24; }
  .status-err { color: #f87171; }
  table { width: 100%; border-collapse: collapse; margin-top: 1.5rem; }
  th, td { text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid #2a2d37; font-size: 0.85rem; }
  th { color: #888; text-transform: uppercase; font-size: 0.75rem; }
  #refresh { float: right; background: #2a2d37; color: #e0e0e0; border: none; padding: 0.4rem 1rem; border-radius: 4px; cursor: pointer; }
  #refresh:hover { background: #3a3d47; }
</style>
</head>
<body>
<h1>MultiHead <button id="refresh" onclick="location.reload()">Refresh</button></h1>
<div class="grid" id="cards"></div>
<table id="runs-table">
  <thead><tr><th>Run ID</th><th>Status</th><th>Events</th></tr></thead>
  <tbody id="runs-body"></tbody>
</table>
<script>
async function load() {
  try {
    const health = await fetch('/health').then(r => r.json());
    const cards = document.getElementById('cards');
    cards.innerHTML = `
      <div class="card"><h2>Status</h2><div class="value status-ok">${health.status}</div><div class="sub">v${health.version}</div></div>
      <div class="card"><h2>Heads</h2><div class="value">${health.heads_loaded ?? '?'}</div><div class="sub">loaded</div></div>
    `;
  } catch(e) {
    document.getElementById('cards').innerHTML = '<div class="card"><h2>Error</h2><div class="value status-err">Offline</div></div>';
  }
}
load();
</script>
</body>
</html>
"""


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    """Serve the monitoring dashboard."""
    return HTMLResponse(content=DASHBOARD_HTML)


@router.get("/metrics")
async def metrics(request: Request, format: str = "json") -> Any:
    """Export metrics in JSON or Prometheus format."""
    collector = getattr(request.app.state, "metrics", None)
    if collector is None:
        return {"counters": {}, "gauges": {}, "histograms": {}}

    if format == "prometheus":
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(collector.to_prometheus(), media_type="text/plain")

    return collector.to_json()
