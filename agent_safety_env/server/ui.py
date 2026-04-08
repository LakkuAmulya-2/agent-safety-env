# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-3-Clause License
"""
Agent Safety Environment — Web Dashboard UI.
Single-page HTML dashboard served at /ui.
Shows all 8 layers in real-time.
"""

UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AgentSafetyEnv Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #f7f8fa; color: #1a202c; }
  .header { background: linear-gradient(135deg, #1e40af, #3b82f6); padding: 20px 30px;
            border-bottom: 2px solid #3b82f6; display: flex; align-items: center; gap: 16px; }
  .header h1 { font-size: 1.4rem; color: #ffffff; }
  .header .badge { background: #48bb78; color: #000; padding: 3px 10px;
                   border-radius: 12px; font-size: 0.75rem; font-weight: bold; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
          gap: 16px; padding: 20px; }
  .card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px;
          padding: 16px; }
  .card h2 { font-size: 0.85rem; color: #64748b; text-transform: uppercase;
             letter-spacing: 0.05em; margin-bottom: 12px; }
  .metric { display: flex; justify-content: space-between; align-items: center;
            padding: 6px 0; border-bottom: 1px solid #e2e8f0; }
  .metric:last-child { border-bottom: none; }
  .metric .label { color: #64748b; font-size: 0.85rem; }
  .metric .value { font-weight: bold; font-size: 0.9rem; }
  .green { color: #48bb78; }
  .red { color: #fc8181; }
  .yellow { color: #f6e05e; }
  .blue { color: #63b3ed; }
  .score-bar { height: 8px; background: #f1f5f9; border-radius: 4px; margin-top: 8px; }
  .score-fill { height: 100%; border-radius: 4px; transition: width 0.5s; }
  .task-row { padding: 8px; background: #f1f5f9; border-radius: 6px; margin-bottom: 6px; }
  .task-row .task-name { font-size: 0.8rem; color: #64748b; }
  .task-row .task-score { font-size: 1.1rem; font-weight: bold; }
  .chaos-item { display: flex; align-items: center; gap: 8px; padding: 5px 0;
                font-size: 0.82rem; border-bottom: 1px solid #e2e8f0; }
  .chaos-item:last-child { border-bottom: none; }
  .dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  .dot.pass { background: #48bb78; }
  .dot.fail { background: #fc8181; }
  .cert-badge { text-align: center; padding: 16px; }
  .cert-badge .badge-icon { font-size: 3rem; }
  .cert-badge .badge-label { font-size: 1.2rem; font-weight: bold; margin-top: 8px; }
  .cert-badge .cert-id { font-size: 0.7rem; color: #94a3b8; margin-top: 4px; }
  .event-item { font-size: 0.78rem; padding: 5px 0; border-bottom: 1px solid #e2e8f0;
                display: flex; gap: 8px; }
  .event-item:last-child { border-bottom: none; }
  .event-type { color: #63b3ed; font-weight: bold; min-width: 120px; }
  .refresh-btn { background: #3b82f6; color: #ffffff; border: none; padding: 8px 16px;
                 border-radius: 6px; cursor: pointer; font-size: 0.85rem; }
  .refresh-btn:hover { background: #2563eb; }
  .loading { color: #94a3b8; font-size: 0.85rem; }
  .endpoint-list { font-size: 0.78rem; }
  .endpoint-list a { color: #63b3ed; text-decoration: none; display: block;
                     padding: 3px 0; border-bottom: 1px solid #e2e8f0; }
  .endpoint-list a:hover { color: #90cdf4; }
  .endpoint-list a:last-child { border-bottom: none; }
  .layer-tag { font-size: 0.65rem; background: #dbeafe; color: #64748b;
               padding: 1px 6px; border-radius: 8px; margin-left: 6px; }
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>AgentSafetyEnv <span class="badge">LIVE</span></h1>
    <div style="font-size:0.75rem;color:#718096;margin-top:4px;">
      Real-world RL environment for AI agent safety training
    </div>
  </div>
  <div style="margin-left:auto;display:flex;gap:8px;align-items:center;">
    <button class="refresh-btn" onclick="loadAll()">Refresh</button>
    <span id="last-updated" class="loading">Loading...</span>
  </div>
</div>

<div class="grid">

  <!-- Safety Metrics -->
  <div class="card">
    <h2>Safety Metrics <span class="layer-tag">Layer 1</span></h2>
    <div id="metrics-content"><div class="loading">Loading...</div></div>
  </div>

  <!-- Task Performance -->
  <div class="card">
    <h2>Task Performance <span class="layer-tag">3 Tasks</span></h2>
    <div id="tasks-content"><div class="loading">Loading...</div></div>
  </div>

  <!-- Compliance -->
  <div class="card">
    <h2>Compliance <span class="layer-tag">Layer 3</span></h2>
    <div id="compliance-content"><div class="loading">Loading...</div></div>
  </div>

  <!-- Observability -->
  <div class="card">
    <h2>Observability <span class="layer-tag">Layer 4</span></h2>
    <div id="observe-content"><div class="loading">Loading...</div></div>
  </div>

  <!-- Chaos Tests -->
  <div class="card">
    <h2>Chaos Tests <span class="layer-tag">Layer 7</span></h2>
    <div id="chaos-content"><div class="loading">Loading...</div></div>
  </div>

  <!-- Trust & Certification -->
  <div class="card">
    <h2>Trust & Certification <span class="layer-tag">Layer 8</span></h2>
    <div id="trust-content"><div class="loading">Loading...</div></div>
  </div>

  <!-- Deployment -->
  <div class="card">
    <h2>Deployment <span class="layer-tag">Layer 6</span></h2>
    <div id="deploy-content"><div class="loading">Loading...</div></div>
  </div>

  <!-- Live Events -->
  <div class="card">
    <h2>Live Events <span class="layer-tag">Real-time</span></h2>
    <div id="events-content"><div class="loading">Loading...</div></div>
  </div>

  <!-- API Endpoints -->
  <div class="card">
    <h2>API Endpoints</h2>
    <div class="endpoint-list">
      <a href="/health" target="_blank">/health — Health check</a>
      <a href="/docs" target="_blank">/docs — Swagger UI</a>
      <a href="/metrics" target="_blank">/metrics — Safety metrics</a>
      <a href="/audit" target="_blank">/audit — Episode audit log</a>
      <a href="/compliance" target="_blank">/compliance — Compliance report</a>
      <a href="/compliance/audit" target="_blank">/compliance/audit — Audit records</a>
      <a href="/observe" target="_blank">/observe — Observability report</a>
      <a href="/observe/traces" target="_blank">/observe/traces — Decision traces</a>
      <a href="/guardrails/config" target="_blank">/guardrails/config — Guardrails</a>
      <a href="/deploy/ab-test" target="_blank">/deploy/ab-test — A/B testing</a>
      <a href="/deploy/canary" target="_blank">/deploy/canary — Canary status</a>
      <a href="/deploy/costs" target="_blank">/deploy/costs — Cost monitoring</a>
      <a href="/chaos/run" target="_blank">/chaos/run — Chaos tests</a>
      <a href="/trust/score" target="_blank">/trust/score — Readiness score</a>
      <a href="/trust/certificate" target="_blank">/trust/certificate — Certificate</a>
      <a href="/stream" target="_blank">/stream — SSE event stream</a>
    </div>
  </div>

</div>

<script>
async function fetchJSON(url) {
  try {
    const r = await fetch(url);
    return await r.json();
  } catch(e) {
    return null;
  }
}

function scoreColor(s) {
  if (s >= 0.8) return 'green';
  if (s >= 0.6) return 'yellow';
  return 'red';
}

function barColor(s) {
  if (s >= 0.8) return '#48bb78';
  if (s >= 0.6) return '#f6e05e';
  return '#fc8181';
}

async function loadMetrics() {
  const d = await fetchJSON('/metrics');
  if (!d) { document.getElementById('metrics-content').innerHTML = '<div class="loading">Unavailable</div>'; return; }
  const total = d.total_episodes || 0;
  const pr = d.pass_rate || 0;
  const avg = d.avg_score || 0;
  document.getElementById('metrics-content').innerHTML = `
    <div class="metric"><span class="label">Total Episodes</span><span class="value blue">${total}</span></div>
    <div class="metric"><span class="label">Pass Rate</span><span class="value ${scoreColor(pr)}">${(pr*100).toFixed(1)}%</span></div>
    <div class="metric"><span class="label">Avg Score</span><span class="value ${scoreColor(avg)}">${avg.toFixed(3)}</span></div>
    <div class="metric"><span class="label">Anomalies</span><span class="value ${d.anomaly_rate > 0.05 ? 'red' : 'green'}">${d.anomalies_detected || 0}</span></div>
    <div class="metric"><span class="label">Circuit Breaker</span><span class="value ${d.circuit_breaker_state === 'CLOSED' ? 'green' : 'red'}">${d.circuit_breaker_state || 'CLOSED'}</span></div>
    <div class="score-bar"><div class="score-fill" style="width:${pr*100}%;background:${barColor(pr)}"></div></div>
  `;
}

async function loadTasks() {
  const d = await fetchJSON('/metrics');
  if (!d || !d.task_breakdown) { document.getElementById('tasks-content').innerHTML = '<div class="loading">No data yet</div>'; return; }
  const tasks = {
    'task1_prompt_injection': 'T1: Prompt Injection (Easy)',
    'task2_data_leakage': 'T2: Data Leakage (Medium)',
    'task3_multi_vector': 'T3: Multi-Vector (Hard)',
  };
  let html = '';
  for (const [tid, label] of Object.entries(tasks)) {
    const t = d.task_breakdown[tid];
    if (!t) { html += `<div class="task-row"><div class="task-name">${label}</div><div class="task-score loading">No data</div></div>`; continue; }
    html += `<div class="task-row">
      <div class="task-name">${label}</div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-top:4px;">
        <span class="task-score ${scoreColor(t.avg_score)}">${t.avg_score.toFixed(3)}</span>
        <span style="font-size:0.75rem;color:#a0aec0">${t.passed}/${t.total} passed</span>
      </div>
      <div class="score-bar" style="margin-top:6px"><div class="score-fill" style="width:${t.pass_rate*100}%;background:${barColor(t.pass_rate)}"></div></div>
    </div>`;
  }
  document.getElementById('tasks-content').innerHTML = html || '<div class="loading">No task data yet</div>';
}

async function loadCompliance() {
  const d = await fetchJSON('/compliance');
  if (!d) { document.getElementById('compliance-content').innerHTML = '<div class="loading">Unavailable</div>'; return; }
  const sev = d.severity_breakdown || {};
  document.getElementById('compliance-content').innerHTML = `
    <div class="metric"><span class="label">Frameworks</span><span class="value blue">${(d.frameworks_covered||[]).length}</span></div>
    <div class="metric"><span class="label">Violation Rate</span><span class="value ${(d.violation_rate||0) > 0.1 ? 'red' : 'green'}">${((d.violation_rate||0)*100).toFixed(1)}%</span></div>
    <div class="metric"><span class="label">CRITICAL</span><span class="value ${sev.CRITICAL > 0 ? 'red' : 'green'}">${sev.CRITICAL || 0}</span></div>
    <div class="metric"><span class="label">HIGH</span><span class="value ${sev.HIGH > 0 ? 'yellow' : 'green'}">${sev.HIGH || 0}</span></div>
    <div class="metric"><span class="label">Retention</span><span class="value blue">${d.retention_policy_days || 90} days</span></div>
    <div style="margin-top:8px;font-size:0.75rem;color:#718096">${(d.frameworks_covered||[]).join(' · ')}</div>
  `;
}

async function loadObserve() {
  const d = await fetchJSON('/observe');
  if (!d) { document.getElementById('observe-content').innerHTML = '<div class="loading">Unavailable</div>'; return; }
  const lat = d.latency || {};
  const tok = d.tokens || {};
  const total_ms = lat.total_ms || {};
  document.getElementById('observe-content').innerHTML = `
    <div class="metric"><span class="label">Latency p50</span><span class="value blue">${total_ms.p50 || 0}ms</span></div>
    <div class="metric"><span class="label">Latency p95</span><span class="value ${(total_ms.p95||0) > 2000 ? 'yellow' : 'blue'}">${total_ms.p95 || 0}ms</span></div>
    <div class="metric"><span class="label">Total Tokens</span><span class="value blue">${tok.total_tokens || 0}</span></div>
    <div class="metric"><span class="label">Est. Cost</span><span class="value green">$${tok.estimated_cost_usd || 0}</span></div>
    <div class="metric"><span class="label">Decision Traces</span><span class="value blue">${d.decision_traces_count || 0}</span></div>
  `;
}

async function loadChaos() {
  const d = await fetchJSON('/chaos/run');
  if (!d) { document.getElementById('chaos-content').innerHTML = '<div class="loading">Unavailable</div>'; return; }
  let html = `<div class="metric" style="margin-bottom:8px"><span class="label">Results</span><span class="value ${d.failed === 0 ? 'green' : 'red'}">${d.passed}/${d.total} passed</span></div>`;
  for (const r of (d.results || [])) {
    html += `<div class="chaos-item"><div class="dot ${r.passed ? 'pass' : 'fail'}"></div><span>${r.test}</span><span style="margin-left:auto;color:#718096;font-size:0.75rem">${r.recovery_ms}ms</span></div>`;
  }
  document.getElementById('chaos-content').innerHTML = html;
}

async function loadTrust() {
  const [score, cert] = await Promise.all([fetchJSON('/trust/score'), fetchJSON('/trust/certificate')]);
  if (!score || !cert) { document.getElementById('trust-content').innerHTML = '<div class="loading">Unavailable</div>'; return; }
  const badgeEmoji = {'GOLD': '🥇', 'SILVER': '🥈', 'BRONZE': '🥉', 'NONE': '❌'}[cert.badge] || '❓';
  const gradeColor = {'A': 'green', 'B': 'green', 'C': 'yellow', 'D': 'yellow', 'F': 'red'}[score.grade] || 'blue';
  document.getElementById('trust-content').innerHTML = `
    <div class="cert-badge">
      <div class="badge-icon">${badgeEmoji}</div>
      <div class="badge-label ${gradeColor}">${cert.badge} CERTIFIED</div>
      <div class="cert-id">ID: ${cert.certificate_id}</div>
    </div>
    <div class="metric"><span class="label">Score</span><span class="value ${gradeColor}">${score.score}/100 (${score.grade})</span></div>
    <div class="metric"><span class="label">Production Ready</span><span class="value ${score.ready_for_production ? 'green' : 'red'}">${score.ready_for_production ? 'YES' : 'NO'}</span></div>
    ${(score.blockers||[]).map(b => `<div style="font-size:0.75rem;color:#fc8181;margin-top:4px">⚠ ${b}</div>`).join('')}
    <div class="score-bar" style="margin-top:8px"><div class="score-fill" style="width:${score.score}%;background:${barColor(score.score/100)}"></div></div>
  `;
}

async function loadDeploy() {
  const [canary, costs, ab] = await Promise.all([
    fetchJSON('/deploy/canary'), fetchJSON('/deploy/costs'), fetchJSON('/deploy/ab-test')
  ]);
  let html = '';
  if (canary) {
    html += `<div class="metric"><span class="label">Canary Traffic</span><span class="value blue">${(canary.current_traffic_pct*100).toFixed(0)}%</span></div>`;
    html += `<div class="metric"><span class="label">Stage</span><span class="value blue">${canary.stage}/${canary.total_stages}</span></div>`;
    html += `<div class="metric"><span class="label">Rolled Back</span><span class="value ${canary.rolled_back ? 'red' : 'green'}">${canary.rolled_back ? 'YES' : 'NO'}</span></div>`;
  }
  if (costs) {
    html += `<div class="metric"><span class="label">Total Cost</span><span class="value green">$${costs.total_cost_usd || 0}</span></div>`;
    html += `<div class="metric"><span class="label">Total Tokens</span><span class="value blue">${costs.total_tokens || 0}</span></div>`;
  }
  document.getElementById('deploy-content').innerHTML = html || '<div class="loading">No data</div>';
}

async function loadEvents() {
  const d = await fetchJSON('/audit?limit=5');
  if (!d || !d.traces) { document.getElementById('events-content').innerHTML = '<div class="loading">No events yet</div>'; return; }
  let html = '';
  for (const t of d.traces.slice(-5).reverse()) {
    const sc = t.final_score || 0;
    html += `<div class="event-item">
      <span class="event-type">${t.task_id.replace('task','T').replace('_prompt_injection','1').replace('_data_leakage','2').replace('_multi_vector','3')}</span>
      <span class="${scoreColor(sc)}">${sc.toFixed(2)}</span>
      <span style="color:#718096;font-size:0.75rem">${t.passed ? 'PASS' : 'FAIL'}</span>
      <span style="color:#718096;font-size:0.7rem;margin-left:auto">${t.duration_ms}ms</span>
    </div>`;
  }
  document.getElementById('events-content').innerHTML = html || '<div class="loading">No episodes yet</div>';
}

async function loadAll() {
  document.getElementById('last-updated').textContent = 'Refreshing...';
  await Promise.all([
    loadMetrics(), loadTasks(), loadCompliance(), loadObserve(),
    loadChaos(), loadTrust(), loadDeploy(), loadEvents()
  ]);
  document.getElementById('last-updated').textContent = 'Updated: ' + new Date().toLocaleTimeString();
}

// Load on start, auto-refresh every 30s
loadAll();
setInterval(loadAll, 30000);
</script>
</body>
</html>"""
