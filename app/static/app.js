const API = "";

async function get(path) {
  const r = await fetch(API + path);
  if (!r.ok) throw new Error(path + " -> " + r.status);
  return r.json();
}

function fmt(x, d = 4) {
  if (x === null || x === undefined) return "—";
  return Number(x).toLocaleString(undefined, { maximumFractionDigits: d });
}

function pill(v) {
  const cls = v > 0.1 ? "pos" : v < -0.1 ? "neg" : "neu";
  return `<span class="pill ${cls}">${Number(v).toFixed(3)}</span>`;
}

function renderCards(s) {
  const r = s.reward || {};
  const cards = [
    ["Trades evaluated", r.total ?? 0, "accent"],
    ["Positive rewards", r.positive ?? 0, "pos"],
    ["Negative rewards", r.negative ?? 0, "neg"],
    ["Neutral rewards", r.neutral ?? 0, "neu"],
    ["Avg reward", fmt(r.avg_reward, 4), "accent"],
    ["Paper fills", s.paper ?? 0, "muted"],
    ["Live fills", s.live ?? 0, "muted"],
    ["Training datasets", (s.training_sets || []).length, "accent"],
  ];
  document.getElementById("cards").innerHTML = cards
    .map(([l, v, c]) => `<div class="card"><div class="label">${l}</div><div class="value ${c}">${v}</div></div>`)
    .join("");
}

function renderExpectancy(e) {
  const rows = [
    ["Overall", e.overall],
    ["Paper", e.paper],
    ["Live", e.live],
  ];
  document.getElementById("expectancy").innerHTML =
    "<tr><th>Scope</th><th>Trades</th><th>Expectancy</th><th>Total P&amp;L</th></tr>" +
    rows
      .map(([k, v]) => `<tr><td>${k}</td><td>${v.count}</td><td>${fmt(v.expectancy)}</td><td>${fmt(v.total_pnl)}</td></tr>`)
      .join("");
}

function renderErrors(errs) {
  const total = errs.reduce((a, b) => a + b.count, 0) || 1;
  document.getElementById("errorCounts").innerHTML =
    "<tr><th>Avoidable error</th><th>Count</th><th>Share</th></tr>" +
    (errs.length
      ? errs
          .map(
            (e) =>
              `<tr><td>${e.category}</td><td>${e.count}</td><td><div class="bar"><span style="width:${100 * e.count / total}%"></span></div></td></tr>`
          )
          .join("")
      : '<tr><td colspan="3" class="muted">none recorded</td></tr>');
}

function drawChart(history) {
  const canvas = document.getElementById("chart");
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!history.length) {
    ctx.fillStyle = "#8b93b0";
    ctx.fillText("no reward history yet", 12, 24);
    return;
  }
  const vals = history.map((d) => d.avg_reward);
  const max = Math.max(0.01, ...vals.map((v) => Math.abs(v)));
  ctx.strokeStyle = "#2a3150";
  ctx.beginPath();
  ctx.moveTo(0, h / 2);
  ctx.lineTo(w, h / 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.strokeStyle = "#7c8cff";
  history.forEach((d, i) => {
    const x = (i / (history.length - 1 || 1)) * w;
    const y = h / 2 - (d.avg_reward / max) * (h / 2 - 8);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = "#8b93b0";
  ctx.fillText("avg reward per bucket (+/− " + max.toFixed(3) + ")", 10, h - 8);
}

function renderGroup(tableId, rows, label) {
  const el = document.getElementById(tableId);
  if (!rows.length) {
    el.innerHTML = `<tr><td colspan="4" class="muted">none</td></tr>`;
    return;
  }
  el.innerHTML =
    `<tr><th>${label}</th><th>Trades</th><th>Avg reward</th><th>Avg P&amp;L</th></tr>` +
    rows
      .map(
        (r) =>
          `<tr><td>${r.group}</td><td>${r.count}</td><td>${pill(r.avg_reward)}</td><td>${fmt(r.avg_pnl)}</td></tr>`
      )
      .join("");
}

function renderModels(models) {
  document.getElementById("models").innerHTML =
    "<tr><th>Version</th><th>Kind</th><th>Status</th><th>Trained</th></tr>" +
    (models.length
      ? models
          .slice(0, 12)
          .map(
            (m) =>
              `<tr><td>${m.version}</td><td>${m.kind}</td><td>${m.status}</td><td>${new Date(m.trained_at * 1000).toLocaleString()}</td></tr>`
          )
          .join("")
      : '<tr><td colspan="4" class="muted">no models</td></tr>');
}

function renderPromotions(promos) {
  document.getElementById("promotions").innerHTML =
    "<tr><th>Candidate</th><th>Decision</th><th>Reason</th></tr>" +
    (promos.length
      ? promos
          .slice(0, 12)
          .map(
            (p) =>
              `<tr><td>${p.candidate_version}</td><td><span class="pill ${p.decision === "promoted" ? "pos" : p.decision === "rolled_back" ? "warn" : "neu"}">${p.decision}</span></td><td class="small">${p.reason}</td></tr>`
          )
          .join("")
      : '<tr><td colspan="3" class="muted">no promotions</td></tr>');
}

function renderDrift(d) {
  document.getElementById("drift").innerHTML = d.drifted
    ? `<span class="pill warn">drift detected</span> ${d.warnings.join("; ")}`
    : `<span class="pill pos">no drift</span>`;
}

function renderDatasets(sets) {
  document.getElementById("datasets").innerHTML =
    "<tr><th>Dataset</th><th>Samples</th><th>Source</th><th>Created</th></tr>" +
    (sets.length
      ? sets
          .slice(0, 20)
          .map(
            (s) =>
              `<tr><td>${s.version}</td><td>${s.n_samples}</td><td>${s.source}</td><td>${new Date(s.created_at * 1000).toLocaleString()}</td></tr>`
          )
          .join("")
      : '<tr><td colspan="4" class="muted">no datasets</td></tr>');
}

function renderTrades(rows) {
  const el = document.getElementById("trades");
  el.innerHTML =
    "<tr><th>Trade</th><th>Symbol</th><th>Kind</th><th>Regime</th><th>Net P&amp;L</th><th>Reward</th><th>Outcome</th><th></th></tr>" +
    (rows.length
      ? rows
          .map(
            (r) =>
              `<tr><td>${r.trade_id}</td><td>${r.symbol}</td><td>${r.execution_kind}</td><td>${r.market_regime}</td><td>${fmt(r.net_pnl)}</td><td>${pill(r.reward)}</td><td><span class="pill ${r.triggers_safety ? "warn" : "neu"}">${r.category || "—"}</span></td><td><button onclick="openTrade('${r.trade_id}')">inspect</button></td></tr>`
          )
          .join("")
      : '<tr><td colspan="8" class="muted">no experiences recorded</td></tr>');
}

async function openTrade(id) {
  const r = await get("/api/learning/experiences/" + id);
  document.getElementById("mTitle").textContent = "Trade " + r.trade_id + " reward breakdown";
  const p = r.payload;
  const rb = p.reward_breakdown || null;
  let breakdown = "";
  if (rb) {
    breakdown =
      `<div class="kv"><div><span>Risk budget:</span> ${fmt(rb.risk_budget)}</div>` +
      `<div><span>Risk-adjusted return:</span> ${fmt(rb.risk_adjusted_return)}</div>` +
      `<div><span>Total reward:</span> ${fmt(rb.total_reward)}</div>` +
      `<div><span>Clipped reward:</span> ${fmt(rb.clipped_reward)} ${rb.clipped ? "(clipped)" : ""}</div></div>` +
      `<table><tr><th>Component</th><th>Raw</th><th>Weight</th><th>Contribution</th></tr>` +
      Object.entries(rb.raw_components)
        .map(([k, v]) => {
          const w = rb.weights[k] || 0;
          const contrib = (rb.weighted_components || {})[k] || 0;
          return `<tr><td>${k}</td><td>${fmt(v)}</td><td>${fmt(w)}</td><td>${fmt(contrib)}</td></tr>`;
        })
        .join("") +
      "</table>" +
      `<p class="small muted">${rb.explanation || ""}</p>`;
  }
  const body =
    `<div class="kv">` +
    `<div><span>Symbol:</span> ${p.symbol}</div>` +
    `<div><span>Side:</span> ${p.side}</div>` +
    `<div><span>Execution:</span> ${p.execution_kind}</div>` +
    `<div><span>Model:</span> ${p.model_version}</div>` +
    `<div><span>Net P&amp;L:</span> ${fmt(p.net_pnl)}</div>` +
    `<div><span>Gross P&amp;L:</span> ${fmt(p.gross_pnl)}</div>` +
    `<div><span>Costs:</span> ${fmt(p.total_costs)}</div>` +
    `<div><span>Exit:</span> ${p.exit_reason || "—"}</div>` +
    `<div><span>Outcome:</span> ${p.category || "—"}</div>` +
    `<div><span>Risk approved:</span> ${p.risk_manager_approved}</div>` +
    `</div>` +
    `<pre>${JSON.stringify(p.prediction_probabilities || {}, null, 2)}</pre>` +
    breakdown;
  document.getElementById("mBody").innerHTML = body;
  document.getElementById("modal").classList.add("open");
}

function closeModal() {
  document.getElementById("modal").classList.remove("open");
}

async function load() {
  try {
    const [summary, expect, errors, history, bySymbol, byRegime, models, promos, drift, datasets, trades] =
      await Promise.all([
        get("/api/learning/summary"),
        get("/api/learning/expected"),
        get("/api/learning/errors"),
        get("/api/learning/reward-history"),
        get("/api/learning/by-symbol"),
        get("/api/learning/by-regime"),
        get("/api/learning/models"),
        get("/api/learning/promotions"),
        get("/api/learning/drift"),
        get("/api/learning/summary").then((s) => s.training_sets || []),
        get("/api/learning/experiences?limit=200"),
      ]);
    renderCards(summary);
    renderExpectancy(expect);
    renderErrors(errors);
    drawChart(history);
    renderGroup("bySymbol", bySymbol, "Symbol");
    renderGroup("byRegime", byRegime, "Regime");
    renderModels(models);
    renderPromotions(promos);
    renderDrift(drift);
    renderDatasets(datasets);
    renderTrades(trades);
  } catch (e) {
    document.getElementById("drift").textContent = "load error: " + e.message;
  }
}

load();
