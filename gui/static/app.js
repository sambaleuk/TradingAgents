const state = {
  config: null,
  runId: null,
  poll: null,
  activeReport: null,
};

const $ = (selector) => document.querySelector(selector);

function clock() {
  $("#clock").textContent = new Date().toLocaleTimeString();
}

function mmss(seconds) {
  const m = String(Math.floor(seconds / 60)).padStart(2, "0");
  const s = String(seconds % 60).padStart(2, "0");
  return `${m}:${s}`;
}

function option(value, label = value) {
  const node = document.createElement("option");
  node.value = value;
  node.textContent = label;
  return node;
}

async function loadConfig() {
  const response = await fetch("/api/config");
  state.config = await response.json();

  const defaults = state.config.defaults;
  const providerSelect = $("#providerSelect");
  Object.keys(state.config.models).forEach((provider) => {
    providerSelect.append(option(provider, provider.toUpperCase()));
  });
  providerSelect.value = defaults.llm_provider;
  $("input[name=analysis_date]").value = defaults.analysis_date;
  updateModelSelects();
}

function updateModelSelects() {
  const provider = $("#providerSelect").value;
  const models = state.config.models[provider] || { quick: [], deep: [] };
  const quick = $("#quickModel");
  const deep = $("#deepModel");
  quick.innerHTML = "";
  deep.innerHTML = "";
  (models.quick || []).forEach(([label, value]) => quick.append(option(value, label)));
  (models.deep || []).forEach(([label, value]) => deep.append(option(value, label)));
}

function payloadFromForm(form) {
  const data = new FormData(form);
  return {
    ticker: data.get("ticker"),
    analysis_date: data.get("analysis_date"),
    analysts: data.getAll("analysts"),
    research_depth: Number(data.get("research_depth")),
    llm_provider: data.get("llm_provider"),
    quick_think_llm: data.get("quick_think_llm"),
    deep_think_llm: data.get("deep_think_llm"),
    output_language: data.get("output_language"),
    checkpoint_enabled: data.get("checkpoint_enabled") === "on",
  };
}

async function startRun(event) {
  event.preventDefault();
  const response = await fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payloadFromForm(event.currentTarget)),
  });
  const body = await response.json();
  if (!response.ok) {
    $("#runStatus").textContent = "ERROR";
    $("#feed").innerHTML = `<div class="feed-item"><span></span><span class="failed">ERROR</span><span>${body.error}</span></div>`;
    return;
  }
  state.runId = body.run_id;
  state.activeReport = null;
  $("#runStatus").textContent = "RUNNING";
  if (state.poll) clearInterval(state.poll);
  await pollRun();
  state.poll = setInterval(pollRun, 1500);
}

async function pollRun() {
  if (!state.runId) return;
  const response = await fetch(`/api/run?id=${state.runId}`);
  const run = await response.json();
  renderRun(run);
  if (["completed", "failed"].includes(run.status)) {
    clearInterval(state.poll);
    state.poll = null;
  }
}

function renderRun(run) {
  $("#runStatus").textContent = run.status.toUpperCase();
  $("#elapsed").textContent = mmss(run.stats.elapsed_seconds || 0);
  $("#agentProgress").textContent = `${run.stats.agents_completed}/${run.stats.agents_total}`;
  $("#reportProgress").textContent = `${run.stats.reports_completed}/${run.stats.reports_total}`;
  renderAgents(run.agent_status || {});
  renderReports(run);
  renderFeed(run);
  if (run.error) {
    $("#reportBody").textContent = run.error;
  }
}

function renderAgents(agentStatus) {
  const root = $("#agents");
  root.innerHTML = "";
  Object.entries(agentStatus).forEach(([name, status]) => {
    const item = document.createElement("div");
    item.className = "agent";
    item.innerHTML = `<div class="agent-name">${name}</div><div class="agent-state ${status}">${status.replace("_", " ")}</div>`;
    root.append(item);
  });
}

function renderReports(run) {
  const tabs = $("#reports");
  tabs.innerHTML = "";
  const reports = run.reports || {};
  const keys = Object.keys(reports);
  if (!state.activeReport || !keys.includes(state.activeReport)) {
    state.activeReport = keys.find((key) => reports[key]) || keys[0];
  }
  keys.forEach((key) => {
    const tab = document.createElement("button");
    tab.type = "button";
    tab.className = `tab ${key === state.activeReport ? "active" : ""}`;
    tab.textContent = run.report_titles[key] || key;
    tab.onclick = () => {
      state.activeReport = key;
      renderReports(run);
    };
    tabs.append(tab);
  });
  const content = reports[state.activeReport] || "Waiting for this desk to publish.";
  $("#reportBody").textContent = content;
}

function renderFeed(run) {
  const feed = $("#feed");
  const messages = (run.messages || []).map((m) => ({
    time: m.time,
    type: m.type,
    content: m.content,
  }));
  const tools = (run.tool_calls || []).map((t) => ({
    time: t.time,
    type: "Tool",
    content: `${t.name} ${JSON.stringify(t.args).slice(0, 180)}`,
  }));
  const items = messages.concat(tools).slice(-80).reverse();
  feed.innerHTML = "";
  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "feed-item";
    row.innerHTML = `<span>${item.time}</span><span class="type">${item.type}</span><span>${escapeHtml(item.content)}</span>`;
    feed.append(row);
  });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function refreshQuote() {
  const ticker = $("input[name=ticker]").value || "SPY";
  const response = await fetch(`/api/market?ticker=${encodeURIComponent(ticker)}`);
  const quote = await response.json();
  $("#quoteSymbol").textContent = quote.ticker || ticker;
  if (quote.error) {
    $("#quoteLast").textContent = "--";
    $("#quoteChange").textContent = quote.error;
    return;
  }
  $("#quoteLast").textContent = quote.last;
  const change = `${quote.change >= 0 ? "+" : ""}${quote.change} (${quote.change_pct}%)`;
  $("#quoteChange").textContent = change;
  $("#quoteChange").className = quote.change >= 0 ? "completed" : "failed";
  drawSparkline(quote.sparkline || []);
}

function drawSparkline(points) {
  const canvas = $("#sparkline");
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  if (!points.length) return;
  const min = Math.min(...points);
  const max = Math.max(...points);
  const range = max - min || 1;
  ctx.strokeStyle = "#24313a";
  ctx.lineWidth = 1;
  for (let y = 20; y < height; y += 24) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
  ctx.strokeStyle = points.at(-1) >= points[0] ? "#19d27c" : "#ff4f5e";
  ctx.lineWidth = 3;
  ctx.beginPath();
  points.forEach((point, index) => {
    const x = (index / (points.length - 1 || 1)) * (width - 20) + 10;
    const y = height - 12 - ((point - min) / range) * (height - 24);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

async function refreshTape() {
  const symbols = ["SPY", "QQQ", "NVDA", "TSLA", "AAPL"];
  const quotes = await Promise.all(symbols.map((ticker) => fetch(`/api/market?ticker=${ticker}`).then((r) => r.json())));
  $("#tickerTape").innerHTML = quotes
    .map((q) => {
      if (q.error) return `<span>${q.ticker} --</span>`;
      const klass = q.change >= 0 ? "completed" : "failed";
      return `<span class="${klass}">${q.ticker} ${q.last} ${q.change >= 0 ? "+" : ""}${q.change_pct}%</span>`;
    })
    .join("");
}

document.addEventListener("DOMContentLoaded", async () => {
  setInterval(clock, 1000);
  clock();
  await loadConfig();
  await refreshQuote();
  await refreshTape();
  $("#providerSelect").addEventListener("change", updateModelSelects);
  $("#runForm").addEventListener("submit", startRun);
  $("input[name=ticker]").addEventListener("change", refreshQuote);
  setInterval(refreshQuote, 60000);
  setInterval(refreshTape, 90000);
});

