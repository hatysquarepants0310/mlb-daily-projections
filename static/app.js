const $ = (id) => document.getElementById(id);

function todayISO() {
  const d = new Date();
  const t = new Date(d.getTime() - d.getTimezoneOffset() * 60000);
  return t.toISOString().slice(0, 10);
}

$("date").value = todayISO();

$("bar").addEventListener("submit", (e) => {
  e.preventDefault();
  load(false);
});
$("force").addEventListener("click", () => load(true));

function fmt(n, d = 2) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return Number(n).toFixed(d);
}

function sortTable(table, col, numeric) {
  const tbody = table.tBodies[0];
  const rows = [...tbody.rows];
  const dir = table.dataset.dir === "asc" ? -1 : 1;
  table.dataset.dir = dir === 1 ? "asc" : "desc";
  rows.sort((a, b) => {
    const av = a.cells[col].dataset.v ?? a.cells[col].innerText;
    const bv = b.cells[col].dataset.v ?? b.cells[col].innerText;
    if (numeric) return dir * (parseFloat(av) - parseFloat(bv));
    return dir * String(av).localeCompare(String(bv));
  });
  rows.forEach((r) => tbody.appendChild(r));
}

function batterTable(rows) {
  const tbl = document.createElement("table");
  tbl.innerHTML = `<caption>Batters — proyección de juego completo</caption>
  <thead><tr>
    <th>Slot</th><th>Jugador</th><th>Vs</th><th>PA</th><th>H</th><th>80%</th>
    <th>HR</th><th>R</th><th>RBI</th><th>BB</th><th>K</th><th>SB</th>
    <th>AVG</th><th>wOBA</th><th>xwOBA</th><th>P(H)</th><th>P(HR)</th><th>Conf</th>
  </tr></thead><tbody></tbody>`;
  const tb = tbl.tBodies[0];
  for (const b of rows) {
    const tr = document.createElement("tr");
    const cells = [
      [b.slot, b.slot, false],
      [`${b.name} ${b.pos}`.trim(), `${b.team} ${b.name}`, false],
      [`${b.bats} vs ${b.pitcher} (${b.vs})`, b.pitcher, false],
      [fmt(b.pa), b.pa, true],
      [fmt(b.h), b.h, true],
      [`${fmt(b.h_lo)}–${fmt(b.h_hi)}`, b.h, true],
      [fmt(b.hr), b.hr, true],
      [fmt(b.r), b.r, true],
      [fmt(b.rbi), b.rbi, true],
      [fmt(b.bb), b.bb, true],
      [fmt(b.k), b.k, true],
      [fmt(b.sb), b.sb, true],
      [fmt(b.avg, 3), b.avg, true],
      [fmt(b.woba, 3), b.woba, true],
      [b.xwoba == null ? "—" : fmt(b.xwoba, 3), b.xwoba ?? -1, true],
      [fmt(b.p_hit, 3), b.p_hit, true],
      [fmt(b.p_hr, 3), b.p_hr, true],
      [fmt(b.confidence, 1), b.confidence, true],
    ];
    cells.forEach(([txt, v], i) => {
      const td = document.createElement("td");
      td.textContent = txt;
      td.dataset.v = v;
      if (i === 1) td.style.fontWeight = "600";
      if (i === 14 && b.p_hit >= 0.72) td.className = "hi";
      if (i === 15 && b.p_hr >= 0.18) td.className = "hot";
      if (i === 16) td.className = "conf";
      tr.appendChild(td);
    });
    tb.appendChild(tr);
  }
  [...tbl.tHead.rows[0].cells].forEach((th, i) => {
    th.addEventListener("click", () => sortTable(tbl, i, i >= 3 || i === 0));
  });
  return tbl;
}

function pitcherTable(rows) {
  const tbl = document.createElement("table");
  tbl.innerHTML = `<caption>Starters</caption>
  <thead><tr>
    <th>Pitcher</th><th>T</th><th>IP</th><th>K</th><th>BB</th><th>H</th><th>HR</th><th>ER</th><th>FIP</th><th>K%</th><th>xwOBA ag</th><th>Conf</th>
  </tr></thead><tbody></tbody>`;
  const tb = tbl.tBodies[0];
  for (const p of rows) {
    const tr = document.createElement("tr");
    const vals = [
      `${p.name} (${p.team})`, p.throws, fmt(p.ip), fmt(p.k), fmt(p.bb), fmt(p.h),
      fmt(p.hr), fmt(p.er), fmt(p.fip), fmt(p.k_rate, 3),
      p.xwoba_against == null ? "—" : fmt(p.xwoba_against, 3), fmt(p.confidence, 1),
    ];
    vals.forEach((t) => {
      const td = document.createElement("td");
      td.textContent = t;
      tr.appendChild(td);
    });
    tb.appendChild(tr);
  }
  return tbl;
}

async function load(force) {
  $("err").hidden = true;
  $("games").innerHTML = "<p class='empty'>Calculando… Stats API + Savant + Log5.</p>";
  const date = $("date").value;
  const url = `/api/projections?date=${encodeURIComponent(date)}${force ? "&force=true" : ""}`;
  let data;
  try {
    const res = await fetch(url);
    data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
  } catch (e) {
    $("games").innerHTML = "";
    $("err").hidden = false;
    $("err").textContent = String(e.message || e);
    return;
  }
  $("meta").textContent = `${data.n_games} juegos · ${data.n_batters} batters · gen ${data.generated_at}`;
  $("honest").textContent = data.model?.honest || "";
  const root = $("games");
  root.innerHTML = "";
  if (!data.games?.length) {
    root.innerHTML = "<p class='empty'>No hay juegos MLB en esa fecha.</p>";
    return;
  }
  for (const g of data.games) {
    const el = document.createElement("article");
    el.className = "game";
    const wx = g.weather || {};
    const inp = g.inputs || {};
    const lampState = inp.state || "esperando";
    el.innerHTML = `<h2>
      <span>${g.away.abbr} @ ${g.home.abbr}
        <span class="lamp lamp-${lampState}" title="${inp.label || "Esperando inputs"}">
          <span class="dot"></span>${inp.ready ? "Listo" : (lampState === "parcial" ? "Parcial" : "Esperando")}
        </span>
        <span class="wx">${g.status} · ${g.venue}</span>
      </span>
      <span class="wx">${wx.temp || "?"}°F · ${wx.condition || ""} · ${wx.wind || ""} · HR×${wx.hr_mult ?? ""}</span>
    </h2>
    <p class="empty" style="margin:8px 16px">${g.lock_clock?.label || ""} · first pitch ${g.gameDate || "?"}</p>`;
    const pills = document.createElement("div");
    pills.className = "pills";
    (g.pitchers || []).forEach((p) => {
      const s = document.createElement("span");
      s.className = "pill";
      s.textContent = `${p.name} ${p.throws}HP · FIP ${fmt(p.fip)} · K ${fmt(p.k)}`;
      pills.appendChild(s);
    });
    el.appendChild(pills);
    if (g.pitchers?.length) el.appendChild(pitcherTable(g.pitchers));
    if (g.batters?.length) el.appendChild(batterTable(g.batters));
    else {
      const p = document.createElement("p");
      p.className = "empty";
      p.textContent = "Sin lineups oficiales todavía.";
      el.appendChild(p);
    }
    el.appendChild(betsTable(g));
    root.appendChild(el);
  }
  renderHistory(data);
}

function spark(ticks) {
  if (!ticks || !ticks.length) return "—";
  return ticks.slice(-8).map((t) => Number(t.implied_p).toFixed(2)).join("→");
}

function betsTable(g) {
  const wrap = document.createElement("div");
  const cap = document.createElement("caption");
  const tbl = document.createElement("table");
  const url = g.poly_url ? ` · <a href="${g.poly_url}" target="_blank" rel="noopener">Polymarket</a>` : "";
  tbl.innerHTML = `<caption>Apuestas (momio Poly) — ${g.bets_note || ""}${url}</caption>
  <thead><tr>
    <th>Mercado</th><th>Lado</th><th>Momio</th><th>Amer</th><th>Poly%</th><th>Modelo</th><th>Edge</th><th>Confiabilidad</th><th>Path</th><th>Estado</th>
  </tr></thead><tbody></tbody>`;
  const tb = tbl.tBodies[0];
  for (const b of g.bets || []) {
    const tr = document.createElement("tr");
    if ((b.take || b.core || b.locked) && b.status !== "void-early") tr.className = "take";
    const st = b.status || "live";
    const stClass = st.includes("ganada") || st.includes("ganado") ? "won" : (st.includes("perdida") || st.includes("perdido") ? "lost" : "");
    const cells = [
      b.note || b.kind,
      b.selection,
      fmt(b.momio_decimal, 3),
      String(b.momio_americano ?? "—"),
      fmt(100 * b.implied_p, 1) + "%",
      fmt(100 * b.model_p, 1) + "%",
      (b.edge >= 0 ? "+" : "") + fmt(100 * b.edge, 1) + " pp",
      fmt(b.reliability, 1),
      spark(b.ticks),
      st,
    ];
    cells.forEach((t, i) => {
      const td = document.createElement("td");
      td.textContent = t;
      if (i === 9) td.className = stClass;
      tr.appendChild(td);
    });
    tb.appendChild(tr);
  }
  if (!(g.bets || []).length) {
    const p = document.createElement("p");
    p.className = "empty";
    p.textContent = g.bets_note || "Sin apuestas Poly.";
    wrap.appendChild(p);
    return wrap;
  }
  wrap.appendChild(tbl);
  return wrap;
}

function renderHistory(data) {
  const s = data.history_summary || {};
  $("hist-sum").textContent = s.n_settled
    ? `Historial locked: ${s.won}–${s.lost} (${Math.round(100 * (s.hit_rate || 0))}% hit) · PnL $100 MXN = ${s.pnl_100mxn} · n_lock ${s.n_locked}`
    : `Historial locked: ${s.n_locked || 0} picks congelados, 0 liquidados. El lock no se reescribe.`;
  const root = $("history");
  root.innerHTML = "";
  const tbl = document.createElement("table");
  tbl.innerHTML = `<caption>Ledger append-only (no delete / no update)</caption>
  <thead><tr><th>Fecha</th><th>Pick</th><th>Momio</th><th>Modelo</th><th>Lock</th><th>Resultado</th></tr></thead><tbody></tbody>`;
  const tb = tbl.tBodies[0];
  for (const r of data.history || []) {
    const tr = document.createElement("tr");
    const st = r.status || "";
    const cells = [
      r.card_date || "",
      `${r.selection} · ${r.note || r.kind}`,
      fmt(r.momio_decimal, 3),
      fmt(100 * r.model_p, 1) + "%",
      (r.locked_at || "").replace("T", " "),
      st,
    ];
    cells.forEach((t, i) => {
      const td = document.createElement("td");
      td.textContent = t;
      if (i === 5) td.className = st === "ganada" ? "won" : st === "perdida" ? "lost" : "";
      tr.appendChild(td);
    });
    tb.appendChild(tr);
  }
  if (!(data.history || []).length) {
    const p = document.createElement("p");
    p.className = "empty";
    p.textContent = "Todavía no hay locks. Solo se congelan picks pregame (Scheduled/Pre-Game).";
    root.appendChild(p);
    return;
  }
  root.appendChild(tbl);
}

load(false);
