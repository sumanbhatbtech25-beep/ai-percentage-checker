const $ = (id) => document.getElementById(id);
const state = { text: "", result: null, edits: 0, editing: false, popFor: -1, panels: {} };

// ------------------------------------------------------------------ model status
async function pollHealth() {
  try {
    const r = await fetch("/api/health").then((r) => r.json());
    if (r.ready) {
      $("status").textContent = `Models ready (${r.device.toUpperCase()})`;
      $("status").className = "status ready";
      updateCheckBtn();
      return;
    }
  } catch {}
  setTimeout(pollHealth, 2000);
}
pollHealth();

// ------------------------------------------------------------------ input
const countWords = (t) => (t.match(/[A-Za-z][A-Za-z'\-]*/g) || []).length;
function updateCheckBtn() {
  const n = countWords($("input").value);
  $("wc").textContent = `${n} words`;
  $("check").disabled = !n || !$("status").classList.contains("ready");
}
$("input").addEventListener("input", updateCheckBtn);

async function loadFile(file) {
  $("filename").textContent = `Reading ${file.name}…`;
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch("/api/extract", { method: "POST", body: fd });
  const j = await r.json();
  if (!r.ok) { $("filename").textContent = j.detail || "Could not read file"; return; }
  $("input").value = j.text;
  $("filename").textContent = file.name;
  updateCheckBtn();
}
$("file").addEventListener("change", (e) => e.target.files[0] && loadFile(e.target.files[0]));
const drop = $("drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && loadFile(e.dataTransfer.files[0]));

$("check").addEventListener("click", () => runAnalysis($("input").value));
$("recheck").addEventListener("click", () => { finishEditing(); runAnalysis(state.text); });

async function runAnalysis(text) {
  busy(true, "Scoring every section with 3 detectors…");
  try {
    const r = await fetch("/api/analyze", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, mode: $("mode").value }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || "Analysis failed");
    state.text = text.replace(/\r\n/g, "\n");
    state.result = j;
    state.edits = 0;
    state.panels = {};
    $("submit").hidden = true;
    $("report").hidden = false;
    renderAll();
    // party crackers when the text reads as human
    if (j.ai_percent < 5 && j.word_count > 0 && window.partyCrackers) setTimeout(window.partyCrackers, 350);
    window.scrollTo({ top: 0 });
  } catch (e) {
    alert(e.message);
  } finally {
    busy(false);
  }
}
function busy(on, msg) { $("busy").hidden = !on; if (msg) $("busyMsg").textContent = msg; }

// ------------------------------------------------------------------ rendering
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

function renderAll() {
  renderScore();
  renderDoc();
  renderSuggList();
  renderParaList();
  renderStale();
}

function renderScore() {
  const r = state.result;
  const pct = r.ai_percent;
  $("pct").textContent = pct;
  const arc = $("arc");
  arc.style.strokeDashoffset = 326.7 * (1 - pct / 100);
  arc.style.stroke = pct >= 50 ? "var(--ai-strong)" : pct >= 20 ? "var(--mid-strong)" : "var(--ok)";
  const modeName = { very_hard: "Very hard", strict: "Strict", standard: "Standard" }[r.mode];
  $("scoreMeta").textContent = `${r.word_count} words · ${modeName} mode`;
  $("scoreText").textContent = pct < 5 ? "Reads as human-written. Nice work!" : "of text likely AI-generated";
  $("warnings").innerHTML = (r.warnings || []).map((w) => `<div class="warn">⚠ ${esc(w)}</div>`).join("");
  const s = r.signals || {};
  setSig("Cls", s.classifier); setSig("Bino", s.binoculars); setSig("Style", s.style);
}
function setSig(k, v) {
  v = v ?? 0;
  $("sig" + k).textContent = Math.round(v * 100) + "%";
  $("bar" + k).style.width = v * 100 + "%";
}

const level = (pct) => (pct >= 50 ? "high" : pct >= 20 ? "mid" : "low");
const pctOf = (v) => Math.round((v ?? 0) * 100) + "%";

function renderRange(a, b) {
  const { text } = state;
  const r = state.result;
  const thr = r.threshold;
  let html = "", pos = a;
  for (const s of r.sentences) {
    if (s.start < a || s.end > b) continue;
    html += esc(text.slice(pos, s.start));
    const cls = s.score >= thr ? "s-high" : s.score >= thr * 0.75 ? "s-mid" : "";
    let inner = "", p = s.start;
    r.suggestions.forEach((g, i) => {
      if (g.start < s.start || g.end > s.end) return;
      inner += esc(text.slice(p, g.start));
      const c = `w sev-${g.severity}${g.applied ? " applied" : ""}`;
      const tip = g.applied ? `Changed from “${g.original}”. Tap to change again or restore.` : `${g.reason}. Try: ${g.alternatives.slice(0, 3).join(", ")}`;
      inner += `<span class="${c}" data-i="${i}" role="button" tabindex="0" title="${esc(tip)}">${esc(text.slice(g.start, g.end))}</span>`;
      p = g.end;
    });
    inner += esc(text.slice(p, s.end));
    html += `<span class="${cls}" title="AI likelihood ${Math.round(s.score * 100)}%">${inner}</span>`;
    pos = s.end;
  }
  return html + esc(text.slice(pos, b));
}

function renderDoc() {
  const paras = state.result.paragraphs || [];
  if (!paras.length) {
    $("doc").innerHTML = `<div class="pbody">${esc(state.text)}</div>`;
    return;
  }
  $("doc").innerHTML = paras.map((p, i) => {
    const lv = level(p.ai_percent);
    const open = !!state.panels[i];
    return `<section class="para lvl-${lv}" id="para-${i}">` +
      `<div class="para-head"><span class="badge ${lv}">¶${i + 1} · ${p.rewritten ? "rewritten · " : ""}${p.ai_percent}% AI</span>` +
      `<button class="small rw-btn" data-p="${i}">${open ? "Hide" : "Why AI? · Suggest better paragraph"}</button></div>` +
      `<div class="pbody">${renderRange(p.start, p.end)}</div>` +
      (open ? renderPanel(i) : "") +
      `</section>`;
  }).join("");
}

function renderPanel(i) {
  const p = state.result.paragraphs[i];
  const st = state.panels[i];
  const reasons = (p.reasons || []).length
    ? p.reasons.map((x) => `<li>${esc(x)}</li>`).join("")
    : `<li>${p.rewritten ? "This paragraph was replaced. Press Re-check score to analyse it again." : "No single pattern stands out. Detectors are reacting to the overall phrasing, so rewriting it in your own voice helps most."}</li>`;
  let html = `<div class="rw-panel"><h4>Why this paragraph looks AI-written</h4><ul>${reasons}</ul>` +
    `<div class="rw-actions"><button class="primary small gen-btn" data-p="${i}" ${st.loading ? "disabled" : ""}>${st.data ? "Generate new versions" : "Suggest rewritten versions"}</button>` +
    `<span class="muted tiny">${st.loading ? "Writing and scoring 3 versions… (~20 s)" : "Each version is scored by the same 3 detectors"}</span></div>`;
  if (st.error) html += `<div class="warn">${esc(st.error)}</div>`;
  if (st.data) {
    const o = st.data.original;
    html += `<div class="muted tiny cand-orig">This paragraph on its own: <b>${o.ai_percent}% AI</b> · classifier ${pctOf(o.signals.classifier)} · Binoculars ${pctOf(o.signals.binoculars)} · style ${pctOf(o.signals.style)}</div>`;
    if (!st.data.candidates.length) html += `<div class="warn">No usable versions came back. Try again.</div>`;
    st.data.candidates.forEach((c, k) => {
      html += `<div class="cand"><div class="cand-top"><span class="badge ${level(c.ai_percent)}">${c.ai_percent}% AI</span>` +
        `<span class="muted tiny">classifier ${pctOf(c.signals.classifier)} · Binoculars ${pctOf(c.signals.binoculars)} · style ${pctOf(c.signals.style)}</span>` +
        `<button class="small use-btn" data-p="${i}" data-c="${k}">Use this</button></div>` +
        `<p>${esc(c.text)}</p>${c.note ? `<div class="note">${esc(c.note)}</div>` : ""}</div>`;
    });
  }
  return html + `</div>`;
}

function renderParaList() {
  const paras = (state.result.paragraphs || []).map((p, i) => ({ p, i }));
  paras.sort((x, y) => y.p.ai_percent - x.p.ai_percent || y.p.score - x.p.score);
  $("paraList").innerHTML = paras.length
    ? paras.map(({ p, i }) => `<li data-p="${i}"><span class="badge ${level(p.ai_percent)}">${p.ai_percent}%</span> ¶${i + 1} ${esc(state.text.slice(p.start, p.start + 55))}…</li>`).join("")
    : "<li class='muted'>Re-check to see paragraph scores.</li>";
}

async function generateRewrites(i) {
  const p = state.result.paragraphs[i];
  state.panels[i] = { ...state.panels[i], loading: true, error: "" };
  renderDoc();
  try {
    const r = await fetch("/api/rewrite", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: state.text.slice(p.start, p.end), mode: state.result.mode }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || "Rewrite failed");
    state.panels[i] = { data: j };
  } catch (e) {
    state.panels[i] = { ...state.panels[i], error: e.message };
  }
  state.panels[i].loading = false;
  renderDoc();
}

function useRewrite(i, k) {
  const r = state.result;
  const p = r.paragraphs[i];
  const c = state.panels[i].data.candidates[k];
  const a = p.start, b = p.end, delta = c.text.length - (b - a);
  state.text = state.text.slice(0, a) + c.text + state.text.slice(b);
  r.sentences = r.sentences.filter((s) => s.end <= a || s.start >= b);
  r.suggestions = r.suggestions.filter((g) => g.end <= a || g.start >= b);
  for (const list of [r.sentences, r.suggestions, r.paragraphs]) {
    for (const s of list) if (s !== p && s.start >= b) { s.start += delta; s.end += delta; }
  }
  Object.assign(p, { end: a + c.text.length, rewritten: true, ai_percent: c.ai_percent, score: c.score, reasons: [] });
  state.panels[i] = null;
  state.edits++;
  renderDoc(); renderSuggList(); renderParaList(); renderStale();
}

function renderSuggList() {
  const sugg = state.result.suggestions;
  $("suggCount").textContent = sugg.length;
  $("suggList").innerHTML = sugg.length
    ? sugg.map((g, i) => `<li data-i="${i}" class="${g.applied ? "done" : ""}"><b>${esc(g.original)}</b> → ${esc(g.alternatives.slice(0, 2).join(", "))}</li>`).join("")
    : "<li class='muted'>No suggestions — nothing stood out.</li>";
}

function renderStale() {
  try { sessionStorage.setItem("aic_text", state.text); } catch {}
  const n = state.edits;
  $("stale").hidden = n === 0;
  $("staleMsg").textContent = `${n} change${n === 1 ? "" : "s"} made. The score above is from before your edits.`;
}

// ------------------------------------------------------------------ dropdown
$("doc").addEventListener("click", (e) => {
  const rw = e.target.closest(".rw-btn");
  if (rw) {
    const i = +rw.dataset.p;
    state.panels[i] = state.panels[i] ? null : {};
    renderDoc();
    return;
  }
  const gen = e.target.closest(".gen-btn");
  if (gen) return generateRewrites(+gen.dataset.p);
  const use = e.target.closest(".use-btn");
  if (use) return useRewrite(+use.dataset.p, +use.dataset.c);
  const w = e.target.closest(".w");
  if (!w) return;
  openPop(+w.dataset.i, w);
});
$("paraList").addEventListener("click", (e) => {
  const li = e.target.closest("li[data-p]");
  if (!li) return;
  const i = +li.dataset.p;
  if (!state.panels[i]) { state.panels[i] = {}; renderDoc(); }
  $("para-" + i).scrollIntoView({ block: "start", behavior: "smooth" });
});
$("suggList").addEventListener("click", (e) => {
  const li = e.target.closest("li[data-i]");
  if (!li) return;
  const i = +li.dataset.i;
  const span = document.querySelector(`.w[data-i="${i}"]`);
  if (span) { span.scrollIntoView({ block: "center", behavior: "smooth" }); setTimeout(() => openPop(i, span), 300); }
});

function openPop(i, anchor) {
  const g = state.result.suggestions[i];
  const current = state.text.slice(g.start, g.end);
  document.querySelectorAll(".w.active").forEach((x) => x.classList.remove("active"));
  anchor.classList.add("active");
  state.popFor = i;
  const opts = g.alternatives.filter((a) => a !== current);
  const pop = $("pop");
  pop.innerHTML =
    `<div class="pop-head">Replace <b>“${esc(current || g.original)}”</b> with:</div>` +
    `<div class="why">${esc(g.reason)}</div>` +
    opts.map((a) => `<button data-v="${esc(a)}">${a === "(remove)" ? "<i>Remove it</i>" : esc(a)}</button>`).join("") +
    (g.applied ? `<button class="orig" data-v="${esc(g.original)}">↺ Restore “${esc(g.original)}”</button>` : "") +
    `<div class="custom"><input placeholder="Type your own…"><button>Use</button></div>`;
  const rect = anchor.getBoundingClientRect();
  pop.hidden = false;
  const vw = document.documentElement.clientWidth;
  const left = Math.min(rect.left + window.scrollX, window.scrollX + vw - pop.offsetWidth - 12);
  pop.style.left = Math.max(8, left) + "px";
  // open upward when there isn't room below the word
  const below = window.innerHeight - rect.bottom;
  const top = below < pop.offsetHeight + 16 && rect.top > pop.offsetHeight + 16
    ? rect.top + window.scrollY - pop.offsetHeight - 6
    : rect.bottom + window.scrollY + 6;
  pop.style.top = top + "px";
  pop.querySelector("button")?.focus({ preventScroll: true });
}
$("doc").addEventListener("keydown", (e) => {
  const w = e.target.closest(".w");
  if (w && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); openPop(+w.dataset.i, w); }
});
function closePop() {
  $("pop").hidden = true;
  state.popFor = -1;
  document.querySelectorAll(".w.active").forEach((x) => x.classList.remove("active"));
}
$("pop").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  let v = b.dataset.v;
  if (v === undefined) v = $("pop").querySelector("input").value.trim();
  if (v === undefined || (v === "" && !b.dataset.v)) return;
  applyReplacement(state.popFor, v === "(remove)" ? "" : v);
  closePop();
});
$("pop").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && e.target.tagName === "INPUT") $("pop").querySelector(".custom button").click();
  if (e.key === "Escape") closePop();
});
document.addEventListener("mousedown", (e) => {
  if (!$("pop").hidden && !e.target.closest("#pop") && !e.target.closest(".w") && !e.target.closest("#suggList")) closePop();
});

function applyReplacement(i, value) {
  const r = state.result;
  const g = r.suggestions[i];
  let { start, end } = g;
  let text = state.text;
  if (value === "") {
    // removing: also swallow one adjacent space so we don't leave a double space
    if (text[end] === " " && (start === 0 || text[start - 1] === " " || text[start - 1] === "\n")) end++;
    else if (text[start - 1] === " ") start--;
  }
  const oldEnd = end;
  const delta = value.length - (end - start);
  state.text = text.slice(0, start) + value + text.slice(end);
  g.start = start;
  g.end = start + value.length;
  g.applied = value !== g.original;
  r.suggestions.forEach((o, k) => { if (k !== i && o.start >= oldEnd) { o.start += delta; o.end += delta; } });
  for (const list of [r.sentences, r.paragraphs || []]) {
    list.forEach((s) => {
      if (s.start >= oldEnd) { s.start += delta; s.end += delta; }
      else if (s.end >= oldEnd || (s.start <= start && s.end >= start)) s.end += delta;
    });
  }
  state.edits++;
  renderDoc();
  renderSuggList();
  renderStale();
}

// ------------------------------------------------------------------ free editing
$("editToggle").addEventListener("click", () => {
  if (!state.editing) {
    state.editing = true;
    $("editor").value = state.text;
    $("editor").style.height = $("doc").offsetHeight + "px";
    $("doc").hidden = true;
    $("editor").hidden = false;
    $("editToggle").textContent = "Done editing";
  } else finishEditing();
});
function finishEditing() {
  if (!state.editing) return;
  state.editing = false;
  const t = $("editor").value;
  if (t !== state.text) {
    state.text = t;
    // offsets are no longer valid: show plain text until re-check
    state.result.sentences = [];
    state.result.suggestions = [];
    state.result.paragraphs = [];
    state.panels = {};
    state.edits++;
  }
  $("editor").hidden = true;
  $("doc").hidden = false;
  $("editToggle").textContent = "Edit text";
  renderDoc(); renderSuggList(); renderStale();
}
$("newDoc").addEventListener("click", () => {
  $("report").hidden = true; $("submit").hidden = false;
  $("input").value = state.text; updateCheckBtn();
});

// ------------------------------------------------------------------ export
const currentText = () => (state.editing ? $("editor").value : state.text);
$("dlTxt").addEventListener("click", () => download(new Blob([currentText()], { type: "text/plain" }), "revised.txt"));
$("dlDocx").addEventListener("click", async () => {
  const r = await fetch("/api/export-docx", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: currentText() }) });
  download(await r.blob(), "revised.docx");
});
$("copy").addEventListener("click", async () => { await navigator.clipboard.writeText(currentText()); $("copy").textContent = "Copied!"; setTimeout(() => ($("copy").textContent = "Copy"), 1500); });
function download(blob, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
