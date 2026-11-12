const $ = (id) => document.getElementById(id);
const state = { text: "", result: null, edits: 0, editing: false, popFor: -1 };

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
    $("submit").hidden = true;
    $("report").hidden = false;
    renderAll();
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
  $("warnings").innerHTML = (r.warnings || []).map((w) => `<div class="warn">⚠ ${esc(w)}</div>`).join("");
  const s = r.signals || {};
  setSig("Cls", s.classifier); setSig("Bino", s.binoculars); setSig("Style", s.style);
}
function setSig(k, v) {
  v = v ?? 0;
  $("sig" + k).textContent = Math.round(v * 100) + "%";
  $("bar" + k).style.width = v * 100 + "%";
}

function renderDoc() {
  const { text } = state;
  const r = state.result;
  const thr = r.threshold;
  const sugg = r.suggestions;
  let html = "", pos = 0, si = 0;
  for (const s of r.sentences) {
    html += esc(text.slice(pos, s.start));
    const cls = s.score >= thr ? "s-high" : s.score >= thr * 0.75 ? "s-mid" : "";
    let inner = "", p = s.start;
    while (si < sugg.length && sugg[si].start < s.start) si++;
    while (si < sugg.length && sugg[si].end <= s.end) {
      const g = sugg[si];
      inner += esc(text.slice(p, g.start));
      const c = `w sev-${g.severity}${g.applied ? " applied" : ""}`;
      inner += `<span class="${c}" data-i="${si}" title="${esc(g.reason)}">${esc(text.slice(g.start, g.end)) || "&#8203;"}</span>`;
      p = g.end;
      si++;
    }
    inner += esc(text.slice(p, s.end));
    html += `<span class="${cls}" title="AI likelihood ${Math.round(s.score * 100)}%">${inner}</span>`;
    pos = s.end;
  }
  html += esc(text.slice(pos));
  $("doc").innerHTML = html;
}

function renderSuggList() {
  const sugg = state.result.suggestions;
  $("suggCount").textContent = sugg.length;
  $("suggList").innerHTML = sugg.length
    ? sugg.map((g, i) => `<li data-i="${i}" class="${g.applied ? "done" : ""}"><b>${esc(g.original)}</b> → ${esc(g.alternatives.slice(0, 2).join(", "))}</li>`).join("")
    : "<li class='muted'>No suggestions — nothing stood out.</li>";
}

function renderStale() {
  const n = state.edits;
  $("stale").hidden = n === 0;
  $("staleMsg").textContent = `${n} change${n === 1 ? "" : "s"} made. The score above is from before your edits.`;
}

// ------------------------------------------------------------------ dropdown
$("doc").addEventListener("click", (e) => {
  const w = e.target.closest(".w");
  if (!w) return;
  openPop(+w.dataset.i, w);
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
    `<div class="why">${esc(g.reason)}</div>` +
    opts.map((a) => `<button data-v="${esc(a)}">${a === "(remove)" ? "<i>Remove it</i>" : esc(a)}</button>`).join("") +
    (g.applied ? `<button class="orig" data-v="${esc(g.original)}">↺ Restore “${esc(g.original)}”</button>` : "") +
    `<div class="custom"><input placeholder="Type your own…"><button>Use</button></div>`;
  const rect = anchor.getBoundingClientRect();
  pop.hidden = false;
  const left = Math.min(rect.left + window.scrollX, window.scrollX + document.documentElement.clientWidth - pop.offsetWidth - 12);
  pop.style.left = Math.max(8, left) + "px";
  pop.style.top = rect.bottom + window.scrollY + 6 + "px";
  pop.querySelector("button")?.focus();
}
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
  r.sentences.forEach((s) => {
    if (s.start >= oldEnd) { s.start += delta; s.end += delta; }
    else if (s.end >= oldEnd || (s.start <= start && s.end >= start)) s.end += delta;
  });
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
