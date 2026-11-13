// "Your saved reports" on the home page: search, sort, rename, continue, delete.
const esc = (s) => String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const $ = (id) => document.getElementById(id);
let reports = [];

function ago(ts) {
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 60) return "just now";
  const units = [["year", 31536000], ["month", 2592000], ["week", 604800], ["day", 86400], ["hour", 3600], ["minute", 60]];
  for (const [name, secs] of units) {
    const n = Math.floor(s / secs);
    if (n >= 1) return `${n} ${name}${n === 1 ? "" : "s"} ago`;
  }
  return "just now";
}

const level = (p) => (p >= 50 ? "high" : p >= 20 ? "mid" : "low");

const SORTS = {
  updated: (a, b) => (b.updated || 0) - (a.updated || 0),
  created: (a, b) => (a.created || 0) - (b.created || 0),
  title: (a, b) => (a.title || "").localeCompare(b.title || "", undefined, { sensitivity: "base" }),
  ai_high: (a, b) => (b.ai_percent ?? -1) - (a.ai_percent ?? -1),
  ai_low: (a, b) => (a.ai_percent ?? 999) - (b.ai_percent ?? 999),
};

try { $("savedSort").value = localStorage.getItem("aic_sort") || "updated"; } catch {}

async function loadSaved() {
  try { reports = (await fetch("/api/reports").then((r) => r.json())).reports || []; } catch { reports = []; }
  render();
}

function render() {
  $("saved").hidden = reports.length === 0;
  $("savedCount").textContent = reports.length;
  // search + sort only earn their space once there are a few reports
  document.querySelector(".saved-tools").hidden = reports.length < 3;
  const q = $("savedSearch").value.trim().toLowerCase();
  const shown = reports.filter((r) => !q || (r.title || "").toLowerCase().includes(q)).sort(SORTS[$("savedSort").value] || SORTS.updated);
  $("savedList").innerHTML = shown.length ? shown.map((r) => `
    <li class="saved-item" data-id="${esc(r.id)}">
      <div class="saved-main">
        <div class="saved-title" title="${esc(r.title)}">${esc(r.title)}</div>
        <div class="saved-meta">
          ${r.ai_percent != null ? `<span class="badge ${level(r.ai_percent)}">${r.ai_percent}% AI</span>` : ""}
          <span>${(r.words || 0).toLocaleString()} words</span>
          <span>· saved ${ago(r.updated)}</span>
        </div>
      </div>
      <div class="saved-actions">
        <a class="continue" href="/editor?report=${encodeURIComponent(r.id)}">Continue this report <span aria-hidden="true">→</span></a>
        <button class="rename" aria-label="Rename ${esc(r.title)}">Rename</button>
        <button class="delete" aria-label="Delete ${esc(r.title)}">Delete</button>
      </div>
    </li>`).join("") : `<li class="muted saved-empty">No saved report matches “${esc(q)}”.</li>`;
}

$("savedSearch").addEventListener("input", render);
$("savedSort").addEventListener("change", () => { try { localStorage.setItem("aic_sort", $("savedSort").value); } catch {} render(); });

function startRename(li) {
  const r = reports.find((x) => x.id === li.dataset.id);
  const titleEl = li.querySelector(".saved-title");
  titleEl.innerHTML = `<input class="rename-input" value="${esc(r.title)}" maxlength="120" aria-label="New name">`;
  const input = titleEl.querySelector("input");
  input.focus();
  input.select();
  let done = false;
  const finish = async (save) => {
    if (done) return;
    done = true;
    const title = input.value.trim();
    if (save && title && title !== r.title) {
      try {
        const res = await fetch(`/api/reports/${encodeURIComponent(r.id)}`, {
          method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title }),
        });
        if (!res.ok) throw new Error("Couldn't rename the report");
        r.title = (await res.json()).title;
      } catch (e) { alert(e.message); }
    }
    render();
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") finish(true);
    if (e.key === "Escape") finish(false);
  });
  input.addEventListener("blur", () => finish(true));
}

$("savedList").addEventListener("click", async (e) => {
  const li = e.target.closest(".saved-item");
  if (!li) return;
  if (e.target.closest(".rename")) return startRename(li);
  const del = e.target.closest(".delete");
  if (!del) return;
  const r = reports.find((x) => x.id === li.dataset.id);
  if (!confirm(`Delete "${r.title}"?\n\nThis removes the saved report from this computer and can't be undone.`)) return;
  del.disabled = true;
  try {
    const res = await fetch(`/api/reports/${encodeURIComponent(r.id)}`, { method: "DELETE" });
    if (!res.ok) throw new Error("Couldn't delete the report");
    reports = reports.filter((x) => x.id !== r.id);
    render();
  } catch (err) {
    alert(err.message);
    del.disabled = false;
  }
});

loadSaved();
// coming back with the browser's Back button can show a cached page; refresh the list
window.addEventListener("pageshow", (e) => { if (e.persisted) loadSaved(); });

// status pill: shows when the detection models are ready
  (async function poll() {
    try {
      const r = await fetch("/api/health").then((r) => r.json());
      const s = document.getElementById("status");
      if (r.ready) { s.textContent = `Ready (${r.device.toUpperCase()})`; s.className = "status ready"; return; }
      s.textContent = "Loading detection models…";
    } catch {}
    setTimeout(poll, 2000);
  })();
