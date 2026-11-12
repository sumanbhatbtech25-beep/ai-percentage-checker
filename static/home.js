// "Your saved reports" on the home page: continue or delete each one.
const esc = (s) => String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

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

function level(p) { return p >= 50 ? "high" : p >= 20 ? "mid" : "low"; }

async function loadSaved() {
  let reports = [];
  try { reports = (await fetch("/api/reports").then((r) => r.json())).reports || []; } catch {}
  const box = document.getElementById("saved");
  box.hidden = reports.length === 0;
  document.getElementById("savedList").innerHTML = reports.map((r) => `
    <li class="saved-item" data-id="${esc(r.id)}">
      <div class="saved-main">
        <div class="saved-title">${esc(r.title)}</div>
        <div class="saved-meta">
          ${r.ai_percent != null ? `<span class="badge ${level(r.ai_percent)}">${r.ai_percent}% AI</span>` : ""}
          <span>${(r.words || 0).toLocaleString()} words</span>
          <span>· saved ${ago(r.updated)}</span>
        </div>
      </div>
      <div class="saved-actions">
        <a class="continue" href="/editor?report=${encodeURIComponent(r.id)}">Continue this report <span aria-hidden="true">→</span></a>
        <button class="delete" data-id="${esc(r.id)}" data-title="${esc(r.title)}">Delete</button>
      </div>
    </li>`).join("");
}

document.getElementById("savedList").addEventListener("click", async (e) => {
  const b = e.target.closest(".delete");
  if (!b) return;
  if (!confirm(`Delete "${b.dataset.title}"?\n\nThis removes the saved report from this computer and can't be undone.`)) return;
  b.disabled = true;
  try {
    const r = await fetch(`/api/reports/${encodeURIComponent(b.dataset.id)}`, { method: "DELETE" });
    if (!r.ok) throw new Error("Couldn't delete the report");
    loadSaved();
  } catch (err) {
    alert(err.message);
    b.disabled = false;
  }
});

loadSaved();
// coming back with the browser's Back button can show a cached page; refresh the list
window.addEventListener("pageshow", (e) => { if (e.persisted) loadSaved(); });
