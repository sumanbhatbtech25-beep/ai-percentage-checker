const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

// prefill with the text from the AI checker, if any
try {
  const t = sessionStorage.getItem("aic_text");
  if (t) { $("text").value = t; $("fromChecker").hidden = false; }
} catch {}

async function loadFile(file) {
  $("filename").textContent = `Reading ${file.name}…`;
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch("/api/extract", { method: "POST", body: fd });
  const j = await r.json();
  if (!r.ok) { $("filename").textContent = j.detail || "Could not read file"; return; }
  $("text").value = j.text;
  $("filename").textContent = file.name;
}
$("file").addEventListener("change", (e) => e.target.files[0] && loadFile(e.target.files[0]));
const drop = $("drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && loadFile(e.dataTransfer.files[0]));

$("search").addEventListener("click", search);
$("query").addEventListener("keydown", (e) => e.key === "Enter" && search());

async function search() {
  const text = $("text").value, query = $("query").value;
  if (!text.trim() && !query.trim()) { $("text").focus(); return; }
  $("busy").hidden = false;
  try {
    const r = await fetch("/api/papers", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, query }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || "Search failed");
    render(j);
  } catch (e) {
    alert(e.message);
  } finally {
    $("busy").hidden = true;
  }
}

function render(j) {
  $("out").hidden = false;
  $("meta").innerHTML =
    `Searched for: ${j.queries.map((q) => `<span class="kw">${esc(q)}</span>`).join("")}` +
    (j.candidates_considered ? ` · ranked ${j.candidates_considered} candidate papers by meaning` : "");
  const errs = j.errors || [];
  $("errors").innerHTML = j.papers.length
    ? (errs.length ? `<div class="muted tiny" style="margin-bottom:10px">Note: ${esc(errs.join(" · "))}</div>` : "")
    : errs.map((e) => `<div class="warn">${esc(e)}</div>`).join("");
  $("results").innerHTML = j.papers.length ? j.papers.map(card).join("") : `<li class="card muted">No matching papers found. Try typing a specific topic in the search box.</li>`;
  $("out").scrollIntoView({ behavior: "smooth" });
}

function card(p, i) {
  const pct = Math.max(0, Math.round(p.similarity * 100));
  const authors = p.authors.length ? esc(p.authors.slice(0, 4).join(", ")) + (p.authors.length > 4 ? " et al." : "") : "Unknown authors";
  return `<li class="paper">
    <div class="sim"><div class="ring" style="--p:${pct}"><i>${pct}%</i></div><span>similar</span></div>
    <div>
      <h3>${i + 1}. <a href="${esc(p.url)}" target="_blank" rel="noopener">${esc(p.title)}</a></h3>
      <div class="by">${authors} · ${p.year ?? "n.d."}${p.venue ? " · " + esc(p.venue) : ""} · ${p.citations} citations</div>
      ${p.shared_terms.length ? `<div>${p.shared_terms.map((t) => `<span class="tag">${esc(t)}</span>`).join("")}</div>` : ""}
      <p class="abs clamp">${esc(p.abstract)}</p>
      <div class="links">
        <button class="more">Show full abstract</button>
        <a href="${esc(p.url)}" target="_blank" rel="noopener">Open paper ↗</a>
        ${p.pdf ? `<a href="${esc(p.pdf)}" target="_blank" rel="noopener">Free PDF ↗</a>` : ""}
        <span class="muted">via ${esc(p.source)}</span>
      </div>
    </div>
  </li>`;
}
$("results").addEventListener("click", (e) => {
  const b = e.target.closest(".more");
  if (!b) return;
  const abs = b.closest(".paper").querySelector(".abs");
  abs.classList.toggle("clamp");
  b.textContent = abs.classList.contains("clamp") ? "Show full abstract" : "Show less";
});
