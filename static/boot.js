// Loaded in <head> on every page (before the page draws), so the saved theme
// applies without a flash. Kept in a file, not inline, so the strict Content
// Security Policy can block any injected inline script.
try {
  const t = localStorage.getItem("aic_theme");
  if (t === "light" || t === "dark") document.documentElement.dataset.theme = t;
} catch {}
if (location.pathname === "/checker" || location.pathname === "/editor") {
  document.documentElement.dataset.page = location.pathname === "/editor" ? "editor" : "checker";
}

// Only ever link to normal web pages (blocks javascript:, data: and other schemes
// in links that come from outside sources, such as paper search results).
window.safeUrl = function safeUrl(u) {
  try {
    const url = new URL(String(u || ""), location.origin);
    return url.protocol === "https:" || url.protocol === "http:" ? url.href : "#";
  } catch {
    return "#";
  }
};

// Read a server reply safely: a crash page or network hiccup becomes a plain message
// instead of "Unexpected token ... is not valid JSON".
window.readJson = async function readJson(r) {
  try {
    return await r.json();
  } catch {
    return { detail: `The server had a problem (error ${r.status}). Please try again.` };
  }
};
