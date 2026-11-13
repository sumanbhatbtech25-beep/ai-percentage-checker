// Shared by every page: highlight the current nav link and add the light/dark switch.
document.querySelectorAll(".nav a").forEach((a) => {
  a.classList.toggle("on", a.getAttribute("href") === location.pathname);
});

(function themeSwitch() {
  const root = document.documentElement;
  const systemDark = () => window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  const current = () => root.dataset.theme || (systemDark() ? "dark" : "light");

  const btn = document.createElement("button");
  btn.className = "theme-btn";
  btn.type = "button";
  const paint = () => {
    const dark = current() === "dark";
    btn.textContent = dark ? "☀" : "☾";
    btn.title = dark ? "Switch to light mode" : "Switch to dark mode";
    btn.setAttribute("aria-label", btn.title);
  };
  btn.addEventListener("click", () => {
    const next = current() === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    try { localStorage.setItem("aic_theme", next); } catch {}
    paint();
  });
  paint();
  const bar = document.querySelector(".topbar");
  const status = bar && bar.querySelector(".status");
  if (bar) status ? bar.insertBefore(btn, status) : bar.appendChild(btn);
  // follow system changes while the user hasn't chosen a theme
  window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", paint);
})();

// Phone mode: a logged-in phone gets a "Log out" link (this computer never needs one).
fetch("/api/health").then((r) => r.json()).then((h) => {
  if (!h.remote) return;
  const a = document.createElement("a");
  a.href = "/logout";
  a.textContent = "Log out";
  document.querySelector(".nav")?.appendChild(a);
}).catch(() => {});
