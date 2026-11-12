// Highlight the current page in the top navigation.
document.querySelectorAll(".nav a").forEach((a) => {
  a.classList.toggle("on", a.getAttribute("href") === location.pathname);
});
