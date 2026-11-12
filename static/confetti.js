// Party-cracker confetti: two bursts fired from the bottom corners.
// No libraries; one temporary full-screen canvas that removes itself.
window.partyCrackers = function partyCrackers() {
  if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const canvas = document.createElement("canvas");
  canvas.className = "confetti";
  document.body.appendChild(canvas);
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const resize = () => {
    canvas.width = innerWidth * dpr;
    canvas.height = innerHeight * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  };
  resize();
  addEventListener("resize", resize);

  const colors = ["#ff4d6d", "#ffd23f", "#3bceac", "#4d96ff", "#9b5de5", "#ff8c42", "#ffffff"];
  const pieces = [];
  const burst = (x, y, dir, delay) => {
    const n = innerWidth < 600 ? 90 : 160;
    for (let i = 0; i < n; i++) {
      const angle = (-Math.PI / 2) + dir * (0.15 + Math.random() * 0.75); // up and inward
      const speed = 9 + Math.random() * 11;
      pieces.push({
        x, y, delay,
        vx: Math.cos(angle) * speed * (0.6 + Math.random() * 0.6),
        vy: Math.sin(angle) * speed,
        w: 5 + Math.random() * 6, h: 8 + Math.random() * 8,
        rot: Math.random() * Math.PI, vr: (Math.random() - 0.5) * 0.35,
        color: colors[(Math.random() * colors.length) | 0],
        shape: Math.random() < 0.25 ? "circle" : "rect",
        wobble: Math.random() * 10,
      });
    }
  };
  burst(0, innerHeight, 1, 0);              // left cracker
  burst(innerWidth, innerHeight, -1, 0);    // right cracker
  burst(0, innerHeight, 1, 22);             // second pop
  burst(innerWidth, innerHeight, -1, 22);

  let frame = 0;
  const tick = () => {
    frame++;
    ctx.clearRect(0, 0, innerWidth, innerHeight);
    let alive = 0;
    for (const p of pieces) {
      if (frame < p.delay) { alive++; continue; }
      p.vy += 0.32;            // gravity
      p.vx *= 0.985;           // air drag
      p.vy *= 0.985;
      p.x += p.vx + Math.sin((frame + p.wobble) / 8) * 0.6;
      p.y += p.vy;
      p.rot += p.vr;
      if (p.y > innerHeight + 40) continue;
      alive++;
      const fade = Math.max(0, Math.min(1, (260 - frame) / 60));
      ctx.save();
      ctx.globalAlpha = fade;
      ctx.translate(p.x, p.y);
      ctx.rotate(p.rot);
      ctx.fillStyle = p.color;
      if (p.shape === "circle") { ctx.beginPath(); ctx.arc(0, 0, p.w / 2, 0, Math.PI * 2); ctx.fill(); }
      else ctx.fillRect(-p.w / 2, -p.h / 2, p.w, p.h * Math.abs(Math.cos(p.rot * 2)) + 2);
      ctx.restore();
    }
    if (alive && frame < 260) requestAnimationFrame(tick);
    else { removeEventListener("resize", resize); canvas.remove(); }
  };
  requestAnimationFrame(tick);
};
