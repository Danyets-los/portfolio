(() => {
  const data = JSON.parse(document.querySelector("#chart-data").textContent);
  const canvas = document.querySelector("#tradeoff");
  const colors = { "max-frame": "#ffd18a", geometry: "#a6c9ff", appearance: "#83e2cc" };
  const visible = new Set(Object.keys(colors));
  let points = [];

  function draw() {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const scale = window.devicePixelRatio || 1;
    canvas.width = width * scale;
    canvas.height = height * scale;
    const ctx = canvas.getContext("2d");
    ctx.scale(scale, scale);
    const left = 50, right = width - 23, top = 25, bottom = height - 52;
    const xMax = Math.max(1, ...data.map(row => row.p95_ms)) * 1.18;
    const yMax = Math.max(0.25, ...data.map(row => row.mae)) * 1.18;
    const x = value => left + value / xMax * (right - left);
    const y = value => bottom - value / yMax * (bottom - top);
    points = [];
    ctx.font = "11px system-ui";
    ctx.fillStyle = "#acc0c8";
    ctx.strokeStyle = "#2b404b";
    for (let tick = 0; tick <= 4; tick += 1) {
      const py = y(tick * yMax / 4);
      const px = x(tick * xMax / 4);
      ctx.beginPath();
      ctx.moveTo(left, py);
      ctx.lineTo(right, py);
      ctx.stroke();
      ctx.textAlign = "right";
      ctx.fillText((tick * yMax / 4).toFixed(2), left - 7, py + 4);
      ctx.textAlign = "center";
      ctx.fillText((tick * xMax / 4).toFixed(xMax < 10 ? 1 : 0), px, bottom + 17);
    }
    ctx.textAlign = "left";
    ctx.fillText("MAE", 6, 12);
    ctx.textAlign = "center";
    ctx.fillText("p95 processing time (ms per clip)", (left + right) / 2, height - 13);
    const labels = [];
    Object.entries(colors).forEach(([method, color], methodIndex) => {
      if (!visible.has(method)) return;
      const rows = data.filter(row => row.method === method).sort((a, b) => a.fps - b.fps);
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.beginPath();
      rows.forEach((row, index) => {
        if (index === 0) ctx.moveTo(x(row.p95_ms), y(row.mae));
        else ctx.lineTo(x(row.p95_ms), y(row.mae));
      });
      ctx.stroke();
      rows.forEach(row => {
        const px = x(row.p95_ms);
        const py = y(row.mae);
        ctx.beginPath();
        ctx.arc(px, py, 4, 0, Math.PI * 2);
        ctx.fill();
        labels.push({ px, py, color, methodIndex, rate: row.fps });
        points.push({px, py, row});
      });
    });
    ctx.textAlign = "left";
    const placed = [];
    labels.forEach(({ px, py, color, methodIndex, rate }) => {
      const text = rate + " fps";
      const textWidth = ctx.measureText(text).width;
      const tx = Math.min(px + 6, width - textWidth - 4);
      let ty = py - 8 - methodIndex * 10;
      while (placed.some(mark => Math.abs(mark.ty - ty) < 10 && Math.abs(mark.tx - tx) < textWidth + 4)) ty -= 11;
      if (ty < 10) return;
      ctx.fillStyle = color;
      ctx.fillText(text, tx, ty);
      placed.push({ tx, ty });
    });
  }

  document.querySelectorAll("button[data-method]").forEach(button => {
    button.addEventListener("click", () => {
      const method = button.dataset.method;
      if (visible.has(method)) visible.delete(method);
      else visible.add(method);
      button.setAttribute("aria-pressed", visible.has(method));
      draw();
    });
  });
  canvas.addEventListener("pointermove", event => {
    const candidate = points.find(({px, py}) => Math.hypot(px - event.offsetX, py - event.offsetY) < 8);
    const row = candidate?.row;
    canvas.title = row ? `${row.method} · ${row.fps} FPS · MAE ${row.mae.toFixed(2)} · p95 ${row.p95_ms.toFixed(1)} ms` : "";
  });
  new ResizeObserver(draw).observe(canvas);
})();
