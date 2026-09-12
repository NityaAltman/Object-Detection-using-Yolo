(() => {
  "use strict";

  const dropzone = document.getElementById("dropzone");
  const canvasWrap = document.getElementById("canvas-wrap");
  const canvasReset = document.getElementById("canvas-reset");
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");
  const fileInput = document.getElementById("file-input");
  const stageError = document.getElementById("stage-error");
  const statusEl = document.getElementById("status");

  const modelSelect = document.getElementById("model-select");
  const modelHint = document.getElementById("model-hint");
  const threshold = document.getElementById("threshold");
  const thresholdValue = document.getElementById("threshold-value");
  const detectionList = document.getElementById("detection-list");
  const detectionCount = document.getElementById("detection-count");
  const countsEl = document.getElementById("counts");
  const metaTime = document.getElementById("meta-time");
  const metaSize = document.getElementById("meta-size");

  const ACCEPTED_TYPES = /^image\/(png|jpeg|webp)$/;

  // A scan is only drawn once the decoded image and its matching response are
  // both in hand. Each upload gets an id so a slow decode belonging to an
  // earlier file can never paint over a newer one.
  const scan = { id: 0, image: null, result: null };
  let currentModel = modelSelect.querySelector(".segmented-option.active").dataset.model;
  let lastFile = null;

  const PALETTE = ["#FF6B35", "#4ADE80", "#60A5FA", "#F0B429", "#C084FC", "#F87171"];
  const colorForClass = (() => {
    const assigned = new Map();
    return (name) => {
      if (!assigned.has(name)) {
        assigned.set(name, PALETTE[assigned.size % PALETTE.length]);
      }
      return assigned.get(name);
    };
  })();

  function setStatus(text, busy = false) {
    statusEl.textContent = text;
    statusEl.classList.toggle("active", busy);
  }

  function showError(message) {
    stageError.textContent = message || "";
    stageError.hidden = !message;
  }

  function resetResults() {
    detectionCount.textContent = "0";
    detectionList.innerHTML = '<p class="empty-note">Nothing scanned yet.</p>';
    countsEl.innerHTML = '<p class="empty-note">Nothing scanned yet.</p>';
    metaTime.textContent = "—";
    metaSize.textContent = "—";
  }

  function showDropzone() {
    scan.id += 1;
    scan.image = null;
    scan.result = null;
    lastFile = null;
    dropzone.hidden = false;
    canvasWrap.hidden = true;
    showError("");
    resetResults();
    setStatus("Ready");
  }

  /* --- Input --- */

  const openFileDialog = () => fileInput.click();

  dropzone.addEventListener("click", openFileDialog);
  canvas.addEventListener("click", openFileDialog);

  canvasReset.addEventListener("click", (event) => {
    event.stopPropagation();
    showDropzone();
  });

  ["dragenter", "dragover"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("drag-over");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("drag-over");
    })
  );
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
    // Allow re-selecting the same file straight after an error.
    fileInput.value = "";
  });

  modelSelect.addEventListener("click", (e) => {
    const btn = e.target.closest(".segmented-option");
    if (!btn || btn.classList.contains("active")) return;

    modelSelect
      .querySelectorAll(".segmented-option")
      .forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    currentModel = btn.dataset.model;

    const unavailable = btn.dataset.available === "false";
    modelHint.textContent = unavailable
      ? `${btn.dataset.label} — not trained yet`
      : btn.dataset.label;

    if (lastFile) runDetection(lastFile);
  });

  threshold.addEventListener("input", () => {
    thresholdValue.textContent = `${threshold.value}%`;
    render();
  });

  window.addEventListener("resize", render);

  function handleFile(file) {
    if (!ACCEPTED_TYPES.test(file.type)) {
      showError("Couldn't read that file. Try a JPG, PNG or WebP.");
      return;
    }
    showError("");
    lastFile = file;
    runDetection(file);
  }

  /* --- Detection --- */

  async function parseResponse(res) {
    // Not every failure comes back as JSON (a proxy or an upload-size refusal
    // may return HTML), so parse defensively and keep the message useful.
    const text = await res.text();
    let data = null;
    try {
      data = JSON.parse(text);
    } catch (_) {
      /* fall through to the status-based message */
    }
    if (!res.ok) {
      throw new Error((data && data.error) || `Request failed (HTTP ${res.status}).`);
    }
    return data;
  }

  function runDetection(file) {
    const id = (scan.id += 1);
    scan.image = null;
    scan.result = null;

    setStatus("Scanning…", true);
    showError("");
    resetResults();
    dropzone.hidden = true;
    canvasWrap.hidden = false;

    const objectUrl = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(objectUrl);
      if (id !== scan.id) return;
      scan.image = img;
      render();
    };
    img.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      if (id !== scan.id) return;
      showError("That image couldn't be decoded by the browser.");
      setStatus("Ready");
    };
    img.src = objectUrl;

    const body = new FormData();
    body.append("image", file);
    body.append("model", currentModel);

    fetch("/api/detect", { method: "POST", body })
      .then(parseResponse)
      .then((data) => {
        if (id !== scan.id) return;
        scan.result = data;
        metaTime.textContent = `${data.inference_ms} ms`;
        metaSize.textContent = `${data.width} × ${data.height}`;
        setStatus("Ready");
        render();
      })
      .catch((err) => {
        if (id !== scan.id) return;
        setStatus("Ready");
        showError(err.message || "Couldn't reach the server.");
      });
  }

  /* --- Rendering --- */

  function drawBase() {
    const rect = canvasWrap.getBoundingClientRect();
    const available = Math.max(rect.width - 32, 80);
    const scale = Math.min(available / scan.image.width, 1);
    canvas.width = Math.round(scan.image.width * scale);
    canvas.height = Math.round(scan.image.height * scale);
    ctx.drawImage(scan.image, 0, 0, canvas.width, canvas.height);
  }

  function render() {
    if (!scan.image || !scan.result) return;

    drawBase();

    const minConfidence = Number(threshold.value) / 100;
    const scaleX = canvas.width / scan.result.width;
    const scaleY = canvas.height / scan.result.height;
    const visible = scan.result.detections.filter((d) => d.confidence >= minConfidence);

    visible.forEach((d) => {
      const [x1, y1, x2, y2] = d.box;
      const color = colorForClass(d.class);
      const bx = x1 * scaleX;
      const by = y1 * scaleY;

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(bx, by, (x2 - x1) * scaleX, (y2 - y1) * scaleY);

      const label = `${d.class} ${Math.round(d.confidence * 100)}%`;
      ctx.font = "500 12px 'IBM Plex Mono', monospace";
      const boxTop = Math.max(by - 18, 0);
      ctx.fillStyle = color;
      ctx.fillRect(bx, boxTop, ctx.measureText(label).width + 10, 18);
      ctx.fillStyle = "#14181F";
      ctx.fillText(label, bx + 5, boxTop + 13);
    });

    renderCounts(visible);
    renderList(visible);
  }

  function renderCounts(visible) {
    detectionCount.textContent = String(visible.length);

    if (visible.length === 0) {
      countsEl.innerHTML = '<p class="empty-note">Nothing above this threshold.</p>';
      return;
    }

    const totals = new Map();
    visible.forEach((d) => totals.set(d.class, (totals.get(d.class) || 0) + 1));

    const rows = [...totals.entries()]
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .map(([name, total]) => {
        const row = document.createElement("div");
        row.className = "count-row";

        const swatch = document.createElement("span");
        swatch.className = "count-swatch";
        swatch.style.background = colorForClass(name);

        const label = document.createElement("span");
        label.className = "count-name";
        label.textContent = name;

        const value = document.createElement("span");
        value.className = "mono";
        value.textContent = String(total);

        row.append(swatch, label, value);
        return row;
      });

    countsEl.replaceChildren(...rows);
  }

  function renderList(visible) {
    if (visible.length === 0) {
      detectionList.innerHTML = '<p class="empty-note">Nothing above this threshold.</p>';
      return;
    }

    detectionList.replaceChildren(
      ...visible.map((d) => {
        const pct = Math.round(d.confidence * 100);
        const color = colorForClass(d.class);

        const row = document.createElement("div");
        row.className = "detection-row";

        const name = document.createElement("span");
        name.className = "detection-class";
        name.textContent = d.class;

        const conf = document.createElement("span");
        conf.className = "detection-confidence";

        const bar = document.createElement("span");
        bar.className = "confidence-bar";
        const fill = document.createElement("span");
        fill.className = "confidence-bar-fill";
        fill.style.width = `${pct}%`;
        fill.style.background = color;
        bar.appendChild(fill);

        const pctEl = document.createElement("span");
        pctEl.className = "mono";
        pctEl.textContent = `${pct}%`;

        conf.append(bar, pctEl);
        row.append(name, conf);
        return row;
      })
    );
  }
})();
