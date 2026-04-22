/**
 * app.js — PneumoScan  (Grad-CAM + LIME dual XAI)
 */

// ── Element refs ─────────────────────────────────────────────
const dropzone    = document.getElementById("dropzone");
const fileInput   = document.getElementById("file-input");
const browseBtn   = document.getElementById("browse-btn");
const previewBox  = document.getElementById("preview-box");
const previewImg  = document.getElementById("preview-img");
const previewMeta = document.getElementById("preview-meta");
const analyzeBtn  = document.getElementById("analyze-btn");
const resetBtn    = document.getElementById("reset-btn");

const resultsPanel  = document.getElementById("results-panel");
const loadingState  = document.getElementById("loading-state");
const resultContent = document.getElementById("result-content");

const resultVerdict = document.getElementById("result-verdict");
const resultMeta    = document.getElementById("result-meta");
const origImg       = document.getElementById("orig-img");
const gradcamImg    = document.getElementById("gradcam-img");
const limeImg       = document.getElementById("lime-img");
const xaiCombined   = document.getElementById("xai-combined-text");
const xaiLimeText   = document.getElementById("xai-lime-text");
const probList      = document.getElementById("prob-list");
const diseaseCard   = document.getElementById("disease-card");
const newScanBtn    = document.getElementById("new-scan-btn");
const downloadBtn   = document.getElementById("download-btn");

let selectedFile = null;
let lastResult   = null;

// ── Drag & Drop ──────────────────────────────────────────────
dropzone.addEventListener("dragover",  e => { e.preventDefault(); dropzone.classList.add("drag-over"); });
dropzone.addEventListener("dragleave", ()  => dropzone.classList.remove("drag-over"));
dropzone.addEventListener("drop",      e  => {
  e.preventDefault(); dropzone.classList.remove("drag-over");
  if (e.dataTransfer.files[0]) handleFileSelect(e.dataTransfer.files[0]);
});
dropzone.addEventListener("click", e => { if (e.target !== browseBtn) fileInput.click(); });
browseBtn.addEventListener("click", e => { e.stopPropagation(); fileInput.click(); });
fileInput.addEventListener("change", () => { if (fileInput.files[0]) handleFileSelect(fileInput.files[0]); });

// ── XAI Tabs ─────────────────────────────────────────────────
document.querySelectorAll(".xai-tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".xai-tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".xai-tab-content").forEach(c => c.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById(`tab-${tab.dataset.tab}`).classList.add("active");
  });
});

// ── File select ───────────────────────────────────────────────
function handleFileSelect(file) {
  const allowed = /\.(png|jpg|jpeg|tiff|bmp|webp)$/i;
  if (!file.name.match(allowed)) { showToast("Unsupported file type.", "error"); return; }
  if (file.size > 16 * 1024 * 1024) { showToast("File too large (max 16 MB).", "error"); return; }
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = ev => {
    previewImg.src = ev.target.result;
    previewMeta.textContent = `${file.name}  ·  ${(file.size / 1024).toFixed(1)} KB`;
    dropzone.classList.add("hidden");
    previewBox.classList.remove("hidden");
  };
  reader.readAsDataURL(file);
}

// ── Reset ────────────────────────────────────────────────────
function reset() {
  selectedFile = lastResult = null;
  fileInput.value = "";
  previewBox.classList.add("hidden");
  dropzone.classList.remove("hidden");
  resultsPanel.classList.add("hidden");
  loadingState.classList.remove("hidden");
  resultContent.classList.add("hidden");
  analyzeBtn.disabled = false;
  analyzeBtn.querySelector(".btn-text").textContent = "Analyse X-Ray";
}
resetBtn?.addEventListener("click", reset);
newScanBtn?.addEventListener("click", reset);

// ── Analyse ───────────────────────────────────────────────────
analyzeBtn?.addEventListener("click", async () => {
  if (!selectedFile) return;
  analyzeBtn.disabled = true;
  analyzeBtn.querySelector(".btn-text").textContent = "Analysing…";

  resultsPanel.classList.remove("hidden");
  loadingState.classList.remove("hidden");
  resultContent.classList.add("hidden");
  resultsPanel.scrollIntoView({ behavior: "smooth", block: "start" });

  const formData = new FormData();
  formData.append("xray", selectedFile);

  try {
    const res  = await fetch("/api/predict", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok || data.error) { showToast(data.error || "Server error.", "error"); reset(); return; }
    lastResult = data;
    renderResults(data);
  } catch (err) {
    showToast("Network error. Is the Flask server running?", "error"); reset();
  }
});

// ── Render results ────────────────────────────────────────────
function renderResults(data) {
  origImg.src    = data.original_url;
  gradcamImg.src = data.gradcam_url;
  limeImg.src    = data.lime_url;

  const pct = (data.top_prob * 100).toFixed(1);
  resultVerdict.textContent = data.top_label === "No Finding"
    ? "No Pathology Detected"
    : `Primary Finding: ${data.top_label}`;
  resultMeta.textContent =
    `Confidence ${pct}%  ·  ${data.inference_ms} ms  ·  LIME R² ${data.lime_r2.toFixed(3)}  ·  ${data.lime_num_segs} superpixels`;

  // Combined XAI text
  xaiCombined.textContent = data.xai_summary || "XAI summary not available.";

  // LIME tab text
  xaiLimeText.textContent =
    `LIME divided the X-ray into ${data.lime_num_segs} superpixels and ran 500 perturbation ` +
    `samples. A surrogate Ridge regression model was fitted (R² = ${data.lime_r2.toFixed(3)}). ` +
    `Green regions positively influenced the ${data.top_label} prediction; red regions opposed it. ` +
    `The top 10 superpixels by absolute coefficient magnitude are highlighted.`;

  // Probability bars
  probList.innerHTML = "";
  data.predictions.forEach(p => {
    const pctVal = (p.probability * 100).toFixed(1);
    const isDanger = p.severity === "high" || p.severity === "critical";
    probList.innerHTML += `
      <div class="prob-item">
        <div class="prob-row">
          <span class="prob-name">${p.label}</span>
          <span class="prob-pct">${pctVal}%</span>
        </div>
        <div class="prob-bar-bg">
          <div class="prob-bar-fill ${isDanger ? "danger" : ""}"
               style="width:0%" data-target="${pctVal}%"></div>
        </div>
      </div>`;
  });
  requestAnimationFrame(() => {
    document.querySelectorAll(".prob-bar-fill").forEach(b => b.style.width = b.dataset.target);
  });

  // Disease card
  if (data.predictions[0]) {
    const p = data.predictions[0];
    diseaseCard.innerHTML = `<strong>${p.label}</strong><br>${p.desc}
      <br><span class="severity-badge sev-${p.severity}">${p.severity} severity</span>`;
  }

  loadingState.classList.add("hidden");
  resultContent.classList.remove("hidden");

  downloadBtn.onclick = () => generateReport(data);
}

// ── Report download ───────────────────────────────────────────
function generateReport(data) {
  const lines = [
    "PNEUMOSCAN — AI LUNG DIAGNOSIS REPORT",
    "======================================",
    `Date: ${new Date().toLocaleString()}`,
    "",
    `PRIMARY FINDING : ${data.top_label}`,
    `CONFIDENCE      : ${(data.top_prob * 100).toFixed(2)}%`,
    `INFERENCE TIME  : ${data.inference_ms} ms`,
    `LIME SURROGATE R²: ${data.lime_r2.toFixed(4)}`,
    `LIME SUPERPIXELS : ${data.lime_num_segs}`,
    "",
    "TOP PREDICTIONS",
    "---------------",
    ...data.predictions.map(p =>
      `  ${p.label.padEnd(22)} ${(p.probability*100).toFixed(2)}%  [${p.severity} severity]`
    ),
    "",
    "XAI SUMMARY",
    "-----------",
    data.xai_summary,
    "",
    "DISCLAIMER",
    "----------",
    "For educational/research use only. Not a substitute for medical diagnosis.",
  ];
  const blob = new Blob([lines.join("\n")], { type: "text/plain" });
  const a    = document.createElement("a");
  a.href     = URL.createObjectURL(blob);
  a.download = "pneumoscan_report.txt";
  a.click();
}

// ── Toast ─────────────────────────────────────────────────────
function showToast(msg, type = "info") {
  const t = document.createElement("div");
  t.textContent = msg;
  t.style.cssText = `
    position:fixed;bottom:2rem;right:2rem;z-index:999;
    background:${type==="error"?"#ef4444":"#00dcc0"};
    color:${type==="error"?"#fff":"#000"};
    padding:.8rem 1.4rem;border-radius:10px;
    font-family:'DM Sans',sans-serif;font-size:.88rem;font-weight:600;
    box-shadow:0 4px 20px rgba(0,0,0,.4);animation:fade-in-up .3s ease;`;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 4000);
}
