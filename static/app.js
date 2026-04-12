/* ========== Tab Switching ========== */
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("panel-" + btn.dataset.tab).classList.add("active");
  });
});

/* ========== Slider value sync ========== */
const sliders = [
  ["designCfg", "designCfgVal"],
  ["designSteps", "designStepsVal"],
  ["cloneCfg", "cloneCfgVal"],
  ["cloneSteps", "cloneStepsVal"],
];
sliders.forEach(([sliderId, valId]) => {
  const slider = document.getElementById(sliderId);
  const valSpan = document.getElementById(valId);
  slider.addEventListener("input", () => {
    valSpan.textContent = slider.value;
  });
});

/* ========== File Upload ========== */
const cloneAudioInput = document.getElementById("cloneAudio");
const uploadZone = document.getElementById("uploadZone");
const uploadPlaceholder = document.getElementById("uploadPlaceholder");
const uploadPreview = document.getElementById("uploadPreview");
const audioFileName = document.getElementById("audioFileName");
const refAudioPlayer = document.getElementById("refAudioPlayer");

cloneAudioInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) showAudioPreview(file);
});

uploadZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadZone.classList.add("dragover");
});

uploadZone.addEventListener("dragleave", () => {
  uploadZone.classList.remove("dragover");
});

uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("audio/")) {
    const dt = new DataTransfer();
    dt.items.add(file);
    cloneAudioInput.files = dt.files;
    showAudioPreview(file);
  }
});

function showAudioPreview(file) {
  audioFileName.textContent = file.name;
  refAudioPlayer.src = URL.createObjectURL(file);
  uploadPlaceholder.style.display = "none";
  uploadPreview.classList.remove("hidden");
  cloneAudioInput.style.display = "none";
}

function clearAudio() {
  cloneAudioInput.value = "";
  refAudioPlayer.src = "";
  uploadPlaceholder.style.display = "";
  uploadPreview.classList.add("hidden");
  cloneAudioInput.style.display = "";
}

/* ========== Status Badge ========== */
function setStatus(state, text) {
  const badge = document.getElementById("statusBadge");
  badge.className = "header-badge " + (state || "");
  badge.querySelector(".status-text").textContent = text;
}

/* ========== Result Rendering ========== */
function showGenerating(containerId) {
  const container = document.getElementById(containerId);
  container.innerHTML = `
    <div class="generating-state">
      <div class="generating-spinner"></div>
      <p>正在生成语音，请稍候...</p>
    </div>
  `;
}

function showError(containerId, message) {
  const container = document.getElementById(containerId);
  container.innerHTML = `
    <div class="error-state">
      <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"/>
        <line x1="15" y1="9" x2="9" y2="15"/>
        <line x1="9" y1="9" x2="15" y2="15"/>
      </svg>
      <p>${escapeHtml(message)}</p>
    </div>
  `;
}

function showResult(containerId, audioBlob, filename) {
  const container = document.getElementById(containerId);
  const url = URL.createObjectURL(audioBlob);
  container.innerHTML = `
    <div class="result-player">
      <audio controls autoplay src="${url}"></audio>
      <div class="result-meta">
        <span>${filename}</span>
        <a class="result-download" href="${url}" download="${filename}">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="7 10 12 15 17 10"/>
            <line x1="12" y1="15" x2="12" y2="3"/>
          </svg>
          下载音频
        </a>
      </div>
    </div>
  `;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

/* ========== Voice Design ========== */
async function handleDesign() {
  const text = document.getElementById("designText").value.trim();
  const description = document.getElementById("designDesc").value.trim();
  const cfg = document.getElementById("designCfg").value;
  const steps = document.getElementById("designSteps").value;

  if (!text) {
    alert("请输入目标文本");
    document.getElementById("designText").focus();
    return;
  }

  const btn = document.getElementById("btnDesign");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> 生成中...';
  setStatus("loading", "Generating...");
  showGenerating("designResult");

  try {
    const form = new FormData();
    form.append("text", text);
    form.append("description", description);
    form.append("cfg_value", cfg);
    form.append("inference_timesteps", steps);

    const resp = await fetch("/api/voice-design", { method: "POST", body: form });

    if (!resp.ok) {
      const errData = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(errData.detail || `HTTP ${resp.status}`);
    }

    const blob = await resp.blob();
    showResult("designResult", blob, "voice_design.wav");
    setStatus("", "Ready");
  } catch (err) {
    showError("designResult", "生成失败: " + err.message);
    setStatus("error", "Error");
  } finally {
    btn.disabled = false;
    btn.innerHTML =
      '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg> 生成语音';
  }
}

/* ========== Voice Clone ========== */
async function handleClone() {
  const targetText = document.getElementById("cloneTargetText").value.trim();
  const refText = document.getElementById("cloneRefText").value.trim();
  const audioFile = cloneAudioInput.files[0];
  const cfg = document.getElementById("cloneCfg").value;
  const steps = document.getElementById("cloneSteps").value;

  if (!audioFile) {
    alert("请上传参考音频");
    return;
  }
  if (!targetText) {
    alert("请输入目标文本");
    document.getElementById("cloneTargetText").focus();
    return;
  }

  const btn = document.getElementById("btnClone");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> 生成中...';
  setStatus("loading", "Cloning...");
  showGenerating("cloneResult");

  try {
    const form = new FormData();
    form.append("target_text", targetText);
    form.append("reference_audio", audioFile);
    form.append("reference_text", refText);
    form.append("auto_transcribe", refText ? "false" : "false");
    form.append("cfg_value", cfg);
    form.append("inference_timesteps", steps);

    const resp = await fetch("/api/voice-clone", { method: "POST", body: form });

    if (!resp.ok) {
      const errData = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(errData.detail || `HTTP ${resp.status}`);
    }

    const blob = await resp.blob();
    showResult("cloneResult", blob, "voice_clone.wav");
    setStatus("", "Ready");
  } catch (err) {
    showError("cloneResult", "克隆失败: " + err.message);
    setStatus("error", "Error");
  } finally {
    btn.disabled = false;
    btn.innerHTML =
      '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg> 生成克隆语音';
  }
}

/* ========== ASR (Auto Transcribe) ========== */
async function handleASR() {
  const audioFile = cloneAudioInput.files[0];
  if (!audioFile) {
    alert("请先上传参考音频");
    return;
  }

  const btn = document.getElementById("btnASR");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner" style="border-color:rgba(91,127,245,0.3);border-top-color:var(--accent);width:14px;height:14px;border-width:2px"></span> 识别中';
  setStatus("loading", "ASR...");

  try {
    const form = new FormData();
    form.append("target_text", "__asr_only__");
    form.append("reference_audio", audioFile);
    form.append("reference_text", "");
    form.append("auto_transcribe", "true");

    const resp = await fetch("/api/voice-clone", { method: "POST", body: form });

    if (!resp.ok) {
      const errData = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(errData.detail || `HTTP ${resp.status}`);
    }

    // ASR-only mode doesn't exist as separate endpoint yet,
    // so we'll use a separate ASR endpoint
    throw new Error("ASR endpoint pending");
  } catch {
    // Fallback: use the dedicated ASR endpoint
    try {
      const form = new FormData();
      form.append("audio", audioFile);
      const resp = await fetch("/api/asr", { method: "POST", body: form });
      if (!resp.ok) throw new Error("ASR failed");
      const data = await resp.json();
      document.getElementById("cloneRefText").value = data.text || "";
      setStatus("", "Ready");
    } catch (err2) {
      alert("语音识别失败: " + err2.message);
      setStatus("error", "ASR Error");
    }
  } finally {
    btn.disabled = false;
    btn.innerHTML =
      '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/></svg> 识别';
  }
}

/* ========== Design Examples ========== */
function fillDesignExample(name, desc, text) {
  document.getElementById("designDesc").value = desc;
  document.getElementById("designText").value = text;
  document.getElementById("designDesc").focus();
}
