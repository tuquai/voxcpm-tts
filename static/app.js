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

/* ========== Abort Controller for Cancellation ========== */
let currentAbortController = null;

function cancelGeneration() {
  if (currentAbortController) {
    currentAbortController.abort();
    currentAbortController = null;
  }
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
      <p class="generating-text">正在生成语音，请稍候...</p>
      <div class="stream-progress hidden" id="${containerId}Progress">
        <div class="stream-progress-bar">
          <div class="stream-progress-fill"></div>
        </div>
        <span class="stream-progress-text">已接收 0 个音频片段</span>
      </div>
      <button class="btn-cancel" onclick="cancelGeneration()">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <line x1="18" y1="6" x2="6" y2="18"/>
          <line x1="6" y1="6" x2="18" y2="18"/>
        </svg>
        取消生成
      </button>
    </div>
  `;
}

function updateStreamProgress(containerId, chunkCount) {
  const progress = document.getElementById(containerId + "Progress");
  if (!progress) return;
  progress.classList.remove("hidden");
  const fill = progress.querySelector(".stream-progress-fill");
  const text = progress.querySelector(".stream-progress-text");
  if (fill) fill.style.width = "100%";
  if (text) text.textContent = `已接收 ${chunkCount} 个音频片段`;
}

function showCancelled(containerId) {
  const container = document.getElementById(containerId);
  container.innerHTML = `
    <div class="cancelled-state">
      <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"/>
        <line x1="8" y1="12" x2="16" y2="12"/>
      </svg>
      <p>生成已取消</p>
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

/* ========== Streaming Audio Fetch & Playback ========== */
async function fetchStreamingAudio(url, formData, signal, resultContainerId) {
  const resp = await fetch(url, { method: "POST", body: formData, signal });
  if (!resp.ok) {
    const errData = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(errData.detail || `HTTP ${resp.status}`);
  }

  const sampleRate = parseInt(resp.headers.get("X-Sample-Rate") || "24000", 10);
  const reader = resp.body.getReader();
  const pcmChunks = [];
  let chunkCount = 0;
  const WAV_HEADER_SIZE = 44;
  let headerSkipped = false;
  let headerBuf = new Uint8Array(0);

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    let data = value;
    if (!headerSkipped) {
      const combined = new Uint8Array(headerBuf.length + data.length);
      combined.set(headerBuf);
      combined.set(data, headerBuf.length);
      if (combined.length >= WAV_HEADER_SIZE) {
        data = combined.slice(WAV_HEADER_SIZE);
        headerSkipped = true;
      } else {
        headerBuf = combined;
        continue;
      }
    }

    if (data.length > 0) {
      pcmChunks.push(data);
      chunkCount++;
      updateStreamProgress(resultContainerId, chunkCount);
    }
  }

  const totalBytes = pcmChunks.reduce((s, c) => s + c.length, 0);
  const merged = new Uint8Array(totalBytes);
  let offset = 0;
  for (const c of pcmChunks) {
    merged.set(c, offset);
    offset += c.length;
  }

  return buildWavBlob(merged, sampleRate);
}

function buildWavBlob(pcmInt16Bytes, sampleRate) {
  const numChannels = 1;
  const bitsPerSample = 16;
  const byteRate = sampleRate * numChannels * bitsPerSample / 8;
  const blockAlign = numChannels * bitsPerSample / 8;
  const dataSize = pcmInt16Bytes.length;

  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeStr = (off, str) => { for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i)); };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);
  writeStr(36, "data");
  view.setUint32(40, dataSize, true);

  new Uint8Array(buffer, 44).set(pcmInt16Bytes);
  return new Blob([buffer], { type: "audio/wav" });
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

  cancelGeneration();
  const controller = new AbortController();
  currentAbortController = controller;

  const btn = document.getElementById("btnDesign");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> 生成中...';
  setStatus("loading", "Streaming...");
  showGenerating("designResult");

  try {
    const form = new FormData();
    form.append("text", text);
    form.append("description", description);
    form.append("cfg_value", cfg);
    form.append("inference_timesteps", steps);

    const blob = await fetchStreamingAudio(
      "/api/voice-design/stream", form, controller.signal, "designResult"
    );
    showResult("designResult", blob, "voice_design.wav");
    setStatus("", "Ready");
  } catch (err) {
    if (err.name === "AbortError") {
      showCancelled("designResult");
      setStatus("", "Ready");
    } else {
      showError("designResult", "生成失败: " + err.message);
      setStatus("error", "Error");
    }
  } finally {
    currentAbortController = null;
    btn.disabled = false;
    btn.innerHTML =
      '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg> 生成语音';
  }
}

/* ========== Voice Source Toggle ========== */
let currentVoiceSource = "upload";
let selectedVoiceId = null;

function switchVoiceSource(source) {
  currentVoiceSource = source;
  document.querySelectorAll(".source-btn").forEach((b) => b.classList.remove("active"));
  document.querySelector(`.source-btn[data-source="${source}"]`).classList.add("active");
  document.getElementById("sourceUpload").classList.toggle("hidden", source !== "upload");
  document.getElementById("sourceSaved").classList.toggle("hidden", source !== "saved");
  if (source === "saved") loadSavedVoices();
}

/* ========== Saved Voices ========== */
async function loadSavedVoices() {
  const container = document.getElementById("savedVoicesList");
  try {
    const resp = await fetch("/api/voices");
    const voices = await resp.json();
    if (voices.length === 0) {
      container.innerHTML = `
        <div class="empty-state small-empty">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.3">
            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
            <circle cx="12" cy="7" r="4"/>
          </svg>
          <p>还没有保存的音色，请先在「上传音频」中保存</p>
        </div>`;
      return;
    }
    container.innerHTML = voices
      .map(
        (v) => `
      <div class="voice-card ${selectedVoiceId === v.id ? "selected" : ""}" data-id="${v.id}" onclick="selectVoice('${v.id}')">
        <div class="voice-card-info">
          <span class="voice-card-name">${escapeHtml(v.name)}</span>
          <span class="voice-card-mode">${v.has_text ? "极致克隆" : "基础克隆"}</span>
        </div>
        <button class="btn-icon voice-delete" onclick="event.stopPropagation(); deleteVoice('${v.id}')" title="删除">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      </div>`
      )
      .join("");
  } catch {
    container.innerHTML = '<div class="empty-state small-empty"><p>加载失败</p></div>';
  }
}

function selectVoice(id) {
  selectedVoiceId = selectedVoiceId === id ? null : id;
  document.querySelectorAll(".voice-card").forEach((c) => {
    c.classList.toggle("selected", c.dataset.id === selectedVoiceId);
  });
}

async function deleteVoice(id) {
  if (!confirm("确定删除这个音色？")) return;
  try {
    await fetch(`/api/voices/${id}`, { method: "DELETE" });
    if (selectedVoiceId === id) selectedVoiceId = null;
    loadSavedVoices();
  } catch {
    alert("删除失败");
  }
}

async function handleSaveVoice() {
  const audioFile = cloneAudioInput.files[0];
  if (!audioFile) {
    alert("请先上传参考音频");
    return;
  }
  const name = document.getElementById("saveVoiceName").value.trim();
  if (!name) {
    alert("请输入音色名称");
    document.getElementById("saveVoiceName").focus();
    return;
  }

  const refText = document.getElementById("cloneRefText").value.trim();
  const btn = document.getElementById("btnSaveVoice");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner" style="border-color:rgba(91,127,245,0.3);border-top-color:var(--accent);width:14px;height:14px;border-width:2px"></span> 保存中...';
  setStatus("loading", "Saving...");

  try {
    const form = new FormData();
    form.append("name", name);
    form.append("reference_audio", audioFile);
    form.append("reference_text", refText);
    form.append("auto_transcribe", refText ? "false" : "true");

    const resp = await fetch("/api/voices", { method: "POST", body: form });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    const voice = await resp.json();
    document.getElementById("saveVoiceName").value = "";
    setStatus("", "Ready");
    alert(`音色「${voice.name}」已保存，可在「已保存音色」中无限次使用`);
  } catch (err) {
    alert("保存失败: " + err.message);
    setStatus("error", "Error");
  } finally {
    btn.disabled = false;
    btn.innerHTML =
      '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg> 保存音色';
  }
}

/* ========== Voice Clone ========== */
async function handleClone() {
  const targetText = document.getElementById("cloneTargetText").value.trim();
  const cfg = document.getElementById("cloneCfg").value;
  const steps = document.getElementById("cloneSteps").value;

  if (!targetText) {
    alert("请输入目标文本");
    document.getElementById("cloneTargetText").focus();
    return;
  }

  const usingSaved = currentVoiceSource === "saved";

  if (usingSaved && !selectedVoiceId) {
    alert("请选择一个已保存的音色");
    return;
  }
  if (!usingSaved && !cloneAudioInput.files[0]) {
    alert("请上传参考音频");
    return;
  }

  cancelGeneration();
  const controller = new AbortController();
  currentAbortController = controller;

  const btn = document.getElementById("btnClone");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> 生成中...';
  setStatus("loading", "Cloning...");
  showGenerating("cloneResult");

  try {
    const form = new FormData();
    form.append("target_text", targetText);
    form.append("cfg_value", cfg);
    form.append("inference_timesteps", steps);

    if (usingSaved) {
      form.append("voice_id", selectedVoiceId);
    } else {
      const audioFile = cloneAudioInput.files[0];
      const refText = document.getElementById("cloneRefText").value.trim();
      form.append("reference_audio", audioFile);
      form.append("reference_text", refText);
      form.append("auto_transcribe", "false");
    }

    const blob = await fetchStreamingAudio(
      "/api/voice-clone/stream", form, controller.signal, "cloneResult"
    );
    showResult("cloneResult", blob, "voice_clone.wav");
    setStatus("", "Ready");
  } catch (err) {
    if (err.name === "AbortError") {
      showCancelled("cloneResult");
      setStatus("", "Ready");
    } else {
      showError("cloneResult", "克隆失败: " + err.message);
      setStatus("error", "Error");
    }
  } finally {
    currentAbortController = null;
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
