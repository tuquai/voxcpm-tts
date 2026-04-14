import os
import re
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import logging
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import struct

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

app = FastAPI(title="VoxCPM Studio", version="1.0.0")

STATIC_DIR = Path(__file__).parent / "static"
OUTPUT_DIR = Path(tempfile.gettempdir()) / "voxcpm_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VOICES_DIR = Path(__file__).parent / "saved_voices"
VOICES_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ModelManager:
    """Lazy-loads the VoxCPM model and optional ASR model on first use."""

    def __init__(self, model_id: str = "openbmb/VoxCPM2"):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._voxcpm = None
        self._asr = None

    @property
    def voxcpm(self):
        if self._voxcpm is None:
            import voxcpm as vcpm

            logger.info("Loading VoxCPM model: %s (device=%s)", self.model_id, self.device)
            self._voxcpm = vcpm.VoxCPM.from_pretrained(self.model_id, optimize=False)
            logger.info("VoxCPM model loaded.")
        return self._voxcpm

    @property
    def asr(self):
        if self._asr is None:
            from funasr import AutoModel

            logger.info("Loading ASR model: iic/SenseVoiceSmall")
            self._asr = AutoModel(
                model="iic/SenseVoiceSmall",
                disable_update=True,
                log_level="WARNING",
                device="cuda:0" if self.device == "cuda" else "cpu",
            )
            logger.info("ASR model loaded.")
        return self._asr

    @property
    def sample_rate(self) -> int:
        return self.voxcpm.tts_model.sample_rate

    def recognize(self, audio_path: str) -> str:
        res = self.asr.generate(input=audio_path, language="auto", use_itn=True)
        return res[0]["text"].split("|>")[-1]


mgr: Optional[ModelManager] = None


def get_mgr() -> ModelManager:
    global mgr
    if mgr is None:
        model_id = os.environ.get("VOXCPM_MODEL_ID", "openbmb/VoxCPM2")
        mgr = ModelManager(model_id=model_id)
    return mgr


def save_wav(wav: np.ndarray, sr: int) -> Path:
    fname = f"{uuid.uuid4().hex[:12]}.wav"
    out_path = OUTPUT_DIR / fname
    sf.write(str(out_path), wav, sr)
    return out_path


# ------------------------------------------------------------------ #
#  Long-text chunking: split → generate per-chunk → rolling cache
#  Prevents quality degradation (noise, speaker drift) on long text.
# ------------------------------------------------------------------ #

CHUNK_MAX_CHARS = 100
MAX_PROMPT_AUDIO_LEN = 75


def split_text_into_chunks(text: str, max_chars: int = CHUNK_MAX_CHARS) -> list:
    """Split *text* at natural sentence / clause boundaries.

    Returns a list of non-empty strings, each at most *max_chars* long
    (best-effort; a single clause longer than *max_chars* is kept intact).
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    parts = re.split(r"(?<=[。！？!?\n])", text)
    parts = [p for p in parts if p.strip()]

    merged: list = []
    buf = ""
    for p in parts:
        if buf and len(buf) + len(p) > max_chars:
            merged.append(buf)
            buf = p
        else:
            buf += p
    if buf:
        merged.append(buf)

    result: list = []
    for seg in merged:
        if len(seg) <= max_chars:
            result.append(seg)
            continue
        clauses = re.split(r"(?<=[，、；：,.;:—])", seg)
        clauses = [c for c in clauses if c.strip()]
        buf = ""
        for c in clauses:
            if buf and len(buf) + len(c) > max_chars:
                result.append(buf)
                buf = c
            else:
                buf += c
        if buf:
            result.append(buf)

    if len(result) > 1 and len(result[-1]) < 10:
        last = result.pop()
        result[-1] += last

    return result


def _build_prompt_cache_from_ref(m, ref_path: str, ref_text: str):
    """Build a reusable prompt_cache from an uploaded reference audio."""
    kwargs: dict = {"reference_wav_path": ref_path}
    if ref_text:
        kwargs["prompt_text"] = ref_text
        kwargs["prompt_wav_path"] = ref_path
    return m.voxcpm.tts_model.build_prompt_cache(**kwargs)


def _rolling_cache(original_ref, chunk_text: str, audio_feat: torch.Tensor) -> dict:
    """Create a bounded rolling cache: original ref identity + last chunk tail."""
    tail = min(audio_feat.shape[0], MAX_PROMPT_AUDIO_LEN)
    cache: dict = {
        "prompt_text": chunk_text,
        "audio_feat": audio_feat[-tail:],
    }
    if original_ref is not None:
        cache["ref_audio_feat"] = original_ref
        cache["mode"] = "ref_continuation"
    else:
        cache["mode"] = "continuation"
    return cache


def _generate_chunked(
    m, target_text: str, prompt_cache: dict,
    cfg_value: float, inference_timesteps: int,
) -> np.ndarray:
    """Generate audio for *target_text*, auto-chunking long text."""
    chunks = split_text_into_chunks(target_text)
    if not chunks:
        raise ValueError("Empty target text")

    tts = m.voxcpm.tts_model
    original_ref = prompt_cache.get("ref_audio_feat") if prompt_cache else None
    current_cache = prompt_cache
    all_wavs: list = []

    for i, chunk in enumerate(chunks):
        logger.info("[Chunked %d/%d] %s", i + 1, len(chunks), chunk[:60])
        wav, _, audio_feat = tts.generate_with_prompt_cache(
            target_text=chunk,
            prompt_cache=current_cache,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
        )
        all_wavs.append(wav.squeeze(0).cpu().numpy())
        current_cache = _rolling_cache(original_ref, chunk, audio_feat)

    return np.concatenate(all_wavs, axis=-1)


def _generate_chunked_streaming(
    m, target_text: str, prompt_cache: dict,
    cfg_value: float, inference_timesteps: int,
):
    """Yield one numpy waveform per text chunk (chunk-level streaming)."""
    chunks = split_text_into_chunks(target_text)
    if not chunks:
        return

    tts = m.voxcpm.tts_model
    original_ref = prompt_cache.get("ref_audio_feat") if prompt_cache else None
    current_cache = prompt_cache

    for i, chunk in enumerate(chunks):
        logger.info("[Chunked Stream %d/%d] %s", i + 1, len(chunks), chunk[:60])
        wav, _, audio_feat = tts.generate_with_prompt_cache(
            target_text=chunk,
            prompt_cache=current_cache,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
        )
        yield wav.squeeze(0).cpu().numpy()
        current_cache = _rolling_cache(original_ref, chunk, audio_feat)


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.post("/api/voice-design")
async def voice_design(
    text: str = Form(...),
    description: str = Form(""),
    cfg_value: float = Form(2.0),
    inference_timesteps: int = Form(10),
):
    """Voice Design: create a voice from description + synthesize text."""
    text = text.strip()
    if not text:
        raise HTTPException(400, "Target text is required.")

    description = description.strip()
    final_text = f"({description}){text}" if description else text

    logger.info("[Voice Design] description=%s, text=%s", description[:60], text[:60])

    m = get_mgr()
    wav = m.voxcpm.generate(
        text=final_text,
        cfg_value=cfg_value,
        inference_timesteps=inference_timesteps,
    )

    out = save_wav(wav, m.sample_rate)
    return FileResponse(str(out), media_type="audio/wav", filename="voice_design.wav")


def _wav_header(sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """Build a WAV header with sizes set to max uint32 (streaming-friendly)."""
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    max_u32 = 0xFFFFFFFF
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", max_u32, b"WAVE",
        b"fmt ", 16, 1, num_channels,
        sample_rate, byte_rate, block_align, bits_per_sample,
        b"data", max_u32,
    )
    return header


def _float32_to_int16_bytes(chunk: np.ndarray) -> bytes:
    clipped = np.clip(chunk, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


@app.post("/api/voice-design/stream")
async def voice_design_stream(
    text: str = Form(...),
    description: str = Form(""),
    cfg_value: float = Form(2.0),
    inference_timesteps: int = Form(10),
):
    """Streaming Voice Design: yields WAV audio chunks as they are generated."""
    text = text.strip()
    if not text:
        raise HTTPException(400, "Target text is required.")

    description = description.strip()
    final_text = f"({description}){text}" if description else text
    logger.info("[Voice Design Stream] description=%s, text=%s", description[:60], text[:60])

    m = get_mgr()
    sr = m.sample_rate

    def audio_stream():
        yield _wav_header(sr)
        for chunk in m.voxcpm.generate_streaming(
            text=final_text,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
        ):
            yield _float32_to_int16_bytes(chunk)

    return StreamingResponse(
        audio_stream(),
        media_type="audio/wav",
        headers={"X-Sample-Rate": str(sr)},
    )


def _build_and_generate(m, target_text: str, ref_path: str, ref_text: str,
                        cfg_value: float, inference_timesteps: int,
                        prompt_cache: dict = None, streaming: bool = False):
    """Build generation kwargs from either a saved prompt_cache or a reference audio path."""
    if prompt_cache is not None:
        func = m.voxcpm.tts_model.generate_with_prompt_cache_streaming if streaming else m.voxcpm.tts_model.generate_with_prompt_cache
        kwargs = dict(
            target_text=target_text,
            prompt_cache=prompt_cache,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
        )
        if streaming:
            def _stream():
                for wav, _, _ in func(**kwargs):
                    yield wav.squeeze(0).cpu().numpy()
            return _stream()
        else:
            wav, _, _ = func(**kwargs)
            return wav.squeeze(0).cpu().numpy()

    generate_kwargs = dict(
        text=target_text,
        reference_wav_path=ref_path,
        cfg_value=cfg_value,
        inference_timesteps=inference_timesteps,
    )
    if ref_text:
        generate_kwargs["prompt_wav_path"] = ref_path
        generate_kwargs["prompt_text"] = ref_text

    if streaming:
        return m.voxcpm.generate_streaming(**generate_kwargs)
    else:
        return m.voxcpm.generate(**generate_kwargs)


# ------------------------------------------------------------------ #
#  Voice Profile Management
# ------------------------------------------------------------------ #

def _voice_meta_path(voice_id: str) -> Path:
    return VOICES_DIR / voice_id / "meta.json"


def _voice_cache_path(voice_id: str) -> Path:
    return VOICES_DIR / voice_id / "prompt_cache.pt"


@app.post("/api/voices")
async def save_voice(
    name: str = Form(...),
    reference_audio: UploadFile = File(...),
    reference_text: str = Form(""),
    auto_transcribe: bool = Form(False),
):
    """Save a voice profile: encode reference audio once, reuse forever."""
    name = name.strip()
    if not name:
        raise HTTPException(400, "Voice name is required.")

    suffix = Path(reference_audio.filename or "ref.wav").suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(OUTPUT_DIR))
    try:
        content = await reference_audio.read()
        tmp.write(content)
        tmp.close()
        ref_path = tmp.name

        m = get_mgr()
        ref_text = reference_text.strip()

        if not ref_text and auto_transcribe:
            logger.info("[Save Voice] Auto-transcribing reference audio...")
            ref_text = m.recognize(ref_path)
            logger.info("[Save Voice] ASR result: %s", ref_text[:80])

        build_kwargs: dict = dict(reference_wav_path=ref_path)
        if ref_text:
            build_kwargs["prompt_text"] = ref_text
            build_kwargs["prompt_wav_path"] = ref_path

        prompt_cache = m.voxcpm.tts_model.build_prompt_cache(**build_kwargs)

        voice_id = uuid.uuid4().hex[:12]
        voice_dir = VOICES_DIR / voice_id
        voice_dir.mkdir(parents=True, exist_ok=True)

        torch.save(prompt_cache, str(_voice_cache_path(voice_id)))

        meta = {
            "id": voice_id,
            "name": name,
            "has_text": bool(ref_text),
            "reference_text": ref_text,
            "created_at": time.time(),
        }
        _voice_meta_path(voice_id).write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

        logger.info("[Save Voice] Saved voice '%s' (id=%s, has_text=%s)", name, voice_id, bool(ref_text))
        return JSONResponse(meta)

    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@app.get("/api/voices")
async def list_voices():
    """List all saved voice profiles."""
    voices = []
    for d in sorted(VOICES_DIR.iterdir()):
        meta_path = d / "meta.json"
        if meta_path.exists():
            try:
                voices.append(json.loads(meta_path.read_text(encoding="utf-8")))
            except Exception:
                pass
    voices.sort(key=lambda v: v.get("created_at", 0), reverse=True)
    return voices


@app.delete("/api/voices/{voice_id}")
async def delete_voice(voice_id: str):
    """Delete a saved voice profile."""
    voice_dir = VOICES_DIR / voice_id
    if not voice_dir.exists():
        raise HTTPException(404, "Voice not found.")
    import shutil
    shutil.rmtree(voice_dir)
    logger.info("[Delete Voice] Removed voice id=%s", voice_id)
    return {"ok": True}


def _load_prompt_cache(voice_id: str) -> dict:
    cache_path = _voice_cache_path(voice_id)
    if not cache_path.exists():
        raise HTTPException(404, f"Voice profile not found: {voice_id}")
    return torch.load(str(cache_path), map_location="cpu", weights_only=False)


# ------------------------------------------------------------------ #
#  Voice Clone (supports both upload and saved voice_id)
# ------------------------------------------------------------------ #

@app.post("/api/voice-clone")
async def voice_clone(
    target_text: str = Form(...),
    reference_audio: Optional[UploadFile] = File(None),
    reference_text: str = Form(""),
    auto_transcribe: bool = Form(False),
    voice_id: str = Form(""),
    cfg_value: float = Form(2.0),
    inference_timesteps: int = Form(10),
):
    """Voice Clone: clone voice from reference audio or saved voice, speak target_text.

    Long text is automatically split into chunks and generated with a rolling
    prompt cache so that quality and speaker identity stay consistent.
    """
    target_text = target_text.strip()
    if not target_text:
        raise HTTPException(400, "Target text is required.")

    m = get_mgr()
    voice_id = voice_id.strip()

    if voice_id:
        logger.info("[Voice Clone - Saved] voice_id=%s", voice_id)
        prompt_cache = _load_prompt_cache(voice_id)
        wav = _generate_chunked(m, target_text, prompt_cache, cfg_value, inference_timesteps)
        out = save_wav(wav, m.sample_rate)
        return FileResponse(str(out), media_type="audio/wav", filename="voice_clone.wav")

    if reference_audio is None:
        raise HTTPException(400, "Either reference_audio or voice_id is required.")

    suffix = Path(reference_audio.filename or "ref.wav").suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(OUTPUT_DIR))
    try:
        content = await reference_audio.read()
        tmp.write(content)
        tmp.close()
        ref_path = tmp.name

        ref_text = reference_text.strip()

        if not ref_text and auto_transcribe:
            logger.info("[Voice Clone] Auto-transcribing reference audio...")
            ref_text = m.recognize(ref_path)
            logger.info("[Voice Clone] ASR result: %s", ref_text[:80])

        if ref_text:
            logger.info("[Voice Clone - Ultimate] prompt_text=%s", ref_text[:60])
        else:
            logger.info("[Voice Clone - Controllable] reference only")

        prompt_cache = _build_prompt_cache_from_ref(m, ref_path, ref_text)
        wav = _generate_chunked(m, target_text, prompt_cache, cfg_value, inference_timesteps)
        out = save_wav(wav, m.sample_rate)
        return FileResponse(str(out), media_type="audio/wav", filename="voice_clone.wav")

    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@app.post("/api/voice-clone/stream")
async def voice_clone_stream(
    target_text: str = Form(...),
    reference_audio: Optional[UploadFile] = File(None),
    reference_text: str = Form(""),
    auto_transcribe: bool = Form(False),
    voice_id: str = Form(""),
    cfg_value: float = Form(2.0),
    inference_timesteps: int = Form(10),
):
    """Streaming Voice Clone: yields WAV audio chunk-by-chunk for long text."""
    target_text = target_text.strip()
    if not target_text:
        raise HTTPException(400, "Target text is required.")

    m = get_mgr()
    sr = m.sample_rate
    voice_id = voice_id.strip()

    if voice_id:
        logger.info("[Voice Clone Stream - Saved] voice_id=%s", voice_id)
        prompt_cache = _load_prompt_cache(voice_id)

        def audio_stream_cached():
            yield _wav_header(sr)
            for wav_chunk in _generate_chunked_streaming(
                m, target_text, prompt_cache, cfg_value, inference_timesteps,
            ):
                yield _float32_to_int16_bytes(wav_chunk)

        return StreamingResponse(
            audio_stream_cached(),
            media_type="audio/wav",
            headers={"X-Sample-Rate": str(sr)},
        )

    if reference_audio is None:
        raise HTTPException(400, "Either reference_audio or voice_id is required.")

    suffix = Path(reference_audio.filename or "ref.wav").suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(OUTPUT_DIR))
    content = await reference_audio.read()
    tmp.write(content)
    tmp.close()
    ref_path = tmp.name

    ref_text = reference_text.strip()

    if not ref_text and auto_transcribe:
        logger.info("[Voice Clone Stream] Auto-transcribing reference audio...")
        ref_text = m.recognize(ref_path)
        logger.info("[Voice Clone Stream] ASR result: %s", ref_text[:80])

    if ref_text:
        logger.info("[Voice Clone Stream - Ultimate] prompt_text=%s", ref_text[:60])
    else:
        logger.info("[Voice Clone Stream - Controllable] reference only")

    prompt_cache = _build_prompt_cache_from_ref(m, ref_path, ref_text)

    def audio_stream():
        yield _wav_header(sr)
        try:
            for wav_chunk in _generate_chunked_streaming(
                m, target_text, prompt_cache, cfg_value, inference_timesteps,
            ):
                yield _float32_to_int16_bytes(wav_chunk)
        finally:
            try:
                os.unlink(ref_path)
            except OSError:
                pass

    return StreamingResponse(
        audio_stream(),
        media_type="audio/wav",
        headers={"X-Sample-Rate": str(sr)},
    )


@app.post("/api/asr")
async def asr(audio: UploadFile = File(...)):
    """Transcribe an audio file using the ASR model."""
    suffix = Path(audio.filename or "asr.wav").suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(OUTPUT_DIR))
    try:
        content = await audio.read()
        tmp.write(content)
        tmp.close()

        m = get_mgr()
        text = m.recognize(tmp.name)
        return {"text": text}
    except Exception as e:
        logger.error("ASR failed: %s", e)
        raise HTTPException(500, f"ASR recognition failed: {e}")
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "device": get_mgr().device,
        "model_id": get_mgr().model_id,
    }


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="VoxCPM Studio Web App")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8808)
    parser.add_argument("--model-id", type=str, default="openbmb/VoxCPM2")
    args = parser.parse_args()

    os.environ["VOXCPM_MODEL_ID"] = args.model_id
    uvicorn.run(app, host=args.host, port=args.port)
