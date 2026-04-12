import os
import sys
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
            self._voxcpm = vcpm.VoxCPM.from_pretrained(self.model_id, optimize=True)
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


@app.post("/api/voice-clone")
async def voice_clone(
    target_text: str = Form(...),
    reference_audio: UploadFile = File(...),
    reference_text: str = Form(""),
    auto_transcribe: bool = Form(False),
    cfg_value: float = Form(2.0),
    inference_timesteps: int = Form(10),
):
    """Voice Clone: clone voice from reference audio, speak target_text."""
    target_text = target_text.strip()
    if not target_text:
        raise HTTPException(400, "Target text is required.")

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
            logger.info("[Voice Clone] Auto-transcribing reference audio...")
            ref_text = m.recognize(ref_path)
            logger.info("[Voice Clone] ASR result: %s", ref_text[:80])

        generate_kwargs = dict(
            text=target_text,
            reference_wav_path=ref_path,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
        )

        if ref_text:
            logger.info("[Voice Clone - Ultimate] prompt_text=%s", ref_text[:60])
            generate_kwargs["prompt_wav_path"] = ref_path
            generate_kwargs["prompt_text"] = ref_text
        else:
            logger.info("[Voice Clone - Controllable] reference only")

        wav = m.voxcpm.generate(**generate_kwargs)
        out = save_wav(wav, m.sample_rate)
        return FileResponse(str(out), media_type="audio/wav", filename="voice_clone.wav")

    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


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
