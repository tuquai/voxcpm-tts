# VoxCPM Studio

A modern web application for text-to-speech generation powered by [VoxCPM2](https://github.com/OpenBMB/VoxCPM).

## Features

- **Voice Design** - Create a brand-new voice from a natural-language description (gender, age, tone, emotion, pace...), no reference audio required.
- **Voice Clone** - Clone any voice from a short reference audio clip. Optionally provide the reference transcript for "Ultimate Cloning" that faithfully reproduces every vocal nuance.
- Built-in ASR auto-transcription for reference audio
- Adjustable CFG guidance and LocDiT flow-matching steps

## Requirements

- Python >= 3.10, < 3.13
- PyTorch >= 2.5.0
- CUDA >= 12.0 (GPU recommended, ~8 GB VRAM for VoxCPM2)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app (model downloads automatically on first use)
python app.py --port 8808

# 3. Open in browser
# http://localhost:8808
```

### Using a local model

```bash
# Download from ModelScope first (optional, for faster loading in China)
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('OpenBMB/VoxCPM2', local_dir='./models/VoxCPM2')"

# Then point to local path
python app.py --model-id ./models/VoxCPM2
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Web UI |
| `POST` | `/api/voice-design` | Voice design (description + text → audio) |
| `POST` | `/api/voice-clone` | Voice cloning (reference audio + target text → audio) |
| `POST` | `/api/asr` | Auto-transcribe audio |
| `GET`  | `/api/health` | Health check |

### Voice Design

```bash
curl -X POST http://localhost:8808/api/voice-design \
  -F "text=Hello, welcome to VoxCPM Studio!" \
  -F "description=A warm young woman with a gentle voice" \
  -F "cfg_value=2.0" \
  -F "inference_timesteps=10" \
  --output voice_design.wav
```

### Voice Clone

```bash
curl -X POST http://localhost:8808/api/voice-clone \
  -F "target_text=This is a cloned voice demo." \
  -F "reference_audio=@path/to/reference.wav" \
  -F "reference_text=The transcript of the reference audio" \
  -F "cfg_value=2.0" \
  -F "inference_timesteps=10" \
  --output voice_clone.wav
```

## Project Structure

```
tts/
├── app.py              # FastAPI backend
├── static/
│   ├── index.html      # Web UI
│   ├── styles.css      # Styles
│   └── app.js          # Frontend logic
├── requirements.txt
└── README.md
```

## License

This project uses VoxCPM2, which is open-sourced under the [Apache-2.0 license](https://github.com/OpenBMB/VoxCPM/blob/main/LICENSE).
