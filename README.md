# VoxCPM Studio

A modern web application for text-to-speech generation powered by [VoxCPM2](https://github.com/OpenBMB/VoxCPM).

## Demo

> **Original** vs **AI-Cloned** — hear the difference:

**Original Voice**

<video src="examples/original.mp4" controls preload="metadata" width="400"></video>

**Cloned Voice (AI)**

<video src="examples/cloned.mp4" controls preload="metadata" width="400"></video>

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

## Known Issues

### `IndexError: Dimension out of range` during model warm-up (PyTorch >= 2.11)

The voxcpm library creates a 1D attention mask in `voxcpm/modules/minicpm4/model.py`, which is incompatible with `torch.nn.functional.scaled_dot_product_attention(..., enable_gqa=True)` in newer PyTorch versions. The GQA code path requires at least a 2D mask.

**Symptom:**

```
IndexError: Dimension out of range (expected to be in range of [-1, 0], but got -2)
```

**Fix:** In `venv/lib/python3.10/site-packages/voxcpm/modules/minicpm4/model.py`, find:

```python
attn_mask = torch.arange(key_cache.size(2), device=key_cache.device) <= position_id
```

Replace with:

```python
attn_mask = (torch.arange(key_cache.size(2), device=key_cache.device) <= position_id).unsqueeze(0)
```

This adds `.unsqueeze(0)` to reshape the mask from 1D `(seq_len,)` to 2D `(1, seq_len)`, which broadcasts correctly. Note: this patch will be overwritten if you reinstall the voxcpm package.

## License

This project uses VoxCPM2, which is open-sourced under the [Apache-2.0 license](https://github.com/OpenBMB/VoxCPM/blob/main/LICENSE).
