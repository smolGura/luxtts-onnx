# luxtts-onnx

Lightweight ONNX Runtime inference for [LuxTTS](https://github.com/ysharma3501/LuxTTS) -- no PyTorch required at runtime.

Supports voice cloning with a reference audio clip. Multilingual (English + Chinese).

## Features

- Pure ONNX Runtime + numpy inference (no torch dependency)
- CPU and GPU (CUDA) support
- Pre-computed prompt for instant startup
- 48 kHz output with dual-path Vocos vocoder

## Quick start

### Install

```bash
# CPU only
uv add luxtts-onnx

# GPU (requires CUDA 12 + cuDNN 9)
uv add "luxtts-onnx[gpu]"
```

### Download models

Models are hosted on HuggingFace. The library auto-downloads on first use, or
you can download manually:

```bash
# Auto-download (default)
uv run python -c "from luxtts_onnx import LuxTTSOnnx; LuxTTSOnnx()"

# Manual download to local directory
uv run huggingface-cli download YatharthS/LuxTTS --local-dir models/
```

Required model files: `text_encoder.onnx`, `fm_decoder.onnx`, `vocos.onnx`,
`tokens.txt`.

### Generate speech

```python
from luxtts_onnx import LuxTTSOnnx

tts = LuxTTSOnnx(model_dir="models/", provider="auto")

# Encode a reference voice (once, then save for reuse)
prompt = tts.encode_prompt(
    audio_path="reference.wav",
    transcript="The transcript of the reference audio.",
    duration=15.0,
)
tts.save_prompt(prompt, "my_voice.npz")

# Load and generate
prompt = tts.load_prompt("my_voice.npz")
tts.generate_to_file(
    "Hello world! This is a test of voice cloning.",
    prompt,
    output_path="output.wav",
    num_steps=8,
    t_shift=0.9,
    guidance_scale=3.0,
)
```

### GPU setup

For CUDA acceleration, install the `gpu` extra and set `LD_LIBRARY_PATH` so that
ONNX Runtime can find cuDNN:

```bash
uv add "luxtts-onnx[gpu]"

# Point to uv-installed NVIDIA libraries
export LD_LIBRARY_PATH=$(uv run python -c "
import os, glob, site
paths = []
for sp in site.getsitepackages():
    paths.extend(glob.glob(os.path.join(sp, 'nvidia', '*', 'lib')))
print(':'.join(paths))
"):$LD_LIBRARY_PATH
```

Then pass `provider="cuda"`:

```python
tts = LuxTTSOnnx(model_dir="models/", provider="cuda")
```

## Benchmarks

Tested on NVIDIA GPU with 8-step ODE, reference audio 15 s:

| Configuration | RTF | Notes |
|---|---|---|
| CPU (4 threads) | ~3.7x | Same as PyTorch |
| GPU CUDA FP32 | ~0.08x | 12x real-time |

RTF = Real-Time Factor (lower is faster; < 1.0 means faster than real-time).

## Re-exporting models

To re-export ONNX models from the PyTorch checkpoint (requires `export` extras):

```bash
uv add "luxtts-onnx[export]"
uv run python scripts/export_vocos.py --output-dir models/
```

`text_encoder.onnx` and `fm_decoder.onnx` are exported by the upstream LuxTTS
project. See the [LuxTTS repo](https://github.com/ysharma3501/LuxTTS) for
details.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for attribution.

This project is a derivative of [LuxTTS](https://github.com/ysharma3501/LuxTTS)
by Yatharth Sharma, which builds on
[ZipVoice](https://github.com/k2-fsa/ZipVoice) by Xiaomi Corp. and
[Vocos](https://github.com/gemelo-ai/vocos) by Gemelo AI.
