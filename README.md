# luxtts-onnx

Lightweight ONNX Runtime inference for [LuxTTS](https://github.com/ysharma3501/LuxTTS) -- no PyTorch required at runtime.

Supports voice cloning with a reference audio clip. Multilingual (English + Chinese).

## Highlights

Compared to the upstream LuxTTS (PyTorch), this package:

- **Zero torch dependency**: Pure ONNX Runtime + numpy + librosa at runtime, saving ~2 GB install size
- **ONNX-compatible Vocos vocoder**: Replaced `torch.fft` ISTFT with real-valued DFT basis matrix (`RealISTFT`), enabling full ONNX export of the dual-path 48 kHz + 24 kHz vocoder
- **Linkwitz-Riley crossover in numpy**: FFT-based 4th-order Butterworth squared filter for merging the dual-path vocoder output, no scipy signal dependency
- **librosa mel extraction**: Matched torchaudio behavior with `norm=None, htk=True` -- the default `norm='slaney'` causes 33x magnitude difference and silent output
- **Pre-computed prompts (.npz)**: Encode reference voice once, save to disk, load instantly on subsequent runs -- skip mel extraction and tokenization on every call
- **Removed Whisper dependency**: User provides transcript directly instead of running ASR on the reference audio
- **GPU on par with PyTorch**: CUDA EP achieves RTF ~0.08x (12x real-time), same speed as torch

## Quick start

### Install

```bash
# CPU only
uv add luxtts-onnx

# GPU (requires CUDA 12 + cuDNN 9)
uv add "luxtts-onnx[gpu]"
```

### Models

Models are automatically downloaded on first use from
[ProgCat/luxtts-onnx](https://huggingface.co/ProgCat/luxtts-onnx) (SHA256
verified per file). No PyTorch or manual setup needed.

Subsequent runs skip the download entirely.

```python
# Auto-download to HuggingFace cache (default)
tts = LuxTTSOnnx()

# Auto-download to a custom directory
tts = LuxTTSOnnx(model_dir="my_models/")
```

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

## Re-exporting vocos

To manually re-export `vocos.onnx` from the PyTorch checkpoint:

```bash
uv add "luxtts-onnx[export]"
uv run python scripts/export_vocos.py --output-dir models/
```

`text_encoder.onnx` and `fm_decoder.onnx` are provided by the upstream
[LuxTTS](https://github.com/ysharma3501/LuxTTS) project and downloaded
automatically.

## Built by

This project was built with [Claude Code](https://claude.ai/claude-code) (Claude Opus 4.6).

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for attribution.

This project is a derivative of [LuxTTS](https://github.com/ysharma3501/LuxTTS)
by Yatharth Sharma, which builds on
[ZipVoice](https://github.com/k2-fsa/ZipVoice) by Xiaomi Corp. and
[Vocos](https://github.com/gemelo-ai/vocos) by Gemelo AI.
