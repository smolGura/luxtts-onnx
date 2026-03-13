# Copyright 2025 smolGura
# Copyright 2024-2025 Yatharth Sharma (LuxTTS)
# Copyright 2024 Xiaomi Corp. (ZipVoice ODE solver, feature extraction)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure numpy + onnxruntime inference pipeline for LuxTTS.

No torch dependency at runtime. Uses librosa for audio loading and mel extraction.
"""
import hashlib
import logging
import shutil
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
from huggingface_hub import snapshot_download

from luxtts_onnx.tokenizer import EmiliaTokenizer

logger = logging.getLogger(__name__)

HF_REPO = "YatharthS/LuxTTS"

# Upstream files: name -> expected SHA256 (from YatharthS/LuxTTS on HuggingFace)
UPSTREAM_FILES = {
    "text_encoder.onnx": "495eca2d5f8a911f5c361bcce5bd55cdd2508ccdd26ce3e9bf1d3c29eb974861",
    "fm_decoder.onnx": "4510d4f5f049f14ef80207fca695e13c820e2cea61635f402954950bc62b1e3c",
    "tokens.txt": "ce98c1afc5f7a20c2484dffdd68a1fff0a4a2cc707328833750c4476c37cdbda",
}


def _sha256(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_models(model_dir: Path) -> None:
    """Ensure all required ONNX models exist and are valid.

    Per-file checks:
    - text_encoder.onnx, fm_decoder.onnx, tokens.txt: verify SHA256 hash
    - vocos.onnx: check existence only (exported locally, hash varies)

    Missing or corrupted files are re-downloaded/re-exported automatically.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    hf_dir = None  # lazy download

    # Check each upstream file: exists + hash match
    for name, expected_hash in UPSTREAM_FILES.items():
        dst = model_dir / name
        need_copy = False
        if not dst.exists():
            logger.info("Missing %s", name)
            need_copy = True
        elif _sha256(dst) != expected_hash:
            logger.warning("Hash mismatch for %s, re-downloading", name)
            need_copy = True

        if need_copy:
            if hf_dir is None:
                logger.info("Downloading upstream models from %s...", HF_REPO)
                hf_dir = Path(snapshot_download(HF_REPO))
            src = hf_dir / name
            if not src.exists():
                raise FileNotFoundError(f"Expected {name} in {hf_dir}")
            shutil.copy2(src, dst)
            logger.info("Installed %s", name)

    # Check vocos.onnx: existence only (exported locally, hash varies by env)
    if not (model_dir / "vocos.onnx").exists():
        logger.info("Missing vocos.onnx, exporting from PyTorch checkpoint...")
        if hf_dir is None:
            hf_dir = Path(snapshot_download(HF_REPO))
        from luxtts_onnx.exporter import export_vocos

        export_vocos(hf_dir, model_dir)
        logger.info("Exported vocos.onnx")

# Mel spectrogram config (matches VocosFbank)
SAMPLE_RATE = 24000
N_MELS = 100
N_FFT = 1024
HOP_LENGTH = 256
FEAT_SCALE = 0.1


def get_time_steps(
    t_start: float = 0.0,
    t_end: float = 1.0,
    num_step: int = 10,
    t_shift: float = 1.0,
) -> np.ndarray:
    """Compute ODE solver time schedule (numpy port)."""
    timesteps = np.linspace(t_start, t_end, num_step + 1, dtype=np.float32)
    timesteps = t_shift * timesteps / (1 + (t_shift - 1) * timesteps)
    return timesteps


def extract_mel_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract log-mel spectrogram features using librosa.

    Matches VocosFbank: power=1 (magnitude), center=True, 100 mels.

    Args:
        audio: 1D float32 audio array
        sr: sample rate (must be 24000)

    Returns:
        features: [1, T, 100] log-mel features scaled by FEAT_SCALE
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        center=True,
        power=1,  # magnitude spectrogram, not power
        norm=None,  # match torchaudio default (librosa defaults to 'slaney')
        htk=True,  # match torchaudio mel scale
    )
    logmel = np.log(np.clip(mel, a_min=1e-7, a_max=None))
    # [n_mels, T] -> [1, T, n_mels]
    features = logmel.T[np.newaxis, :, :]
    return (features * FEAT_SCALE).astype(np.float32)


def rms_norm(audio: np.ndarray, target_rms: float) -> tuple[np.ndarray, float]:
    """Normalize RMS (only scale UP, never down).

    Returns:
        audio: normalized audio
        original_rms: original RMS value for volume matching
    """
    original_rms = float(np.sqrt(np.mean(audio ** 2)))
    if original_rms < target_rms:
        audio = audio * (target_rms / original_rms)
    return audio, original_rms


def crossover_merge(
    audio_48k: np.ndarray,
    audio_24k: np.ndarray,
    crossover_freq: float = 12000.0,
    sr_48k: int = 48000,
    sr_24k: int = 24000,
) -> np.ndarray:
    """Linkwitz-Riley crossover merge of 48kHz and 24kHz paths.

    Uses FFT-based 4th-order Linkwitz-Riley (Butterworth squared) filter.
    48kHz path provides high frequencies, 24kHz path provides low frequencies.
    """
    # Resample 24kHz to 48kHz
    audio_24k_up = librosa.resample(audio_24k, orig_sr=sr_24k, target_sr=sr_48k)

    # Match lengths
    min_len = min(len(audio_48k), len(audio_24k_up))
    audio_48k = audio_48k[:min_len]
    audio_24k_up = audio_24k_up[:min_len]

    # FFT-based Linkwitz-Riley crossover
    n = len(audio_48k)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr_48k)

    # 4th-order Butterworth magnitude response squared = Linkwitz-Riley
    ratio = freqs / crossover_freq
    ratio = np.clip(ratio, 0, 1e6)
    butter_sq = 1.0 / (1.0 + ratio ** 8)  # 4th order squared

    low_gain = np.sqrt(butter_sq)
    high_gain = np.sqrt(1.0 - butter_sq)

    # Apply in frequency domain
    spec_24k = np.fft.rfft(audio_24k_up)
    spec_48k = np.fft.rfft(audio_48k)

    merged = np.fft.irfft(spec_24k * low_gain + spec_48k * high_gain, n=n)
    return merged.astype(np.float32)


class LuxTTSOnnx:
    """Pure ONNX Runtime inference for LuxTTS.

    No torch dependency. Uses numpy for all computation.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        provider: str = "auto",
        num_threads: int = 4,
    ):
        """Initialize LuxTTS ONNX inference.

        Args:
            model_dir: path to directory containing ONNX models and tokens.txt.
                If None, downloads to HuggingFace cache.
            provider: "auto", "cuda", or "cpu"
            num_threads: number of threads for CPU inference
        """
        if model_dir is not None:
            model_dir = Path(model_dir)
        else:
            model_dir = Path(snapshot_download(HF_REPO))

        _ensure_models(model_dir)

        # Resolve provider
        if provider == "auto":
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
        elif provider == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = num_threads
        sess_opts.intra_op_num_threads = num_threads

        # Load ONNX sessions
        te_path = model_dir / "text_encoder.onnx"
        fm_path = model_dir / "fm_decoder.onnx"
        vocos_path = model_dir / "vocos.onnx"

        logger.info("Loading text_encoder from %s", te_path)
        self.text_encoder = ort.InferenceSession(
            str(te_path), sess_options=sess_opts, providers=providers,
        )

        logger.info("Loading fm_decoder from %s", fm_path)
        self.fm_decoder = ort.InferenceSession(
            str(fm_path), sess_options=sess_opts, providers=providers,
        )

        logger.info("Loading vocos from %s", vocos_path)
        self.vocos = ort.InferenceSession(
            str(vocos_path), sess_options=sess_opts, providers=providers,
        )

        # Get feat_dim from fm_decoder metadata
        meta = self.fm_decoder.get_modelmeta().custom_metadata_map
        self.feat_dim = int(meta["feat_dim"])

        # Load tokenizer
        token_file = model_dir / "tokens.txt"
        logger.info("Loading tokenizer from %s", token_file)
        self.tokenizer = EmiliaTokenizer(token_file=str(token_file))

        logger.info("LuxTTSOnnx initialized (providers=%s, feat_dim=%d)",
                     providers, self.feat_dim)

    def encode_prompt(
        self,
        audio_path: str,
        transcript: str,
        duration: float = 15.0,
        target_rms: float = 0.01,
    ) -> dict:
        """Encode a reference audio prompt.

        Args:
            audio_path: path to reference audio file
            transcript: transcript of the reference audio
            duration: max duration in seconds to use from reference
            target_rms: target RMS for normalization

        Returns:
            dict with prompt_tokens, prompt_features, prompt_features_len, prompt_rms
        """
        # Load and resample to 24kHz
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, duration=duration)

        # RMS normalization
        audio, prompt_rms = rms_norm(audio, target_rms)

        # Extract mel features
        features = extract_mel_features(audio)  # [1, T, 100]

        # Tokenize transcript
        prompt_tokens = self.tokenizer.texts_to_token_ids([transcript])

        return {
            "prompt_tokens": prompt_tokens,
            "prompt_features": features,
            "prompt_features_len": np.array(features.shape[1], dtype=np.int64),
            "prompt_rms": prompt_rms,
        }

    def save_prompt(self, prompt: dict, path: str) -> None:
        """Save pre-computed prompt to .npz file."""
        # Convert prompt_tokens (list of list of int) to padded array
        tokens = prompt["prompt_tokens"][0]
        np.savez(
            path,
            prompt_tokens=np.array(tokens, dtype=np.int64),
            prompt_features=prompt["prompt_features"],
            prompt_features_len=prompt["prompt_features_len"],
            prompt_rms=np.array(prompt["prompt_rms"], dtype=np.float32),
        )
        logger.info("Saved prompt to %s", path)

    def load_prompt(self, path: str) -> dict:
        """Load pre-computed prompt from .npz file."""
        data = np.load(path, allow_pickle=False)
        return {
            "prompt_tokens": [data["prompt_tokens"].tolist()],
            "prompt_features": data["prompt_features"],
            "prompt_features_len": data["prompt_features_len"],
            "prompt_rms": float(data["prompt_rms"]),
        }

    def generate(
        self,
        text: str,
        prompt: dict,
        num_steps: int = 8,
        t_shift: float = 0.9,
        guidance_scale: float = 3.0,
        speed: float = 1.0,
        target_rms: float = 0.1,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate speech from text using a pre-computed prompt.

        Args:
            text: text to synthesize
            prompt: dict from encode_prompt() or load_prompt()
            num_steps: ODE solver steps (8 is good default)
            t_shift: time schedule shift (0.9 recommended)
            guidance_scale: classifier-free guidance scale (3.0 recommended)
            speed: speech speed multiplier (1.0 = default, internally * 1.3)
            target_rms: target RMS for output volume matching
            seed: random seed for reproducibility

        Returns:
            audio: float32 array at 48kHz sample rate
        """
        if seed is not None:
            np.random.seed(seed)

        # Tokenize input text
        tokens = self.tokenizer.texts_to_token_ids([text])
        tokens_np = np.array(tokens, dtype=np.int64)
        prompt_tokens_np = np.array(prompt["prompt_tokens"], dtype=np.int64)
        prompt_features = prompt["prompt_features"]  # [1, T_prompt, 100]
        prompt_features_len = prompt["prompt_features_len"]
        prompt_rms = prompt["prompt_rms"]

        # Speed adjustment (original code multiplies by 1.3)
        speed_np = np.array(speed * 1.3, dtype=np.float32)

        # Run text encoder
        te_inputs = self.text_encoder.get_inputs()
        te_outputs = self.text_encoder.get_outputs()
        text_condition = self.text_encoder.run(
            [te_outputs[0].name],
            {
                te_inputs[0].name: tokens_np,
                te_inputs[1].name: prompt_tokens_np,
                te_inputs[2].name: prompt_features_len,
                te_inputs[3].name: speed_np,
            },
        )[0]  # [1, num_frames, dim]

        batch_size, num_frames, _ = text_condition.shape

        # Time schedule
        timesteps = get_time_steps(
            t_start=0.0, t_end=1.0, num_step=num_steps, t_shift=t_shift,
        )

        # Initialize x with noise
        x = np.random.randn(batch_size, num_frames, self.feat_dim).astype(np.float32)

        # Pad or truncate prompt features to match num_frames
        prompt_t = prompt_features.shape[1]
        if prompt_t <= num_frames:
            pad_width = num_frames - prompt_t
            speech_condition = np.pad(
                prompt_features, ((0, 0), (0, pad_width), (0, 0)), mode="constant",
            ).astype(np.float32)
        else:
            speech_condition = prompt_features[:, :num_frames, :].astype(np.float32)

        guidance_np = np.array(guidance_scale, dtype=np.float32)

        # ODE sampling loop
        fm_inputs = self.fm_decoder.get_inputs()
        fm_outputs = self.fm_decoder.get_outputs()

        for step in range(num_steps):
            t_cur = np.array(timesteps[step], dtype=np.float32)
            t_next = timesteps[step + 1]

            # Predict velocity
            v = self.fm_decoder.run(
                [fm_outputs[0].name],
                {
                    fm_inputs[0].name: t_cur,
                    fm_inputs[1].name: x,
                    fm_inputs[2].name: text_condition,
                    fm_inputs[3].name: speech_condition,
                    fm_inputs[4].name: guidance_np,
                },
            )[0]

            # Anchor-based ODE update
            x_1_pred = x + (1.0 - float(t_cur)) * v
            x_0_pred = x - float(t_cur) * v

            if step < num_steps - 1:
                x = (1.0 - t_next) * x_0_pred + t_next * x_1_pred
            else:
                x = x_1_pred

        # Remove prompt portion
        prompt_len = int(prompt_features_len)
        x = x[:, prompt_len:, :]  # [1, gen_frames, feat_dim]

        # Convert features to vocos input: [B, feat_dim, T] / scale
        vocos_input = (x.transpose(0, 2, 1) / FEAT_SCALE).astype(np.float32)

        # Run vocos vocoder
        vocos_inputs = self.vocos.get_inputs()
        vocos_outputs = self.vocos.get_outputs()
        audio_48k, audio_24k = self.vocos.run(
            [vocos_outputs[0].name, vocos_outputs[1].name],
            {vocos_inputs[0].name: vocos_input},
        )

        # Crossover merge
        audio_48k = audio_48k[0]  # [samples]
        audio_24k = audio_24k[0]  # [samples]
        merged = crossover_merge(audio_48k, audio_24k)

        # Clamp
        merged = np.clip(merged, -1.0, 1.0)

        # Volume matching
        if prompt_rms < target_rms:
            merged = merged * (prompt_rms / target_rms)

        return merged

    def generate_to_file(
        self,
        text: str,
        prompt: dict,
        output_path: str,
        **kwargs,
    ) -> str:
        """Generate speech and save to file.

        Returns:
            output_path
        """
        audio = self.generate(text, prompt, **kwargs)
        sf.write(output_path, audio, 48000)
        logger.info("Saved audio to %s (%.2fs)", output_path, len(audio) / 48000)
        return output_path
