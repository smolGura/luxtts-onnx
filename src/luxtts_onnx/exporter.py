# Copyright 2025 smolGura
# Copyright 2024-2025 Yatharth Sharma (LuxTTS)
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

"""Export Vocos vocoder to ONNX.

Requires the ``[export]`` optional dependencies (torch, linacodec, vocos).
All heavy imports are deferred so this module can be imported without torch
installed -- the import error surfaces only when ``export_vocos`` is called.
"""

import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)


def export_vocos(hf_dir: Path, output_dir: Path, opset: int = 18) -> Path:
    """Export Vocos vocoder from PyTorch checkpoint to ONNX.

    Args:
        hf_dir: Path to downloaded HuggingFace repo containing
            ``vocoder/config.yaml`` and ``vocoder/vocos.bin``.
        output_dir: Directory to write ``vocos.onnx`` into.
        opset: ONNX opset version.

    Returns:
        Path to the exported ``vocos.onnx``.
    """
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.nn.utils import parametrize

    from linacodec.vocoder.vocos import Vocos

    # -- Helper classes -------------------------------------------------------

    class RealISTFT(nn.Module):
        """ONNX-compatible ISTFT using real-valued DFT basis matrices."""

        def __init__(self, n_fft: int, hop_length: int, win_length: int):
            super().__init__()
            self.n_fft = n_fft
            self.hop_length = hop_length
            self.win_length = win_length

            n_bins = n_fft // 2 + 1
            n = torch.arange(n_fft).float().unsqueeze(1)
            k = torch.arange(n_bins).float().unsqueeze(0)
            angles = 2.0 * math.pi * k * n / n_fft

            cos_basis = torch.cos(angles) / n_fft
            sin_basis = torch.sin(angles) / n_fft

            scale = torch.ones(1, n_bins)
            scale[0, 1:-1] = 2.0
            cos_basis = cos_basis * scale
            sin_basis = sin_basis * scale

            self.register_buffer("cos_basis", cos_basis)
            self.register_buffer("sin_basis", sin_basis)

            window = torch.hann_window(win_length)
            self.register_buffer("window", window)

        def forward(self, spec_real: torch.Tensor, spec_imag: torch.Tensor) -> torch.Tensor:
            B, K, T = spec_real.shape
            ifft = torch.matmul(self.cos_basis, spec_real) - torch.matmul(
                self.sin_basis, spec_imag
            )
            ifft = ifft * self.window[:, None]

            pad = (self.win_length - self.hop_length) // 2
            identity = torch.eye(self.n_fft, device=ifft.device, dtype=ifft.dtype).unsqueeze(1)
            y = torch.nn.functional.conv_transpose1d(ifft, identity, stride=self.hop_length)[
                :, 0, pad:-pad
            ]

            win_sq = (self.window**2).view(1, self.n_fft, 1).expand(1, -1, T)
            window_envelope = torch.nn.functional.conv_transpose1d(
                win_sq, identity, stride=self.hop_length
            )[0, 0, pad:-pad]

            y = y / (window_envelope + 1e-11)
            return y

    class RealISTFTHead(nn.Module):
        """ONNX-compatible replacement for ISTFTHead."""

        def __init__(self, original_head):
            super().__init__()
            self.out = original_head.out
            istft = original_head.istft
            self.istft = RealISTFT(
                n_fft=istft.n_fft,
                hop_length=istft.hop_length,
                win_length=istft.win_length,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.out(x).transpose(1, 2)
            mag, p = x.chunk(2, dim=1)
            mag = torch.exp(mag)
            mag = torch.clamp(mag, max=1e2)
            spec_real = mag * torch.cos(p)
            spec_imag = mag * torch.sin(p)
            audio = self.istft(spec_real, spec_imag)
            return audio.unsqueeze(1)

    class VocosExportWrapper(nn.Module):
        """Wrapper that outputs both 48kHz and 24kHz audio paths."""

        def __init__(self, vocos: Vocos):
            super().__init__()
            self.backbone = vocos.backbone
            self.upsampler = vocos.upsampler
            self.head_48k = RealISTFTHead(vocos.head_48k)
            self.head = RealISTFTHead(vocos.head)

        def forward(
            self, features_input: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            features = self.backbone(features_input).transpose(1, 2)
            upsampled = self.upsampler(features).transpose(1, 2)
            audio_48k = self.head_48k(upsampled).squeeze(1)
            audio_24k = self.head(features.transpose(1, 2)).squeeze(1)
            return audio_48k, audio_24k

    def _remove_weight_norm(module: nn.Module) -> None:
        for name, child in module.named_children():
            if hasattr(child, "parametrizations"):
                try:
                    parametrize.remove_parametrizations(child, "weight")
                except Exception:
                    pass
            _remove_weight_norm(child)

    # -- Export ---------------------------------------------------------------

    vocos_config = hf_dir / "vocoder" / "config.yaml"
    vocos_weights = hf_dir / "vocoder" / "vocos.bin"

    if not vocos_config.exists():
        raise FileNotFoundError(f"vocoder/config.yaml not found in {hf_dir}")
    if not vocos_weights.exists():
        raise FileNotFoundError(f"vocoder/vocos.bin not found in {hf_dir}")

    logger.info("Loading Vocos from %s", vocos_config)
    vocos = Vocos.from_hparams(str(vocos_config))
    _remove_weight_norm(vocos)
    state_dict = torch.load(str(vocos_weights), map_location="cpu", weights_only=True)
    vocos.load_state_dict(state_dict)

    wrapper = VocosExportWrapper(vocos)
    wrapper.eval()

    dummy_input = torch.randn(1, 100, 50)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "vocos.onnx"

    logger.info("Exporting vocos.onnx to %s", onnx_path)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy_input)
    torch.onnx.export(
        traced,
        dummy_input,
        str(onnx_path),
        opset_version=opset,
        dynamo=False,
        input_names=["features"],
        output_names=["audio_48k", "audio_24k"],
        dynamic_axes={
            "features": {0: "batch", 2: "time"},
            "audio_48k": {0: "batch", 1: "samples"},
            "audio_24k": {0: "batch", 1: "samples"},
        },
    )

    size_mb = onnx_path.stat().st_size / 1024 / 1024
    logger.info("Exported vocos.onnx (%.1f MB)", size_mb)

    # Quick sanity check with onnxruntime
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_48k, onnx_24k = sess.run(None, {"features": dummy_input.numpy()})

    with torch.no_grad():
        ref_48k, ref_24k = wrapper(dummy_input)
    min_48 = min(ref_48k.shape[1], onnx_48k.shape[1])
    diff = np.abs(ref_48k.numpy()[:, :min_48] - onnx_48k[:, :min_48]).max()
    logger.info("Numerical verification: max_diff=%.6f", diff)

    return onnx_path
