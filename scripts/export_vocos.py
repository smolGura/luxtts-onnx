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

"""Export Vocos vocoder components to ONNX.

Strategy: Export the neural network parts as a single ONNX model that outputs
both 48kHz and 24kHz path audio. Resample + crossover merge done in numpy.

The ISTFTHead uses complex numbers and torch.fft which ONNX doesn't support.
We replace it with a real-valued implementation using precomputed DFT basis matrices.

Requires: pip install -e ".[export]"
"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parametrize
from huggingface_hub import snapshot_download

from linacodec.vocoder.vocos import Vocos


class RealISTFT(nn.Module):
    """ONNX-compatible ISTFT using real-valued DFT basis matrices.

    Replaces torch.fft.irfft + complex number operations with matmul.
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int, padding: str = "same"):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.padding = padding

        # Precompute inverse DFT basis (real part): cos_basis[n, k] = cos(2*pi*k*n/N)
        # For irfft: N real outputs from N/2+1 complex bins
        n_bins = n_fft // 2 + 1
        n = torch.arange(n_fft).float().unsqueeze(1)  # [N, 1]
        k = torch.arange(n_bins).float().unsqueeze(0)  # [1, K]
        angles = 2.0 * math.pi * k * n / n_fft  # [N, K]

        cos_basis = torch.cos(angles) / n_fft  # [N, K]
        sin_basis = torch.sin(angles) / n_fft  # [N, K]

        # DC and Nyquist bins appear once, others appear twice
        scale = torch.ones(1, n_bins)
        scale[0, 1:-1] = 2.0
        cos_basis = cos_basis * scale
        sin_basis = sin_basis * scale

        self.register_buffer("cos_basis", cos_basis)  # [N, K]
        self.register_buffer("sin_basis", sin_basis)  # [N, K]

        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spec_real: torch.Tensor, spec_imag: torch.Tensor) -> torch.Tensor:
        """ISTFT from separate real and imaginary parts.

        Args:
            spec_real: [B, N/2+1, T] real part of STFT
            spec_imag: [B, N/2+1, T] imaginary part of STFT

        Returns:
            audio: [B, L] reconstructed waveform
        """
        B, K, T = spec_real.shape

        # irfft via matmul: output[n] = sum_k(real[k]*cos[n,k] - imag[k]*sin[n,k])
        # [B, N, T] = [N, K] @ [B, K, T]
        ifft = torch.matmul(self.cos_basis, spec_real) - torch.matmul(self.sin_basis, spec_imag)

        # Apply window
        ifft = ifft * self.window[:, None]  # [B, N, T]

        # Overlap-add using conv_transpose1d (ONNX-friendly, no fold)
        pad = (self.win_length - self.hop_length) // 2
        # Identity kernel: [N, 1, N] - each input channel maps to one position
        identity = torch.eye(self.n_fft, device=ifft.device, dtype=ifft.dtype).unsqueeze(1)
        y = torch.nn.functional.conv_transpose1d(
            ifft, identity, stride=self.hop_length
        )[:, 0, pad:-pad]

        # Window envelope via same conv_transpose1d
        win_sq = (self.window ** 2).view(1, self.n_fft, 1).expand(1, -1, T)
        window_envelope = torch.nn.functional.conv_transpose1d(
            win_sq, identity, stride=self.hop_length
        )[0, 0, pad:-pad]

        y = y / (window_envelope + 1e-11)

        return y


class RealISTFTHead(nn.Module):
    """ONNX-compatible replacement for ISTFTHead.

    Replaces complex spec creation with real/imag decomposition.
    """

    def __init__(self, original_head):
        super().__init__()
        self.out = original_head.out
        istft = original_head.istft
        self.istft = RealISTFT(
            n_fft=istft.n_fft,
            hop_length=istft.hop_length,
            win_length=istft.win_length,
            padding=istft.padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clamp(mag, max=1e2)

        # Real/imag decomposition instead of complex: mag * (cos(p) + j*sin(p))
        spec_real = mag * torch.cos(p)  # [B, K, T]
        spec_imag = mag * torch.sin(p)  # [B, K, T]

        audio = self.istft(spec_real, spec_imag)
        return audio.unsqueeze(1)  # [B, 1, L]


class VocosExportWrapper(nn.Module):
    """Wrapper that outputs both audio paths for numpy post-processing.

    Input:  features [B, 100, T]  (mel features)
    Output: audio_48k [B, samples_48k]  (48kHz head)
            audio_24k [B, samples_24k]  (24kHz head)
    """

    def __init__(self, vocos: Vocos):
        super().__init__()
        self.backbone = vocos.backbone
        self.upsampler = vocos.upsampler
        self.head_48k = RealISTFTHead(vocos.head_48k)
        self.head = RealISTFTHead(vocos.head)

    def forward(self, features_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(features_input).transpose(1, 2)

        # 48kHz path
        upsampled = self.upsampler(features).transpose(1, 2)
        audio_48k = self.head_48k(upsampled).squeeze(1)

        # 24kHz path
        audio_24k = self.head(features.transpose(1, 2)).squeeze(1)

        return audio_48k, audio_24k


def remove_weight_norm_recursive(module: nn.Module):
    """Remove weight_norm parametrization from all submodules."""
    for name, child in module.named_children():
        if hasattr(child, "parametrizations"):
            try:
                parametrize.remove_parametrizations(child, "weight")
                print(f"  Removed weight_norm from {name}")
            except Exception:
                pass
        remove_weight_norm_recursive(child)


def load_vocos(model_dir: str | None) -> Vocos:
    """Load Vocos model from local dir or HuggingFace."""
    if model_dir is None:
        model_dir = snapshot_download("YatharthS/LuxTTS")

    vocos_config = Path(model_dir) / "vocoder" / "config.yaml"
    vocos_weights = Path(model_dir) / "vocoder" / "vocos.bin"

    print(f"[export] Loading Vocos from {vocos_config}")
    vocos = Vocos.from_hparams(str(vocos_config))

    # Remove weight_norm before loading (saved weights are raw, not parametrized)
    remove_weight_norm_recursive(vocos)

    state_dict = torch.load(str(vocos_weights), map_location="cpu", weights_only=True)
    vocos.load_state_dict(state_dict)
    vocos.eval()
    return vocos


def export_onnx(wrapper: nn.Module, dummy: torch.Tensor, path: Path, opset: int):
    """Export a traced model to ONNX."""
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy)
    torch.onnx.export(
        traced,
        dummy,
        str(path),
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


def verify_numerics(vocos: Vocos, wrapper: VocosExportWrapper, onnx_path: Path):
    """Verify ONNX output matches PyTorch output."""
    import onnxruntime as ort

    test_input = torch.randn(1, 100, 50)

    # PyTorch reference (original vocos)
    with torch.no_grad():
        vocos.return_48k = True
        vocos.freq_range = 12000
        ref_features = vocos.backbone(test_input).transpose(1, 2)
        ref_upsampled = vocos.upsampler(ref_features).transpose(1, 2)
        ref_48k = vocos.head_48k(ref_upsampled).squeeze(1)
        ref_24k = vocos.head(ref_features.transpose(1, 2)).squeeze(1)

    # Our wrapper
    with torch.no_grad():
        our_48k, our_24k = wrapper(test_input)

    # Compare over shared length
    min_48k = min(ref_48k.shape[1], our_48k.shape[1])
    min_24k = min(ref_24k.shape[1], our_24k.shape[1])
    diff_48k = (ref_48k[:, :min_48k] - our_48k[:, :min_48k]).abs().max().item()
    diff_24k = (ref_24k[:, :min_24k] - our_24k[:, :min_24k]).abs().max().item()
    print(f"  PyTorch vs Wrapper: 48k max_diff={diff_48k:.6f} (len {ref_48k.shape[1]} vs {our_48k.shape[1]})")
    print(f"                      24k max_diff={diff_24k:.6f} (len {ref_24k.shape[1]} vs {our_24k.shape[1]})")

    # ONNX
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_48k, onnx_24k = sess.run(None, {"features": test_input.numpy()})
    min_o48 = min(ref_48k.shape[1], onnx_48k.shape[1])
    min_o24 = min(ref_24k.shape[1], onnx_24k.shape[1])
    diff_onnx_48k = np.abs(ref_48k.numpy()[:, :min_o48] - onnx_48k[:, :min_o48]).max()
    diff_onnx_24k = np.abs(ref_24k.numpy()[:, :min_o24] - onnx_24k[:, :min_o24]).max()
    print(f"  PyTorch vs ONNX:    48k max_diff={diff_onnx_48k:.6f}, 24k max_diff={diff_onnx_24k:.6f}")


def export(model_dir: str | None, output_dir: Path, opset: int = 18):
    output_dir.mkdir(parents=True, exist_ok=True)

    vocos = load_vocos(model_dir)

    wrapper = VocosExportWrapper(vocos)
    wrapper.eval()

    # Dummy input: [batch=1, mel_bins=100, time_frames=50]
    dummy_input = torch.randn(1, 100, 50)

    # Export fp32
    fp32_path = output_dir / "vocos.onnx"
    print(f"[export] Exporting fp32 -> {fp32_path}")
    export_onnx(wrapper, dummy_input, fp32_path, opset)

    # Verify numerics
    print("\n[verify] Checking numerical accuracy...")
    verify_numerics(vocos, wrapper, fp32_path)

    # Export fp16
    fp16_path = output_dir / "vocos_fp16.onnx"
    print(f"\n[export] Exporting fp16 -> {fp16_path}")
    wrapper_fp16 = wrapper.half()
    dummy_fp16 = dummy_input.half()
    export_onnx(wrapper_fp16, dummy_fp16, fp16_path, opset)

    # Int8 quantization
    from onnxruntime.quantization import quantize_dynamic, QuantType

    int8_path = output_dir / "vocos_int8.onnx"
    print(f"[export] Quantizing int8 -> {int8_path}")
    quantize_dynamic(
        str(fp32_path),
        str(int8_path),
        weight_type=QuantType.QInt8,
    )

    # Summary
    print("\n[export] Done!")
    for f in sorted(output_dir.glob("vocos*.onnx")):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=None, help="Local model dir (default: download)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "models",
    )
    args = parser.parse_args()
    export(args.model_dir, args.output_dir)
