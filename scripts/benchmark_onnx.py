"""Benchmark luxtts-onnx on CPU.

Run with: uv run python scripts/benchmark_onnx.py
"""
import time

import numpy as np
import soundfile as sf

TEXT_EN = "Hello! I am Gura, your AI assistant. How can I help you today?"
TEXT_ZH = "你好！我是Gura，你的AI助手。今天有什么可以帮你的吗？"
NUM_RUNS = 3


def benchmark(text: str, label: str, tts, prompt):
    print(f"\n--- {label} ---")
    # Warmup
    _ = tts.generate(
        "This is a warmup sentence for the model.",
        prompt, num_steps=2, seed=0,
    )

    times = []
    durations = []
    for i in range(NUM_RUNS):
        t0 = time.time()
        audio = tts.generate(
            text, prompt,
            num_steps=8, t_shift=0.9, guidance_scale=3.0, seed=42,
        )
        elapsed = time.time() - t0
        dur = len(audio) / 48000
        times.append(elapsed)
        durations.append(dur)
        print(f"  Run {i+1}: {elapsed:.2f}s -> {dur:.2f}s audio (RTF={elapsed/dur:.2f}x)")

    avg_time = np.mean(times)
    avg_dur = np.mean(durations)
    print(f"  Average: {avg_time:.2f}s -> {avg_dur:.2f}s audio (RTF={avg_time/avg_dur:.2f}x)")
    return avg_time, avg_dur


def main():
    from luxtts_onnx.inference import LuxTTSOnnx

    print("=" * 60)
    print("luxtts-onnx (numpy + onnxruntime, CPU, 4 threads)")
    print("=" * 60)

    t0 = time.time()
    tts = LuxTTSOnnx(model_dir="models", provider="cpu", num_threads=4)
    print(f"Init: {time.time()-t0:.2f}s")

    prompt = tts.load_prompt("models/gura_prompt.npz")

    en_time, en_dur = benchmark(TEXT_EN, "English", tts, prompt)
    zh_time, zh_dur = benchmark(TEXT_ZH, "Chinese", tts, prompt)

    print(f"\nONNX EN: {en_time:.2f}s gen -> {en_dur:.2f}s audio (RTF={en_time/en_dur:.2f}x)")
    print(f"ONNX ZH: {zh_time:.2f}s gen -> {zh_dur:.2f}s audio (RTF={zh_time/zh_dur:.2f}x)")


if __name__ == "__main__":
    main()
