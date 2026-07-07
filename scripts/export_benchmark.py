#!/usr/bin/env python3
"""Export a YOLO-seg checkpoint to ONNX and benchmark inference latency.

Exports the model once, then times GPU (CUDAExecutionProvider) and CPU
(CPUExecutionProvider) inference separately with the same ONNX Runtime
session: a warmup phase discarded from timing, followed by a fixed number of
timed single-image runs, reporting mean/p50/p95 latency in milliseconds.

Example:
    uv run python scripts/export_benchmark.py --weights weights/best.pt --imgsz 1024
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path

import numpy as np

# onnxruntime-gpu's CUDAExecutionProvider needs pip-installed CUDA/cuDNN shared
# libraries (per `ldd libonnxruntime_providers_cuda.so`) that ship inside
# nvidia-* packages but aren't on the default loader search path. Preload
# exactly those by absolute path before importing onnxruntime so this works
# without the caller having to set LD_LIBRARY_PATH themselves. Preloading
# every .so under nvidia/*/lib (e.g. nccl, nvshmem) instead of this exact set
# pulls in unrelated libraries and segfaults on export.
_REQUIRED_CUDA_LIBS = (
    "libcudart.so.13",
    "libcublasLt.so.13",
    "libcublas.so.13",
    "libcurand.so.10",
    "libcufft.so.12",
    "libnvrtc.so.13",
    "libcudnn.so.9",
)
_nvidia_root = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "nvidia"
for _lib_name in _REQUIRED_CUDA_LIBS:
    _matches = list(_nvidia_root.glob(f"*/lib/{_lib_name}"))
    if _matches:
        try:
            ctypes.CDLL(str(_matches[0]), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

import onnxruntime as ort  # noqa: E402
from ultralytics import YOLO  # noqa: E402

DEVICES = {
    "GPU (CUDAExecutionProvider)": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "CPU (CPUExecutionProvider)": ["CPUExecutionProvider"],
}


def time_session(session: ort.InferenceSession, dummy_input: np.ndarray, warmup: int, iters: int) -> dict[str, float]:
    """Run warmup + timed inference, returning mean/p50/p95 latency in ms."""
    input_name = session.get_inputs()[0].name
    for _ in range(warmup):
        session.run(None, {input_name: dummy_input})
    times_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        times_ms.append((time.perf_counter() - t0) * 1000)
    return {
        "mean_ms": float(np.mean(times_ms)),
        "p50_ms": float(np.percentile(times_ms, 50)),
        "p95_ms": float(np.percentile(times_ms, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True, help="path to a trained .pt checkpoint")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=Path("reports"), help="dir for benchmark.md")
    args = parser.parse_args()

    weights = args.weights.expanduser()
    model = YOLO(str(weights))
    onnx_path = model.export(format="onnx", imgsz=args.imgsz, batch=args.batch, simplify=True)
    print(f"Exported ONNX model to {onnx_path}")

    # Only ever try CUDA/CPU here; TensorrtExecutionProvider reports as "available" (the
    # provider .so ships in onnxruntime-gpu) even when the separate TensorRT runtime isn't
    # installed, which would otherwise print a harmless-but-noisy load-failure/fallback warning.
    session = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    dummy_input = np.random.default_rng(42).random(input_meta.shape, dtype=np.float32)

    lines = [
        "# Export & latency benchmark",
        "",
        f"- weights: `{weights}`, onnx: `{onnx_path}`",
        f"- imgsz={args.imgsz}, batch={args.batch}, warmup={args.warmup}, iters={args.iters}",
        f"- available onnxruntime providers: {ort.get_available_providers()}",
        "",
        "| device | mean (ms) | p50 (ms) | p95 (ms) |",
        "|--------|----------:|---------:|---------:|",
    ]
    available = set(ort.get_available_providers())
    for device_name, providers in DEVICES.items():
        if providers[0] not in available:
            print(f"Skipping {device_name}: {providers[0]} not available on this machine")
            continue
        session = ort.InferenceSession(str(onnx_path), providers=providers)
        stats = time_session(session, dummy_input, args.warmup, args.iters)
        lines.append(f"| {device_name} | {stats['mean_ms']:.2f} | {stats['p50_ms']:.2f} | {stats['p95_ms']:.2f} |")
        print(f"{device_name}: mean={stats['mean_ms']:.2f}ms p50={stats['p50_ms']:.2f}ms p95={stats['p95_ms']:.2f}ms")

    report = "\n".join(lines) + "\n"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "benchmark.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
