#!/usr/bin/env python3
"""Gradio demo for steel surface defect segmentation.

Upload a steel surface image and see predicted defect masks overlaid with
class labels and confidence scores. Defaults to an ONNX checkpoint so it
runs CPU-only (e.g. a Hugging Face Space); pass a .pt path to use PyTorch
instead.

Example:
    uv run python app/app.py --weights weights/best.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
from ultralytics import YOLO

DESCRIPTION = (
    "Upload a steel surface image to detect and segment surface defects "
    "(YOLO26-seg trained on the Severstal Steel Defect Detection dataset)."
)


def build_demo(weights: Path, imgsz: int) -> gr.Interface:
    model = YOLO(str(weights))

    def predict(image):
        if image is None:
            return None
        result = model.predict(image, imgsz=imgsz, verbose=False)[0]
        return result.plot(pil=True)

    return gr.Interface(
        fn=predict,
        inputs=gr.Image(type="numpy", label="Steel surface image"),
        outputs=gr.Image(type="pil", label="Predicted defects"),
        title="Steel Defect Segmentation (YOLO26-seg)",
        description=DESCRIPTION,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=Path("weights/best.onnx"), help="path to .pt or .onnx weights")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--share", action="store_true", help="create a public Gradio share link")
    parser.add_argument("--server-port", type=int, default=None)
    args = parser.parse_args()

    demo = build_demo(args.weights, args.imgsz)
    demo.launch(share=args.share, server_port=args.server_port)


if __name__ == "__main__":
    main()
