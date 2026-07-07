#!/usr/bin/env python3
"""Gradio demo for steel surface defect segmentation.

Upload a steel surface image and see predicted defect masks overlaid with
class labels and confidence scores. Defaults to an ONNX checkpoint so it
runs CPU-only (e.g. a Hugging Face Space); pass a .pt path to use PyTorch
instead. If --weights doesn't exist locally (e.g. a fresh Space container),
it's downloaded from the Hugging Face model repo instead.

Example:
    uv run python app/app.py --weights weights/steel_defect_yolo26s_seg_best.onnx
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
HF_MODEL_REPO = "betty0/steel-defect-segmentation"
HF_MODEL_FILE = "steel_defect_yolo26s_seg_best.onnx"


def resolve_weights(weights: Path) -> Path:
    """Return weights as-is if present locally, else download from the HF model repo."""
    if weights.exists():
        return weights
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(HF_MODEL_REPO, HF_MODEL_FILE))


def build_demo(weights: Path, imgsz: int) -> gr.Blocks:
    # task="segment" is required: loading a bare .onnx (as opposed to the
    # original .pt) can't always auto-detect the task and silently falls back
    # to "detect", which drops mask output entirely.
    model = YOLO(str(weights), task="segment")

    def predict(image):
        if image is None:
            return None
        result = model.predict(image, imgsz=imgsz, verbose=False)[0]
        return result.plot(pil=True)

    # Severstal images are 1600x256 (very wide, short strips) -- a
    # side-by-side input/output layout halves the already-tight width, so
    # stack them instead and let each use the full page width.
    with gr.Blocks(title="Steel Defect Segmentation (YOLO26-seg)") as demo:
        gr.Markdown(f"# Steel Defect Segmentation (YOLO26-seg)\n\n{DESCRIPTION}")
        input_image = gr.Image(type="numpy", label="Steel surface image")
        with gr.Row():
            clear_btn = gr.ClearButton(value="Clear")
            submit_btn = gr.Button("Submit", variant="primary")
        output_image = gr.Image(type="pil", label="Predicted defects")
        clear_btn.add([input_image, output_image])
        submit_btn.click(fn=predict, inputs=input_image, outputs=output_image)

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("weights/steel_defect_yolo26s_seg_best.onnx"),
        help="path to .pt or .onnx weights",
    )
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--share", action="store_true", help="create a public Gradio share link")
    parser.add_argument("--server-port", type=int, default=None)
    args = parser.parse_args()

    demo = build_demo(resolve_weights(args.weights), args.imgsz)
    demo.launch(share=args.share, server_port=args.server_port)


if __name__ == "__main__":
    main()
