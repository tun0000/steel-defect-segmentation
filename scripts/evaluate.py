#!/usr/bin/env python3
"""Evaluate a trained YOLO-seg checkpoint and produce a per-class report.

Runs ``model.val()`` to compute overall and per-class mask/box mAP, writes a
markdown report, and saves a handful of prediction overlays (original image +
predicted masks/labels) per class for visual inspection.

Example:
    uv run python scripts/evaluate.py \\
        --weights weights/steel_defect_yolo26s_seg_best.pt --data ~/datasets/severstal/yolo/data.yaml
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import yaml
from ultralytics import YOLO


def sample_images_per_class(labels_dir: Path, num_per_class: int, seed: int) -> dict[int, list[str]]:
    """Return up to num_per_class image stems containing each class id, keyed by class id."""
    by_class: dict[int, list[str]] = defaultdict(list)
    for label_path in sorted(labels_dir.glob("*.txt")):
        if label_path.stat().st_size == 0:
            continue
        classes_present = {int(line.split()[0]) for line in label_path.read_text().splitlines() if line.strip()}
        for class_id in classes_present:
            by_class[class_id].append(label_path.stem)
    rng = random.Random(seed)
    return {c: rng.sample(stems, min(num_per_class, len(stems))) for c, stems in by_class.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True, help="path to a trained .pt checkpoint")
    parser.add_argument("--data", type=Path, required=True, help="path to the dataset's data.yaml")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--out-dir", type=Path, default=Path("reports"), help="dir for eval_results.md and figures/")
    parser.add_argument("--num-per-class", type=int, default=4, help="overlay images to save per class")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_cfg = yaml.safe_load(args.data.expanduser().read_text())
    dataset_root = Path(data_cfg["path"]).expanduser()
    val_images_dir = dataset_root / data_cfg["val"]
    val_labels_dir = dataset_root / "labels" / Path(data_cfg["val"]).name

    model = YOLO(str(args.weights.expanduser()))
    metrics = model.val(data=str(args.data.expanduser()), imgsz=args.imgsz, plots=False, verbose=False)

    lines = [
        "# Evaluation report",
        "",
        f"- weights: `{args.weights}`",
        f"- data: `{args.data}`, imgsz={args.imgsz}",
        "",
        f"**Overall — mask mAP50: {metrics.seg.map50:.4f}, mask mAP50-95: {metrics.seg.map:.4f}** "
        f"(box mAP50: {metrics.box.map50:.4f}, box mAP50-95: {metrics.box.map:.4f})",
        "",
        "| class | images | instances | mask P | mask R | mask mAP50 | mask mAP50-95 | box mAP50 | box mAP50-95 |",
        "|-------|-------:|----------:|-------:|-------:|-----------:|--------------:|----------:|-------------:|",
    ]
    for i, class_id in enumerate(metrics.ap_class_index):
        mask_p, mask_r, mask_ap50, mask_ap = metrics.seg.class_result(i)
        _, _, box_ap50, box_ap = metrics.box.class_result(i)
        lines.append(
            f"| {metrics.names[class_id]} | {metrics.nt_per_image[class_id]} | {metrics.nt_per_class[class_id]} "
            f"| {mask_p:.4f} | {mask_r:.4f} | {mask_ap50:.4f} | {mask_ap:.4f} | {box_ap50:.4f} | {box_ap:.4f} |"
        )
    report = "\n".join(lines) + "\n"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "eval_results.md").write_text(report)
    print(report)

    figures_dir = args.out_dir / "figures"
    samples = sample_images_per_class(val_labels_dir, args.num_per_class, args.seed)
    for class_id, stems in samples.items():
        class_name = data_cfg["names"][class_id]
        class_dir = figures_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            image_path = val_images_dir / f"{stem}.jpg"
            result = model.predict(str(image_path), imgsz=args.imgsz, verbose=False)[0]
            result.plot(pil=True).save(class_dir / f"{stem}.jpg")
    print(f"Saved overlays to {figures_dir}")


if __name__ == "__main__":
    main()
