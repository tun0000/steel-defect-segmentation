#!/usr/bin/env python3
"""Convert Severstal Steel Defect Detection annotations to YOLO-seg format.

Pipeline:
1. Parse train.csv (handles both the original ``ImageId_ClassId`` layout and the
   cleaned ``ImageId,ClassId,EncodedPixels`` layout). RLE is column-major
   (top-to-bottom, then left-to-right).
2. Verify EVERY annotation round-trips: decode -> re-encode must reproduce the
   original RLE string exactly, otherwise abort.
3. Split each (image, class) mask into instances via connected components;
   fragments smaller than --min-area are dropped (and counted).
4. Convert each instance to one YOLO-seg polygon using the outer contour only.
   Instances containing holes are counted and reported, not carved out.
5. Stratified train/val split keyed on the set of defect classes present in
   each image (defect-free images form their own stratum). A fixed fraction of
   defect-free images is kept as background negatives (empty label files).
6. Emit images/ + labels/ trees, data.yaml, reports/dataset_stats.md and a few
   sanity overlay images (decoded mask + generated polygons) for manual checks.

Example:
    uv run python scripts/convert_severstal_to_yolo.py \\
        --raw-dir ~/datasets/severstal/raw --out-dir ~/datasets/severstal/yolo
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

IMG_HEIGHT = 256
IMG_WIDTH = 1600
NUM_CLASSES = 4
CLASS_NAMES = {i: f"defect_{i + 1}" for i in range(NUM_CLASSES)}
# BGR colors for sanity overlays, keyed by Severstal class id (1-4)
OVERLAY_COLORS = {1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 0), 4: (0, 255, 255)}


def rle_decode(rle: str, height: int = IMG_HEIGHT, width: int = IMG_WIDTH) -> np.ndarray:
    """Decode a column-major RLE string into a binary uint8 mask of (height, width)."""
    tokens = np.array(rle.split(), dtype=np.int64)
    starts, lengths = tokens[0::2] - 1, tokens[1::2]  # RLE positions are 1-indexed
    flat = np.zeros(height * width, dtype=np.uint8)
    for start, length in zip(starts, lengths):
        flat[start : start + length] = 1
    return flat.reshape((height, width), order="F")


def rle_encode(mask: np.ndarray) -> str:
    """Encode a binary mask into a column-major RLE string (inverse of rle_decode)."""
    flat = mask.flatten(order="F")
    padded = np.concatenate([[0], flat, [0]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    starts = changes[0::2] + 1  # back to 1-indexed
    lengths = changes[1::2] - changes[0::2]
    return " ".join(f"{s} {l}" for s, l in zip(starts, lengths))


def load_annotations(csv_path: Path) -> pd.DataFrame:
    """Load train.csv into columns [ImageId, ClassId, EncodedPixels], defects only."""
    df = pd.read_csv(csv_path)
    if "ImageId_ClassId" in df.columns:
        split = df["ImageId_ClassId"].str.rsplit("_", n=1, expand=True)
        df["ImageId"], df["ClassId"] = split[0], split[1]
    df = df.dropna(subset=["EncodedPixels"]).copy()
    df["ClassId"] = df["ClassId"].astype(int)
    df["EncodedPixels"] = df["EncodedPixels"].astype(str).str.strip()
    return df[["ImageId", "ClassId", "EncodedPixels"]].reset_index(drop=True)


def verify_roundtrip(df: pd.DataFrame) -> None:
    """Abort unless every RLE decodes and re-encodes to the identical string."""
    for row in df.itertuples(index=False):
        re_encoded = rle_encode(rle_decode(row.EncodedPixels))
        if re_encoded.split() != row.EncodedPixels.split():
            sys.exit(
                f"RLE round-trip FAILED for {row.ImageId} class {row.ClassId} — "
                "column-major decode assumption is wrong, aborting."
            )
    print(f"RLE round-trip check passed for all {len(df)} annotations.")


def mask_to_instances(
    mask: np.ndarray, min_area: int
) -> tuple[list[np.ndarray], int, int]:
    """Split a class mask into instance masks via 8-connected components.

    Returns (instance_masks, n_dropped_fragments, n_instances_with_holes).
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    instances, dropped, with_holes = [], 0, 0
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            dropped += 1
            continue
        inst = (labels == i).astype(np.uint8)
        _, hierarchy = cv2.findContours(inst, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None and any(h[3] != -1 for h in hierarchy[0]):
            with_holes += 1
        instances.append(inst)
    return instances, dropped, with_holes


def instance_to_polygon(inst_mask: np.ndarray) -> np.ndarray | None:
    """Return the outer contour of one instance as an (N, 2) pixel-coord array."""
    contours, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).squeeze(axis=1)
    if contour.ndim != 2 or len(contour) < 3:
        return None
    return contour


def polygon_to_label_line(class_id: int, polygon: np.ndarray) -> str:
    """Format one instance as a YOLO-seg label line with normalized coordinates."""
    normalized = polygon.astype(np.float64) / [IMG_WIDTH, IMG_HEIGHT]
    normalized = normalized.clip(0.0, 1.0)
    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in normalized)
    return f"{class_id - 1} {coords}"  # Severstal 1-4 -> YOLO 0-3


def stratified_split(
    strata: dict[str, list[str]], val_frac: float, rng: random.Random
) -> tuple[set[str], set[str]]:
    """Split image ids into train/val per stratum so rare class combos stay balanced."""
    train_ids, val_ids = set(), set()
    for key in sorted(strata):
        ids = sorted(strata[key])
        rng.shuffle(ids)
        n_val = round(len(ids) * val_frac)
        if n_val == 0 and len(ids) >= 5:
            n_val = 1
        val_ids.update(ids[:n_val])
        train_ids.update(ids[n_val:])
    return train_ids, val_ids


def write_sanity_overlay(
    image_path: Path, class_masks: dict[int, np.ndarray], polygons: list[tuple[int, np.ndarray]], out_path: Path
) -> None:
    """Save original image blended with class masks plus generated polygon outlines."""
    image = cv2.imread(str(image_path))
    overlay = image.copy()
    for class_id, mask in class_masks.items():
        overlay[mask.astype(bool)] = OVERLAY_COLORS[class_id]
    blended = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
    for class_id, polygon in polygons:
        cv2.polylines(blended, [polygon.reshape(-1, 1, 2)], True, OVERLAY_COLORS[class_id], 1)
    cv2.imwrite(str(out_path), blended)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, required=True, help="dir containing train.csv and train_images/")
    parser.add_argument("--out-dir", type=Path, required=True, help="output dir for the YOLO-seg dataset")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"), help="dir for stats and sanity overlays")
    parser.add_argument("--val-frac", type=float, default=0.1, help="validation fraction (default 0.1)")
    parser.add_argument("--neg-ratio", type=float, default=0.1, help="negatives kept as a ratio of defect images")
    parser.add_argument("--min-area", type=int, default=16, help="drop instance fragments smaller than this (px)")
    parser.add_argument("--seed", type=int, default=42, help="random seed for split and sampling")
    parser.add_argument("--num-sanity", type=int, default=5, help="number of sanity overlay images")
    args = parser.parse_args()

    raw_dir = args.raw_dir.expanduser()
    out_dir = args.out_dir.expanduser()
    images_dir = raw_dir / "train_images"
    rng = random.Random(args.seed)

    df = load_annotations(raw_dir / "train.csv")
    verify_roundtrip(df)

    all_images = sorted(p.name for p in images_dir.glob("*.jpg"))
    annotations = defaultdict(dict)  # ImageId -> {ClassId: rle}
    for row in df.itertuples(index=False):
        annotations[row.ImageId][row.ClassId] = row.EncodedPixels
    defect_images = sorted(annotations)
    negative_pool = sorted(set(all_images) - set(defect_images))
    negatives = sorted(rng.sample(negative_pool, min(int(len(defect_images) * args.neg_ratio), len(negative_pool))))

    # stratify on the exact combination of classes present; negatives are one stratum
    strata: dict[str, list[str]] = defaultdict(list)
    for image_id in defect_images:
        strata[",".join(map(str, sorted(annotations[image_id])))].append(image_id)
    for image_id in negatives:
        strata["neg"].append(image_id)
    train_ids, val_ids = stratified_split(strata, args.val_frac, rng)

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    sanity_dir = args.reports_dir / "sanity"
    sanity_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "images": Counter(),  # (split, class) -> images containing class
        "instances": Counter(),  # (split, class) -> instance count
        "dropped_fragments": Counter(),  # class -> dropped tiny fragments
        "with_holes": Counter(),  # class -> instances containing holes
        "degenerate": 0,  # instances whose contour had < 3 points
    }
    sanity_ids = set(rng.sample(defect_images, min(args.num_sanity, len(defect_images))))

    for n_done, image_id in enumerate(sorted(train_ids | val_ids), 1):
        split = "train" if image_id in train_ids else "val"
        label_lines: list[str] = []
        class_masks: dict[int, np.ndarray] = {}
        polygons: list[tuple[int, np.ndarray]] = []
        for class_id, rle in sorted(annotations.get(image_id, {}).items()):
            mask = rle_decode(rle)
            class_masks[class_id] = mask
            instances, dropped, with_holes = mask_to_instances(mask, args.min_area)
            stats["dropped_fragments"][class_id] += dropped
            stats["with_holes"][class_id] += with_holes
            if instances:
                stats["images"][(split, class_id)] += 1
            for inst in instances:
                polygon = instance_to_polygon(inst)
                if polygon is None:
                    stats["degenerate"] += 1
                    continue
                label_lines.append(polygon_to_label_line(class_id, polygon))
                polygons.append((class_id, polygon))
                stats["instances"][(split, class_id)] += 1
        (out_dir / "labels" / split / f"{Path(image_id).stem}.txt").write_text(
            "\n".join(label_lines) + ("\n" if label_lines else "")
        )
        shutil.copy2(images_dir / image_id, out_dir / "images" / split / image_id)
        if image_id in sanity_ids:
            write_sanity_overlay(images_dir / image_id, class_masks, polygons, sanity_dir / image_id)
        if n_done % 1000 == 0:
            print(f"  processed {n_done} images...")

    data_yaml = {
        "path": str(out_dir),
        "train": "images/train",
        "val": "images/val",
        "names": CLASS_NAMES,
    }
    (out_dir / "data.yaml").write_text(yaml.dump(data_yaml, sort_keys=False))

    # ---- stats report -------------------------------------------------------
    n_neg_train = len(set(negatives) & train_ids)
    n_neg_val = len(set(negatives) & val_ids)
    lines = [
        "# Severstal -> YOLO-seg dataset stats",
        "",
        f"- seed: {args.seed}, val_frac: {args.val_frac}, neg_ratio: {args.neg_ratio}, min_area: {args.min_area}px",
        f"- source images: {len(all_images)} total, {len(defect_images)} with defects, {len(negative_pool)} defect-free",
        f"- kept: {len(train_ids)} train / {len(val_ids)} val images "
        f"(negatives: {n_neg_train} train / {n_neg_val} val)",
        f"- RLE round-trip: PASSED ({len(df)} annotations)",
        f"- degenerate polygons skipped (<3 points): {stats['degenerate']}",
        "",
        "| class | train imgs | val imgs | train instances | val instances | dropped <min_area | instances w/ holes |",
        "|-------|-----------:|---------:|----------------:|--------------:|------------------:|-------------------:|",
    ]
    for class_id in range(1, NUM_CLASSES + 1):
        lines.append(
            f"| defect_{class_id} | {stats['images'][('train', class_id)]} | {stats['images'][('val', class_id)]} "
            f"| {stats['instances'][('train', class_id)]} | {stats['instances'][('val', class_id)]} "
            f"| {stats['dropped_fragments'][class_id]} | {stats['with_holes'][class_id]} |"
        )
    report = "\n".join(lines) + "\n"
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "dataset_stats.md").write_text(report)
    print(report)
    print(f"Dataset written to {out_dir}, data.yaml at {out_dir / 'data.yaml'}")
    print(f"Stats at {args.reports_dir / 'dataset_stats.md'}, overlays in {sanity_dir}")


if __name__ == "__main__":
    main()
