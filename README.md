# Steel Defect Segmentation with YOLO26

Instance segmentation of steel surface defects (4 defect classes) using
[Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/) segmentation models,
trained on the [Severstal: Steel Defect Detection](https://www.kaggle.com/competitions/severstal-steel-defect-detection)
dataset.

> **Status: work in progress** — Phase 1 (data pipeline + training notebook) done, training and evaluation pending.

## Why this matters for steel / manufacturing quality inspection

<!-- TODO(Phase 2): value proposition for automated surface inspection -->

## Results

<!-- TODO(Phase 2): mask mAP50 / mAP50-95 table (overall + per class), measured latency (RTX 4090 / CPU, ONNX) -->

| Model | imgsz | mask mAP50 | mask mAP50-95 | GPU latency | CPU latency |
|-------|-------|-----------|---------------|-------------|-------------|
| yolo26s-seg | 1024 | TBD | TBD | TBD | TBD |

## Demo

<!-- TODO(Phase 2): demo GIF + Hugging Face Space link -->

## Reproduce

Prerequisites: [uv](https://docs.astral.sh/uv/), a Kaggle account that has joined the
competition and accepted its rules, and `~/.kaggle/kaggle.json` API credentials.

```bash
# 1. install dependencies
uv sync

# 2. download the competition data (requires accepted competition rules)
uv run kaggle competitions download -c severstal-steel-defect-detection -p ~/datasets/severstal/raw
unzip -q ~/datasets/severstal/raw/severstal-steel-defect-detection.zip -d ~/datasets/severstal/raw

# 3. convert RLE annotations to YOLO-seg format (stats + sanity overlays in reports/)
uv run python scripts/convert_severstal_to_yolo.py \
    --raw-dir ~/datasets/severstal/raw --out-dir ~/datasets/severstal/yolo

# 4. train — open notebooks/train_colab.ipynb in Google Colab (Runtime -> Run all)
```

## Dataset

Data comes from the Severstal: Steel Defect Detection Kaggle competition
(12,568 annotated 1600x256 grayscale images, 4 defect classes, RLE masks).
Competition rules do not permit redistribution, so **no image data is included in
this repository** — download it from Kaggle with your own account.

## License

Code is MIT licensed (see [LICENSE](LICENSE)). The dataset is subject to the
[competition rules](https://www.kaggle.com/competitions/severstal-steel-defect-detection/rules).
