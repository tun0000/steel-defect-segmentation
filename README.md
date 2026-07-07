# Steel Defect Segmentation with YOLO26

Instance segmentation of steel surface defects (4 defect classes) using
[Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/) segmentation models,
trained on the [Severstal: Steel Defect Detection](https://www.kaggle.com/competitions/severstal-steel-defect-detection)
dataset.

Trained on Google Colab (A100), evaluated and benchmarked locally on an RTX 4090.
Weights + model card: **https://huggingface.co/betty0/steel-defect-segmentation**

## Why this matters for steel / manufacturing quality inspection

Manual visual inspection of steel strip surfaces is slow, inconsistent between
inspectors, and hard to scale to full production-line speed. Instance
segmentation — as opposed to plain classification or bounding-box detection —
recovers the actual defect *shape and area*, which is what quality control
actually needs to decide severity (a small edge nick vs. a large-area scale
patch) and to feed downstream metrics like defect area per coil. An end-to-end,
NMS-free model like YOLO26-seg also keeps per-image latency low enough
(single-digit milliseconds on a GPU, see below) for inline inspection rather
than offline sampling.

## Results

Real numbers from [`scripts/evaluate.py`](scripts/evaluate.py) and
[`scripts/export_benchmark.py`](scripts/export_benchmark.py) against the held-out
validation split (734 images, seed 42), full reports in
[`reports/eval_results.md`](reports/eval_results.md) and
[`reports/benchmark.md`](reports/benchmark.md).

| Model | imgsz | mask mAP50 | mask mAP50-95 | GPU latency (RTX 4090, ONNX) | CPU latency (ONNX) |
|-------|-------|-----------:|--------------:|------------------------------:|--------------------:|
| yolo26s-seg | 1024 | 0.587 | 0.232 | 8.04 ms mean (p95 8.46 ms) | 167.39 ms mean (p95 179.35 ms) |

Per-class breakdown (mask metrics; `images`/`instances` are validation-split counts):

| class | images | instances | mask mAP50 | mask mAP50-95 |
|-------|-------:|----------:|-----------:|--------------:|
| defect_1 | 90 | 293 | 0.537 | 0.173 |
| defect_2 | 25 | 30 | 0.543 | 0.181 |
| defect_3 | 514 | 1479 | 0.625 | 0.260 |
| defect_4 | 80 | 210 | 0.642 | 0.316 |

Sample prediction overlays per class are in [`reports/figures/`](reports/figures/).

Training ran 99 epochs (early stopped, `patience=20`) before selecting epoch
79 as `best.pt` — Ultralytics picks the best checkpoint by combined box+mask
mAP50-95, not mask mAP50-95 alone, which does land on a different epoch (92)
if you (like I initially did) only look at the mask curve. Validation mask
mAP is noisy epoch-to-epoch on a 734-image validation split with 4 imbalanced
classes, but trends up and stabilizes in the second half of training:

![training curve](reports/figures/training_curve.png)

### Class imbalance: an honest look

The training set is heavily imbalanced — defect_3 appears in 4,636 training
images vs. only 222 for defect_2 (see [`reports/dataset_stats.md`](reports/dataset_stats.md))
— but instance count alone does not predict per-class difficulty. defect_2, the
*rarest* class, actually scores higher on mask mAP50-95 (0.181) than defect_1
(0.173), which has roughly 10x more training instances. defect_1's defects tend
to be thin, elongated scratches with ambiguous boundaries, which likely hurts
mask IoU at stricter thresholds regardless of how much data it has — while
defect_2's shape appears more consistent and learnable, and `copy_paste`
augmentation (enabled during training specifically to help rare classes) may
have offset some of its rarity. See the v1.1 experiment below for an attempt
at per-class loss weighting to address this.

### v1.1 experiment: per-class loss weighting for defect_1

Hypothesis: since defect_1 underperforms despite having roughly 10x more
training instances than defect_2, forcing the model to weight defect_1's
classification loss more heavily during training might close the gap without
touching the data pipeline.

Implementation: a custom `on_pretrain_routine_end` callback (see
[`notebooks/steel_defect_yolo26_train.ipynb`](notebooks/steel_defect_yolo26_train.ipynb))
sets `model.class_weights = [2.0, 1.0, 1.0, 1.0]` — double weight on defect_1,
unchanged for the rest. This is a different mechanism from Ultralytics' built-in
`cls_pw` hyperparameter, which reweights by *inverse frequency* and would have
pushed weight *away* from defect_1 (the more frequent of the two weak classes)
— the wrong direction for this specific goal. Everything else — architecture,
dataset, `imgsz=1024`, `epochs=100`/`patience=20`, seed, batch size,
augmentation — is identical to the baseline run, so any difference is
attributable to this one change.

Real results (held-out validation split, same [`scripts/evaluate.py`](scripts/evaluate.py)
methodology as the baseline table above; full report in
[`reports/defect1_weighted/eval_results.md`](reports/defect1_weighted/eval_results.md)):

| class | baseline mask mAP50-95 | weighted mask mAP50-95 | Δ | baseline mask mAP50 | weighted mask mAP50 | Δ |
|-------|------------------------:|------------------------:|---:|----------------------:|-----------------------:|---:|
| defect_1 | 0.173 | 0.179 | +0.006 | 0.537 | 0.523 | −0.014 |
| defect_2 | 0.181 | 0.145 | −0.036 | 0.543 | 0.491 | −0.052 |
| defect_3 | 0.260 | 0.265 | +0.005 | 0.625 | 0.609 | −0.016 |
| defect_4 | 0.316 | 0.274 | −0.042 | 0.642 | 0.617 | −0.026 |
| **overall** | **0.232** | **0.216** | **−0.017** | **0.587** | **0.560** | **−0.027** |

![baseline vs defect_1-weighted per-class mask mAP](reports/figures/defect1_weighted_comparison.png)

This run's own training curve confirms the earlier finish — validation mask
mAP is noisy in the same way as the baseline, but plateaus by epoch 40 instead
of continuing to climb through epoch 79:

![defect_1-weighted training curve](reports/defect1_weighted/figures/training_curve.png)

Verdict: it didn't pay off net. defect_1's mask mAP50-95 improved slightly
(+0.006), but its own mAP50 actually dropped (−0.014), and the cost landed
disproportionately on defect_2 (−0.036) and defect_4 (−0.042) — dragging the
combined fitness metric down enough that this run's best checkpoint landed at
epoch 40 instead of baseline's epoch 79. A flat per-class weight is too blunt
an instrument here: it doesn't distinguish *why* defect_1 underperforms (thin,
ambiguous-boundary scratches, per above) from simply needing a bigger number,
so it borrows capacity from other classes without addressing the actual cause.
**The baseline (`steel_defect_yolo26s_seg_best.pt`) remains the model backing
the demo and Hugging Face release.** A more targeted follow-up — e.g.
weighting only the mask loss term, or oversampling defect_1 images instead of
reweighting loss — would be the next thing worth trying.

## Demo

![demo](reports/figures/demo.gif)

**Live demo: https://huggingface.co/spaces/betty0/steel-defect-segmentation**
(CPU-only Space; downloads the ONNX weights from the model repo on startup).

Run it locally: `uv run python app/app.py --weights weights/steel_defect_yolo26s_seg_best.onnx`

## Reproduce

Prerequisites: [uv](https://docs.astral.sh/uv/), a Kaggle account that has joined the
competition and accepted its rules, and a Kaggle API credential in one of:
- `~/.kaggle/access_token` (current Kaggle API token — kaggle.com → profile → Settings → API), or
- `~/.kaggle/kaggle.json` (legacy username/key pair, via "Create Legacy API Key").

```bash
# 1. install dependencies
uv sync

# 2. download the competition data (requires accepted competition rules)
uv run kaggle competitions download -c severstal-steel-defect-detection -p ~/datasets/severstal/raw
unzip -q ~/datasets/severstal/raw/severstal-steel-defect-detection.zip -d ~/datasets/severstal/raw

# 3. convert RLE annotations to YOLO-seg format (stats + sanity overlays in reports/)
uv run python scripts/convert_severstal_to_yolo.py \
    --raw-dir ~/datasets/severstal/raw --out-dir ~/datasets/severstal/yolo

# 4. train — open notebooks/steel_defect_yolo26_train.ipynb in Google Colab (Runtime -> Run all)
# download the resulting best.pt into weights/ when done

# 5. evaluate (writes reports/eval_results.md + per-class overlays in reports/figures/)
uv run python scripts/evaluate.py \
    --weights weights/steel_defect_yolo26s_seg_best.pt --data ~/datasets/severstal/yolo/data.yaml

# 6. export to ONNX and benchmark GPU/CPU latency (writes reports/benchmark.md)
uv run python scripts/export_benchmark.py --weights weights/steel_defect_yolo26s_seg_best.pt

# 7. run the Gradio demo locally
uv run python app/app.py --weights weights/steel_defect_yolo26s_seg_best.onnx
```

## Dataset

Data comes from the Severstal: Steel Defect Detection Kaggle competition
(12,568 annotated 1600x256 grayscale images, 4 defect classes, RLE masks).
Competition rules do not permit redistribution, so **no image data is included in
this repository** — download it from Kaggle with your own account.

## License

Code is MIT licensed (see [LICENSE](LICENSE)). The dataset is subject to the
[competition rules](https://www.kaggle.com/competitions/severstal-steel-defect-detection/rules).
