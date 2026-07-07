# Evaluation report

- weights: `weights/steel_defect_yolo26s_seg_defect1_weighted.pt`
- data: `/home/tun2404/datasets/severstal/yolo/data.yaml`, imgsz=1024

**Overall — mask mAP50: 0.5599, mask mAP50-95: 0.2156** (box mAP50: 0.6221, box mAP50-95: 0.3188)

| class | images | instances | mask P | mask R | mask mAP50 | mask mAP50-95 | box mAP50 | box mAP50-95 |
|-------|-------:|----------:|-------:|-------:|-----------:|--------------:|----------:|-------------:|
| defect_1 | 90 | 293 | 0.6099 | 0.4539 | 0.5226 | 0.1789 | 0.5730 | 0.2363 |
| defect_2 | 25 | 30 | 0.5547 | 0.5333 | 0.4911 | 0.1449 | 0.6400 | 0.3380 |
| defect_3 | 514 | 1479 | 0.6330 | 0.5727 | 0.6089 | 0.2645 | 0.6437 | 0.3482 |
| defect_4 | 80 | 210 | 0.6198 | 0.5667 | 0.6168 | 0.2739 | 0.6315 | 0.3525 |
