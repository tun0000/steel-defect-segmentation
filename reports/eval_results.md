# Evaluation report

- weights: `weights/steel_defect_yolo26s_seg_best.pt`
- data: `/home/tun2404/datasets/severstal/yolo/data.yaml`, imgsz=1024

**Overall — mask mAP50: 0.5869, mask mAP50-95: 0.2324** (box mAP50: 0.6586, box mAP50-95: 0.3428)

| class | images | instances | mask P | mask R | mask mAP50 | mask mAP50-95 | box mAP50 | box mAP50-95 |
|-------|-------:|----------:|-------:|-------:|-----------:|--------------:|----------:|-------------:|
| defect_1 | 90 | 293 | 0.6013 | 0.4642 | 0.5370 | 0.1728 | 0.5989 | 0.2644 |
| defect_2 | 25 | 30 | 0.6185 | 0.5667 | 0.5433 | 0.1807 | 0.7066 | 0.3327 |
| defect_3 | 514 | 1479 | 0.6515 | 0.6017 | 0.6252 | 0.2599 | 0.6801 | 0.3765 |
| defect_4 | 80 | 210 | 0.7385 | 0.5781 | 0.6424 | 0.3161 | 0.6486 | 0.3976 |
