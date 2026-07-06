# Severstal -> YOLO-seg dataset stats

- seed: 42, val_frac: 0.1, neg_ratio: 0.1, min_area: 16px
- source images: 12568 total, 6666 with defects, 5902 defect-free
- kept: 6598 train / 734 val images (negatives: 599 train / 67 val)
- RLE round-trip: PASSED (7095 annotations)
- degenerate polygons skipped (<3 points): 3

| class | train imgs | val imgs | train instances | val instances | dropped <min_area | instances w/ holes |
|-------|-----------:|---------:|----------------:|--------------:|------------------:|-------------------:|
| defect_1 | 807 | 90 | 2789 | 293 | 0 | 17 |
| defect_2 | 222 | 25 | 291 | 30 | 0 | 1 |
| defect_3 | 4636 | 514 | 13132 | 1479 | 34 | 155 |
| defect_4 | 721 | 80 | 1687 | 210 | 10 | 55 |
