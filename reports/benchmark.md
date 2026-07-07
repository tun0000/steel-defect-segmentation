# Export & latency benchmark

- weights: `weights/steel_defect_yolo26s_seg_best.pt`, onnx: `weights/steel_defect_yolo26s_seg_best.onnx`
- imgsz=1024, batch=1, warmup=10, iters=100
- available onnxruntime providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']

| device | mean (ms) | p50 (ms) | p95 (ms) |
|--------|----------:|---------:|---------:|
| GPU (CUDAExecutionProvider) | 8.04 | 8.08 | 8.46 |
| CPU (CPUExecutionProvider) | 167.39 | 166.70 | 179.35 |
