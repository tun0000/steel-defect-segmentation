# steel-defect-segmentation 專案計畫

## Context

求職作品集專案：用 Ultralytics **YOLO26 instance segmentation** 做鋼材表面瑕疵分割，資料為 Kaggle「Severstal: Steel Defect Detection」比賽資料（4 類瑕疵、RLE 遮罩標註）。已接受比賽規則、kaggle.json 已設定。

已確認的前提與決策：
- **YOLO26-seg 已於 2026-01 正式釋出**（`yolo26n-seg.pt` / `yolo26s-seg.pt` 皆有 COCO 預訓練權重，NMS-free、ONNX 匯出 end-to-end）。實作時仍須先查 ultralytics 官方文件確認最低版本與 API（CLAUDE.md 約定）。
- **imgsz=1024**：Severstal 影像為 1600×256 寬長條圖，640 letterbox 會把有效解析度壓到 640×102、細刮痕消失；1024 是精度與 Colab 訓練時間的平衡點。
- **Colab 資料流**：notebook 內直接 `kaggle` CLI 下載 + 重跑 repo 同一支轉換腳本（比賽規則禁止再散布資料，不上傳資料到 Drive 共享或 HF）。
- **資料存放**：WSL ext4（`~/datasets/severstal/`），程式碼留在目前 Windows 側專案資料夾；所有指令經 `wsl -e bash -lc "..."` 在 WSL 內執行（CLAUDE.md 約定，且避免 /mnt/c 小檔 I/O 瓶頸）。

## 資料集已知事實（規劃參考用；實作報告一律以腳本實際輸出為準）

- train 影像 12,568 張、全部 1600×256 JPEG；`train.csv` 欄位：ImageId, ClassId (1–4), EncodedPixels。
- RLE 為 **column-major**（由上到下、再由左到右）→ decode 用 `flat.reshape(256, 1600, order='F')`。
- 約 6,666 張含瑕疵 / 5,902 張無瑕疵；image-level 類別數約 class1≈900、class2≈250、class3≈5,150、class4≈800 —— **嚴重不平衡，class 2 極少**。

## 專案結構

```
steel-defect-segmentation/
├── plan.md                  # 本計畫
├── README.md / LICENSE(MIT) / .gitignore / .env.example
├── pyproject.toml           # uv 鎖定；deps: ultralytics, opencv-python, pandas, numpy, python-dotenv, kaggle
├── scripts/
│   ├── convert_severstal_to_yolo.py   # Phase 1 核心
│   ├── evaluate.py                    # Phase 2
│   └── export_benchmark.py            # Phase 2
├── notebooks/steel_defect_yolo26_train.ipynb
├── app/                     # Phase 2 Gradio demo（可部署 HF Space）
├── reports/                 # 統計表、sanity 疊圖、結果圖（小檔可進 git）
└── weights/                 # gitignored；Phase 2 放 best.pt / best.onnx
資料實體：WSL ~/datasets/severstal/{raw, yolo}/（不進 git）
```

## Phase 1（本機，v1 實作範圍）

### Step 0：初始化專案
- `git init`；`.gitignore` 先寫好（data、weights、.env、runs/、*.pt、*.onnx、大檔）；MIT LICENSE；README skeleton；`.env.example`。
- WSL 內用 uv 建 venv、`pyproject.toml` 鎖依賴。動工前先上網確認 ultralytics 目前版本的 YOLO26-seg API 用法。

### Step 1：下載資料
- `kaggle competitions download -c severstal-steel-defect-detection -p ~/datasets/severstal/raw` → unzip。

### Step 2：轉換腳本 `convert_severstal_to_yolo.py`（argparse + docstring，seed=42）
1. **RLE decode（column-major）**＋**round-trip 驗證**：全部標註列 decode→re-encode 必須完全還原原字串，任何一筆不符即中止；另隨機輸出 5 張「原圖+mask 疊圖」到 `reports/sanity/` 供人工抽查。
2. **instance 拆分**：每個 (image, class) 的合併 mask 用 `cv2.connectedComponentsWithStats` 拆成獨立 instance；面積 < `--min-area`（預設 16 px）的碎片丟棄並計數。
3. **polygon 轉換**：v1 用 `cv2.findContours(RETR_EXTERNAL)` 只取外輪廓；同時用 RETR_CCOMP 的 hierarchy **統計含孔洞的 instance 數**寫進報告（數字顯著再升級成 ultralytics merge_multi_segment 做法）。輪廓 ≥3 點、座標 normalize 到 0–1；class id 1–4 → 0–3。
4. **切分**：90/10 **stratified split**，分層鍵 = 該圖的瑕疵類別組合（無瑕疵圖自成一層），seed 固定 —— 避免 class 2（僅約 250 張）在 val 中波動過大。
5. **負樣本**：`--neg-ratio`（預設 0.1）保留約 10% 無瑕疵圖為背景負樣本（空 label .txt），符合 ultralytics 0–10% background 建議。
6. **輸出**：`~/datasets/severstal/yolo/{images,labels}/{train,val}`、`data.yaml`（WSL 絕對路徑＋4 類名稱）、統計報告 `reports/dataset_stats.md`（每類 instance 數 train/val、圖片數、負樣本數、含孔洞數、被過濾碎片數）——**統計結果輸出給使用者看**。

### Step 3：4090 smoke test（WSL）
- 小子集（~200 train / 50 val）+ `yolo26n-seg` + 1 epoch + imgsz=1024：確認 loss 正常、val 能跑、無 NaN。
- 用產出權重對 2–3 張 val 圖 predict 疊圖，人工確認 mask 位置合理。

### Step 4：`notebooks/steel_defect_yolo26_train.ipynb`（Runtime → Run all 一鍵跑完）
- Cells：GPU check → mount Drive → Kaggle 憑證（優先 Colab Secrets 的 KAGGLE_USERNAME/KAGGLE_KEY，fallback 讀 Drive 上的 kaggle.json）→ `git clone` 本 repo → pip install → kaggle 下載 → **跑 repo 同一支轉換腳本** → 訓練 → 驗證 → 產物存 Drive。
- 訓練設定：`yolo26s-seg.pt`、imgsz=1024、epochs=100 + patience=20、seed 固定；augmentation 針對鋼帶特性：`degrees=0`（方向固定）、`fliplr=0.5`、`flipud=0.5`（鋼帶上下翻轉合理）、`copy_paste=0.3`（seg 專用、幫助稀有類）、mosaic 預設、hsv 小值。
- `project=` 指到 Drive 路徑 → checkpoint 自動落在 Drive，附 `resume=True` 說明 cell 供斷線續訓；結束後 best.pt / results.csv 複製到 Drive 固定路徑。

### Step 5：git commit（英文、一個里程碑一個 commit）
**Phase 1 到此停止，等使用者去 Colab 訓練。**

## Phase 2（`weights/best.pt` 就位後）

6. **評估** `evaluate.py`：`model.val()` 取 mask mAP50 / mAP50-95 與 per-class AP → 寫進 README 結果表；每類挑 3–4 張 val 圖輸出「原圖 + 半透明彩色遮罩 + 類別標籤」疊圖到 `reports/figures/`。
7. **匯出與 benchmark** `export_benchmark.py`：export ONNX（YOLO26 NMS-free、end-to-end）；4090（onnxruntime-gpu）與 CPU（onnxruntime）各量 batch=1、imgsz=1024、warmup 10 次後跑 100 次，報 mean/p50/p95。
8. **Gradio demo**（`app/`）：上傳鋼材影像 → 疊上分割遮罩與類別/信心值；推論直接用 `YOLO("best.onnx")` 走 ultralytics ONNX 路徑（不手寫 proto 解碼），CPU 可跑、附 requirements 可部署 HF Space。
9. **發佈**：權重 + model card 上傳 HF Hub（**repo 名稱與可見性先確認**；model card 註明資料出處與 Severstal 競賽規則、資料本身不隨附）；完成 README：專案動機、「對鋼鐵/製造業品檢的價值」段落、結果表（實測數字）、demo GIF 佔位、重現步驟、HF 連結、資料授權聲明、**class 2 類別不平衡的誠實分析與改進方向**（作品集亮點）。

## 風險與備案

- **Severstal 下載失敗**（授權/API 問題）→ 改用其他有完整遮罩的公開 crack segmentation 資料集，並說明差異（類別數、標註型式、影像尺寸）。
- **YOLO26-seg API 與記憶不符** → 實作前先查官方文件（CLAUDE.md 約定），必要時降版或 fallback YOLO11-seg 並說明。
- **class 2 AP 偏低** → v1 不做 oversampling，README 誠實分析；v1.1 可選：複製 class 2 影像做 naive oversampling。
- **Colab T4 太慢** → 降 epochs 或換 A100；imgsz 保持 1024 不輕易降。

## Verification（各階段驗收）

- 轉換：RLE round-trip 100% 通過；`reports/sanity/` 疊圖人工抽查無錯位；統計表數字合理（總 instance 數、類別分布與文獻量級一致）。
- Smoke test：1 epoch loss 下降、val 跑通、predict 疊圖合理。
- Notebook：本機以同一支腳本 + 小子集驗證過每個步驟；`jupyter nbconvert --to script` 檢查可解析。
- Phase 2：所有報告數字來自實際執行輸出，嚴禁估計（CLAUDE.md 約定）。
