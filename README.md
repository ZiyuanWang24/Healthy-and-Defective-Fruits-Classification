# Fruit Defect Classification with Multi-Input MobileNetV2 (RGB + Silhouette Fusion)

A **PyTorch** image-classification project that detects **healthy vs defective** apples by fusing:
1) the original **RGB image**, and  
2) a corresponding **silhouette / defect mask image** (e.g., a precomputed segmentation output).

The core idea is that segmentation-derived cues (shape/defect region) can complement RGB texture/color, improving robustness when lighting or color varies.

---

## Why this is interesting
- **Multi-modal / multi-input learning**: two encoders + learnable fusion (concat / sum / gated).
- **Practical data engineering**: pairs RGB↔mask files by ID-stem, supports mixed defect categories.
- **Imbalance handling**: `WeightedRandomSampler` for balanced training.
- **Reproducible training**: deterministic seeds + run metadata saved to `results/run_info.txt`.
- **Clear evaluation**: accuracy + confusion matrix on the validation split.

---

## Project structure
```
.
├── train.py
└── src
    ├── config.py
    ├── data.py
    ├── engine.py
    ├── model.py
    └── utils.py
```

---

## Dataset layout expected
Create an `images/` folder with **RGB** folders and matching **silhouette** folders:

```
images/
  fresh/
  fresh_silhouette/
  rot_defect/
  rot_defect_silhouette/
  bruise_defect/
  bruise_defect_silhouette/
  scab_defect/
  scab_defect_silhouette/
```

### Pairing rule (important)
RGB files are paired to silhouette files by **ID-stem**.

Example:
- RGB: `SD_REAL_0001.jpg`  → stem `SD_REAL_0001`
- Silhouette: `SD_REAL_0001_step3_binary.png` → base stem `SD_REAL_0001`

(See `pair_by_id_stem()` in `src/data.py`.)

---

## Setup
```bash
# (recommended) create a clean env
conda create -n fruit_fusion python=3.11 -y
conda activate fruit_fusion

pip install torch torchvision pillow matplotlib
```

---

## Train
```bash
python train.py --epochs 10 --batch_size 32 --lr 3e-4
```

Logs print per-epoch metrics like:
```
[01/10] train loss=... acc=... | val loss=... acc=... | CM [[TN=..., FP=...], [FN=..., TP=...]]
```

Tip: save logs for plotting:
```bash
python train.py | tee results/train.log
```

---

## Generate figures (training curves + confusion matrix)
After you have `results/train.log`:
```bash
python scripts/make_figures.py --log results/train.log --out results/figures
```

Outputs:
- `training_curves.png`
- `confusion_matrix_best.png`

---

## Model overview
- Two **MobileNetV2** encoders (RGB branch + silhouette branch) used as **feature extractors**
- Feature fusion: `concat` / `sum` / **gated** (default)
- MLP head: Linear → ReLU → Dropout → Linear

---

## Notes / improvements you can add later
- Save a CSV of metrics each epoch (easy to parse + share).
- Add AMP (`autocast`) support (a `GradScaler` is instantiated, but not used in the current loop).
- Add test-time inference script + export to TorchScript/ONNX.

---

## License
Choose a license (MIT/Apache-2.0) before publishing.
