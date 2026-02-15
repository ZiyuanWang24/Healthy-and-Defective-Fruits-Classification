# Technical Report — Fruit Defect Classification via RGB + Silhouette Fusion

## 1. Problem statement
Food quality inspection is a high-impact, real-world computer vision task. In this project, the goal is to **classify apples as healthy vs defective** using paired inputs:
- **RGB images** provide texture and color cues.
- **Silhouette / mask images** provide shape and defect-region cues (often more invariant to lighting).

The project is designed to be:
- simple enough to reproduce quickly,
- but “real-world” enough to demonstrate data engineering + modeling choices recruiters care about.

## 2. Data pipeline

### 2.1 Folder structure
The dataloader expects an `images/` root directory with **healthy** (`fresh/`) and multiple defect subtypes:
- `rot_defect/`, `bruise_defect/`, `scab_defect/`

Each category has a paired silhouette folder:
- `*_silhouette/`

### 2.2 Pairing logic
Pairs are created by matching an RGB file stem (e.g., `SD_REAL_0001`) with a silhouette stem that contains the suffix `_step3_binary` (e.g., `SD_REAL_0001_step3_binary.png`). Missing pairs are skipped with a warning.

### 2.3 Train/val split
- A reproducible random split is created by shuffling indices with a seed.
- Separate seeds are used for healthy and defect splits.

### 2.4 Augmentation + normalization
- Resize to 224×224
- Random horizontal flip + small random rotation (train only)
- ImageNet normalization (aligned with pretrained MobileNetV2 weights)

### 2.5 Class imbalance handling
If the dataset has far more healthy than defective (or vice versa), training can become biased.  
This project uses `WeightedRandomSampler` to approximately balance class sampling during training.

## 3. Model

### 3.1 Architecture
The network is a two-branch encoder + fusion classifier:
1) **RGB encoder**: MobileNetV2 backbone → global average pooling → 1280-d vector  
2) **Silhouette encoder**: MobileNetV2 backbone → global average pooling → 1280-d vector  
3) **Fusion block** (three supported modes):
   - **concat**: [f_rgb ; f_sil] → 2560-d
   - **sum**: f_rgb + f_sil → 1280-d
   - **gated (default)**: learn g∈(0,1) and compute g·f_rgb + (1−g)·f_sil

4) **MLP head**:
   - Linear → ReLU → Dropout → Linear → logits (2 classes)

### 3.2 Why MobileNetV2?
MobileNetV2 is lightweight, widely used in production, and a strong baseline for embedded/real-time vision.

## 4. Training and evaluation

### 4.1 Loss and optimizer
- Cross-entropy loss
- AdamW optimizer

### 4.2 Metrics
- Accuracy
- Confusion matrix (TN, FP, FN, TP), printed each epoch

### 4.3 Checkpointing
The script saves the best validation-accuracy checkpoint to:
- `results/best_model.pt`

### 4.4 Reproducibility
A deterministic seed function is used to reduce run-to-run variance. Run configuration and dataset counts are written to:
- `results/run_info.txt`

## 5. Results (how to report)
After training, capture your best epoch metrics and include:
- Best validation accuracy
- Confusion matrix at best epoch
- Training curve plot (loss/acc vs epoch)

This repo includes a plotting utility that parses your `train.log` and generates:
- training curves
- best-epoch confusion matrix heatmap

## 6. Limitations
- No separate test set (easy to add).
- AMP scaffolding exists but is not fully wired into the training loop.
- Current `train.py` builds data loaders with fixed constants (can be refactored to fully use CLI config).

## 7. Future work (nice add-ons)
- Add inference script: classify a folder of images and export a CSV.
- Export model to ONNX and benchmark latency.
- Try fusion ablations (concat vs sum vs gated).
- Replace silhouette encoder with a smaller branch (e.g., fewer layers) to reduce compute.

---
**Author:** Ziyuan Wang  
**Stack:** PyTorch, torchvision, PIL, matplotlib
