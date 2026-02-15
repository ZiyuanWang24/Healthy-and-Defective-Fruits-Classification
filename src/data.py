from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# -----------------------------
# Helpers
# -----------------------------
def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    files: List[Path] = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    if not files:
        raise RuntimeError(f"No image files found in: {folder}")
    return sorted(files)


def split_paths(paths: List[Path], val_ratio: float, seed: int) -> Tuple[List[Path], List[Path]]:
    rng = random.Random(seed)
    idxs = list(range(len(paths)))
    rng.shuffle(idxs)
    n_val = int(round(len(paths) * val_ratio))
    val_set = set(idxs[:n_val])

    train, val = [], []
    for i, p in enumerate(paths):
        (val if i in val_set else train).append(p)
    return train, val


def make_transforms(img_size: int = 224):
    # ImageNet normalization (works well with pretrained MobileNetV2)
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tf, val_tf

import re
from pathlib import Path
from typing import Dict, List, Tuple

def pair_by_id_stem(
    rgb_dir: Path,
    sil_dir: Path,
    sil_suffix: str = "_step3_binary",
) -> List[Tuple[Path, Path]]:
    """
    Pair (rgb_path, sil_path) by ID-stem.

    Example:
      RGB stem: SD_REAL_0001
      SIL stem: SD_REAL_0001_step3_binary  -> base = SD_REAL_0001

    Keeps pairs only when both exist.
    """

    rgb_files = list_images(rgb_dir)
    sil_files = list_images(sil_dir)

    def rgb_key(p: Path) -> str:
        return p.stem  # SD_REAL_0001

    def sil_key(p: Path) -> str:
        s = p.stem  # SD_REAL_0001_step3_binary
        # Most robust: remove suffix if present (even if more text comes after it)
        if sil_suffix in s:
            return s.split(sil_suffix)[0]  # SD_REAL_0001
        return s  # fallback

    # build silhouette map: key -> silhouette path
    sil_map: Dict[str, Path] = {}
    dup = 0
    for sp in sil_files:
        k = sil_key(sp)

        if k in sil_map:
            dup += 1
            # keep the first one; you can change this behavior if needed
            continue
        sil_map[k] = sp

    if dup > 0:
        print(f"[pair_by_id_stem] Warning: {dup} duplicate silhouette keys in {sil_dir} (kept first occurrence).")

    pairs: List[Tuple[Path, Path]] = []
    missing = 0
    for rp in rgb_files:
        k = rgb_key(rp)
        sp = sil_map.get(k)
        if sp is None:
            missing += 1
            continue
        pairs.append((rp, sp))

    if not pairs:
        raise RuntimeError(
            f"No paired files found between:\n  RGB: {rgb_dir}\n  SIL: {sil_dir}\n"
            f"Expected silhouette names like: <ID>{sil_suffix}.* (e.g., SD_REAL_0001_step3_binary.png)"
        )

    if missing > 0:
        print(f"[pair_by_id_stem] Warning: {missing} RGB files in {rgb_dir} had no silhouette match in {sil_dir}")

    return pairs



# def pair_by_filename(rgb_dir: Path, sil_dir: Path) -> List[Tuple[Path, Path]]:
#     """
#     Create (rgb_path, sil_path) pairs by matching *filename*.
#     Only keeps files that exist in BOTH dirs.
#     """
#     rgb_files = list_images(rgb_dir)
#     sil_map: Dict[str, Path] = {p.name: p for p in list_images(sil_dir)}

#     pairs: List[Tuple[Path, Path]] = []
#     missing = 0
#     for rp in rgb_files:
#         sp = sil_map.get(rp.name)
#         if sp is None:
#             missing += 1
#             continue
#         pairs.append((rp, sp))

#     if not pairs:
#         raise RuntimeError(
#             f"No paired files found between:\n  RGB: {rgb_dir}\n  SIL: {sil_dir}\n"
#             f"Check if filenames match exactly."
#         )

#     if missing > 0:
#         # not fatal, but useful to know
#         print(f"[pair_by_filename] Warning: {missing} RGB files in {rgb_dir} had no silhouette match in {sil_dir}")

#     return pairs


# -----------------------------
# Dataset
# -----------------------------
class PairedBinaryDataset(Dataset):
    """
    samples: list of (rgb_path, sil_path, label)
    returns: (rgb_tensor, sil_tensor, label_tensor, rgb_path_str)
    """
    def __init__(self, samples: List[Tuple[Path, Path, int]], transform_rgb=None, transform_sil=None):
        self.samples = samples
        self.transform_rgb = transform_rgb
        self.transform_sil = transform_sil

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        rgb_path, sil_path, label = self.samples[idx]

        rgb = Image.open(rgb_path).convert("RGB")
        sil = Image.open(sil_path).convert("RGB")  # keep 3 channels for MobileNetV2

        if self.transform_rgb is not None:
            rgb = self.transform_rgb(rgb)
        if self.transform_sil is not None:
            sil = self.transform_sil(sil)

        y = torch.tensor(label, dtype=torch.long)
        return rgb, sil, y, str(rgb_path)


# -----------------------------
# Build loaders
# -----------------------------
@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    train_counts: Dict[str, int]
    val_counts: Dict[str, int]


def build_data_loaders(
    images_root: Path,
    img_size: int = 224,
    batch_size: int = 32,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 2,
    balance: bool = True,   # use WeightedRandomSampler
) -> DataBundle:
    """
    images_root should contain:
      fresh/, fresh_silhouette/,
      rot_defect/, rot_defect_silhouette/,
      bruise_defect/, bruise_defect_silhouette/,
      scab_defect/, scab_defect_silhouette/
    """
    # --- define dirs
    fresh_rgb = images_root / "fresh"
    fresh_sil = images_root / "fresh_silhouette"

    defect_types = ["rot_defect", "bruise_defect", "scab_defect"]
    defect_rgb_dirs = [images_root / d for d in defect_types]
    defect_sil_dirs = [images_root / f"{d}_silhouette" for d in defect_types]

    # --- pair healthy
    healthy_pairs = pair_by_id_stem(fresh_rgb, fresh_sil)  # [(rgb,sil), ...]
    healthy_train, healthy_val = split_paths(healthy_pairs, val_ratio, seed)

    # --- pair defects (merge all types)
    defect_pairs_all: List[Tuple[Path, Path]] = []
    for rgb_d, sil_d in zip(defect_rgb_dirs, defect_sil_dirs):
        defect_pairs_all.extend(pair_by_id_stem(rgb_d, sil_d))

    defect_train, defect_val = split_paths(defect_pairs_all, val_ratio, seed + 1)

    # --- label them
    train_samples: List[Tuple[Path, Path, int]] = [(r, s, 0) for (r, s) in healthy_train] + [(r, s, 1) for (r, s) in defect_train]
    val_samples:   List[Tuple[Path, Path, int]] = [(r, s, 0) for (r, s) in healthy_val]   + [(r, s, 1) for (r, s) in defect_val]

    # --- transforms
    train_tf, val_tf = make_transforms(img_size)

    train_ds = PairedBinaryDataset(train_samples, transform_rgb=train_tf, transform_sil=train_tf)
    val_ds   = PairedBinaryDataset(val_samples,   transform_rgb=val_tf,   transform_sil=val_tf)

    # --- sampler (optional)
    if balance:
        n_healthy = len(healthy_train)
        n_defect  = len(defect_train)
        w_healthy = 1.0 / max(1, n_healthy)
        w_defect  = 1.0 / max(1, n_defect)

        # IMPORTANT: must match order of train_samples (healthy first then defect)
        sample_weights = [w_healthy] * n_healthy + [w_defect] * n_defect
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )

    val_loader = DataLoader(
        val_ds, batch_size=max(64, batch_size), shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    train_counts = {"healthy": len(healthy_train), "defect": len(defect_train)}
    val_counts   = {"healthy": len(healthy_val),   "defect": len(defect_val)}

    print(f"[Data] Train counts: {train_counts} | Val counts: {val_counts}")
    return DataBundle(train_loader=train_loader, val_loader=val_loader, train_counts=train_counts, val_counts=val_counts)
