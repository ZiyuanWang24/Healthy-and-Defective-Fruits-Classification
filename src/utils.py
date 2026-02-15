# src/utils.py
import random
from pathlib import Path
from typing import List
import torch

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def write_run_info(out_dir: Path, text: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_info.txt").write_text(text)