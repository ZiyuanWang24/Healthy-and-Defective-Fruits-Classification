# src/config.py
from __future__ import annotations
from dataclasses import dataclass
import argparse

'''
Step 1 — src/config.py (all settings in one place)
'''
@dataclass
class Config:
    healthy_dir: str = "images/fresh"
    rot_defect_dir: str = "images/rot_defect"
    bruise_defect_dir: str = "images/bruise_defect"
    scab_defect_dir: str = "images/scab_defect"
    healthy_silhouette_dir: str = "images/fresh_silhouette"
    rot_defect_silhouette_dir: str = "images/rot_defect_silhouette"
    bruise_defect_silhouette_dir: str = "images/bruise_defect_silhouette"
    scab_defect_silhouette_dir: str = "images/scab_defect_silhouette"
    out_dir: str = "results"

    epochs: int = 10
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4

    val_ratio: float = 0.2
    img_size: int = 224
    seed: int = 42

    num_workers: int = 2
    pretrained: bool = True

def parse_args() -> Config:
    ap = argparse.ArgumentParser()

    ap.add_argument("--epochs", type=int, default=Config.epochs)
    ap.add_argument("--batch_size", type=int, default=Config.batch_size)
    ap.add_argument("--lr", type=float, default=Config.lr)
    ap.add_argument("--weight_decay", type=float, default=Config.weight_decay)

    ap.add_argument("--val_ratio", type=float, default=Config.val_ratio)
    ap.add_argument("--img_size", type=int, default=Config.img_size)
    ap.add_argument("--seed", type=int, default=Config.seed)

    ap.add_argument("--num_workers", type=int, default=Config.num_workers)
    ap.add_argument("--no_pretrained", action="store_true")

    args = ap.parse_args()

    cfg = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        img_size=args.img_size,
        seed=args.seed,
        num_workers=args.num_workers,
        pretrained=(not args.no_pretrained),
    )
    return cfg