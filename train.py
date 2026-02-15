# train.py
from __future__ import annotations
from pathlib import Path
import torch

from src.config import parse_args
from src.utils import set_seed, write_run_info
from src.data import build_data_loaders
from src.model import MultiInputMobileNetV2
from src.engine import train_one_epoch, evaluate

def main():
    cfg = parse_args()
    set_seed(cfg.seed)

    rot_defect_dir = Path(cfg.rot_defect_dir)
    bruise_defect_dir = Path(cfg.bruise_defect_dir)
    scab_defect_dir = Path(cfg.scab_defect_dir)

    rot_defect_silhouette_dir = Path(cfg.rot_defect_silhouette_dir)
    bruise_defect_silhouette_dir = Path(cfg.bruise_defect_silhouette_dir)
    scab_defect_silhouette_dir = Path(cfg.scab_defect_silhouette_dir)

    healthy_dir = Path(cfg.healthy_dir)
    defect_dir  = [rot_defect_dir, bruise_defect_dir, scab_defect_dir]
    healthy_silhouette_dir = Path(cfg.healthy_silhouette_dir)
    defect_silhouette_dir = [rot_defect_silhouette_dir,
                             bruise_defect_silhouette_dir,
                             scab_defect_silhouette_dir]

    out_dir     = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = build_data_loaders(
        images_root=Path("images"),
        img_size=224,
        batch_size=32,
        val_ratio=0.2,
        seed=42,
        num_workers=2,
        balance=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = MultiInputMobileNetV2(pretrained=cfg.pretrained).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    best_val_acc = -1.0
    best_path = out_dir / "best_model.pt"

    write_run_info(
        out_dir,
        text=(
            f"healthy_dir={healthy_dir}\n"
            f"defect_dir={defect_dir}\n"
            f"model=mobilenet_v2\n"
            f"pretrained={cfg.pretrained}\n"
            f"epochs={cfg.epochs}\n"
            f"batch_size={cfg.batch_size}\n"
            f"lr={cfg.lr}\n"
            f"weight_decay={cfg.weight_decay}\n"
            f"val_ratio={cfg.val_ratio}\n"
            f"img_size={cfg.img_size}\n"
            f"seed={cfg.seed}\n"
            f"train_counts: healthy={data.train_counts['healthy']} defect={data.train_counts['defect']}\n"
            f"val_counts: healthy={data.val_counts['healthy']} defect={data.val_counts['defect']}\n"
        ),
    )

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, data.train_loader, optimizer, device)
        va_loss, va_acc, (tn, fp, fn, tp) = evaluate(model, data.val_loader, device)

        print(
            f"[{epoch:02d}/{cfg.epochs}] "
            f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
            f"val loss={va_loss:.4f} acc={va_acc:.4f} | "
            f"CM [[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]"
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(
                {
                    "model_name": "mobilenet_v2",
                    "state_dict": model.state_dict(),
                    "val_acc": best_val_acc,
                    "img_size": cfg.img_size,
                    "pretrained": cfg.pretrained,
                },
                best_path,
            )

    print(f"Done. Best val acc={best_val_acc:.4f}")
    print(f"Saved best model to: {best_path}")

if __name__ == "__main__":
    main()
