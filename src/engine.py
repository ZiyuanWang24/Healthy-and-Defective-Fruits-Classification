'''
Step 5 — src/engine.py (training + evaluation loops)

This is model-agnostic training logic.
'''
# src/engine.py
from typing import Tuple
import torch
import torch.nn as nn

@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[float, float, Tuple[int, int, int, int]]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total = 0
    correct = 0
    loss_sum = 0.0
    tn = fp = fn = tp = 0

    for rgb, sil, y, _paths in loader:
        rgb = rgb.to(device, non_blocking=True)
        sil = sil.to(device, non_blocking=True)
        y   = y.to(device, non_blocking=True)

        logits = model(rgb, sil)             # [B, 2]
        loss = criterion(logits, y)
        loss_sum += loss.item() * y.size(0)

        preds = logits.argmax(dim=1)   # [B]
        total += y.size(0)
        correct += (preds == y).sum().item()

        for pi, yi in zip(preds.tolist(), y.tolist()):
            if yi == 0 and pi == 0: tn += 1
            elif yi == 0 and pi == 1: fp += 1
            elif yi == 1 and pi == 0: fn += 1
            elif yi == 1 and pi == 1: tp += 1

    acc = correct / max(1, total)
    avg_loss = loss_sum / max(1, total)
    return avg_loss, acc, (tn, fp, fn, tp)

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    crit = nn.CrossEntropyLoss()
    loss_sum, correct, total = 0.0, 0, 0

    for rgb, sil, y, _paths in loader:
        rgb = rgb.to(device, non_blocking=True)
        sil = sil.to(device, non_blocking=True)
        y   = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(rgb, sil)
        loss = crit(logits, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return loss_sum / max(1, total), correct / max(1, total)