#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title
-----
Defensive Distillation for Vision Mamba (Teacher–Student Fine-tuning)

Purpose
-------
This script implements *defensive distillation* (Papernot et al., 2016) for a Vision Mamba
image classifier on an ImageFolder dataset (train/valid/test).

It trains:
1) A **teacher** model using temperature-scaled logits (high temperature).
2) A **student** model using a combined objective:
   - KL divergence between teacher and student *soft targets* at the same temperature
   - Cross-entropy with ground-truth labels (hard labels)

The main artifact outputs are:
- best_teacher_model.pth (checkpoint with model weights + metadata)
- best_student_model.pth (checkpoint with model weights + metadata)
- teacher_training_history.csv (+ optional PNG plot)
- student_training_history.csv (+ optional PNG plot)
- config.json and class_mapping.json for reproducibility

Pseudocode
----------
load ImageFolder dataloaders (train + valid)
build teacher Vision Mamba
for each teacher epoch:
    forward teacher logits
    compute CE loss on (logits / T_teacher)
    backprop + update
    validate; save best checkpoint
load best teacher checkpoint
build student Vision Mamba
freeze teacher
for each student epoch:
    forward student logits
    forward teacher logits (no grad)
    compute distillation loss:
        alpha * KL( softmax(teacher/T), log_softmax(student/T) ) * T^2
        + (1-alpha) * CE(student, y)
    backprop + update
    validate; save best checkpoint
load best student checkpoint
report clean accuracy on validation set

Usage
-----
DATA_ROOT=dataset_rambu_lalu_lintas \
python defensive_distillation_vim_github.py

Optional environment variables (defaults shown):
- DATA_ROOT=dataset_rambu_lalu_lintas
- BATCH_SIZE=64
- MAX_EPOCHS_TEACHER=50
- MAX_EPOCHS_STUDENT=50
- TEMPERATURE_TEACHER=20.0
- TEMPERATURE_STUDENT=20.0
- ALPHA=0.7
- BASE_LR=1e-4
- WEIGHT_DECAY=1e-4
- EARLY_STOP_PATIENCE=10
- TARGET_ACC=0.70
- SEED=42
- IMG_SIZE=224
- OUTPUT_PREFIX=defensive_distillation_vim

Notes
-----
- This script is meant as a *research artifact* for a thesis/repository.
- Defensive distillation is widely known to be weak under modern adaptive attacks;
  it can still be included as a baseline/negative control, but should be discussed carefully.
"""

from __future__ import annotations

import inspect
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

from vision_mamba import Vim


# -----------------------------------------------------------------------------
# 1) Configuration
# -----------------------------------------------------------------------------
DATA_ROOT = Path(os.getenv("DATA_ROOT", "dataset_rambu_lalu_lintas"))
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "valid"
TEST_DIR = DATA_ROOT / "test"

IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
MAX_EPOCHS_TEACHER = int(os.getenv("MAX_EPOCHS_TEACHER", "50"))
MAX_EPOCHS_STUDENT = int(os.getenv("MAX_EPOCHS_STUDENT", "50"))

TEMPERATURE_TEACHER = float(os.getenv("TEMPERATURE_TEACHER", "20.0"))
TEMPERATURE_STUDENT = float(os.getenv("TEMPERATURE_STUDENT", "20.0"))

ALPHA = float(os.getenv("ALPHA", "0.7"))  # distillation weight vs hard-label weight

BASE_LR = float(os.getenv("BASE_LR", "1e-4"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "1e-4"))
TARGET_ACC = float(os.getenv("TARGET_ACC", "0.70"))
EARLY_STOP_PATIENCE = int(os.getenv("EARLY_STOP_PATIENCE", "10"))

SEED = int(os.getenv("SEED", "42"))
OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX", "defensive_distillation_vim")

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


# -----------------------------------------------------------------------------
# 2) Utilities
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Note: deterministic mode can reduce performance; enable if needed.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def create_save_dir(prefix: str = OUTPUT_PREFIX) -> Path:
    """Create a timestamped output directory."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"{prefix}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def build_vim_kwargs(num_classes: int, img_size: int = IMG_SIZE) -> Dict:
    """
    Build Vim constructor kwargs in a signature-safe way.
    Adjust values here to match your Vision Mamba configuration.
    """
    sig = inspect.signature(Vim).parameters
    kwargs = {
        "dim": 192,
        "dt_rank": 24,
        "dim_inner": 192,
        "d_state": 64,
        "num_classes": num_classes,
        "image_size": img_size,
        "patch_size": 32,
        "channels": 3,
        "dropout": 0.10,
        "depth": 6,
    }
    if "heads" in sig:
        kwargs["heads"] = 4
    # Keep only supported args for the current Vim implementation.
    return {k: v for k, v in kwargs.items() if k in sig}


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute accuracy from raw logits."""
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total if total > 0 else 0.0


def plot_history(df: pd.DataFrame, out_path: Path, title: str) -> None:
    """Save a simple training curve plot."""
    if df.empty:
        return

    plt.figure(figsize=(10, 6))
    if "train_loss" in df.columns:
        plt.plot(df["epoch"], df["train_loss"], marker="o", label="train_loss")
    if "val_loss" in df.columns:
        plt.plot(df["epoch"], df["val_loss"], marker="o", label="val_loss")
    if "train_acc" in df.columns:
        plt.plot(df["epoch"], df["train_acc"], marker="o", label="train_acc")
    if "val_acc" in df.columns:
        plt.plot(df["epoch"], df["val_acc"], marker="o", label="val_acc")

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# 3) Distillation loss
# -----------------------------------------------------------------------------
class DistillationLoss(nn.Module):
    """
    Combined knowledge distillation loss:
    - KL divergence on temperature-softened probabilities
    - Cross entropy on hard labels

    total = alpha * KL(teacher || student) * T^2 + (1-alpha) * CE(student, y)
    """

    def __init__(self, temperature: float = 1.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.kldiv = nn.KLDivLoss(reduction="batchmean")
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        # Soft targets from teacher and student at temperature T.
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student_log = F.log_softmax(student_logits / self.temperature, dim=1)

        # KL divergence term; multiply by T^2 as in standard distillation recipes.
        distill = self.kldiv(soft_student_log, soft_teacher) * (self.temperature**2)

        # Hard-label loss.
        hard = self.cross_entropy(student_logits, targets)

        return self.alpha * distill + (1.0 - self.alpha) * hard


# -----------------------------------------------------------------------------
# 4) Data loaders
# -----------------------------------------------------------------------------
def get_dataloaders() -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
    """Create train/validation dataloaders from ImageFolder directories."""
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Train folder not found: {TRAIN_DIR}")

    train_tfms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )

    eval_tfms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)

    use_val_dir = VAL_DIR.exists() and any(VAL_DIR.rglob("*.*"))
    if use_val_dir:
        val_ds = datasets.ImageFolder(VAL_DIR, transform=eval_tfms)
        print("[INFO] Using 'valid' as the validation set.")
    else:
        if not TEST_DIR.exists():
            raise FileNotFoundError(f"Neither 'valid' nor 'test' folder found under: {DATA_ROOT}")
        val_ds = datasets.ImageFolder(TEST_DIR, transform=eval_tfms)
        print("[INFO] 'valid' is missing/empty; using 'test' as the validation set.")

    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}
    print(f"[INFO] num_classes={len(idx_to_class)}")

    num_workers = min(4, (os.cpu_count() // 2) if os.cpu_count() else 2)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, idx_to_class


# -----------------------------------------------------------------------------
# 5) Evaluation helpers
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Return (loss, accuracy) on a loader."""
    model.eval()
    running_loss = 0.0
    running_correct = 0.0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)
        acc = accuracy_from_logits(logits, targets)

        bs = targets.size(0)
        running_loss += float(loss.item()) * bs
        running_correct += float(acc) * bs
        total += bs

    if total == 0:
        return 0.0, 0.0
    return running_loss / total, running_correct / total


@torch.no_grad()
def evaluate_clean_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute clean accuracy (no attack) on a loader."""
    model.eval()
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return correct / total if total > 0 else 0.0


# -----------------------------------------------------------------------------
# 6) Training: teacher
# -----------------------------------------------------------------------------
def train_teacher_model(
    model: nn.Module,
    vim_kwargs: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    save_dir: Path,
) -> nn.Module:
    """
    Train the teacher with temperature scaling on logits:
        loss = CE(logits / T_teacher, y)
    """
    print("\n" + "=" * 60)
    print("TRAINING TEACHER MODEL (temperature-scaled logits)")
    print("=" * 60)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS_TEACHER)

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    patience = 0

    best_path = save_dir / "best_teacher_model.pth"

    for epoch in range(1, MAX_EPOCHS_TEACHER + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            # Key step: train teacher under temperature (divide logits by T).
            loss = criterion(logits / TEMPERATURE_TEACHER, targets)

            loss.backward()
            optimizer.step()

            acc = accuracy_from_logits(logits, targets)
            bs = targets.size(0)
            running_loss += float(loss.item()) * bs
            running_correct += float(acc) * bs
            total += bs

        train_loss = running_loss / total if total else 0.0
        train_acc = running_correct / total if total else 0.0

        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"[Teacher {epoch:03d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:6.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc*100:6.2f}%"
        )

        # Save best teacher checkpoint.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "val_acc": float(best_val_acc),
                    "temperature_train": float(TEMPERATURE_TEACHER),
                    "vim_kwargs": vim_kwargs,
                },
                best_path,
            )
            print(f"  -> saved best teacher (val_acc={best_val_acc*100:.2f}%)")
        else:
            patience += 1

        # Optional early stop when reaching a target accuracy.
        if best_val_acc >= TARGET_ACC:
            print(f"[EARLY STOP] target accuracy reached: {best_val_acc*100:.2f}% >= {TARGET_ACC*100:.2f}%")
            break

        if patience >= EARLY_STOP_PATIENCE:
            print(f"[EARLY STOP] no improvement for {EARLY_STOP_PATIENCE} epochs")
            break

    df = pd.DataFrame(history)
    df.to_csv(save_dir / "teacher_training_history.csv", index=False)
    plot_history(df, save_dir / "teacher_training_history.png", "Teacher Training History")

    # Load best checkpoint before returning (important for student distillation).
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()

    return model


# -----------------------------------------------------------------------------
# 7) Training: student
# -----------------------------------------------------------------------------
def train_student_model(
    teacher_model: nn.Module,
    student_model: nn.Module,
    student_vim_kwargs: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    save_dir: Path,
) -> nn.Module:
    """Train student using defensive distillation (teacher frozen)."""
    print("\n" + "=" * 60)
    print("TRAINING STUDENT MODEL (defensive distillation)")
    print("=" * 60)

    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    criterion = DistillationLoss(temperature=TEMPERATURE_STUDENT, alpha=ALPHA)
    val_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS_STUDENT)

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    patience = 0

    best_path = save_dir / "best_student_model.pth"

    for epoch in range(1, MAX_EPOCHS_STUDENT + 1):
        student_model.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            student_logits = student_model(images)
            with torch.no_grad():
                teacher_logits = teacher_model(images)

            loss = criterion(student_logits, teacher_logits, targets)
            loss.backward()
            optimizer.step()

            acc = accuracy_from_logits(student_logits, targets)
            bs = targets.size(0)
            running_loss += float(loss.item()) * bs
            running_correct += float(acc) * bs
            total += bs

        train_loss = running_loss / total if total else 0.0
        train_acc = running_correct / total if total else 0.0

        val_loss, val_acc = evaluate_model(student_model, val_loader, val_criterion, device)
        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"[Student {epoch:03d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:6.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc*100:6.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model": student_model.state_dict(),
                    "val_acc": float(best_val_acc),
                    "temperature_train": float(TEMPERATURE_STUDENT),
                    "temperature_inference": 1.0,
                    "alpha": float(ALPHA),
                    "vim_kwargs": student_vim_kwargs,
                },
                best_path,
            )
            print(f"  -> saved best student (val_acc={best_val_acc*100:.2f}%)")
        else:
            patience += 1

        if best_val_acc >= TARGET_ACC:
            print(f"[EARLY STOP] target accuracy reached: {best_val_acc*100:.2f}% >= {TARGET_ACC*100:.2f}%")
            break

        if patience >= EARLY_STOP_PATIENCE:
            print(f"[EARLY STOP] no improvement for {EARLY_STOP_PATIENCE} epochs")
            break

    df = pd.DataFrame(history)
    df.to_csv(save_dir / "student_training_history.csv", index=False)
    plot_history(df, save_dir / "student_training_history.png", "Student Training History")

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        student_model.load_state_dict(ckpt["model"])
        student_model.eval()

    return student_model


# -----------------------------------------------------------------------------
# 8) Main
# -----------------------------------------------------------------------------
def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    print(f"[DEVICE] {device}")
    print(f"[DATA]   {DATA_ROOT.resolve()}")
    print(
        f"[HP] batch={BATCH_SIZE} lr={BASE_LR} wd={WEIGHT_DECAY} "
        f"T_teacher={TEMPERATURE_TEACHER} T_student={TEMPERATURE_STUDENT} alpha={ALPHA}"
    )

    train_loader, val_loader, idx_to_class = get_dataloaders()
    num_classes = len(idx_to_class)

    save_dir = create_save_dir()
    print(f"[OUTPUT] {save_dir.resolve()}")

    # Save config and class mapping for reproducibility.
    teacher_vim_kwargs = build_vim_kwargs(num_classes=num_classes, img_size=IMG_SIZE)
    student_vim_kwargs = build_vim_kwargs(num_classes=num_classes, img_size=IMG_SIZE)

    config = {
        "data_root": str(DATA_ROOT),
        "num_classes": num_classes,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "max_epochs_teacher": MAX_EPOCHS_TEACHER,
        "max_epochs_student": MAX_EPOCHS_STUDENT,
        "temperature_teacher": TEMPERATURE_TEACHER,
        "temperature_student": TEMPERATURE_STUDENT,
        "alpha": ALPHA,
        "base_lr": BASE_LR,
        "weight_decay": WEIGHT_DECAY,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "target_acc": TARGET_ACC,
        "seed": SEED,
        "defense_method": "defensive_distillation",
        "reference": "Papernot et al. (2016) - Distillation as a Defense to Adversarial Perturbations",
        "teacher_vim_kwargs": teacher_vim_kwargs,
        "student_vim_kwargs": student_vim_kwargs,
    }

    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    with open(save_dir / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"idx_to_class": {str(k): v for k, v in idx_to_class.items()}}, f, indent=2)

    # Step 1) Train teacher
    teacher = Vim(**teacher_vim_kwargs).to(device)
    teacher = train_teacher_model(teacher, teacher_vim_kwargs, train_loader, val_loader, device, save_dir)

    # Step 2) Train student (distillation)
    student = Vim(**student_vim_kwargs).to(device)
    student = train_student_model(teacher, student, student_vim_kwargs, train_loader, val_loader, device, save_dir)

    # Final evaluation: clean accuracy on validation set
    clean_acc = evaluate_clean_accuracy(student, val_loader, device)
    print(f"[RESULT] clean_accuracy={clean_acc*100:.2f}%")

    print("\n[DONE] Artifacts:")
    print(f" - {save_dir / 'best_teacher_model.pth'}")
    print(f" - {save_dir / 'best_student_model.pth'}")
    print(f" - {save_dir / 'teacher_training_history.csv'}")
    print(f" - {save_dir / 'student_training_history.csv'}")
    print(f" - {save_dir / 'config.json'}")
    print(f" - {save_dir / 'class_mapping.json'}")


if __name__ == "__main__":
    main()
