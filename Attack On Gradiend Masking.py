#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os, json, time, re, math, warnings, inspect, contextlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from PIL import Image

warnings.filterwarnings("ignore")
torch.set_printoptions(threshold=50, edgeitems=2, linewidth=120, sci_mode=False)

# Disable dynamo (mengurangi kejadian aneh terkait import/compile)
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# =========================
# AutoAttack import/install
# =========================
try:
    from autoattack import AutoAttack
except Exception:
    print("[INFO] Installing AutoAttack...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/fra31/auto-attack"])
    from autoattack import AutoAttack


# =========================================================
# CONFIG (Gradient Masking paths)
# =========================================================
class CFG:
    CKPT_PATH = Path("FINAL RESULTS/Gradiend Masking/best_gradient_masking_model.pth")
    GM_CONFIG_PATH = Path("FINAL RESULTS/Gradiend Masking/gradient_masking_config.json")
    ARCH_PATH = Path("FINAL RESULTS/Gradiend Masking/model_architecture.json")
    CLASSMAP_PATH = Path("FINAL RESULTS/Gradiend Masking/class_mapping.json")
    GRAD_STATS_PATH = Path("FINAL RESULTS/Gradiend Masking/gradient_statistics.json")

    DATA_ROOT = Path("dataset_rambu_lalu_lintas")
    TEST_DIR  = DATA_ROOT / "test"
    VAL_DIR   = DATA_ROOT / "valid"

    OUT_DIR = Path("Auto_attack_on_Model_Gradient_Masking")

    EPS_LIST = [0.5, 1, 2, 3, 4, 5, 6, 7, 8]
    AA_VERSION = "standard"
    AA_ATTACKS = ["apgd-ce", "apgd-dlr", "fab-t", "square"]

    IMG_SIZE = 224
    BATCH_SIZE = 16
    AA_BS = 16
    MAX_SAMPLES = 16    # percepat; boleh 8/12/16
    NUM_WORKERS = 0

    USE_IMAGENET_NORM = True
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    SILENCE_MODEL_PRINTS = True
    TQDM_MININTERVAL = 0.25


# =========================================================
# Helpers
# =========================================================
def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def safe_torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)  # type: ignore
    except TypeError:
        return torch.load(path, map_location=device)

def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@contextlib.contextmanager
def suppress_stdout_stderr(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as fnull:
        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            yield

def strip_module(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k.replace("module.", ""): v for k, v in sd.items()}

def try_get(cfg: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in cfg:
            return cfg[k]
    return default


# =========================================================
# Dataset WITHOUT torchvision (PIL-based ImageFolder)
# =========================================================
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class SimpleImageFolder(Dataset):
    def __init__(self, root: Path, img_size: int):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Folder dataset tidak ditemukan: {self.root}")

        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        if not self.classes:
            raise RuntimeError(f"Tidak ada subfolder class di: {self.root}")

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.img_size = int(img_size)

        samples: List[Tuple[Path, int]] = []
        for c in self.classes:
            cdir = self.root / c
            files = [p for p in cdir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
            files.sort()
            cid = self.class_to_idx[c]
            samples.extend([(p, cid) for p in files])

        if not samples:
            raise RuntimeError(f"Tidak ada file gambar di: {self.root}")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = img.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC [0,1]
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW
        return x, int(y)

def make_loader(img_size: int, device: torch.device) -> Tuple[DataLoader, Any, int]:
    if CFG.TEST_DIR.exists():
        ds = SimpleImageFolder(CFG.TEST_DIR, img_size=img_size)
        data_dir = CFG.TEST_DIR
    elif CFG.VAL_DIR.exists():
        ds = SimpleImageFolder(CFG.VAL_DIR, img_size=img_size)
        data_dir = CFG.VAL_DIR
    else:
        raise FileNotFoundError(f"Tidak ada folder test/ atau valid/ di: {CFG.DATA_ROOT.resolve()}")

    num_classes = len(ds.classes)

    if CFG.MAX_SAMPLES is not None and len(ds) > int(CFG.MAX_SAMPLES):
        idx = torch.randperm(len(ds))[: int(CFG.MAX_SAMPLES)].tolist()
        ds = Subset(ds, idx)

    loader = DataLoader(
        ds,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    print(f"[DATA] dir={data_dir} | N={len(ds)} | classes={num_classes} | img={img_size}")
    return loader, ds, num_classes

def collect_all(loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    x_all = torch.cat(xs, dim=0).to(device, non_blocking=True)
    y_all = torch.cat(ys, dim=0).to(device, non_blocking=True)
    return x_all, y_all


# =========================================================
# Model wrappers
# =========================================================
class NormalizeWrapper(nn.Module):
    def __init__(self, core: nn.Module, mean: List[float], std: List[float]):
        super().__init__()
        self.core = core
        m = torch.tensor(mean).view(1, 3, 1, 1)
        s = torch.tensor(std).view(1, 3, 1, 1)
        self.register_buffer("mean", m)
        self.register_buffer("std", s)

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.core(x)

class SilentForwardWrapper(nn.Module):
    def __init__(self, core: nn.Module, enabled: bool = True):
        super().__init__()
        self.core = core
        self.enabled = enabled

    def forward(self, x):
        with suppress_stdout_stderr(self.enabled):
            return self.core(x)

class GradientMaskingEvalWrapper(nn.Module):
    """
    Untuk AutoAttack: output harus LOGITS (bukan softmax probabilities),
    tapi tetap membawa efek temperature scaling (distillation_temperature).
    Ini menjaga AutoAttack valid & tetap merepresentasikan defense (skala grad).
    """
    def __init__(self, base_logits_model: nn.Module, temperature: float = 20.0, use_temp: bool = True):
        super().__init__()
        self.base = base_logits_model
        self.temperature = float(temperature)
        self.use_temp = bool(use_temp)

    def forward(self, x):
        logits = self.base(x)
        if self.use_temp and self.temperature > 0:
            logits = logits / self.temperature
        return logits


# =========================================================
# Vim builder (pakai model_architecture.json dari training)
# =========================================================
def import_vim_class():
    # ikut training script kamu: from vision_mamba import Vim
    from vision_mamba import Vim
    return Vim

def build_vim_from_arch(num_classes: int, arch_json: Dict[str, Any]) -> nn.Module:
    Vim = import_vim_class()
    sig = inspect.signature(Vim).parameters

    # arch_json biasanya berisi vim_kwargs yang disimpan saat training
    kwargs = dict(arch_json) if isinstance(arch_json, dict) else {}

    # pastikan num_classes benar (head classifier)
    kwargs["num_classes"] = int(kwargs.get("num_classes", num_classes))
    if kwargs["num_classes"] != num_classes:
        # lebih aman ikuti checkpoint (biasanya sama)
        pass

    # filter agar hanya argumen yang dikenali Vim
    kwargs = {k: v for k, v in kwargs.items() if k in sig}

    print("[MODEL ARCH] Vim kwargs:", kwargs)
    return Vim(**kwargs)

def extract_base_model_state_dict(wrapper_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    checkpoint kamu: model_state_dict dari GradientMaskingWrapper
    -> key biasanya 'base_model.xxx'
    """
    wrapper_sd = strip_module(wrapper_sd)

    # coba beberapa prefix umum
    prefixes = ["base_model.", "model.base_model.", "module.base_model."]
    for pref in prefixes:
        hits = {k[len(pref):]: v for k, v in wrapper_sd.items() if k.startswith(pref)}
        if len(hits) > 0:
            print(f"[SD] extracted base_model keys using prefix='{pref}' | n={len(hits)}")
            return hits

    # fallback: kalau ternyata state_dict sudah langsung core model
    print("[SD] WARNING: tidak menemukan prefix base_model.*; mencoba pakai state_dict apa adanya")
    return wrapper_sd


# =========================================================
# Eval + AutoAttack
# =========================================================
@torch.no_grad()
def forward_preds(model: nn.Module, x: torch.Tensor, bs: int) -> torch.Tensor:
    preds = []
    for i in range(0, x.size(0), bs):
        preds.append(model(x[i:i+bs]).argmax(1))
    return torch.cat(preds, dim=0)

def attack_full_with_oom_fallback(adversary: AutoAttack, x: torch.Tensor, y: torch.Tensor, aa_bs: int, device: torch.device):
    cur = int(aa_bs)
    while True:
        try:
            with suppress_stdout_stderr(CFG.SILENCE_MODEL_PRINTS):
                x_adv = adversary.run_standard_evaluation(x, y, bs=cur)
            return x_adv.detach(), cur
        except RuntimeError as e:
            if device.type == "cuda" and "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                if cur <= 1:
                    raise
                cur = max(1, cur // 2)
                tqdm.write(f"[OOM] AA_BS turun -> {cur}")
                continue
            raise


# =========================================================
# MAIN
# =========================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        props = torch.cuda.get_device_properties(0)
        print(f"[GPU] {props.name} | {props.total_memory/1e9:.2f} GB")

    # paths existence
    assert CFG.CKPT_PATH.exists(), f"CKPT tidak ada: {CFG.CKPT_PATH}"
    assert CFG.ARCH_PATH.exists(), f"model_architecture.json tidak ada: {CFG.ARCH_PATH}"
    assert CFG.CLASSMAP_PATH.exists(), f"class_mapping.json tidak ada: {CFG.CLASSMAP_PATH}"

    CFG.OUT_DIR.mkdir(parents=True, exist_ok=True)

    # dataset
    loader, ds, _ = make_loader(CFG.IMG_SIZE, device)

    # class mapping -> num_classes
    classmap = read_json(CFG.CLASSMAP_PATH)
    idx_to_class = classmap.get("idx_to_class", None)
    if isinstance(idx_to_class, dict):
        num_classes = len(idx_to_class)
    elif isinstance(idx_to_class, list):
        num_classes = len(idx_to_class)
    else:
        # fallback: gunakan folder dataset classes
        num_classes = len(getattr(ds, "classes", [])) or 0

    if num_classes <= 0:
        raise RuntimeError("Tidak bisa menentukan num_classes dari class_mapping maupun dataset.")

    print(f"[CLASSES] num_classes={num_classes}")

    # build Vim from architecture json
    arch_json = read_json(CFG.ARCH_PATH)
    base_model = build_vim_from_arch(num_classes=num_classes, arch_json=arch_json).to(device)

    # load checkpoint
    ckpt = safe_torch_load(CFG.CKPT_PATH, device)
    if not isinstance(ckpt, dict):
        raise RuntimeError("Checkpoint format tidak dikenali (bukan dict).")

    # get wrapper model_state_dict
    if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        wrapper_sd = ckpt["model_state_dict"]
    elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        wrapper_sd = ckpt["state_dict"]
    else:
        # fallback: mungkin ckpt langsung sd
        wrapper_sd = ckpt

    wrapper_sd = strip_module(wrapper_sd)
    base_sd = extract_base_model_state_dict(wrapper_sd)

    # load to base_model
    incompat = base_model.load_state_dict(base_sd, strict=False)
    print(f"[LOAD] into Vim | missing={len(incompat.missing_keys)} | unexpected={len(incompat.unexpected_keys)}")
    if len(incompat.missing_keys) > 0:
        print("  missing sample:", incompat.missing_keys[:10])
    if len(incompat.unexpected_keys) > 0:
        print("  unexpected sample:", incompat.unexpected_keys[:10])

    # Wrap normalization + gradient masking eval (logits with temperature)
    gm_cfg = read_json(CFG.GM_CONFIG_PATH) if CFG.GM_CONFIG_PATH.exists() else {}
    temperature = float(gm_cfg.get("distillation_temperature", 20))

    model_logits = base_model
    if CFG.USE_IMAGENET_NORM:
        model_logits = NormalizeWrapper(model_logits, CFG.IMAGENET_MEAN, CFG.IMAGENET_STD)
        print("[NORM] ImageNet wrapper ON")

    model_logits = GradientMaskingEvalWrapper(model_logits, temperature=temperature, use_temp=True)
    model_logits = SilentForwardWrapper(model_logits, enabled=CFG.SILENCE_MODEL_PRINTS)
    model_logits.to(device).eval()

    # cache all samples (kecil)
    x_all, y_all = collect_all(loader, device)
    print(f"[CACHE] N={int(x_all.size(0))}")

    # clean acc
    t0 = time.time()
    pred_clean = forward_preds(model_logits, x_all, bs=CFG.BATCH_SIZE)
    clean_acc = 100.0 * (pred_clean == y_all).float().mean().item()
    print(f"[CLEAN] acc={clean_acc:.2f}% | time={(time.time()-t0):.2f}s")

    tag = now_tag()
    rows = []

    eps_pbar = tqdm(CFG.EPS_LIST, desc="Epoch(Epsilon) AutoAttack", mininterval=CFG.TQDM_MININTERVAL)
    for i, eps_val in enumerate(eps_pbar, start=1):
        eps_pbar.set_postfix({"epoch": f"{i}/{len(CFG.EPS_LIST)}", "eps/255": f"{eps_val:g}"})
        eps = float(eps_val) / 255.0

        adversary = AutoAttack(model_logits, norm="Linf", eps=eps, version=CFG.AA_VERSION, device=device)
        adversary.verbose = False
        adversary.attacks_to_run = list(CFG.AA_ATTACKS)

        start = time.time()
        x_adv, used_bs = attack_full_with_oom_fallback(adversary, x_all, y_all, CFG.AA_BS, device)

        with torch.no_grad():
            pred_adv = forward_preds(model_logits, x_adv, bs=CFG.BATCH_SIZE)

        adv_acc = 100.0 * (pred_adv == y_all).float().mean().item()
        asr = 100.0 - adv_acc
        sec = time.time() - start

        tqdm.write(f"[RES] eps={eps_val:g}/255 | clean={clean_acc:.2f}% | adv={adv_acc:.2f}% | ASR={asr:.2f}% | time={sec:.1f}s | AA_BS={used_bs}")

        rows.append({
            "epsilon": eps_val,
            "clean_acc": clean_acc,
            "adv_acc": adv_acc,
            "asr": asr,
            "seconds": sec,
            "aa_bs": used_bs,
            "n_samples": int(x_all.size(0)),
            "temperature": temperature,
        })

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # save csv + plots
    import pandas as pd
    df = pd.DataFrame(rows)
    csv_path = CFG.OUT_DIR / f"autoattack_report_{tag}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[SAVE] CSV: {csv_path}")

    plt.figure(figsize=(11, 6))
    plt.plot(df["epsilon"], df["clean_acc"], marker="o", linewidth=2, label="Clean Accuracy")
    plt.plot(df["epsilon"], df["adv_acc"], marker="o", linewidth=2, label="Adversarial Accuracy (AutoAttack)")
    plt.xlabel("Epsilon (pixel/255)")
    plt.ylabel("Accuracy (%)")
    plt.title("Gradient Masking (Vim base) Robustness vs AutoAttack (FAST SAMPLE)")
    plt.grid(True, alpha=0.3)
    plt.xticks(df["epsilon"])
    plt.ylim(0, 100)
    plt.legend()
    acc_png = CFG.OUT_DIR / f"accuracy_vs_epsilon_{tag}.png"
    plt.tight_layout()
    plt.savefig(acc_png, dpi=160)
    plt.close()
    print(f"[SAVE] Plot: {acc_png}")

    plt.figure(figsize=(11, 6))
    plt.plot(df["epsilon"], df["asr"], marker="o", linewidth=2)
    plt.xlabel("Epsilon (pixel/255)")
    plt.ylabel("Attack Success Rate (%)")
    plt.title("AutoAttack Success Rate (ASR) vs Epsilon (FAST SAMPLE)")
    plt.grid(True, alpha=0.3)
    plt.xticks(df["epsilon"])
    plt.ylim(0, 100)
    asr_png = CFG.OUT_DIR / f"asr_vs_epsilon_{tag}.png"
    plt.tight_layout()
    plt.savefig(asr_png, dpi=160)
    plt.close()
    print(f"[SAVE] Plot: {asr_png}")

    print("\nDONE ✅ Output di:", CFG.OUT_DIR.resolve())


if __name__ == "__main__":
    main()
