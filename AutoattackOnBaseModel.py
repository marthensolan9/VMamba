#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os, json, re, math, time, warnings, inspect
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import contextlib

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
torch.set_printoptions(threshold=50, edgeitems=2, linewidth=120, sci_mode=False)

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
# CONFIG (eps sama, tujuan AutoAttack sama; dipercepat via sample kecil + 1-tensor)
# =========================================================
class CFG:
    # ====== HANYA BAGIAN PATH TARGET MODEL YANG DIGANTI ======
    BASE_DIR = Path("outputs_vim_rambu_small_20251119_220259")
    CKPT_CANDIDATES = [
        BASE_DIR / "best_vim_rambu_small.pth",
    ]
    CONFIG_CANDIDATES = [
        BASE_DIR / "config.json",
    ]
    CLASSMAP_PATH = BASE_DIR / "class_mapping.json"
    # ====== END PATH TARGET MODEL ======

    DATA_ROOT = Path("dataset_rambu_lalu_lintas")
    TEST_DIR  = DATA_ROOT / "test"
    VAL_DIR   = DATA_ROOT / "valid"

    OUT_DIR = Path("Auto_attack_on_Model_Adversarial Train New1")

    # EPSILON TETAP
    EPS_LIST = [0.5, 1, 2, 3, 4, 5, 6, 7, 8]

    # AutoAttack standard suite (tetap)
    AA_VERSION = "standard"
    AA_ATTACKS = ["apgd-ce", "apgd-dlr", "fab-t", "square"]

    # Percepat: sample kecil + single-tensor evaluation
    IMG_SIZE = 224
    BATCH_SIZE = 64          # untuk clean/adv forward (bisa 64)
    AA_BS = 16               # AA internal batch
    MAX_SAMPLES = 64         # <<< TURUNKAN LAGI kalau masih lama: 8 / 12 / 16
    NUM_WORKERS = 0          # lebih stabil di Jupyter

    USE_IMAGENET_NORM = True
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    STRICT_LOAD_THRESHOLD = 300

    # Matikan saving gambar untuk ngebut (tetap save CSV + plot)
    SAVE_VIS = False

    # Anti hang: bunuh print tensor dari model internal
    SILENCE_MODEL_PRINTS = True

    TQDM_MININTERVAL = 0.25


# =========================================================
# Utils
# =========================================================
def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def pick_existing(paths: List[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError("Tidak ada checkpoint ditemukan dari CKPT_CANDIDATES. Cek path kamu.")

def pick_existing_or_none(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None

def safe_torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)  # type: ignore
    except TypeError:
        return torch.load(path, map_location=device)

def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict):
        for k in ["model_state_dict", "state_dict", "model", "net", "weights"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
        if ckpt_obj and all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj
        raise RuntimeError("Checkpoint dict tidak berisi state_dict yang dikenali.")
    if hasattr(ckpt_obj, "state_dict"):
        return ckpt_obj.state_dict()
    raise RuntimeError("Format checkpoint tidak didukung.")

def strip_module(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k.replace("module.", ""): v for k, v in sd.items()}

def try_get(cfg: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in cfg:
            return cfg[k]
    return default

@contextlib.contextmanager
def suppress_stdout_stderr(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as fnull:
        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            yield


# =========================================================
# Dataset
# =========================================================
def make_loader(img_size: int, device: torch.device) -> Tuple[DataLoader, Any, int]:
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    if CFG.TEST_DIR.exists():
        ds = datasets.ImageFolder(str(CFG.TEST_DIR), transform=tfm)
        data_dir = CFG.TEST_DIR
    elif CFG.VAL_DIR.exists():
        ds = datasets.ImageFolder(str(CFG.VAL_DIR), transform=tfm)
        data_dir = CFG.VAL_DIR
    else:
        raise FileNotFoundError(f"Tidak ada folder test/ atau valid/ di: {CFG.DATA_ROOT.resolve()}")

    if len(ds) == 0:
        raise RuntimeError(f"Dataset kosong: {data_dir.resolve()}")

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


# =========================================================
# Vim build (infer)
# =========================================================
def import_vim_class():
    from vision_mamba.model import Vim
    return Vim

def infer_depth_from_sd(sd: Dict[str, torch.Tensor]) -> Optional[int]:
    pat = re.compile(r"^layers\.(\d+)\.")
    idxs = []
    for k in sd.keys():
        m = pat.match(k)
        if m:
            idxs.append(int(m.group(1)))
    return (max(idxs) + 1) if idxs else None

def infer_patch_size_from_sd(sd: Dict[str, torch.Tensor]) -> Optional[int]:
    for k, v in sd.items():
        if k.endswith("to_patch_embedding.1.weight") and v.ndim == 2:
            in_features = int(v.shape[1])
            if in_features % 3 == 0:
                p2 = in_features // 3
                p = int(round(math.sqrt(p2)))
                if p * p == p2:
                    return p
    return None

def infer_dim_dt_from_sd(sd: Dict[str, torch.Tensor]) -> Dict[str, Optional[int]]:
    dim = None
    dt_rank = None
    dim_inner = None
    d_state = None

    for k, v in sd.items():
        if k.endswith("ssm.dt_proj_layer.weight") and v.ndim == 2:
            dim_inner = int(v.shape[0])
            dt_rank = int(v.shape[1])
            break

    for k, v in sd.items():
        if k.endswith("ssm.deltaBC_layer.weight") and v.ndim == 2:
            out_features = int(v.shape[0])
            in_features = int(v.shape[1])
            dim = in_features
            if dt_rank is not None:
                rem = out_features - dt_rank
                if rem > 0 and rem % 2 == 0:
                    d_state = rem // 2
            break

    return {"dim": dim, "dt_rank": dt_rank, "dim_inner": dim_inner, "d_state": d_state}

def build_vim_true(num_classes: int, cfg_json: Dict[str, Any], sd: Dict[str, torch.Tensor]) -> nn.Module:
    Vim = import_vim_class()

    image_size = int(try_get(cfg_json, ["image_size", "img_size", "input_size"], CFG.IMG_SIZE))
    patch_size = try_get(cfg_json, ["patch_size"], None)
    channels = int(try_get(cfg_json, ["channels", "in_chans"], 3))
    dropout = float(try_get(cfg_json, ["dropout", "drop_rate"], 0.0))
    depth = try_get(cfg_json, ["depth", "num_layers", "n_layers"], None)

    infer = infer_dim_dt_from_sd(sd)
    dim = infer["dim"]
    dt_rank = infer["dt_rank"]
    dim_inner = infer["dim_inner"]
    d_state = infer["d_state"]

    if patch_size is None:
        patch_size = infer_patch_size_from_sd(sd) or 16
    patch_size = int(patch_size)

    if depth is None:
        depth = infer_depth_from_sd(sd) or 24
    depth = int(depth)

    if dim is None:
        for k, v in sd.items():
            if k.endswith("to_patch_embedding.1.weight") and v.ndim == 2:
                dim = int(v.shape[0])
                break
    if dim is None:
        dim = 192
    if dt_rank is None:
        dt_rank = min(16, max(4, dim // 8))
    if dim_inner is None:
        dim_inner = int(2 * dim)
    if d_state is None:
        d_state = 16

    print("[BUILD] Vim params:")
    print(f"  dim={dim} | dt_rank={dt_rank} | dim_inner={dim_inner} | d_state={d_state}")
    print(f"  image_size={image_size} | patch_size={patch_size} | channels={channels} | dropout={dropout} | depth={depth} | num_classes={num_classes}")

    sig = inspect.signature(Vim.__init__)
    allowed = set(sig.parameters.keys())
    kwargs = {}
    for k, v in {
        "dim": dim,
        "dt_rank": dt_rank,
        "dim_inner": dim_inner,
        "d_state": d_state,
        "num_classes": num_classes,
        "image_size": image_size,
        "patch_size": patch_size,
        "channels": channels,
        "dropout": dropout,
        "depth": depth,
    }.items():
        if k in allowed:
            kwargs[k] = v
    return Vim(**kwargs)


# =========================================================
# Attack helpers (single-tensor)
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
        print(f"[GPU] {torch.cuda.get_device_properties(0).name} | {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")

    CFG.OUT_DIR.mkdir(parents=True, exist_ok=True)

    loader, ds, num_classes = make_loader(CFG.IMG_SIZE, device)

    ckpt_path = pick_existing(CFG.CKPT_CANDIDATES)
    cfg_path = pick_existing_or_none(CFG.CONFIG_CANDIDATES)
    cfg_json = read_json(cfg_path) if cfg_path else {}

    print(f"[CKPT] {ckpt_path}")
    print(f"[CFG ] {cfg_path}" if cfg_path else "[CFG ] tidak ditemukan (OK, infer dari state_dict).")
    print(f"[CLASSMAP] {CFG.CLASSMAP_PATH}")

    ckpt_obj = safe_torch_load(ckpt_path, device)
    sd = strip_module(extract_state_dict(ckpt_obj))

    model_core = build_vim_true(num_classes=num_classes, cfg_json=cfg_json, sd=sd)
    incompat = model_core.load_state_dict(sd, strict=False)
    miss = len(incompat.missing_keys)
    unex = len(incompat.unexpected_keys)
    print(f"[LOAD] missing_keys={miss} | unexpected_keys={unex}")
    if miss + unex > CFG.STRICT_LOAD_THRESHOLD:
        print("[FATAL] mismatch besar antara checkpoint vs arsitektur Vim yang terbentuk.")
        print("  contoh missing:", incompat.missing_keys[:10])
        print("  contoh unexpected:", incompat.unexpected_keys[:10])
        raise RuntimeError("STOP agar tidak menghasilkan angka palsu.")

    if CFG.USE_IMAGENET_NORM:
        model = NormalizeWrapper(model_core, CFG.IMAGENET_MEAN, CFG.IMAGENET_STD)
        print("[NORM] Wrapper ImageNet mean/std aktif.")
    else:
        model = model_core
        print("[NORM] Wrapper OFF.")

    model = SilentForwardWrapper(model, enabled=CFG.SILENCE_MODEL_PRINTS)
    model.to(device).eval()

    # Kumpulkan semua sample jadi 1 tensor (super cepat untuk loop eps)
    x_all, y_all = collect_all(loader, device)
    print(f"[CACHE] x_all={tuple(x_all.shape)} | y_all={tuple(y_all.shape)}")

    # Clean acc (cepat)
    t0 = time.time()
    pred_clean = forward_preds(model, x_all, bs=CFG.BATCH_SIZE)
    clean_acc = 100.0 * (pred_clean == y_all).float().mean().item()
    print(f"[CLEAN] acc={clean_acc:.2f}% | time={(time.time()-t0):.2f}s")

    tag = now_tag()
    rows = []

    eps_pbar = tqdm(CFG.EPS_LIST, desc="Epoch(Epsilon) AutoAttack", mininterval=CFG.TQDM_MININTERVAL)
    for i, eps_val in enumerate(eps_pbar, start=1):
        eps_pbar.set_postfix({"epoch": f"{i}/{len(CFG.EPS_LIST)}", "eps/255": f"{eps_val:g}"})
        eps_norm = float(eps_val) / 255.0

        adversary = AutoAttack(model, norm="Linf", eps=eps_norm, version=CFG.AA_VERSION, device=device)
        adversary.verbose = False
        adversary.attacks_to_run = list(CFG.AA_ATTACKS)

        start = time.time()
        x_adv, used_bs = attack_full_with_oom_fallback(adversary, x_all, y_all, CFG.AA_BS, device)

        with torch.no_grad():
            pred_adv = forward_preds(model, x_adv, bs=CFG.BATCH_SIZE)

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
        })

        if device.type == "cuda":
            torch.cuda.empty_cache()

    import pandas as pd
    df = pd.DataFrame(rows)
    csv_path = CFG.OUT_DIR / f"autoattack_report_{tag}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[SAVE] CSV: {csv_path}")

    # Plot
    plt.figure(figsize=(11, 6))
    plt.plot(df["epsilon"], df["clean_acc"], marker="o", linewidth=2, label="Clean Accuracy")
    plt.plot(df["epsilon"], df["adv_acc"], marker="o", linewidth=2, label="Adversarial Accuracy (AutoAttack)")
    plt.xlabel("Epsilon (pixel/255)")
    plt.ylabel("Accuracy (%)")
    plt.title("Vision Mamba (Vim) Robustness vs AutoAttack (FAST SAMPLE)")
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
