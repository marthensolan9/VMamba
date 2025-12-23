#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os, json, re, math, time, warnings, inspect
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

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
# CONFIG (sesuaikan seperlunya)
# =========================================================
class CFG:
    # === Kandidat model fine-tuned kamu (pilih yang ada) ===
    CKPT_CANDIDATES = [
        Path("FINAL RESULTS/Finetune_Certied_accuracy/best_certified_vim_rs.pth"),
    ]

    # === Kandidat config untuk membantu build arsitektur (optional) ===
    CONFIG_CANDIDATES = [
        Path("FINAL RESULTS/Finetune_Certied_accuracy/run_config.json"),
    ]

    # === Dataset wajib asli ===
    DATA_ROOT = Path("dataset_rambu_lalu_lintas")
    TEST_DIR  = DATA_ROOT / "test"
    VAL_DIR   = DATA_ROOT / "valid"

    # === Output ===
    OUT_DIR = Path(f"autoattack_time_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # === AutoAttack params ===
    EPS_LIST = [0.5, 1, 2, 3, 4, 5, 6, 7, 8]     # epsilon pixel/255
    AA_VERSION = "standard"
    AA_ATTACKS = ["apgd-ce", "apgd-dlr", "fab-t", "square"]  # standard suite

    # === Runtime untuk GPU 16GB ===
    IMG_SIZE = 224
    BATCH_SIZE = 16
    AA_BS = 16             # internal AA microbatch; auto turun kalau OOM
    MAX_SAMPLES = 64       # batasi agar cepat & tidak OOM; set None untuk full
    NUM_WORKERS = 2

    # === Normalisasi (sesuai script fine tuning kamu) ===
    # AutoAttack harus menerima input [0,1], jadi Normalize dilakukan di model wrapper.
    USE_IMAGENET_NORM = True
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    # === Validasi mismatch checkpoint vs model ===
    STRICT_LOAD_THRESHOLD = 300  # kalau missing+unexpected > ini -> stop (hindari angka palsu)

    # Visual
    SAVE_EPS_EXAMPLES = {0.5, 4, 8}
    SAVE_EXAMPLES_PER_EPS = 4


# =========================================================
# Helper utils
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
    # kompatibel PyTorch 2.6+ (weights_only default berubah)
    try:
        return torch.load(path, map_location=device, weights_only=False)  # type: ignore
    except TypeError:
        return torch.load(path, map_location=device)

def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    """
    Ambil state_dict dari berbagai format checkpoint.
    """
    if isinstance(ckpt_obj, dict):
        for k in ["model_state_dict", "state_dict", "model", "net", "weights"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
        # kalau dict langsung tensor-tensor
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


# =========================================================
# Dataset loader (Wajib ImageFolder — NO FakeData)
# =========================================================
def make_loader(img_size: int, device: torch.device) -> Tuple[DataLoader, Any, int]:
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),     # IMPORTANT: tetap [0,1]
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

    base_classes = ds.classes
    num_classes = len(base_classes)

    if CFG.MAX_SAMPLES is not None and len(ds) > int(CFG.MAX_SAMPLES):
        idx = torch.randperm(len(ds))[: int(CFG.MAX_SAMPLES)].tolist()
        ds = Subset(ds, idx)

    pin = (device.type == "cuda")
    loader = DataLoader(
        ds,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=pin,
        drop_last=False,
    )

    real_len = len(ds)
    print(f"[DATA] dir={data_dir} | N={real_len} | classes={num_classes} | img={img_size}")
    return loader, ds, num_classes


# =========================================================
# Model wrapper untuk normalisasi (AutoAttack input tetap [0,1])
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


# =========================================================
# Import Vim (vision_mamba) + infer params dari state_dict
# =========================================================
def import_vim_class():
    # environment kamu error stack: ~/.local/lib/python3.10/site-packages/vision_mamba/model.py
    from vision_mamba.model import Vim
    return Vim

def infer_depth_from_sd(sd: Dict[str, torch.Tensor]) -> Optional[int]:
    # keys seperti layers.0..., layers.1..., dst
    pat = re.compile(r"^layers\.(\d+)\.")
    idxs = []
    for k in sd.keys():
        m = pat.match(k)
        if m:
            idxs.append(int(m.group(1)))
    if idxs:
        return max(idxs) + 1
    return None

def infer_patch_size_from_sd(sd: Dict[str, torch.Tensor]) -> Optional[int]:
    # to_patch_embedding.1.weight bentuk [dim, 3*P*P]
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
    """
    Infer param SSM dari bentuk layer:
    - deltaBC_layer.weight: [dt_rank+2*d_state, dim]
    - dt_proj_layer.weight: [dim_inner, dt_rank]
    """
    dim = None
    dt_rank = None
    dim_inner = None
    d_state = None

    # cari dt_proj_layer.weight dulu -> dt_rank & dim_inner
    for k, v in sd.items():
        if k.endswith("ssm.dt_proj_layer.weight") and v.ndim == 2:
            dim_inner = int(v.shape[0])
            dt_rank = int(v.shape[1])
            break

    # cari deltaBC_layer.weight -> dim & (dt_rank+2*d_state)
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

    # 1) coba dari config jika ada
    image_size = int(try_get(cfg_json, ["image_size", "img_size", "input_size"], CFG.IMG_SIZE))
    patch_size = try_get(cfg_json, ["patch_size"], None)
    channels = int(try_get(cfg_json, ["channels", "in_chans"], 3))
    dropout = float(try_get(cfg_json, ["dropout", "drop_rate"], 0.0))
    depth = try_get(cfg_json, ["depth", "num_layers", "n_layers"], None)

    # 2) infer dari state_dict (lebih kuat)
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

    # ==== FALLBACK aman kalau sebagian belum ter-infer ====
    # dim wajib ada; kalau tidak, pakai info linear patch embedding
    if dim is None:
        for k, v in sd.items():
            if k.endswith("to_patch_embedding.1.weight") and v.ndim == 2:
                dim = int(v.shape[0])
                break
    if dim is None:
        dim = 192

    if dt_rank is None:
        # heuristik umum: min(16, dim//8) (cukup aman)
        dt_rank = min(16, max(4, dim // 8))
    if dim_inner is None:
        # dim_inner biasanya >= dim; aman: 2*dim (atau dim)
        dim_inner = int(2 * dim)
    if d_state is None:
        # umum: 16 / 64; aman: 16
        d_state = 16

    print("[BUILD] Vim params:")
    print(f"  dim={dim} | dt_rank={dt_rank} | dim_inner={dim_inner} | d_state={d_state}")
    print(f"  image_size={image_size} | patch_size={patch_size} | channels={channels} | dropout={dropout} | depth={depth} | num_classes={num_classes}")

    # pastikan kita kirim argumen yang benar ke signature Vim environment kamu
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

    model = Vim(**kwargs)
    return model


# =========================================================
# Evaluasi & attack helpers
# =========================================================
@torch.no_grad()
def eval_clean(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / max(1, total)

def attack_with_oom_fallback(adversary: AutoAttack, x: torch.Tensor, y: torch.Tensor, aa_bs: int, device: torch.device):
    cur = int(aa_bs)
    while True:
        try:
            outs = []
            n = x.size(0)
            for i in range(0, n, cur):
                xb = x[i:i+cur]
                yb = y[i:i+cur]
                x_adv_b = adversary.run_standard_evaluation(xb, yb, bs=xb.size(0))
                outs.append(x_adv_b.detach())
            return torch.cat(outs, dim=0), cur
        except RuntimeError as e:
            if device.type == "cuda" and "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                if cur <= 1:
                    raise
                cur = max(1, cur // 2)
                print(f"[OOM] AA_BS turun -> {cur}")
                continue
            raise

def save_examples_grid(path: Path, clean: torch.Tensor, adv: torch.Tensor, y: torch.Tensor, pc: torch.Tensor, pa: torch.Tensor, eps_val: float, max_n: int = 4):
    n = min(max_n, clean.size(0))
    clean_np = clean[:n].permute(0, 2, 3, 1).cpu().numpy()
    adv_np   = adv[:n].permute(0, 2, 3, 1).cpu().numpy()

    fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
    fig.suptitle(f"AutoAttack Examples (eps={eps_val:.1f}/255)", fontsize=14, fontweight="bold")
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for i in range(n):
        axes[0, i].imshow(np.clip(clean_np[i], 0, 1))
        axes[0, i].set_title(f"Clean\nT:{int(y[i])} P:{int(pc[i])}")
        axes[0, i].axis("off")

        axes[1, i].imshow(np.clip(adv_np[i], 0, 1))
        axes[1, i].set_title(f"Adv\nT:{int(y[i])} P:{int(pa[i])}")
        axes[1, i].axis("off")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def save_heatmap(path: Path, clean: torch.Tensor, adv: torch.Tensor, eps_val: float):
    c = clean[0].cpu()
    a = adv[0].cpu()
    pert = (a - c).abs().mean(dim=0).numpy()
    pert = (pert - pert.min()) / (pert.max() - pert.min() + 1e-8)

    c_np = c.permute(1, 2, 0).numpy()
    a_np = a.permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Perturbation Heatmap (eps={eps_val:.1f}/255)", fontsize=14, fontweight="bold")

    axes[0].imshow(np.clip(c_np, 0, 1)); axes[0].set_title("Clean"); axes[0].axis("off")
    axes[1].imshow(np.clip(a_np, 0, 1)); axes[1].set_title("Adversarial"); axes[1].axis("off")
    im = axes[2].imshow(pert, cmap="hot", vmin=0, vmax=1)
    axes[2].set_title("|Δ| Heatmap"); axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    (CFG.OUT_DIR / "examples").mkdir(exist_ok=True)
    (CFG.OUT_DIR / "heatmaps").mkdir(exist_ok=True)

    # Dataset TRUE
    loader, ds, num_classes = make_loader(CFG.IMG_SIZE, device)

    # Pick checkpoint + config
    ckpt_path = pick_existing(CFG.CKPT_CANDIDATES)
    cfg_path = pick_existing_or_none(CFG.CONFIG_CANDIDATES)
    cfg_json = read_json(cfg_path) if cfg_path else {}

    print(f"[CKPT] {ckpt_path}")
    if cfg_path:
        print(f"[CFG ] {cfg_path}")
    else:
        print("[CFG ] tidak ditemukan (OK, akan infer dari state_dict).")

    # Load checkpoint -> state_dict
    ckpt_obj = safe_torch_load(ckpt_path, device)
    sd = strip_module(extract_state_dict(ckpt_obj))

    # === Build Vim TRUE (fix dt_rank/d_state/dim_inner) ===
    model_core = build_vim_true(num_classes=num_classes, cfg_json=cfg_json, sd=sd)

    # Load weights + mismatch check
    incompat = model_core.load_state_dict(sd, strict=False)
    miss = len(incompat.missing_keys)
    unex = len(incompat.unexpected_keys)
    print(f"[LOAD] missing_keys={miss} | unexpected_keys={unex}")
    if miss + unex > CFG.STRICT_LOAD_THRESHOLD:
        print("[FATAL] mismatch besar antara checkpoint vs arsitektur Vim yang terbentuk.")
        print("  contoh missing:", incompat.missing_keys[:10])
        print("  contoh unexpected:", incompat.unexpected_keys[:10])
        raise RuntimeError("STOP agar tidak menghasilkan angka palsu. Perbaiki konfigurasi arsitektur/ckpt.")

    # Wrap normalization (sesuai fine-tuning kamu) — input AutoAttack tetap [0,1]
    if CFG.USE_IMAGENET_NORM:
        model = NormalizeWrapper(model_core, CFG.IMAGENET_MEAN, CFG.IMAGENET_STD)
        print("[NORM] Wrapper ImageNet mean/std aktif.")
    else:
        model = model_core
        print("[NORM] Wrapper OFF (pastikan ini sesuai training kamu).")

    model.to(device).eval()

    # Clean baseline
    t0 = time.time()
    clean_acc = eval_clean(model, loader, device)
    print(f"[CLEAN] acc={clean_acc:.2f}% | time={(time.time()-t0):.1f}s")

    tag = now_tag()
    rows = []

    for eps_val in CFG.EPS_LIST:
        eps_norm = float(eps_val) / 255.0
        print(f"\n[AA] eps={eps_val:.1f}/255 | attacks={CFG.AA_ATTACKS}")

        adversary = AutoAttack(model, norm="Linf", eps=eps_norm, version=CFG.AA_VERSION, device=device)
        adversary.verbose = False
        adversary.attacks_to_run = list(CFG.AA_ATTACKS)

        total = 0
        correct_adv = 0
        used_aa_bs = CFG.AA_BS

        saved = None

        start = time.time()
        for bidx, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            x_adv, used_aa_bs = attack_with_oom_fallback(adversary, x, y, used_aa_bs, device)

            with torch.no_grad():
                pred_adv = model(x_adv).argmax(1)

            correct_adv += (pred_adv == y).sum().item()
            total += y.numel()

            if bidx == 0 and eps_val in CFG.SAVE_EPS_EXAMPLES:
                with torch.no_grad():
                    pred_clean = model(x).argmax(1)
                saved = (x.detach().cpu(), x_adv.detach().cpu(), y.detach().cpu(),
                         pred_clean.detach().cpu(), pred_adv.detach().cpu())

            if device.type == "cuda" and (bidx % 10 == 0):
                torch.cuda.empty_cache()

        sec = time.time() - start
        adv_acc = 100.0 * correct_adv / max(1, total)
        asr = 100.0 - adv_acc
        print(f"[RES] clean={clean_acc:.2f}% | adv={adv_acc:.2f}% | ASR={asr:.2f}% | time={sec:.1f}s | AA_BS={used_aa_bs}")

        rows.append({
            "epsilon": eps_val,
            "clean_acc": clean_acc,
            "adv_acc": adv_acc,
            "asr": asr,
            "seconds": sec,
            "aa_bs": used_aa_bs,
        })

        if saved is not None:
            clean_b, adv_b, y_b, pc_b, pa_b = saved
            grid_path = CFG.OUT_DIR / "examples" / f"examples_eps_{eps_val:.1f}_{tag}.png"
            heat_path = CFG.OUT_DIR / "heatmaps" / f"heatmap_eps_{eps_val:.1f}_{tag}.png"
            save_examples_grid(grid_path, clean_b, adv_b, y_b, pc_b, pa_b, eps_val, max_n=CFG.SAVE_EXAMPLES_PER_EPS)
            save_heatmap(heat_path, clean_b, adv_b, eps_val)

    # Save CSV + plots
    import pandas as pd
    df = pd.DataFrame(rows)
    csv_path = CFG.OUT_DIR / f"autoattack_report_{tag}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[SAVE] CSV: {csv_path}")

    # Plot Acc
    plt.figure(figsize=(11, 6))
    plt.plot(df["epsilon"], df["clean_acc"], marker="o", linewidth=2, label="Clean Accuracy")
    plt.plot(df["epsilon"], df["adv_acc"], marker="o", linewidth=2, label="Adversarial Accuracy (AutoAttack)")
    plt.xlabel("Epsilon (pixel/255)")
    plt.ylabel("Accuracy (%)")
    plt.title("Vision Mamba (Vim) Robustness vs AutoAttack")
    plt.grid(True, alpha=0.3)
    plt.xticks(df["epsilon"])
    plt.ylim(0, 100)
    plt.legend()
    acc_png = CFG.OUT_DIR / f"accuracy_vs_epsilon_{tag}.png"
    plt.tight_layout()
    plt.savefig(acc_png, dpi=160)
    plt.close()
    print(f"[SAVE] Plot: {acc_png}")

    # Plot ASR
    plt.figure(figsize=(11, 6))
    plt.plot(df["epsilon"], df["asr"], marker="o", linewidth=2)
    plt.xlabel("Epsilon (pixel/255)")
    plt.ylabel("Attack Success Rate (%)")
    plt.title("AutoAttack Success Rate (ASR) vs Epsilon")
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
