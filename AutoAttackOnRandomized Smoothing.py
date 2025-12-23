#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os, json, time, warnings, inspect, contextlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union

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
# CONFIG (RS VIM final paths)
# =========================================================
class CFG:
    # ---- RS fine-tune output terbaru (sesuai user)
    RS_DIR = Path("Randomized Smoothing Fine-tuning_new")
    CKPT_PATH = RS_DIR / "final_certified_model.pth"
    VIM_KWARGS_PATH = RS_DIR / "vim_kwargs.json"
    CLASSES_PATH = RS_DIR / "classes.json"

    # dataset
    DATA_ROOT = Path("dataset_rambu_lalu_lintas")
    TEST_DIR  = DATA_ROOT / "test"
    VAL_DIR   = DATA_ROOT / "valid"

    # output
    OUT_DIR = Path("AutoAttack_on_RS_VIM_FINAL")

    # AutoAttack config
    EPS_LIST = [0.5, 1, 2, 3, 4, 5, 6, 7, 8]  # eps in /255 space
    AA_VERSION = "standard"
    AA_ATTACKS = ["apgd-ce", "apgd-dlr", "fab-t", "square"]

    # perf knobs
    BATCH_SIZE = 16
    AA_BS = 16
    MAX_SAMPLES = 16     # percepat; boleh 8/12/16/32/64
    NUM_WORKERS = 0

    # normalization: RS training Anda pakai NormalizeWrapper (ImageNet mean/std)
    USE_IMAGENET_NORM = True
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    SILENCE_MODEL_PRINTS = True
    TQDM_MININTERVAL = 0.25

    # Safety: jangan lanjut kalau num_classes mismatch (hindari angka palsu)
    STRICT_NUM_CLASSES_CHECK = True


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

def read_json_any(path: Path) -> Any:
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
    out = {}
    for k, v in sd.items():
        if k.startswith("module."):
            out[k.replace("module.", "", 1)] = v
        else:
            out[k] = v
    return out

def pick_img_size_from_vim_kwargs(vim_kwargs: Dict[str, Any]) -> Tuple[int, int]:
    # training script Anda pakai image_size atau img_size
    v = vim_kwargs.get("image_size", vim_kwargs.get("img_size", 224))
    if isinstance(v, (list, tuple)) and len(v) == 2:
        h, w = int(v[0]), int(v[1])
        return (h, w)
    s = int(v)
    return (s, s)

def import_vim_class():
    # ikut training: from vision_mamba import Vim
    from vision_mamba import Vim
    return Vim

def build_vim_from_kwargs(vim_kwargs: Dict[str, Any]) -> nn.Module:
    Vim = import_vim_class()
    sig = inspect.signature(Vim).parameters
    kwargs = {k: v for k, v in dict(vim_kwargs).items() if k in sig}
    print("[MODEL] Vim kwargs(filtered):", kwargs)
    return Vim(**kwargs)

def extract_state_dict_any(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "model", "net", "weights"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        # ckpt langsung sd?
        if ckpt and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt  # type: ignore
    raise RuntimeError("Checkpoint tidak berisi state_dict yang dikenali.")

def maybe_strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    hits = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    return hits if len(hits) > 0 else sd


# =========================================================
# Dataset (PIL) dengan urutan kelas dari classes.json (ANTI label ketukar)
# =========================================================
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class ClassOrderedImageFolder(Dataset):
    def __init__(self, root: Path, classes: List[str], img_size_hw: Tuple[int, int]):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Folder dataset tidak ditemukan: {self.root}")

        self.classes = list(classes)
        if not self.classes:
            raise RuntimeError("classes kosong (classes.json invalid).")

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.h, self.w = int(img_size_hw[0]), int(img_size_hw[1])

        samples: List[Tuple[Path, int]] = []
        missing_dirs = 0
        for c in self.classes:
            cdir = self.root / c
            if not cdir.exists():
                missing_dirs += 1
                continue
            files = [p for p in cdir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
            files.sort()
            cid = self.class_to_idx[c]
            samples.extend([(p, cid) for p in files])

        if missing_dirs > 0:
            print(f"[WARN] Ada {missing_dirs} folder kelas yang tidak ada di split ini ({self.root}).")

        if not samples:
            raise RuntimeError(f"Tidak ada file gambar di: {self.root}")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = img.resize((self.w, self.h), resample=Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC [0,1]
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW
        return x, int(y)

def make_loader(classes: List[str], img_hw: Tuple[int,int], device: torch.device) -> Tuple[DataLoader, Dataset, int, Path]:
    if CFG.TEST_DIR.exists():
        ds = ClassOrderedImageFolder(CFG.TEST_DIR, classes=classes, img_size_hw=img_hw)
        data_dir = CFG.TEST_DIR
    elif CFG.VAL_DIR.exists():
        ds = ClassOrderedImageFolder(CFG.VAL_DIR, classes=classes, img_size_hw=img_hw)
        data_dir = CFG.VAL_DIR
    else:
        raise FileNotFoundError(f"Tidak ada folder test/ atau valid/ di: {CFG.DATA_ROOT.resolve()}")

    num_classes = len(classes)

    if CFG.MAX_SAMPLES is not None and len(ds) > int(CFG.MAX_SAMPLES):
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(len(ds), generator=g)[: int(CFG.MAX_SAMPLES)].tolist()
        ds = Subset(ds, idx)

    loader = DataLoader(
        ds,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    print(f"[DATA] dir={data_dir} | N={len(ds)} | classes={num_classes} | img={img_hw}")
    return loader, ds, num_classes, data_dir

def collect_all(loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    x_all = torch.cat(xs, dim=0).to(device, non_blocking=True)
    y_all = torch.cat(ys, dim=0).to(device, non_blocking=True)
    return x_all, y_all


# =========================================================
# Model wrappers (match training behavior)
# =========================================================
class NormalizeWrapper(nn.Module):
    """
    Match RS training script Anda:
    - clamp ke [0,1]
    - (x-mean)/std
    - forward ke core
    State dict (jika disave wrapper): keys biasanya 'core.*', 'mean', 'std'
    """
    def __init__(self, core: nn.Module, mean: List[float], std: List[float]):
        super().__init__()
        self.core = core
        m = torch.tensor(mean).view(1, 3, 1, 1)
        s = torch.tensor(std).view(1, 3, 1, 1)
        self.register_buffer("mean", m)
        self.register_buffer("std", s)

    def forward(self, x):
        x = torch.clamp(x, 0.0, 1.0)
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

    # validate paths
    assert CFG.CKPT_PATH.exists(), f"CKPT tidak ada: {CFG.CKPT_PATH}"
    assert CFG.VIM_KWARGS_PATH.exists(), f"vim_kwargs.json tidak ada: {CFG.VIM_KWARGS_PATH}"
    assert CFG.CLASSES_PATH.exists(), f"classes.json tidak ada: {CFG.CLASSES_PATH}"

    CFG.OUT_DIR.mkdir(parents=True, exist_ok=True)

    # load meta
    vim_kwargs = read_json_any(CFG.VIM_KWARGS_PATH)
    classes = read_json_any(CFG.CLASSES_PATH)
    if not isinstance(vim_kwargs, dict):
        raise RuntimeError("vim_kwargs.json bukan dict.")
    if not (isinstance(classes, list) and all(isinstance(c, str) for c in classes)):
        raise RuntimeError("classes.json harus list[str].")

    num_classes = len(classes)
    img_hw = pick_img_size_from_vim_kwargs(vim_kwargs)

    # strict check: num_classes in kwargs vs classes.json
    if CFG.STRICT_NUM_CLASSES_CHECK and ("num_classes" in vim_kwargs):
        if int(vim_kwargs["num_classes"]) != num_classes:
            raise RuntimeError(
                f"STOP: num_classes mismatch! vim_kwargs.num_classes={vim_kwargs['num_classes']} "
                f"vs len(classes.json)={num_classes}. Ini bisa bikin label ketukar."
            )

    print(f"[META] num_classes={num_classes} | img_size={img_hw} | classes(sample)={classes[:8]}")

    # dataset
    loader, ds, _, data_dir = make_loader(classes=classes, img_hw=img_hw, device=device)

    # build model core
    core = build_vim_from_kwargs(vim_kwargs).to(device)

    # wrap normalization (match training)
    model_logits: nn.Module = core
    if CFG.USE_IMAGENET_NORM:
        model_logits = NormalizeWrapper(model_logits, CFG.IMAGENET_MEAN, CFG.IMAGENET_STD)
        print("[NORM] ImageNet wrapper ON (clamp+normalize inside model)")

    model_logits = SilentForwardWrapper(model_logits, enabled=CFG.SILENCE_MODEL_PRINTS).to(device).eval()

    # load checkpoint (handle wrapper/core prefix safely)
    ckpt = safe_torch_load(CFG.CKPT_PATH, device=torch.device("cpu"))
    sd = strip_module(extract_state_dict_any(ckpt))

    # Jika checkpoint disimpan dari wrapper NormalizeWrapper, keys biasanya:
    # - 'mean', 'std', 'core.xxx'
    # Jika disimpan dari core saja, keys biasanya langsung cocok.
    # Kita coba load ke model_logits (wrapper) dulu; kalau gagal berat, fallback strip 'core.' dan load ke core.
    loaded_ok = False
    try:
        incompat = model_logits.core.load_state_dict(sd, strict=False) if isinstance(model_logits, SilentForwardWrapper) and isinstance(model_logits.core, NormalizeWrapper) else None
        # ^ trik: kalau model_logits = Silent(Normalize(core)), kita akan load full sd ke wrapper di bawah (lebih benar).
    except Exception:
        incompat = None

    # 1) coba load ke wrapper NormalizeWrapper (paling cocok jika sd punya mean/std/core.*)
    try:
        # unwrap Silent
        inner = model_logits.core if isinstance(model_logits, SilentForwardWrapper) else model_logits
        incompat = inner.load_state_dict(sd, strict=False)  # type: ignore
        print(f"[LOAD] into wrapper-model | missing={len(incompat.missing_keys)} | unexpected={len(incompat.unexpected_keys)}")
        if len(incompat.missing_keys) > 0: print("  missing sample:", incompat.missing_keys[:10])
        if len(incompat.unexpected_keys) > 0: print("  unexpected sample:", incompat.unexpected_keys[:10])
        loaded_ok = True
    except Exception as e1:
        print("[LOAD] gagal load ke wrapper, coba load ke core dengan strip 'core.' ...")
        # 2) fallback: strip core. lalu load ke core
        sd2 = maybe_strip_prefix(sd, "core.")
        try:
            incompat2 = core.load_state_dict(sd2, strict=False)
            print(f"[LOAD] into core | missing={len(incompat2.missing_keys)} | unexpected={len(incompat2.unexpected_keys)}")
            if len(incompat2.missing_keys) > 0: print("  missing sample:", incompat2.missing_keys[:10])
            if len(incompat2.unexpected_keys) > 0: print("  unexpected sample:", incompat2.unexpected_keys[:10])
            loaded_ok = True
        except Exception as e2:
            raise RuntimeError(f"Gagal load checkpoint ke wrapper maupun core.\n- wrapper err: {e1}\n- core err: {e2}")

    if not loaded_ok:
        raise RuntimeError("Gagal load checkpoint (loaded_ok False).")

    # cache all samples (kecil)
    x_all, y_all = collect_all(loader, device)
    print(f"[CACHE] dir={data_dir} | N={int(x_all.size(0))}")

    # clean acc
    t0 = time.time()
    pred_clean = forward_preds(model_logits, x_all, bs=CFG.BATCH_SIZE)
    clean_acc = 100.0 * (pred_clean == y_all).float().mean().item()
    print(f"[CLEAN] acc={clean_acc:.2f}% | time={(time.time()-t0):.2f}s")

    tag = now_tag()
    rows = []

    eps_pbar = tqdm(CFG.EPS_LIST, desc="AutoAttack (RS-Vim)", mininterval=CFG.TQDM_MININTERVAL)
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
            "epsilon": float(eps_val),
            "clean_acc": float(clean_acc),
            "adv_acc": float(adv_acc),
            "asr": float(asr),
            "seconds": float(sec),
            "aa_bs": int(used_bs),
            "n_samples": int(x_all.size(0)),
            "ckpt": str(CFG.CKPT_PATH),
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
    plt.title("RS Fine-tuned Vision Mamba Robustness vs AutoAttack (FAST SAMPLE)")
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
