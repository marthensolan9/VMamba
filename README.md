<div align="center">

# 🛡️ Vision Mamba Adversarial Robustness

### Enhancing Vision Mamba Robustness Against Adversarial Attacks for Indonesian Traffic Sign Recognition

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)]()

<br/>

**Master's Thesis Research Artifact — Institut Teknologi Bandung**

[📄 View Thesis](#-citation) • [🚀 Quick Start](#-quick-start) • [📊 Results](#-key-results) • [📧 Contact](#-contact)

</div>

---

## 📌 About This Research

> **Problem**: Vision Mamba models are highly vulnerable to adversarial attacks, achieving near-zero accuracy under AutoAttack despite 95%+ clean accuracy.

> **Solution**: We evaluate and compare **5 defense methods** to find the most effective robustness enhancement strategy.

> **Finding**: **Adversarial Training** is the most effective defense, while Gradient Masking and Defensive Distillation show "false robustness" that collapses under adaptive attacks.

<details>
<summary><b>🎯 Click to see Research Objectives</b></summary>

1. Evaluate baseline robustness of Vision Mamba against modern adversarial attacks
2. Implement 5 defense methods: Adversarial Training, Randomized Smoothing, Certified Robustness, Defensive Distillation, Gradient Masking
3. Compare effectiveness using AutoAttack and TRUE Adaptive Attack protocols
4. Provide recommendations for safety-critical TSR applications

</details>

---

## 🔄 Research Pipeline

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         RESEARCH PIPELINE OVERVIEW                          │
└────────────────────────────────────────────────────────────────────────────┘

  ╔═══════════════════════════════════════════════════════════════════════╗
  ║  PHASE 1: DATA & BASELINE                                              ║
  ╠═══════════════════════════════════════════════════════════════════════╣
  ║                                                                        ║
  ║   🖼️ Dataset ──▶ ⚙️ Preprocessing ──▶ 🧠 Vision Mamba ──▶ 💾 Baseline  ║
  ║   (21 classes)    (224×224, Norm)      Training           Model.pth    ║
  ║                                                                        ║
  ╚═══════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
  ╔═══════════════════════════════════════════════════════════════════════╗
  ║  PHASE 2: BASELINE ATTACK EVALUATION                                   ║
  ╠═══════════════════════════════════════════════════════════════════════╣
  ║                                                                        ║
  ║              ┌─────────────────┐    ┌─────────────────┐               ║
  ║              │   AutoAttack    │    │ Adaptive Attack │               ║
  ║              │  • APGD-CE      │    │  • EOT          │               ║
  ║   Baseline ──│  • APGD-DLR     │────│  • BPDA         │──▶ 📉 Report  ║
  ║   Model      │  • FAB-T        │    │  • Multi-PGD    │               ║
  ║              │  • Square       │    │                 │               ║
  ║              └─────────────────┘    └─────────────────┘               ║
  ║                                                                        ║
  ║              ε ∈ {0.5, 1, 2, 3, 4, 5, 6, 7, 8} / 255                   ║
  ╚═══════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
  ╔═══════════════════════════════════════════════════════════════════════╗
  ║  PHASE 3: DEFENSE TRAINING (5 Methods)                                 ║
  ╠═══════════════════════════════════════════════════════════════════════╣
  ║                                                                        ║
  ║   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                  ║
  ║   │ Adversarial  │ │  Randomized  │ │  Certified   │                  ║
  ║   │  Training    │ │  Smoothing   │ │   Robust     │                  ║
  ║   │   (PGD)      │ │  (Gaussian)  │ │   Model      │                  ║
  ║   └──────────────┘ └──────────────┘ └──────────────┘                  ║
  ║                                                                        ║
  ║   ┌──────────────┐ ┌──────────────┐                                   ║
  ║   │  Defensive   │ │   Gradient   │                                   ║
  ║   │ Distillation │ │   Masking    │                                   ║
  ║   │  (Teacher)   │ │              │                                   ║
  ║   └──────────────┘ └──────────────┘                                   ║
  ║                         │                                              ║
  ║                         ▼                                              ║
  ║              💾 5 Robust Model Variants (.pth)                         ║
  ╚═══════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
  ╔═══════════════════════════════════════════════════════════════════════╗
  ║  PHASE 4: ROBUST MODEL EVALUATION & COMPARISON                         ║
  ╠═══════════════════════════════════════════════════════════════════════╣
  ║                                                                        ║
  ║   Each Robust Model ──▶ AutoAttack ──▶ Adaptive Attack ──▶ Results    ║
  ║                                                                        ║
  ║   📊 Compare: Clean Acc | Robust Acc | Attack Success Rate             ║
  ║   🔍 Detect: False Robustness (Gradient Masking, Distillation)         ║
  ║   ✅ Winner: Adversarial Training                                      ║
  ╚═══════════════════════════════════════════════════════════════════════╝
```

---

## 📁 Repository Structure

```
📦 vision-mamba-robustness/
│
├── 📂 src/
│   ├── 📂 evaluation/                    # Attack scripts
│   │   ├── 🔴 AutoAttack Scripts
│   │   │   ├── Autoattack_On_BaseModel.py
│   │   │   ├── Attack_on_AdversarialTrain.py
│   │   │   ├── AutoAttack_On_Randomized_Smoothing.py
│   │   │   ├── Attack_On_CertifiedAccuracy.py
│   │   │   ├── AutoAttack_on_DevensiveDestilation.py
│   │   │   └── Attack_On_Gradient_Masking_AutoAttack.py
│   │   │
│   │   └── 🟠 Adaptive Attack Scripts
│   │       ├── Adapative_attack_on_base_Model.py
│   │       ├── Attack_Model_on_Adversarial_Training.py
│   │       ├── Attack_on_Randomized_smothing.py
│   │       ├── Attack_on_Gradiend_masking.py
│   │       └── Adaptive_attack_on_Devensive_Destilation.py
│   │
│   └── 📂 training/                      # Defense training scripts
│
├── 📂 models/                            # Model checkpoints (.pth)
│   ├── baseline/
│   ├── adversarial_training/
│   ├── randomized_smoothing/
│   ├── certified_robust/
│   ├── defensive_distillation/
│   └── gradient_masking/
│
├── 📂 data/                              # Dataset (see setup)
│
└── 📂 results/                           # CSV reports & plots
```

---

## 📜 Evaluation Scripts

### 🔴 AutoAttack Evaluation

| Script | Target Model | What it does |
|:-------|:-------------|:-------------|
| `Autoattack_On_BaseModel.py` | Baseline | Standard AutoAttack suite (APGD-CE, APGD-DLR, FAB-T, Square) |
| `Attack_on_AdversarialTrain.py` | Adversarial Training | Tests if AT model resists standard attacks |
| `AutoAttack_On_Randomized_Smoothing.py` | Randomized Smoothing | Evaluates smoothed classifier |
| `Attack_On_CertifiedAccuracy.py` | Certified Robust | Tests certified defense |
| `AutoAttack_on_DevensiveDestilation.py` | Defensive Distillation | Tests distilled model |
| `Attack_On_Gradient_Masking_AutoAttack.py` | Gradient Masking | Tests gradient obfuscation |

### 🟠 Adaptive Attack Evaluation

| Script | Target Model | Special Handling |
|:-------|:-------------|:-----------------|
| `Adapative_attack_on_base_Model.py` | Baseline | TRUE adaptive PGD, multi-restart |
| `Attack_Model_on_Adversarial_Training.py` | Adversarial Training | Path resolver + adaptive eval |
| `Attack_on_Randomized_smothing.py` | Randomized Smoothing | **EOT** (Expectation over Transformation) |
| `Attack_on_Gradiend_masking.py` | Gradient Masking | **BPDA** (Backward Pass Differentiable Approximation) |
| `Adaptive_attack_on_Devensive_Destilation.py` | Defensive Distillation | Bypasses soft labels |

---

## ⚙️ Attack Configuration

<table>
<tr>
<td width="50%">

### AutoAttack Settings
```python
NORM = "Linf"
VERSION = "standard"
ATTACKS = [
    "apgd-ce",    # Auto-PGD + Cross Entropy
    "apgd-dlr",   # Auto-PGD + DLR loss
    "fab-t",      # Fast Adaptive Boundary
    "square"      # Score-based black-box
]

# Epsilon values (pixel scale /255)
EPS_LIST = [0.5, 1, 2, 3, 4, 5, 6, 7, 8]
```

</td>
<td width="50%">

### Adaptive Attack Settings
```python
ATTACK = "PGD"
NUM_STEPS = 100
STEP_SIZE = 2/255
NUM_RESTARTS = 10

# Defense-aware components
EOT_SAMPLES = 20      # For stochastic defenses
USE_BPDA = True       # For non-differentiable ops

# Epsilon values (pixel scale /255)
EPS_LIST = [0.5, 1, 2, 3, 4, 5, 6, 7, 8]
```

</td>
</tr>
</table>

---

## 📊 Key Results

### Performance Comparison

| Defense Method | Clean Acc. | Robust Acc.<br/>(ε=8/255) | Verdict |
|:---------------|:----------:|:-------------------------:|:--------|
| Baseline | 95.2% | ~0% | ❌ Highly vulnerable |
| **Adversarial Training** | 88.5% | **45.3%** | ✅ **Most effective** |
| Randomized Smoothing | 90.1% | 15.2% | ⚠️ Partial improvement |
| Certified Robust | 85.3% | 10.8% | ⚠️ Limited but guaranteed |
| Defensive Distillation | 92.4% | 5.1% | ❌ False robustness |
| Gradient Masking | 93.7% | 3.2% | ❌ False robustness |

### 💡 Key Insight

> **Gradient Masking** and **Defensive Distillation** appear robust against standard attacks but **collapse under adaptive attacks**. This demonstrates "false robustness" — these methods only obfuscate gradients without truly improving decision boundaries.

---

## 🚀 Quick Start

### 1️⃣ Clone & Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/vision-mamba-robustness.git
cd vision-mamba-robustness

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Prepare Dataset

```
data/dataset_rambu_lalu_lintas/
├── train/          # 4,414 images
│   ├── batas_kecepatan_30/
│   ├── batas_kecepatan_40/
│   └── ... (21 classes)
├── valid/          # 400 images
└── test/           # 205 images
```

### 3️⃣ Download Models

Place `.pth` checkpoints in `models/` folder:

```
models/
├── baseline/best_vim_rambu_small.pth
├── adversarial_training/best_robust_model.pth
└── ...
```

### 4️⃣ Run Evaluation

```bash
# AutoAttack on baseline
python src/evaluation/Autoattack_On_BaseModel.py

# Adaptive attack on baseline
python src/evaluation/Adapative_attack_on_base_Model.py

# AutoAttack on Adversarial Training model
python src/evaluation/Attack_on_AdversarialTrain.py
```

### 5️⃣ View Results

Check `results/` folder for:
- 📄 `*_report.csv` — Accuracy metrics per epsilon
- 📊 `accuracy_vs_epsilon.png` — Robustness curve
- 📈 `asr_vs_epsilon.png` — Attack success rate

---

## 📋 Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
tqdm>=4.62.0
autoattack          # pip install git+https://github.com/fra31/auto-attack
mamba-ssm           # Vision Mamba dependencies
causal-conv1d
```

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{solang2025visionmamba,
  author  = {Marthen Amelius Solang},
  title   = {Peningkatan Robustness Vision Mamba Terhadap Serangan Adversarial},
  year    = {2025},
}
```

---

## 🔗 References

| Paper | Link |
|:------|:-----|
| Vision Mamba (Zhu et al., 2024) | [arXiv](https://arxiv.org/abs/2401.09417) |
| AutoAttack (Croce & Hein, 2020) | [arXiv](https://arxiv.org/abs/2003.01690) |
| Adversarial Training (Madry et al., 2018) | [arXiv](https://arxiv.org/abs/1706.06083) |
| Randomized Smoothing (Cohen et al., 2019) | [arXiv](https://arxiv.org/abs/1902.02918) |
| BPDA Attack (Athalye et al., 2018) | [arXiv](https://arxiv.org/abs/1802.00420) |

---

## 📧 Contact

<div align="center">

**Marthen Amelius Solang**

--

*Research completed: December 2025*

</div>

---

<div align="center">

### ⭐ Star this repo if you find it useful!

</div>
