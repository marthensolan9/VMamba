# Vision Mamba Robustness Against Adversarial Attacks
## Research Artifact Package — Indonesian Traffic Sign Recognition

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the **research artifact** for a Design Science Research Methodology (DSRM) study on improving and evaluating the robustness of **Vision Mamba** against adversarial attacks in Indonesian Traffic Sign Recognition (TSR).

---

## 📋 Citation & Identity

| Field | Value |
|-------|-------|
| **Thesis Title** | Peningkatan Robustness Vision Mamba Terhadap Serangan Adversarial |
| **English Title** | Enhancing the Robustness of Vision Mamba Against Adversarial Attacks |
| **Author** | Marthen Amelius Solang |
| **NIM** | 23523305 |
| **Program** | Master's Program in Informatics |
| **Institution** | Institut Teknologi Bandung (ITB) |
| **Date** | December 2025 |

---

## 🎯 Research Overview

### Problem Statement
Vision Mamba, while efficient for visual representation learning, is highly vulnerable to adversarial attacks. This research evaluates and improves the robustness of Vision Mamba for safety-critical traffic sign recognition.

### Research Objectives
1. Evaluate baseline robustness of Vision Mamba against AutoAttack and Adaptive Attack
2. Implement and compare five defense methods for robustness enhancement
3. Determine the most effective defense strategy for Indonesian TSR applications

### Key Findings
- **Adversarial Training** is the most effective defense method
- Gradient Masking and Defensive Distillation exhibit "false robustness" that collapses under adaptive attacks
- Randomized Smoothing and Certified Robustness provide additional stability layers

---

## 🔄 Research Pipeline Flowchart

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           RESEARCH PIPELINE OVERVIEW                                 │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: DATA PREPARATION & BASELINE TRAINING                                       │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│   ┌──────────────┐    ┌──────────────────┐    ┌─────────────────────┐               │
│   │   Dataset    │───▶│  Preprocessing   │───▶│   Vision Mamba      │               │
│   │  (21 classes)│    │  • Resize 224×224│    │   Baseline Training │               │
│   │  Train/Val/  │    │  • Normalize     │    │   • dim=192         │               │
│   │    Test      │    │  • Patch (32×32) │    │   • depth=6         │               │
│   └──────────────┘    └──────────────────┘    │   • dropout=0.20    │               │
│                                               └──────────┬──────────┘               │
│                                                          │                          │
│                                                          ▼                          │
│                                               ┌─────────────────────┐               │
│                                               │  Baseline Model     │               │
│                                               │  (.pth checkpoint)  │               │
│                                               │  + config.json      │               │
│                                               └──────────┬──────────┘               │
└──────────────────────────────────────────────────────────┼──────────────────────────┘
                                                           │
                                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: BASELINE ATTACK EVALUATION                                                 │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│   ┌─────────────────────┐         ┌─────────────────────────────────────┐           │
│   │   Baseline Model    │────────▶│         ATTACK EVALUATION           │           │
│   └─────────────────────┘         │                                     │           │
│                                   │  ┌─────────────┐  ┌─────────────┐   │           │
│                                   │  │ AutoAttack  │  │  Adaptive   │   │           │
│                                   │  │ • APGD-CE   │  │   Attack    │   │           │
│                                   │  │ • APGD-DLR  │  │ • EOT       │   │           │
│                                   │  │ • FAB-T     │  │ • BPDA      │   │           │
│                                   │  │ • Square    │  │ • Multi-    │   │           │
│                                   │  │             │  │   restart   │   │           │
│                                   │  └─────────────┘  └─────────────┘   │           │
│                                   │                                     │           │
│                                   │  ε ∈ {0.5, 1, 2, 3, 4, 5, 6, 7, 8} │           │
│                                   │         /255 (L∞ norm)              │           │
│                                   └─────────────────────────────────────┘           │
│                                                    │                                │
│                                                    ▼                                │
│                                   ┌─────────────────────────────────────┐           │
│                                   │   Baseline Vulnerability Report    │           │
│                                   │   • Clean Accuracy                 │           │
│                                   │   • Robust Accuracy per ε          │           │
│                                   │   • Attack Success Rate            │           │
│                                   └─────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                                           │
                                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: ROBUSTNESS ENHANCEMENT (5 DEFENSE METHODS)                                 │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│   ┌─────────────────────────────────────────────────────────────────────────┐       │
│   │                     DEFENSE METHODS TRAINING                             │       │
│   │                                                                          │       │
│   │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐    │       │
│   │  │  Randomized  │ │   Gradient   │ │  Certified   │ │  Adversarial │    │       │
│   │  │  Smoothing   │ │   Masking    │ │   Robust     │ │   Training   │    │       │
│   │  │              │ │              │ │    Model     │ │    (PGD)     │    │       │
│   │  │ σ = 0.25     │ │ Gradient     │ │              │ │              │    │       │
│   │  │ Gaussian     │ │ obfuscation  │ │ RS + formal  │ │ ε-train      │    │       │
│   │  │ noise        │ │              │ │ certification│ │ = 8/255      │    │       │
│   │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘    │       │
│   │                                                                          │       │
│   │  ┌──────────────────────────────────────────────────────────────────┐   │       │
│   │  │                    Defensive Distillation                        │   │       │
│   │  │           Teacher-Student with soft labels (T > 1)               │   │       │
│   │  └──────────────────────────────────────────────────────────────────┘   │       │
│   └─────────────────────────────────────────────────────────────────────────┘       │
│                                           │                                         │
│                                           ▼                                         │
│                            ┌──────────────────────────────┐                         │
│                            │    5 Robust Model Variants   │                         │
│                            │    (.pth checkpoints)        │                         │
│                            └──────────────────────────────┘                         │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                                           │
                                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 4: ROBUST MODEL EVALUATION                                                    │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│   ┌─────────────────────────────────────────────────────────────────────────┐       │
│   │                    FOR EACH DEFENSE METHOD:                              │       │
│   │                                                                          │       │
│   │    ┌────────────┐    ┌────────────────┐    ┌────────────────────┐       │       │
│   │    │   Robust   │───▶│   AutoAttack   │───▶│  Robust Accuracy   │       │       │
│   │    │   Model    │    │   Evaluation   │    │  vs Baseline       │       │       │
│   │    └────────────┘    └────────────────┘    └────────────────────┘       │       │
│   │                              │                                           │       │
│   │                              ▼                                           │       │
│   │                      ┌────────────────┐    ┌────────────────────┐       │       │
│   │                      │   Adaptive     │───▶│  True Robustness   │       │       │
│   │                      │    Attack      │    │  Verification      │       │       │
│   │                      └────────────────┘    └────────────────────┘       │       │
│   └─────────────────────────────────────────────────────────────────────────┘       │
│                                           │                                         │
│                                           ▼                                         │
│                  ┌──────────────────────────────────────────────┐                   │
│                  │            COMPARATIVE ANALYSIS               │                   │
│                  │  • Clean Accuracy comparison                  │                   │
│                  │  • Robust Accuracy per method                 │                   │
│                  │  • Attack Success Rate analysis               │                   │
│                  │  • False robustness detection                 │                   │
│                  └──────────────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                                           │
                                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 5: RESULTS & ARTIFACTS                                                        │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                     │
│   │   CSV Reports   │  │  Visualization  │  │  Model Files    │                     │
│   │   per attack    │  │   (PNG plots)   │  │  (.pth)         │                     │
│   └─────────────────┘  └─────────────────┘  └─────────────────┘                     │
│                                                                                      │
│   Final Recommendation: Adversarial Training as primary defense,                     │
│   Randomized Smoothing + Certified Robust as supplementary layers                    │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
vision-mamba-robustness/
│
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
│
├── src/                                   # Source code
│   ├── training/                          # Training scripts
│   │   ├── baseline_training.py           # Baseline Vision Mamba training
│   │   ├── adversarial_training.py        # Adversarial Training (PGD-based)
│   │   ├── randomized_smoothing.py        # Randomized Smoothing fine-tuning
│   │   ├── certified_robust.py            # Certified Robustness training
│   │   ├── defensive_distillation.py      # Teacher-Student distillation
│   │   └── gradient_masking.py            # Gradient Masking wrapper
│   │
│   └── evaluation/                        # Attack evaluation scripts
│       ├── Autoattack_On_BaseModel.py     # AutoAttack on baseline model
│       ├── Adapative_attack_on_base_Model.py  # Adaptive Attack on baseline
│       └── ... (see Evaluation Scripts section)
│
├── models/                                # Model checkpoints (.pth files)
│   ├── baseline/
│   │   ├── best_vim_rambu_small.pth       # Baseline model checkpoint
│   │   ├── config.json                    # Architecture configuration
│   │   └── class_mapping.json             # Class index mapping
│   │
│   ├── adversarial_training/
│   │   └── best_robust_model.pth
│   │
│   ├── randomized_smoothing/
│   │   └── best_smoothed_model.pth
│   │
│   ├── certified_robust/
│   │   └── best_certified_model.pth
│   │
│   ├── defensive_distillation/
│   │   └── student_model.pth
│   │
│   └── gradient_masking/
│       └── masked_model.pth
│
├── data/                                  # Dataset (not included, see Dataset section)
│   └── dataset_rambu_lalu_lintas/
│       ├── train/
│       ├── valid/
│       └── test/
│
└── results/                               # Experimental outputs
    ├── autoattack/
    │   └── *.csv, *.png
    └── adaptive_attack/
        └── *.csv, *.png
```

---

## 📜 Evaluation Scripts Overview

### AutoAttack Evaluation Scripts

| Script | Target Model | Description |
|--------|--------------|-------------|
| `Autoattack_On_BaseModel.py` | Baseline | Evaluates baseline Vision Mamba against AutoAttack (L∞) |
| `Attack_on_AdversarialTrain.py` | Adversarial Training | AutoAttack on adversarially trained model |
| `AutoAttack_On_Randomized_Smoothing.py` | Randomized Smoothing | AutoAttack on smoothed model |
| `Attack_On_CertifiedAccuracy.py` | Certified Robust | AutoAttack on certified robust model |
| `AutoAttack_on_DevensiveDestilation.py` | Defensive Distillation | AutoAttack on distilled model |
| `Attack_On_Gradient_Masking_AutoAttack.py` | Gradient Masking | AutoAttack on gradient-masked model |

### Adaptive Attack Evaluation Scripts

| Script | Target Model | Description |
|--------|--------------|-------------|
| `Adapative_attack_on_base_Model.py` | Baseline | TRUE Adaptive PGD attack on baseline |
| `Attack_Model_on_Adversarial_Training.py` | Adversarial Training | Adaptive attack path resolver + evaluation |
| `Attack_on_Randomized_smothing.py` | Randomized Smoothing | Adaptive attack with EOT for stochastic defense |
| `Attack_on_Gradiend_masking.py` | Gradient Masking | Adaptive attack with BPDA for gradient obfuscation |
| `Adaptive_attack_on_Devensive_Destilation.py` | Defensive Distillation | Adaptive attack on distilled model |

---

## ⚙️ Attack Configuration

### AutoAttack Settings
```python
ATTACK_TYPE = "standard"  # Standard AutoAttack suite
ATTACKS = ["apgd-ce", "apgd-dlr", "fab-t", "square"]
NORM = "Linf"
EPS_LIST = [0.5, 1, 2, 3, 4, 5, 6, 7, 8]  # /255 scale
```

### Adaptive Attack Settings
```python
ATTACK_TYPE = "adaptive_pgd"
NUM_STEPS = 100           # PGD iterations
STEP_SIZE = 2/255         # Step size per iteration
NUM_RESTARTS = 10         # Random restarts
LOSS = "CE"               # Cross-entropy loss
# Defense-aware components:
EOT_SAMPLES = 20          # For Randomized Smoothing
BPDA = True               # For Gradient Masking
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/<your-username>/vision-mamba-robustness.git
cd vision-mamba-robustness

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

The Indonesian Traffic Sign dataset should be organized as:
```
data/dataset_rambu_lalu_lintas/
├── train/
│   ├── batas_kecepatan_30/
│   ├── batas_kecepatan_40/
│   ├── ... (21 classes)
├── valid/
│   └── ... (same structure)
└── test/
    └── ... (same structure)
```

### 3. Download Model Checkpoints

Place model checkpoints in the `models/` directory. Each defense method has its own subfolder.

### 4. Run Evaluation

```bash
# AutoAttack on baseline model
python src/evaluation/Autoattack_On_BaseModel.py

# Adaptive Attack on baseline model
python src/evaluation/Adapative_attack_on_base_Model.py

# AutoAttack on Adversarial Training model
python src/evaluation/Attack_on_AdversarialTrain.py
```

---

## 📊 Expected Outputs

Each evaluation script generates:

1. **CSV Report**: `{attack_type}_report_{timestamp}.csv`
   - Columns: epsilon, clean_accuracy, robust_accuracy, attack_success_rate

2. **Accuracy Plot**: `accuracy_vs_epsilon_{timestamp}.png`
   - X-axis: Epsilon (perturbation budget)
   - Y-axis: Accuracy (%)

3. **Attack Success Rate Plot**: `asr_vs_epsilon_{timestamp}.png`
   - Shows how attack effectiveness increases with epsilon

4. **Adversarial Examples Grid** (Adaptive Attack only):
   - Visual comparison of clean vs. adversarial images

---

## 📈 Key Results Summary

| Defense Method | Clean Acc. | Robust Acc. (ε=8/255) | Remarks |
|----------------|------------|----------------------|---------|
| Baseline | ~95% | ~0% | Highly vulnerable |
| Adversarial Training | ~88% | ~45% | **Most effective** |
| Randomized Smoothing | ~90% | ~15% | Partial improvement |
| Certified Robust | ~85% | ~10% | Formal guarantees but limited |
| Defensive Distillation | ~92% | ~5% | False robustness detected |
| Gradient Masking | ~93% | ~3% | False robustness detected |

> **Note**: Exact values may vary based on training configuration and random seeds.

---

## 🔬 Methodology (DSRM Mapping)

This repository supports Design Science Research Methodology:

| DSRM Phase | Artifact/Output |
|------------|-----------------|
| **Problem Identification** | Vision Mamba vulnerability analysis |
| **Design & Development** | Defense method implementations (`src/training/`) |
| **Demonstration** | Training scripts with checkpoints (`models/`) |
| **Evaluation** | Attack scripts with CSV/PNG outputs (`results/`) |
| **Communication** | Thesis document + this repository |

---

## 📚 Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
tqdm>=4.62.0
autoattack  # pip install git+https://github.com/fra31/auto-attack
mamba-ssm   # Vision Mamba core
causal-conv1d
```

---

## ⚠️ Important Notes

1. **GPU Memory**: Evaluation requires at least 8GB GPU memory
2. **Attack Duration**: Full AutoAttack evaluation takes ~2-4 hours per model
3. **Reproducibility**: Set `torch.manual_seed(42)` for consistent results
4. **Path Configuration**: Update `BASE_DIR`, `DATA_ROOT` in each script to match your setup

---

## 📖 References

1. Goodfellow, I. J., et al. (2014). "Explaining and Harnessing Adversarial Examples"
2. Madry, A., et al. (2018). "Towards Deep Learning Models Resistant to Adversarial Attacks"
3. Croce, F., & Hein, M. (2020). "Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks"
4. Zhu, L., et al. (2024). "Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model"
5. Cohen, J., et al. (2019). "Certified Adversarial Robustness via Randomized Smoothing"

---

## 📄 License

This research artifact is provided for academic and research purposes. Please cite the thesis if you use this code.

---

## 📧 Contact

For questions or collaboration inquiries:
- **Author**: Marthen Amelius Solang
- **Institution**: Institut Teknologi Bandung (ITB)
- **Program**: Master's in Informatics

---

*Last updated: December 2025*
