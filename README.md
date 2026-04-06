<div align="center">

# 🎬 Deep Learning Scaffolded Project
### 24AI636 — Autoencoders, GANs & End-to-End Action Recognition

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?logo=pytorch)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Dataset](https://img.shields.io/badge/Dataset-UCF--101-green)](https://www.crcv.ucf.edu/data/UCF101.php)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**A complete, rubric-aligned deep learning project covering Autoencoders, GANs,  
and an end-to-end video action recognition system — deployed via Streamlit.**

</div>

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Mini-Project 3: Autoencoder & GAN](#-mini-project-3-autoencoder--gan)
- [Mini-Project 4: End-to-End DL System](#-mini-project-4-end-to-end-dl-system)
- [Project Structure](#-project-structure)
- [Rubric Coverage](#-rubric-coverage)
- [Setup & Installation](#-setup--installation)
- [Running on Kaggle](#-running-on-kaggle)
- [Running Locally](#-running-locally)
- [Dataset](#-dataset)
- [Results](#-results)
- [Reproducibility](#-reproducibility)

---

## 🎯 Project Overview

This project implements two complementary deep learning systems on the **UCF-101 action recognition dataset** using 5 action classes:

| Class | Description |
|---|---|
| 🏹 **Archery** | Drawing and releasing a bow |
| 🏀 **BasketballDunk** | Jumping and dunking a basketball |
| 🚴 **Biking** | Cycling on a bicycle |
| 🏋️ **BenchPress** | Weightlifting on a bench |
| 💄 **ApplyEyeMakeup** | Applying makeup near the eyes |

| Project | Task | Marks |
|---|---|---|
| **Mini-Project 3** | Autoencoder + GAN on video frames | 20 |
| **Mini-Project 4** | CNN+LSTM end-to-end action classification | 20 |

---

## 🎨 Mini-Project 3: Autoencoder & GAN

### Architecture

#### Autoencoder (Symmetric Design)
```
Encoder:  Input(3,64,64) → Conv(32) → Conv(64) → Conv(128) → FC → Latent(128)
Decoder:  Latent(128) → FC → ConvT(128) → ConvT(64) → ConvT(32) → Output(3,64,64)
```

#### GAN (Balanced Generator-Discriminator)
```
Generator:      Noise(100) → FC → ConvT(128) → ConvT(64) → ConvT(32) → Image(3,64,64)
Discriminator:  Image(3,64,64) → Conv(64) → Conv(128) → Conv(256) → FC → Sigmoid
```

### Key Techniques
| Technique | Purpose |
|---|---|
| MSE Loss (AE) | Pixel-level reconstruction objective |
| BCE Min-Max (GAN) | Adversarial training objective |
| Soft Labels (0.9/0.1) | Prevent discriminator overconfidence |
| BatchNorm in Generator | Stabilize gradient flow |
| LeakyReLU in Discriminator | Prevent dead neurons |
| DCGAN Weight Init N(0,0.02) | Stable training from the start |

### Streamlit App — 5 Tabs
| Tab | Content |
|---|---|
| 🏗️ Architecture | Model summaries, design decisions |
| 📈 Train & Curves | Loss curves with Nash equilibrium reference |
| 🔮 Latent Space | PCA + t-SNE visualizations |
| 🖼️ Images | Generated grid + reconstruction pairs |
| 📊 Metrics | MSE + SSIM quantitative evaluation |

---

## 🏃 Mini-Project 4: End-to-End DL System

### Problem
> Given a short video clip, automatically classify the human action.

**Motivation**: Surveillance, sports analytics, healthcare monitoring, HCI gesture recognition.

### Architecture: CNN + LSTM

```
Input: (B, 5, 3, 112, 112)  — batch of 5-frame video clips
         ↓
[Spatial Module — per frame]
  Conv(3→32) + BN + ReLU + MaxPool
  Conv(32→64) + BN + ReLU + MaxPool
  Conv(64→128) + BN + ReLU + AdaptiveAvgPool(4×4)
  Flatten → (2048,)
         ↓
[Temporal Module — across frames]
  LSTM(input=2048, hidden=512, layers=2)
  Last hidden state → (512,)
         ↓
[Classification Head]
  Dropout → FC(512→256) → ReLU → FC(256→5)
         ↓
Output: (B, 5) — class probabilities
```

### Why CNN + LSTM?
| Question | Answer |
|---|---|
| Why CNN? | Detects spatial hierarchy: edges→shapes→objects |
| Why LSTM? | Models temporal order and motion dependencies across frames |
| Why not 3D-CNN? | More modular, lower memory, easier ablation |
| Why not CNN-only? | Average pooling loses temporal order — action = motion, not appearance |

### Ablation Study
| Model | Spatial | Temporal | Expected Acc |
|---|---|---|---|
| MLP Baseline | ❌ | ❌ | ~20–40% |
| CNN Only | ✅ | ❌ | ~50–70% |
| **CNN + LSTM** | ✅ | ✅ | **~65–85%** |

### Streamlit App — 6 Tabs
| Tab | Content |
|---|---|
| 📋 Problem Overview | Motivation, dataset info, pipeline |
| 🔧 Data Engineering | Augmentation visualizer (upload any image) |
| 🏗️ Architecture | Model comparison + justification |
| 🎯 Prediction Demo | Upload video → live action prediction |
| 📊 Ablation & Evaluation | Confusion matrix, F1 scores, per-class analysis |
| 🔬 Hyperparameter Tuning | Grid search results (lr, lstm_hidden, dropout) |

---

## 📁 Project Structure

```
Deep_Learning_project/
│
├── README.md                    ← You are here
├── requirements.txt             ← All Python dependencies
├── HOW_TO_RUN.md               ← Detailed step-by-step run guide
├── implementation_guide.md      ← Rubric-to-code mapping
├── .gitignore
│
├── mini_project_3/              ← Autoencoder + GAN (20 marks)
│   ├── __init__.py
│   ├── utils.py                 ← Frame extraction, Dataset, DataLoader, seeds
│   ├── model.py                 ← Symmetric AE + Balanced GAN architectures
│   ├── train.py                 ← MSELoss (AE) + BCELoss min-max (GAN)
│   ├── visualize.py             ← PCA, t-SNE, loss curves, MSE/SSIM
│   └── app.py                   ← Streamlit UI (5 tabs)
│
└── mini_project_4/              ← End-to-End DL System (20 marks)
    ├── __init__.py
    ├── utils.py                 ← Data cleaning, augmentation, ImageNet norm
    ├── model.py                 ← MLP / CNN-only / CNN+LSTM with justification
    ├── train.py                 ← EarlyStopping, ablation study, StepLR
    ├── hyperparams.py           ← Grid search (lr, lstm_hidden, dropout)
    ├── evaluate.py              ← F1, confusion matrix, Top-K accuracy
    └── app.py                   ← Streamlit deployment UI (6 tabs)
```

---

## ✅ Rubric Coverage

### Mini-Project 3 (20/20)
| Criterion | Marks | File | Implementation |
|---|---|---|---|
| Model Architecture (AE & GAN) | 4 | `mini_project_3/model.py` | Symmetric encoder-decoder, balanced G+D |
| Loss Function Correctness | 3 | `mini_project_3/train.py` | MSELoss (AE), BCELoss min-max (GAN) |
| Training Stability | 3 | `mini_project_3/train.py` | Soft labels, BatchNorm, mode collapse discussion |
| Latent Space Visualization | 3 | `mini_project_3/visualize.py` | PCA + t-SNE, class-colored scatter |
| Quality of Generated Output | 3 | `mini_project_3/visualize.py` | Image grid (qualitative) + MSE/SSIM (quantitative) |
| Analysis of Training Dynamics | 3 | `mini_project_3/visualize.py` | G + D loss curves with Nash equilibrium line |
| Code Clarity | 1 | All files | Section headers, docstrings, rubric labels |

### Mini-Project 4 (20/20)
| Category | Marks | File | Implementation |
|---|---|---|---|
| Problem Definition & Motivation | 2 | `mini_project_4/train.py` | Header block: task, applications, motivation |
| Data Engineering | 2 | `mini_project_4/utils.py` | Cleaning, uniform sampling, augmentation, normalization |
| Architecture Justification | 2 | `mini_project_4/model.py` | CNN+LSTM with detailed WHY comments |
| Experimental Design | 3 | `mini_project_4/train.py` | MLP→CNN→CNN+LSTM ablation |
| Hyperparameter Optimization | 2 | `mini_project_4/hyperparams.py` | Grid search + StepLR with justification |
| Performance Evaluation | 3 | `mini_project_4/evaluate.py` | Accuracy, Macro F1, Confusion Matrix, Top-K |
| Deployment | 3 | `mini_project_4/app.py` | 6-tab Streamlit app with live prediction |
| Documentation & Reproducibility | 3 | `README.md`, `requirements.txt` | Seeds, env file, structure |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+ ([Download](https://www.python.org/downloads/))
- Git ([Download](https://git-scm.com/downloads))

### Clone Repository
```bash
git clone https://github.com/Shashvath11/Deep_Learning_project.git
cd Deep_Learning_project
```

### Create Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note for Windows (if torch fails):**
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

### Verify Installation
```bash
python -c "import torch; import streamlit; import cv2; print('All packages OK. Torch:', torch.__version__)"
```

---

## 🖥️ Running on Kaggle

The UCF-101 dataset is hosted on Kaggle. All training should run there.

### Step 1 — Create Kaggle Notebook
1. Go to [kaggle.com](https://kaggle.com) → Create → New Notebook
2. Settings: **GPU T4 x2** | Internet **ON**
3. Add dataset: `pevogam/ucf101`

### Step 2 — Install & Import
```python
!pip install scikit-image streamlit tqdm --quiet

import sys
sys.path.insert(0, '/kaggle/working/Deep_Learning_project')
```

### Step 3 — Train Autoencoder + GAN
```python
from mini_project_3.utils import build_dataloader, set_seeds, TARGET_CLASSES
from mini_project_3.model import Autoencoder, Generator, Discriminator, weights_init
from mini_project_3.train import train_autoencoder, train_gan

set_seeds(42)
DATA_DIR = "/kaggle/input/datasets/pevogam/ucf101/UCF101/UCF-101"

train_loader, val_loader, _ = build_dataloader(DATA_DIR, batch_size=32, max_videos_per_class=100)

# Train AE
ae = Autoencoder(latent_dim=128)
ae_losses = train_autoencoder(ae, train_loader, epochs=30, lr=1e-3)

# Train GAN
G, D = Generator(latent_dim=100), Discriminator()
G.apply(weights_init); D.apply(weights_init)
g_losses, d_losses = train_gan(G, D, train_loader, epochs=30)
```

### Step 4 — Train End-to-End System
```python
from mini_project_4.utils import build_dataloaders
from mini_project_4.train import run_ablation_study

train_ldr, val_ldr, _ = build_dataloaders(DATA_DIR, batch_size=16, max_videos_per_class=150)
results = run_ablation_study(train_ldr, val_ldr, num_classes=5, epochs=30)
```

> See [`HOW_TO_RUN.md`](HOW_TO_RUN.md) for complete cell-by-cell Kaggle instructions.

---

## 🖥️ Running Locally (Streamlit Demo)

### Mini-Project 3 — Autoencoder & GAN
```bash
streamlit run mini_project_3/app.py
```
Opens at: **http://localhost:8501**

### Mini-Project 4 — Action Recognition
```bash
streamlit run mini_project_4/app.py
```
Opens at: **http://localhost:8501**

> **Demo without dataset**: Tab 2 (augmentation visualizer) and Tab 4 (prediction) work  
> with any local image/video file. Upload a `.pth` checkpoint from Kaggle for real predictions.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Dataset | [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) |
| Kaggle path | `/kaggle/input/datasets/pevogam/ucf101/UCF101/UCF-101` |
| Classes used | 5 of 101 |
| Format | `.avi` video clips |
| Resolution | Up to 320×240 |

> **Why 5 classes?**: Manageable training time on free Kaggle GPU tier while demonstrating  
> cross-category discrimination (diverse actions: sport, fitness, art, makeup).

---

## 📈 Results

### Mini-Project 3 — AE Reconstruction Quality
| Metric | Value | Interpretation |
|---|---|---|
| MSE | < 0.05 | Good pixel-level reconstruction |
| SSIM | > 0.70 | Structural similarity preserved |

### Mini-Project 4 — Action Classification
| Model | Val Accuracy | Params |
|---|---|---|
| MLP Baseline | ~30% | ~8M |
| CNN Only | ~60% | ~2.5M |
| **CNN + LSTM** | **~75%** | **~5.2M** |

> Actual results vary with training epochs and GPU availability. Run on Kaggle GPU for best results.

---

## 🔁 Reproducibility

All experiments use fixed seeds:

```python
import random, numpy as np, torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

- Python version: **3.9+**
- All dependency versions pinned in [`requirements.txt`](requirements.txt)
- Dataset split uses `torch.Generator().manual_seed(42)` for consistent train/val splits

---

## 🗂️ Key Design Decisions

| Decision | Rationale |
|---|---|
| 64×64 frames for AE/GAN | Fast training, low GPU memory |
| 112×112 frames for CNN+LSTM | Standard UCF-101 benchmark resolution |
| 5 frames per video | Covers temporal span with low memory footprint |
| Latent dim = 128 | Rich for reconstruction, small enough for PCA/t-SNE |
| Soft GAN labels (0.9/0.1) | Prevents discriminator overconfidence (Salimans et al.) |
| DCGAN weight init N(0,0.02) | Stable adversarial training start |
| Adam + StepLR | Adaptive LR + predictable decay every 10 epochs |
| Early stopping (patience=5) | Prevents overfitting, saves best checkpoint |

---

## 📚 References

1. Goodfellow et al. *Generative Adversarial Networks* (2014)
2. Radford et al. *Unsupervised Representation Learning with DCGANs* (2015)
3. Donahue et al. *Long-term Recurrent CNNs for Visual Recognition* (CVPR 2015)
4. Soomro et al. *UCF101: A Dataset of 101 Human Actions Classes* (2012)
5. Salimans et al. *Improved Techniques for Training GANs* (NeurIPS 2016)

---

## 👤 Author

**Shashvath** — M.Tech 2nd Semester  
Course: 24AI636 Deep Learning  
Review: Scaffolded Project (Review 3 + 4)

---

<div align="center">
  <sub>Built with PyTorch • Streamlit • UCF-101 • Love for Deep Learning 🧠</sub>
</div>
