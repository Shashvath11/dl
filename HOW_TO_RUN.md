# 🚀 HOW TO RUN — Step-by-Step Guide
## 24AI636 DL Scaffolded Project (Mini-Project 3 + 4)

---

## QUICK OVERVIEW

This project runs in TWO places:
| Where | What |
|---|---|
| **Kaggle Notebook** | For actual training (GPU + dataset access) |
| **Local Machine** | For Streamlit UI (after downloading trained weights) |

> The dataset path `/kaggle/input/datasets/pevogam/ucf101/UCF101/UCF-101`
> only exists on Kaggle. All training must happen there.

---

## ─────────────────────────────────────────────────
## PART A: RUNNING ON KAGGLE (Recommended — GPU + Data)
## ─────────────────────────────────────────────────

### STEP 1: Upload Project to Kaggle

**Option A — Upload as Dataset:**
1. Go to https://www.kaggle.com/datasets → "New Dataset"
2. Upload the entire `Final_review/` folder as a zip
3. Name it: `dl-final-review`

**Option B — Upload files directly to Notebook:**
1. Create a new Kaggle Notebook
2. Click "+ Add data" → search for `pevogam/ucf101` → Add it
3. In the notebook, use the file upload button to upload your `.py` files

---

### STEP 2: Create a Kaggle Notebook

1. Go to https://www.kaggle.com → "Create" → "New Notebook"
2. Settings (right panel):
   - **Accelerator**: GPU T4 x2 (free tier)
   - **Language**: Python
   - **Internet**: ON (for pip installs)
3. Add UCF-101 dataset:
   - Click "+ Add Input" → search "ucf101 pevogam" → Add

---

### STEP 3: Install Dependencies in Kaggle

Paste this in the first cell:

```python
# Cell 1: Install dependencies
!pip install scikit-image==0.22.0 streamlit==1.29.0 tqdm==4.66.1 --quiet
# torch, torchvision, numpy, pandas, sklearn, opencv, matplotlib, seaborn
# are all pre-installed in Kaggle kernels
```

---

### STEP 4: Set Up Project Files in Kaggle

**Option A (if uploaded as dataset):**
```python
# Cell 2: Copy project files
import shutil, os
shutil.copytree('/kaggle/input/dl-final-review/Final_review', '/kaggle/working/Final_review')
os.chdir('/kaggle/working/Final_review')
```

**Option B (manual upload):**
```python
# Cell 2: Manually set working directory after uploading files
import os
os.chdir('/kaggle/working')
# Upload files via Kaggle sidebar → your files appear in /kaggle/working/
```

---

### STEP 5: Run Mini-Project 3 (Autoencoder + GAN)

```python
# Cell 3: Test data loading first
import sys
sys.path.insert(0, '/kaggle/working/Final_review')

from mini_project_3.utils import build_dataloader, set_seeds, TARGET_CLASSES

set_seeds(42)
DATA_DIR = "/kaggle/input/datasets/pevogam/ucf101/UCF101/UCF-101"

train_loader, val_loader, class_to_idx = build_dataloader(
    DATA_DIR,
    batch_size=32,
    max_videos_per_class=100,  # increase to 200+ for better results
    num_workers=2,
)
print("Classes:", class_to_idx)
batch, labels = next(iter(train_loader))
print("Batch shape:", batch.shape)   # Expected: (32, 3, 64, 64)
```

```python
# Cell 4: Train Autoencoder
from mini_project_3.model import Autoencoder, Generator, Discriminator, weights_init
from mini_project_3.train import train_autoencoder, train_gan

ae = Autoencoder(latent_dim=128)
ae_losses = train_autoencoder(ae, train_loader, epochs=30, lr=1e-3)
```

```python
# Cell 5: Train GAN
G = Generator(latent_dim=100)
D = Discriminator()
G.apply(weights_init)
D.apply(weights_init)

g_losses, d_losses = train_gan(
    G, D, train_loader,
    epochs=30, lr_g=2e-4, lr_d=2e-4,
    latent_dim=100
)
```

```python
# Cell 6: Visualize results
from mini_project_3.visualize import (
    plot_loss_curves, plot_latent_space_pca, plot_latent_space_tsne,
    plot_generated_images, plot_reconstructed_images, compute_reconstruction_metrics
)

# Loss curves
plot_loss_curves(ae_losses, g_losses, d_losses)

# Latent space
plot_latent_space_pca(ae.encoder, val_loader, TARGET_CLASSES)
plot_latent_space_tsne(ae.encoder, val_loader, TARGET_CLASSES)

# Generated images
plot_generated_images(G, latent_dim=100, n=16)
plot_reconstructed_images(ae, val_loader, n=8)

# Quantitative metrics
metrics = compute_reconstruction_metrics(ae, val_loader)
print(metrics)
```

```python
# Cell 7: Save models for download
import torch, os
os.makedirs('/kaggle/working/checkpoints', exist_ok=True)
torch.save(ae.state_dict(), '/kaggle/working/checkpoints/autoencoder_best.pth')
torch.save(G.state_dict(), '/kaggle/working/checkpoints/generator_best.pth')
torch.save(D.state_dict(), '/kaggle/working/checkpoints/discriminator_best.pth')
print("Models saved!")
```

---

### STEP 6: Run Mini-Project 4 (End-to-End DL System)

```python
# Cell 8: Load data for MP4
from mini_project_4.utils import build_dataloaders, TARGET_CLASSES, set_seeds

set_seeds(42)
train_ldr, val_ldr, c2i = build_dataloaders(
    DATA_DIR,
    batch_size=16,
    max_videos_per_class=150,
    n_frames=5,
    num_workers=2,
)
print("Loaded:", len(train_ldr.dataset), "train samples")
```

```python
# Cell 9: Run full ablation study (MLP vs CNN vs CNN+LSTM)
from mini_project_4.train import run_ablation_study, plot_ablation_results

results = run_ablation_study(
    train_ldr, val_ldr,
    num_classes=5,
    epochs=30,
    save_dir='/kaggle/working/checkpoints',
)
plot_ablation_results(results)
```

```python
# Cell 10: Run hyperparameter grid search
from mini_project_4.hyperparams import get_hyperparameter_grid, grid_search, plot_hyperparam_results

grid = get_hyperparameter_grid()
results_df = grid_search(train_ldr, val_ldr, param_grid=grid, epochs_per_config=5)
plot_hyperparam_results(results_df)
print(results_df)
```

```python
# Cell 11: Evaluate best model
from mini_project_4.evaluate import compute_metrics, plot_confusion_matrix, top_k_accuracy
from mini_project_4.model import CNNLSTM
import torch

# Load best checkpoint
best_model = CNNLSTM(num_classes=5, lstm_hidden=512, lstm_layers=2)
best_model.load_state_dict(torch.load('/kaggle/working/checkpoints/cnn_lstm_best.pth'))

# Compute all metrics
metrics = compute_metrics(best_model, val_ldr, TARGET_CLASSES)
top3 = top_k_accuracy(best_model, val_ldr, k=3)

plot_confusion_matrix(metrics['confusion_matrix'], TARGET_CLASSES)
```

```python
# Cell 12: Save CNN+LSTM for download
torch.save(best_model.state_dict(), '/kaggle/working/checkpoints/cnn_lstm_best.pth')
print("Saved! Download from Kaggle Output panel.")
```

---

### STEP 7: Download Checkpoints from Kaggle

1. In the Kaggle notebook, go to **Output** tab (right panel)
2. Find `/kaggle/working/checkpoints/` folder
3. Download these files to your local machine:
   - `autoencoder_best.pth`
   - `generator_best.pth`
   - `cnn_lstm_best.pth`
4. Place them in: `Final_review/checkpoints/`

---

## ─────────────────────────────────────────────────
## PART B: RUNNING STREAMLIT LOCALLY (After Training)
## ─────────────────────────────────────────────────

### STEP 1: Check Python Version

```powershell
python --version
# Must be Python 3.9, 3.10, or 3.11
# If not installed: https://www.python.org/downloads/
```

---

### STEP 2: Open Terminal in Project Folder

```powershell
# In PowerShell or Command Prompt:
cd "c:\Users\shash\Desktop\Mtech - 2nd sem\Deep Learning\Final_review"
```

---

### STEP 3: Create Virtual Environment

```powershell
# Create virtual environment (do this ONCE)
python -m venv venv

# Activate it (do this EVERY time you open a new terminal)
.\venv\Scripts\activate

# You should see (venv) in your terminal prompt
```

---

### STEP 4: Install Dependencies

```powershell
# With (venv) active:
pip install --upgrade pip

# Install all packages (this takes 5-10 minutes first time)
pip install -r requirements.txt
```

> ⚠️ **If torch install fails** (Windows sometimes has issues):
> ```powershell
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

---

### STEP 5: Verify Installation

```powershell
python -c "import torch; import streamlit; import cv2; print('All good! Torch:', torch.__version__)"
```

Expected output: `All good! Torch: 2.x.x`

---

### STEP 6: Test Module Imports (No Training)

```powershell
# From Final_review/ folder with (venv) active:
python mini_project_3/model.py
python mini_project_4/model.py
```

You should see architecture summaries printed.

---

### STEP 7: Run Streamlit — Mini-Project 3

```powershell
# Make sure you're in the Final_review/ folder
# Make sure (venv) is active

streamlit run mini_project_3/app.py
```

Your browser will automatically open at **http://localhost:8501**

**In the Streamlit app:**
1. Set dataset path in sidebar (point to your local UCF-101 OR leave as-is for Kaggle)
2. Click **"Start Training (AE + GAN)"** in Tab 2
3. View latent space, images, and metrics in other tabs

---

### STEP 8: Run Streamlit — Mini-Project 4

```powershell
# Stop the previous app first (Ctrl+C), then:
streamlit run mini_project_4/app.py
```

Your browser opens at **http://localhost:8501**

**In the Streamlit app:**
- **Tab 1** — Problem Overview (no action needed)
- **Tab 2** — Upload a photo to see augmentation
- **Tab 3** — View architecture comparison
- **Tab 4** — Upload any `.avi/.mp4` video → predict action class
  - You can upload a `.pth` checkpoint too (downloaded from Kaggle)
- **Tab 5** — Run ablation study (needs dataset path)
- **Tab 6** — Run hyperparameter grid search (needs dataset path)

---

## ─────────────────────────────────────────────────
## PART C: DEMO WITHOUT DATASET (Presentation Mode)
## ─────────────────────────────────────────────────

If you don't have UCI-101 data locally, you can still demo:

### For Mini-Project 3:
1. Open `mini_project_3/app.py` in Streamlit
2. **Tab 1** — Shows architecture (no data needed)
3. **Tab 4** — Click "Generate Images" → works with random weights
4. **Tab 5** — Metrics will use preloaded model state

### For Mini-Project 4:
1. Open `mini_project_4/app.py` in Streamlit
2. **Tab 1** — Problem overview (no data needed)
3. **Tab 2** — Upload any JPEG photo → shows augmentation live
4. **Tab 3** — Architecture comparison (no data needed)
5. **Tab 4** — Upload any short video → prediction with random weights
   - Upload the `.pth` checkpoint from Kaggle for real predictions

---

## ─────────────────────────────────────────────────
## COMMON ERRORS & FIXES
## ─────────────────────────────────────────────────

| Error | Fix |
|---|---|
| `ModuleNotFoundError: mini_project_3` | Make sure you're running from the `Final_review/` folder |
| `cv2.error: (-215)` | Video file is corrupt or wrong format |
| `CUDA out of memory` | Reduce `batch_size` in sidebar |
| `No such file: UCF-101` | Update dataset path in sidebar |
| `streamlit: command not found` | Run `pip install streamlit` or activate venv |
| `torch not found` | Run `pip install torch torchvision` |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` |

---

## ─────────────────────────────────────────────────
## FILE REFERENCE
## ─────────────────────────────────────────────────

```
Final_review/
├── HOW_TO_RUN.md          ← This file
├── implementation_guide.md ← Rubric-to-code mapping
├── README.md              ← Project overview
├── requirements.txt       ← All Python dependencies
│
├── mini_project_3/        ← Run: streamlit run mini_project_3/app.py
│   ├── utils.py           → Data loading, frame extraction
│   ├── model.py           → Autoencoder + GAN architectures
│   ├── train.py           → Training loops (AE + GAN)
│   ├── visualize.py       → PCA, t-SNE, loss curves, metrics
│   └── app.py             → Streamlit UI (5 tabs)
│
├── mini_project_4/        ← Run: streamlit run mini_project_4/app.py
│   ├── utils.py           → Video dataset, augmentation
│   ├── model.py           → MLP / CNN / CNN+LSTM models
│   ├── train.py           → Training + ablation study
│   ├── hyperparams.py     → Grid search + LR scheduler
│   ├── evaluate.py        → Metrics + confusion matrix
│   └── app.py             → Streamlit UI (6 tabs)
│
└── checkpoints/           ← Create this; put .pth files here after Kaggle training
    ├── autoencoder_best.pth
    ├── generator_best.pth
    └── cnn_lstm_best.pth
```

---

## ─────────────────────────────────────────────────
## RECOMMENDED WORKFLOW FOR PRESENTATION
## ─────────────────────────────────────────────────

```
Day 1 (Kaggle):
  1. Upload project files to Kaggle
  2. Run Cells 3-7 (MP3 training ~30-60 min on GPU)
  3. Run Cells 8-12 (MP4 training ~60-120 min on GPU)
  4. Download .pth checkpoints

Day 2 (Local Demo):
  1. Install requirements locally (STEP 4 above)
  2. Run MP3 Streamlit app
  3. Load checkpoint in Tab 4 for prediction
  4. Screenshot all plots for slides

Presentation:
  1. Show Streamlit app live
  2. Use Tab 1 (Architecture) to explain model
  3. Tab 3 (Latent Space) to show PCA/t-SNE
  4. Tab 4 (Prediction) for live demo
  5. Tab 5 (Ablation/Metrics) to show performance
```
