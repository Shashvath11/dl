# 24AI636 DL Scaffolded Project — Implementation Guide
# Dataset: UCF-101 (5 classes: Archery, BasketballDunk, Biking, BenchPress, ApplyEyeMakeup)
# Dataset Path: /kaggle/input/datasets/pevogam/ucf101/UCF101/UCF-101

## Project File Structure

```
Final_review/
│
├── implementation_guide.md          ← This file
│
├── mini_project_3/                  ← Autoencoder + GAN (20 marks)
│   ├── model.py                     ← AE + GAN architectures
│   ├── train.py                     ← Training loops for AE + GAN
│   ├── utils.py                     ← Data loading, preprocessing, metrics
│   ├── visualize.py                 ← Latent space (PCA/t-SNE), loss curves
│   └── app.py                       ← Streamlit UI for Mini-Project 3
│
├── mini_project_4/                  ← End-to-End DL System (20 marks)
│   ├── model.py                     ← CNN + LSTM / 3D-CNN architecture
│   ├── train.py                     ← Full training pipeline with ablation
│   ├── utils.py                     ← Data engineering + augmentation
│   ├── evaluate.py                  ← Metrics, statistical reasoning
│   ├── hyperparams.py               ← Hyperparameter tuning strategy
│   └── app.py                       ← Streamlit deployment UI
│
├── requirements.txt                 ← Reproducibility
└── README.md                        ← Documentation
```

---

## Rubric Mapping — Mini-Project 3 (20 Marks)

| Criterion | Marks | File(s) | What is Implemented |
|---|---|---|---|
| Model architecture correctness (AE & GAN) | 4 | model.py | Symmetric encoder-decoder (Conv/Deconv), Generator + Discriminator with balanced depth |
| Loss function correctness | 3 | train.py | MSE reconstruction loss (AE), BCELoss min-max objective (GAN) |
| Training stability handling | 3 | train.py, model.py | BatchNorm, LeakyReLU, soft labels, mode collapse discussion in comments |
| Latent space visualization | 3 | visualize.py | PCA + t-SNE of encoder latent codes, color-coded by class |
| Quality of generated output | 3 | visualize.py, train.py | Qualitative: image grid; Quantitative: MSE/SSIM/FID proxy |
| Analysis of training dynamics | 3 | visualize.py | Generator + Discriminator loss curves plotted per epoch |
| Code clarity | 1 | All files | Section headers, docstrings, rubric labels in comments |

---

## Rubric Mapping — Mini-Project 4 (20 Marks)

| Category | Marks | File(s) | What is Implemented |
|---|---|---|---|
| Problem Definition & Motivation | 2 | train.py, app.py | Clear comments on research relevance (video action recognition) |
| Data Engineering | 2 | utils.py | Frame extraction, normalization, augmentation (flip, crop, color jitter) |
| Model Architecture Justification | 2 | model.py | Commented reasoning: why CNN+LSTM over plain CNN |
| Experimental Design | 3 | train.py | Baseline (MLP) + ablation (CNN-only vs CNN+LSTM vs full model) |
| Hyperparameter Optimization | 2 | hyperparams.py | Grid search strategy, LR scheduler, early stopping |
| Performance Evaluation | 3 | evaluate.py | Accuracy, F1, Confusion Matrix, per-class analysis + reasoning |
| Deployment (API/UI/Cloud) | 3 | app.py | Streamlit app: upload video → predict + visualize |
| Documentation & Reproducibility | 3 | requirements.txt, README.md | Environment file, seed setting, folder structure |

---

## Module-by-Module Task Checklist

### PHASE 1 — Shared Utilities & Data Pipeline

#### mini_project_3/utils.py and mini_project_4/utils.py
- [ ] load_ucf101_frames(data_dir, classes, n_frames) — reads video, samples N frames uniformly
- [ ] preprocess_frame(frame) — resize to 64x64 (MP3) / 112x112 (MP4), normalize [-1,1] or [0,1]
- [ ] augment_frame(frame) — random horizontal flip, random crop, color jitter (MP4 only)
- [ ] build_dataloader(data_dir, classes, batch_size, augment) — torch Dataset + DataLoader
- [ ] Set random.seed(42), torch.manual_seed(42), numpy.random.seed(42) for reproducibility

---

### PHASE 2 — Mini-Project 3 Models

#### mini_project_3/model.py
**[RUBRIC: Architecture Correctness — 4 marks]**

##### Autoencoder
- [ ] Encoder: Conv2d(3→32→64→128) + ReLU + BatchNorm → Flatten → Linear → latent_dim (128)
- [ ] Decoder: Linear → Reshape → ConvTranspose2d(128→64→32→3) + ReLU/Tanh → 64x64 output
- [ ] Autoencoder: nn.Module wrapping Encoder + Decoder with encode(), decode(), forward()
- [ ] Comment: encoder-decoder symmetry (same depth, mirrored channel sizes)

##### GAN
- [ ] Generator: Linear(latent→256) → Reshape(4x4) → ConvTranspose2d x3 → Tanh → 64x64x3
- [ ] Discriminator: Conv2d(3→64→128→256) → Flatten → Linear → Sigmoid
- [ ] Comment: generator-discriminator balance (matched capacity), LeakyReLU in Discriminator

---

### PHASE 3 — Mini-Project 3 Training

#### mini_project_3/train.py
**[RUBRIC: Loss Functions — 3 marks | Training Stability — 3 marks]**

- [ ] train_autoencoder(model, dataloader, epochs, lr):
  - Loss: nn.MSELoss() (reconstruction loss) — labeled in comment
  - Adam optimizer, log loss per epoch to list
- [ ] train_gan(generator, discriminator, dataloader, epochs, lr):
  - D loss: BCE(D(real), 1) + BCE(D(fake), 0) — min-max labeled in comment
  - G loss: BCE(D(G(z)), 1) — generator tries to fool D
  - Stability tricks: soft labels (0.9 / 0.1), BatchNorm in G, LeakyReLU in D
  - Comment block: mode collapse discussion, what symptoms look like, how BatchNorm/soft labels help
- [ ] Return ae_losses, g_losses, d_losses lists for plotting

---

### PHASE 4 — Mini-Project 3 Visualization

#### mini_project_3/visualize.py
**[RUBRIC: Latent Space — 3 marks | Output Quality — 3 marks | Training Dynamics — 3 marks]**

- [ ] plot_loss_curves(ae_losses, g_losses, d_losses) — side-by-side matplotlib plots
- [ ] plot_latent_space_pca(encoder, dataloader, classes) — PCA of 128-d latent vectors, 2D scatter, color by class
- [ ] plot_latent_space_tsne(encoder, dataloader, classes) — t-SNE same data
- [ ] plot_generated_images(generator, n=16) — grid of GAN-generated images
- [ ] plot_reconstructed_images(autoencoder, dataloader, n=8) — original vs reconstructed side by side
- [ ] compute_reconstruction_metrics(autoencoder, dataloader) — MSE + SSIM scores (quantitative)

---

### PHASE 5 — Mini-Project 4 Models

#### mini_project_4/model.py
**[RUBRIC: Architecture Justification — 2 marks]**

- [ ] BaselineMLP(input_dim, num_classes) — flat frame → MLP (ablation baseline)
- [ ] CNNFeatureExtractor — lightweight CNN (Conv2d x3 + Pool) for spatial features per frame
- [ ] CNNOnlyClassifier — CNN + global avg pool → Linear (ablation variant)
- [ ] CNNLSTM(num_classes, hidden_dim, lstm_layers) — CNN per frame → LSTM over sequence → Linear
  - Comment: why LSTM? temporal dependencies in video; why CNN? spatial hierarchy
- [ ] All models use forward(x) where x shape = (batch, frames, C, H, W)

---

### PHASE 6 — Mini-Project 4 Training & Ablation

#### mini_project_4/train.py
**[RUBRIC: Problem Definition — 2 marks | Experimental Design — 3 marks]**

- [ ] Header comment block: Problem, Motivation, Dataset (UCF-101, 5 classes)
- [ ] train_model(model, dataloader, val_loader, epochs, lr, model_name) — generic training loop
- [ ] run_ablation_study(data_dir):
  - Train Baseline MLP → record val accuracy
  - Train CNN-only → record val accuracy
  - Train CNN+LSTM (full) → record val accuracy
  - Return comparison dict
- [ ] plot_ablation_results(results_dict) — bar chart comparing models
- [ ] Save checkpoints: torch.save(state_dict, f'{model_name}_best.pth')
- [ ] Early stopping (patience=5) implemented as class EarlyStopping

---

### PHASE 7 — Mini-Project 4 Hyperparameter Tuning

#### mini_project_4/hyperparams.py
**[RUBRIC: Hyperparameter Optimization — 2 marks]**

- [ ] Comment block: tuning strategy explanation (grid search over small space)
- [ ] get_hyperparameter_grid() — returns dict of {lr, batch_size, lstm_hidden, dropout}
- [ ] grid_search(model_class, train_loader, val_loader, param_grid) — nested loop, log results to DataFrame
- [ ] LR_scheduler: torch.optim.lr_scheduler.StepLR (step decay) — justified in comment
- [ ] plot_hyperparam_results(df) — heatmap or line plot of val accuracy vs hyperparams

---

### PHASE 8 — Mini-Project 4 Evaluation

#### mini_project_4/evaluate.py
**[RUBRIC: Performance Evaluation — 3 marks]**

- [ ] compute_metrics(model, dataloader, classes):
  - Accuracy, per-class Precision, Recall, F1 (sklearn)
  - Confusion matrix (seaborn heatmap)
  - Comment: why F1? (balanced assessment, handles class imbalance)
- [ ] plot_confusion_matrix(cm, classes) — annotated heatmap
- [ ] statistical_reasoning_comment() — docstring explaining metric choices
- [ ] top_k_accuracy(model, dataloader, k=3) — Top-3 accuracy for 5-class problem

---

### PHASE 9 — Streamlit Apps

#### mini_project_3/app.py
- [ ] Sidebar: choose AE or GAN mode, set latent dim, epochs
- [ ] Tab 1 — Architecture: display model summary
- [ ] Tab 2 — Train & Loss Curves: run train (cached), show g/d/ae loss plots
- [ ] Tab 3 — Latent Space: PCA + t-SNE interactive plots (plotly)
- [ ] Tab 4 — Generated/Reconstructed Images: image grids
- [ ] Tab 5 — Metrics: MSE/SSIM table

#### mini_project_4/app.py
**[RUBRIC: Deployment — 3 marks]**
- [ ] Sidebar: upload .avi/.mp4 video OR select sample from dataset
- [ ] Tab 1 — Problem Overview: motivation text + dataset info
- [ ] Tab 2 — Data Engineering: show augmentation pipeline visually
- [ ] Tab 3 — Model & Architecture: architecture diagram, justification
- [ ] Tab 4 — Prediction: extract frames → run CNN+LSTM → show predicted class + confidence bar
- [ ] Tab 5 — Ablation & Evaluation: show comparison table, confusion matrix
- [ ] Tab 6 — Hyperparameter Tuning: show grid search table/plot

---

### PHASE 10 — Reproducibility & Documentation

#### requirements.txt
- [ ] torch, torchvision, streamlit, scikit-learn, matplotlib, seaborn, numpy, opencv-python, Pillow, plotly, scikit-image, pandas, tqdm

#### README.md
- [ ] Project overview, how to run, dataset setup instructions
- [ ] Rubric mapping table (criterion → file → function)
- [ ] Reproducibility notes (seed, environment, folder setup)

---

## Implementation Order (recommended)

```
1. requirements.txt + README.md skeleton
2. mini_project_3/utils.py (shared data loading)
3. mini_project_3/model.py (AE + GAN architectures)
4. mini_project_3/train.py (training loops)
5. mini_project_3/visualize.py (all plots)
6. mini_project_3/app.py (Streamlit UI)
7. mini_project_4/utils.py (video data engineering)
8. mini_project_4/model.py (CNN+LSTM + baselines)
9. mini_project_4/train.py (ablation study)
10. mini_project_4/hyperparams.py (grid search)
11. mini_project_4/evaluate.py (metrics)
12. mini_project_4/app.py (Streamlit deployment)
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Frame size 64x64 for MP3 | Fast training for AE/GAN, manageable memory |
| Frame size 112x112 for MP4 | Standard for action recognition (UCF-101 benchmarks) |
| 5 frames per video for MP4 | Uniform sampling, covers temporal span, lightweight |
| Latent dim = 128 | Rich enough for reconstruction, small enough to visualize |
| CNN+LSTM over 3D-CNN | Simpler, more interpretable, easier to ablate |
| Soft labels 0.9/0.1 | Proven GAN stabilization technique (Salimans et al.) |
| t-SNE perplexity = 30 | Recommended default for datasets up to ~10k points |

---

NOTE: All training functions accept epochs as parameter. For demo/Streamlit, use small epochs (5-10). For full training, use 50-100 epochs on Kaggle GPU.
