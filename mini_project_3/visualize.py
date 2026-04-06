# =============================================================
# mini_project_3/visualize.py
# [RUBRIC MP3: Latent Space Visualization — 3 marks]
# [RUBRIC MP3: Quality of Generated Output — 3 marks]
# [RUBRIC MP3: Analysis of Training Dynamics — 3 marks]
#
# This file implements:
#   1. Loss curve plotting (AE + G + D losses over epochs)
#   2. PCA latent space visualization
#   3. t-SNE latent space visualization
#   4. Generated image grid (GAN qualitative output)
#   5. Reconstructed image comparison (AE qualitative output)
#   6. Quantitative metrics: MSE + SSIM (AE quality)
# =============================================================

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (safe for Streamlit)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from skimage.metrics import structural_similarity as ssim_score
import io
from PIL import Image


# ──────────────────────────────────────────────────────────────────
# SECTION 1: TRAINING DYNAMICS — LOSS CURVES
# [RUBRIC: Analysis of training dynamics — loss curves for both networks]
# ──────────────────────────────────────────────────────────────────

def plot_loss_curves(
    ae_losses: list,
    g_losses: list = None,
    d_losses: list = None,
    title: str = "Training Loss Curves",
    return_fig: bool = False,
):
    """
    Plot training loss curves for AE and/or GAN.

    For AE: single reconstruction loss curve
    For GAN: dual curves for Generator and Discriminator separately

    [RUBRIC: Loss curves for both networks — G and D shown separately
             to detect mode collapse by watching D_loss → 0]

    Args:
        ae_losses:  AE reconstruction loss per epoch
        g_losses:   GAN Generator loss per epoch
        d_losses:   GAN Discriminator loss per epoch
        title:      Plot title
        return_fig: If True, return matplotlib Figure instead of showing

    Returns:
        fig if return_fig else None
    """
    n_plots = sum([
        ae_losses is not None and len(ae_losses) > 0,
        g_losses is not None and len(g_losses) > 0,
    ])
    if n_plots == 0:
        print("[visualize] No loss data provided.")
        return None

    if g_losses is not None and d_losses is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(title, fontsize=14, fontweight='bold')

        # AE reconstruction loss
        if ae_losses and len(ae_losses) > 0:
            ax = axes[0]
            epochs = range(1, len(ae_losses) + 1)
            ax.plot(epochs, ae_losses, 'b-o', markersize=4, linewidth=2, label='Recon Loss')
            ax.set_title("Autoencoder — Reconstruction Loss (MSE)")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MSE Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # GAN G + D losses on same plot
        ax2 = axes[1]
        epochs_gan = range(1, len(g_losses) + 1)
        ax2.plot(epochs_gan, g_losses, 'r-o', markersize=4, linewidth=2, label='Generator Loss')
        ax2.plot(epochs_gan, d_losses, 'g-s', markersize=4, linewidth=2, label='Discriminator Loss')
        ax2.set_title("GAN — Generator vs Discriminator Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("BCE Loss")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Annotate: ideal GAN training → both losses near ~0.69 (ln 2)
        ax2.axhline(y=0.693, color='black', linestyle='--', alpha=0.4,
                    label='Nash Equilibrium (ln2 ≈ 0.693)')
        ax2.legend()

    else:
        # Only AE losses
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        epochs = range(1, len(ae_losses) + 1)
        ax.plot(epochs, ae_losses, 'b-o', markersize=4, linewidth=2)
        ax.set_title("Autoencoder — Reconstruction Loss (MSE)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if return_fig:
        return fig
    plt.savefig("loss_curves.png", dpi=150, bbox_inches='tight')
    plt.show()
    return None


# ──────────────────────────────────────────────────────────────────
# SECTION 2: LATENT SPACE EXTRACTION
# Helper to collect latent codes and labels from the encoder
# ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_latent_codes(
    encoder: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
    max_samples: int = 1000,
) -> tuple:
    """
    Pass data through the encoder and collect (latent_vector, label) pairs.

    Returns:
        latents: numpy array (N, latent_dim)
        labels:  numpy array (N,)
    """
    encoder.eval()
    encoder = encoder.to(device)
    all_latents = []
    all_labels = []
    total = 0

    for images, labels in dataloader:
        if total >= max_samples:
            break
        images = images.to(device)
        z = encoder(images)
        all_latents.append(z.cpu().numpy())
        all_labels.append(labels.numpy())
        total += images.size(0)

    latents = np.vstack(all_latents)[:max_samples]
    labels = np.concatenate(all_labels)[:max_samples]
    return latents, labels


# ──────────────────────────────────────────────────────────────────
# SECTION 3: PCA LATENT SPACE VISUALIZATION
# [RUBRIC: Latent space visualization — PCA plot]
# ──────────────────────────────────────────────────────────────────

def plot_latent_space_pca(
    encoder: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_names: list,
    device: str = "cpu",
    max_samples: int = 500,
    return_fig: bool = False,
):
    """
    Apply PCA to reduce latent codes to 2D and create a scatter plot.

    PCA (Principal Component Analysis):
    - Linear dimensionality reduction
    - Finds directions of maximum variance in latent space
    - Fast, deterministic, good for initial visualization
    - If classes form clusters → encoder has learned class structure

    [RUBRIC: PCA latent space visualization, color-coded by class]

    Args:
        encoder:     Trained Encoder model
        dataloader:  DataLoader with labeled data
        class_names: List of class name strings
        device:      Compute device
        max_samples: Max points to visualize
        return_fig:  Return figure instead of showing

    Returns:
        fig if return_fig else None
    """
    print("[PCA] Extracting latent codes...")
    latents, labels = extract_latent_codes(encoder, dataloader, device, max_samples)

    print("[PCA] Running PCA (2 components)...")
    pca = PCA(n_components=2, random_state=42)
    latents_2d = pca.fit_transform(latents)

    explained = pca.explained_variance_ratio_
    print(f"[PCA] Variance explained: PC1={explained[0]:.2%}, PC2={explained[1]:.2%}")

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

    for i, cls in enumerate(class_names):
        mask = labels == i
        ax.scatter(
            latents_2d[mask, 0], latents_2d[mask, 1],
            c=[colors[i]], label=cls, alpha=0.7, s=30, edgecolors='none'
        )

    ax.set_title(
        f"Latent Space — PCA (2D)\n"
        f"Variance Explained: {explained[0]:.1%} + {explained[1]:.1%} = {sum(explained):.1%}",
        fontsize=13
    )
    ax.set_xlabel(f"PC1 ({explained[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%} variance)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if return_fig:
        return fig
    plt.savefig("pca_latent.png", dpi=150, bbox_inches='tight')
    plt.show()
    return None


# ──────────────────────────────────────────────────────────────────
# SECTION 4: t-SNE LATENT SPACE VISUALIZATION
# [RUBRIC: Latent space visualization — t-SNE plot]
# ──────────────────────────────────────────────────────────────────

def plot_latent_space_tsne(
    encoder: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_names: list,
    device: str = "cpu",
    max_samples: int = 500,
    perplexity: int = 30,
    return_fig: bool = False,
):
    """
    Apply t-SNE to reduce latent codes to 2D and create a scatter plot.

    t-SNE (t-Distributed Stochastic Neighbor Embedding):
    - Non-linear dimensionality reduction
    - Preserves LOCAL structure — nearby points in high-D stay nearby in 2D
    - Perplexity ~30 recommended for N < 10,000
    - Computationally slower than PCA, but reveals cluster structure better
    - If classes form tight clusters → encoder separates semantic content

    INTERPRETATION:
    - Tight clusters → encoder learned discriminative features
    - Overlapping clusters → encoder hasn't fully separated classes
    - Compare PCA vs t-SNE: t-SNE reveals non-linear structure PCA misses

    [RUBRIC: t-SNE latent space visualization, class-colored]

    Args:
        encoder:    Trained Encoder
        dataloader: Labeled DataLoader
        class_names: Class name list
        perplexity: t-SNE hyperparameter (typical: 5-50)
        return_fig: Return figure if True
    """
    print("[t-SNE] Extracting latent codes...")
    latents, labels = extract_latent_codes(encoder, dataloader, device, max_samples)

    print(f"[t-SNE] Running t-SNE (perplexity={perplexity})... this may take a moment")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=1000,
        random_state=42,
        learning_rate='auto',
        init='pca',
    )
    latents_2d = tsne.fit_transform(latents)
    print(f"[t-SNE] Done. KL divergence: {tsne.kl_divergence_:.4f}")

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

    for i, cls in enumerate(class_names):
        mask = labels == i
        ax.scatter(
            latents_2d[mask, 0], latents_2d[mask, 1],
            c=[colors[i]], label=cls, alpha=0.8, s=30, edgecolors='none'
        )

    ax.set_title(
        f"Latent Space — t-SNE (2D)\n"
        f"Perplexity={perplexity} | KL Divergence={tsne.kl_divergence_:.4f}",
        fontsize=13
    )
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if return_fig:
        return fig
    plt.savefig("tsne_latent.png", dpi=150, bbox_inches='tight')
    plt.show()
    return None


# ──────────────────────────────────────────────────────────────────
# SECTION 5: QUALITATIVE OUTPUT — GENERATED IMAGES (GAN)
# [RUBRIC: Quality of generated output — qualitative evaluation]
# ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def plot_generated_images(
    generator: nn.Module,
    latent_dim: int = 100,
    n: int = 16,
    device: str = "cpu",
    return_fig: bool = False,
):
    """
    Generate a grid of synthetic images from the GAN Generator.

    Qualitative evaluation: visually inspect whether generated images
    resemble real video frames from UCF-101 classes.

    [RUBRIC: Qualitative output — GAN image grid]

    Args:
        generator:  Trained Generator model
        latent_dim: Noise vector size
        n:          Number of images to generate (should be a perfect square)
        device:     Compute device
        return_fig: Return figure if True
    """
    generator.eval()
    generator = generator.to(device)

    noise = torch.randn(n, latent_dim, device=device)
    fake_imgs = generator(noise).cpu()

    # Denormalize from [-1,1] to [0,1] for display
    fake_imgs = (fake_imgs * 0.5 + 0.5).clamp(0, 1)

    grid_size = int(np.sqrt(n))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))
    fig.suptitle("GAN Generated Images (Qualitative Evaluation)", fontsize=13, fontweight='bold')

    for i, ax in enumerate(axes.flatten()):
        img = fake_imgs[i].permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.axis('off')

    plt.tight_layout()

    if return_fig:
        return fig
    plt.savefig("generated_images.png", dpi=150, bbox_inches='tight')
    plt.show()
    return None


# ──────────────────────────────────────────────────────────────────
# SECTION 6: QUALITATIVE OUTPUT — RECONSTRUCTIONS (AE)
# [RUBRIC: Quality of generated output — AE reconstruction quality]
# ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def plot_reconstructed_images(
    autoencoder: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    n: int = 8,
    device: str = "cpu",
    return_fig: bool = False,
):
    """
    Show original vs reconstructed image pairs side by side.

    Qualitative evaluation: if reconstructions are sharp and preserve
    colors/shapes → AE has learned meaningful representations.

    [RUBRIC: AE qualitative output — original vs reconstructed pairs]
    """
    autoencoder.eval()
    autoencoder = autoencoder.to(device)

    images, labels = next(iter(dataloader))
    images = images[:n].to(device)
    x_hat, _ = autoencoder(images)

    # Denormalize both
    orig = (images.cpu() * 0.5 + 0.5).clamp(0, 1)
    recon = (x_hat.cpu() * 0.5 + 0.5).clamp(0, 1)

    fig, axes = plt.subplots(2, n, figsize=(n * 2, 5))
    fig.suptitle("Autoencoder: Original (top) vs Reconstructed (bottom)",
                 fontsize=13, fontweight='bold')

    for i in range(n):
        axes[0, i].imshow(orig[i].permute(1, 2, 0).numpy())
        axes[0, i].axis('off')
        axes[0, i].set_title("Original", fontsize=8)

        axes[1, i].imshow(recon[i].permute(1, 2, 0).numpy())
        axes[1, i].axis('off')
        axes[1, i].set_title("Recon", fontsize=8)

    plt.tight_layout()

    if return_fig:
        return fig
    plt.savefig("reconstructed_images.png", dpi=150, bbox_inches='tight')
    plt.show()
    return None


# ──────────────────────────────────────────────────────────────────
# SECTION 7: QUANTITATIVE METRICS — MSE + SSIM
# [RUBRIC: Quality of generated output — quantitative evaluation]
# ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_reconstruction_metrics(
    autoencoder: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
    max_batches: int = 10,
) -> dict:
    """
    Compute quantitative reconstruction quality metrics:

    1. MSE (Mean Squared Error):
       - Lower is better
       - Measures per-pixel squared difference
       - Directly optimized by the AE

    2. SSIM (Structural Similarity Index):
       - Range [0, 1] — higher is better
       - Measures structural similarity (luminance, contrast, structure)
       - Correlates better with human perception than MSE
       - SSIM > 0.7 indicates good reconstruction

    [RUBRIC: Quantitative evaluation — MSE and SSIM scores]

    Returns:
        dict with 'avg_mse', 'avg_ssim', 'interpretation' keys
    """
    autoencoder.eval()
    autoencoder = autoencoder.to(device)

    mse_scores = []
    ssim_scores = []
    mse_fn = nn.MSELoss()

    for batch_idx, (images, _) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        images = images.to(device)
        x_hat, _ = autoencoder(images)

        # MSE per batch
        mse = mse_fn(x_hat, images).item()
        mse_scores.append(mse)

        # SSIM: convert to numpy [0,1] range
        orig_np = ((images.cpu() * 0.5 + 0.5).clamp(0, 1)).numpy()  # (B, 3, H, W)
        recon_np = ((x_hat.cpu() * 0.5 + 0.5).clamp(0, 1)).numpy()

        for j in range(orig_np.shape[0]):
            o = orig_np[j].transpose(1, 2, 0)  # (H, W, 3)
            r = recon_np[j].transpose(1, 2, 0)
            s = ssim_score(o, r, channel_axis=2, data_range=1.0)
            ssim_scores.append(s)

    avg_mse = np.mean(mse_scores)
    avg_ssim = np.mean(ssim_scores)

    # Qualitative interpretation
    if avg_ssim > 0.80:
        interpretation = "Excellent reconstruction quality"
    elif avg_ssim > 0.60:
        interpretation = "Good reconstruction — minor detail loss"
    elif avg_ssim > 0.40:
        interpretation = "Moderate — structure preserved, details blurry"
    else:
        interpretation = "Poor reconstruction — model needs more training"

    result = {
        "avg_mse": round(avg_mse, 6),
        "avg_ssim": round(avg_ssim, 4),
        "interpretation": interpretation,
        "n_samples": len(ssim_scores),
    }
    print(f"[Metrics] MSE: {avg_mse:.6f} | SSIM: {avg_ssim:.4f} | {interpretation}")
    return result


# ──────────────────────────────────────────────────────────────────
# SECTION 8: Helper — Convert matplotlib fig to PIL Image (for Streamlit)
# ──────────────────────────────────────────────────────────────────

def fig_to_pil(fig) -> Image.Image:
    """Convert a matplotlib Figure to a PIL Image for Streamlit display."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return Image.open(buf)


# ──────────────────────────────────────────────────────────────────
# Entry point: test visualize functions are importable
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[visualize.py] Available functions:")
    print("  - plot_loss_curves(ae_losses, g_losses, d_losses)")
    print("  - plot_latent_space_pca(encoder, dataloader, class_names)")
    print("  - plot_latent_space_tsne(encoder, dataloader, class_names)")
    print("  - plot_generated_images(generator, latent_dim, n)")
    print("  - plot_reconstructed_images(autoencoder, dataloader, n)")
    print("  - compute_reconstruction_metrics(autoencoder, dataloader)")
    print("  - fig_to_pil(fig)")
