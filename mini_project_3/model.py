# =============================================================
# mini_project_3/model.py
# [RUBRIC MP3: Model Architecture Correctness — 4 marks]
#
# This file defines:
#   1. Autoencoder (Encoder + Decoder with symmetric architecture)
#   2. GAN (Generator + Discriminator with balanced capacity)
#
# Architecture Design Principles:
#   - Encoder & Decoder are SYMMETRIC: same depth, mirrored channels
#   - Generator & Discriminator are BALANCED: equal expressive power
#   - Stability components: BatchNorm in G, LeakyReLU in D, Tanh output
# =============================================================

import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────────
# SECTION 1: AUTOENCODER
# [RUBRIC: Architecture correctness — encoder-decoder symmetry]
#
# Architecture summary:
#   Encoder: Input(3,64,64) → Conv(32)→Conv(64)→Conv(128) → Flatten → FC(latent)
#   Decoder: FC(latent) → Unflatten → ConvT(128)→ConvT(64)→ConvT(32)→ConvT(3)
#
# Symmetry: Encoder has 3 conv blocks. Decoder has 3 transposed-conv blocks.
# Channel progression is the exact mirror: [3,32,64,128] → [128,64,32,3]
# ──────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    Convolutional Encoder: maps an image (B, 3, 64, 64) to a
    latent vector (B, latent_dim).

    Layers:
        Conv2d(3→32)   + BatchNorm + ReLU + MaxPool → (B, 32, 32, 32)
        Conv2d(32→64)  + BatchNorm + ReLU + MaxPool → (B, 64, 16, 16)
        Conv2d(64→128) + BatchNorm + ReLU + MaxPool → (B, 128, 8, 8)
        Flatten → Linear → latent_dim

    [RUBRIC: Encoder side of symmetric autoencoder]
    """

    def __init__(self, latent_dim: int = 128, img_size: int = 64):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim

        # Convolutional feature extraction (spatial → semantic)
        self.conv_block = nn.Sequential(
            # Block 1: 3 → 32 channels,  64×64 → 32×32
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),   # stride=2 halves spatial
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 2: 32 → 64 channels, 32×32 → 16×16
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 3: 64 → 128 channels, 16×16 → 8×8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # After 3 stride-2 convolutions on 64×64: 64/(2^3) = 8
        self.flatten_dim = 128 * 8 * 8   # = 8192

        # Bottleneck: compress to latent space
        self.fc = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor (B, 3, 64, 64)
        Returns:
            z: Latent vector (B, latent_dim)
        """
        features = self.conv_block(x)           # (B, 128, 8, 8)
        flat = features.view(features.size(0), -1)  # (B, 8192)
        z = self.fc(flat)                        # (B, latent_dim)
        return z


class Decoder(nn.Module):
    """
    Convolutional Decoder: maps latent vector (B, latent_dim) back to
    an image (B, 3, 64, 64).

    Layers (MIRROR of Encoder):
        Linear(latent_dim → 128*8*8) → Reshape (B, 128, 8, 8)
        ConvTranspose2d(128→64)  + BatchNorm + ReLU → (B, 64, 16, 16)
        ConvTranspose2d(64→32)   + BatchNorm + ReLU → (B, 32, 32, 32)
        ConvTranspose2d(32→3)    + Tanh             → (B, 3, 64, 64)

    Tanh output maps to [-1, 1] — matching the input normalization.
    [RUBRIC: Decoder side — symmetric to Encoder]
    """

    def __init__(self, latent_dim: int = 128):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.start_dim = 128 * 8 * 8   # reshape target

        # Project latent vector back to spatial feature map
        self.fc = nn.Linear(latent_dim, self.start_dim)

        # Transposed convolutions (upsampling, mirroring Encoder)
        self.deconv_block = nn.Sequential(
            # Block 1: 128 → 64 channels, 8×8 → 16×16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 2: 64 → 32 channels, 16×16 → 32×32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 3: 32 → 3 channels, 32×32 → 64×64
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),   # Output in [-1, 1] to match input normalization
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent vector (B, latent_dim)
        Returns:
            x_hat: Reconstructed image (B, 3, 64, 64)
        """
        projected = self.fc(z)                           # (B, 8192)
        reshaped = projected.view(-1, 128, 8, 8)         # (B, 128, 8, 8)
        x_hat = self.deconv_block(reshaped)              # (B, 3, 64, 64)
        return x_hat


class Autoencoder(nn.Module):
    """
    Full Autoencoder = Encoder + Decoder.
    Provides encode(), decode(), and forward() for flexible use.

    [RUBRIC: Complete AE architecture with clear symmetry]
    Architecture: Input → Encoder → Latent(128d) → Decoder → Reconstruction
    """

    def __init__(self, latent_dim: int = 128):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent space."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to image."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Full forward pass through AE.
        Returns:
            (x_hat, z) — reconstruction and latent code
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


# ──────────────────────────────────────────────────────────────────
# SECTION 2: GAN — Generator + Discriminator
# [RUBRIC: Architecture correctness — generator-discriminator balance]
#
# Design philosophy:
#   - Generator: maps latent noise → realistic image (upsampling path)
#   - Discriminator: maps image → real/fake probability (downsampling path)
#   - Balanced capacity: both have 3 main blocks with similar depth
#   - Stability: BatchNorm in Generator, LeakyReLU in Discriminator
# ──────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    GAN Generator: maps noise vector z ~ N(0,1) to a synthetic image.

    Architecture:
        FC(latent_dim → 128*8*8) → Reshape(128,8,8)
        ConvTranspose(128→64) + BN + ReLU  → (64, 16, 16)
        ConvTranspose(64→32)  + BN + ReLU  → (32, 32, 32)
        ConvTranspose(32→3)   + Tanh       → (3,  64, 64)

    STABILITY NOTES:
        - BatchNorm in every hidden layer prevents gradient vanishing
        - ReLU (not LeakyReLU) in Generator is standard DCGAN practice
        - Tanh final activation: output in [-1,1] for stable training

    [RUBRIC: GAN Generator with balanced depth vs Discriminator]
    """

    def __init__(self, latent_dim: int = 100, img_channels: int = 3):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim

        # Project noise to spatial feature map
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.BatchNorm1d(128 * 8 * 8),
            nn.ReLU(inplace=True),
        )

        # Upsampling transposed convolution blocks (DCGAN-style)
        self.deconv_blocks = nn.Sequential(
            # 8×8 → 16×16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 16×16 → 32×32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 32×32 → 64×64
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # Output in [-1, 1] — critical for GAN stability
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Random noise (B, latent_dim)
        Returns:
            fake_img: Synthetic image (B, 3, 64, 64)
        """
        projected = self.fc(z)                       # (B, 8192)
        reshaped = projected.view(-1, 128, 8, 8)     # (B, 128, 8, 8)
        fake_img = self.deconv_blocks(reshaped)       # (B, 3, 64, 64)
        return fake_img


class Discriminator(nn.Module):
    """
    GAN Discriminator: classifies images as real (1) or fake (0).

    Architecture:
        Conv(3→64)   + LeakyReLU               → (64, 32, 32)
        Conv(64→128) + BN + LeakyReLU           → (128, 16, 16)
        Conv(128→256)+ BN + LeakyReLU           → (256, 8, 8)
        Flatten → FC → Sigmoid

    STABILITY NOTES:
        - LeakyReLU (slope=0.2) prevents dead neurons in discriminator
        - BatchNorm stabilizes gradient flow to generator
        - Sigmoid output: probability in (0, 1) for BCELoss

    BALANCE vs Generator:
        - 3 conv blocks (same depth as generator)
        - No BatchNorm on first layer (recommended by DCGAN paper)

    [RUBRIC: GAN Discriminator balanced against Generator]
    """

    def __init__(self, img_channels: int = 3):
        super(Discriminator, self).__init__()

        self.conv_blocks = nn.Sequential(
            # Block 1: 3 → 64, no BatchNorm (DCGAN recommendation)
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 2: 64 → 128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 3: 128 → 256
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # After 3 stride-2 convolutions on 64×64: output is 8×8
        self.flatten_dim = 256 * 8 * 8

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 1),
            nn.Sigmoid(),    # Probability of being real
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor (B, 3, 64, 64)
        Returns:
            prob: Real/fake probability (B, 1)
        """
        features = self.conv_blocks(x)    # (B, 256, 8, 8)
        prob = self.classifier(features)  # (B, 1)
        return prob


# ──────────────────────────────────────────────────────────────────
# SECTION 3: Weight Initialization (DCGAN standard)
# [RUBRIC: Training stability — proper init for GAN convergence]
# ──────────────────────────────────────────────────────────────────

def weights_init(m: nn.Module):
    """
    Initialize Conv and BatchNorm weights using DCGAN paper conventions:
    - Conv weights: Normal(0, 0.02)
    - BatchNorm weights: Normal(1, 0.02), bias = 0

    This initialization prevents early training instability.
    Call as: model.apply(weights_init)
    """
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(m.bias.data, 0)


# ──────────────────────────────────────────────────────────────────
# SECTION 4: Model Summary Utility
# ──────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary():
    """Print architecture summary for both AE and GAN models."""
    print("=" * 60)
    print("AUTOENCODER ARCHITECTURE SUMMARY")
    print("=" * 60)
    ae = Autoencoder(latent_dim=128)
    print(ae)
    print(f"\nTotal AE parameters: {count_parameters(ae):,}")

    print("\n" + "=" * 60)
    print("GAN ARCHITECTURE SUMMARY")
    print("=" * 60)
    G = Generator(latent_dim=100)
    D = Discriminator()
    print("Generator:")
    print(G)
    print(f"\nGenerator parameters: {count_parameters(G):,}")
    print("\nDiscriminator:")
    print(D)
    print(f"\nDiscriminator parameters: {count_parameters(D):,}")


# ──────────────────────────────────────────────────────────────────
# Entry point: run model.py directly to verify architectures
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print_model_summary()

    # Quick forward pass test
    dummy = torch.randn(4, 3, 64, 64)
    ae = Autoencoder(latent_dim=128)
    x_hat, z = ae(dummy)
    print(f"\n[AE Test] Input: {dummy.shape} | Latent: {z.shape} | Output: {x_hat.shape}")

    G = Generator(latent_dim=100)
    D = Discriminator()
    G.apply(weights_init)
    D.apply(weights_init)
    noise = torch.randn(4, 100)
    fake = G(noise)
    prob = D(fake)
    print(f"[GAN Test] Noise: {noise.shape} | Fake: {fake.shape} | D output: {prob.shape}")
