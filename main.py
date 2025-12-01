#!/usr/bin/env python
"""
Complete implementation of Zero-DCE Deep Learning for enhancing Permanently
Shadowed Regions (PSRs) in Chandrayaan-2 OHRC imagery.

This single file contains the entire pipeline:
  1. PDS4 Data Loading (memory-efficient tile extraction)
  2. Preprocessing (12-bit normalization, gamma correction)
  3. Zero-DCE Model Architecture (unsupervised light enhancement)
  4. Training Loop (with custom loss functions)
  5. Inference & Visualization

Usage:
    uv run python main.py

Author: Rohit
Project: ISRO Internship Application - Lunar Shadow Enhancement
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from pathlib import Path
from typing import Tuple, List, Optional

import pds4_tools


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Data paths
    "data_path": (
        "data/ch2_ohr_ncp_20230303T0152168201_d_img_n18/"
        "data/calibrated/20230303/"
        "ch2_ohr_ncp_20230303T0152168201_d_img_n18.xml"
    ),
    # Tile extraction
    "tile_size": 512,
    "tile_x": None,  # None = auto (center)
    "tile_y": None,  # None = auto (center)
    # Model settings
    "n_curves": 8,  # Number of enhancement curve iterations
    "bit_depth": 12,  # OHRC sensor bit-depth
    # Training settings
    "enable_training": True,
    "epochs": 100,
    "learning_rate": 1e-4,
    "target_exposure": 0.5,  # Target brightness level (0-1)
    # Loss weights
    "spa_weight": 1.0,  # Spatial consistency
    "exp_weight": 10.0,  # Exposure control
    "col_weight": 5.0,  # Color constancy
    "smooth_weight": 200.0,  # Illumination smoothness
    # Comparison
    "gamma": 2.2,  # Gamma for traditional correction
    # Output
    "output_dir": "results",
    "save_results": True,
    "show_plots": True,
}


# =============================================================================
# DATA LOADING
# =============================================================================


class OHRCDataLoader:
    """
    Memory-efficient data loader for Chandrayaan-2 OHRC PDS4 imagery.

    Uses lazy loading to handle ~1GB images without loading entire
    array into RAM. Provides tile-based access via get_tile().
    """

    def __init__(self, xml_path: str):
        """Load PDS4 metadata (lazy - doesn't load full array)."""
        self.xml_path = Path(xml_path)

        if not self.xml_path.exists():
            raise FileNotFoundError(f"PDS4 file not found: {self.xml_path}")

        print(f"[INFO] Loading: {self.xml_path.name}")

        # Lazy load - only reads metadata, not full array
        self._structure = pds4_tools.read(str(self.xml_path), lazy_load=True)
        self._data = self._structure[0].data

        self.shape = self._data.shape
        self.dtype = self._data.dtype
        self.bit_depth = self._detect_bit_depth()

        print(f"   Shape: {self.shape[0]:,} × {self.shape[1]:,} pixels")
        print(f"   Dtype: {self.dtype}, Bit-depth: {self.bit_depth}-bit")

    def _detect_bit_depth(self) -> int:
        """Detect bit-depth from sample data."""
        sample = self._data[:1000, :1000]
        max_val = np.max(sample)
        return 12 if max_val > 1023 else 10 if max_val > 255 else 8

    def get_tile(self, x: int, y: int, size: int = 512) -> np.ndarray:
        """
        Extract a square tile from the image.

        Args:
            x: Horizontal center position (column)
            y: Vertical center position (row)
            size: Tile size (default 512×512)

        Returns:
            Extracted tile as numpy array
        """
        height, width = self.shape
        half = size // 2

        # Calculate bounds with clipping
        y_start = max(0, y - half)
        y_end = min(height, y + half)
        x_start = max(0, x - half)
        x_end = min(width, x + half)

        return self._data[y_start:y_end, x_start:x_end].copy()

    def get_statistics(self, n_samples: int = 10000) -> dict:
        """Compute statistics from random samples."""
        y_idx = np.random.randint(0, self.shape[0], n_samples)
        x_idx = np.random.randint(0, self.shape[1], n_samples)
        samples = self._data[y_idx, x_idx]

        return {
            "min": float(np.min(samples)),
            "max": float(np.max(samples)),
            "mean": float(np.mean(samples)),
            "std": float(np.std(samples)),
        }


# =============================================================================
# PREPROCESSING
# =============================================================================


class Preprocessor:
    """
    Preprocessing utilities for OHRC imagery.

    Handles normalization from raw bit-depth to [0,1] range,
    gamma correction, and PyTorch tensor conversion.
    """

    def __init__(self, bit_depth: int = 12, device: str = "auto"):
        self.bit_depth = bit_depth
        self.max_value = (2**bit_depth) - 1
        self.device = (
            "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        )

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize raw integers to [0, 1] float32."""
        return data.astype(np.float32) / self.max_value

    def apply_gamma(self, data: np.ndarray, gamma: float = 2.2) -> np.ndarray:
        """Apply gamma correction: pixel^(1/gamma)."""
        return np.power(np.clip(data, 0, 1), 1.0 / gamma).astype(np.float32)

    def to_tensor(self, data: np.ndarray) -> torch.Tensor:
        """Convert numpy array to PyTorch tensor (B, 3, H, W)."""
        tensor = torch.from_numpy(data).unsqueeze(0)  # Add channel dim
        tensor = tensor.repeat(3, 1, 1)  # Grayscale to RGB (3 channels)
        tensor = tensor.unsqueeze(0)  # Add batch dim
        return tensor.to(self.device)

    def from_tensor(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor back to numpy (grayscale)."""
        array = tensor.detach().cpu().numpy()[0]  # Remove batch dim
        return np.mean(array, axis=0)  # RGB to grayscale


# =============================================================================
# ZERO-DCE MODEL
# =============================================================================


class ZeroDCE(nn.Module):
    """
    Zero-Reference Deep Curve Estimation Network.

    Estimates pixel-wise light-enhancement curves for low-light image
    enhancement without requiring paired training data.

    Reference: "Zero-Reference Deep Curve Estimation for Low-Light
               Image Enhancement" - Guo et al., CVPR 2020

    Architecture:
        - 7 convolutional layers with skip connections
        - Outputs n_curves × 3 channel curve parameter maps
        - Light Enhancement: LE(x) = x + α × x × (1 - x)
    """

    def __init__(self, n_curves: int = 8):
        super().__init__()
        self.n_curves = n_curves
        out_ch = n_curves * 3  # n curves × 3 RGB channels

        # Encoder
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 32, 3, 1, 1)

        # Decoder with skip connections
        self.conv5 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv6 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv7 = nn.Conv2d(64, out_ch, 3, 1, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input image (B, 3, H, W) in [0, 1]

        Returns:
            enhanced: Enhanced image (B, 3, H, W)
            curves: Curve parameters (B, n×3, H, W)
        """
        # Encoder path
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))

        # Decoder with skip connections
        x5 = F.relu(self.conv5(torch.cat([x3, x4], dim=1)))
        x6 = F.relu(self.conv6(torch.cat([x2, x5], dim=1)))
        curves = torch.tanh(self.conv7(torch.cat([x1, x6], dim=1)))

        # Apply enhancement curves iteratively
        enhanced = x
        for i in range(self.n_curves):
            alpha = curves[:, i * 3 : (i + 1) * 3, :, :]
            # LE-curve: enhanced = enhanced + α × enhanced × (1 - enhanced)
            enhanced = enhanced + alpha * enhanced * (1 - enhanced)

        return torch.clamp(enhanced, 0, 1), curves

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# ZERO-DCE LOSS FUNCTIONS
# =============================================================================


class ZeroDCELoss(nn.Module):
    """
    Combined loss function for unsupervised Zero-DCE training.

    Components:
        - Spatial Consistency: Preserve local structure
        - Exposure Control: Achieve target brightness
        - Color Constancy: Prevent color shifts
        - Illumination Smoothness: Smooth curve parameters
    """

    def __init__(
        self,
        spa_weight: float = 1.0,
        exp_weight: float = 10.0,
        col_weight: float = 5.0,
        smooth_weight: float = 200.0,
        target_exposure: float = 0.6,
        patch_size: int = 16,
    ):
        super().__init__()
        self.spa_weight = spa_weight
        self.exp_weight = exp_weight
        self.col_weight = col_weight
        self.smooth_weight = smooth_weight
        self.target_exposure = target_exposure
        self.patch_size = patch_size

        # Gradient kernels for spatial loss
        self.register_buffer(
            "kernel_left",
            torch.tensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=torch.float32).view(
                1, 1, 3, 3
            ),
        )
        self.register_buffer(
            "kernel_right",
            torch.tensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]], dtype=torch.float32).view(
                1, 1, 3, 3
            ),
        )
        self.register_buffer(
            "kernel_up",
            torch.tensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32).view(
                1, 1, 3, 3
            ),
        )
        self.register_buffer(
            "kernel_down",
            torch.tensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=torch.float32).view(
                1, 1, 3, 3
            ),
        )

    def forward(
        self, enhanced: torch.Tensor, original: torch.Tensor, curves: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Compute total loss and individual components."""

        L_spa = self._spatial_consistency_loss(enhanced, original)
        L_exp = self._exposure_control_loss(enhanced)
        L_col = self._color_constancy_loss(enhanced)
        L_smooth = self._illumination_smoothness_loss(curves)

        total = (
            self.spa_weight * L_spa
            + self.exp_weight * L_exp
            + self.col_weight * L_col
            + self.smooth_weight * L_smooth
        )

        return total, {
            "total": total.item(),
            "spatial": L_spa.item(),
            "exposure": L_exp.item(),
            "color": L_col.item(),
            "smoothness": L_smooth.item(),
        }

    def _spatial_consistency_loss(
        self, enhanced: torch.Tensor, original: torch.Tensor
    ) -> torch.Tensor:
        """Preserve local structure between original and enhanced."""
        enh_gray = torch.mean(enhanced, dim=1, keepdim=True)
        org_gray = torch.mean(original, dim=1, keepdim=True)

        enh_pool = F.avg_pool2d(enh_gray, 4)
        org_pool = F.avg_pool2d(org_gray, 4)

        d_left = F.conv2d(enh_pool, self.kernel_left, padding=1) - F.conv2d(
            org_pool, self.kernel_left, padding=1
        )
        d_right = F.conv2d(enh_pool, self.kernel_right, padding=1) - F.conv2d(
            org_pool, self.kernel_right, padding=1
        )
        d_up = F.conv2d(enh_pool, self.kernel_up, padding=1) - F.conv2d(
            org_pool, self.kernel_up, padding=1
        )
        d_down = F.conv2d(enh_pool, self.kernel_down, padding=1) - F.conv2d(
            org_pool, self.kernel_down, padding=1
        )

        return torch.mean(d_left**2 + d_right**2 + d_up**2 + d_down**2)

    def _exposure_control_loss(self, enhanced: torch.Tensor) -> torch.Tensor:
        """Guide enhancement toward target exposure level."""
        luminance = torch.mean(enhanced, dim=1, keepdim=True)
        patches = F.avg_pool2d(luminance, self.patch_size)
        return torch.mean((patches - self.target_exposure) ** 2)

    def _color_constancy_loss(self, enhanced: torch.Tensor) -> torch.Tensor:
        """Prevent color shifts (keep RGB balanced for grayscale)."""
        mean_r = torch.mean(enhanced[:, 0, :, :])
        mean_g = torch.mean(enhanced[:, 1, :, :])
        mean_b = torch.mean(enhanced[:, 2, :, :])
        return (
            (mean_r - mean_g) ** 2 + (mean_r - mean_b) ** 2 + (mean_g - mean_b) ** 2
        ) / 3

    def _illumination_smoothness_loss(self, curves: torch.Tensor) -> torch.Tensor:
        """Encourage smooth curve parameters (total variation)."""
        grad_x = torch.abs(curves[:, :, :, :-1] - curves[:, :, :, 1:])
        grad_y = torch.abs(curves[:, :, :-1, :] - curves[:, :, 1:, :])
        return torch.mean(grad_x) + torch.mean(grad_y)


# =============================================================================
# SHADOW ENHANCER (MAIN PIPELINE)
# =============================================================================


class ShadowEnhancer:
    """
    Complete shadow enhancement pipeline.

    Combines data loading, preprocessing, model inference,
    and optional unsupervised training.
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"\n[INFO] Initializing Shadow Enhancer on {self.device.upper()}")

        # Create model
        self.model = ZeroDCE(n_curves=config["n_curves"]).to(self.device)
        print(f"   Model: Zero-DCE ({self.model.count_parameters():,} parameters)")

        # Create loss function
        self.criterion = ZeroDCELoss(
            spa_weight=config["spa_weight"],
            exp_weight=config["exp_weight"],
            col_weight=config["col_weight"],
            smooth_weight=config["smooth_weight"],
            target_exposure=config["target_exposure"],
        ).to(self.device)

        # Preprocessor
        self.preprocessor = Preprocessor(
            bit_depth=config["bit_depth"], device=self.device
        )

    def train(self, tile: np.ndarray, epochs: int, lr: float = 1e-4) -> List[dict]:
        """
        Train the model on a single tile (unsupervised).

        Args:
            tile: Raw image tile
            epochs: Number of training epochs
            lr: Learning rate

        Returns:
            Training history
        """
        print(f"\n[TRAIN] Training for {epochs} epochs...")

        # Prepare data
        normalized = self.preprocessor.normalize(tile)
        tensor = self.preprocessor.to_tensor(normalized)

        # Optimizer
        optimizer = Adam(self.model.parameters(), lr=lr)

        # Training loop
        self.model.train()
        history = []

        for epoch in range(epochs):
            optimizer.zero_grad()

            enhanced, curves = self.model(tensor)
            loss, components = self.criterion(enhanced, tensor, curves)

            loss.backward()
            optimizer.step()

            history.append(components)

            # Progress logging
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(
                    f"   Epoch {epoch+1:3d}/{epochs} | "
                    f"Loss: {components['total']:.4f} | "
                    f"Exp: {components['exposure']:.4f}"
                )

        self.model.eval()
        print(f"   [DONE] Training complete. Final loss: {history[-1]['total']:.4f}")

        return history

    def enhance(self, tile: np.ndarray) -> np.ndarray:
        """Enhance a single tile using the trained model."""
        normalized = self.preprocessor.normalize(tile)
        tensor = self.preprocessor.to_tensor(normalized)

        self.model.eval()
        with torch.no_grad():
            enhanced, _ = self.model(tensor)

        return self.preprocessor.from_tensor(enhanced)

    def compare(
        self, tile: np.ndarray, gamma: float = 2.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate comparison outputs.

        Returns:
            (original_normalized, gamma_corrected, ai_enhanced)
        """
        original = self.preprocessor.normalize(tile)
        gamma_corrected = self.preprocessor.apply_gamma(original, gamma)
        ai_enhanced = self.enhance(tile)

        return original, gamma_corrected, ai_enhanced


# =============================================================================
# VISUALIZATION
# =============================================================================


def plot_comparison(
    original: np.ndarray,
    gamma: np.ndarray,
    enhanced: np.ndarray,
    gamma_value: float = 2.2,
    save_path: Optional[str] = None,
):
    """
    Create side-by-side comparison figure.

    Shows: Original | Gamma Corrected | AI Enhanced
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

    fig.suptitle("Lunar Shadow Enhancement Comparison", fontsize=14, fontweight="bold")

    images = [
        (original, "Original\n(Normalized from 12-bit)"),
        (gamma, f"Gamma Corrected\n(γ = {gamma_value})"),
        (enhanced, "Zero-DCE Enhanced\n(AI)"),
    ]

    for ax, (img, title) in zip(axes, images):
        im = ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

        # Stats overlay
        stats = f"μ={np.mean(img):.3f}\nσ={np.std(img):.3f}"
        ax.text(
            0.02,
            0.98,
            stats,
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            ha="left",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"   [SAVED] {save_path}")

    return fig


def plot_histograms(
    original: np.ndarray, enhanced: np.ndarray, save_path: Optional[str] = None
):
    """Plot intensity distribution comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150)

    fig.suptitle("Intensity Distribution Analysis", fontsize=14, fontweight="bold")

    axes[0].hist(original.ravel(), bins=100, color="steelblue", alpha=0.7, density=True)
    axes[0].axvline(
        np.mean(original),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(original):.3f}",
    )
    axes[0].set_title("Original")
    axes[0].set_xlabel("Pixel Intensity")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    axes[1].hist(enhanced.ravel(), bins=100, color="coral", alpha=0.7, density=True)
    axes[1].axvline(
        np.mean(enhanced),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(enhanced):.3f}",
    )
    axes[1].set_title("AI Enhanced")
    axes[1].set_xlabel("Pixel Intensity")
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"   [SAVED] {save_path}")

    return fig


def plot_training_curves(history: List[dict], save_path: Optional[str] = None):
    """Plot training loss curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    fig.suptitle("Zero-DCE Training Progress", fontsize=14, fontweight="bold")

    epochs = range(1, len(history) + 1)

    axes[0, 0].plot(epochs, [h["total"] for h in history], "b-", lw=2)
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, [h["exposure"] for h in history], "r-", lw=2)
    axes[0, 1].set_title("Exposure Control Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, [h["spatial"] for h in history], "g-", lw=2)
    axes[1, 0].set_title("Spatial Consistency Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, [h["smoothness"] for h in history], "m-", lw=2)
    axes[1, 1].set_title("Smoothness Loss")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"   [SAVED] {save_path}")

    return fig


def plot_enhancement_heatmap(
    original: np.ndarray,
    enhanced: np.ndarray,
    colormap: str = "magma",
    save_path: Optional[str] = None,
):
    """
    Generate a heatmap overlay showing where the AI made changes.

    This visualization is crucial for scientific analysis as it reveals
    exactly which regions (hidden craters, PSR details) were enhanced.

    Args:
        original: Original normalized image
        enhanced: AI-enhanced image
        colormap: Colormap for difference visualization ('magma', 'jet', 'inferno', 'hot')
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    # Calculate the enhancement difference
    difference = enhanced - original

    # Create figure with 4 panels
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=150)
    fig.suptitle(
        "Enhancement Analysis: Where AI Revealed Hidden Features",
        fontsize=14,
        fontweight="bold",
    )

    # Panel 1: Original image
    im1 = axes[0, 0].imshow(original, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Original (Dark Regions)", fontsize=11)
    axes[0, 0].axis("off")
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, label="Intensity")

    # Panel 2: Enhanced image
    im2 = axes[0, 1].imshow(enhanced, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("AI Enhanced", fontsize=11)
    axes[0, 1].axis("off")
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, label="Intensity")

    # Panel 3: Difference heatmap (where enhancement occurred)
    # Normalize difference for better visualization
    diff_min, diff_max = difference.min(), difference.max()
    im3 = axes[1, 0].imshow(difference, cmap=colormap, vmin=diff_min, vmax=diff_max)
    axes[1, 0].set_title(
        f"Enhancement Heatmap (Δ = Enhanced - Original)\n"
        f"Range: [{diff_min:.3f}, {diff_max:.3f}]",
        fontsize=11,
    )
    axes[1, 0].axis("off")
    cbar3 = plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
    cbar3.set_label("Enhancement Intensity (Δ)", fontsize=10)

    # Panel 4: Overlay - Original with heatmap overlay
    # Show original in grayscale with enhancement regions highlighted
    axes[1, 1].imshow(original, cmap="gray", vmin=0, vmax=0.5)  # Slightly brighten base

    # Overlay the positive differences (where brightening occurred)
    positive_diff = np.clip(difference, 0, None)  # Only positive changes
    im4 = axes[1, 1].imshow(
        positive_diff,
        cmap=colormap,
        alpha=0.7,  # Semi-transparent overlay
        vmin=0,
        vmax=positive_diff.max(),
    )
    axes[1, 1].set_title(
        "Overlay: Hidden Features Revealed\n(Brighter = More Enhancement)",
        fontsize=11,
    )
    axes[1, 1].axis("off")
    cbar4 = plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
    cbar4.set_label("Enhancement Strength", fontsize=10)

    # Add statistics annotation
    stats_text = (
        f"Enhancement Statistics:\n"
        f"  Mean Δ: {np.mean(difference):.4f}\n"
        f"  Max Δ:  {np.max(difference):.4f}\n"
        f"  Std Δ:  {np.std(difference):.4f}\n"
        f"  Pixels enhanced: {np.sum(difference > 0.01) / difference.size * 100:.1f}%"
    )
    fig.text(
        0.02,
        0.02,
        stats_text,
        fontsize=9,
        fontfamily="monospace",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9),
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.96])  # Leave room for stats

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"   [SAVED] {save_path}")

    return fig


def plot_crater_detection_heatmap(
    original: np.ndarray,
    enhanced: np.ndarray,
    threshold: float = 0.05,
    save_path: Optional[str] = None,
):
    """
    Specialized heatmap for detecting revealed craters/features.

    Uses edge detection on the difference to highlight structural
    features (crater rims, boulders) that were hidden in shadows.

    Args:
        original: Original normalized image
        enhanced: AI-enhanced image
        threshold: Minimum enhancement to consider significant
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    from scipy import ndimage

    # Calculate difference
    difference = enhanced - original

    # Edge detection on difference to find structural features
    # Sobel filter highlights crater rims and other edges
    sobel_x = ndimage.sobel(difference, axis=1)
    sobel_y = ndimage.sobel(difference, axis=0)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Threshold to find significant enhancements
    significant_mask = difference > threshold

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=150)
    fig.suptitle(
        "Crater & Feature Detection in Permanently Shadowed Regions",
        fontsize=14,
        fontweight="bold",
    )

    # Row 1: Basic comparison
    axes[0, 0].imshow(original, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Original (PSR)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(enhanced, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("AI Enhanced")
    axes[0, 1].axis("off")

    im_diff = axes[0, 2].imshow(difference, cmap="magma")
    axes[0, 2].set_title("Enhancement Map")
    axes[0, 2].axis("off")
    plt.colorbar(im_diff, ax=axes[0, 2], fraction=0.046)

    # Row 2: Feature detection
    im_edge = axes[1, 0].imshow(edge_magnitude, cmap="hot")
    axes[1, 0].set_title("Edge Detection on Δ\n(Crater Rims & Features)")
    axes[1, 0].axis("off")
    plt.colorbar(im_edge, ax=axes[1, 0], fraction=0.046)

    axes[1, 1].imshow(significant_mask, cmap="Greens")
    axes[1, 1].set_title(f"Significant Enhancement\n(Δ > {threshold})")
    axes[1, 1].axis("off")

    # Combined visualization: enhanced image with detected features overlay
    axes[1, 2].imshow(enhanced, cmap="gray", vmin=0, vmax=1)
    # Overlay edges in color
    edge_overlay = np.ma.masked_where(
        edge_magnitude < np.percentile(edge_magnitude, 90), edge_magnitude
    )
    axes[1, 2].imshow(edge_overlay, cmap="spring", alpha=0.8)
    axes[1, 2].set_title("Detected Features Overlay\n(Yellow = Strong Edges)")
    axes[1, 2].axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"   [SAVED] {save_path}")

    return fig


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def print_banner():
    """Print project banner."""
    print(
        """
================================================================================
                    LUNAR SHADOW ENHANCEMENT PIPELINE                           
================================================================================

      Chandrayaan-2 OHRC Data x Zero-DCE Deep Learning                        
      Recovering Permanently Shadowed Region (PSR) Details                    
                                                                              
      ISRO Internship Application Project                                     

================================================================================
    """
    )


def main():
    """Run the complete shadow enhancement pipeline."""

    print_banner()

    # =========================================================================
    # STEP 1: LOAD DATA
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Loading Chandrayaan-2 OHRC Data")
    print("=" * 70)

    try:
        loader = OHRCDataLoader(CONFIG["data_path"])
        stats = loader.get_statistics()
        print(
            f"\n   Statistics: min={stats['min']:.0f}, max={stats['max']:.0f}, "
            f"mean={stats['mean']:.1f}, std={stats['std']:.1f}"
        )
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("   Please ensure the OHRC data is in the correct location.")
        return

    # =========================================================================
    # STEP 2: EXTRACT TILE
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Extracting Shadow Region Tile")
    print("=" * 70)

    # Use center of image if not specified
    tile_x = CONFIG["tile_x"] or loader.shape[1] // 2
    tile_y = CONFIG["tile_y"] or loader.shape[0] // 2
    tile_size = CONFIG["tile_size"]

    print(f"   Location: ({tile_x}, {tile_y})")
    print(f"   Size: {tile_size}×{tile_size} pixels")

    tile = loader.get_tile(tile_x, tile_y, tile_size)
    print(f"   Tile range: [{tile.min()}, {tile.max()}]")
    print(f"   Tile mean: {tile.mean():.1f}")

    # =========================================================================
    # STEP 3: INITIALIZE ENHANCER
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Initializing Zero-DCE Model")
    print("=" * 70)

    # Update config with detected bit-depth
    CONFIG["bit_depth"] = loader.bit_depth

    enhancer = ShadowEnhancer(CONFIG)

    # =========================================================================
    # STEP 4: TRAIN MODEL (OPTIONAL)
    # =========================================================================
    history = None
    if CONFIG["enable_training"]:
        print("\n" + "=" * 70)
        print("STEP 4: Unsupervised Training")
        print("=" * 70)

        history = enhancer.train(
            tile=tile, epochs=CONFIG["epochs"], lr=CONFIG["learning_rate"]
        )

    # =========================================================================
    # STEP 5: ENHANCE & COMPARE
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Shadow Enhancement")
    print("=" * 70)

    original, gamma_corrected, ai_enhanced = enhancer.compare(tile, CONFIG["gamma"])

    # Metrics
    print(f"\n   Enhancement Results:")
    print(f"   Original mean:      {np.mean(original):.4f}")
    print(f"   Gamma corrected:    {np.mean(gamma_corrected):.4f}")
    print(f"   AI enhanced:        {np.mean(ai_enhanced):.4f}")

    boost = (np.mean(ai_enhanced) / np.mean(original)) if np.mean(original) > 0 else 0
    print(f"\n   Brightness boost: {boost:.1f}x")

    # =========================================================================
    # STEP 6: VISUALIZATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Visualization")
    print("=" * 70)

    # Create output directory
    if CONFIG["save_results"]:
        output_dir = Path(CONFIG["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"   Output directory: {output_dir}")

    # Main comparison
    save_comparison = (
        str(output_dir / "comparison.png") if CONFIG["save_results"] else None
    )
    plot_comparison(
        original, gamma_corrected, ai_enhanced, CONFIG["gamma"], save_comparison
    )

    # Histograms
    save_hist = str(output_dir / "histograms.png") if CONFIG["save_results"] else None
    plot_histograms(original, ai_enhanced, save_hist)

    # Training curves
    if history and CONFIG["save_results"]:
        save_train = str(output_dir / "training_curves.png")
        plot_training_curves(history, save_train)

    # Enhancement heatmap (shows where AI made changes)
    save_heatmap = (
        str(output_dir / "enhancement_heatmap.png") if CONFIG["save_results"] else None
    )
    plot_enhancement_heatmap(
        original, ai_enhanced, colormap="magma", save_path=save_heatmap
    )

    # Crater detection heatmap
    save_crater = (
        str(output_dir / "crater_detection.png") if CONFIG["save_results"] else None
    )
    plot_crater_detection_heatmap(
        original, ai_enhanced, threshold=0.02, save_path=save_crater
    )

    # Save model checkpoint
    if CONFIG["save_results"]:
        checkpoint_path = output_dir / "zerodce_checkpoint.pth"
        torch.save(enhancer.model.state_dict(), checkpoint_path)
        print(f"   [SAVED] {checkpoint_path}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    if CONFIG["save_results"]:
        print(f"\nResults saved to: {output_dir}/")
        print("   - comparison.png         : Side-by-side enhancement comparison")
        print("   - histograms.png         : Intensity distribution analysis")
        print("   - enhancement_heatmap.png: Where AI revealed hidden features")
        print("   - crater_detection.png   : Detected craters & structural features")
        if history:
            print("   - training_curves.png    : Training loss visualization")
        print("   - zerodce_checkpoint.pth : Trained model weights")

    print("\nKey Findings:")
    print(f"   - Original shadow regions: mean intensity {np.mean(original):.4f}")
    print(
        f"   - Gamma correction (gamma={CONFIG['gamma']}): {np.mean(gamma_corrected):.4f}"
    )
    print(f"   - Zero-DCE AI enhancement: {np.mean(ai_enhanced):.4f}")
    print(f"   - Brightness improvement: {boost:.1f}x over original")

    # Show plots
    if CONFIG["show_plots"]:
        print("\nDisplaying figures...")
        plt.show()

    return {
        "original": original,
        "gamma_corrected": gamma_corrected,
        "ai_enhanced": ai_enhanced,
        "history": history,
    }


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    results = main()
