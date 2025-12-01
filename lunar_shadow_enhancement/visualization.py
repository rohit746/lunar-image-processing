"""
Visualization Module for Lunar Shadow Enhancement
==================================================

Provides plotting utilities for comparing original, gamma-corrected,
and AI-enhanced imagery side-by-side.

Key Features:
- Side-by-side comparison plots
- Histogram visualization
- Training loss curves
- Image statistics overlay
"""

from pathlib import Path
from typing import Optional, Tuple, List, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize


class Visualizer:
    """
    Visualization utilities for lunar shadow enhancement.

    Creates publication-quality figures comparing enhancement results.

    Attributes:
        figsize: Default figure size
        dpi: Figure resolution
        style: Matplotlib style

    Example:
        >>> viz = Visualizer()
        >>> viz.plot_comparison(original, gamma, enhanced)
        >>> viz.plot_histograms(original, enhanced)
    """

    def __init__(
        self, figsize: Tuple[int, int] = (15, 5), dpi: int = 150, style: str = "default"
    ):
        """
        Initialize visualizer.

        Args:
            figsize: Default figure size (width, height)
            dpi: Figure resolution in dots per inch
            style: Matplotlib style name
        """
        self.figsize = figsize
        self.dpi = dpi

        # Set style
        if style != "default":
            plt.style.use(style)

        # Configure defaults for scientific visualization
        plt.rcParams.update(
            {
                "font.size": 10,
                "axes.titlesize": 12,
                "axes.labelsize": 10,
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "savefig.facecolor": "white",
                "savefig.dpi": dpi,
            }
        )

    def plot_comparison(
        self,
        original: np.ndarray,
        gamma_corrected: np.ndarray,
        ai_enhanced: np.ndarray,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show_stats: bool = True,
        gamma_value: float = 2.2,
    ) -> plt.Figure:
        """
        Create side-by-side comparison of enhancement methods.

        Args:
            original: Original normalized image [0, 1]
            gamma_corrected: Gamma corrected image [0, 1]
            ai_enhanced: AI enhanced image [0, 1]
            title: Optional figure title
            save_path: Path to save figure
            show_stats: Overlay image statistics
            gamma_value: Gamma value used (for label)

        Returns:
            Matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 3, figsize=self.figsize, dpi=self.dpi)

        # Main title
        if title:
            fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

        # Plot images
        images = [
            (original, f"Original\n(Normalized)", "viridis"),
            (gamma_corrected, f"Gamma Corrected\n(γ = {gamma_value})", "viridis"),
            (ai_enhanced, "Zero-DCE Enhanced\n(AI)", "viridis"),
        ]

        for ax, (img, label, cmap) in zip(axes, images):
            im = ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            ax.set_title(label, fontsize=11)
            ax.axis("off")

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Intensity", fontsize=9)

            # Show statistics if requested
            if show_stats:
                stats_text = (
                    f"μ={np.mean(img):.3f}\n"
                    f"σ={np.std(img):.3f}\n"
                    f"min={np.min(img):.3f}\n"
                    f"max={np.max(img):.3f}"
                )
                ax.text(
                    0.02,
                    0.98,
                    stats_text,
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment="top",
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

        plt.tight_layout()

        # Save if requested
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=self.dpi)
            print(f"[SAVED] Figure saved: {save_path}")

        return fig

    def plot_histograms(
        self,
        original: np.ndarray,
        enhanced: np.ndarray,
        bins: int = 100,
        title: str = "Intensity Distribution Comparison",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot histograms comparing intensity distributions.

        Args:
            original: Original image
            enhanced: Enhanced image
            bins: Number of histogram bins
            title: Figure title
            save_path: Path to save figure

        Returns:
            Matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=self.dpi)

        fig.suptitle(title, fontsize=14, fontweight="bold")

        # Original histogram
        axes[0].hist(
            original.ravel(),
            bins=bins,
            color="steelblue",
            alpha=0.7,
            edgecolor="none",
            density=True,
        )
        axes[0].set_title("Original Image")
        axes[0].set_xlabel("Pixel Intensity")
        axes[0].set_ylabel("Density")
        axes[0].set_xlim(0, 1)
        axes[0].axvline(
            np.mean(original),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(original):.3f}",
        )
        axes[0].legend()

        # Enhanced histogram
        axes[1].hist(
            enhanced.ravel(),
            bins=bins,
            color="coral",
            alpha=0.7,
            edgecolor="none",
            density=True,
        )
        axes[1].set_title("Enhanced Image")
        axes[1].set_xlabel("Pixel Intensity")
        axes[1].set_ylabel("Density")
        axes[1].set_xlim(0, 1)
        axes[1].axvline(
            np.mean(enhanced),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(enhanced):.3f}",
        )
        axes[1].legend()

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=self.dpi)
            print(f"[SAVED] Figure saved: {save_path}")

        return fig

    def plot_training_curves(
        self,
        history: List[dict],
        title: str = "Zero-DCE Training Progress",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot training loss curves.

        Args:
            history: List of loss component dictionaries
            title: Figure title
            save_path: Path to save figure

        Returns:
            Matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=self.dpi)
        fig.suptitle(title, fontsize=14, fontweight="bold")

        epochs = range(1, len(history) + 1)

        # Total loss
        axes[0, 0].plot(
            epochs, [h["total"] for h in history], color="steelblue", linewidth=2
        )
        axes[0, 0].set_title("Total Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].grid(True, alpha=0.3)

        # Exposure loss
        axes[0, 1].plot(
            epochs, [h["exposure"] for h in history], color="coral", linewidth=2
        )
        axes[0, 1].set_title("Exposure Control Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].grid(True, alpha=0.3)

        # Spatial loss
        axes[1, 0].plot(
            epochs, [h["spatial"] for h in history], color="seagreen", linewidth=2
        )
        axes[1, 0].set_title("Spatial Consistency Loss")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].grid(True, alpha=0.3)

        # Color & smoothness
        axes[1, 1].plot(
            epochs, [h["color"] for h in history], label="Color Constancy", linewidth=2
        )
        axes[1, 1].plot(
            epochs, [h["smoothness"] for h in history], label="Smoothness", linewidth=2
        )
        axes[1, 1].set_title("Color & Smoothness Losses")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Loss")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=self.dpi)
            print(f"[SAVED] Figure saved: {save_path}")

        return fig

    def plot_tile_grid(
        self,
        tiles: List[np.ndarray],
        ncols: int = 4,
        title: str = "Tile Overview",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot a grid of tiles.

        Args:
            tiles: List of tile arrays
            ncols: Number of columns in grid
            title: Figure title
            save_path: Path to save figure

        Returns:
            Matplotlib Figure object
        """
        n_tiles = len(tiles)
        nrows = (n_tiles + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(3 * ncols, 3 * nrows), dpi=self.dpi
        )

        fig.suptitle(title, fontsize=14, fontweight="bold")

        axes = np.atleast_2d(axes)

        for idx, tile in enumerate(tiles):
            row = idx // ncols
            col = idx % ncols

            axes[row, col].imshow(tile, cmap="gray", vmin=0, vmax=1)
            axes[row, col].set_title(f"Tile {idx + 1}", fontsize=9)
            axes[row, col].axis("off")

        # Hide empty subplots
        for idx in range(n_tiles, nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            axes[row, col].axis("off")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=self.dpi)
            print(f"[SAVED] Figure saved: {save_path}")

        return fig

    def plot_before_after(
        self,
        before: np.ndarray,
        after: np.ndarray,
        before_label: str = "Before",
        after_label: str = "After",
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Simple before/after comparison.

        Args:
            before: Before image
            after: After image
            before_label: Label for before image
            after_label: Label for after image
            title: Optional figure title
            save_path: Path to save figure

        Returns:
            Matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=self.dpi)

        if title:
            fig.suptitle(title, fontsize=14, fontweight="bold")

        axes[0].imshow(before, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title(before_label)
        axes[0].axis("off")

        axes[1].imshow(after, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title(after_label)
        axes[1].axis("off")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=self.dpi)
            print(f"[SAVED] Figure saved: {save_path}")

        return fig

    def plot_curve_maps(
        self,
        curves: np.ndarray,
        title: str = "Enhancement Curve Parameters",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Visualize the curve parameter maps from Zero-DCE.

        Args:
            curves: Curve parameter array (n_curves*3, H, W)
            title: Figure title
            save_path: Path to save figure

        Returns:
            Matplotlib Figure object
        """
        n_channels = curves.shape[0]
        n_curves = n_channels // 3

        fig, axes = plt.subplots(n_curves, 3, figsize=(12, 2 * n_curves), dpi=self.dpi)

        fig.suptitle(title, fontsize=14, fontweight="bold")

        for i in range(n_curves):
            for c in range(3):
                ax = axes[i, c] if n_curves > 1 else axes[c]
                im = ax.imshow(curves[i * 3 + c], cmap="RdBu", vmin=-1, vmax=1)
                ax.set_title(f'Curve {i+1} - Ch {["R","G","B"][c]}', fontsize=9)
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=self.dpi)
            print(f"[SAVED] Figure saved: {save_path}")

        return fig

    @staticmethod
    def show():
        """Display all pending figures."""
        plt.show()


def create_visualizer(figsize: Tuple[int, int] = (15, 5), dpi: int = 150) -> Visualizer:
    """
    Factory function to create a Visualizer.

    Args:
        figsize: Default figure size
        dpi: Figure resolution

    Returns:
        Configured Visualizer instance
    """
    return Visualizer(figsize=figsize, dpi=dpi)
