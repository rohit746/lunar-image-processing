"""
Inference Pipeline for Lunar Shadow Enhancement
================================================

This module provides the main inference pipeline combining all components:
data loading, preprocessing, model inference, and result assembly.

Key Features:
- End-to-end shadow enhancement workflow
- Tile-based processing for large images
- Training and fine-tuning support
- Checkpoint saving/loading
"""

from pathlib import Path
from typing import Optional, Union, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from .data_loader import OHRCDataLoader
from .preprocessing import Preprocessor
from .model import ZeroDCE, ZeroDCELoss, create_model


class ShadowEnhancer:
    """
    Complete pipeline for lunar shadow enhancement.

    Orchestrates the data loading, preprocessing, model inference,
    and post-processing stages for enhancing PSR regions.

    Attributes:
        model: Zero-DCE neural network
        preprocessor: Data preprocessing utility
        device: PyTorch device (cuda/cpu)

    Example:
        >>> enhancer = ShadowEnhancer()
        >>> enhanced = enhancer.enhance_tile(dark_tile)
        >>> enhancer.train_on_tile(training_tile, epochs=100)
    """

    def __init__(
        self,
        model_variant: str = "standard",
        n_curves: int = 8,
        device: str = "auto",
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initialize the shadow enhancement pipeline.

        Args:
            model_variant: Model variant ('standard' or 'lightweight')
            n_curves: Number of curve iterations
            device: Target device ('cuda', 'cpu', or 'auto')
            checkpoint_path: Path to load pretrained weights
        """
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[INFO] Initializing Shadow Enhancer on {self.device.upper()}")

        # Create model
        self.model = create_model(variant=model_variant, n_curves=n_curves)
        self.model = self.model.to(self.device)

        # Create loss function for training
        self.criterion = ZeroDCELoss()
        self.criterion = self.criterion.to(self.device)

        # Create preprocessor (will be configured per-image)
        self.preprocessor = None

        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

        # Set model to eval mode by default
        self.model.eval()

    def configure_preprocessor(self, bit_depth: int = 12):
        """
        Configure the preprocessor for a specific image.

        Args:
            bit_depth: Bit-depth of the source data
        """
        self.preprocessor = Preprocessor(bit_depth=bit_depth, device=self.device)

    def enhance_tile(
        self, tile: np.ndarray, normalize_method: str = "linear"
    ) -> np.ndarray:
        """
        Enhance a single tile using Zero-DCE.

        Args:
            tile: Raw image tile (H, W) as numpy array
            normalize_method: Preprocessing normalization method

        Returns:
            Enhanced tile as numpy array in [0, 1]
        """
        if self.preprocessor is None:
            raise RuntimeError(
                "Preprocessor not configured. Call configure_preprocessor() first."
            )

        # Preprocess
        normalized = self.preprocessor.normalize(tile, method=normalize_method)
        tensor = self.preprocessor.to_tensor(normalized)

        # Inference
        self.model.eval()
        with torch.no_grad():
            enhanced, curves = self.model(tensor)

        # Convert back to numpy
        result = self.preprocessor.from_tensor(enhanced)

        return result

    def enhance_with_comparison(
        self, tile: np.ndarray, gamma: float = 2.2, normalize_method: str = "linear"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Enhance tile and provide comparison with gamma correction.

        Args:
            tile: Raw image tile (H, W)
            gamma: Gamma value for comparison correction
            normalize_method: Normalization method

        Returns:
            Tuple of (original_normalized, gamma_corrected, ai_enhanced)
        """
        if self.preprocessor is None:
            raise RuntimeError(
                "Preprocessor not configured. Call configure_preprocessor() first."
            )

        # Normalize original
        original = self.preprocessor.normalize(tile, method=normalize_method)

        # Gamma correction: pixel^(1/gamma)
        gamma_corrected = self.preprocessor.apply_gamma(original, gamma=gamma)

        # AI enhancement
        ai_enhanced = self.enhance_tile(tile, normalize_method=normalize_method)

        return original, gamma_corrected, ai_enhanced

    def train_on_tile(
        self,
        tile: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 1e-4,
        target_exposure: float = 0.6,
        verbose: bool = True,
    ) -> List[dict]:
        """
        Fine-tune the model on a specific tile (unsupervised).

        Zero-DCE can be trained without paired data using its custom
        loss functions that encourage proper enhancement.

        Args:
            tile: Training tile (H, W)
            epochs: Number of training epochs
            learning_rate: Optimizer learning rate
            target_exposure: Target brightness level
            verbose: Print training progress

        Returns:
            List of loss dictionaries per epoch
        """
        if self.preprocessor is None:
            raise RuntimeError(
                "Preprocessor not configured. Call configure_preprocessor() first."
            )

        # Prepare training data
        normalized = self.preprocessor.normalize(tile, method="linear")
        tensor = self.preprocessor.to_tensor(normalized)

        # Configure loss with target exposure
        self.criterion.target_exposure = target_exposure

        # Setup optimizer
        optimizer = Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        self.model.train()
        history = []

        iterator = tqdm(range(epochs), desc="Training") if verbose else range(epochs)

        for epoch in iterator:
            optimizer.zero_grad()

            # Forward pass
            enhanced, curves = self.model(tensor)

            # Compute loss
            loss, components = self.criterion(enhanced, tensor, curves)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Record history
            history.append(components)

            if verbose and (epoch + 1) % 20 == 0:
                tqdm.write(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Loss: {components['total']:.4f} | "
                    f"Exp: {components['exposure']:.4f}"
                )

        self.model.eval()
        print(f"[DONE] Training complete. Final loss: {history[-1]['total']:.4f}")

        return history

    def train_on_tiles(
        self,
        tiles: List[np.ndarray],
        epochs: int = 100,
        learning_rate: float = 1e-4,
        batch_size: int = 4,
        target_exposure: float = 0.6,
        verbose: bool = True,
    ) -> List[dict]:
        """
        Train on multiple tiles for better generalization.

        Args:
            tiles: List of training tiles
            epochs: Number of epochs
            learning_rate: Learning rate
            batch_size: Batch size for training
            target_exposure: Target exposure level
            verbose: Show progress

        Returns:
            Training history
        """
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor not configured.")

        # Prepare all tiles
        tensors = []
        for tile in tiles:
            normalized = self.preprocessor.normalize(tile, method="linear")
            tensor = self.preprocessor.to_tensor(normalized, add_batch_dim=False)
            tensors.append(tensor)

        # Stack into dataset
        dataset = torch.stack(tensors, dim=0)  # (N, 3, H, W)
        n_samples = len(dataset)

        print(f"[INFO] Training on {n_samples} tiles...")

        # Configure
        self.criterion.target_exposure = target_exposure
        optimizer = Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        self.model.train()
        history = []

        for epoch in tqdm(range(epochs), desc="Training", disable=not verbose):
            epoch_losses = []

            # Mini-batch training
            indices = torch.randperm(n_samples)

            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i : i + batch_size]
                batch = dataset[batch_idx].to(self.device)

                optimizer.zero_grad()
                enhanced, curves = self.model(batch)
                loss, components = self.criterion(enhanced, batch, curves)
                loss.backward()
                optimizer.step()

                epoch_losses.append(components)

            # Average epoch loss
            avg_loss = {
                k: np.mean([l[k] for l in epoch_losses]) for k in epoch_losses[0].keys()
            }
            history.append(avg_loss)

        self.model.eval()
        print(f"[DONE] Training complete. Final loss: {history[-1]['total']:.4f}")

        return history

    def process_full_image(
        self,
        loader: OHRCDataLoader,
        tile_size: int = 512,
        overlap: int = 64,
        output_path: Optional[str] = None,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Process entire image tile-by-tile and stitch results.

        Args:
            loader: Configured OHRCDataLoader
            tile_size: Size of processing tiles
            overlap: Overlap between tiles for smooth stitching
            output_path: Optional path to save result
            verbose: Show progress bar

        Returns:
            Full enhanced image as numpy array
        """
        # Configure preprocessor
        self.configure_preprocessor(bit_depth=loader.bit_depth)

        height, width = loader.shape

        # Create output array
        result = np.zeros((height, width), dtype=np.float32)
        weight_map = np.zeros((height, width), dtype=np.float32)

        # Calculate grid dimensions
        num_rows, num_cols = loader.get_grid_dimensions(tile_size, overlap)
        total_tiles = num_rows * num_cols

        print(f"🔄 Processing {total_tiles} tiles ({num_rows}×{num_cols})...")

        iterator = range(total_tiles)
        if verbose:
            iterator = tqdm(iterator, desc="Enhancing")

        for idx in iterator:
            row = idx // num_cols
            col = idx % num_cols

            # Get tile
            tile, (y_start, y_end, x_start, x_end) = loader.get_tile_by_index(
                row, col, tile_size, overlap
            )

            # Enhance
            enhanced = self.enhance_tile(tile)

            # Create blending weights (feather edges)
            tile_h, tile_w = enhanced.shape
            weights = self._create_blend_weights(tile_h, tile_w, overlap)

            # Add to result with weights
            result[y_start:y_end, x_start:x_end] += enhanced * weights
            weight_map[y_start:y_end, x_start:x_end] += weights

        # Normalize by weights
        weight_map = np.maximum(weight_map, 1e-8)  # Avoid division by zero
        result = result / weight_map

        # Save if requested
        if output_path:
            np.save(output_path, result)
            print(f"[SAVED] Result saved to: {output_path}")

        return result

    def _create_blend_weights(self, height: int, width: int, margin: int) -> np.ndarray:
        """Create feathered blend weights for tile stitching."""
        weights = np.ones((height, width), dtype=np.float32)

        # Feather edges
        if margin > 0:
            for i in range(margin):
                factor = (i + 1) / (margin + 1)
                weights[i, :] *= factor
                weights[-(i + 1), :] *= factor
                weights[:, i] *= factor
                weights[:, -(i + 1)] *= factor

        return weights

    def save_checkpoint(self, path: Union[str, Path]):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "model_class": self.model.__class__.__name__,
        }

        torch.save(checkpoint, path)
        print(f"[SAVED] Checkpoint saved: {path}")

    def load_checkpoint(self, path: Union[str, Path]):
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint file
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        print(f"[OK] Loaded checkpoint: {path}")

    def get_model_info(self) -> dict:
        """Return model information."""
        return {
            "class": self.model.__class__.__name__,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "device": self.device,
            "n_curves": self.model.n_curves,
        }


def create_enhancer(
    model_variant: str = "standard",
    device: str = "auto",
    checkpoint: Optional[str] = None,
) -> ShadowEnhancer:
    """
    Factory function to create a ShadowEnhancer.

    Args:
        model_variant: Model variant to use
        device: Target device
        checkpoint: Optional checkpoint path

    Returns:
        Configured ShadowEnhancer instance
    """
    return ShadowEnhancer(
        model_variant=model_variant, device=device, checkpoint_path=checkpoint
    )
