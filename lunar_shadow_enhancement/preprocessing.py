"""
Preprocessing Module for Lunar Shadow Enhancement
==================================================

This module provides preprocessing utilities for preparing raw OHRC imagery
for deep learning inference. Handles bit-depth normalization, contrast
stretching, and tensor conversion for PyTorch models.

Key Features:
- Raw integer to float32 conversion
- Bit-depth aware normalization (10-bit/12-bit to [0, 1])
- Histogram-based adaptive normalization
- Contrast Limited Adaptive Histogram Equalization (CLAHE) option
- PyTorch tensor conversion utilities
"""

from typing import Tuple, Optional, Union
import numpy as np
import torch


class Preprocessor:
    """
    Preprocessing pipeline for OHRC lunar imagery.

    Converts raw integer image data to normalized float tensors suitable
    for deep learning models, with special handling for low-light/shadow regions.

    Attributes:
        bit_depth (int): Source bit-depth (10 or 12)
        max_value (int): Maximum possible raw value
        device (str): PyTorch device ('cuda' or 'cpu')

    Example:
        >>> preprocessor = Preprocessor(bit_depth=12)
        >>> normalized = preprocessor.normalize(raw_tile)
        >>> tensor = preprocessor.to_tensor(normalized)
    """

    def __init__(self, bit_depth: int = 12, device: str = "auto"):
        """
        Initialize the preprocessor.

        Args:
            bit_depth: Bit-depth of source data (10 or 12)
            device: PyTorch device ('cuda', 'cpu', or 'auto')
        """
        if bit_depth not in [8, 10, 12, 14, 16]:
            raise ValueError(f"Unsupported bit-depth: {bit_depth}")

        self.bit_depth = bit_depth
        self.max_value = (2**bit_depth) - 1

        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(
            f"🔧 Preprocessor initialized: {bit_depth}-bit → [0,1], device={self.device}"
        )

    def normalize(self, data: np.ndarray, method: str = "linear") -> np.ndarray:
        """
        Normalize raw integer data to [0, 1] float32 range.

        Args:
            data: Raw integer image data
            method: Normalization method:
                - 'linear': Simple division by max_value
                - 'minmax': Min-max normalization per tile
                - 'percentile': Robust normalization using percentiles
                - 'adaptive': Shadow-aware adaptive normalization

        Returns:
            Normalized float32 array in [0, 1] range
        """
        # Convert to float32
        normalized = data.astype(np.float32)

        if method == "linear":
            # Standard linear normalization from bit-depth
            normalized = normalized / self.max_value

        elif method == "minmax":
            # Min-max normalization (per-tile)
            min_val = np.min(normalized)
            max_val = np.max(normalized)
            if max_val > min_val:
                normalized = (normalized - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(normalized)

        elif method == "percentile":
            # Robust percentile-based normalization
            p1 = np.percentile(normalized, 1)
            p99 = np.percentile(normalized, 99)
            normalized = np.clip((normalized - p1) / (p99 - p1 + 1e-8), 0, 1)

        elif method == "adaptive":
            # Shadow-aware adaptive normalization
            normalized = self._adaptive_normalize(normalized)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return normalized.astype(np.float32)

    def _adaptive_normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Adaptive normalization that preserves shadow detail.

        Uses a two-stage approach:
        1. Log transformation to expand shadow dynamic range
        2. Percentile-based clipping for robust normalization
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1.0

        # Log transform to expand shadow range
        log_data = np.log1p(data + epsilon)

        # Normalize log-transformed data
        p1 = np.percentile(log_data, 0.5)
        p99 = np.percentile(log_data, 99.5)

        normalized = (log_data - p1) / (p99 - p1 + 1e-8)
        normalized = np.clip(normalized, 0, 1)

        return normalized

    def apply_gamma(self, data: np.ndarray, gamma: float = 2.2) -> np.ndarray:
        """
        Apply gamma correction for shadow enhancement.

        The formula pixel^(1/gamma) brightens dark regions.
        For gamma=2.2, this approximates sRGB-like curve.

        Args:
            data: Normalized float array in [0, 1]
            gamma: Gamma value (default 2.2 for standard correction)

        Returns:
            Gamma-corrected array
        """
        # Ensure data is in [0, 1] range
        data = np.clip(data, 0, 1)

        # Apply gamma correction: output = input^(1/gamma)
        corrected = np.power(data, 1.0 / gamma)

        return corrected.astype(np.float32)

    def to_tensor(
        self, data: np.ndarray, add_batch_dim: bool = True, add_channel_dim: bool = True
    ) -> torch.Tensor:
        """
        Convert numpy array to PyTorch tensor.

        Args:
            data: Normalized float32 numpy array (H, W)
            add_batch_dim: Add batch dimension (B, C, H, W)
            add_channel_dim: Add channel dimension for grayscale

        Returns:
            PyTorch tensor on configured device
        """
        tensor = torch.from_numpy(data)

        # Add channel dimension for grayscale: (H, W) → (1, H, W)
        if add_channel_dim and tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)

        # Repeat to 3 channels for Zero-DCE compatibility: (1, H, W) → (3, H, W)
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)

        # Add batch dimension: (C, H, W) → (1, C, H, W)
        if add_batch_dim:
            tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    def from_tensor(
        self,
        tensor: torch.Tensor,
        remove_batch_dim: bool = True,
        to_grayscale: bool = True,
    ) -> np.ndarray:
        """
        Convert PyTorch tensor back to numpy array.

        Args:
            tensor: PyTorch tensor (B, C, H, W) or (C, H, W)
            remove_batch_dim: Remove batch dimension
            to_grayscale: Convert RGB back to grayscale

        Returns:
            Numpy array in [0, 1] range
        """
        # Move to CPU and detach from computation graph
        array = tensor.detach().cpu().numpy()

        # Remove batch dimension
        if remove_batch_dim and array.ndim == 4:
            array = array[0]  # (C, H, W)

        # Convert to grayscale by averaging channels
        if to_grayscale and array.ndim == 3 and array.shape[0] == 3:
            array = np.mean(array, axis=0)  # (H, W)

        return array.astype(np.float32)

    def denormalize(
        self, data: np.ndarray, target_bit_depth: Optional[int] = None
    ) -> np.ndarray:
        """
        Convert normalized [0, 1] data back to integer range.

        Args:
            data: Normalized float array in [0, 1]
            target_bit_depth: Output bit-depth (defaults to source)

        Returns:
            Integer array in original range
        """
        if target_bit_depth is None:
            target_bit_depth = self.bit_depth

        max_val = (2**target_bit_depth) - 1

        # Scale and clip
        result = np.clip(data * max_val, 0, max_val)

        # Choose appropriate integer type
        if target_bit_depth <= 8:
            return result.astype(np.uint8)
        elif target_bit_depth <= 16:
            return result.astype(np.uint16)
        else:
            return result.astype(np.uint32)

    def prepare_batch(
        self, tiles: list, normalize_method: str = "linear"
    ) -> torch.Tensor:
        """
        Prepare a batch of tiles for model inference.

        Args:
            tiles: List of raw numpy arrays
            normalize_method: Normalization method to apply

        Returns:
            Batched tensor (B, 3, H, W)
        """
        processed = []

        for tile in tiles:
            # Normalize
            normalized = self.normalize(tile, method=normalize_method)
            # Convert to tensor (without batch dim)
            tensor = self.to_tensor(normalized, add_batch_dim=False)
            processed.append(tensor)

        # Stack into batch
        batch = torch.stack(processed, dim=0)

        return batch


def create_preprocessor(bit_depth: int = 12, device: str = "auto") -> Preprocessor:
    """
    Factory function to create a preprocessor instance.

    Args:
        bit_depth: Source data bit-depth
        device: Target device for tensors

    Returns:
        Configured Preprocessor instance
    """
    return Preprocessor(bit_depth=bit_depth, device=device)
