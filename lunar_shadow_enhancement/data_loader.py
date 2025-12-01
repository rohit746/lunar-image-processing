"""
Data Loader Module for Chandrayaan-2 OHRC PDS4 Imagery
======================================================

This module provides memory-efficient loading of large PDS4 formatted lunar
imagery using pds4_tools. Implements lazy loading and tile-based access to
handle ~1GB images without loading entire arrays into RAM.

Key Features:
- Lazy loading via pds4_tools for metadata access without full load
- Tile-based access (get_tile) for memory-efficient processing
- Automatic detection of image dimensions and bit-depth
- Shadow region detection utilities

References:
- PDS4 Standard: https://pds.nasa.gov/datastandards/pds4/
- OHRC Instrument: Chandrayaan-2 Orbiter High Resolution Camera
"""

from pathlib import Path
from typing import Tuple, Optional, Union
import numpy as np

import pds4_tools


class OHRCDataLoader:
    """
    Memory-efficient data loader for Chandrayaan-2 OHRC PDS4 imagery.

    This class provides on-demand access to large lunar images through
    tile-based loading, avoiding RAM overflow for ~1GB datasets.

    Attributes:
        xml_path (Path): Path to PDS4 XML label file
        structure: pds4_tools structure object (lazy loaded)
        shape (tuple): Image dimensions (height, width)
        dtype: NumPy data type of the image
        bit_depth (int): Detected bit-depth (10-bit or 12-bit)

    Example:
        >>> loader = OHRCDataLoader("path/to/image.xml")
        >>> print(f"Image shape: {loader.shape}")
        >>> tile = loader.get_tile(x=1000, y=2000, size=512)
    """

    def __init__(self, xml_path: Union[str, Path], lazy_load: bool = True):
        """
        Initialize the OHRC data loader.

        Args:
            xml_path: Path to the PDS4 XML label file
            lazy_load: If True, defer full data loading until accessed
        """
        self.xml_path = Path(xml_path)

        if not self.xml_path.exists():
            raise FileNotFoundError(f"PDS4 XML file not found: {self.xml_path}")

        if not self.xml_path.suffix.lower() == ".xml":
            raise ValueError(f"Expected .xml file, got: {self.xml_path.suffix}")

        print(f"[INFO] Loading PDS4 metadata from: {self.xml_path.name}")

        # Load structure with lazy_load for memory efficiency
        self._structure = pds4_tools.read(str(self.xml_path), lazy_load=lazy_load)

        # Cache image array reference (lazy - not fully loaded yet)
        self._data = self._structure[0].data

        # Extract metadata
        self._shape = self._data.shape
        self._dtype = self._data.dtype

        # Detect bit-depth from data type and statistics
        self._bit_depth = self._detect_bit_depth()

        print(f"[OK] Image loaded: {self._shape[0]:,} x {self._shape[1]:,} pixels")
        print(f"   Data type: {self._dtype}, Bit-depth: {self._bit_depth}-bit")

    @property
    def shape(self) -> Tuple[int, int]:
        """Return image dimensions (height, width)."""
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        """Return NumPy data type of the image."""
        return self._dtype

    @property
    def bit_depth(self) -> int:
        """Return detected bit-depth (10 or 12)."""
        return self._bit_depth

    @property
    def max_value(self) -> int:
        """Return maximum possible value based on bit-depth."""
        return (2**self._bit_depth) - 1

    def _detect_bit_depth(self) -> int:
        """
        Detect the bit-depth of the image data.

        OHRC typically uses 10-bit or 12-bit encoding. We detect this
        by checking the data type and sampling pixel values.

        Returns:
            int: Detected bit-depth (10 or 12)
        """
        # Sample a small region to determine actual data range
        sample_size = min(1000, min(self._shape))
        sample = self._data[:sample_size, :sample_size]
        max_val = np.max(sample)

        # Determine bit-depth based on maximum value
        if max_val > 1023:  # > 10-bit max
            return 12
        elif max_val > 255:  # > 8-bit max
            return 10
        else:
            # Default to 12-bit for OHRC data
            return 12

    def get_tile(
        self, x: int, y: int, size: int = 512, allow_partial: bool = True
    ) -> np.ndarray:
        """
        Extract a square tile from the image at the specified location.

        This method provides memory-efficient access to specific regions
        of the large image without loading the entire array.

        Args:
            x: Horizontal position (column) of tile center
            y: Vertical position (row) of tile center
            size: Size of the square tile (default 512x512)
            allow_partial: If True, allow tiles at image edges that may be smaller

        Returns:
            np.ndarray: Extracted tile as numpy array

        Raises:
            ValueError: If coordinates are out of bounds and allow_partial=False

        Example:
            >>> tile = loader.get_tile(x=1000, y=2000, size=512)
            >>> print(tile.shape)  # (512, 512)
        """
        height, width = self._shape
        half_size = size // 2

        # Calculate bounding box
        y_start = max(0, y - half_size)
        y_end = min(height, y + half_size)
        x_start = max(0, x - half_size)
        x_end = min(width, x + half_size)

        # Validate bounds
        if not allow_partial:
            if (y_end - y_start) < size or (x_end - x_start) < size:
                raise ValueError(
                    f"Tile at ({x}, {y}) with size {size} exceeds image bounds. "
                    f"Image shape: {self._shape}"
                )

        # Extract tile from memory-mapped data
        tile = self._data[y_start:y_end, x_start:x_end].copy()

        return tile

    def get_tile_by_index(
        self, row_idx: int, col_idx: int, tile_size: int = 512, overlap: int = 0
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Get a tile by grid index position.

        Divides the image into a grid of tiles and returns the tile
        at the specified grid position.

        Args:
            row_idx: Row index in the tile grid (0-indexed)
            col_idx: Column index in the tile grid (0-indexed)
            tile_size: Size of each tile
            overlap: Overlap between adjacent tiles

        Returns:
            Tuple of (tile_array, (y_start, y_end, x_start, x_end))
        """
        stride = tile_size - overlap

        y_start = row_idx * stride
        x_start = col_idx * stride
        y_end = min(y_start + tile_size, self._shape[0])
        x_end = min(x_start + tile_size, self._shape[1])

        tile = self._data[y_start:y_end, x_start:x_end].copy()

        return tile, (y_start, y_end, x_start, x_end)

    def get_grid_dimensions(
        self, tile_size: int = 512, overlap: int = 0
    ) -> Tuple[int, int]:
        """
        Calculate the number of tiles in each dimension.

        Args:
            tile_size: Size of each tile
            overlap: Overlap between adjacent tiles

        Returns:
            Tuple of (num_rows, num_cols)
        """
        stride = tile_size - overlap
        num_rows = int(np.ceil(self._shape[0] / stride))
        num_cols = int(np.ceil(self._shape[1] / stride))
        return num_rows, num_cols

    def find_dark_regions(
        self,
        threshold: float = 0.05,
        min_region_size: int = 100,
        sample_stride: int = 100,
    ) -> list:
        """
        Find regions in the image that likely contain shadows (PSRs).

        Samples the image at regular intervals to identify dark regions
        that may benefit from shadow enhancement.

        Args:
            threshold: Fraction of max_value below which pixels are "dark"
            min_region_size: Minimum number of dark pixels to qualify
            sample_stride: Stride for sampling (larger = faster but coarser)

        Returns:
            List of (center_y, center_x, darkness_score) tuples
        """
        dark_threshold = self.max_value * threshold
        dark_regions = []

        height, width = self._shape
        tile_size = 512

        print(
            f"[INFO] Scanning for shadow regions (threshold: {threshold:.1%} of max)..."
        )

        for y in range(0, height - tile_size, sample_stride):
            for x in range(0, width - tile_size, sample_stride):
                tile = self.get_tile(x + tile_size // 2, y + tile_size // 2, tile_size)

                # Calculate darkness metrics
                dark_pixels = np.sum(tile < dark_threshold)
                total_pixels = tile.size
                darkness_ratio = dark_pixels / total_pixels
                mean_intensity = np.mean(tile)

                # Check if region qualifies as a shadow candidate
                if dark_pixels >= min_region_size and darkness_ratio > 0.3:
                    dark_regions.append(
                        {
                            "center_y": y + tile_size // 2,
                            "center_x": x + tile_size // 2,
                            "darkness_ratio": darkness_ratio,
                            "mean_intensity": mean_intensity,
                            "dark_pixel_count": dark_pixels,
                        }
                    )

        # Sort by darkness ratio (most dark first)
        dark_regions.sort(key=lambda r: r["darkness_ratio"], reverse=True)

        print(f"   Found {len(dark_regions)} potential shadow regions")

        return dark_regions

    def get_statistics(self, sample_size: Optional[int] = None) -> dict:
        """
        Compute statistics of the image data.

        Args:
            sample_size: If provided, compute stats on a random sample

        Returns:
            Dictionary containing min, max, mean, std, percentiles
        """
        if sample_size:
            # Random sampling for large images
            y_samples = np.random.randint(0, self._shape[0], sample_size)
            x_samples = np.random.randint(0, self._shape[1], sample_size)
            data = self._data[y_samples, x_samples]
        else:
            data = self._data

        return {
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "percentile_1": float(np.percentile(data, 1)),
            "percentile_99": float(np.percentile(data, 99)),
            "bit_depth": self._bit_depth,
            "shape": self._shape,
        }

    def __repr__(self) -> str:
        return (
            f"OHRCDataLoader(\n"
            f"  path='{self.xml_path.name}',\n"
            f"  shape={self._shape},\n"
            f"  dtype={self._dtype},\n"
            f"  bit_depth={self._bit_depth}\n"
            f")"
        )


# Utility function for quick loading
def load_ohrc_image(xml_path: Union[str, Path]) -> OHRCDataLoader:
    """
    Convenience function to load an OHRC image.

    Args:
        xml_path: Path to PDS4 XML label file

    Returns:
        Configured OHRCDataLoader instance
    """
    return OHRCDataLoader(xml_path)
