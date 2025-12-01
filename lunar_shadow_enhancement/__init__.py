"""
Lunar Shadow Enhancement Pipeline for Chandrayaan-2 OHRC Data
==============================================================

This package provides a complete pipeline for enhancing Permanently Shadowed
Regions (PSRs) in lunar imagery from Chandrayaan-2 Orbiter High Resolution
Camera (OHRC) using Zero-DCE (Zero-Reference Deep Curve Estimation) deep learning.

Modules:
--------
- data_loader: Memory-efficient loading of large PDS4 imagery
- preprocessing: Normalization and data preparation utilities
- model: Zero-DCE neural network architecture
- inference: Shadow enhancement pipeline
- visualization: Plotting and comparison utilities

Author: Rohit
Project: ISRO Internship Application - Lunar Shadow Enhancement
"""

__version__ = "0.1.0"
__author__ = "Rohit"

from .data_loader import OHRCDataLoader
from .preprocessing import Preprocessor
from .model import ZeroDCE, ZeroDCELoss
from .inference import ShadowEnhancer
from .visualization import Visualizer

__all__ = [
    "OHRCDataLoader",
    "Preprocessor",
    "ZeroDCE",
    "ZeroDCELoss",
    "ShadowEnhancer",
    "Visualizer",
]
