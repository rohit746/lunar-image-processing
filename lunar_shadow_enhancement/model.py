"""
Zero-DCE Model for Lunar Shadow Enhancement
============================================

Implementation of Zero-Reference Deep Curve Estimation (Zero-DCE) for
enhancing Permanently Shadowed Regions (PSRs) in lunar imagery.

Zero-DCE is an unsupervised learning approach that estimates light-enhancement
curves without requiring paired training data. This makes it ideal for lunar
shadow enhancement where ground-truth illuminated images don't exist.

Reference Paper:
    "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement"
    Guo et al., CVPR 2020
    https://arxiv.org/abs/2001.06826

Key Concepts:
- Light-Enhancement Curve (LE-curve): Pixel-wise adjustment curves
- Curve Parameter Maps: Network predicts α parameters for enhancement
- Iterative Enhancement: Curves applied multiple times for stronger effect
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroDCE(nn.Module):
    """
    Zero-Reference Deep Curve Estimation Network.

    A lightweight CNN that estimates pixel-wise curve parameters for
    low-light image enhancement. The network predicts alpha parameters
    that define light-enhancement curves.

    Architecture:
        - 7 convolutional layers with ReLU activation
        - Skip connections for feature preservation
        - Symmetric encoder-decoder style design
        - Final layer outputs curve parameters (n_curves * 3 channels)

    Enhancement Formula:
        LE(x) = x + α * x * (1 - x)

    This formula is applied iteratively for stronger enhancement.

    Attributes:
        n_curves (int): Number of curve iterations (default 8)

    Example:
        >>> model = ZeroDCE(n_curves=8)
        >>> enhanced, curves = model(low_light_image)
    """

    def __init__(self, n_curves: int = 8):
        """
        Initialize Zero-DCE network.

        Args:
            n_curves: Number of light-enhancement curve iterations
        """
        super(ZeroDCE, self).__init__()

        self.n_curves = n_curves

        # Number of output channels: n_curves iterations × 3 RGB channels
        out_channels = n_curves * 3

        # Encoder layers (feature extraction)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)

        # Decoder layers with skip connections
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv7 = nn.Conv2d(
            64, out_channels, kernel_size=3, stride=1, padding=1, bias=True
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights with Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Zero-DCE network.

        Args:
            x: Input low-light image tensor (B, 3, H, W) in [0, 1]

        Returns:
            Tuple of:
                - enhanced: Enhanced image tensor (B, 3, H, W)
                - curves: Curve parameter maps (B, n_curves*3, H, W)
        """
        # Encoder path
        x1 = F.relu(self.conv1(x))  # (B, 32, H, W)
        x2 = F.relu(self.conv2(x1))  # (B, 32, H, W)
        x3 = F.relu(self.conv3(x2))  # (B, 32, H, W)
        x4 = F.relu(self.conv4(x3))  # (B, 32, H, W)

        # Decoder path with skip connections
        x5 = F.relu(self.conv5(torch.cat([x3, x4], dim=1)))  # (B, 32, H, W)
        x6 = F.relu(self.conv6(torch.cat([x2, x5], dim=1)))  # (B, 32, H, W)

        # Final layer: curve parameters with tanh activation
        # Tanh bounds output to [-1, 1], which works well for curve adjustments
        curves = torch.tanh(self.conv7(torch.cat([x1, x6], dim=1)))  # (B, n*3, H, W)

        # Apply light-enhancement curves iteratively
        enhanced = self._apply_curves(x, curves)

        return enhanced, curves

    def _apply_curves(self, x: torch.Tensor, curves: torch.Tensor) -> torch.Tensor:
        """
        Apply light-enhancement curves to the input image.

        The LE-curve formula: LE(x) = x + α * x * (1 - x)
        This is applied n_curves times with different α values.

        Args:
            x: Input image tensor (B, 3, H, W)
            curves: Curve parameter tensor (B, n_curves*3, H, W)

        Returns:
            Enhanced image tensor (B, 3, H, W)
        """
        # Start with input image
        enhanced = x

        # Apply each set of curve parameters
        for i in range(self.n_curves):
            # Extract alpha values for this iteration
            # Each iteration uses 3 channels (one per RGB)
            alpha = curves[:, i * 3 : (i + 1) * 3, :, :]

            # Apply LE-curve: enhanced = enhanced + α * enhanced * (1 - enhanced)
            enhanced = enhanced + alpha * enhanced * (1 - enhanced)

        # Clamp to valid range [0, 1]
        enhanced = torch.clamp(enhanced, 0, 1)

        return enhanced

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ZeroDCELoss(nn.Module):
    """
    Comprehensive loss function for Zero-DCE training.

    Combines multiple loss terms for unsupervised training:
        - Spatial Consistency Loss: Preserves local structure
        - Exposure Control Loss: Controls overall brightness
        - Color Constancy Loss: Prevents color shifts
        - Illumination Smoothness Loss: Ensures smooth enhancement

    These losses enable training without paired ground-truth data.

    Attributes:
        weights: Dictionary of loss term weights
        patch_size: Patch size for exposure control
        target_exposure: Target mean exposure level
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
        """
        Initialize Zero-DCE loss function.

        Args:
            spa_weight: Weight for spatial consistency loss
            exp_weight: Weight for exposure control loss
            col_weight: Weight for color constancy loss
            smooth_weight: Weight for illumination smoothness loss
            target_exposure: Target exposure level (0-1)
            patch_size: Patch size for exposure calculation
        """
        super(ZeroDCELoss, self).__init__()

        self.spa_weight = spa_weight
        self.exp_weight = exp_weight
        self.col_weight = col_weight
        self.smooth_weight = smooth_weight
        self.target_exposure = target_exposure
        self.patch_size = patch_size

        # Sobel kernels for gradient computation
        self.register_buffer(
            "kernel_left",
            torch.tensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0),
        )
        self.register_buffer(
            "kernel_right",
            torch.tensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0),
        )
        self.register_buffer(
            "kernel_up",
            torch.tensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0),
        )
        self.register_buffer(
            "kernel_down",
            torch.tensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0),
        )

    def forward(
        self, enhanced: torch.Tensor, original: torch.Tensor, curves: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute total Zero-DCE loss.

        Args:
            enhanced: Enhanced image (B, 3, H, W)
            original: Original input image (B, 3, H, W)
            curves: Curve parameter maps (B, n*3, H, W)

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # Individual loss components
        loss_spa = self.spatial_consistency_loss(enhanced, original)
        loss_exp = self.exposure_control_loss(enhanced)
        loss_col = self.color_constancy_loss(enhanced)
        loss_smooth = self.illumination_smoothness_loss(curves)

        # Weighted sum
        total_loss = (
            self.spa_weight * loss_spa
            + self.exp_weight * loss_exp
            + self.col_weight * loss_col
            + self.smooth_weight * loss_smooth
        )

        # Return individual components for monitoring
        components = {
            "spatial": loss_spa.item(),
            "exposure": loss_exp.item(),
            "color": loss_col.item(),
            "smoothness": loss_smooth.item(),
            "total": total_loss.item(),
        }

        return total_loss, components

    def spatial_consistency_loss(
        self, enhanced: torch.Tensor, original: torch.Tensor
    ) -> torch.Tensor:
        """
        Spatial consistency loss to preserve local structure.

        Computes difference in local gradients between original and enhanced.
        """
        # Convert to grayscale
        enhanced_gray = torch.mean(enhanced, dim=1, keepdim=True)
        original_gray = torch.mean(original, dim=1, keepdim=True)

        # Compute gradients in 4 directions
        enhanced_pool = F.avg_pool2d(enhanced_gray, kernel_size=4)
        original_pool = F.avg_pool2d(original_gray, kernel_size=4)

        # Gradient differences
        d_left = F.conv2d(enhanced_pool, self.kernel_left, padding=1) - F.conv2d(
            original_pool, self.kernel_left, padding=1
        )
        d_right = F.conv2d(enhanced_pool, self.kernel_right, padding=1) - F.conv2d(
            original_pool, self.kernel_right, padding=1
        )
        d_up = F.conv2d(enhanced_pool, self.kernel_up, padding=1) - F.conv2d(
            original_pool, self.kernel_up, padding=1
        )
        d_down = F.conv2d(enhanced_pool, self.kernel_down, padding=1) - F.conv2d(
            original_pool, self.kernel_down, padding=1
        )

        loss = (
            torch.pow(d_left, 2)
            + torch.pow(d_right, 2)
            + torch.pow(d_up, 2)
            + torch.pow(d_down, 2)
        )

        return torch.mean(loss)

    def exposure_control_loss(self, enhanced: torch.Tensor) -> torch.Tensor:
        """
        Exposure control loss to achieve target brightness.

        Computes distance from target exposure level using patch means.
        """
        # Average over channels to get luminance
        luminance = torch.mean(enhanced, dim=1, keepdim=True)

        # Compute patch-wise means
        patches = F.avg_pool2d(luminance, kernel_size=self.patch_size)

        # Distance from target exposure
        loss = torch.mean(torch.pow(patches - self.target_exposure, 2))

        return loss

    def color_constancy_loss(self, enhanced: torch.Tensor) -> torch.Tensor:
        """
        Color constancy loss to prevent color shifts.

        Penalizes differences between mean channel values.
        """
        mean_r = torch.mean(enhanced[:, 0, :, :])
        mean_g = torch.mean(enhanced[:, 1, :, :])
        mean_b = torch.mean(enhanced[:, 2, :, :])

        loss = (
            torch.pow(mean_r - mean_g, 2)
            + torch.pow(mean_r - mean_b, 2)
            + torch.pow(mean_g - mean_b, 2)
        ) / 3

        return loss

    def illumination_smoothness_loss(self, curves: torch.Tensor) -> torch.Tensor:
        """
        Illumination smoothness loss for smooth curve parameters.

        Penalizes rapid changes in curve parameters (total variation).
        """
        # Compute gradients
        grad_x = torch.abs(curves[:, :, :, :-1] - curves[:, :, :, 1:])
        grad_y = torch.abs(curves[:, :, :-1, :] - curves[:, :, 1:, :])

        # Total variation
        loss = torch.mean(grad_x) + torch.mean(grad_y)

        return loss


class ZeroDCELightweight(nn.Module):
    """
    Lightweight variant of Zero-DCE for resource-constrained inference.

    Uses fewer channels and layers for faster processing while
    maintaining reasonable enhancement quality.
    """

    def __init__(self, n_curves: int = 8):
        """Initialize lightweight Zero-DCE."""
        super(ZeroDCELightweight, self).__init__()

        self.n_curves = n_curves
        out_channels = n_curves * 3

        # Reduced architecture
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(torch.cat([x2, x3], dim=1)))
        curves = torch.tanh(self.conv5(torch.cat([x1, x4], dim=1)))

        enhanced = self._apply_curves(x, curves)
        return enhanced, curves

    def _apply_curves(self, x: torch.Tensor, curves: torch.Tensor) -> torch.Tensor:
        enhanced = x
        for i in range(self.n_curves):
            alpha = curves[:, i * 3 : (i + 1) * 3, :, :]
            enhanced = enhanced + alpha * enhanced * (1 - enhanced)
        return torch.clamp(enhanced, 0, 1)


def create_model(
    variant: str = "standard", n_curves: int = 8, pretrained: bool = False
) -> nn.Module:
    """
    Factory function to create Zero-DCE model.

    Args:
        variant: Model variant ('standard' or 'lightweight')
        n_curves: Number of enhancement curve iterations
        pretrained: Load pretrained weights (not implemented yet)

    Returns:
        Zero-DCE model instance
    """
    if variant == "standard":
        model = ZeroDCE(n_curves=n_curves)
    elif variant == "lightweight":
        model = ZeroDCELightweight(n_curves=n_curves)
    else:
        raise ValueError(f"Unknown model variant: {variant}")

    print(
        f"[INFO] Created Zero-DCE ({variant}) with {model.get_num_parameters():,} parameters"
    )

    return model
