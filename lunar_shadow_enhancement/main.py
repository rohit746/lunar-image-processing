"""
Main Entry Point for Lunar Shadow Enhancement
==============================================

Complete pipeline demonstrating:
1. Loading Chandrayaan-2 OHRC PDS4 imagery
2. Extracting tiles from shadow regions
3. Enhancing using Zero-DCE deep learning
4. Comparing with traditional gamma correction
5. Visualizing results

Usage:
    uv run python -m lunar_shadow_enhancement.main

Author: Rohit
Project: ISRO Internship Application - Lunar Shadow Enhancement
"""

import argparse
from pathlib import Path
import numpy as np
import torch

from .data_loader import OHRCDataLoader
from .preprocessing import Preprocessor
from .inference import ShadowEnhancer
from .visualization import Visualizer


# Default paths (relative to project root)
DEFAULT_DATA_PATH = (
    "data/ch2_ohr_ncp_20230303T0152168201_d_img_n18/"
    "data/calibrated/20230303/"
    "ch2_ohr_ncp_20230303T0152168201_d_img_n18.xml"
)


def main():
    """
    Main entry point for lunar shadow enhancement pipeline.

    Demonstrates the complete workflow from data loading to visualization.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Lunar Shadow Enhancement using Zero-DCE Deep Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with defaults (uses sample data)
    uv run python -m lunar_shadow_enhancement.main
    
    # Specify custom data path
    uv run python -m lunar_shadow_enhancement.main --data path/to/image.xml
    
    # Enable training fine-tuning
    uv run python -m lunar_shadow_enhancement.main --train --epochs 100
    
    # Save outputs
    uv run python -m lunar_shadow_enhancement.main --output results/
        """,
    )

    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to PDS4 XML label file",
    )

    parser.add_argument(
        "--tile-size",
        "-t",
        type=int,
        default=512,
        help="Size of processing tiles (default: 512)",
    )

    parser.add_argument(
        "--tile-x",
        type=int,
        default=None,
        help="X coordinate for tile center (auto-detect if not specified)",
    )

    parser.add_argument(
        "--tile-y",
        type=int,
        default=None,
        help="Y coordinate for tile center (auto-detect if not specified)",
    )

    parser.add_argument(
        "--train", action="store_true", help="Enable unsupervised training on the tile"
    )

    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )

    parser.add_argument(
        "--target-exposure",
        type=float,
        default=0.5,
        help="Target exposure level for training (default: 0.5)",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=2.2,
        help="Gamma value for comparison (default: 2.2)",
    )

    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output directory for results"
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display figures (useful for headless execution)",
    )

    args = parser.parse_args()

    # Print banner
    print_banner()

    # Run pipeline
    run_pipeline(
        data_path=args.data,
        tile_size=args.tile_size,
        tile_x=args.tile_x,
        tile_y=args.tile_y,
        enable_training=args.train,
        epochs=args.epochs,
        target_exposure=args.target_exposure,
        gamma=args.gamma,
        output_dir=args.output,
        display=not args.no_display,
    )


def print_banner():
    """Print project banner."""
    banner = """
======================================================================
              LUNAR SHADOW ENHANCEMENT PIPELINE                       
                                                                      
   Chandrayaan-2 OHRC Data x Zero-DCE Deep Learning                  
   Recovering Permanently Shadowed Region (PSR) Details              
                                                                      
   ISRO Internship Application Project                                
======================================================================
    """
    print(banner)


def run_pipeline(
    data_path: str,
    tile_size: int = 512,
    tile_x: int = None,
    tile_y: int = None,
    enable_training: bool = False,
    epochs: int = 100,
    target_exposure: float = 0.5,
    gamma: float = 2.2,
    output_dir: str = None,
    display: bool = True,
):
    """
    Execute the complete shadow enhancement pipeline.

    Args:
        data_path: Path to PDS4 XML file
        tile_size: Size of processing tiles
        tile_x: Optional X coordinate for tile
        tile_y: Optional Y coordinate for tile
        enable_training: Enable unsupervised training
        epochs: Number of training epochs
        target_exposure: Target brightness for training
        gamma: Gamma value for comparison
        output_dir: Output directory for results
        display: Whether to display figures
    """
    # Setup output directory
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None

    # =========================================================================
    # STEP 1: DATA LOADING
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 1: Loading Chandrayaan-2 OHRC Data")
    print("=" * 60)

    try:
        loader = OHRCDataLoader(data_path)
        print(loader)

        # Get image statistics
        stats = loader.get_statistics(sample_size=10000)
        print(f"\n   Image Statistics (sampled):")
        print(f"   Range: [{stats['min']:.0f}, {stats['max']:.0f}]")
        print(f"   Mean: {stats['mean']:.1f}, Std: {stats['std']:.1f}")
        print(f"   Bit-depth: {stats['bit_depth']}-bit")

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("\n   Make sure you have the OHRC data in the expected location.")
        print(f"   Expected: {data_path}")
        return

    # =========================================================================
    # STEP 2: SELECT SHADOW REGION
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Selecting Shadow Region")
    print("=" * 60)

    if tile_x is None or tile_y is None:
        # Auto-detect dark regions
        print("   Auto-detecting potential shadow regions...")

        # Use center of image as default (likely to have varied terrain)
        height, width = loader.shape
        tile_y = height // 2
        tile_x = width // 2

        print(f"   Using center tile at ({tile_x}, {tile_y})")
    else:
        print(f"   Using specified coordinates: ({tile_x}, {tile_y})")

    # Extract tile
    print(f"\n📦 Extracting {tile_size}×{tile_size} tile...")
    tile = loader.get_tile(x=tile_x, y=tile_y, size=tile_size)
    print(f"   Tile shape: {tile.shape}")
    print(f"   Tile range: [{tile.min()}, {tile.max()}]")
    print(f"   Tile mean: {tile.mean():.1f}")

    # =========================================================================
    # STEP 3: INITIALIZE ENHANCER
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Initializing Zero-DCE Model")
    print("=" * 60)

    enhancer = ShadowEnhancer(model_variant="standard", n_curves=8, device="auto")

    # Configure preprocessor based on image bit-depth
    enhancer.configure_preprocessor(bit_depth=loader.bit_depth)

    model_info = enhancer.get_model_info()
    print(f"\n   Model Configuration:")
    print(f"   Architecture: {model_info['class']}")
    print(f"   Parameters: {model_info['parameters']:,}")
    print(f"   Device: {model_info['device']}")
    print(f"   Enhancement Curves: {model_info['n_curves']}")

    # =========================================================================
    # STEP 4: OPTIONAL TRAINING
    # =========================================================================
    if enable_training:
        print("\n" + "=" * 60)
        print("STEP 4: Unsupervised Training")
        print("=" * 60)

        print(f"   Training for {epochs} epochs...")
        print(f"   Target exposure: {target_exposure}")

        history = enhancer.train_on_tile(
            tile=tile,
            epochs=epochs,
            target_exposure=target_exposure,
            learning_rate=1e-4,
            verbose=True,
        )

        # Save training curves if output specified
        if output_path:
            viz = Visualizer()
            viz.plot_training_curves(
                history,
                title="Zero-DCE Training Progress",
                save_path=str(output_path / "training_curves.png"),
            )

    # =========================================================================
    # STEP 5: ENHANCEMENT & COMPARISON
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 5: Shadow Enhancement")
    print("=" * 60)

    print("   Running enhancement pipeline...")

    # Get all three versions
    original, gamma_corrected, ai_enhanced = enhancer.enhance_with_comparison(
        tile=tile, gamma=gamma, normalize_method="linear"
    )

    print(f"\n   Enhancement Results:")
    print(f"   Original mean intensity: {np.mean(original):.4f}")
    print(f"   Gamma corrected mean: {np.mean(gamma_corrected):.4f}")
    print(f"   AI enhanced mean: {np.mean(ai_enhanced):.4f}")

    # Improvement metrics
    original_contrast = np.std(original)
    enhanced_contrast = np.std(ai_enhanced)
    contrast_improvement = (enhanced_contrast / original_contrast - 1) * 100

    print(f"\n   Quality Metrics:")
    print(f"   Contrast improvement: {contrast_improvement:+.1f}%")
    print(f"   Brightness boost: {np.mean(ai_enhanced) / np.mean(original):.2f}x")

    # =========================================================================
    # STEP 6: VISUALIZATION
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 6: Visualization")
    print("=" * 60)

    viz = Visualizer(figsize=(15, 5), dpi=150)

    # Main comparison figure
    save_comparison = str(output_path / "comparison.png") if output_path else None
    fig_comparison = viz.plot_comparison(
        original=original,
        gamma_corrected=gamma_corrected,
        ai_enhanced=ai_enhanced,
        title=f"Lunar Shadow Enhancement - Tile at ({tile_x}, {tile_y})",
        gamma_value=gamma,
        save_path=save_comparison,
        show_stats=True,
    )

    # Histogram comparison
    save_hist = str(output_path / "histograms.png") if output_path else None
    fig_hist = viz.plot_histograms(
        original=original,
        enhanced=ai_enhanced,
        title="Intensity Distribution: Before vs After Enhancement",
        save_path=save_hist,
    )

    # Save checkpoint if output specified
    if output_path:
        checkpoint_path = output_path / "zerodce_checkpoint.pth"
        enhancer.save_checkpoint(checkpoint_path)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    if output_path:
        print(f"\nResults saved to: {output_path}")
        print("   - comparison.png: Side-by-side enhancement comparison")
        print("   - histograms.png: Intensity distribution analysis")
        print("   - zerodce_checkpoint.pth: Model weights")
        if enable_training:
            print("   - training_curves.png: Training loss plots")

    print("\nKey Findings:")
    print(f"   - Original shadow regions had mean intensity: {np.mean(original):.3f}")
    print(f"   - Zero-DCE enhanced mean intensity to: {np.mean(ai_enhanced):.3f}")
    print(f"   - Traditional gamma correction achieved: {np.mean(gamma_corrected):.3f}")

    if display:
        print("\nDisplaying figures...")
        viz.show()

    return {
        "original": original,
        "gamma_corrected": gamma_corrected,
        "ai_enhanced": ai_enhanced,
        "loader": loader,
        "enhancer": enhancer,
    }


if __name__ == "__main__":
    main()
