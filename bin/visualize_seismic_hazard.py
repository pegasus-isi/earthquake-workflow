#!/usr/bin/env python3

"""
Visualize seismic hazard assessment results.

This script creates visualizations of seismic hazard analysis including:
- Hazard map showing PGA distribution
- Risk level distribution map
- Hazard curves for selected sites
- Exceedance probability summary
- Regional hazard statistics

Usage:
    python visualize_seismic_hazard.py \
        --input seismic_hazard.json \
        --output hazard_visualization.png

    # With original catalog for context
    python visualize_seismic_hazard.py \
        --input seismic_hazard.json \
        --catalog earthquake_catalog.csv \
        --output hazard_visualization.png
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')

# Color schemes for risk levels
RISK_COLORS = {
    'very_high': '#67000d',  # Dark red
    'high': '#d62728',       # Red
    'moderate': '#ff7f0e',   # Orange
    'low': '#2ca02c',        # Green
    'very_low': '#1f77b4'    # Blue
}

# PGA colormap (green -> yellow -> orange -> red -> dark red)
PGA_COLORS = ['#2ca02c', '#98df8a', '#ffff00', '#ff7f0e', '#d62728', '#67000d']
PGA_CMAP = LinearSegmentedColormap.from_list('pga', PGA_COLORS)


def load_hazard_data(input_file: str) -> Dict:
    """Load seismic hazard data from JSON file."""
    logger.info(f"Loading hazard data from {input_file}")
    with open(input_file, 'r') as f:
        return json.load(f)


def load_catalog(catalog_file: str) -> Optional[pd.DataFrame]:
    """Load earthquake catalog if provided."""
    if catalog_file:
        logger.info(f"Loading catalog from {catalog_file}")
        return pd.read_csv(catalog_file)
    return None


def create_hazard_map(hazard_data: Dict, ax: plt.Axes, catalog: Optional[pd.DataFrame] = None):
    """
    Create geographic map showing PGA hazard levels.

    Args:
        hazard_data: Hazard analysis results
        ax: Matplotlib axes
        catalog: Optional earthquake catalog for background events
    """
    logger.info("Creating hazard map...")

    grid = hazard_data.get('hazard_grid', [])

    if not grid:
        ax.text(0.5, 0.5, 'No hazard data to display',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Seismic Hazard Map', fontsize=14, fontweight='bold')
        return

    # Extract data
    lats = [p['lat'] for p in grid]
    lons = [p['lon'] for p in grid]
    pga_values = [p.get('max_pga_historical', 0) for p in grid]

    # Plot background catalog events if available
    if catalog is not None and len(catalog) > 0:
        ax.scatter(catalog['longitude'], catalog['latitude'],
                  s=3, c='gray', alpha=0.2, label='Earthquakes', zorder=1)

    # Create scatter plot with PGA colors
    pga_array = np.array(pga_values)
    vmin, vmax = 0, max(0.5, pga_array.max())

    scatter = ax.scatter(lons, lats, c=pga_values, cmap=PGA_CMAP,
                        s=50, alpha=0.8, edgecolors='k', linewidths=0.3,
                        vmin=vmin, vmax=vmax, zorder=2)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Peak Ground Acceleration (g)', fontsize=10)

    # Mark highest hazard locations
    if pga_array.max() > 0:
        top_indices = np.argsort(pga_array)[-3:]
        for idx in top_indices:
            ax.scatter(lons[idx], lats[idx], s=150, facecolors='none',
                      edgecolors='black', linewidths=2, zorder=3)

    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.set_title('Seismic Hazard Map (PGA)', fontsize=14, fontweight='bold')


def create_risk_distribution_map(hazard_data: Dict, ax: plt.Axes):
    """
    Create map showing risk level distribution.

    Args:
        hazard_data: Hazard analysis results
        ax: Matplotlib axes
    """
    logger.info("Creating risk distribution map...")

    grid = hazard_data.get('hazard_grid', [])

    if not grid:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title('Risk Level Distribution', fontsize=14, fontweight='bold')
        return

    # Group by risk level
    risk_levels = ['very_low', 'low', 'moderate', 'high', 'very_high']

    for level in risk_levels:
        points = [(p['lon'], p['lat']) for p in grid if p.get('risk_level') == level]
        if points:
            lons, lats = zip(*points)
            ax.scatter(lons, lats, c=RISK_COLORS.get(level, 'gray'),
                      s=40, alpha=0.7, label=level.replace('_', ' ').title(),
                      edgecolors='k', linewidths=0.2)

    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.set_title('Risk Level Distribution', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', ncol=2, fontsize=7, framealpha=0.9)


def create_hazard_curves(hazard_data: Dict, ax: plt.Axes):
    """
    Create hazard curves for selected high-risk sites.

    Args:
        hazard_data: Hazard analysis results
        ax: Matplotlib axes
    """
    logger.info("Creating hazard curves...")

    grid = hazard_data.get('hazard_grid', [])
    curves = hazard_data.get('hazard_curves', {})

    if not curves:
        ax.text(0.5, 0.5, 'No hazard curves available',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Hazard Curves', fontsize=14, fontweight='bold')
        return

    # Select top sites by hazard
    sorted_grid = sorted(grid, key=lambda x: x.get('max_pga_historical', 0), reverse=True)
    top_sites = sorted_grid[:5]

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(top_sites)))

    for i, site in enumerate(top_sites):
        key = f"{site['lat']}_{site['lon']}"
        if key in curves:
            curve = curves[key]
            pga_vals = curve.get('pga_values', [])
            annual_prob = curve.get('annual_exceedance_prob', [])

            if pga_vals and annual_prob:
                label = f"({site['lat']:.1f}, {site['lon']:.1f})"
                ax.semilogy(pga_vals, annual_prob, '-o', color=colors[i],
                           markersize=3, linewidth=1.5, label=label)

    # Add reference lines for return periods
    ax.axhline(y=1/475, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(0.02, 1/475 * 1.5, '475-yr', fontsize=8, color='red', alpha=0.7)

    ax.axhline(y=1/2475, color='darkred', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(0.02, 1/2475 * 1.5, '2475-yr', fontsize=8, color='darkred', alpha=0.7)

    ax.set_xlabel('PGA (g)', fontsize=10)
    ax.set_ylabel('Annual Exceedance Probability', fontsize=10)
    ax.set_title('Hazard Curves (Top 5 Sites)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.set_xlim(0, None)
    ax.set_ylim(1e-5, 1)
    ax.grid(True, alpha=0.3)


def create_exceedance_summary(hazard_data: Dict, ax: plt.Axes):
    """
    Create bar chart showing exceedance probability summary.

    Args:
        hazard_data: Hazard analysis results
        ax: Matplotlib axes
    """
    logger.info("Creating exceedance summary...")

    grid = hazard_data.get('hazard_grid', [])
    meta = hazard_data.get('metadata', {})
    thresholds = meta.get('pga_thresholds_g', [0.1, 0.2, 0.4])

    if not grid:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title('Exceedance Probability Summary', fontsize=14, fontweight='bold')
        return

    # Calculate percentage of sites exceeding each threshold
    exceed_pcts = []
    labels = []

    for threshold in thresholds:
        key = f'exceedance_prob_{threshold}g_50yr'
        probs = [p.get(key, 0) for p in grid]

        # Percentage with >10% probability of exceedance
        pct_high = 100 * sum(1 for p in probs if p > 0.1) / len(probs)
        exceed_pcts.append(pct_high)
        labels.append(f'{threshold}g')

    colors = ['#2ca02c', '#ff7f0e', '#d62728'][:len(thresholds)]
    bars = ax.bar(labels, exceed_pcts, color=colors, edgecolor='black', alpha=0.8)

    # Add value labels on bars
    for bar, pct in zip(bars, exceed_pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('PGA Threshold', fontsize=10)
    ax.set_ylabel('% Sites with >10% Exceedance Prob (50yr)', fontsize=10)
    ax.set_title('Exceedance Probability Summary', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(exceed_pcts) * 1.2 if exceed_pcts else 100)


def create_regional_statistics(hazard_data: Dict, ax: plt.Axes):
    """
    Create summary statistics panel.

    Args:
        hazard_data: Hazard analysis results
        ax: Matplotlib axes
    """
    logger.info("Creating regional statistics...")

    ax.axis('off')

    meta = hazard_data.get('metadata', {})
    summary = hazard_data.get('regional_summary', {})
    grid = hazard_data.get('hazard_grid', [])

    # Build statistics text
    stats_lines = [
        "REGIONAL HAZARD SUMMARY",
        "=" * 35,
        "",
        f"Analysis Date: {meta.get('generated_at', 'N/A')[:10]}",
        f"Events Analyzed: {meta.get('total_events_used', 'N/A')}",
        f"Time Window: {meta.get('time_window_years', 'N/A'):.1f} years",
        f"Grid Points: {len(grid)}",
        f"Grid Resolution: {meta.get('grid_resolution_deg', 'N/A')}°",
        "",
        "=" * 35,
        "",
        f"Max PGA Observed: {summary.get('max_pga_observed', 0):.3f} g",
        f"Mean PGA: {summary.get('mean_pga', 0):.4f} g",
        f"High Hazard Area: {summary.get('high_hazard_area_pct', 0):.1f}%",
        f"Dominant Mag Range: M{summary.get('dominant_magnitude_range', 'N/A')}",
        "",
        "=" * 35,
        "",
        "RISK DISTRIBUTION:",
    ]

    risk_dist = summary.get('risk_distribution', {})
    total = sum(risk_dist.values()) if risk_dist else 1
    for level in ['very_high', 'high', 'moderate', 'low', 'very_low']:
        count = risk_dist.get(level, 0)
        pct = 100 * count / total if total > 0 else 0
        stats_lines.append(f"  {level.replace('_', ' ').title()}: {count} ({pct:.1f}%)")

    stats_text = '\n'.join(stats_lines)

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=9, fontfamily='monospace',
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    ax.set_title('Regional Statistics', fontsize=14, fontweight='bold')


def create_pga_histogram(hazard_data: Dict, ax: plt.Axes):
    """
    Create histogram of PGA values across grid.

    Args:
        hazard_data: Hazard analysis results
        ax: Matplotlib axes
    """
    logger.info("Creating PGA histogram...")

    grid = hazard_data.get('hazard_grid', [])

    if not grid:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title('PGA Distribution', fontsize=14, fontweight='bold')
        return

    pga_values = [p.get('max_pga_historical', 0) for p in grid]
    pga_values = [v for v in pga_values if v > 0]

    if not pga_values:
        ax.text(0.5, 0.5, 'No PGA data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title('PGA Distribution', fontsize=14, fontweight='bold')
        return

    # Create histogram
    bins = np.linspace(0, max(pga_values) * 1.1, 20)
    n, bins_out, patches = ax.hist(pga_values, bins=bins, edgecolor='black', alpha=0.7)

    # Color bars by PGA level
    for i, patch in enumerate(patches):
        bin_center = (bins_out[i] + bins_out[i+1]) / 2
        patch.set_facecolor(PGA_CMAP(bin_center / max(pga_values)))

    # Add vertical lines for thresholds
    thresholds = [(0.1, 'Moderate'), (0.2, 'High'), (0.4, 'Very High')]
    for thresh, label in thresholds:
        if thresh <= max(pga_values):
            ax.axvline(x=thresh, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
            ax.text(thresh + 0.01, ax.get_ylim()[1] * 0.9, label,
                   fontsize=8, rotation=90, va='top')

    ax.set_xlabel('PGA (g)', fontsize=10)
    ax.set_ylabel('Number of Grid Points', fontsize=10)
    ax.set_title('PGA Distribution', fontsize=14, fontweight='bold')


def create_visualization(hazard_data: Dict, output_file: str, title: str,
                         catalog: Optional[pd.DataFrame] = None):
    """
    Create complete hazard visualization.

    Args:
        hazard_data: Hazard analysis results
        output_file: Output image path
        title: Plot title
        catalog: Optional earthquake catalog
    """
    logger.info("Creating hazard visualization...")

    # Create figure with subplots - use constrained_layout for proper spacing
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08,
                        hspace=0.25, wspace=0.25)

    # Flatten axes for easy access
    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]

    # Create each panel
    create_hazard_map(hazard_data, ax1, catalog)
    create_risk_distribution_map(hazard_data, ax2)
    create_hazard_curves(hazard_data, ax3)
    create_exceedance_summary(hazard_data, ax4)
    create_pga_histogram(hazard_data, ax5)
    create_regional_statistics(hazard_data, ax6)

    # Main title
    fig.suptitle(title.replace('_', ' '), fontsize=16, fontweight='bold')

    # Save figure with fixed dimensions (no bbox_inches='tight' to avoid expansion)
    plt.savefig(output_file, dpi=150, facecolor='white', edgecolor='none')
    plt.close()

    logger.info(f"Visualization saved to {output_file}")


def print_summary(hazard_data: Dict):
    """Print summary of hazard results."""
    meta = hazard_data.get('metadata', {})
    summary = hazard_data.get('regional_summary', {})

    print(f"\n{'='*60}")
    print("SEISMIC HAZARD VISUALIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"Events analyzed: {meta.get('total_events_used', 'N/A')}")
    print(f"Grid points: {summary.get('total_grid_points', 'N/A')}")
    print(f"Max PGA: {summary.get('max_pga_observed', 0):.3f} g")
    print(f"High hazard area: {summary.get('high_hazard_area_pct', 0):.1f}%")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize seismic hazard assessment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --input hazard.json --output hazard_viz.png
    %(prog)s --input hazard.json --catalog catalog.csv --output hazard_viz.png
        """
    )

    parser.add_argument("--input", type=str, required=True,
                        help="Input seismic hazard JSON file")
    parser.add_argument("--output", type=str, required=True,
                        help="Output visualization image file")
    parser.add_argument("--catalog", type=str, default=None,
                        help="Optional earthquake catalog CSV for context")
    parser.add_argument("--title", type=str, default="Seismic_Hazard_Assessment",
                        help="Plot title")

    args = parser.parse_args()

    try:
        # Load data
        hazard_data = load_hazard_data(args.input)
        catalog = load_catalog(args.catalog) if args.catalog else None

        # Create visualization
        create_visualization(hazard_data, args.output, args.title, catalog)

        # Print summary
        print_summary(hazard_data)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
