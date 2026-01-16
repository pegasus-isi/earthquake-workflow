#!/usr/bin/env python3

"""
Visualize seismic gap analysis results.

This script creates visualizations of seismic gap analysis including:
- Geographic map of identified seismic gaps
- Rate ratio heatmap
- Risk score distribution
- Temporal comparison (historical vs recent)
- Gap statistics summary
- Potential magnitude chart

Usage:
    python visualize_seismic_gaps.py \
        --input seismic_gaps.json \
        --output gaps_visualization.png

    # With original catalog for context
    python visualize_seismic_gaps.py \
        --input seismic_gaps.json \
        --catalog earthquake_catalog.csv \
        --output gaps_visualization.png
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
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
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
    'critical': '#67000d',   # Dark red
    'high': '#d62728',       # Red
    'moderate': '#ff7f0e',   # Orange
    'low': '#2ca02c',        # Green
    'very_low': '#1f77b4',   # Blue
    'not_gap': '#cccccc'     # Gray
}

# Rate ratio colormap (high ratio = blue/safe, low ratio = red/gap)
RATE_COLORS = ['#67000d', '#d62728', '#ff7f0e', '#ffff00', '#2ca02c', '#1f77b4']
RATE_CMAP = LinearSegmentedColormap.from_list('rate_ratio', RATE_COLORS)


def load_gap_data(input_file: str) -> Dict:
    """Load seismic gap data from JSON file."""
    logger.info(f"Loading gap data from {input_file}")
    with open(input_file, 'r') as f:
        return json.load(f)


def load_catalog(catalog_file: str) -> Optional[pd.DataFrame]:
    """Load earthquake catalog if provided."""
    if catalog_file:
        logger.info(f"Loading catalog from {catalog_file}")
        return pd.read_csv(catalog_file)
    return None


def create_gap_map(gap_data: Dict, ax: plt.Axes, catalog: Optional[pd.DataFrame] = None):
    """
    Create geographic map of identified seismic gaps.

    Args:
        gap_data: Gap analysis results
        ax: Matplotlib axes
        catalog: Optional earthquake catalog for background events
    """
    logger.info("Creating gap map...")

    gaps = gap_data.get('seismic_gaps', [])
    grid = gap_data.get('grid_analysis', [])

    # Plot background catalog events if available
    if catalog is not None and len(catalog) > 0:
        ax.scatter(catalog['longitude'], catalog['latitude'],
                  s=3, c='gray', alpha=0.3, label='Earthquakes', zorder=1)

    # Plot grid cells (non-gaps in light color)
    non_gaps = [g for g in grid if not g.get('is_gap', False)]
    if non_gaps:
        lons = [g['lon'] for g in non_gaps]
        lats = [g['lat'] for g in non_gaps]
        ax.scatter(lons, lats, s=15, c='lightblue', alpha=0.3, zorder=2)

    # Plot gap cells
    gap_cells = [g for g in grid if g.get('is_gap', False)]
    if gap_cells:
        lons = [g['lon'] for g in gap_cells]
        lats = [g['lat'] for g in gap_cells]
        colors = [RISK_COLORS.get(g.get('risk_level', 'moderate'), '#ff7f0e') for g in gap_cells]
        ax.scatter(lons, lats, s=40, c=colors, alpha=0.7, edgecolors='k',
                  linewidths=0.3, zorder=3)

    # Highlight merged gap regions with bounding boxes
    for gap in gaps[:10]:  # Limit to top 10
        bbox = gap.get('bounding_box', {})
        if bbox:
            width = bbox.get('max_lon', 0) - bbox.get('min_lon', 0)
            height = bbox.get('max_lat', 0) - bbox.get('min_lat', 0)
            rect = Rectangle(
                (bbox.get('min_lon', 0), bbox.get('min_lat', 0)),
                width, height,
                fill=False, edgecolor=RISK_COLORS.get(gap.get('risk_level', 'moderate')),
                linewidth=2, linestyle='--', zorder=4
            )
            ax.add_patch(rect)

            # Add gap ID label
            centroid = gap.get('centroid', {})
            ax.text(centroid.get('lon', 0), centroid.get('lat', 0),
                   f"#{gap.get('gap_id', '')}", fontsize=8, fontweight='bold',
                   ha='center', va='center', zorder=5,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    # Legend
    legend_elements = [
        mpatches.Patch(color=RISK_COLORS['critical'], label='Critical'),
        mpatches.Patch(color=RISK_COLORS['high'], label='High'),
        mpatches.Patch(color=RISK_COLORS['moderate'], label='Moderate'),
        mpatches.Patch(color=RISK_COLORS['low'], label='Low'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.12),
              ncol=4, fontsize=8, title='Gap Risk Level', title_fontsize=9)

    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.set_title('Seismic Gap Locations', fontsize=14, fontweight='bold')


def create_rate_ratio_heatmap(gap_data: Dict, ax: plt.Axes):
    """
    Create heatmap of rate ratios across grid.

    Args:
        gap_data: Gap analysis results
        ax: Matplotlib axes
    """
    logger.info("Creating rate ratio heatmap...")

    grid = gap_data.get('grid_analysis', [])

    if not grid:
        ax.text(0.5, 0.5, 'No grid data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title('Rate Ratio Heatmap', fontsize=14, fontweight='bold')
        return

    # Extract data
    lats = [g['lat'] for g in grid]
    lons = [g['lon'] for g in grid]
    ratios = [min(g.get('rate_ratio', 1.0), 2.0) for g in grid]  # Cap at 2.0

    # Create scatter plot
    scatter = ax.scatter(lons, lats, c=ratios, cmap=RATE_CMAP,
                        s=30, alpha=0.8, edgecolors='k', linewidths=0.2,
                        vmin=0, vmax=1.5)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Rate Ratio (Recent/Historical)', fontsize=9)

    # Add threshold line annotation
    ax.text(0.02, 0.98, 'Low ratio = Potential gap', transform=ax.transAxes,
           fontsize=8, va='top', style='italic', color='darkred')

    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.set_title('Seismicity Rate Ratio', fontsize=14, fontweight='bold')


def create_risk_score_chart(gap_data: Dict, ax: plt.Axes):
    """
    Create bar chart of gap risk scores.

    Args:
        gap_data: Gap analysis results
        ax: Matplotlib axes
    """
    logger.info("Creating risk score chart...")

    gaps = gap_data.get('seismic_gaps', [])

    if not gaps:
        ax.text(0.5, 0.5, 'No gaps identified', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title('Gap Risk Scores', fontsize=14, fontweight='bold')
        return

    # Sort by risk score and limit to top 12
    sorted_gaps = sorted(gaps, key=lambda x: x.get('risk_score', 0), reverse=True)[:12]

    # Create bar chart
    gap_ids = [f"Gap #{g.get('gap_id', i)}" for i, g in enumerate(sorted_gaps)]
    scores = [g.get('risk_score', 0) for g in sorted_gaps]
    colors = [RISK_COLORS.get(g.get('risk_level', 'moderate'), '#ff7f0e') for g in sorted_gaps]

    bars = ax.barh(gap_ids, scores, color=colors, edgecolor='black', alpha=0.8)

    # Add value labels
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{score:.2f}', va='center', fontsize=8)

    # Add threshold lines
    ax.axvline(x=0.65, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axvline(x=0.45, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)

    ax.set_xlabel('Risk Score', fontsize=10)
    ax.set_xlim(0, 1.1)
    ax.set_title('Gap Risk Scores', fontsize=14, fontweight='bold')
    ax.invert_yaxis()


def create_temporal_comparison(gap_data: Dict, ax: plt.Axes):
    """
    Create comparison of historical vs recent seismicity.

    Args:
        gap_data: Gap analysis results
        ax: Matplotlib axes
    """
    logger.info("Creating temporal comparison...")

    gaps = gap_data.get('seismic_gaps', [])
    meta = gap_data.get('metadata', {})

    if not gaps:
        ax.text(0.5, 0.5, 'No gaps identified', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title('Temporal Comparison', fontsize=14, fontweight='bold')
        return

    # Sort and limit
    sorted_gaps = sorted(gaps, key=lambda x: x.get('risk_score', 0), reverse=True)[:10]

    gap_labels = [f"#{g.get('gap_id', i)}" for i, g in enumerate(sorted_gaps)]
    historical = [g.get('historical_events', 0) for g in sorted_gaps]
    recent = [g.get('recent_events', 0) for g in sorted_gaps]

    x = np.arange(len(gap_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, historical, width, label='Historical', color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x + width/2, recent, width, label='Recent', color='#d62728', alpha=0.8)

    ax.set_xlabel('Gap ID', fontsize=10)
    ax.set_ylabel('Event Count', fontsize=10)
    ax.set_title('Historical vs Recent Events', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(gap_labels, fontsize=8)
    ax.legend(loc='upper right', fontsize=9)

    # Add period info
    hist_period = meta.get('historical_period', {})
    recent_period = meta.get('recent_period', {})
    period_text = f"Historical: {hist_period.get('years', 'N/A')}yr, Recent: {recent_period.get('years', 'N/A')}yr"
    ax.text(0.5, -0.15, period_text, transform=ax.transAxes, ha='center', fontsize=8, style='italic')


def create_potential_magnitude_chart(gap_data: Dict, ax: plt.Axes):
    """
    Create chart showing potential magnitudes for gaps.

    Args:
        gap_data: Gap analysis results
        ax: Matplotlib axes
    """
    logger.info("Creating potential magnitude chart...")

    gaps = gap_data.get('seismic_gaps', [])

    if not gaps:
        ax.text(0.5, 0.5, 'No gaps identified', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title('Potential Magnitudes', fontsize=14, fontweight='bold')
        return

    # Sort by potential magnitude
    sorted_gaps = sorted(gaps, key=lambda x: x.get('potential_magnitude', 0), reverse=True)[:10]

    gap_ids = [f"#{g.get('gap_id', i)}" for i, g in enumerate(sorted_gaps)]
    potential_mags = [g.get('potential_magnitude', 0) for g in sorted_gaps]
    historical_max = [g.get('historical_max_magnitude', 0) for g in sorted_gaps]

    x = np.arange(len(gap_ids))
    width = 0.35

    bars1 = ax.bar(x - width/2, historical_max, width, label='Historical Max',
                  color='#2ca02c', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, potential_mags, width, label='Potential',
                  color='#d62728', alpha=0.8, edgecolor='black')

    ax.set_xlabel('Gap ID', fontsize=10)
    ax.set_ylabel('Magnitude', fontsize=10)
    ax.set_title('Historical vs Potential Magnitude', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(gap_ids, fontsize=8)
    ax.legend(loc='upper right', fontsize=9)

    # Add reference lines
    ax.axhline(y=6.0, color='orange', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=7.0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(len(x) - 0.5, 6.05, 'M6', fontsize=8, color='orange')
    ax.text(len(x) - 0.5, 7.05, 'M7', fontsize=8, color='red')


def create_summary_statistics(gap_data: Dict, ax: plt.Axes):
    """
    Create summary statistics panel.

    Args:
        gap_data: Gap analysis results
        ax: Matplotlib axes
    """
    logger.info("Creating summary statistics...")

    ax.axis('off')

    meta = gap_data.get('metadata', {})
    summary = gap_data.get('summary', {})
    gaps = gap_data.get('seismic_gaps', [])
    hist_period = meta.get('historical_period', {})
    recent_period = meta.get('recent_period', {})

    # Build statistics text
    stats_lines = [
        "SEISMIC GAP ANALYSIS SUMMARY",
        "=" * 40,
        "",
        f"Analysis Date: {meta.get('generated_at', 'N/A')[:10]}",
        f"Events Analyzed: {meta.get('total_events_analyzed', 'N/A')}",
        "",
        "TIME PERIODS:",
        f"  Historical: {hist_period.get('years', 'N/A')} years ({hist_period.get('n_events', 'N/A')} events)",
        f"  Recent: {recent_period.get('years', 'N/A')} years ({recent_period.get('n_events', 'N/A')} events)",
        "",
        "=" * 40,
        "",
        "GAP DETECTION RESULTS:",
        f"  Total Cells Analyzed: {summary.get('total_cells_analyzed', 'N/A')}",
        f"  Gap Cells Found: {summary.get('gap_cells_identified', 'N/A')}",
        f"  Merged Gap Regions: {summary.get('total_gaps', 'N/A')}",
        f"  High/Critical Gaps: {summary.get('high_risk_gaps', 0)}",
        "",
        f"  Total Gap Area: {summary.get('total_gap_area_sq_km', 0):,.0f} sq km",
        f"  Max Potential Mag: M{summary.get('max_potential_magnitude', 0):.1f}",
        f"  Overall Rate Change: {summary.get('overall_rate_change', 1.0):.2f}x",
        "",
        "=" * 40,
        "",
        "DETECTION PARAMETERS:",
        f"  Grid Resolution: {meta.get('grid_resolution_deg', 'N/A')}°",
        f"  Rate Threshold: {meta.get('rate_threshold', 'N/A')}",
        f"  Min Significance: {meta.get('min_significance', 'N/A')}",
    ]

    stats_text = '\n'.join(stats_lines)

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=8, fontfamily='monospace',
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    ax.set_title('Analysis Summary', fontsize=14, fontweight='bold')


def create_visualization(gap_data: Dict, output_file: str, title: str,
                         catalog: Optional[pd.DataFrame] = None):
    """
    Create complete gap visualization.

    Args:
        gap_data: Gap analysis results
        output_file: Output image path
        title: Plot title
        catalog: Optional earthquake catalog
    """
    logger.info("Creating gap visualization...")

    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))

    # Define grid layout
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])  # Gap map
    ax2 = fig.add_subplot(gs[0, 1])  # Rate ratio heatmap
    ax3 = fig.add_subplot(gs[0, 2])  # Risk score chart
    ax4 = fig.add_subplot(gs[1, 0])  # Temporal comparison
    ax5 = fig.add_subplot(gs[1, 1])  # Potential magnitude
    ax6 = fig.add_subplot(gs[1, 2])  # Summary statistics

    # Create each panel
    create_gap_map(gap_data, ax1, catalog)
    create_rate_ratio_heatmap(gap_data, ax2)
    create_risk_score_chart(gap_data, ax3)
    create_temporal_comparison(gap_data, ax4)
    create_potential_magnitude_chart(gap_data, ax5)
    create_summary_statistics(gap_data, ax6)

    # Main title
    fig.suptitle(title.replace('_', ' '), fontsize=16, fontweight='bold', y=0.98)

    # Save figure
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Visualization saved to {output_file}")


def print_summary(gap_data: Dict):
    """Print summary of gap results."""
    summary = gap_data.get('summary', {})
    gaps = gap_data.get('seismic_gaps', [])

    print(f"\n{'='*60}")
    print("SEISMIC GAP VISUALIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total gaps identified: {summary.get('total_gaps', 0)}")
    print(f"High/Critical risk gaps: {summary.get('high_risk_gaps', 0)}")
    print(f"Total gap area: {summary.get('total_gap_area_sq_km', 0):,.0f} sq km")
    print(f"Max potential magnitude: M{summary.get('max_potential_magnitude', 0):.1f}")

    if gaps:
        print(f"\nTop 3 Gaps by Risk Score:")
        sorted_gaps = sorted(gaps, key=lambda x: x.get('risk_score', 0), reverse=True)[:3]
        for g in sorted_gaps:
            centroid = g.get('centroid', {})
            print(f"  - Gap #{g.get('gap_id')}: score={g.get('risk_score', 0):.2f}, "
                  f"level={g.get('risk_level')}, "
                  f"potential=M{g.get('potential_magnitude', 0):.1f}, "
                  f"loc=({centroid.get('lat', 0):.2f}, {centroid.get('lon', 0):.2f})")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize seismic gap analysis results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --input gaps.json --output gaps_viz.png
    %(prog)s --input gaps.json --catalog catalog.csv --output gaps_viz.png
        """
    )

    parser.add_argument("--input", type=str, required=True,
                        help="Input seismic gaps JSON file")
    parser.add_argument("--output", type=str, required=True,
                        help="Output visualization image file")
    parser.add_argument("--catalog", type=str, default=None,
                        help="Optional earthquake catalog CSV for context")
    parser.add_argument("--title", type=str, default="Seismic_Gap_Analysis",
                        help="Plot title")

    args = parser.parse_args()

    try:
        # Load data
        gap_data = load_gap_data(args.input)
        catalog = load_catalog(args.catalog) if args.catalog else None

        # Create visualization
        create_visualization(gap_data, args.output, args.title, catalog)

        # Print summary
        print_summary(gap_data)

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
