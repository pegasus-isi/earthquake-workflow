#!/usr/bin/env python3

"""
Visualize aftershock predictions from predict_aftershocks.py output.

This script creates visualizations of aftershock prediction results including:
- Geographic map of mainshocks with aftershock zones
- Probability comparison charts
- Omori-Utsu decay curves
- Risk level summary
- Expected aftershock count heatmap

Usage:
    python visualize_aftershock_predictions.py \
        --input aftershock_predictions.json \
        --output aftershock_visualization.png

    # With original catalog for context
    python visualize_aftershock_predictions.py \
        --input aftershock_predictions.json \
        --catalog earthquake_catalog.csv \
        --output aftershock_visualization.png
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
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
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

# Color schemes
RISK_COLORS = {
    'high': '#d62728',      # Red
    'moderate': '#ff7f0e',  # Orange
    'low': '#2ca02c'        # Green
}

MAGNITUDE_COLORS = plt.cm.plasma


def load_predictions(input_file: str) -> Dict:
    """Load aftershock predictions from JSON file."""
    logger.info(f"Loading predictions from {input_file}")
    with open(input_file, 'r') as f:
        return json.load(f)


def load_catalog(catalog_file: str) -> Optional[pd.DataFrame]:
    """Load earthquake catalog if provided."""
    if catalog_file:
        logger.info(f"Loading catalog from {catalog_file}")
        return pd.read_csv(catalog_file)
    return None


def create_mainshock_map(predictions: Dict, ax: plt.Axes, catalog: Optional[pd.DataFrame] = None):
    """
    Create geographic map of mainshocks with aftershock zones.

    Args:
        predictions: Prediction data dictionary
        ax: Matplotlib axes
        catalog: Optional earthquake catalog for background events
    """
    logger.info("Creating mainshock map...")

    mainshocks = predictions.get('mainshock_predictions', [])

    if not mainshocks:
        ax.text(0.5, 0.5, 'No mainshocks to display',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Mainshock Locations', fontsize=14, fontweight='bold')
        return

    # Plot background catalog events if available
    if catalog is not None and len(catalog) > 0:
        ax.scatter(catalog['longitude'], catalog['latitude'],
                  s=5, c='lightgray', alpha=0.3, label='All events')

    # Plot mainshocks and aftershock zones
    circles = []
    colors = []

    for ms in mainshocks:
        lat = ms['mainshock_location']['latitude']
        lon = ms['mainshock_location']['longitude']
        mag = ms['mainshock_magnitude']
        radius_km = ms['aftershock_zone']['radius_km']

        # Get risk level color
        risk = ms.get('predictions', {}).get('combined', {}).get('risk_level', 'moderate')
        color = RISK_COLORS.get(risk, '#ff7f0e')

        # Convert radius from km to approximate degrees
        radius_deg = radius_km / 111  # rough conversion

        # Add aftershock zone circle
        circle = Circle((lon, lat), radius_deg, fill=True, alpha=0.2)
        circles.append(circle)
        colors.append(color)

        # Plot mainshock point
        size = (mag ** 2) * 10
        ax.scatter(lon, lat, s=size, c=color, edgecolors='black',
                  linewidth=1.5, zorder=5, alpha=0.8)

        # Add magnitude label
        ax.annotate(f'M{mag:.1f}', (lon, lat), fontsize=8,
                   xytext=(5, 5), textcoords='offset points',
                   fontweight='bold', zorder=6)

    # Add circles collection
    if circles:
        for circle, color in zip(circles, colors):
            circle.set_facecolor(color)
            circle.set_edgecolor(color)
            circle.set_alpha(0.2)
            ax.add_patch(circle)

    # Set axis limits with padding
    lons = [ms['mainshock_location']['longitude'] for ms in mainshocks]
    lats = [ms['mainshock_location']['latitude'] for ms in mainshocks]
    radii = [ms['aftershock_zone']['radius_km'] / 111 for ms in mainshocks]

    lon_pad = max(radii) * 2 if radii else 1
    lat_pad = max(radii) * 2 if radii else 1

    ax.set_xlim(min(lons) - lon_pad, max(lons) + lon_pad)
    ax.set_ylim(min(lats) - lat_pad, max(lats) + lat_pad)

    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title(f'Mainshock Locations & Aftershock Zones (N={len(mainshocks)})',
                fontsize=14, fontweight='bold')

    # Legend
    legend_elements = [
        mpatches.Patch(color=RISK_COLORS['high'], alpha=0.5, label='High Risk'),
        mpatches.Patch(color=RISK_COLORS['moderate'], alpha=0.5, label='Moderate Risk'),
        mpatches.Patch(color=RISK_COLORS['low'], alpha=0.5, label='Low Risk'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)


def create_probability_chart(predictions: Dict, ax: plt.Axes):
    """
    Create bar chart comparing aftershock probabilities for each mainshock.

    Args:
        predictions: Prediction data dictionary
        ax: Matplotlib axes
    """
    logger.info("Creating probability chart...")

    mainshocks = predictions.get('mainshock_predictions', [])

    if not mainshocks:
        ax.text(0.5, 0.5, 'No data available',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Aftershock Probabilities', fontsize=14, fontweight='bold')
        return

    # Extract data
    labels = []
    probs_m4 = []
    probs_m5 = []
    probs_m6 = []

    for ms in mainshocks:
        label = f"M{ms['mainshock_magnitude']:.1f}"
        labels.append(label)

        stat_probs = ms.get('predictions', {}).get('statistical', {}).get('probability', {})
        probs_m4.append(stat_probs.get('M>=4.0_within_7_days', 0) * 100)
        probs_m5.append(stat_probs.get('M>=5.0_within_7_days', 0) * 100)
        probs_m6.append(stat_probs.get('M>=6.0_within_7_days', 0) * 100)

    x = np.arange(len(labels))
    width = 0.25

    bars1 = ax.bar(x - width, probs_m4, width, label='M>=4.0', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x, probs_m5, width, label='M>=5.0', color='#f39c12', alpha=0.8)
    bars3 = ax.bar(x + width, probs_m6, width, label='M>=6.0', color='#e74c3c', alpha=0.8)

    ax.set_xlabel('Mainshock', fontsize=11)
    ax.set_ylabel('Probability (%)', fontsize=11)
    ax.set_title('Aftershock Probability (7-day window)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 5:
                ax.annotate(f'{height:.0f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=7)


def create_omori_decay_plot(predictions: Dict, ax: plt.Axes):
    """
    Create Omori-Utsu decay curves for mainshocks.

    Args:
        predictions: Prediction data dictionary
        ax: Matplotlib axes
    """
    logger.info("Creating Omori decay plot...")

    mainshocks = predictions.get('mainshock_predictions', [])

    if not mainshocks:
        ax.text(0.5, 0.5, 'No data available',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Omori-Utsu Decay', fontsize=14, fontweight='bold')
        return

    t = np.linspace(0.01, 30, 200)  # days
    colors = plt.cm.viridis(np.linspace(0, 1, len(mainshocks)))

    for i, ms in enumerate(mainshocks):
        omori = ms.get('predictions', {}).get('statistical', {}).get('omori_parameters', {})
        K = omori.get('K', 10)
        c = omori.get('c', 0.05)
        p = omori.get('p', 1.1)

        # Calculate rate: n(t) = K / (t + c)^p
        rate = K / np.power(t + c, p)

        label = f"M{ms['mainshock_magnitude']:.1f} (K={K:.0f})"
        ax.plot(t, rate, color=colors[i], linewidth=2, label=label, alpha=0.8)

    ax.set_xlabel('Days Since Mainshock', fontsize=11)
    ax.set_ylabel('Aftershock Rate (events/day)', fontsize=11)
    ax.set_title('Omori-Utsu Aftershock Decay', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(0, 30)


def create_risk_summary(predictions: Dict, ax: plt.Axes):
    """
    Create risk level summary pie chart.

    Args:
        predictions: Prediction data dictionary
        ax: Matplotlib axes
    """
    logger.info("Creating risk summary...")

    summary = predictions.get('summary', {})

    high = summary.get('high_risk_mainshocks', 0)
    moderate = summary.get('moderate_risk_mainshocks', 0)
    low = summary.get('low_risk_mainshocks', 0)

    if high + moderate + low == 0:
        ax.text(0.5, 0.5, 'No data available',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Risk Level Distribution', fontsize=14, fontweight='bold')
        return

    sizes = [high, moderate, low]
    labels = [f'High\n({high})', f'Moderate\n({moderate})', f'Low\n({low})']
    colors = [RISK_COLORS['high'], RISK_COLORS['moderate'], RISK_COLORS['low']]
    explode = (0.05, 0, 0)

    # Filter out zero values
    non_zero = [(s, l, c, e) for s, l, c, e in zip(sizes, labels, colors, explode) if s > 0]
    if non_zero:
        sizes, labels, colors, explode = zip(*non_zero)

    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                       autopct='%1.0f%%', shadow=False, startangle=90,
                                       textprops={'fontsize': 10})

    for autotext in autotexts:
        autotext.set_fontweight('bold')

    ax.set_title('Risk Level Distribution', fontsize=14, fontweight='bold')

    # Add average probability text
    avg_prob = summary.get('average_m5_probability', 0)
    ax.text(0, -1.3, f'Avg M5+ Probability: {avg_prob:.1%}',
           ha='center', fontsize=11, fontweight='bold')


def create_expected_counts_heatmap(predictions: Dict, ax: plt.Axes):
    """
    Create heatmap of expected aftershock counts.

    Args:
        predictions: Prediction data dictionary
        ax: Matplotlib axes
    """
    logger.info("Creating expected counts heatmap...")

    mainshocks = predictions.get('mainshock_predictions', [])

    if not mainshocks:
        ax.text(0.5, 0.5, 'No data available',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Expected Aftershock Counts', fontsize=14, fontweight='bold')
        return

    # Build data matrix
    mag_labels = [f"M{ms['mainshock_magnitude']:.1f}" for ms in mainshocks]
    time_windows = ['1_day', '7_day', '30_day']
    time_labels = ['1 day', '7 days', '30 days']

    # Get M>=4.0 expected counts
    data = []
    for ms in mainshocks:
        counts = ms.get('predictions', {}).get('statistical', {}).get('expected_counts', {})
        m4_counts = counts.get('M>=4.0', {})
        row = [m4_counts.get(tw, 0) for tw in time_windows]
        data.append(row)

    data = np.array(data)

    if data.size == 0:
        ax.text(0.5, 0.5, 'No count data available',
                ha='center', va='center', transform=ax.transAxes)
        return

    # Create heatmap
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Expected Count (M>=4.0)', rotation=270, labelpad=15)

    # Set ticks
    ax.set_xticks(np.arange(len(time_labels)))
    ax.set_yticks(np.arange(len(mag_labels)))
    ax.set_xticklabels(time_labels, fontsize=10)
    ax.set_yticklabels(mag_labels, fontsize=10)

    # Add text annotations
    for i in range(len(mag_labels)):
        for j in range(len(time_labels)):
            value = data[i, j]
            color = 'white' if value > data.max() / 2 else 'black'
            ax.text(j, i, f'{value:.0f}', ha='center', va='center',
                   color=color, fontsize=10, fontweight='bold')

    ax.set_xlabel('Time Window', fontsize=11)
    ax.set_ylabel('Mainshock', fontsize=11)
    ax.set_title('Expected Aftershock Counts (M>=4.0)', fontsize=14, fontweight='bold')


def create_bath_law_plot(predictions: Dict, ax: plt.Axes):
    """
    Create Bath's Law visualization showing expected largest aftershock.

    Args:
        predictions: Prediction data dictionary
        ax: Matplotlib axes
    """
    logger.info("Creating Bath's Law plot...")

    mainshocks = predictions.get('mainshock_predictions', [])

    if not mainshocks:
        ax.text(0.5, 0.5, 'No data available',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Bath's Law Predictions", fontsize=14, fontweight='bold')
        return

    mainshock_mags = []
    expected_largest = []
    ranges_low = []
    ranges_high = []

    for ms in mainshocks:
        mainshock_mags.append(ms['mainshock_magnitude'])
        bath = ms.get('predictions', {}).get('statistical', {}).get('bath_law', {})
        expected = bath.get('expected_largest_aftershock', ms['mainshock_magnitude'] - 1.2)
        expected_largest.append(expected)
        range_vals = bath.get('range', [expected - 0.5, expected + 0.5])
        ranges_low.append(expected - range_vals[0])
        ranges_high.append(range_vals[1] - expected)

    x = np.arange(len(mainshocks))
    labels = [f"M{m:.1f}" for m in mainshock_mags]

    # Plot mainshock magnitudes
    ax.bar(x - 0.2, mainshock_mags, 0.35, label='Mainshock', color='#3498db', alpha=0.8)

    # Plot expected largest aftershock with error bars
    ax.bar(x + 0.2, expected_largest, 0.35, label='Expected Largest Aftershock',
          color='#e74c3c', alpha=0.8, yerr=[ranges_low, ranges_high], capsize=4)

    ax.set_xlabel('Event', fontsize=11)
    ax.set_ylabel('Magnitude', fontsize=11)
    ax.set_title("Bath's Law: Expected Largest Aftershock", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')


def create_comprehensive_visualization(predictions: Dict, output_file: str,
                                       catalog: Optional[pd.DataFrame] = None,
                                       title: Optional[str] = None):
    """
    Create comprehensive aftershock prediction visualization.

    Args:
        predictions: Prediction data dictionary
        output_file: Output PNG file path
        catalog: Optional earthquake catalog for context
        title: Optional main title
    """
    logger.info("Creating comprehensive aftershock visualization...")

    # Create figure
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Main title
    if title:
        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    else:
        n_mainshocks = len(predictions.get('mainshock_predictions', []))
        fig.suptitle(f'Aftershock Prediction Analysis (N={n_mainshocks} mainshocks)',
                    fontsize=18, fontweight='bold', y=0.98)

    # Create subplots
    ax1 = fig.add_subplot(gs[0:2, 0:2])  # Map (large)
    ax2 = fig.add_subplot(gs[0, 2])       # Risk summary
    ax3 = fig.add_subplot(gs[1, 2])       # Expected counts heatmap
    ax4 = fig.add_subplot(gs[2, 0])       # Probability chart
    ax5 = fig.add_subplot(gs[2, 1])       # Omori decay
    ax6 = fig.add_subplot(gs[2, 2])       # Bath's law

    # Generate plots
    create_mainshock_map(predictions, ax1, catalog)
    create_risk_summary(predictions, ax2)
    create_expected_counts_heatmap(predictions, ax3)
    create_probability_chart(predictions, ax4)
    create_omori_decay_plot(predictions, ax5)
    create_bath_law_plot(predictions, ax6)

    # Add metadata
    metadata = predictions.get('metadata', {})
    metadata_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"
    if metadata.get('total_events_analyzed'):
        metadata_text += f"\nTotal Events: {metadata['total_events_analyzed']}"
    if metadata.get('mainshocks_identified'):
        metadata_text += f"\nMainshocks: {metadata['mainshocks_identified']}"

    fig.text(0.99, 0.01, metadata_text, ha='right', va='bottom',
            fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Save figure
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Visualization saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize aftershock predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic visualization
  %(prog)s --input aftershock_predictions.json --output visualization.png

  # With original catalog for context
  %(prog)s --input aftershock_predictions.json --catalog earthquakes.csv \
           --output visualization.png --title "California Aftershock Predictions"
        """
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input aftershock predictions JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output PNG file for visualization"
    )
    parser.add_argument(
        "--catalog",
        type=str,
        default=None,
        help="Optional earthquake catalog CSV for background context"
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Main title for visualization"
    )

    args = parser.parse_args()

    try:
        # Load predictions
        predictions = load_predictions(args.input)

        # Load catalog if provided
        catalog = load_catalog(args.catalog) if args.catalog else None

        # Create visualization
        create_comprehensive_visualization(
            predictions, args.output, catalog, args.title
        )

        logger.info("Aftershock visualization completed successfully")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON file: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to create visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
