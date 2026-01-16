#!/usr/bin/env python3

"""
Visualize earthquake data with maps and plots.

This script creates visualizations of earthquake data including:
- Geographic map of earthquake locations (colored by magnitude)
- Magnitude-depth scatter plot
- Time series of earthquake frequency
- Magnitude distribution histogram
- Cumulative magnitude plot

Usage:
    python visualize_earthquakes.py --input earthquake_catalog.csv \
                                    --output analysis/earthquake_map.png \
                                    --title "California Earthquakes 2024"
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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


def create_geographic_map(df: pd.DataFrame, ax: plt.Axes, title: str = "Earthquake Locations"):
    """
    Create geographic map of earthquake locations.

    Args:
        df: DataFrame with earthquake data
        ax: Matplotlib axes
        title: Plot title
    """
    logger.info("Creating geographic map...")

    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        logger.warning("No spatial data available")
        ax.text(0.5, 0.5, 'No spatial data available',
                ha='center', va='center', transform=ax.transAxes)
        return

    # Size by magnitude
    sizes = (df['magnitude'] ** 2) * 5 if 'magnitude' in df.columns else 20

    # Color by depth
    if 'depth_km' in df.columns:
        colors = df['depth_km']
        scatter = ax.scatter(df['longitude'], df['latitude'],
                           c=colors, s=sizes, alpha=0.6,
                           cmap='viridis_r', edgecolors='black', linewidth=0.5)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Depth (km)', rotation=270, labelpad=20)
    else:
        ax.scatter(df['longitude'], df['latitude'],
                  s=sizes, alpha=0.6, edgecolors='black', linewidth=0.5)

    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add legend for magnitude
    if 'magnitude' in df.columns:
        legend_mags = [3.0, 5.0, 7.0]
        legend_sizes = [(m ** 2) * 5 for m in legend_mags]
        for mag, size in zip(legend_mags, legend_sizes):
            ax.scatter([], [], s=size, c='gray', alpha=0.6,
                      edgecolors='black', linewidth=0.5,
                      label=f'M {mag:.1f}')
        ax.legend(scatterpoints=1, frameon=True, labelspacing=2, title='Magnitude',
                 loc='upper right')


def create_magnitude_depth_plot(df: pd.DataFrame, ax: plt.Axes):
    """
    Create magnitude vs depth scatter plot.

    Args:
        df: DataFrame with earthquake data
        ax: Matplotlib axes
    """
    logger.info("Creating magnitude-depth plot...")

    if 'magnitude' not in df.columns or 'depth_km' not in df.columns:
        logger.warning("Missing magnitude or depth data")
        ax.text(0.5, 0.5, 'Missing magnitude or depth data',
                ha='center', va='center', transform=ax.transAxes)
        return

    valid_data = df[['magnitude', 'depth_km']].dropna()

    if len(valid_data) == 0:
        ax.text(0.5, 0.5, 'No valid magnitude-depth data',
                ha='center', va='center', transform=ax.transAxes)
        return

    ax.scatter(valid_data['magnitude'], valid_data['depth_km'],
              alpha=0.5, s=30, edgecolors='black', linewidth=0.5)

    ax.set_xlabel('Magnitude', fontsize=12)
    ax.set_ylabel('Depth (km)', fontsize=12)
    ax.set_title('Magnitude vs Depth', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Depth increases downward

    # Add depth category lines
    ax.axhline(y=70, color='red', linestyle='--', alpha=0.5, linewidth=1.5,
              label='Shallow/Intermediate boundary')
    ax.axhline(y=300, color='orange', linestyle='--', alpha=0.5, linewidth=1.5,
              label='Intermediate/Deep boundary')
    ax.legend(fontsize=9, loc='lower right')


def create_time_series_plot(df: pd.DataFrame, ax: plt.Axes):
    """
    Create time series of earthquake frequency.

    Args:
        df: DataFrame with earthquake data
        ax: Matplotlib axes
    """
    logger.info("Creating time series plot...")

    if 'time' not in df.columns:
        logger.warning("No time data available")
        ax.text(0.5, 0.5, 'No time data available',
                ha='center', va='center', transform=ax.transAxes)
        return

    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], format='mixed', utc=True)
    df = df.sort_values('time')

    # Daily event counts
    daily_counts = df.groupby(df['time'].dt.date).size()
    dates = pd.to_datetime(daily_counts.index, utc=True)

    ax.plot(dates, daily_counts.values, linewidth=2, color='steelblue', label='Daily events')

    # Add 7-day rolling average
    if len(daily_counts) >= 7:
        rolling_avg = daily_counts.rolling(window=7, center=True).mean()
        ax.plot(dates, rolling_avg.values, linewidth=2, color='red',
               linestyle='--', label='7-day average')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Number of Events', fontsize=12)
    ax.set_title('Earthquake Frequency Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')


def create_magnitude_histogram(df: pd.DataFrame, ax: plt.Axes):
    """
    Create magnitude distribution histogram.

    Args:
        df: DataFrame with earthquake data
        ax: Matplotlib axes
    """
    logger.info("Creating magnitude histogram...")

    if 'magnitude' not in df.columns:
        logger.warning("No magnitude data available")
        ax.text(0.5, 0.5, 'No magnitude data available',
                ha='center', va='center', transform=ax.transAxes)
        return

    magnitudes = df['magnitude'].dropna()

    if len(magnitudes) == 0:
        ax.text(0.5, 0.5, 'No valid magnitude data',
                ha='center', va='center', transform=ax.transAxes)
        return

    # Create histogram
    bins = np.arange(np.floor(magnitudes.min()), np.ceil(magnitudes.max()) + 0.5, 0.5)
    ax.hist(magnitudes, bins=bins, color='steelblue', alpha=0.7,
           edgecolor='black', linewidth=1.2)

    ax.set_xlabel('Magnitude', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Magnitude Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add statistics
    stats_text = (f"N = {len(magnitudes)}\n"
                 f"Mean = {magnitudes.mean():.2f}\n"
                 f"Median = {magnitudes.median():.2f}\n"
                 f"Std = {magnitudes.std():.2f}")
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def create_cumulative_magnitude_plot(df: pd.DataFrame, ax: plt.Axes):
    """
    Create cumulative magnitude plot (Gutenberg-Richter).

    Args:
        df: DataFrame with earthquake data
        ax: Matplotlib axes
    """
    logger.info("Creating cumulative magnitude plot...")

    if 'magnitude' not in df.columns:
        logger.warning("No magnitude data available")
        ax.text(0.5, 0.5, 'No magnitude data available',
                ha='center', va='center', transform=ax.transAxes)
        return

    magnitudes = df['magnitude'].dropna().sort_values()

    if len(magnitudes) == 0:
        ax.text(0.5, 0.5, 'No valid magnitude data',
                ha='center', va='center', transform=ax.transAxes)
        return

    # Calculate cumulative counts (N >= M)
    mag_values = np.linspace(magnitudes.min(), magnitudes.max(), 100)
    cumulative_counts = [np.sum(magnitudes >= m) for m in mag_values]

    ax.semilogy(mag_values, cumulative_counts, linewidth=2, color='steelblue',
               label='Observed')

    ax.set_xlabel('Magnitude', fontsize=12)
    ax.set_ylabel('Cumulative Number (N ≥ M)', fontsize=12)
    ax.set_title('Gutenberg-Richter Relation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10)


def create_depth_histogram(df: pd.DataFrame, ax: plt.Axes):
    """
    Create depth distribution histogram.

    Args:
        df: DataFrame with earthquake data
        ax: Matplotlib axes
    """
    logger.info("Creating depth histogram...")

    if 'depth_km' not in df.columns:
        logger.warning("No depth data available")
        ax.text(0.5, 0.5, 'No depth data available',
                ha='center', va='center', transform=ax.transAxes)
        return

    depths = df['depth_km'].dropna()

    if len(depths) == 0:
        ax.text(0.5, 0.5, 'No valid depth data',
                ha='center', va='center', transform=ax.transAxes)
        return

    # Create histogram with bins appropriate for the data range
    max_depth = depths.max()
    standard_bins = [0, 10, 20, 40, 70, 150, 300]
    # Keep only bins <= max_depth, then add the final edge
    bins = [b for b in standard_bins if b <= max_depth] + [max_depth + 1]
    if len(bins) < 2:
        bins = [0, max_depth + 1]
    ax.hist(depths, bins=bins, color='coral', alpha=0.7,
           edgecolor='black', linewidth=1.2)

    ax.set_xlabel('Depth (km)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Depth Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add category annotations
    ax.axvline(x=70, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axvline(x=300, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)

    # Add statistics
    stats_text = (f"N = {len(depths)}\n"
                 f"Mean = {depths.mean():.1f} km\n"
                 f"Median = {depths.median():.1f} km")
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def create_comprehensive_visualization(df: pd.DataFrame, output_file: str,
                                      title: Optional[str] = None):
    """
    Create comprehensive earthquake visualization with multiple subplots.

    Args:
        df: DataFrame with earthquake data
        output_file: Output PNG file path
        title: Optional main title
    """
    logger.info(f"Creating comprehensive visualization for {len(df)} events...")

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Main title
    if title:
        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)

    # Create subplots
    ax1 = fig.add_subplot(gs[0:2, 0:2])  # Geographic map (large)
    ax2 = fig.add_subplot(gs[0, 2])      # Magnitude histogram
    ax3 = fig.add_subplot(gs[1, 2])      # Depth histogram
    ax4 = fig.add_subplot(gs[2, 0])      # Time series
    ax5 = fig.add_subplot(gs[2, 1])      # Magnitude-depth
    ax6 = fig.add_subplot(gs[2, 2])      # Cumulative magnitude

    # Generate plots
    create_geographic_map(df, ax1, title=f"Geographic Distribution (N={len(df)})")
    create_magnitude_histogram(df, ax2)
    create_depth_histogram(df, ax3)
    create_time_series_plot(df, ax4)
    create_magnitude_depth_plot(df, ax5)
    create_cumulative_magnitude_plot(df, ax6)

    # Add metadata text
    metadata_text = (
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"Total Events: {len(df)}"
    )
    if 'magnitude' in df.columns:
        mags = df['magnitude'].dropna()
        if len(mags) > 0:
            metadata_text += f"\nMagnitude Range: {mags.min():.1f} - {mags.max():.1f}"
    if 'time' in df.columns:
        times = pd.to_datetime(df['time'].dropna(), format='mixed', utc=True)
        if len(times) > 0:
            metadata_text += f"\nDate Range: {times.min().date()} to {times.max().date()}"

    fig.text(0.99, 0.01, metadata_text, ha='right', va='bottom',
            fontsize=9, style='italic', bbox=dict(boxstyle='round',
            facecolor='wheat', alpha=0.3))

    # Save figure
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Visualization saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize earthquake data with maps and plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create visualization
  %(prog)s --input earthquake_catalog.csv --output analysis/earthquake_map.png

  # With custom title
  %(prog)s --input california_2024.csv --output analysis/california_map.png \
           --title "California Earthquakes 2024"
        """
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input earthquake catalog CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output PNG file for visualization"
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Main title for visualization"
    )

    args = parser.parse_args()

    try:
        # Load earthquake data
        logger.info(f"Loading earthquake data from {args.input}")
        df = pd.read_csv(args.input)

        if df.empty:
            logger.error("Input file is empty")
            sys.exit(1)

        logger.info(f"Loaded {len(df)} earthquake events")

        # Create visualization
        create_comprehensive_visualization(df, args.output, args.title)

        logger.info("Earthquake visualization completed successfully")

    except FileNotFoundError:
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to create visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
