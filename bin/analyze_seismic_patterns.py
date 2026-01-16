#!/usr/bin/env python3

"""
Analyze seismic patterns from earthquake catalog data.

This script performs statistical analysis on earthquake data including:
- Magnitude-frequency distribution (Gutenberg-Richter law)
- Depth profile analysis
- Temporal patterns and trends
- Spatial distribution analysis
- Event clustering metrics

Usage:
    python analyze_seismic_patterns.py --input earthquake_catalog.csv \
                                       --output analysis/patterns.json
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_magnitude_distribution(df: pd.DataFrame) -> Dict:
    """
    Analyze magnitude distribution and fit Gutenberg-Richter law.

    The Gutenberg-Richter law: log10(N) = a - b*M
    where N is the number of earthquakes >= magnitude M

    Args:
        df: DataFrame with earthquake data

    Returns:
        Dictionary with magnitude distribution statistics
    """
    logger.info("Analyzing magnitude distribution...")

    magnitudes = df['magnitude'].dropna()

    if len(magnitudes) == 0:
        logger.warning("No magnitude data available")
        return {}

    # Basic statistics
    mag_stats = {
        'count': int(len(magnitudes)),
        'min': float(magnitudes.min()),
        'max': float(magnitudes.max()),
        'mean': float(magnitudes.mean()),
        'median': float(magnitudes.median()),
        'std': float(magnitudes.std()),
        'quartiles': {
            'q25': float(magnitudes.quantile(0.25)),
            'q50': float(magnitudes.quantile(0.50)),
            'q75': float(magnitudes.quantile(0.75))
        }
    }

    # Magnitude bins
    bins = np.arange(np.floor(magnitudes.min()), np.ceil(magnitudes.max()) + 0.5, 0.5)
    hist, bin_edges = np.histogram(magnitudes, bins=bins)

    mag_stats['histogram'] = {
        'bins': bin_edges.tolist(),
        'counts': hist.tolist()
    }

    # Gutenberg-Richter law fitting
    # Calculate cumulative frequency (N >= M)
    mag_bins = np.arange(magnitudes.min(), magnitudes.max(), 0.1)
    cumulative_counts = []

    for mag in mag_bins:
        count = len(magnitudes[magnitudes >= mag])
        if count > 0:
            cumulative_counts.append(count)
        else:
            cumulative_counts.append(1)  # Avoid log(0)

    # Fit log10(N) = a - b*M
    log_counts = np.log10(cumulative_counts)
    valid_indices = np.isfinite(log_counts)

    if np.sum(valid_indices) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            mag_bins[valid_indices],
            log_counts[valid_indices]
        )

        mag_stats['gutenberg_richter'] = {
            'b_value': float(-slope),  # b-value (typically ~1.0 for natural earthquakes)
            'a_value': float(intercept),  # a-value (log of productivity)
            'r_squared': float(r_value ** 2),
            'p_value': float(p_value),
            'std_error': float(std_err),
            'interpretation': (
                'Natural seismicity' if 0.8 <= -slope <= 1.2
                else 'Induced seismicity' if -slope < 0.8
                else 'Aftershock sequence' if -slope > 1.2
                else 'Unknown'
            )
        }

        logger.info(f"Gutenberg-Richter b-value: {-slope:.3f} (R²={r_value**2:.3f})")
    else:
        logger.warning("Insufficient data for Gutenberg-Richter fitting")
        mag_stats['gutenberg_richter'] = None

    # Count by magnitude category
    mag_stats['magnitude_categories'] = {
        'micro (M<3.0)': int(len(magnitudes[magnitudes < 3.0])),
        'minor (3.0≤M<4.0)': int(len(magnitudes[(magnitudes >= 3.0) & (magnitudes < 4.0)])),
        'light (4.0≤M<5.0)': int(len(magnitudes[(magnitudes >= 4.0) & (magnitudes < 5.0)])),
        'moderate (5.0≤M<6.0)': int(len(magnitudes[(magnitudes >= 5.0) & (magnitudes < 6.0)])),
        'strong (6.0≤M<7.0)': int(len(magnitudes[(magnitudes >= 6.0) & (magnitudes < 7.0)])),
        'major (7.0≤M<8.0)': int(len(magnitudes[(magnitudes >= 7.0) & (magnitudes < 8.0)])),
        'great (M≥8.0)': int(len(magnitudes[magnitudes >= 8.0]))
    }

    return mag_stats


def analyze_depth_profile(df: pd.DataFrame) -> Dict:
    """
    Analyze depth distribution of earthquakes.

    Args:
        df: DataFrame with earthquake data

    Returns:
        Dictionary with depth profile statistics
    """
    logger.info("Analyzing depth profile...")

    depths = df['depth_km'].dropna()

    if len(depths) == 0:
        logger.warning("No depth data available")
        return {}

    depth_stats = {
        'count': int(len(depths)),
        'min': float(depths.min()),
        'max': float(depths.max()),
        'mean': float(depths.mean()),
        'median': float(depths.median()),
        'std': float(depths.std()),
        'quartiles': {
            'q25': float(depths.quantile(0.25)),
            'q50': float(depths.quantile(0.50)),
            'q75': float(depths.quantile(0.75))
        }
    }

    # Depth categories
    depth_stats['depth_categories'] = {
        'shallow (0-70 km)': int(len(depths[(depths >= 0) & (depths < 70)])),
        'intermediate (70-300 km)': int(len(depths[(depths >= 70) & (depths < 300)])),
        'deep (>300 km)': int(len(depths[depths >= 300]))
    }

    # Histogram
    bins = [0, 10, 20, 40, 70, 150, 300, 700]
    hist, bin_edges = np.histogram(depths, bins=bins)

    depth_stats['histogram'] = {
        'bins': bin_edges.tolist(),
        'counts': hist.tolist()
    }

    # Magnitude vs depth correlation
    if 'magnitude' in df.columns:
        valid_data = df[['magnitude', 'depth_km']].dropna()
        if len(valid_data) > 2:
            correlation, p_value = stats.pearsonr(
                valid_data['magnitude'],
                valid_data['depth_km']
            )
            depth_stats['magnitude_correlation'] = {
                'coefficient': float(correlation),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05)
            }

    return depth_stats


def analyze_temporal_patterns(df: pd.DataFrame) -> Dict:
    """
    Analyze temporal patterns and trends in earthquake occurrence.

    Args:
        df: DataFrame with earthquake data

    Returns:
        Dictionary with temporal pattern statistics
    """
    logger.info("Analyzing temporal patterns...")

    if 'time' not in df.columns or df['time'].isna().all():
        logger.warning("No time data available")
        return {}

    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], format='mixed', utc=True)
    df = df.sort_values('time')

    temporal_stats = {
        'date_range': {
            'start': df['time'].min().isoformat(),
            'end': df['time'].max().isoformat(),
            'duration_days': float((df['time'].max() - df['time'].min()).total_seconds() / 86400)
        },
        'total_events': int(len(df))
    }

    # Events per day
    daily_counts = df.groupby(df['time'].dt.date).size()
    temporal_stats['daily_rate'] = {
        'mean': float(daily_counts.mean()),
        'median': float(daily_counts.median()),
        'std': float(daily_counts.std()),
        'min': int(daily_counts.min()),
        'max': int(daily_counts.max())
    }

    # Events per hour of day
    hourly_distribution = df.groupby(df['time'].dt.hour).size()
    temporal_stats['hourly_distribution'] = hourly_distribution.to_dict()

    # Events per day of week
    dow_distribution = df.groupby(df['time'].dt.dayofweek).size()
    temporal_stats['day_of_week_distribution'] = {
        int(k): int(v) for k, v in dow_distribution.items()
    }

    # Trend analysis (linear regression on cumulative count)
    df['days_since_start'] = (df['time'] - df['time'].min()).dt.total_seconds() / 86400
    df['event_number'] = range(1, len(df) + 1)

    if len(df) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df['days_since_start'],
            df['event_number']
        )

        temporal_stats['temporal_trend'] = {
            'events_per_day': float(slope),
            'r_squared': float(r_value ** 2),
            'p_value': float(p_value),
            'trend': (
                'increasing' if slope > 0 and p_value < 0.05
                else 'decreasing' if slope < 0 and p_value < 0.05
                else 'stable'
            )
        }

    # Inter-event time statistics
    df['inter_event_time'] = df['time'].diff().dt.total_seconds() / 3600  # hours
    inter_event_times = df['inter_event_time'].dropna()

    if len(inter_event_times) > 0:
        temporal_stats['inter_event_time_hours'] = {
            'mean': float(inter_event_times.mean()),
            'median': float(inter_event_times.median()),
            'std': float(inter_event_times.std()),
            'min': float(inter_event_times.min()),
            'max': float(inter_event_times.max())
        }

    return temporal_stats


def analyze_spatial_distribution(df: pd.DataFrame) -> Dict:
    """
    Analyze spatial distribution of earthquakes.

    Args:
        df: DataFrame with earthquake data

    Returns:
        Dictionary with spatial distribution statistics
    """
    logger.info("Analyzing spatial distribution...")

    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        logger.warning("No spatial data available")
        return {}

    spatial_data = df[['latitude', 'longitude']].dropna()

    if len(spatial_data) == 0:
        logger.warning("No valid spatial data")
        return {}

    spatial_stats = {
        'count': int(len(spatial_data)),
        'latitude': {
            'min': float(spatial_data['latitude'].min()),
            'max': float(spatial_data['latitude'].max()),
            'mean': float(spatial_data['latitude'].mean()),
            'std': float(spatial_data['latitude'].std())
        },
        'longitude': {
            'min': float(spatial_data['longitude'].min()),
            'max': float(spatial_data['longitude'].max()),
            'mean': float(spatial_data['longitude'].mean()),
            'std': float(spatial_data['longitude'].std())
        }
    }

    # Calculate geographic extent
    lat_range = spatial_data['latitude'].max() - spatial_data['latitude'].min()
    lon_range = spatial_data['longitude'].max() - spatial_data['longitude'].min()

    spatial_stats['geographic_extent'] = {
        'latitude_range_degrees': float(lat_range),
        'longitude_range_degrees': float(lon_range),
        'approximate_area_km2': float(lat_range * 111 * lon_range * 111)  # Rough approximation
    }

    # Spatial clustering metrics (simple grid-based)
    lat_bins = 20
    lon_bins = 20

    H, xedges, yedges = np.histogram2d(
        spatial_data['latitude'],
        spatial_data['longitude'],
        bins=[lat_bins, lon_bins]
    )

    # Find hotspots (cells with >50% of max count)
    threshold = H.max() * 0.5
    hotspot_cells = np.sum(H > threshold)

    spatial_stats['spatial_clustering'] = {
        'total_grid_cells': int(lat_bins * lon_bins),
        'occupied_cells': int(np.sum(H > 0)),
        'hotspot_cells': int(hotspot_cells),
        'concentration_ratio': float(hotspot_cells / (lat_bins * lon_bins))
    }

    return spatial_stats


def analyze_event_types(df: pd.DataFrame) -> Dict:
    """
    Analyze distribution of event types.

    Args:
        df: DataFrame with earthquake data

    Returns:
        Dictionary with event type statistics
    """
    logger.info("Analyzing event types...")

    if 'event_type' not in df.columns:
        return {}

    event_counts = df['event_type'].value_counts()

    event_stats = {
        'total': int(len(df)),
        'types': event_counts.to_dict()
    }

    # Tsunami-related events
    if 'tsunami' in df.columns:
        tsunami_count = int(df['tsunami'].sum())
        event_stats['tsunami_events'] = tsunami_count
        event_stats['tsunami_percentage'] = float(tsunami_count / len(df) * 100)

    # Significance analysis
    if 'significance' in df.columns:
        sig_data = df['significance'].dropna()
        if len(sig_data) > 0:
            event_stats['significance'] = {
                'mean': float(sig_data.mean()),
                'median': float(sig_data.median()),
                'max': float(sig_data.max()),
                'high_significance_count': int(len(sig_data[sig_data >= 600]))
            }

    return event_stats


def generate_summary(df: pd.DataFrame, output_file: str):
    """
    Generate comprehensive seismic pattern analysis.

    Args:
        df: DataFrame with earthquake data
        output_file: Output JSON file path
    """
    logger.info(f"Generating seismic pattern analysis for {len(df)} events...")

    analysis = {
        'metadata': {
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'total_events': int(len(df)),
            'analysis_version': '1.0'
        },
        'magnitude_distribution': analyze_magnitude_distribution(df),
        'depth_profile': analyze_depth_profile(df),
        'temporal_patterns': analyze_temporal_patterns(df),
        'spatial_distribution': analyze_spatial_distribution(df),
        'event_types': analyze_event_types(df)
    }

    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)

    logger.info(f"Analysis saved to {output_file}")

    # Print summary
    print(f"\n{'='*70}")
    print("SEISMIC PATTERN ANALYSIS SUMMARY")
    print(f"{'='*70}")

    if analysis['magnitude_distribution']:
        mag = analysis['magnitude_distribution']
        print(f"\nMagnitude Distribution:")
        print(f"  Range: {mag['min']:.1f} - {mag['max']:.1f}")
        print(f"  Mean: {mag['mean']:.2f} ± {mag['std']:.2f}")
        if mag.get('gutenberg_richter'):
            gr = mag['gutenberg_richter']
            print(f"  Gutenberg-Richter b-value: {gr['b_value']:.3f} ({gr['interpretation']})")

    if analysis['depth_profile']:
        depth = analysis['depth_profile']
        print(f"\nDepth Profile:")
        print(f"  Range: {depth['min']:.1f} - {depth['max']:.1f} km")
        print(f"  Mean: {depth['mean']:.1f} ± {depth['std']:.1f} km")
        print(f"  Shallow (<70 km): {depth['depth_categories']['shallow (0-70 km)']} events")

    if analysis['temporal_patterns']:
        temp = analysis['temporal_patterns']
        print(f"\nTemporal Patterns:")
        print(f"  Duration: {temp['date_range']['duration_days']:.1f} days")
        print(f"  Daily rate: {temp['daily_rate']['mean']:.1f} events/day")
        if temp.get('temporal_trend'):
            trend = temp['temporal_trend']
            print(f"  Trend: {trend['trend']} ({trend['events_per_day']:.2f} events/day)")

    if analysis['spatial_distribution']:
        spatial = analysis['spatial_distribution']
        print(f"\nSpatial Distribution:")
        print(f"  Latitude: {spatial['latitude']['min']:.2f}° to {spatial['latitude']['max']:.2f}°")
        print(f"  Longitude: {spatial['longitude']['min']:.2f}° to {spatial['longitude']['max']:.2f}°")
        print(f"  Area: ~{spatial['geographic_extent']['approximate_area_km2']:.0f} km²")

    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze seismic patterns from earthquake catalog data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze earthquake catalog
  %(prog)s --input earthquake_catalog.csv --output analysis/patterns.json

  # Analyze regional data
  %(prog)s --input california_earthquakes.csv --output analysis/california_patterns.json
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
        help="Output JSON file for analysis results"
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

        # Generate analysis
        generate_summary(df, args.output)

        logger.info("Seismic pattern analysis completed successfully")

    except FileNotFoundError:
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to analyze seismic patterns: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
