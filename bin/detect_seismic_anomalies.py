#!/usr/bin/env python3

"""
Detect seismic anomalies in earthquake data.

This script identifies unusual seismic activity patterns including:
- Earthquake swarms (temporal and spatial clustering)
- Mainshock-aftershock sequences
- Unusual magnitude increases
- Depth anomalies
- Sudden changes in seismic rate

Usage:
    python detect_seismic_anomalies.py --input earthquake_catalog.csv \
                                       --output anomalies/anomalies.json
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
from scipy.spatial.distance import cdist

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_swarms(df: pd.DataFrame, time_window_hours: int = 24,
                 spatial_radius_km: int = 50, min_events: int = 10) -> List[Dict]:
    """
    Detect earthquake swarms (clustered events in time and space).

    Args:
        df: DataFrame with earthquake data
        time_window_hours: Time window for clustering (hours)
        spatial_radius_km: Spatial radius for clustering (km)
        min_events: Minimum events to qualify as swarm

    Returns:
        List of detected swarms
    """
    logger.info("Detecting earthquake swarms...")

    if 'time' not in df.columns or 'latitude' not in df.columns or 'longitude' not in df.columns:
        logger.warning("Missing required columns for swarm detection")
        return []

    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], format='mixed', utc=True)
    df = df.sort_values('time').reset_index(drop=True)

    swarms = []
    processed = set()

    for i in range(len(df)):
        if i in processed:
            continue

        event = df.iloc[i]
        event_time = event['time']
        event_lat = event['latitude']
        event_lon = event['longitude']

        # Find events within time and space window
        time_mask = (df['time'] >= event_time) & \
                   (df['time'] <= event_time + timedelta(hours=time_window_hours))

        # Calculate distances (rough approximation)
        lat_diff = (df['latitude'] - event_lat) * 111  # km per degree
        lon_diff = (df['longitude'] - event_lon) * 111 * np.cos(np.radians(event_lat))
        distances = np.sqrt(lat_diff**2 + lon_diff**2)

        spatial_mask = distances <= spatial_radius_km
        cluster_mask = time_mask & spatial_mask

        cluster_indices = df[cluster_mask].index.tolist()

        if len(cluster_indices) >= min_events:
            cluster_df = df.loc[cluster_indices]

            swarm = {
                'swarm_id': len(swarms) + 1,
                'start_time': cluster_df['time'].min().isoformat(),
                'end_time': cluster_df['time'].max().isoformat(),
                'duration_hours': float((cluster_df['time'].max() - cluster_df['time'].min()).total_seconds() / 3600),
                'num_events': int(len(cluster_indices)),
                'center_latitude': float(cluster_df['latitude'].mean()),
                'center_longitude': float(cluster_df['longitude'].mean()),
                'magnitude_range': {
                    'min': float(cluster_df['magnitude'].min()),
                    'max': float(cluster_df['magnitude'].max()),
                    'mean': float(cluster_df['magnitude'].mean())
                },
                'depth_range_km': {
                    'min': float(cluster_df['depth_km'].min()),
                    'max': float(cluster_df['depth_km'].max()),
                    'mean': float(cluster_df['depth_km'].mean())
                },
                'event_ids': cluster_df['id'].tolist() if 'id' in cluster_df.columns else []
            }

            swarms.append(swarm)
            processed.update(cluster_indices)

    logger.info(f"Detected {len(swarms)} earthquake swarms")
    return swarms


def detect_mainshock_aftershock_sequences(df: pd.DataFrame,
                                         magnitude_threshold: float = 5.0,
                                         time_window_days: int = 30,
                                         spatial_radius_km: int = 100) -> List[Dict]:
    """
    Identify mainshock-aftershock sequences.

    A mainshock is defined as:
    - Magnitude >= threshold
    - Largest magnitude in local time-space window

    Aftershocks are:
    - Events within time and space window after mainshock
    - Magnitude < mainshock magnitude

    Args:
        df: DataFrame with earthquake data
        magnitude_threshold: Minimum magnitude for mainshock
        time_window_days: Time window for aftershocks
        spatial_radius_km: Spatial radius for aftershocks

    Returns:
        List of mainshock-aftershock sequences
    """
    logger.info("Detecting mainshock-aftershock sequences...")

    if 'magnitude' not in df.columns or 'time' not in df.columns:
        logger.warning("Missing required columns for sequence detection")
        return []

    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], format='mixed', utc=True)
    df = df.sort_values('time').reset_index(drop=True)

    # Find potential mainshocks
    mainshock_candidates = df[df['magnitude'] >= magnitude_threshold].copy()

    sequences = []

    for idx, mainshock in mainshock_candidates.iterrows():
        mainshock_time = mainshock['time']
        mainshock_mag = mainshock['magnitude']
        mainshock_lat = mainshock['latitude']
        mainshock_lon = mainshock['longitude']

        # Find aftershocks
        time_mask = (df['time'] > mainshock_time) & \
                   (df['time'] <= mainshock_time + timedelta(days=time_window_days))

        # Calculate distances
        lat_diff = (df['latitude'] - mainshock_lat) * 111
        lon_diff = (df['longitude'] - mainshock_lon) * 111 * np.cos(np.radians(mainshock_lat))
        distances = np.sqrt(lat_diff**2 + lon_diff**2)

        spatial_mask = distances <= spatial_radius_km
        magnitude_mask = df['magnitude'] < mainshock_mag

        aftershock_mask = time_mask & spatial_mask & magnitude_mask
        aftershocks = df[aftershock_mask]

        if len(aftershocks) >= 5:  # Require at least 5 aftershocks
            # Calculate Omori's law parameters (rate = K / (c + t)^p)
            time_diffs = (aftershocks['time'] - mainshock_time).dt.total_seconds() / 86400  # days

            sequence = {
                'mainshock_id': mainshock['id'] if 'id' in mainshock else None,
                'mainshock_time': mainshock_time.isoformat(),
                'mainshock_magnitude': float(mainshock_mag),
                'mainshock_latitude': float(mainshock_lat),
                'mainshock_longitude': float(mainshock_lon),
                'mainshock_depth_km': float(mainshock['depth_km']) if 'depth_km' in mainshock else None,
                'num_aftershocks': int(len(aftershocks)),
                'aftershock_duration_days': float(time_diffs.max()),
                'largest_aftershock_magnitude': float(aftershocks['magnitude'].max()),
                'aftershock_magnitude_range': {
                    'min': float(aftershocks['magnitude'].min()),
                    'max': float(aftershocks['magnitude'].max()),
                    'mean': float(aftershocks['magnitude'].mean())
                },
                'aftershock_rate_decay': {
                    'first_day': int(len(aftershocks[time_diffs <= 1])),
                    'first_week': int(len(aftershocks[time_diffs <= 7])),
                    'first_month': int(len(aftershocks[time_diffs <= 30]))
                }
            }

            sequences.append(sequence)

    logger.info(f"Detected {len(sequences)} mainshock-aftershock sequences")
    return sequences


def detect_magnitude_anomalies(df: pd.DataFrame, z_threshold: float = 2.5) -> List[Dict]:
    """
    Detect unusual magnitude patterns using statistical methods.

    Args:
        df: DataFrame with earthquake data
        z_threshold: Z-score threshold for anomaly detection

    Returns:
        List of magnitude anomalies
    """
    logger.info("Detecting magnitude anomalies...")

    if 'magnitude' not in df.columns or 'time' not in df.columns:
        logger.warning("Missing required columns")
        return []

    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], format='mixed', utc=True)
    df = df.sort_values('time')

    # Calculate rolling statistics
    window_size = min(50, len(df) // 4)  # Adaptive window size
    if window_size < 10:
        logger.warning("Insufficient data for magnitude anomaly detection")
        return []

    df['mag_rolling_mean'] = df['magnitude'].rolling(window=window_size, center=True).mean()
    df['mag_rolling_std'] = df['magnitude'].rolling(window=window_size, center=True).std()

    # Calculate z-scores
    df['mag_z_score'] = (df['magnitude'] - df['mag_rolling_mean']) / df['mag_rolling_std']

    # Find anomalies
    anomalies_df = df[np.abs(df['mag_z_score']) > z_threshold].copy()

    anomalies = []
    for idx, row in anomalies_df.iterrows():
        anomaly = {
            'event_id': row['id'] if 'id' in row else None,
            'time': row['time'].isoformat(),
            'magnitude': float(row['magnitude']),
            'z_score': float(row['mag_z_score']),
            'rolling_mean': float(row['mag_rolling_mean']),
            'rolling_std': float(row['mag_rolling_std']),
            'latitude': float(row['latitude']) if 'latitude' in row else None,
            'longitude': float(row['longitude']) if 'longitude' in row else None,
            'depth_km': float(row['depth_km']) if 'depth_km' in row else None,
            'anomaly_type': 'unusually_large' if row['mag_z_score'] > 0 else 'unusually_small'
        }
        anomalies.append(anomaly)

    logger.info(f"Detected {len(anomalies)} magnitude anomalies")
    return anomalies


def detect_rate_changes(df: pd.DataFrame, window_hours: int = 24,
                       threshold_factor: float = 3.0) -> List[Dict]:
    """
    Detect sudden changes in earthquake rate.

    Args:
        df: DataFrame with earthquake data
        window_hours: Time window for rate calculation (hours)
        threshold_factor: Factor above baseline for anomaly

    Returns:
        List of rate change anomalies
    """
    logger.info("Detecting seismic rate changes...")

    if 'time' not in df.columns:
        logger.warning("No time data available")
        return []

    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], format='mixed', utc=True)
    df = df.sort_values('time')

    # Calculate hourly event counts
    df['hour'] = df['time'].dt.floor('H')
    hourly_counts = df.groupby('hour').size()

    # Calculate rolling baseline
    window_size = max(24, window_hours)  # At least 24 hours
    baseline = hourly_counts.rolling(window=window_size, center=True).mean()
    std = hourly_counts.rolling(window=window_size, center=True).std()

    # Detect anomalies
    rate_changes = []

    for hour, count in hourly_counts.items():
        if pd.isna(baseline[hour]) or pd.isna(std[hour]):
            continue

        if std[hour] == 0:
            continue

        z_score = (count - baseline[hour]) / std[hour]

        if z_score > threshold_factor:
            # Get events in this hour
            hour_events = df[df['hour'] == hour]

            rate_change = {
                'time': hour.isoformat(),
                'event_count': int(count),
                'baseline_rate': float(baseline[hour]),
                'z_score': float(z_score),
                'rate_increase_factor': float(count / baseline[hour]),
                'num_events': int(len(hour_events)),
                'magnitude_range': {
                    'min': float(hour_events['magnitude'].min()),
                    'max': float(hour_events['magnitude'].max()),
                    'mean': float(hour_events['magnitude'].mean())
                } if 'magnitude' in hour_events.columns else None,
                'spatial_extent': {
                    'latitude_range': [float(hour_events['latitude'].min()),
                                      float(hour_events['latitude'].max())],
                    'longitude_range': [float(hour_events['longitude'].min()),
                                       float(hour_events['longitude'].max())]
                } if 'latitude' in hour_events.columns else None
            }

            rate_changes.append(rate_change)

    logger.info(f"Detected {len(rate_changes)} rate change anomalies")
    return rate_changes


def detect_depth_anomalies(df: pd.DataFrame, z_threshold: float = 2.5) -> List[Dict]:
    """
    Detect unusual depth patterns.

    Args:
        df: DataFrame with earthquake data
        z_threshold: Z-score threshold for anomaly detection

    Returns:
        List of depth anomalies
    """
    logger.info("Detecting depth anomalies...")

    if 'depth_km' not in df.columns or 'time' not in df.columns:
        logger.warning("Missing required columns")
        return []

    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], format='mixed', utc=True)
    df = df.sort_values('time')

    # Calculate rolling statistics
    window_size = min(50, len(df) // 4)
    if window_size < 10:
        logger.warning("Insufficient data for depth anomaly detection")
        return []

    df['depth_rolling_mean'] = df['depth_km'].rolling(window=window_size, center=True).mean()
    df['depth_rolling_std'] = df['depth_km'].rolling(window=window_size, center=True).std()

    # Calculate z-scores
    df['depth_z_score'] = (df['depth_km'] - df['depth_rolling_mean']) / df['depth_rolling_std']

    # Find anomalies
    anomalies_df = df[np.abs(df['depth_z_score']) > z_threshold].copy()

    anomalies = []
    for idx, row in anomalies_df.iterrows():
        anomaly = {
            'event_id': row['id'] if 'id' in row else None,
            'time': row['time'].isoformat(),
            'depth_km': float(row['depth_km']),
            'z_score': float(row['depth_z_score']),
            'rolling_mean': float(row['depth_rolling_mean']),
            'rolling_std': float(row['depth_rolling_std']),
            'magnitude': float(row['magnitude']) if 'magnitude' in row else None,
            'latitude': float(row['latitude']) if 'latitude' in row else None,
            'longitude': float(row['longitude']) if 'longitude' in row else None,
            'anomaly_type': 'unusually_deep' if row['depth_z_score'] > 0 else 'unusually_shallow'
        }
        anomalies.append(anomaly)

    logger.info(f"Detected {len(anomalies)} depth anomalies")
    return anomalies


def generate_anomaly_report(df: pd.DataFrame, output_file: str):
    """
    Generate comprehensive seismic anomaly report.

    Args:
        df: DataFrame with earthquake data
        output_file: Output JSON file path
    """
    logger.info(f"Generating seismic anomaly report for {len(df)} events...")

    report = {
        'metadata': {
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'total_events': int(len(df)),
            'analysis_version': '1.0'
        },
        'swarms': detect_swarms(df),
        'mainshock_aftershock_sequences': detect_mainshock_aftershock_sequences(df),
        'magnitude_anomalies': detect_magnitude_anomalies(df),
        'rate_changes': detect_rate_changes(df),
        'depth_anomalies': detect_depth_anomalies(df)
    }

    # Summary statistics
    report['summary'] = {
        'total_swarms': len(report['swarms']),
        'total_sequences': len(report['mainshock_aftershock_sequences']),
        'total_magnitude_anomalies': len(report['magnitude_anomalies']),
        'total_rate_changes': len(report['rate_changes']),
        'total_depth_anomalies': len(report['depth_anomalies']),
        'total_anomalous_events': sum([
            len(report['magnitude_anomalies']),
            len(report['depth_anomalies'])
        ])
    }

    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Anomaly report saved to {output_file}")

    # Print summary
    print(f"\n{'='*70}")
    print("SEISMIC ANOMALY DETECTION SUMMARY")
    print(f"{'='*70}")
    print(f"Total Events Analyzed: {len(df)}")
    print(f"\nDetected Anomalies:")
    print(f"  Earthquake Swarms: {report['summary']['total_swarms']}")
    print(f"  Mainshock-Aftershock Sequences: {report['summary']['total_sequences']}")
    print(f"  Magnitude Anomalies: {report['summary']['total_magnitude_anomalies']}")
    print(f"  Rate Change Events: {report['summary']['total_rate_changes']}")
    print(f"  Depth Anomalies: {report['summary']['total_depth_anomalies']}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Detect seismic anomalies in earthquake data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect anomalies in catalog
  %(prog)s --input earthquake_catalog.csv --output anomalies/anomalies.json

  # With custom thresholds
  %(prog)s --input california_2024.csv --output anomalies/california_anomalies.json
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
        help="Output JSON file for anomaly report"
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

        # Generate anomaly report
        generate_anomaly_report(df, args.output)

        logger.info("Seismic anomaly detection completed successfully")

    except FileNotFoundError:
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to detect seismic anomalies: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
