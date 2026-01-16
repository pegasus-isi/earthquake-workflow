#!/usr/bin/env python3

"""
Cluster earthquakes into seismic zones using spatial clustering algorithms.

This script identifies distinct seismic zones by clustering earthquake events
based on their geographic locations. It supports multiple clustering methods:
- DBSCAN: Density-based clustering (good for irregular shapes)
- K-Means: Centroid-based clustering (good for spherical clusters)
- Hierarchical: Agglomerative clustering (good for nested structures)

Each identified zone is characterized by:
- Geographic extent (bounding box, centroid)
- Magnitude distribution statistics
- Depth profile
- Temporal activity patterns
- Event density

Usage:
    python cluster_seismic_zones.py --input earthquake_catalog.csv \
                                    --output zones/seismic_zones.json \
                                    --method dbscan \
                                    --eps 50 --min-samples 10
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth.

    Args:
        lat1, lon1: Coordinates of first point (degrees)
        lat2, lon2: Coordinates of second point (degrees)

    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth's radius in km

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def compute_distance_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Compute pairwise haversine distances between all earthquake locations.

    Args:
        df: DataFrame with 'latitude' and 'longitude' columns

    Returns:
        Distance matrix in kilometers
    """
    coords = df[['latitude', 'longitude']].values
    n = len(coords)
    distances = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_distance(coords[i, 0], coords[i, 1],
                                  coords[j, 0], coords[j, 1])
            distances[i, j] = d
            distances[j, i] = d

    return distances


def cluster_dbscan(df: pd.DataFrame, eps_km: float = 50.0,
                   min_samples: int = 10) -> np.ndarray:
    """
    Cluster earthquakes using DBSCAN algorithm.

    DBSCAN is effective for identifying clusters of arbitrary shape and
    automatically determines the number of clusters.

    Args:
        df: DataFrame with earthquake data
        eps_km: Maximum distance (km) between samples in a cluster
        min_samples: Minimum samples in a neighborhood for core points

    Returns:
        Array of cluster labels (-1 for noise)
    """
    from sklearn.cluster import DBSCAN

    logger.info(f"Running DBSCAN clustering (eps={eps_km}km, min_samples={min_samples})")

    # Compute distance matrix
    logger.info("Computing pairwise distances...")
    distances = compute_distance_matrix(df)

    # Run DBSCAN with precomputed distances
    clustering = DBSCAN(eps=eps_km, min_samples=min_samples, metric='precomputed')
    labels = clustering.fit_predict(distances)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    logger.info(f"DBSCAN found {n_clusters} clusters, {n_noise} noise points")

    return labels


def cluster_kmeans(df: pd.DataFrame, n_clusters: int = 5) -> np.ndarray:
    """
    Cluster earthquakes using K-Means algorithm.

    K-Means partitions data into k spherical clusters. Note: Uses
    Euclidean distance on lat/lon, which is approximate for geographic data.

    Args:
        df: DataFrame with earthquake data
        n_clusters: Number of clusters to form

    Returns:
        Array of cluster labels
    """
    from sklearn.cluster import KMeans

    logger.info(f"Running K-Means clustering (n_clusters={n_clusters})")

    # Scale coordinates for better clustering
    coords = df[['latitude', 'longitude']].values

    # Apply cosine correction for longitude
    mean_lat = np.mean(coords[:, 0])
    scaled_coords = coords.copy()
    scaled_coords[:, 1] = coords[:, 1] * np.cos(np.radians(mean_lat))

    clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = clustering.fit_predict(scaled_coords)

    logger.info(f"K-Means assigned {len(df)} events to {n_clusters} clusters")

    return labels


def cluster_hierarchical(df: pd.DataFrame, n_clusters: Optional[int] = None,
                        distance_threshold: Optional[float] = 100.0) -> np.ndarray:
    """
    Cluster earthquakes using Agglomerative Hierarchical clustering.

    Args:
        df: DataFrame with earthquake data
        n_clusters: Number of clusters (if None, use distance_threshold)
        distance_threshold: Distance threshold for cluster formation (km)

    Returns:
        Array of cluster labels
    """
    from sklearn.cluster import AgglomerativeClustering

    logger.info("Running Hierarchical clustering")

    # Compute distance matrix
    logger.info("Computing pairwise distances...")
    distances = compute_distance_matrix(df)

    if n_clusters is not None:
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric='precomputed',
            linkage='average'
        )

    labels = clustering.fit_predict(distances)
    n_found = len(set(labels))
    logger.info(f"Hierarchical clustering found {n_found} clusters")

    return labels


def characterize_zone(df: pd.DataFrame, zone_id: int) -> Dict:
    """
    Generate comprehensive statistics for a seismic zone.

    Args:
        df: DataFrame with earthquake data for the zone
        zone_id: Identifier for the zone

    Returns:
        Dictionary with zone characteristics
    """
    zone = {
        'zone_id': int(zone_id),
        'num_events': int(len(df)),
    }

    # Geographic extent
    zone['geographic_extent'] = {
        'centroid': {
            'latitude': float(df['latitude'].mean()),
            'longitude': float(df['longitude'].mean())
        },
        'bounding_box': {
            'min_latitude': float(df['latitude'].min()),
            'max_latitude': float(df['latitude'].max()),
            'min_longitude': float(df['longitude'].min()),
            'max_longitude': float(df['longitude'].max())
        },
        'area_sq_km': _estimate_area(df)
    }

    # Magnitude statistics
    if 'magnitude' in df.columns and not df['magnitude'].isna().all():
        mags = df['magnitude'].dropna()
        zone['magnitude_stats'] = {
            'min': float(mags.min()),
            'max': float(mags.max()),
            'mean': float(mags.mean()),
            'median': float(mags.median()),
            'std': float(mags.std()),
            'count_by_range': {
                '3.0-3.9': int(((mags >= 3.0) & (mags < 4.0)).sum()),
                '4.0-4.9': int(((mags >= 4.0) & (mags < 5.0)).sum()),
                '5.0-5.9': int(((mags >= 5.0) & (mags < 6.0)).sum()),
                '6.0+': int((mags >= 6.0).sum())
            }
        }

        # Estimate Gutenberg-Richter b-value
        try:
            b_value = _estimate_b_value(mags)
            zone['magnitude_stats']['b_value'] = b_value
        except Exception:
            pass

    # Depth statistics
    if 'depth_km' in df.columns and not df['depth_km'].isna().all():
        depths = df['depth_km'].dropna()
        zone['depth_stats'] = {
            'min_km': float(depths.min()),
            'max_km': float(depths.max()),
            'mean_km': float(depths.mean()),
            'median_km': float(depths.median()),
            'std_km': float(depths.std()),
            'classification': _classify_depth(depths.mean())
        }

    # Temporal statistics
    if 'time' in df.columns:
        df_temp = df.copy()
        df_temp['time'] = pd.to_datetime(df_temp['time'], format='mixed', utc=True)
        times = df_temp['time'].dropna()

        if len(times) > 0:
            zone['temporal_stats'] = {
                'first_event': times.min().isoformat(),
                'last_event': times.max().isoformat(),
                'duration_days': float((times.max() - times.min()).total_seconds() / 86400),
                'events_per_day': float(len(times) / max(1, (times.max() - times.min()).total_seconds() / 86400))
            }

    # Event density
    if zone['geographic_extent']['area_sq_km'] > 0:
        zone['event_density_per_1000_sq_km'] = float(
            zone['num_events'] / zone['geographic_extent']['area_sq_km'] * 1000
        )

    return zone


def _estimate_area(df: pd.DataFrame) -> float:
    """Estimate the area covered by earthquakes in square kilometers."""
    lat_range = df['latitude'].max() - df['latitude'].min()
    lon_range = df['longitude'].max() - df['longitude'].min()
    mean_lat = df['latitude'].mean()

    # Convert to km (approximate)
    lat_km = lat_range * 111
    lon_km = lon_range * 111 * np.cos(np.radians(mean_lat))

    return float(lat_km * lon_km)


def _estimate_b_value(magnitudes: pd.Series) -> Optional[Dict]:
    """
    Estimate Gutenberg-Richter b-value using maximum likelihood.

    b-value describes the relative frequency of large vs small earthquakes.
    Typical values are around 1.0.
    """
    mags = magnitudes.dropna().values
    if len(mags) < 20:
        return None

    # Estimate magnitude of completeness (Mc) using maximum curvature
    bins = np.arange(mags.min(), mags.max() + 0.1, 0.1)
    hist, edges = np.histogram(mags, bins=bins)
    mc_idx = np.argmax(hist)
    mc = edges[mc_idx]

    # Filter events above Mc
    mags_complete = mags[mags >= mc]
    if len(mags_complete) < 10:
        return None

    # Maximum likelihood b-value (Aki, 1965)
    mean_mag = np.mean(mags_complete)
    b_value = np.log10(np.e) / (mean_mag - mc + 0.05)

    # Standard error
    n = len(mags_complete)
    b_std = b_value / np.sqrt(n)

    return {
        'value': float(b_value),
        'std_error': float(b_std),
        'magnitude_completeness': float(mc),
        'n_events_used': int(n)
    }


def _classify_depth(mean_depth: float) -> str:
    """Classify seismic zone by mean depth."""
    if mean_depth < 20:
        return 'shallow'
    elif mean_depth < 70:
        return 'intermediate_shallow'
    elif mean_depth < 300:
        return 'intermediate'
    else:
        return 'deep'


def generate_zone_report(df: pd.DataFrame, labels: np.ndarray,
                        method: str, output_file: str, params: Dict):
    """
    Generate comprehensive seismic zone clustering report.

    Args:
        df: DataFrame with earthquake data
        labels: Cluster labels for each event
        method: Clustering method used
        output_file: Output JSON file path
        params: Clustering parameters used
    """
    logger.info("Generating seismic zone report...")

    # Add cluster labels to dataframe
    df = df.copy()
    df['zone_id'] = labels

    # Identify unique zones (excluding noise if DBSCAN)
    unique_zones = sorted([z for z in set(labels) if z >= 0])

    report = {
        'metadata': {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'total_events': int(len(df)),
            'clustering_method': method,
            'clustering_params': params,
            'num_zones': len(unique_zones),
            'num_noise_events': int((labels == -1).sum()) if -1 in labels else 0
        },
        'zones': []
    }

    # Characterize each zone
    for zone_id in unique_zones:
        zone_df = df[df['zone_id'] == zone_id]
        zone_stats = characterize_zone(zone_df, zone_id)
        report['zones'].append(zone_stats)

    # Sort zones by number of events (descending)
    report['zones'].sort(key=lambda x: x['num_events'], reverse=True)

    # Add summary statistics
    report['summary'] = {
        'largest_zone': {
            'zone_id': report['zones'][0]['zone_id'] if report['zones'] else None,
            'num_events': report['zones'][0]['num_events'] if report['zones'] else 0
        },
        'smallest_zone': {
            'zone_id': report['zones'][-1]['zone_id'] if report['zones'] else None,
            'num_events': report['zones'][-1]['num_events'] if report['zones'] else 0
        },
        'avg_events_per_zone': float(np.mean([z['num_events'] for z in report['zones']])) if report['zones'] else 0,
        'zones_with_large_events': sum(
            1 for z in report['zones']
            if 'magnitude_stats' in z and z['magnitude_stats'].get('max', 0) >= 5.0
        )
    }

    # Add zone event IDs for reference
    if 'id' in df.columns:
        for zone in report['zones']:
            zone_df = df[df['zone_id'] == zone['zone_id']]
            # Convert to native Python types for JSON serialization
            event_ids = zone_df['id'].tolist()[:100]  # Limit to 100 IDs
            zone['event_ids'] = [str(eid) if not isinstance(eid, str) else eid for eid in event_ids]

    # Save to file
    output_path = Path(output_file)
    if output_path.parent and str(output_path.parent) not in ('.', ''):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Zone report saved to {output_file}")

    # Print summary
    print(f"\n{'='*70}")
    print("SEISMIC ZONE CLUSTERING SUMMARY")
    print(f"{'='*70}")
    print(f"Clustering Method: {method.upper()}")
    print(f"Total Events: {len(df)}")
    print(f"Zones Identified: {len(unique_zones)}")
    if -1 in labels:
        print(f"Noise Events: {(labels == -1).sum()}")
    print(f"\nZone Statistics:")
    for i, zone in enumerate(report['zones'][:5]):  # Show top 5 zones
        print(f"  Zone {zone['zone_id']}: {zone['num_events']} events", end='')
        if 'magnitude_stats' in zone:
            print(f" (M{zone['magnitude_stats']['min']:.1f}-{zone['magnitude_stats']['max']:.1f})", end='')
        if 'depth_stats' in zone:
            print(f" [{zone['depth_stats']['classification']}]", end='')
        print()
    if len(report['zones']) > 5:
        print(f"  ... and {len(report['zones']) - 5} more zones")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Cluster earthquakes into seismic zones",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # DBSCAN clustering (density-based, auto-determines clusters)
  %(prog)s --input earthquakes.csv --output zones.json --method dbscan --eps 50 --min-samples 10

  # K-Means clustering (specify number of clusters)
  %(prog)s --input earthquakes.csv --output zones.json --method kmeans --n-clusters 8

  # Hierarchical clustering with distance threshold
  %(prog)s --input earthquakes.csv --output zones.json --method hierarchical --distance-threshold 75

Clustering Methods:
  dbscan       - Density-based (good for irregular shapes, finds outliers)
  kmeans       - Centroid-based (good for spherical clusters)
  hierarchical - Agglomerative (good for nested structures)
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
        help="Output JSON file for zone report"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=['dbscan', 'kmeans', 'hierarchical'],
        default='dbscan',
        help="Clustering method (default: dbscan)"
    )

    # DBSCAN parameters
    parser.add_argument(
        "--eps",
        type=float,
        default=50.0,
        help="DBSCAN: Maximum distance (km) between samples (default: 50)"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="DBSCAN: Minimum samples for core points (default: 10)"
    )

    # K-Means parameters
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=5,
        help="K-Means/Hierarchical: Number of clusters (default: 5)"
    )

    # Hierarchical parameters
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=None,
        help="Hierarchical: Distance threshold in km (overrides n-clusters)"
    )

    args = parser.parse_args()

    try:
        # Load earthquake data
        logger.info(f"Loading earthquake data from {args.input}")
        df = pd.read_csv(args.input)

        if df.empty:
            logger.error("Input file is empty")
            sys.exit(1)

        # Validate required columns
        required_cols = ['latitude', 'longitude']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            sys.exit(1)

        logger.info(f"Loaded {len(df)} earthquake events")

        # Remove rows with missing coordinates
        df = df.dropna(subset=['latitude', 'longitude'])
        logger.info(f"After removing missing coordinates: {len(df)} events")

        if len(df) < 10:
            logger.error("Insufficient data for clustering (need at least 10 events)")
            sys.exit(1)

        # Run clustering
        params = {}
        if args.method == 'dbscan':
            params = {'eps_km': args.eps, 'min_samples': args.min_samples}
            labels = cluster_dbscan(df, eps_km=args.eps, min_samples=args.min_samples)

        elif args.method == 'kmeans':
            params = {'n_clusters': args.n_clusters}
            labels = cluster_kmeans(df, n_clusters=args.n_clusters)

        elif args.method == 'hierarchical':
            if args.distance_threshold:
                params = {'distance_threshold_km': args.distance_threshold}
                labels = cluster_hierarchical(df, distance_threshold=args.distance_threshold)
            else:
                params = {'n_clusters': args.n_clusters}
                labels = cluster_hierarchical(df, n_clusters=args.n_clusters)

        # Generate report
        generate_zone_report(df, labels, args.method, args.output, params)

        logger.info("Seismic zone clustering completed successfully")

    except FileNotFoundError:
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    except ImportError as e:
        logger.error(f"Missing required package: {e}")
        logger.error("Install with: pip install scikit-learn")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to cluster seismic zones: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
