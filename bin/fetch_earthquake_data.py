#!/usr/bin/env python3

"""
Fetch earthquake data from USGS Earthquake API.

This script fetches earthquake events from the USGS FDSNWS Event web service,
providing real-time and historical seismic data worldwide.

API Documentation: https://earthquake.usgs.gov/fdsnws/event/1/

Usage:
    python fetch_earthquake_data.py --start-date 2024-01-01 \
                                     --end-date 2024-01-31 \
                                     --min-magnitude 4.0 \
                                     --output earthquake_catalog.csv
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_earthquake_data(
    start_date: datetime,
    end_date: datetime,
    min_magnitude: float = 2.5,
    max_magnitude: float = 10.0,
    min_latitude: Optional[float] = None,
    max_latitude: Optional[float] = None,
    min_longitude: Optional[float] = None,
    max_longitude: Optional[float] = None,
    limit: int = 20000
) -> pd.DataFrame:
    """
    Fetch earthquake data from USGS API.

    Args:
        start_date: Start date for earthquake data
        end_date: End date for earthquake data
        min_magnitude: Minimum magnitude (default: 2.5)
        max_magnitude: Maximum magnitude (default: 10.0)
        min_latitude: Minimum latitude for bounding box
        max_latitude: Maximum latitude for bounding box
        min_longitude: Minimum longitude for bounding box
        max_longitude: Maximum longitude for bounding box
        limit: Maximum number of events to return (default: 20000)

    Returns:
        DataFrame with earthquake events
    """
    base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    params = {
        'format': 'geojson',
        'starttime': start_date.strftime('%Y-%m-%d'),
        'endtime': end_date.strftime('%Y-%m-%d'),
        'minmagnitude': min_magnitude,
        'maxmagnitude': max_magnitude,
        'limit': limit,
        'orderby': 'time'
    }

    # Add bounding box if specified
    if all(v is not None for v in [min_latitude, max_latitude, min_longitude, max_longitude]):
        params['minlatitude'] = min_latitude
        params['maxlatitude'] = max_latitude
        params['minlongitude'] = min_longitude
        params['maxlongitude'] = max_longitude
        logger.info(f"Bounding box: ({min_latitude}, {min_longitude}) to ({max_latitude}, {max_longitude})")

    logger.info(f"Fetching earthquakes from {start_date.date()} to {end_date.date()}")
    logger.info(f"Magnitude range: {min_magnitude} - {max_magnitude}")

    try:
        response = requests.get(base_url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        features = data.get('features', [])
        logger.info(f"Fetched {len(features)} earthquake events")

        if not features:
            logger.warning("No earthquake data found for specified criteria")
            return pd.DataFrame()

        # Parse earthquake data
        earthquakes = []
        for feature in features:
            props = feature.get('properties', {})
            geom = feature.get('geometry', {})
            coords = geom.get('coordinates', [None, None, None])

            earthquakes.append({
                'id': feature.get('id'),
                'time': pd.to_datetime(props.get('time'), unit='ms', utc=True),
                'latitude': coords[1],
                'longitude': coords[0],
                'depth_km': coords[2],
                'magnitude': props.get('mag'),
                'magnitude_type': props.get('magType'),
                'place': props.get('place'),
                'event_type': props.get('type'),
                'status': props.get('status'),
                'tsunami': props.get('tsunami', 0),
                'significance': props.get('sig'),
                'net': props.get('net'),
                'nst': props.get('nst'),  # Number of stations
                'dmin': props.get('dmin'),  # Distance to nearest station
                'rms': props.get('rms'),  # RMS travel time residual
                'gap': props.get('gap'),  # Azimuthal gap
                'url': props.get('url')
            })

        df = pd.DataFrame(earthquakes)

        # Remove events with missing critical data
        initial_count = len(df)
        df = df.dropna(subset=['time', 'latitude', 'longitude', 'magnitude'])
        dropped = initial_count - len(df)
        if dropped > 0:
            logger.warning(f"Dropped {dropped} events with missing critical data")

        # Sort by time
        df = df.sort_values('time').reset_index(drop=True)

        logger.info(f"Final dataset: {len(df)} earthquake events")
        logger.info(f"Magnitude range: {df['magnitude'].min():.1f} - {df['magnitude'].max():.1f}")
        logger.info(f"Depth range: {df['depth_km'].min():.1f} - {df['depth_km'].max():.1f} km")

        return df

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching earthquake data: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status: {e.response.status_code}")
            logger.error(f"Response body: {e.response.text[:500]}")
        raise


def fetch_by_region(
    region_name: str,
    start_date: datetime,
    end_date: datetime,
    min_magnitude: float = 2.5
) -> pd.DataFrame:
    """
    Fetch earthquakes for predefined regions.

    Args:
        region_name: Region name (pacific_ring, california, japan, etc.)
        start_date: Start date
        end_date: End date
        min_magnitude: Minimum magnitude

    Returns:
        DataFrame with earthquake events
    """
    # Predefined regions of seismic interest
    regions = {
        'pacific_ring': {'minlat': -60, 'maxlat': 60, 'minlon': 120, 'maxlon': -60},
        'california': {'minlat': 32, 'maxlat': 42, 'minlon': -125, 'maxlon': -114},
        'japan': {'minlat': 30, 'maxlat': 46, 'minlon': 128, 'maxlon': 146},
        'indonesia': {'minlat': -11, 'maxlat': 6, 'minlon': 95, 'maxlon': 141},
        'turkey': {'minlat': 36, 'maxlat': 42, 'minlon': 26, 'maxlon': 45},
        'chile': {'minlat': -56, 'maxlat': -17, 'minlon': -76, 'maxlon': -66},
        'worldwide': None  # No bounding box
    }

    if region_name not in regions:
        raise ValueError(f"Unknown region: {region_name}. Available: {list(regions.keys())}")

    logger.info(f"Fetching earthquakes for region: {region_name}")

    region_bounds = regions[region_name]
    if region_bounds:
        return fetch_earthquake_data(
            start_date=start_date,
            end_date=end_date,
            min_magnitude=min_magnitude,
            min_latitude=region_bounds['minlat'],
            max_latitude=region_bounds['maxlat'],
            min_longitude=region_bounds['minlon'],
            max_longitude=region_bounds['maxlon']
        )
    else:
        return fetch_earthquake_data(
            start_date=start_date,
            end_date=end_date,
            min_magnitude=min_magnitude
        )


def save_catalog(df: pd.DataFrame, output_file: str):
    """
    Save earthquake catalog to CSV.

    Args:
        df: DataFrame with earthquake data
        output_file: Output file path
    """
    output_path = Path(output_file)

    # Only create parent directories if there's an actual parent path (not current dir)
    if output_path.parent and str(output_path.parent) not in ('.', ''):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_file, index=False)
    logger.info(f"Catalog saved to {output_file}")

    # Calculate summary statistics
    summary = {
        'total_events': len(df),
        'date_range': {
            'start': df['time'].min().isoformat(),
            'end': df['time'].max().isoformat()
        },
        'magnitude': {
            'min': float(df['magnitude'].min()),
            'max': float(df['magnitude'].max()),
            'mean': float(df['magnitude'].mean()),
            'std': float(df['magnitude'].std())
        },
        'depth_km': {
            'min': float(df['depth_km'].min()),
            'max': float(df['depth_km'].max()),
            'mean': float(df['depth_km'].mean())
        },
        'geographic_extent': {
            'min_lat': float(df['latitude'].min()),
            'max_lat': float(df['latitude'].max()),
            'min_lon': float(df['longitude'].min()),
            'max_lon': float(df['longitude'].max())
        },
        'event_types': df['event_type'].value_counts().to_dict(),
        'tsunami_events': int(df['tsunami'].sum())
    }

    # Log summary instead of writing to separate file (avoids undeclared output issues)
    logger.info(f"Summary: {json.dumps(summary, indent=2)}")

    # Print summary
    print(f"\n{'='*70}")
    print("EARTHQUAKE DATA SUMMARY")
    print(f"{'='*70}")
    print(f"Total events: {summary['total_events']}")
    print(f"Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"Magnitude: {summary['magnitude']['min']:.1f} - {summary['magnitude']['max']:.1f} (mean: {summary['magnitude']['mean']:.2f})")
    print(f"Depth: {summary['depth_km']['min']:.1f} - {summary['depth_km']['max']:.1f} km (mean: {summary['depth_km']['mean']:.1f})")
    print(f"Tsunami events: {summary['tsunami_events']}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch earthquake data from USGS API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch all earthquakes M4.0+ in January 2024
  %(prog)s --start-date 2024-01-01 --end-date 2024-01-31 --min-magnitude 4.0

  # Fetch earthquakes in California
  %(prog)s --start-date 2024-01-01 --end-date 2024-01-31 --region california

  # Fetch with custom bounding box
  %(prog)s --start-date 2024-01-01 --end-date 2024-01-31 \\
           --min-lat 30 --max-lat 50 --min-lon -130 --max-lon -110
        """
    )

    parser.add_argument(
        "--start-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        required=True,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        default=None,
        help="End date (YYYY-MM-DD), defaults to start_date + 30 days"
    )
    parser.add_argument(
        "--min-magnitude",
        type=float,
        default=4.0,
        help="Minimum magnitude (default: 4.0)"
    )
    parser.add_argument(
        "--max-magnitude",
        type=float,
        default=10.0,
        help="Maximum magnitude (default: 10.0)"
    )
    parser.add_argument(
        "--region",
        type=str,
        choices=['pacific_ring', 'california', 'japan', 'indonesia', 'turkey', 'chile', 'worldwide'],
        help="Predefined region (overrides bounding box)"
    )
    parser.add_argument(
        "--min-lat",
        type=float,
        help="Minimum latitude"
    )
    parser.add_argument(
        "--max-lat",
        type=float,
        help="Maximum latitude"
    )
    parser.add_argument(
        "--min-lon",
        type=float,
        help="Minimum longitude"
    )
    parser.add_argument(
        "--max-lon",
        type=float,
        help="Maximum longitude"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="earthquake_catalog.csv",
        help="Output CSV file (default: earthquake_catalog.csv)"
    )

    args = parser.parse_args()

    if not args.end_date:
        args.end_date = args.start_date + timedelta(days=30)

    try:
        # Fetch data
        if args.region:
            df = fetch_by_region(
                region_name=args.region,
                start_date=args.start_date,
                end_date=args.end_date,
                min_magnitude=args.min_magnitude
            )
        else:
            df = fetch_earthquake_data(
                start_date=args.start_date,
                end_date=args.end_date,
                min_magnitude=args.min_magnitude,
                max_magnitude=args.max_magnitude,
                min_latitude=args.min_lat,
                max_latitude=args.max_lat,
                min_longitude=args.min_lon,
                max_longitude=args.max_lon
            )

        if df.empty:
            logger.error("No earthquake data fetched. Please check your parameters.")
            sys.exit(1)

        # Save catalog
        save_catalog(df, args.output)

        logger.info("Earthquake data fetch completed successfully")

    except Exception as e:
        logger.error(f"Failed to fetch earthquake data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
