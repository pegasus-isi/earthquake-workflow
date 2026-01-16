#!/usr/bin/env python3
"""
Seismic Gap Analysis Tool

Identifies seismic gaps - regions with historical seismicity but anomalous
recent quiescence, indicating potential strain accumulation and elevated
future earthquake risk.

Features:
- Temporal rate comparison (historical vs recent)
- Grid-based spatial analysis
- Statistical significance testing (Poisson)
- Moment deficit estimation
- Potential magnitude calculation
- Risk scoring and classification

Usage:
    # Basic gap analysis
    python analyze_seismic_gaps.py --input catalog.csv --output gaps.json

    # With custom time periods
    python analyze_seismic_gaps.py --input catalog.csv --output gaps.json \\
        --historical-years 30 --recent-years 5 --grid-resolution 0.5

    # With significance threshold
    python analyze_seismic_gaps.py --input catalog.csv --output gaps.json \\
        --rate-threshold 0.3 --min-significance 0.95
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Earth radius in km
EARTH_RADIUS_KM = 6371.0

# Risk score weights
RISK_WEIGHTS = {
    'rate_ratio': 0.25,      # Lower ratio = higher risk
    'historical_mag': 0.20,   # Higher historical max mag = higher risk
    'time_since_large': 0.20, # Longer time = higher risk
    'significance': 0.15,     # Higher significance = higher risk
    'moment_deficit': 0.20    # Higher deficit = higher risk
}

# Risk level thresholds
RISK_THRESHOLDS = {
    'critical': 0.85,
    'high': 0.65,
    'moderate': 0.45,
    'low': 0.25
}


# =============================================================================
# Utility Functions
# =============================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points in km."""
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return EARTH_RADIUS_KM * c


def calculate_cell_area(lat: float, resolution: float) -> float:
    """
    Calculate approximate area of a grid cell in sq km.

    Args:
        lat: Center latitude of cell
        resolution: Cell size in degrees

    Returns:
        Area in square kilometers
    """
    # Approximate degrees to km at given latitude
    lat_km = 111.0  # km per degree latitude
    lon_km = 111.0 * np.cos(np.radians(lat))  # km per degree longitude

    return lat_km * lon_km * resolution**2


def magnitude_to_moment(magnitude: float) -> float:
    """Convert moment magnitude to seismic moment (N·m)."""
    return 10**(1.5 * magnitude + 9.1)


def moment_to_magnitude(moment: float) -> float:
    """Convert seismic moment (N·m) to moment magnitude."""
    return (np.log10(moment) - 9.1) / 1.5


def calculate_b_value(magnitudes: np.ndarray, mc: Optional[float] = None) -> Tuple[float, float]:
    """
    Calculate Gutenberg-Richter b-value using maximum likelihood.

    Args:
        magnitudes: Array of magnitudes
        mc: Magnitude of completeness (auto-estimated if None)

    Returns:
        Tuple of (b-value, standard error)
    """
    if len(magnitudes) < 10:
        return 1.0, 0.5  # Default with high uncertainty

    if mc is None:
        # Estimate Mc using maximum curvature method
        hist, bin_edges = np.histogram(magnitudes, bins=np.arange(0, 10, 0.1))
        mc = bin_edges[np.argmax(hist)]

    # Filter magnitudes above completeness
    mags_complete = magnitudes[magnitudes >= mc]

    if len(mags_complete) < 10:
        return 1.0, 0.5

    # Aki's (1965) maximum likelihood estimator
    mean_mag = np.mean(mags_complete)
    b_value = np.log10(np.e) / (mean_mag - (mc - 0.05))

    # Standard error (Shi & Bolt, 1982)
    n = len(mags_complete)
    std_error = 2.3 * b_value**2 * np.sqrt(np.sum((mags_complete - mean_mag)**2) / (n * (n - 1)))

    return max(0.5, min(2.0, b_value)), std_error


# =============================================================================
# Temporal Analysis
# =============================================================================

def split_catalog_by_time(df: pd.DataFrame, historical_years: float,
                          recent_years: float) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Split catalog into historical and recent periods.

    Args:
        df: Earthquake catalog with 'time' column
        historical_years: Length of historical period
        recent_years: Length of recent period

    Returns:
        Tuple of (historical_df, recent_df, period_info)
    """
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], format='mixed', utc=True)
    df = df.sort_values('time')

    # Define time boundaries
    end_time = df['time'].max()
    recent_start = end_time - timedelta(days=recent_years * 365.25)
    historical_end = recent_start
    historical_start = historical_end - timedelta(days=historical_years * 365.25)

    # Split catalog
    historical_df = df[(df['time'] >= historical_start) & (df['time'] < historical_end)]
    recent_df = df[df['time'] >= recent_start]

    period_info = {
        'historical': {
            'start': historical_start.isoformat(),
            'end': historical_end.isoformat(),
            'years': historical_years,
            'n_events': len(historical_df)
        },
        'recent': {
            'start': recent_start.isoformat(),
            'end': end_time.isoformat(),
            'years': recent_years,
            'n_events': len(recent_df)
        }
    }

    logger.info(f"Historical period: {len(historical_df)} events over {historical_years} years")
    logger.info(f"Recent period: {len(recent_df)} events over {recent_years} years")

    return historical_df, recent_df, period_info


# =============================================================================
# Spatial Gridding
# =============================================================================

def create_analysis_grid(df: pd.DataFrame, resolution: float,
                         buffer: float = 0.5) -> List[Dict[str, float]]:
    """
    Create a grid of analysis cells covering the earthquake region.

    Args:
        df: Earthquake catalog
        resolution: Grid spacing in degrees
        buffer: Buffer around data extent in degrees

    Returns:
        List of grid cell dictionaries
    """
    min_lat = df['latitude'].min() - buffer
    max_lat = df['latitude'].max() + buffer
    min_lon = df['longitude'].min() - buffer
    max_lon = df['longitude'].max() + buffer

    # Create grid
    lats = np.arange(min_lat, max_lat + resolution, resolution)
    lons = np.arange(min_lon, max_lon + resolution, resolution)

    grid_cells = []
    for lat in lats:
        for lon in lons:
            area = calculate_cell_area(lat + resolution/2, resolution)
            grid_cells.append({
                'lat': float(lat),
                'lon': float(lon),
                'lat_max': float(lat + resolution),
                'lon_max': float(lon + resolution),
                'area_sq_km': float(area)
            })

    logger.info(f"Created analysis grid: {len(lats)} x {len(lons)} = {len(grid_cells)} cells")
    return grid_cells


def count_events_in_cell(df: pd.DataFrame, cell: Dict[str, float]) -> pd.DataFrame:
    """Count events within a grid cell."""
    mask = (
        (df['latitude'] >= cell['lat']) &
        (df['latitude'] < cell['lat_max']) &
        (df['longitude'] >= cell['lon']) &
        (df['longitude'] < cell['lon_max'])
    )
    return df[mask]


# =============================================================================
# Gap Detection
# =============================================================================

def calculate_poisson_significance(historical_count: int, historical_years: float,
                                   recent_count: int, recent_years: float) -> float:
    """
    Calculate statistical significance of rate decrease using Poisson test.

    Returns probability that the recent rate decrease is NOT due to chance
    (i.e., 1 - p-value for one-sided test).
    """
    if historical_count == 0 or historical_years <= 0:
        return 0.0

    # Expected count in recent period based on historical rate
    historical_rate = historical_count / historical_years
    expected_recent = historical_rate * recent_years

    if expected_recent <= 0:
        return 0.0

    # Poisson probability of observing <= recent_count given expected_recent
    # Low p-value means significant decrease
    p_value = stats.poisson.cdf(recent_count, expected_recent)

    # Return significance (1 - p_value)
    return 1 - p_value


def estimate_moment_deficit(historical_df: pd.DataFrame, recent_df: pd.DataFrame,
                            recent_years: float) -> Tuple[float, float]:
    """
    Estimate seismic moment deficit based on historical vs recent release.

    Args:
        historical_df: Historical period events
        recent_df: Recent period events
        recent_years: Length of recent period

    Returns:
        Tuple of (moment_deficit in N·m, potential_magnitude)
    """
    if len(historical_df) == 0:
        return 0.0, 0.0

    # Calculate historical moment release rate
    historical_moments = [magnitude_to_moment(m) for m in historical_df['magnitude']]
    total_historical_moment = sum(historical_moments)

    # Calculate expected vs actual recent moment
    expected_rate = total_historical_moment / len(historical_df) * (len(historical_df) / (historical_df['time'].max() - historical_df['time'].min()).days * 365.25)
    expected_recent_moment = expected_rate * recent_years

    if len(recent_df) > 0:
        recent_moments = [magnitude_to_moment(m) for m in recent_df['magnitude']]
        actual_recent_moment = sum(recent_moments)
    else:
        actual_recent_moment = 0.0

    # Moment deficit
    moment_deficit = max(0, expected_recent_moment - actual_recent_moment)

    # Potential magnitude from accumulated deficit
    if moment_deficit > 0:
        potential_mag = moment_to_magnitude(moment_deficit)
    else:
        potential_mag = 0.0

    return moment_deficit, potential_mag


def calculate_risk_score(rate_ratio: float, historical_max_mag: float,
                         days_since_large: int, significance: float,
                         moment_deficit: float, historical_mean_mag: float = 4.0) -> float:
    """
    Calculate composite risk score for a seismic gap.

    Args:
        rate_ratio: Recent/historical rate ratio (lower = higher risk)
        historical_max_mag: Maximum magnitude in historical period
        days_since_large: Days since last M5+ event
        significance: Statistical significance of rate decrease
        moment_deficit: Accumulated moment deficit
        historical_mean_mag: Mean magnitude in historical period

    Returns:
        Risk score between 0 and 1
    """
    # Normalize each factor to 0-1 range

    # Rate ratio (lower = higher risk): 1 - ratio, capped at 0-1
    rate_score = max(0, min(1, 1 - rate_ratio))

    # Historical max magnitude (higher = higher risk): normalize to typical range
    mag_score = max(0, min(1, (historical_max_mag - 4.0) / 4.0))

    # Time since large event (longer = higher risk): normalize by years
    time_score = max(0, min(1, days_since_large / (20 * 365)))  # Max at 20 years

    # Significance (higher = higher risk): already 0-1
    sig_score = max(0, min(1, significance))

    # Moment deficit (higher = higher risk): normalize by typical values
    deficit_mag = moment_to_magnitude(moment_deficit) if moment_deficit > 0 else 0
    deficit_score = max(0, min(1, deficit_mag / 7.0))  # Normalize to M7 equivalent

    # Weighted combination
    risk_score = (
        RISK_WEIGHTS['rate_ratio'] * rate_score +
        RISK_WEIGHTS['historical_mag'] * mag_score +
        RISK_WEIGHTS['time_since_large'] * time_score +
        RISK_WEIGHTS['significance'] * sig_score +
        RISK_WEIGHTS['moment_deficit'] * deficit_score
    )

    return float(risk_score)


def classify_risk_level(risk_score: float) -> str:
    """Classify gap risk level based on risk score."""
    if risk_score >= RISK_THRESHOLDS['critical']:
        return 'critical'
    elif risk_score >= RISK_THRESHOLDS['high']:
        return 'high'
    elif risk_score >= RISK_THRESHOLDS['moderate']:
        return 'moderate'
    elif risk_score >= RISK_THRESHOLDS['low']:
        return 'low'
    else:
        return 'very_low'


# =============================================================================
# Gap Characterization
# =============================================================================

def analyze_grid_cell(cell: Dict, historical_df: pd.DataFrame, recent_df: pd.DataFrame,
                      historical_years: float, recent_years: float,
                      rate_threshold: float, min_historical: int,
                      min_significance: float) -> Optional[Dict[str, Any]]:
    """
    Analyze a single grid cell for seismic gap characteristics.

    Returns None if cell doesn't qualify as a gap.
    """
    # Get events in this cell
    hist_events = count_events_in_cell(historical_df, cell)
    recent_events = count_events_in_cell(recent_df, cell)

    hist_count = len(hist_events)
    recent_count = len(recent_events)

    # Skip cells with insufficient historical activity
    if hist_count < min_historical:
        return None

    # Calculate rates
    hist_rate = hist_count / historical_years
    recent_rate = recent_count / recent_years if recent_years > 0 else 0

    # Rate ratio
    rate_ratio = recent_rate / hist_rate if hist_rate > 0 else 1.0

    # Statistical significance
    significance = calculate_poisson_significance(
        hist_count, historical_years, recent_count, recent_years
    )

    # Check if qualifies as a gap
    is_gap = (rate_ratio <= rate_threshold and significance >= min_significance)

    # Calculate additional metrics
    hist_max_mag = hist_events['magnitude'].max() if hist_count > 0 else 0
    hist_mean_mag = hist_events['magnitude'].mean() if hist_count > 0 else 0

    # Time since last significant event (M5+)
    all_events = pd.concat([hist_events, recent_events])
    large_events = all_events[all_events['magnitude'] >= 5.0]
    if len(large_events) > 0:
        last_large = pd.to_datetime(large_events['time'].max())
        days_since_large = (datetime.now(timezone.utc) - last_large).days
    else:
        # Use time since last event of any magnitude
        if len(all_events) > 0:
            last_event = pd.to_datetime(all_events['time'].max())
            days_since_large = (datetime.now(timezone.utc) - last_event).days
        else:
            days_since_large = int(historical_years * 365)

    # Moment deficit
    moment_deficit, potential_mag = estimate_moment_deficit(
        hist_events, recent_events, recent_years
    )

    # B-value comparison
    hist_b, hist_b_err = calculate_b_value(hist_events['magnitude'].values) if hist_count >= 10 else (1.0, 0.5)
    recent_b, recent_b_err = calculate_b_value(recent_events['magnitude'].values) if recent_count >= 10 else (1.0, 0.5)

    # Calculate risk score
    risk_score = calculate_risk_score(
        rate_ratio, hist_max_mag, days_since_large,
        significance, moment_deficit, hist_mean_mag
    )

    result = {
        'lat': cell['lat'],
        'lon': cell['lon'],
        'lat_center': (cell['lat'] + cell['lat_max']) / 2,
        'lon_center': (cell['lon'] + cell['lon_max']) / 2,
        'area_sq_km': cell['area_sq_km'],
        'historical_count': int(hist_count),
        'recent_count': int(recent_count),
        'historical_rate': float(hist_rate),
        'recent_rate': float(recent_rate),
        'rate_ratio': float(rate_ratio),
        'significance': float(significance),
        'p_value': float(1 - significance),
        'is_gap': is_gap,
        'historical_max_magnitude': float(hist_max_mag),
        'historical_mean_magnitude': float(hist_mean_mag),
        'days_since_last_m5': int(days_since_large),
        'moment_deficit_nm': float(moment_deficit),
        'potential_magnitude': float(potential_mag),
        'historical_b_value': float(hist_b),
        'recent_b_value': float(recent_b),
        'risk_score': float(risk_score),
        'risk_level': classify_risk_level(risk_score) if is_gap else 'not_gap'
    }

    return result


def merge_adjacent_gaps(gap_cells: List[Dict], resolution: float) -> List[Dict]:
    """
    Merge adjacent gap cells into contiguous gap regions.

    Args:
        gap_cells: List of gap cell dictionaries
        resolution: Grid resolution in degrees

    Returns:
        List of merged gap regions
    """
    if not gap_cells:
        return []

    # Simple merging: group cells that are within resolution distance
    merged_gaps = []
    used = set()

    for i, cell in enumerate(gap_cells):
        if i in used:
            continue

        # Start a new gap region
        region_cells = [cell]
        used.add(i)

        # Find adjacent cells
        for j, other in enumerate(gap_cells):
            if j in used:
                continue

            # Check if adjacent (within 1.5 * resolution)
            for rc in region_cells:
                lat_diff = abs(other['lat_center'] - rc['lat_center'])
                lon_diff = abs(other['lon_center'] - rc['lon_center'])

                if lat_diff <= resolution * 1.5 and lon_diff <= resolution * 1.5:
                    region_cells.append(other)
                    used.add(j)
                    break

        # Characterize merged region
        gap_region = characterize_gap_region(region_cells, len(merged_gaps) + 1)
        merged_gaps.append(gap_region)

    return merged_gaps


def characterize_gap_region(cells: List[Dict], gap_id: int) -> Dict[str, Any]:
    """
    Characterize a gap region from its constituent cells.

    Args:
        cells: List of gap cell dictionaries
        gap_id: Unique identifier for this gap

    Returns:
        Gap region dictionary with aggregated statistics
    """
    lats = [c['lat_center'] for c in cells]
    lons = [c['lon_center'] for c in cells]

    # Aggregate statistics
    total_hist_count = sum(c['historical_count'] for c in cells)
    total_recent_count = sum(c['recent_count'] for c in cells)
    total_area = sum(c['area_sq_km'] for c in cells)

    # Weighted averages by historical count
    weights = [c['historical_count'] for c in cells]
    total_weight = sum(weights) if sum(weights) > 0 else 1

    avg_rate_ratio = sum(c['rate_ratio'] * w for c, w in zip(cells, weights)) / total_weight
    avg_significance = sum(c['significance'] * w for c, w in zip(cells, weights)) / total_weight
    avg_risk_score = sum(c['risk_score'] * w for c, w in zip(cells, weights)) / total_weight

    # Maximum values
    max_hist_mag = max(c['historical_max_magnitude'] for c in cells)
    max_potential_mag = max(c['potential_magnitude'] for c in cells)
    total_moment_deficit = sum(c['moment_deficit_nm'] for c in cells)
    min_days_since = min(c['days_since_last_m5'] for c in cells)

    return {
        'gap_id': gap_id,
        'centroid': {
            'lat': float(np.mean(lats)),
            'lon': float(np.mean(lons))
        },
        'bounding_box': {
            'min_lat': float(min(c['lat'] for c in cells)),
            'max_lat': float(max(c['lat_center'] for c in cells) + (cells[0]['lat_center'] - cells[0]['lat'])),
            'min_lon': float(min(c['lon'] for c in cells)),
            'max_lon': float(max(c['lon_center'] for c in cells) + (cells[0]['lon_center'] - cells[0]['lon']))
        },
        'n_cells': len(cells),
        'area_sq_km': float(total_area),
        'historical_events': int(total_hist_count),
        'recent_events': int(total_recent_count),
        'historical_rate': float(total_hist_count / cells[0].get('historical_years', 20) if cells else 0),
        'recent_rate': float(total_recent_count / cells[0].get('recent_years', 5) if cells else 0),
        'rate_ratio': float(avg_rate_ratio),
        'quiescence_significance': float(avg_significance),
        'historical_max_magnitude': float(max_hist_mag),
        'days_since_last_m5': int(min_days_since),
        'estimated_moment_deficit': float(total_moment_deficit),
        'potential_magnitude': float(max_potential_mag),
        'risk_score': float(avg_risk_score),
        'risk_level': classify_risk_level(avg_risk_score)
    }


# =============================================================================
# Main Pipeline
# =============================================================================

def run_gap_analysis(df: pd.DataFrame, args) -> Dict[str, Any]:
    """
    Run the full seismic gap analysis pipeline.

    Args:
        df: Earthquake catalog DataFrame
        args: Command-line arguments

    Returns:
        Complete gap analysis results
    """
    logger.info("Starting seismic gap analysis...")

    # Filter by magnitude
    df_filtered = df[df['magnitude'] >= args.min_magnitude].copy()
    logger.info(f"Using {len(df_filtered)} events with M >= {args.min_magnitude}")

    if len(df_filtered) == 0:
        logger.warning("No events meet magnitude threshold")
        return {'error': 'No events meet magnitude threshold'}

    # Split catalog into historical and recent periods
    historical_df, recent_df, period_info = split_catalog_by_time(
        df_filtered, args.historical_years, args.recent_years
    )

    if len(historical_df) == 0:
        logger.warning("No events in historical period")
        return {'error': 'No events in historical period'}

    # Create analysis grid
    grid_cells = create_analysis_grid(df_filtered, args.grid_resolution)

    # Analyze each grid cell
    grid_analysis = []
    gap_cells = []

    for i, cell in enumerate(grid_cells):
        if (i + 1) % 100 == 0:
            logger.info(f"Analyzing cell {i + 1}/{len(grid_cells)}")

        result = analyze_grid_cell(
            cell, historical_df, recent_df,
            args.historical_years, args.recent_years,
            args.rate_threshold, args.min_historical_events,
            args.min_significance
        )

        if result:
            grid_analysis.append(result)
            if result['is_gap']:
                gap_cells.append(result)

    logger.info(f"Identified {len(gap_cells)} gap cells out of {len(grid_analysis)} analyzed")

    # Merge adjacent gaps into regions
    seismic_gaps = merge_adjacent_gaps(gap_cells, args.grid_resolution)
    logger.info(f"Merged into {len(seismic_gaps)} gap regions")

    # Calculate summary statistics
    if seismic_gaps:
        high_risk_gaps = [g for g in seismic_gaps if g['risk_level'] in ['high', 'critical']]
        total_gap_area = sum(g['area_sq_km'] for g in seismic_gaps)
        max_potential = max(g['potential_magnitude'] for g in seismic_gaps)
    else:
        high_risk_gaps = []
        total_gap_area = 0
        max_potential = 0

    summary = {
        'total_gaps': len(seismic_gaps),
        'high_risk_gaps': len(high_risk_gaps),
        'critical_gaps': len([g for g in seismic_gaps if g['risk_level'] == 'critical']),
        'total_gap_area_sq_km': float(total_gap_area),
        'max_potential_magnitude': float(max_potential),
        'total_cells_analyzed': len(grid_analysis),
        'gap_cells_identified': len(gap_cells),
        'overall_rate_change': float(len(recent_df) / args.recent_years) / (len(historical_df) / args.historical_years) if len(historical_df) > 0 else 1.0
    }

    # Build final report
    report = {
        'metadata': {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'analysis_version': '1.0',
            'historical_period': period_info['historical'],
            'recent_period': period_info['recent'],
            'grid_resolution_deg': args.grid_resolution,
            'rate_threshold': args.rate_threshold,
            'min_significance': args.min_significance,
            'min_historical_events': args.min_historical_events,
            'min_magnitude': args.min_magnitude,
            'total_events_analyzed': int(len(df_filtered)),
            'analysis_bounds': {
                'min_lat': float(df_filtered['latitude'].min()),
                'max_lat': float(df_filtered['latitude'].max()),
                'min_lon': float(df_filtered['longitude'].min()),
                'max_lon': float(df_filtered['longitude'].max())
            }
        },
        'seismic_gaps': seismic_gaps,
        'grid_analysis': grid_analysis,
        'summary': summary
    }

    return report


def print_summary(report: Dict[str, Any]) -> None:
    """Print a formatted summary of gap analysis results."""
    print(f"\n{'='*70}")
    print("SEISMIC GAP ANALYSIS SUMMARY")
    print(f"{'='*70}")

    meta = report['metadata']
    summary = report['summary']

    print(f"\nAnalysis Parameters:")
    print(f"  - Events analyzed: {meta['total_events_analyzed']}")
    print(f"  - Historical period: {meta['historical_period']['years']} years ({meta['historical_period']['n_events']} events)")
    print(f"  - Recent period: {meta['recent_period']['years']} years ({meta['recent_period']['n_events']} events)")
    print(f"  - Grid resolution: {meta['grid_resolution_deg']}°")
    print(f"  - Rate threshold: {meta['rate_threshold']}")
    print(f"  - Min significance: {meta['min_significance']}")

    print(f"\nGap Detection Results:")
    print(f"  - Total cells analyzed: {summary['total_cells_analyzed']}")
    print(f"  - Gap cells identified: {summary['gap_cells_identified']}")
    print(f"  - Merged gap regions: {summary['total_gaps']}")
    print(f"  - High/Critical risk gaps: {summary['high_risk_gaps']}")
    print(f"  - Total gap area: {summary['total_gap_area_sq_km']:.1f} sq km")
    print(f"  - Max potential magnitude: M{summary['max_potential_magnitude']:.1f}")
    print(f"  - Overall rate change: {summary['overall_rate_change']:.2f}x")

    # List gaps by risk level
    gaps = report['seismic_gaps']
    if gaps:
        print(f"\nIdentified Seismic Gaps (sorted by risk):")
        sorted_gaps = sorted(gaps, key=lambda x: x['risk_score'], reverse=True)

        for gap in sorted_gaps[:10]:  # Top 10
            print(f"\n  Gap #{gap['gap_id']} [{gap['risk_level'].upper()}]")
            print(f"    Location: ({gap['centroid']['lat']:.2f}°, {gap['centroid']['lon']:.2f}°)")
            print(f"    Area: {gap['area_sq_km']:.0f} sq km ({gap['n_cells']} cells)")
            print(f"    Rate ratio: {gap['rate_ratio']:.2f} (hist: {gap['historical_events']}, recent: {gap['recent_events']})")
            print(f"    Significance: {gap['quiescence_significance']:.2f}")
            print(f"    Historical max: M{gap['historical_max_magnitude']:.1f}")
            print(f"    Days since M5+: {gap['days_since_last_m5']}")
            print(f"    Potential magnitude: M{gap['potential_magnitude']:.1f}")
            print(f"    Risk score: {gap['risk_score']:.2f}")

        if len(gaps) > 10:
            print(f"\n  ... and {len(gaps) - 10} more gaps")
    else:
        print(f"\nNo significant seismic gaps identified.")

    print(f"\n{'='*70}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Seismic Gap Analysis Tool - Identify regions with anomalous quiescence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --input catalog.csv --output gaps.json
    %(prog)s --input catalog.csv --output gaps.json --historical-years 30 --recent-years 5
    %(prog)s --input catalog.csv --output gaps.json --rate-threshold 0.2 --min-significance 0.95
        """
    )

    # Required arguments
    parser.add_argument("--input", type=str, required=True,
                        help="Input earthquake catalog CSV file")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON file for gap analysis results")

    # Time period parameters
    parser.add_argument("--historical-years", type=float, default=20.0,
                        help="Length of historical period in years (default: 20)")
    parser.add_argument("--recent-years", type=float, default=5.0,
                        help="Length of recent period in years (default: 5)")

    # Grid parameters
    parser.add_argument("--grid-resolution", type=float, default=0.5,
                        help="Grid spacing in degrees (default: 0.5)")

    # Gap detection parameters
    parser.add_argument("--rate-threshold", type=float, default=0.3,
                        help="Max recent/historical rate ratio for gap detection (default: 0.3)")
    parser.add_argument("--min-historical-events", type=int, default=5,
                        help="Minimum events in historical period per cell (default: 5)")
    parser.add_argument("--min-significance", type=float, default=0.9,
                        help="Minimum statistical significance for gap (default: 0.9)")
    parser.add_argument("--min-magnitude", type=float, default=3.0,
                        help="Minimum magnitude to consider (default: 3.0)")

    args = parser.parse_args()

    try:
        # Load earthquake catalog
        logger.info(f"Loading earthquake catalog from {args.input}")
        df = pd.read_csv(args.input)
        logger.info(f"Loaded {len(df)} events")

        # Validate required columns
        required_cols = ['latitude', 'longitude', 'magnitude', 'time']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            sys.exit(1)

        # Run gap analysis
        report = run_gap_analysis(df, args)

        if 'error' in report:
            logger.error(report['error'])
            sys.exit(1)

        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Gap analysis saved to {args.output}")

        # Print summary
        print_summary(report)

    except FileNotFoundError:
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Gap analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
