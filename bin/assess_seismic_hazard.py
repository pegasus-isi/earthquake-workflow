#!/usr/bin/env python3
"""
Seismic Hazard Assessment Tool

Performs probabilistic seismic hazard analysis (PSHA) using Ground Motion
Prediction Equations (GMPEs) to estimate ground shaking probability at
grid points based on historical earthquake catalogs.

Features:
- Multiple GMPE models (simplified NGA, Boore-Atkinson)
- Grid-based hazard mapping
- Exceedance probability calculations
- Risk level classification
- Hazard curve generation

Usage:
    # Basic hazard assessment
    python assess_seismic_hazard.py --input catalog.csv --output hazard.json

    # With custom grid resolution
    python assess_seismic_hazard.py --input catalog.csv --output hazard.json \\
        --grid-resolution 0.5 --pga-thresholds 0.1 0.2 0.4

    # With specific analysis region
    python assess_seismic_hazard.py --input catalog.csv --output hazard.json \\
        --min-lat 32 --max-lat 42 --min-lon -125 --max-lon -114
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
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

# GMPE coefficients for simplified NGA-West2 model
# ln(PGA) = c0 + c1*M + c2*M^2 + c3*ln(R + c4*exp(c5*M)) + c6*ln(Vs30/760)
GMPE_SIMPLIFIED = {
    'c0': -1.997,
    'c1': 0.977,
    'c2': -0.038,
    'c3': -1.485,
    'c4': 7.688,
    'c5': 0.524,
    'c6': -0.406,
    'sigma': 0.65  # Total standard deviation (log units)
}

# Boore-Atkinson (2008) simplified coefficients
GMPE_BOORE_ATKINSON = {
    'e1': -0.53804,
    'e2': -0.50350,
    'e3': -0.75472,
    'e4': -0.50970,
    'e5': 0.28805,
    'e6': -0.10164,
    'e7': 0.0,
    'mh': 6.75,
    'c1': -0.66050,
    'c2': 0.11970,
    'c3': -0.01151,
    'h': 1.35,
    'sigma': 0.564
}

# Risk level thresholds (PGA in g)
RISK_THRESHOLDS = {
    'very_high': 0.4,
    'high': 0.2,
    'moderate': 0.1,
    'low': 0.05
}

# Default PGA values for hazard curves
DEFAULT_PGA_VALUES = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]


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


def calculate_rupture_distance(lat_site: float, lon_site: float, lat_eq: float,
                                lon_eq: float, depth_km: float) -> float:
    """
    Calculate rupture distance (Rrup) as hypocentral distance.
    For point sources, this is sqrt(Repi^2 + depth^2).
    """
    repi = haversine_distance(lat_site, lon_site, lat_eq, lon_eq)
    rrup = np.sqrt(repi**2 + depth_km**2)
    return max(rrup, 1.0)  # Minimum 1 km to avoid singularities


def magnitude_to_moment(magnitude: float) -> float:
    """Convert moment magnitude to seismic moment (N·m)."""
    return 10**(1.5 * magnitude + 9.1)


# =============================================================================
# GMPE Models
# =============================================================================

def gmpe_simplified(magnitude: float, distance_km: float, depth_km: float = 10.0,
                    vs30: float = 760.0) -> Tuple[float, float]:
    """
    Simplified NGA-West2-like GMPE for PGA estimation.

    Args:
        magnitude: Moment magnitude
        distance_km: Rupture distance in km
        depth_km: Hypocentral depth in km
        vs30: Time-averaged shear-wave velocity (m/s)

    Returns:
        Tuple of (median PGA in g, sigma in log units)
    """
    c = GMPE_SIMPLIFIED

    # Effective distance with magnitude-dependent near-source saturation
    r_eff = np.sqrt(distance_km**2 + (c['c4'] * np.exp(c['c5'] * magnitude))**2)

    # Log PGA calculation
    ln_pga = (c['c0'] + c['c1'] * magnitude + c['c2'] * magnitude**2 +
              c['c3'] * np.log(r_eff) + c['c6'] * np.log(vs30 / 760.0))

    # Depth adjustment (shallow events amplify)
    if depth_km < 20:
        ln_pga += 0.1 * (20 - depth_km) / 20

    pga = np.exp(ln_pga)
    return pga, c['sigma']


def gmpe_boore_atkinson(magnitude: float, distance_km: float, depth_km: float = 10.0,
                         vs30: float = 760.0) -> Tuple[float, float]:
    """
    Boore-Atkinson (2008) GMPE for PGA estimation (simplified version).

    Args:
        magnitude: Moment magnitude
        distance_km: Joyner-Boore distance in km
        depth_km: Hypocentral depth in km
        vs30: Time-averaged shear-wave velocity (m/s)

    Returns:
        Tuple of (median PGA in g, sigma in log units)
    """
    c = GMPE_BOORE_ATKINSON

    # Magnitude scaling
    if magnitude <= c['mh']:
        f_mag = c['e1'] + c['e2'] * (magnitude - c['mh']) + c['e3'] * (magnitude - c['mh'])**2
    else:
        f_mag = c['e1'] + c['e4'] * (magnitude - c['mh'])

    # Distance scaling
    r = np.sqrt(distance_km**2 + c['h']**2)
    f_dis = (c['c1'] + c['c2'] * (magnitude - 4.5)) * np.log(r / 1.0) + c['c3'] * (r - 1.0)

    # Site response (simplified linear)
    f_site = -0.36 * np.log(vs30 / 760.0)

    ln_pga = f_mag + f_dis + f_site
    pga = np.exp(ln_pga)

    return pga, c['sigma']


def calculate_pga(magnitude: float, distance_km: float, depth_km: float = 10.0,
                  vs30: float = 760.0, gmpe_model: str = 'simplified') -> Tuple[float, float]:
    """
    Calculate PGA using specified GMPE model.

    Args:
        magnitude: Moment magnitude
        distance_km: Distance in km
        depth_km: Depth in km
        vs30: Site Vs30 in m/s
        gmpe_model: 'simplified' or 'boore_atkinson'

    Returns:
        Tuple of (median PGA in g, sigma in log units)
    """
    if gmpe_model == 'boore_atkinson':
        return gmpe_boore_atkinson(magnitude, distance_km, depth_km, vs30)
    else:
        return gmpe_simplified(magnitude, distance_km, depth_km, vs30)


# =============================================================================
# Hazard Analysis Functions
# =============================================================================

def create_analysis_grid(df: pd.DataFrame, resolution: float,
                         min_lat: Optional[float] = None, max_lat: Optional[float] = None,
                         min_lon: Optional[float] = None, max_lon: Optional[float] = None,
                         buffer: float = 0.5) -> List[Dict[str, float]]:
    """
    Create a grid of analysis points covering the earthquake region.

    Args:
        df: Earthquake catalog DataFrame
        resolution: Grid spacing in degrees
        min_lat, max_lat, min_lon, max_lon: Optional bounds
        buffer: Buffer around data extent in degrees

    Returns:
        List of grid point dictionaries with lat, lon
    """
    # Determine bounds from data if not specified
    if min_lat is None:
        min_lat = df['latitude'].min() - buffer
    if max_lat is None:
        max_lat = df['latitude'].max() + buffer
    if min_lon is None:
        min_lon = df['longitude'].min() - buffer
    if max_lon is None:
        max_lon = df['longitude'].max() + buffer

    # Create grid
    lats = np.arange(min_lat, max_lat + resolution, resolution)
    lons = np.arange(min_lon, max_lon + resolution, resolution)

    grid_points = []
    for lat in lats:
        for lon in lons:
            grid_points.append({'lat': float(lat), 'lon': float(lon)})

    logger.info(f"Created analysis grid: {len(lats)} x {len(lons)} = {len(grid_points)} points")
    return grid_points


def calculate_exceedance_probability(pga_threshold: float, pga_median: float,
                                      sigma: float, duration_years: float,
                                      annual_rate: float = 1.0) -> float:
    """
    Calculate probability of exceeding a PGA threshold.

    Uses lognormal distribution for ground motion variability.

    Args:
        pga_threshold: PGA threshold in g
        pga_median: Median PGA from GMPE
        sigma: Standard deviation in log units
        duration_years: Time period for probability
        annual_rate: Annual rate of similar earthquakes

    Returns:
        Probability of exceedance
    """
    if pga_median <= 0 or pga_threshold <= 0:
        return 0.0

    # Probability of exceeding threshold given the event occurs
    # P(PGA > threshold | event) = 1 - CDF(ln(threshold), ln(median), sigma)
    z = (np.log(pga_threshold) - np.log(pga_median)) / sigma
    prob_exceed_given_event = 1 - stats.norm.cdf(z)

    # Annual probability of exceedance
    annual_prob = annual_rate * prob_exceed_given_event

    # Probability over duration (Poisson model)
    prob_duration = 1 - np.exp(-annual_prob * duration_years)

    return min(prob_duration, 1.0)


def classify_risk_level(pga: float) -> str:
    """Classify hazard risk level based on PGA."""
    if pga >= RISK_THRESHOLDS['very_high']:
        return 'very_high'
    elif pga >= RISK_THRESHOLDS['high']:
        return 'high'
    elif pga >= RISK_THRESHOLDS['moderate']:
        return 'moderate'
    elif pga >= RISK_THRESHOLDS['low']:
        return 'low'
    else:
        return 'very_low'


def calculate_site_hazard(site_lat: float, site_lon: float, df: pd.DataFrame,
                          vs30: float, gmpe_model: str, pga_thresholds: List[float],
                          return_periods: List[int], time_window_years: float) -> Dict[str, Any]:
    """
    Calculate seismic hazard for a single site.

    Args:
        site_lat, site_lon: Site coordinates
        df: Earthquake catalog
        vs30: Site Vs30
        gmpe_model: GMPE model name
        pga_thresholds: PGA thresholds for exceedance
        return_periods: Return periods in years
        time_window_years: Historical time window

    Returns:
        Dictionary with hazard results for the site
    """
    # Calculate distance and PGA from each event
    pga_values = []
    dominant_source = None
    max_pga = 0.0

    for _, event in df.iterrows():
        distance = calculate_rupture_distance(
            site_lat, site_lon,
            event['latitude'], event['longitude'],
            event.get('depth_km', 10.0)
        )

        pga, sigma = calculate_pga(
            event['magnitude'], distance, event.get('depth_km', 10.0),
            vs30, gmpe_model
        )

        pga_values.append({
            'pga': pga,
            'sigma': sigma,
            'magnitude': event['magnitude'],
            'distance': distance,
            'event_id': str(event.get('id', ''))
        })

        if pga > max_pga:
            max_pga = pga
            dominant_source = {
                'event_id': str(event.get('id', '')),
                'magnitude': float(event['magnitude']),
                'distance_km': float(distance),
                'pga_contribution': float(pga)
            }

    # Calculate annual rate of significant events
    n_events = len(df)
    annual_rate = n_events / time_window_years if time_window_years > 0 else 0

    # Calculate exceedance probabilities for thresholds
    exceedance_probs = {}
    for threshold in pga_thresholds:
        # Sum contributions from all events (simplified PSHA)
        total_annual_rate = 0
        for pv in pga_values:
            z = (np.log(threshold) - np.log(pv['pga'])) / pv['sigma']
            prob_exceed = 1 - stats.norm.cdf(z)
            total_annual_rate += prob_exceed / time_window_years

        # 50-year exceedance probability
        prob_50yr = 1 - np.exp(-total_annual_rate * 50)
        exceedance_probs[f'exceedance_prob_{threshold}g_50yr'] = float(prob_50yr)

    # Calculate hazard for return periods (PGA at X% in Y years)
    hazard_levels = {}
    for rp in return_periods:
        # Find PGA corresponding to return period
        target_annual_rate = 1.0 / rp

        # Binary search for PGA
        pga_low, pga_high = 0.001, 2.0
        for _ in range(50):  # Iteration limit
            pga_mid = (pga_low + pga_high) / 2

            total_rate = 0
            for pv in pga_values:
                z = (np.log(pga_mid) - np.log(pv['pga'])) / pv['sigma']
                prob_exceed = 1 - stats.norm.cdf(z)
                total_rate += prob_exceed / time_window_years

            if total_rate > target_annual_rate:
                pga_low = pga_mid
            else:
                pga_high = pga_mid

        hazard_levels[f'pga_{rp}yr_return'] = float(pga_mid)

    # Build hazard curve
    hazard_curve = {'pga_values': [], 'annual_exceedance_prob': []}
    for pga_val in DEFAULT_PGA_VALUES:
        total_rate = 0
        for pv in pga_values:
            z = (np.log(pga_val) - np.log(pv['pga'])) / pv['sigma']
            prob_exceed = 1 - stats.norm.cdf(z)
            total_rate += prob_exceed / time_window_years

        hazard_curve['pga_values'].append(pga_val)
        hazard_curve['annual_exceedance_prob'].append(float(total_rate))

    # Determine risk level based on 475-year return period PGA (if calculated)
    design_pga = hazard_levels.get('pga_475yr_return', max_pga)
    risk_level = classify_risk_level(design_pga)

    return {
        'lat': float(site_lat),
        'lon': float(site_lon),
        'max_pga_historical': float(max_pga),
        **exceedance_probs,
        **hazard_levels,
        'risk_level': risk_level,
        'dominant_source': dominant_source,
        'n_contributing_events': len(pga_values),
        'hazard_curve': hazard_curve
    }


# =============================================================================
# Main Pipeline
# =============================================================================

def run_hazard_assessment(df: pd.DataFrame, args) -> Dict[str, Any]:
    """
    Run the full seismic hazard assessment pipeline.

    Args:
        df: Earthquake catalog DataFrame
        args: Command-line arguments

    Returns:
        Complete hazard assessment results
    """
    logger.info("Starting seismic hazard assessment...")

    # Filter by magnitude
    df_filtered = df[df['magnitude'] >= args.min_magnitude].copy()
    logger.info(f"Using {len(df_filtered)} events with M >= {args.min_magnitude}")

    if len(df_filtered) == 0:
        logger.warning("No events meet magnitude threshold")
        return {'error': 'No events meet magnitude threshold'}

    # Calculate time window
    df_filtered['time'] = pd.to_datetime(df_filtered['time'], format='mixed', utc=True)
    time_span_days = (df_filtered['time'].max() - df_filtered['time'].min()).days
    time_window_years = max(time_span_days / 365.25, 1.0)

    if args.time_window:
        time_window_years = min(time_window_years, args.time_window)

    logger.info(f"Analysis time window: {time_window_years:.1f} years")

    # Create analysis grid
    grid_points = create_analysis_grid(
        df_filtered, args.grid_resolution,
        args.min_lat, args.max_lat, args.min_lon, args.max_lon
    )

    # Parse PGA thresholds and return periods
    pga_thresholds = args.pga_thresholds
    return_periods = args.return_periods

    # Calculate hazard for each grid point
    hazard_grid = []
    hazard_curves = {}

    total_points = len(grid_points)
    for i, point in enumerate(grid_points):
        if (i + 1) % 50 == 0:
            logger.info(f"Processing grid point {i + 1}/{total_points}")

        site_hazard = calculate_site_hazard(
            point['lat'], point['lon'], df_filtered,
            args.vs30, args.gmpe_model, pga_thresholds,
            return_periods, time_window_years
        )

        # Extract hazard curve and store separately
        curve = site_hazard.pop('hazard_curve')
        hazard_curves[f"{point['lat']}_{point['lon']}"] = curve

        hazard_grid.append(site_hazard)

    # Calculate regional summary
    pga_values = [h['max_pga_historical'] for h in hazard_grid]
    risk_counts = {}
    for h in hazard_grid:
        level = h['risk_level']
        risk_counts[level] = risk_counts.get(level, 0) + 1

    # Find dominant magnitude range
    mag_bins = [(4, 5), (5, 6), (6, 7), (7, 8), (8, 9)]
    mag_counts = {f"{lo}-{hi}": len(df_filtered[(df_filtered['magnitude'] >= lo) & (df_filtered['magnitude'] < hi)])
                  for lo, hi in mag_bins}
    dominant_range = max(mag_counts, key=mag_counts.get)

    regional_summary = {
        'max_pga_observed': float(max(pga_values)) if pga_values else 0.0,
        'mean_pga': float(np.mean(pga_values)) if pga_values else 0.0,
        'high_hazard_area_pct': float(100 * (risk_counts.get('high', 0) + risk_counts.get('very_high', 0)) / len(hazard_grid)) if hazard_grid else 0.0,
        'risk_distribution': {k: int(v) for k, v in risk_counts.items()},
        'dominant_magnitude_range': dominant_range,
        'total_grid_points': len(hazard_grid)
    }

    # Build final report
    report = {
        'metadata': {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'analysis_version': '1.0',
            'grid_resolution_deg': args.grid_resolution,
            'total_events_used': int(len(df_filtered)),
            'time_window_years': float(time_window_years),
            'vs30_ms': args.vs30,
            'gmpe_model': args.gmpe_model,
            'pga_thresholds_g': pga_thresholds,
            'return_periods_years': return_periods,
            'min_magnitude': args.min_magnitude,
            'analysis_bounds': {
                'min_lat': float(df_filtered['latitude'].min()),
                'max_lat': float(df_filtered['latitude'].max()),
                'min_lon': float(df_filtered['longitude'].min()),
                'max_lon': float(df_filtered['longitude'].max())
            }
        },
        'hazard_grid': hazard_grid,
        'hazard_curves': hazard_curves,
        'regional_summary': regional_summary,
        'gmpe_parameters': {
            'model': args.gmpe_model,
            'coefficients': GMPE_SIMPLIFIED if args.gmpe_model == 'simplified' else GMPE_BOORE_ATKINSON
        }
    }

    return report


def print_summary(report: Dict[str, Any]) -> None:
    """Print a formatted summary of hazard assessment results."""
    print(f"\n{'='*70}")
    print("SEISMIC HAZARD ASSESSMENT SUMMARY")
    print(f"{'='*70}")

    meta = report['metadata']
    summary = report['regional_summary']

    print(f"\nAnalysis Parameters:")
    print(f"  - Events analyzed: {meta['total_events_used']}")
    print(f"  - Time window: {meta['time_window_years']:.1f} years")
    print(f"  - Grid resolution: {meta['grid_resolution_deg']}°")
    print(f"  - Grid points: {summary['total_grid_points']}")
    print(f"  - GMPE model: {meta['gmpe_model']}")
    print(f"  - Site Vs30: {meta['vs30_ms']} m/s")

    print(f"\nHazard Summary:")
    print(f"  - Maximum PGA observed: {summary['max_pga_observed']:.3f} g")
    print(f"  - Mean PGA: {summary['mean_pga']:.4f} g")
    print(f"  - High hazard area: {summary['high_hazard_area_pct']:.1f}%")
    print(f"  - Dominant magnitude range: M{summary['dominant_magnitude_range']}")

    print(f"\nRisk Distribution:")
    for level, count in sorted(summary['risk_distribution'].items()):
        pct = 100 * count / summary['total_grid_points']
        print(f"  - {level}: {count} points ({pct:.1f}%)")

    # Find highest hazard locations
    high_hazard = [h for h in report['hazard_grid'] if h['risk_level'] in ['high', 'very_high']]
    if high_hazard:
        print(f"\nTop 5 Highest Hazard Locations:")
        sorted_hazard = sorted(high_hazard, key=lambda x: x['max_pga_historical'], reverse=True)[:5]
        for i, h in enumerate(sorted_hazard, 1):
            print(f"  {i}. ({h['lat']:.2f}°, {h['lon']:.2f}°) - PGA: {h['max_pga_historical']:.3f} g [{h['risk_level']}]")

    print(f"{'='*70}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Seismic Hazard Assessment Tool - Probabilistic seismic hazard analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --input catalog.csv --output hazard.json
    %(prog)s --input catalog.csv --output hazard.json --grid-resolution 0.5
    %(prog)s --input catalog.csv --output hazard.json --min-lat 32 --max-lat 42
        """
    )

    # Required arguments
    parser.add_argument("--input", type=str, required=True,
                        help="Input earthquake catalog CSV file")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON file for hazard results")

    # Grid parameters
    parser.add_argument("--grid-resolution", type=float, default=1.0,
                        help="Grid spacing in degrees (default: 1.0)")
    parser.add_argument("--min-lat", type=float, default=None,
                        help="Minimum latitude for analysis")
    parser.add_argument("--max-lat", type=float, default=None,
                        help="Maximum latitude for analysis")
    parser.add_argument("--min-lon", type=float, default=None,
                        help="Minimum longitude for analysis")
    parser.add_argument("--max-lon", type=float, default=None,
                        help="Maximum longitude for analysis")

    # Hazard parameters
    parser.add_argument("--pga-thresholds", type=float, nargs='+', default=[0.1, 0.2, 0.4],
                        help="PGA thresholds in g (default: 0.1 0.2 0.4)")
    parser.add_argument("--return-periods", type=int, nargs='+', default=[50, 100, 475],
                        help="Return periods in years (default: 50 100 475)")
    parser.add_argument("--time-window", type=float, default=None,
                        help="Historical time window in years (default: auto from data)")
    parser.add_argument("--min-magnitude", type=float, default=4.0,
                        help="Minimum magnitude for hazard calculation (default: 4.0)")

    # Site parameters
    parser.add_argument("--vs30", type=float, default=760.0,
                        help="Default site Vs30 in m/s (default: 760, rock site)")
    parser.add_argument("--gmpe-model", type=str, default='simplified',
                        choices=['simplified', 'boore_atkinson'],
                        help="GMPE model to use (default: simplified)")

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

        # Handle depth column
        if 'depth_km' not in df.columns:
            logger.warning("No 'depth_km' column found, using default 10 km")
            df['depth_km'] = 10.0

        # Run hazard assessment
        report = run_hazard_assessment(df, args)

        if 'error' in report:
            logger.error(report['error'])
            sys.exit(1)

        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Hazard assessment saved to {args.output}")

        # Print summary
        print_summary(report)

    except FileNotFoundError:
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Hazard assessment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
