#!/usr/bin/env python3

"""
Predict aftershock probabilities using statistical and ML models.

This script identifies mainshock events from earthquake catalogs and predicts:
- Probability of aftershocks at different magnitude thresholds
- Expected aftershock counts over time windows
- Temporal decay patterns following Omori-Utsu law
- Feature importance for seismological interpretation

Supports both:
- Statistical baselines: Omori-Utsu decay law, Bath's law, Gutenberg-Richter
- ML models: Random Forest, Gradient Boosting for probability estimation

Usage:
    # Inference mode (statistical models only)
    python predict_aftershocks.py --input earthquake_catalog.csv \
                                  --output predictions/aftershock_predictions.json

    # Inference with ML model
    python predict_aftershocks.py --input earthquake_catalog.csv \
                                  --output predictions/aftershock_predictions.json \
                                  --model-path models/aftershock_model.pkl

    # Training mode (train new model from historical data)
    python predict_aftershocks.py --input historical_catalog.csv \
                                  --output models/ \
                                  --mode train
"""

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Wells-Coppersmith (1994) rupture length scaling
WELLS_COPPERSMITH_A = -2.44
WELLS_COPPERSMITH_B = 0.59

# Omori-Utsu law defaults
OMORI_DEFAULT_C = 0.05  # days
OMORI_DEFAULT_P = 1.1

# Bath's law constant (largest aftershock ~1.2 less than mainshock)
BATH_CONSTANT = 1.2

# Default b-value for Gutenberg-Richter
DEFAULT_B_VALUE = 1.0


# =============================================================================
# Utility Functions
# =============================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great circle distance between two points on Earth.

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


def wells_coppersmith_radius(magnitude: float) -> float:
    """
    Estimate aftershock zone radius using Wells-Coppersmith (1994) relations.

    Based on rupture length scaling: log10(L) = a + b*M

    Args:
        magnitude: Mainshock magnitude

    Returns:
        Estimated aftershock zone radius in km
    """
    log_length = WELLS_COPPERSMITH_A + WELLS_COPPERSMITH_B * magnitude
    rupture_length = 10 ** log_length
    # Aftershock zone typically extends ~1.5x rupture length on each side
    return rupture_length * 1.5


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Quick distance calculation using flat-earth approximation.

    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates

    Returns:
        Approximate distance in km
    """
    lat_diff = (lat2 - lat1) * 111  # km per degree
    lon_diff = (lon2 - lon1) * 111 * np.cos(np.radians((lat1 + lat2) / 2))
    return np.sqrt(lat_diff**2 + lon_diff**2)


# =============================================================================
# Mainshock Identification
# =============================================================================

def identify_mainshocks(
    df: pd.DataFrame,
    magnitude_threshold: float = 5.0,
    decluster_window_days: int = 7,
    decluster_radius_km: Optional[float] = None
) -> pd.DataFrame:
    """
    Identify mainshock events from earthquake catalog.

    A mainshock is defined as the largest event in a space-time window.
    Uses magnitude-dependent radius if not specified.

    Args:
        df: DataFrame with columns: time, latitude, longitude, depth_km, magnitude
        magnitude_threshold: Minimum magnitude to consider as mainshock
        decluster_window_days: Time window for declustering (days before event)
        decluster_radius_km: Fixed radius for declustering (km), or None for magnitude-dependent

    Returns:
        DataFrame containing mainshock events with aftershock_zone_radius_km column
    """
    logger.info(f"Identifying mainshocks (M >= {magnitude_threshold})...")

    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], format='mixed', utc=True)
    df = df.sort_values('time').reset_index(drop=True)

    # Filter to potential mainshocks
    candidates = df[df['magnitude'] >= magnitude_threshold].copy()

    if len(candidates) == 0:
        logger.warning(f"No events found with magnitude >= {magnitude_threshold}")
        return pd.DataFrame()

    mainshocks = []

    for idx, event in candidates.iterrows():
        event_time = event['time']
        event_mag = event['magnitude']
        event_lat = event['latitude']
        event_lon = event['longitude']

        # Use magnitude-dependent radius if not specified
        radius = decluster_radius_km if decluster_radius_km else wells_coppersmith_radius(event_mag)

        # Check if this is the largest event in the preceding window
        time_mask = (df['time'] >= event_time - timedelta(days=decluster_window_days)) & \
                   (df['time'] < event_time)

        if time_mask.sum() > 0:
            preceding_events = df[time_mask]

            # Calculate distances
            distances = preceding_events.apply(
                lambda row: calculate_distance(event_lat, event_lon, row['latitude'], row['longitude']),
                axis=1
            )

            nearby_preceding = preceding_events[distances <= radius]

            # Check if any preceding nearby event is larger
            if len(nearby_preceding) > 0 and nearby_preceding['magnitude'].max() >= event_mag:
                continue  # This event is likely an aftershock of a previous mainshock

        # This is a mainshock
        mainshock_data = event.to_dict()
        mainshock_data['aftershock_zone_radius_km'] = float(radius)
        mainshocks.append(mainshock_data)

    mainshocks_df = pd.DataFrame(mainshocks)
    logger.info(f"Identified {len(mainshocks_df)} mainshocks")

    return mainshocks_df


# =============================================================================
# Feature Engineering
# =============================================================================

def calculate_b_value(magnitudes: np.ndarray) -> Optional[float]:
    """
    Calculate Gutenberg-Richter b-value using maximum likelihood.

    Args:
        magnitudes: Array of earthquake magnitudes

    Returns:
        Estimated b-value or None if insufficient data
    """
    if len(magnitudes) < 20:
        return None

    # Estimate magnitude of completeness (simple method: mode)
    bins = np.arange(magnitudes.min(), magnitudes.max() + 0.1, 0.1)
    hist, edges = np.histogram(magnitudes, bins=bins)
    mc_idx = np.argmax(hist)
    mc = edges[mc_idx]

    # Filter events above Mc
    mags_complete = magnitudes[magnitudes >= mc]
    if len(mags_complete) < 10:
        return None

    # Aki's (1965) maximum likelihood b-value
    mean_mag = np.mean(mags_complete)
    b_value = np.log10(np.e) / (mean_mag - mc + 0.05)

    return float(b_value)


def extract_mainshock_features(
    df: pd.DataFrame,
    mainshock: pd.Series,
    lookback_days: int = 365
) -> Dict[str, float]:
    """
    Extract features for a mainshock event for ML prediction.

    Args:
        df: Full earthquake catalog DataFrame
        mainshock: Series containing mainshock event data
        lookback_days: Days of historical data to consider

    Returns:
        Dictionary of feature names to values
    """
    mainshock_time = pd.to_datetime(mainshock['time'])
    mainshock_lat = mainshock['latitude']
    mainshock_lon = mainshock['longitude']
    mainshock_mag = mainshock['magnitude']
    mainshock_depth = mainshock.get('depth_km', 10.0)

    # Define region around mainshock
    region_radius_km = 200  # km

    # Get historical events in region
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], format='mixed', utc=True)

    historical_mask = (df['time'] < mainshock_time) & \
                     (df['time'] >= mainshock_time - timedelta(days=lookback_days))
    historical = df[historical_mask].copy()

    if len(historical) > 0:
        # Calculate distances from mainshock
        historical['distance_km'] = historical.apply(
            lambda row: calculate_distance(mainshock_lat, mainshock_lon, row['latitude'], row['longitude']),
            axis=1
        )
        regional = historical[historical['distance_km'] <= region_radius_km]
    else:
        regional = pd.DataFrame()

    # Build feature dictionary
    features = {
        # Mainshock properties
        'mainshock_magnitude': float(mainshock_mag),
        'mainshock_depth_km': float(mainshock_depth) if pd.notna(mainshock_depth) else 10.0,
        'mainshock_latitude': float(mainshock_lat),
        'mainshock_longitude': float(mainshock_lon),

        # Aftershock zone estimate
        'aftershock_zone_radius_km': float(wells_coppersmith_radius(mainshock_mag)),
        'aftershock_zone_area_km2': float(np.pi * wells_coppersmith_radius(mainshock_mag) ** 2),
    }

    # Historical seismicity features
    if len(regional) > 0:
        features['historical_event_count'] = int(len(regional))
        features['historical_event_rate'] = float(len(regional) / lookback_days)
        features['historical_max_magnitude'] = float(regional['magnitude'].max())
        features['historical_mean_magnitude'] = float(regional['magnitude'].mean())
        features['historical_mean_depth'] = float(regional['depth_km'].mean()) if 'depth_km' in regional else 10.0
        features['historical_depth_std'] = float(regional['depth_km'].std()) if 'depth_km' in regional else 5.0

        # B-value
        b_val = calculate_b_value(regional['magnitude'].values)
        features['historical_b_value'] = float(b_val) if b_val else DEFAULT_B_VALUE

        # Regional seismic moment (proxy for energy release)
        moments = 10 ** (1.5 * regional['magnitude'].values + 9.1)  # Hanks & Kanamori
        features['regional_moment_rate'] = float(np.sum(moments) / lookback_days)

        # Days since last significant event
        m5_events = regional[regional['magnitude'] >= 5.0]
        if len(m5_events) > 0:
            last_m5_time = m5_events['time'].max()
            features['days_since_last_m5'] = float((mainshock_time - last_m5_time).total_seconds() / 86400)
        else:
            features['days_since_last_m5'] = float(lookback_days)
    else:
        # No historical data - use defaults
        features['historical_event_count'] = 0
        features['historical_event_rate'] = 0.0
        features['historical_max_magnitude'] = 0.0
        features['historical_mean_magnitude'] = 0.0
        features['historical_mean_depth'] = 10.0
        features['historical_depth_std'] = 5.0
        features['historical_b_value'] = DEFAULT_B_VALUE
        features['regional_moment_rate'] = 0.0
        features['days_since_last_m5'] = float(lookback_days)

    # Depth category (shallow/intermediate/deep)
    depth = features['mainshock_depth_km']
    if depth < 20:
        features['depth_category'] = 0  # shallow
    elif depth < 70:
        features['depth_category'] = 1  # intermediate-shallow
    else:
        features['depth_category'] = 2  # deep

    # Temporal features
    features['hour_of_day'] = int(mainshock_time.hour)

    return features


# =============================================================================
# Statistical Models
# =============================================================================

def omori_utsu_rate(t: np.ndarray, K: float, c: float, p: float) -> np.ndarray:
    """
    Calculate aftershock rate using Omori-Utsu law.

    n(t) = K / (t + c)^p

    Args:
        t: Time since mainshock (days)
        K: Productivity parameter
        c: Time offset parameter (typically 0.01-0.1 days)
        p: Decay exponent (typically ~1.0)

    Returns:
        Expected aftershock rate at each time
    """
    return K / np.power(t + c, p)


def estimate_omori_productivity(mainshock_magnitude: float, b_value: float = 1.0) -> float:
    """
    Estimate Omori K parameter based on mainshock magnitude.

    Uses Reasenberg-Jones (1989) productivity relation.

    Args:
        mainshock_magnitude: Mainshock magnitude
        b_value: Gutenberg-Richter b-value

    Returns:
        Estimated K parameter (events/day at t=0)
    """
    # Reasenberg-Jones: K ~ 10^(a + b*(M - Mc))
    # Simplified: K proportional to 10^(0.8 * M)
    a = -1.67  # Typical value
    K = 10 ** (a + 0.8 * mainshock_magnitude)
    return float(K)


def predict_aftershock_count_statistical(
    mainshock_magnitude: float,
    target_magnitude: float,
    time_window_days: float,
    b_value: float = 1.0
) -> Dict[str, float]:
    """
    Predict aftershock counts using statistical relationships.

    Uses:
    - Reasenberg-Jones model for productivity
    - Gutenberg-Richter for magnitude distribution
    - Omori-Utsu for temporal integration

    Args:
        mainshock_magnitude: Magnitude of mainshock
        target_magnitude: Minimum aftershock magnitude to count
        time_window_days: Prediction time window (days)
        b_value: Gutenberg-Richter b-value

    Returns:
        Dictionary with expected counts and rates
    """
    K = estimate_omori_productivity(mainshock_magnitude, b_value)
    c = OMORI_DEFAULT_C
    p = OMORI_DEFAULT_P

    # Integrate Omori-Utsu over time window
    # N(T) = K * [(T + c)^(1-p) - c^(1-p)] / (1 - p) for p != 1
    if abs(p - 1.0) > 0.01:
        total_aftershocks = K * ((time_window_days + c)**(1 - p) - c**(1 - p)) / (1 - p)
    else:
        # For p ~ 1: N(T) = K * ln((T + c) / c)
        total_aftershocks = K * np.log((time_window_days + c) / c)

    # Apply Gutenberg-Richter to get count at target magnitude
    # N(M >= target) = N_total * 10^(-b * (target - M_min))
    magnitude_completeness = mainshock_magnitude - 3.0  # Rough Mc estimate
    magnitude_completeness = max(magnitude_completeness, 2.0)

    if target_magnitude >= magnitude_completeness:
        magnitude_factor = 10 ** (-b_value * (target_magnitude - magnitude_completeness))
    else:
        magnitude_factor = 1.0

    expected_count = total_aftershocks * magnitude_factor

    return {
        'expected_count': float(max(0, expected_count)),
        'omori_K': float(K),
        'omori_c': float(c),
        'omori_p': float(p),
        'total_aftershocks': float(max(0, total_aftershocks)),
        'magnitude_factor': float(magnitude_factor)
    }


def predict_statistical(
    mainshock: pd.Series,
    b_value: float = 1.0,
    magnitude_thresholds: List[float] = [4.0, 5.0, 6.0],
    time_windows_days: List[int] = [1, 7, 30]
) -> Dict[str, Any]:
    """
    Generate statistical predictions for a mainshock.

    Args:
        mainshock: Series with mainshock data
        b_value: Gutenberg-Richter b-value
        magnitude_thresholds: Magnitude thresholds for predictions
        time_windows_days: Time windows for predictions (days)

    Returns:
        Dictionary with statistical predictions
    """
    mainshock_mag = mainshock['magnitude']

    # Omori parameters
    K = estimate_omori_productivity(mainshock_mag, b_value)

    predictions = {
        'omori_parameters': {
            'K': float(K),
            'c': float(OMORI_DEFAULT_C),
            'p': float(OMORI_DEFAULT_P)
        },
        'expected_counts': {},
        'probability': {},
        'bath_law': {
            'expected_largest_aftershock': float(mainshock_mag - BATH_CONSTANT),
            'range': [float(mainshock_mag - BATH_CONSTANT - 0.5),
                     float(mainshock_mag - BATH_CONSTANT + 0.5)]
        }
    }

    # Calculate expected counts for each threshold and window
    for target_mag in magnitude_thresholds:
        mag_key = f'M>={target_mag}'
        predictions['expected_counts'][mag_key] = {}

        for window in time_windows_days:
            result = predict_aftershock_count_statistical(
                mainshock_mag, target_mag, window, b_value
            )
            predictions['expected_counts'][mag_key][f'{window}_day'] = float(result['expected_count'])

    # Calculate probabilities (Poisson: P(N >= 1) = 1 - exp(-lambda))
    for target_mag in magnitude_thresholds:
        for window in time_windows_days:
            expected = predictions['expected_counts'][f'M>={target_mag}'][f'{window}_day']
            prob = 1 - np.exp(-expected) if expected > 0 else 0.0
            prob = min(prob, 0.99)  # Cap at 99%
            predictions['probability'][f'M>={target_mag}_within_{window}_days'] = float(prob)

    return predictions


# =============================================================================
# ML Model
# =============================================================================

class AftershockPredictor:
    """
    ML-based aftershock probability predictor using Random Forest or Gradient Boosting.
    """

    def __init__(
        self,
        model_type: str = 'random_forest',
        target_magnitude: float = 5.0,
        time_window_days: int = 7,
        n_estimators: int = 100
    ):
        """
        Initialize aftershock predictor.

        Args:
            model_type: 'random_forest' or 'gradient_boosting'
            target_magnitude: Threshold for significant aftershock
            time_window_days: Prediction time window
            n_estimators: Number of trees in ensemble
        """
        self.model_type = model_type
        self.target_magnitude = target_magnitude
        self.time_window_days = time_window_days
        self.n_estimators = n_estimators
        self.model = None
        self.feature_names = None
        self.training_metrics = None

    def _create_model(self):
        """Create the underlying ML model."""
        try:
            if self.model_type == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
            else:  # gradient_boosting
                from sklearn.ensemble import GradientBoostingClassifier
                return GradientBoostingClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
        except ImportError:
            logger.error("scikit-learn not available. Install with: pip install scikit-learn")
            raise

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        val_split: float = 0.2
    ) -> Dict:
        """
        Train ML model on historical mainshock-aftershock data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary target array (1 = significant aftershock occurred)
            feature_names: List of feature names
            val_split: Validation split fraction

        Returns:
            Dictionary with training history and metrics
        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, brier_score_loss
        )

        logger.info(f"Training {self.model_type} model on {len(X)} samples...")

        self.feature_names = feature_names
        self.model = self._create_model()

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )

        # Train
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_val)
        y_prob = self.model.predict_proba(X_val)[:, 1] if hasattr(self.model, 'predict_proba') else y_pred

        metrics = {
            'accuracy': float(accuracy_score(y_val, y_pred)),
            'precision': float(precision_score(y_val, y_pred, zero_division=0)),
            'recall': float(recall_score(y_val, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_val, y_pred, zero_division=0)),
            'n_train': int(len(X_train)),
            'n_val': int(len(X_val)),
            'positive_rate_train': float(np.mean(y_train)),
            'positive_rate_val': float(np.mean(y_val))
        }

        # AUC and Brier score if we have both classes
        if len(np.unique(y_val)) > 1:
            metrics['auc_roc'] = float(roc_auc_score(y_val, y_prob))
            metrics['brier_score'] = float(brier_score_loss(y_val, y_prob))

        self.training_metrics = metrics
        logger.info(f"Training complete. Validation accuracy: {metrics['accuracy']:.3f}")

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate probability predictions for mainshocks.

        Args:
            X: Feature matrix

        Returns:
            Predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a saved model.")

        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, 1]
        else:
            return self.model.predict(X).astype(float)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.model is None or self.feature_names is None:
            return {}

        importances = self.model.feature_importances_
        return {name: float(imp) for name, imp in zip(self.feature_names, importances)}

    def save(self, filepath: str):
        """Save trained model to file."""
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'target_magnitude': self.target_magnitude,
            'time_window_days': self.time_window_days,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'n_estimators': self.n_estimators
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'AftershockPredictor':
        """Load trained model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        predictor = cls(
            model_type=model_data['model_type'],
            target_magnitude=model_data['target_magnitude'],
            time_window_days=model_data['time_window_days'],
            n_estimators=model_data.get('n_estimators', 100)
        )
        predictor.model = model_data['model']
        predictor.feature_names = model_data['feature_names']
        predictor.training_metrics = model_data.get('training_metrics')

        logger.info(f"Model loaded from {filepath}")
        return predictor


# =============================================================================
# Training Pipeline
# =============================================================================

def prepare_training_data(
    df: pd.DataFrame,
    mainshock_threshold: float = 5.0,
    aftershock_window_days: int = 30,
    target_magnitude: float = 5.0,
    target_window_days: int = 7
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare training data from historical earthquake catalog.

    For each historical mainshock:
    1. Extract features from pre-mainshock data
    2. Calculate target (did significant aftershock occur?)
    3. Build feature matrix and target vector

    Args:
        df: Historical earthquake catalog
        mainshock_threshold: Minimum mainshock magnitude
        aftershock_window_days: Window to look for aftershocks
        target_magnitude: Threshold for "significant" aftershock
        target_window_days: Prediction window for target

    Returns:
        Tuple of (X, y, feature_names)
    """
    logger.info("Preparing training data...")

    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], format='mixed', utc=True)
    df = df.sort_values('time').reset_index(drop=True)

    # Identify mainshocks
    mainshocks = identify_mainshocks(df, magnitude_threshold=mainshock_threshold)

    if len(mainshocks) == 0:
        logger.error("No mainshocks found in catalog")
        return np.array([]), np.array([]), []

    features_list = []
    targets = []

    for idx, mainshock in mainshocks.iterrows():
        mainshock_time = pd.to_datetime(mainshock['time'])
        mainshock_lat = mainshock['latitude']
        mainshock_lon = mainshock['longitude']
        mainshock_mag = mainshock['magnitude']

        # Extract features
        features = extract_mainshock_features(df, mainshock)

        # Calculate target: did a significant aftershock occur?
        aftershock_radius = wells_coppersmith_radius(mainshock_mag)

        aftershock_mask = (df['time'] > mainshock_time) & \
                         (df['time'] <= mainshock_time + timedelta(days=target_window_days)) & \
                         (df['magnitude'] >= target_magnitude) & \
                         (df['magnitude'] < mainshock_mag)

        if aftershock_mask.sum() > 0:
            potential_aftershocks = df[aftershock_mask]
            distances = potential_aftershocks.apply(
                lambda row: calculate_distance(mainshock_lat, mainshock_lon, row['latitude'], row['longitude']),
                axis=1
            )
            significant_aftershock = (distances <= aftershock_radius).any()
        else:
            significant_aftershock = False

        features_list.append(features)
        targets.append(1 if significant_aftershock else 0)

    # Convert to arrays
    feature_names = list(features_list[0].keys())
    X = np.array([[f[name] for name in feature_names] for f in features_list])
    y = np.array(targets)

    logger.info(f"Prepared {len(X)} training samples, {sum(y)} positive ({100*sum(y)/len(y):.1f}%)")

    return X, y, feature_names


def run_training_pipeline(
    input_file: str,
    output_dir: str,
    model_type: str = 'random_forest',
    mainshock_threshold: float = 5.0,
    target_magnitude: float = 5.0,
    n_estimators: int = 100,
    val_split: float = 0.2
) -> Dict:
    """
    Run complete training pipeline.

    Args:
        input_file: Path to earthquake catalog CSV
        output_dir: Output directory for model files
        model_type: 'random_forest' or 'gradient_boosting'
        mainshock_threshold: Minimum mainshock magnitude
        target_magnitude: Target aftershock magnitude threshold
        n_estimators: Number of trees in ensemble
        val_split: Validation split fraction

    Returns:
        Training results dictionary
    """
    logger.info(f"Starting training pipeline...")

    # Load data
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} events from {input_file}")

    # Prepare training data
    X, y, feature_names = prepare_training_data(
        df,
        mainshock_threshold=mainshock_threshold,
        target_magnitude=target_magnitude
    )

    if len(X) < 20:
        logger.error(f"Insufficient training data: {len(X)} samples (need at least 20)")
        return {'error': 'Insufficient training data'}

    # Train model
    predictor = AftershockPredictor(
        model_type=model_type,
        target_magnitude=target_magnitude,
        n_estimators=n_estimators
    )

    metrics = predictor.train(X, y, feature_names, val_split=val_split)

    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_file = output_path / f'aftershock_model_{model_type}.pkl'
    predictor.save(str(model_file))

    # Save training report
    report = {
        'metadata': {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'input_file': input_file,
            'model_type': model_type,
            'target_magnitude': target_magnitude,
            'n_estimators': n_estimators
        },
        'training_metrics': metrics,
        'feature_importance': predictor.get_feature_importance()
    }

    report_file = output_path / 'training_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Training complete. Model saved to {model_file}")

    return report


# =============================================================================
# Inference Pipeline
# =============================================================================

def run_inference_pipeline(
    input_file: str,
    output_file: str,
    model_path: Optional[str] = None,
    mainshock_threshold: float = 5.0,
    magnitude_thresholds: List[float] = [4.0, 5.0, 6.0],
    time_windows_days: List[int] = [1, 7, 30],
    use_statistical: bool = True,
    use_ml: bool = True
) -> Dict:
    """
    Run aftershock prediction inference pipeline.

    Args:
        input_file: Path to earthquake catalog CSV
        output_file: Output JSON file path
        model_path: Path to trained ML model (optional)
        mainshock_threshold: Minimum magnitude for mainshock identification
        magnitude_thresholds: Magnitude thresholds for predictions
        time_windows_days: Time windows for predictions (days)
        use_statistical: Include statistical model predictions
        use_ml: Include ML model predictions

    Returns:
        Prediction report dictionary
    """
    logger.info(f"Starting aftershock prediction inference...")

    # Load data
    df = pd.read_csv(input_file)
    df['time'] = pd.to_datetime(df['time'], format='mixed', utc=True)
    logger.info(f"Loaded {len(df)} events from {input_file}")

    # Identify mainshocks
    mainshocks = identify_mainshocks(df, magnitude_threshold=mainshock_threshold)

    if len(mainshocks) == 0:
        logger.warning("No mainshocks found in catalog")
        report = {
            'metadata': {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'total_events_analyzed': int(len(df)),
                'mainshocks_identified': 0,
                'warning': f'No events with magnitude >= {mainshock_threshold}'
            },
            'mainshock_predictions': []
        }
        return report

    # Load ML model if specified
    ml_predictor = None
    if use_ml and model_path:
        try:
            ml_predictor = AftershockPredictor.load(model_path)
            logger.info(f"Loaded ML model from {model_path}")
        except Exception as e:
            logger.warning(f"Could not load ML model: {e}. Using statistical only.")
            use_ml = False

    # Generate predictions for each mainshock
    predictions = []

    for idx, mainshock in mainshocks.iterrows():
        mainshock_time = pd.to_datetime(mainshock['time'])

        prediction = {
            'mainshock_id': str(mainshock.get('id', f'mainshock_{idx}')),
            'mainshock_time': mainshock_time.isoformat(),
            'mainshock_magnitude': float(mainshock['magnitude']),
            'mainshock_depth_km': float(mainshock.get('depth_km', 0)),
            'mainshock_location': {
                'latitude': float(mainshock['latitude']),
                'longitude': float(mainshock['longitude']),
                'place': str(mainshock.get('place', 'Unknown'))
            },
            'aftershock_zone': {
                'radius_km': float(mainshock['aftershock_zone_radius_km']),
                'area_km2': float(np.pi * mainshock['aftershock_zone_radius_km'] ** 2)
            },
            'predictions': {}
        }

        # Extract features
        features = extract_mainshock_features(df, mainshock)
        b_value = features.get('historical_b_value', DEFAULT_B_VALUE)

        # Statistical predictions
        if use_statistical:
            prediction['predictions']['statistical'] = predict_statistical(
                mainshock, b_value, magnitude_thresholds, time_windows_days
            )

        # ML predictions
        if use_ml and ml_predictor:
            try:
                X = np.array([[features[name] for name in ml_predictor.feature_names]])
                prob = float(ml_predictor.predict(X)[0])

                prediction['predictions']['ml'] = {
                    'model_type': ml_predictor.model_type,
                    'target_magnitude': ml_predictor.target_magnitude,
                    'time_window_days': ml_predictor.time_window_days,
                    'probability': float(prob),
                    'risk_level': 'high' if prob >= 0.7 else 'moderate' if prob >= 0.3 else 'low'
                }
            except Exception as e:
                logger.warning(f"ML prediction failed for mainshock {idx}: {e}")

        # Combined prediction (average if both available)
        if use_statistical and use_ml and 'ml' in prediction['predictions']:
            stat_prob = prediction['predictions']['statistical']['probability'].get('M>=5.0_within_7_days', 0.5)
            ml_prob = prediction['predictions']['ml']['probability']
            combined_prob = (stat_prob + ml_prob) / 2

            prediction['predictions']['combined'] = {
                'probability_M5_7days': float(combined_prob),
                'risk_level': 'high' if combined_prob >= 0.7 else 'moderate' if combined_prob >= 0.3 else 'low'
            }
        elif use_statistical:
            stat_prob = prediction['predictions']['statistical']['probability'].get('M>=5.0_within_7_days', 0.5)
            prediction['predictions']['combined'] = {
                'probability_M5_7days': float(stat_prob),
                'risk_level': 'high' if stat_prob >= 0.7 else 'moderate' if stat_prob >= 0.3 else 'low'
            }

        predictions.append(prediction)

    # Build report
    report = {
        'metadata': {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'analysis_version': '1.0',
            'total_events_analyzed': int(len(df)),
            'mainshocks_identified': int(len(mainshocks)),
            'prediction_settings': {
                'mainshock_threshold': mainshock_threshold,
                'magnitude_thresholds': magnitude_thresholds,
                'time_windows_days': time_windows_days,
                'statistical_model': 'omori_utsu' if use_statistical else None,
                'ml_model': ml_predictor.model_type if ml_predictor else None
            }
        },
        'mainshock_predictions': predictions
    }

    # Add feature importance if ML model used
    if ml_predictor:
        report['feature_importance'] = ml_predictor.get_feature_importance()
        if ml_predictor.training_metrics:
            report['model_performance'] = ml_predictor.training_metrics

    # Summary statistics
    if predictions:
        probs = [p['predictions']['combined']['probability_M5_7days'] for p in predictions
                 if 'combined' in p['predictions']]

        report['summary'] = {
            'total_mainshocks': int(len(predictions)),
            'high_risk_mainshocks': int(sum(1 for p in probs if p >= 0.7)),
            'moderate_risk_mainshocks': int(sum(1 for p in probs if 0.3 <= p < 0.7)),
            'low_risk_mainshocks': int(sum(1 for p in probs if p < 0.3)),
            'average_m5_probability': float(np.mean(probs)) if probs else 0.0
        }

    # Save report
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Predictions saved to {output_file}")

    # Print summary
    print(f"\n{'='*70}")
    print("AFTERSHOCK PREDICTION SUMMARY")
    print(f"{'='*70}")
    print(f"Total Events Analyzed: {len(df)}")
    print(f"Mainshocks Identified: {len(mainshocks)}")
    if 'summary' in report:
        print(f"\nRisk Assessment:")
        print(f"  High Risk (P >= 0.7): {report['summary']['high_risk_mainshocks']}")
        print(f"  Moderate Risk (0.3 <= P < 0.7): {report['summary']['moderate_risk_mainshocks']}")
        print(f"  Low Risk (P < 0.3): {report['summary']['low_risk_mainshocks']}")
        print(f"  Average M5+ Probability: {report['summary']['average_m5_probability']:.1%}")
    print(f"\nPredictions saved to: {output_file}")
    print(f"{'='*70}\n")

    return report


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Predict aftershock probabilities using statistical and ML models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inference mode with statistical models only
  %(prog)s --input catalog.csv --output predictions.json

  # Inference with ML model
  %(prog)s --input catalog.csv --output predictions.json \\
           --model-path models/aftershock_model.pkl

  # Training mode
  %(prog)s --input historical_catalog.csv --output models/ --mode train \\
           --model-type random_forest --n-estimators 200

  # Custom thresholds
  %(prog)s --input catalog.csv --output predictions.json \\
           --magnitude-thresholds 4.5 5.0 5.5 6.0 --time-windows 1 7 14 30
        """
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=['inference', 'train'],
        default='inference',
        help="Operating mode: 'inference' for predictions, 'train' for model training"
    )

    # Input/Output
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
        help="Output file (JSON for inference, directory for training)"
    )

    # Model configuration
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained ML model (for inference mode)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=['random_forest', 'gradient_boosting'],
        default='random_forest',
        help="ML model type for training (default: random_forest)"
    )

    # Prediction parameters
    parser.add_argument(
        "--mainshock-threshold",
        type=float,
        default=5.0,
        help="Minimum magnitude to identify as mainshock (default: 5.0)"
    )
    parser.add_argument(
        "--magnitude-thresholds",
        type=float,
        nargs='+',
        default=[4.0, 5.0, 6.0],
        help="Magnitude thresholds for predictions (default: 4.0 5.0 6.0)"
    )
    parser.add_argument(
        "--time-windows",
        type=int,
        nargs='+',
        default=[1, 7, 30],
        help="Time windows in days for predictions (default: 1 7 30)"
    )

    # Training parameters
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in ensemble (default: 100)"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split fraction (default: 0.2)"
    )
    parser.add_argument(
        "--target-magnitude",
        type=float,
        default=5.0,
        help="Target magnitude threshold for training (default: 5.0)"
    )

    # Statistical model options
    parser.add_argument(
        "--no-statistical",
        action='store_true',
        help="Disable statistical model predictions"
    )
    parser.add_argument(
        "--no-ml",
        action='store_true',
        help="Disable ML model predictions"
    )

    args = parser.parse_args()

    try:
        if args.mode == 'train':
            # Training mode
            run_training_pipeline(
                input_file=args.input,
                output_dir=args.output,
                model_type=args.model_type,
                mainshock_threshold=args.mainshock_threshold,
                target_magnitude=args.target_magnitude,
                n_estimators=args.n_estimators,
                val_split=args.val_split
            )
        else:
            # Inference mode
            run_inference_pipeline(
                input_file=args.input,
                output_file=args.output,
                model_path=args.model_path,
                mainshock_threshold=args.mainshock_threshold,
                magnitude_thresholds=args.magnitude_thresholds,
                time_windows_days=args.time_windows,
                use_statistical=not args.no_statistical,
                use_ml=not args.no_ml and args.model_path is not None
            )

        logger.info("Aftershock prediction completed successfully")

    except FileNotFoundError:
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to predict aftershocks: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
