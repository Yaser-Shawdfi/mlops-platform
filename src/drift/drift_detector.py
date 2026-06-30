"""
Drift Detection Module
Uses Evidently for data drift and concept drift detection.
Generates reports, triggers alerts, and feeds into retraining pipeline.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict
from loguru import logger
import json

from src.config import settings


class DriftDetector:
    """Detect data drift using statistical tests and Evidently."""

    def __init__(self):
        self.reports_dir = settings.reports_dir / "drift"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.threshold_psi = settings.drift_threshold_psi
        self.threshold_ks = settings.drift_threshold_ks

    @staticmethod
    def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI).
        PSI < 0.1: No drift
        PSI 0.1-0.2: Moderate drift
        PSI > 0.2: Significant drift
        """
        expected = np.array(expected)
        actual = np.array(actual)

        # Handle NaN
        expected = expected[~np.isnan(expected)]
        actual = actual[~np.isnan(actual)]

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        expected_perc = np.percentile(expected, breakpoints)
        actual_perc = np.percentile(actual, breakpoints)

        # Ensure unique breakpoints
        expected_perc = np.unique(expected_perc)
        actual_perc = np.unique(actual_perc)

        if len(expected_perc) < 3:
            return 0.0

        # Calculate proportions
        expected_counts = np.histogram(expected, bins=expected_perc)[0]
        actual_counts = np.histogram(actual, bins=expected_perc)[0]

        expected_proportions = expected_counts / len(expected)
        actual_proportions = actual_counts / len(actual)

        # Add small epsilon to avoid division by zero
        expected_proportions = np.where(expected_proportions == 0, 0.0001, expected_proportions)
        actual_proportions = np.where(actual_proportions == 0, 0.0001, actual_proportions)

        psi = np.sum((actual_proportions - expected_proportions) * np.log(actual_proportions / expected_proportions))
        return float(psi)

    def detect_feature_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        numerical_features: List[str] = None,
        categorical_features: List[str] = None,
    ) -> Dict:
        """
        Detect drift for each feature.

        Returns dict with per-feature PSI values and drift status.
        """
        if numerical_features is None:
            numerical_features = reference_data.select_dtypes(include=[np.number]).columns.tolist()
        if categorical_features is None:
            categorical_features = reference_data.select_dtypes(exclude=[np.number]).columns.tolist()

        results = {
            "timestamp": datetime.now().isoformat(),
            "reference_size": len(reference_data),
            "current_size": len(current_data),
            "features": {},
            "drift_detected": False,
            "drifted_features": [],
        }

        # Numerical features - PSI
        for col in numerical_features:
            if col not in current_data.columns:
                continue
            psi = self.calculate_psi(reference_data[col].values, current_data[col].values)
            drifted = psi > self.threshold_psi
            results["features"][col] = {
                "type": "numerical",
                "psi": round(psi, 6),
                "drifted": drifted,
                "threshold": self.threshold_psi,
            }
            if drifted:
                results["drifted_features"].append(col)
                results["drift_detected"] = True

        # Categorical features - distribution shift
        for col in categorical_features:
            if col not in current_data.columns:
                continue
            ref_dist = reference_data[col].value_counts(normalize=True)
            cur_dist = current_data[col].value_counts(normalize=True)

            # Align distributions
            all_cats = set(ref_dist.index) | set(cur_dist.index)
            ref_vals = np.array([ref_dist.get(c, 0.0001) for c in all_cats])
            cur_vals = np.array([cur_dist.get(c, 0.0001) for c in all_cats])

            # PSI for categorical
            psi = float(np.sum((cur_vals - ref_vals) * np.log(cur_vals / ref_vals)))
            drifted = psi > self.threshold_psi
            results["features"][col] = {
                "type": "categorical",
                "psi": round(psi, 6),
                "drifted": drifted,
                "threshold": self.threshold_psi,
            }
            if drifted:
                results["drifted_features"].append(col)
                results["drift_detected"] = True

        # Save report
        report_path = self.reports_dir / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        results["report_path"] = str(report_path)

        if results["drift_detected"]:
            logger.warning(
                f"Drift detected in {len(results['drifted_features'])} features: {results['drifted_features']}"
            )
        else:
            logger.info("No drift detected across all features")

        return results

    def detect_target_drift(
        self,
        reference_targets: np.ndarray,
        current_targets: np.ndarray,
    ) -> Dict:
        """
        Detect concept drift by comparing target distributions.
        Uses Kolmogorov-Smirnov test for numerical, chi-square for categorical.
        """
        from scipy import stats

        # KS test for numerical targets
        if len(np.unique(reference_targets)) > 2:
            ks_stat, ks_pvalue = stats.ks_2samp(reference_targets, current_targets)
            drifted = ks_pvalue < self.threshold_ks
            test_type = "kolmogorov_smirnov"
        else:
            # Chi-square for binary/categorical
            ref_counts = np.bincount(reference_targets.astype(int))
            cur_counts = np.bincount(current_targets.astype(int))
            min_len = min(len(ref_counts), len(cur_counts))
            chi_stat, chi_pvalue = stats.chisquare(cur_counts[:min_len], ref_counts[:min_len] + 1e-10)
            drifted = chi_pvalue < self.threshold_ks
            test_type = "chi_square"
            ks_stat = float(chi_stat)
            ks_pvalue = float(chi_pvalue)

        result = {
            "timestamp": datetime.now().isoformat(),
            "test": test_type,
            "statistic": float(ks_stat),
            "p_value": float(ks_pvalue),
            "drifted": drifted,
            "threshold": self.threshold_ks,
        }

        if drifted:
            logger.warning(f"Target drift detected: p-value={ks_pvalue:.6f}")
        else:
            logger.info(f"No target drift: p-value={ks_pvalue:.6f}")

        return result

    def detect_prediction_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray,
    ) -> Dict:
        """Detect drift in model predictions (output distribution shift)."""
        psi = self.calculate_psi(reference_predictions, current_predictions)
        drifted = psi > self.threshold_psi
        result = {
            "timestamp": datetime.now().isoformat(),
            "psi": round(psi, 6),
            "drifted": drifted,
            "threshold": self.threshold_psi,
        }

        if drifted:
            logger.warning(f"Prediction drift detected: PSI={psi:.6f}")
        else:
            logger.info(f"No prediction drift: PSI={psi:.6f}")

        return result

    def generate_drift_report(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> Dict:
        """Generate a comprehensive drift report combining all drift types."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "feature_drift": self.detect_feature_drift(reference_data, current_data),
        }

        # Save full report
        report_path = self.reports_dir / f"full_drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        report["report_path"] = str(report_path)
        return report
