"""
A/B Testing Framework
Statistical comparison of model versions in production.
Routes traffic, collects outcomes, and evaluates significance.
"""

import numpy as np
from datetime import datetime
from typing import Dict, List
from loguru import logger
from scipy import stats
import json

from src.config import settings


class ABTestManager:
    """Manages A/B tests between model versions."""

    def __init__(self):
        self.tests_dir = settings.data_dir / "ab_tests"
        self.tests_dir.mkdir(parents=True, exist_ok=True)

    def create_test(
        self,
        test_name: str,
        model_name: str,
        version_a: str,
        version_b: str,
        traffic_split: float = 0.5,
        primary_metric: str = "accuracy",
        min_sample_size: int = None,
    ) -> Dict:
        """Create a new A/B test."""
        test = {
            "test_name": test_name,
            "model_name": model_name,
            "version_a": version_a,
            "version_b": version_b,
            "traffic_split": traffic_split,
            "primary_metric": primary_metric,
            "min_sample_size": min_sample_size or settings.ab_min_sample_size,
            "significance_level": settings.ab_significance_level,
            "status": "running",
            "created_at": datetime.now().isoformat(),
            "results_a": [],
            "results_b": [],
            "outcomes_a": [],
            "outcomes_b": [],
        }

        # Save test config
        test_path = self.tests_dir / f"{test_name}_config.json"
        with open(test_path, "w") as f:
            json.dump(test, f, indent=2, default=str)

        logger.info(f"Created A/B test '{test_name}': v{version_a} vs v{version_b}")
        return test

    def route_request(self, test_name: str) -> str:
        """Route a request to version A or B based on traffic split."""
        test_path = self.tests_dir / f"{test_name}_config.json"
        if not test_path.exists():
            return "A"

        with open(test_path) as f:
            test = json.load(f)

        if np.random.random() < test["traffic_split"]:
            return "B"
        return "A"

    def record_result(
        self,
        test_name: str,
        version: str,
        prediction: float,
        actual: float = None,
    ) -> Dict:
        """Record a prediction result for the A/B test."""
        test_path = self.tests_dir / f"{test_name}_config.json"
        if not test_path.exists():
            return {"error": f"Test '{test_name}' not found"}

        with open(test_path) as f:
            test = json.load(f)

        if version == "A":
            test["results_a"].append(float(prediction))
            if actual is not None:
                test["outcomes_a"].append({"prediction": float(prediction), "actual": float(actual)})
        elif version == "B":
            test["results_b"].append(float(prediction))
            if actual is not None:
                test["outcomes_b"].append({"prediction": float(prediction), "actual": float(actual)})

        # Save updated test
        with open(test_path, "w") as f:
            json.dump(test, f, indent=2, default=str)

        return {
            "recorded": True,
            "version": version,
            "total_a": len(test["results_a"]),
            "total_b": len(test["results_b"]),
        }

    def evaluate_test(self, test_name: str) -> Dict:
        """
        Evaluate A/B test results using statistical tests.

        Returns:
            Dict with statistical test results and recommendation.
        """
        test_path = self.tests_dir / f"{test_name}_config.json"
        if not test_path.exists():
            return {"error": f"Test '{test_name}' not found"}

        with open(test_path) as f:
            test = json.load(f)

        results_a = test["results_a"]
        results_b = test["results_b"]
        outcomes_a = test["outcomes_a"]
        outcomes_b = test["outcomes_b"]

        n_a = len(results_a)
        n_b = len(results_b)

        if n_a < test["min_sample_size"] or n_b < test["min_sample_size"]:
            return {
                "test_name": test_name,
                "status": "insufficient_data",
                "sample_a": n_a,
                "sample_b": n_b,
                "min_required": test["min_sample_size"],
                "message": f"Need at least {test['min_sample_size']} samples per variant",
            }

        # Calculate metrics if we have outcomes
        if outcomes_a and outcomes_b:
            # Accuracy
            preds_a = np.array([o["prediction"] for o in outcomes_a])
            acts_a = np.array([o["actual"] for o in outcomes_a])
            preds_b = np.array([o["prediction"] for o in outcomes_b])
            acts_b = np.array([o["actual"] for o in outcomes_b])

            # For classification (threshold at 0.5)
            acc_a = np.mean((preds_a > 0.5).astype(int) == acts_a.astype(int))
            acc_b = np.mean((preds_b > 0.5).astype(int) == acts_b.astype(int))

            # Two-proportion z-test
            p1 = acc_a
            p2 = acc_b
            p_pool = (p1 * n_a + p2 * n_b) / (n_a + n_b)
            se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))

            if se > 0:
                z_stat = (p2 - p1) / se
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            else:
                z_stat = 0
                p_value = 1.0

            # Effect size (Cohen's h)
            h = 2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1))

            significant = p_value < test["significance_level"]

            # Recommendation
            if significant and acc_b > acc_a:
                recommendation = "deploy_b"
                message = f"Version B is significantly better ({acc_b:.4f} vs {acc_a:.4f}, p={p_value:.6f}). Deploy B."
            elif significant and acc_b < acc_a:
                recommendation = "keep_a"
                message = f"Version A is significantly better ({acc_a:.4f} vs {acc_b:.4f}, p={p_value:.6f}). Keep A."
            else:
                recommendation = "inconclusive"
                message = f"No significant difference (p={p_value:.6f}). Need more data or keep A."

            result = {
                "test_name": test_name,
                "status": "completed",
                "sample_a": n_a,
                "sample_b": n_b,
                "accuracy_a": round(float(acc_a), 6),
                "accuracy_b": round(float(acc_b), 6),
                "delta": round(float(acc_b - acc_a), 6),
                "z_statistic": round(float(z_stat), 6),
                "p_value": round(float(p_value), 6),
                "effect_size_h": round(float(h), 6),
                "significant": bool(significant),
                "recommendation": recommendation,
                "message": message,
                "evaluated_at": datetime.now().isoformat(),
            }
        else:
            # No outcomes - compare prediction distributions
            if len(results_a) > 1 and len(results_b) > 1:
                ks_stat, ks_pvalue = stats.ks_2samp(results_a, results_b)
                result = {
                    "test_name": test_name,
                    "status": "completed_no_outcomes",
                    "sample_a": n_a,
                    "sample_b": n_b,
                    "mean_a": round(float(np.mean(results_a)), 6),
                    "mean_b": round(float(np.mean(results_b)), 6),
                    "ks_statistic": round(float(ks_stat), 6),
                    "ks_p_value": round(float(ks_pvalue), 6),
                    "message": "No ground truth available. Compared prediction distributions only.",
                }
            else:
                result = {
                    "test_name": test_name,
                    "status": "insufficient_data",
                    "sample_a": n_a,
                    "sample_b": n_b,
                    "message": "Not enough data for evaluation",
                }

        # Save evaluation
        eval_path = self.tests_dir / f"{test_name}_evaluation.json"
        with open(eval_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        logger.info(f"A/B test '{test_name}' evaluated: {result.get('recommendation', result.get('status'))}")
        return result

    def list_tests(self) -> List[Dict]:
        """List all A/B tests."""
        tests = []
        for path in sorted(self.tests_dir.glob("*_config.json")):
            with open(path) as f:
                test = json.load(f)
                tests.append(
                    {
                        "test_name": test["test_name"],
                        "model_name": test["model_name"],
                        "version_a": test["version_a"],
                        "version_b": test["version_b"],
                        "status": test["status"],
                        "sample_a": len(test["results_a"]),
                        "sample_b": len(test["results_b"]),
                        "created_at": test["created_at"],
                    }
                )
        return tests

    def stop_test(self, test_name: str) -> Dict:
        """Stop an A/B test."""
        test_path = self.tests_dir / f"{test_name}_config.json"
        if not test_path.exists():
            return {"error": f"Test '{test_name}' not found"}

        with open(test_path) as f:
            test = json.load(f)
        test["status"] = "stopped"
        test["stopped_at"] = datetime.now().isoformat()
        with open(test_path, "w") as f:
            json.dump(test, f, indent=2, default=str)

        logger.info(f"A/B test '{test_name}' stopped")
        return {"test_name": test_name, "status": "stopped"}
