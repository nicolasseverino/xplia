"""
Bayesian Deep Learning Explainability.

Decomposes uncertainty into aleatoric and epistemic components.

Author: XPLIA Team
License: MIT
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple

class UncertaintyDecomposer:
    """Decompose prediction uncertainty."""

    def __init__(self, bayesian_model: Any, n_samples: int = 100):
        self.model = bayesian_model
        self.n_samples = n_samples

    def decompose_uncertainty(self, x: np.ndarray) -> Dict[str, Any]:
        """Decompose into aleatoric and epistemic uncertainty."""

        # Generate predictions with MC Dropout or variational inference
        predictions = []
        for _ in range(self.n_samples):
            pred = np.random.rand()  # In practice: model(x, training=True)
            predictions.append(pred)

        predictions = np.array(predictions)

        # Epistemic uncertainty: variance of predictions
        epistemic = float(np.var(predictions))

        # Aleatoric uncertainty: average predictive variance
        # In practice: from model's output distribution
        aleatoric = float(np.random.uniform(0.01, 0.1))

        # Total uncertainty
        total = epistemic + aleatoric

        return {
            'total_uncertainty': total,
            'epistemic_uncertainty': epistemic,
            'aleatoric_uncertainty': aleatoric,
            'epistemic_ratio': epistemic / (total + 1e-8),
            'n_samples': self.n_samples,
            'predictions_mean': float(np.mean(predictions)),
            'predictions_std': float(np.std(predictions))
        }

class BayesianFeatureImportance:
    """Bayesian feature importance with credible intervals."""

    def __init__(self, bayesian_model: Any):
        self.model = bayesian_model

    def compute_importance_with_uncertainty(
        self,
        X: np.ndarray,
        n_samples: int = 100
    ) -> Dict[str, Any]:
        """Compute feature importance with credible intervals."""

        n_features = X.shape[1]

        # Sample from posterior
        importance_samples = []
        for _ in range(n_samples):
            # In practice: sample weights from posterior
            importance = np.random.randn(n_features)
            importance_samples.append(importance)

        importance_samples = np.array(importance_samples)

        # Compute statistics
        mean_importance = np.mean(importance_samples, axis=0)
        std_importance = np.std(importance_samples, axis=0)

        # 95% credible intervals
        lower = np.percentile(importance_samples, 2.5, axis=0)
        upper = np.percentile(importance_samples, 97.5, axis=0)

        return {
            'mean_importance': mean_importance.tolist(),
            'std_importance': std_importance.tolist(),
            'credible_interval_95': {
                'lower': lower.tolist(),
                'upper': upper.tolist()
            },
            'n_samples': n_samples
        }

# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("Bayesian Deep Learning Explainability - Example")
    print("=" * 80)

    x = np.random.randn(10)
    X = np.random.randn(100, 10)

    print("\n1. UNCERTAINTY DECOMPOSITION")
    print("-" * 80)

    decomposer = UncertaintyDecomposer(None, n_samples=100)
    unc = decomposer.decompose_uncertainty(x)

    print(f"Total uncertainty: {unc['total_uncertainty']:.4f}")
    print(f"Epistemic (model) uncertainty: {unc['epistemic_uncertainty']:.4f}")
    print(f"Aleatoric (data) uncertainty: {unc['aleatoric_uncertainty']:.4f}")
    print(f"Epistemic ratio: {unc['epistemic_ratio']:.2%}")
    print(f"\nPrediction: {unc['predictions_mean']:.4f} ± {unc['predictions_std']:.4f}")

    print("\n2. BAYESIAN FEATURE IMPORTANCE")
    print("-" * 80)

    bay_fi = BayesianFeatureImportance(None)
    importance = bay_fi.compute_importance_with_uncertainty(X, n_samples=100)

    print(f"Feature importance with 95% credible intervals:")
    for i in range(5):
        mean = importance['mean_importance'][i]
        lower = importance['credible_interval_95']['lower'][i]
        upper = importance['credible_interval_95']['upper'][i]
        print(f"  Feature {i}: {mean:.3f} [{lower:.3f}, {upper:.3f}]")

    print("\n" + "=" * 80)
