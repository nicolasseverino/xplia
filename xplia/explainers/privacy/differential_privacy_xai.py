"""
Privacy-Preserving XAI with Differential Privacy.

Implements differential privacy mechanisms for explanations to protect
sensitive information while maintaining explanation utility.

Author: XPLIA Team
License: MIT
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import warnings

try:
    from scipy import stats
except ImportError:
    stats = None

from xplia.core.base import ExplainerBase, ExplanationResult


@dataclass
class PrivacyBudget:
    """
    Privacy budget for differential privacy.

    Attributes
    ----------
    epsilon : float
        Privacy parameter (lower = more private).
    delta : float
        Failure probability.
    spent : float
        Already spent privacy budget.
    """
    epsilon: float
    delta: float
    spent: float = 0.0

    def has_budget(self, required_epsilon: float) -> bool:
        """Check if sufficient budget remains."""
        return (self.spent + required_epsilon) <= self.epsilon

    def spend(self, amount: float):
        """Spend privacy budget."""
        if not self.has_budget(amount):
            raise ValueError(f"Insufficient privacy budget. Required: {amount}, Available: {self.epsilon - self.spent}")
        self.spent += amount

    def remaining(self) -> float:
        """Get remaining budget."""
        return self.epsilon - self.spent

    def reset(self):
        """Reset spent budget."""
        self.spent = 0.0


class LaplaceMechanism:
    """
    Laplace mechanism for differential privacy.

    Adds Laplace noise calibrated to sensitivity and epsilon.

    Parameters
    ----------
    epsilon : float
        Privacy parameter.
    sensitivity : float
        L1 sensitivity of the query.

    Examples
    --------
    >>> mechanism = LaplaceMechanism(epsilon=1.0, sensitivity=1.0)
    >>> noisy_value = mechanism.add_noise(true_value)
    """

    def __init__(self, epsilon: float, sensitivity: float):
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if sensitivity < 0:
            raise ValueError("Sensitivity must be non-negative")

        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.scale = sensitivity / epsilon

    def add_noise(self, value: np.ndarray) -> np.ndarray:
        """Add Laplace noise to value."""
        noise = np.random.laplace(0, self.scale, size=value.shape)
        return value + noise

    def add_noise_scalar(self, value: float) -> float:
        """Add Laplace noise to scalar value."""
        noise = np.random.laplace(0, self.scale)
        return value + noise


class GaussianMechanism:
    """
    Gaussian mechanism for (epsilon, delta)-differential privacy.

    Adds Gaussian noise for approximate differential privacy.

    Parameters
    ----------
    epsilon : float
        Privacy parameter.
    delta : float
        Failure probability.
    sensitivity : float
        L2 sensitivity of the query.

    Examples
    --------
    >>> mechanism = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
    >>> noisy_value = mechanism.add_noise(true_value)
    """

    def __init__(self, epsilon: float, delta: float, sensitivity: float):
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if delta <= 0 or delta >= 1:
            raise ValueError("Delta must be in (0, 1)")
        if sensitivity < 0:
            raise ValueError("Sensitivity must be non-negative")

        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity

        # Calculate sigma for (epsilon, delta)-DP
        # sigma >= sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon
        self.sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon

    def add_noise(self, value: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to value."""
        noise = np.random.normal(0, self.sigma, size=value.shape)
        return value + noise

    def add_noise_scalar(self, value: float) -> float:
        """Add Gaussian noise to scalar value."""
        noise = np.random.normal(0, self.sigma)
        return value + noise


class ExponentialMechanism:
    """
    Exponential mechanism for differential privacy.

    Selects from discrete set with probability proportional to utility.

    Parameters
    ----------
    epsilon : float
        Privacy parameter.
    sensitivity : float
        Sensitivity of utility function.

    Examples
    --------
    >>> mechanism = ExponentialMechanism(epsilon=1.0, sensitivity=1.0)
    >>> selected = mechanism.select(candidates, utility_scores)
    """

    def __init__(self, epsilon: float, sensitivity: float):
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if sensitivity < 0:
            raise ValueError("Sensitivity must be non-negative")

        self.epsilon = epsilon
        self.sensitivity = sensitivity

    def select(self, candidates: List[Any], utilities: np.ndarray) -> Any:
        """
        Select candidate with probability proportional to utility.

        Parameters
        ----------
        candidates : list
            List of candidate items.
        utilities : ndarray
            Utility scores for each candidate.

        Returns
        -------
        selected : Any
            Selected candidate.
        """
        if len(candidates) != len(utilities):
            raise ValueError("Candidates and utilities must have same length")

        # Compute probabilities
        scaled_utilities = (self.epsilon * utilities) / (2 * self.sensitivity)

        # Numerical stability: subtract max
        scaled_utilities = scaled_utilities - np.max(scaled_utilities)

        probabilities = np.exp(scaled_utilities)
        probabilities = probabilities / np.sum(probabilities)

        # Sample
        selected_idx = np.random.choice(len(candidates), p=probabilities)
        return candidates[selected_idx]


class DPFeatureImportanceExplainer(ExplainerBase):
    """
    Differentially private feature importance explanations.

    Computes feature importance with differential privacy guarantees.

    Parameters
    ----------
    base_explainer : ExplainerBase
        Base explainer to make private.
    privacy_budget : PrivacyBudget
        Privacy budget.
    mechanism : str
        'laplace' or 'gaussian'.
    clip_threshold : float, optional
        Clipping threshold for gradients/importance.

    Examples
    --------
    >>> from xplia.explainers.shap import SHAPExplainer
    >>> base = SHAPExplainer(model)
    >>> budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
    >>> dp_explainer = DPFeatureImportanceExplainer(base, budget)
    >>> explanation = dp_explainer.explain(X_test[0])
    """

    def __init__(
        self,
        base_explainer: ExplainerBase,
        privacy_budget: PrivacyBudget,
        mechanism: str = 'gaussian',
        clip_threshold: float = 1.0
    ):
        super().__init__(base_explainer.model)
        self.base_explainer = base_explainer
        self.privacy_budget = privacy_budget
        self.mechanism_type = mechanism
        self.clip_threshold = clip_threshold

        if mechanism not in ['laplace', 'gaussian']:
            raise ValueError("Mechanism must be 'laplace' or 'gaussian'")

    def _clip_importance(self, importance: np.ndarray) -> np.ndarray:
        """Clip feature importance to bound sensitivity."""
        norm = np.linalg.norm(importance, ord=2)
        if norm > self.clip_threshold:
            importance = importance * (self.clip_threshold / norm)
        return importance

    def explain(
        self,
        X: np.ndarray,
        epsilon_per_query: Optional[float] = None,
        **kwargs
    ) -> ExplanationResult:
        """
        Generate differentially private explanation.

        Parameters
        ----------
        X : ndarray
            Instance to explain.
        epsilon_per_query : float, optional
            Epsilon to use for this query. If None, uses remaining budget.
        **kwargs
            Additional arguments for base explainer.

        Returns
        -------
        explanation : ExplanationResult
            Private explanation with noise added.
        """
        # Get base explanation
        base_explanation = self.base_explainer.explain(X, **kwargs)

        # Extract feature importance
        if 'feature_importance' not in base_explanation.explanation_data:
            raise ValueError("Base explainer must provide feature_importance")

        importance = np.array(base_explanation.explanation_data['feature_importance'])

        # Clip for bounded sensitivity
        importance_clipped = self._clip_importance(importance)

        # Determine epsilon for this query
        if epsilon_per_query is None:
            epsilon_per_query = self.privacy_budget.remaining()

        if not self.privacy_budget.has_budget(epsilon_per_query):
            raise ValueError(f"Insufficient privacy budget for query")

        # Add noise
        if self.mechanism_type == 'laplace':
            # L1 sensitivity = clip_threshold (after clipping to L2 ball)
            mechanism = LaplaceMechanism(
                epsilon=epsilon_per_query,
                sensitivity=self.clip_threshold
            )
            noisy_importance = mechanism.add_noise(importance_clipped)
        else:  # gaussian
            # L2 sensitivity = clip_threshold
            mechanism = GaussianMechanism(
                epsilon=epsilon_per_query,
                delta=self.privacy_budget.delta,
                sensitivity=self.clip_threshold
            )
            noisy_importance = mechanism.add_noise(importance_clipped)

        # Spend budget
        self.privacy_budget.spend(epsilon_per_query)

        # Create private explanation
        private_data = base_explanation.explanation_data.copy()
        private_data['feature_importance'] = noisy_importance.tolist()
        private_data['privacy'] = {
            'epsilon': epsilon_per_query,
            'delta': self.privacy_budget.delta if self.mechanism_type == 'gaussian' else 0.0,
            'mechanism': self.mechanism_type,
            'clip_threshold': self.clip_threshold,
            'budget_remaining': self.privacy_budget.remaining()
        }

        return ExplanationResult(
            method=f"dp_{base_explanation.method}",
            explanation_data=private_data,
            metadata=base_explanation.metadata
        )


class DPAggregatedExplainer:
    """
    Aggregate explanations from multiple samples with differential privacy.

    Useful for computing average feature importance across a dataset
    while preserving privacy.

    Parameters
    ----------
    base_explainer : ExplainerBase
        Base explainer.
    privacy_budget : PrivacyBudget
        Privacy budget.
    n_samples : int
        Number of samples to aggregate.
    mechanism : str
        'laplace' or 'gaussian'.

    Examples
    --------
    >>> dp_agg = DPAggregatedExplainer(shap_explainer, budget, n_samples=100)
    >>> avg_explanation = dp_agg.aggregate_explanations(X_test)
    """

    def __init__(
        self,
        base_explainer: ExplainerBase,
        privacy_budget: PrivacyBudget,
        n_samples: int,
        mechanism: str = 'gaussian'
    ):
        self.base_explainer = base_explainer
        self.privacy_budget = privacy_budget
        self.n_samples = n_samples
        self.mechanism = mechanism

    def aggregate_explanations(
        self,
        X: np.ndarray,
        epsilon: Optional[float] = None
    ) -> ExplanationResult:
        """
        Compute differentially private aggregated explanation.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples to aggregate over.
        epsilon : float, optional
            Privacy budget for aggregation.

        Returns
        -------
        explanation : ExplanationResult
            Aggregated private explanation.
        """
        if X.shape[0] != self.n_samples:
            warnings.warn(f"Expected {self.n_samples} samples, got {X.shape[0]}")

        # Collect base explanations
        importance_list = []
        for i in range(X.shape[0]):
            exp = self.base_explainer.explain(X[i:i+1])
            if 'feature_importance' in exp.explanation_data:
                importance_list.append(np.array(exp.explanation_data['feature_importance']))

        if not importance_list:
            raise ValueError("No feature importances found")

        # Stack and compute mean
        importance_matrix = np.stack(importance_list, axis=0)
        mean_importance = np.mean(importance_matrix, axis=0)

        # Sensitivity of mean: max individual contribution / n
        # Assuming bounded importance [-1, 1] per feature
        sensitivity = 2.0 / X.shape[0]

        if epsilon is None:
            epsilon = self.privacy_budget.remaining()

        # Add noise
        if self.mechanism == 'laplace':
            mech = LaplaceMechanism(epsilon=epsilon, sensitivity=sensitivity)
            noisy_mean = mech.add_noise(mean_importance)
        else:
            mech = GaussianMechanism(
                epsilon=epsilon,
                delta=self.privacy_budget.delta,
                sensitivity=sensitivity
            )
            noisy_mean = mech.add_noise(mean_importance)

        self.privacy_budget.spend(epsilon)

        return ExplanationResult(
            method=f"dp_aggregated_{self.base_explainer.method if hasattr(self.base_explainer, 'method') else 'unknown'}",
            explanation_data={
                'feature_importance': noisy_mean.tolist(),
                'n_samples': X.shape[0],
                'privacy': {
                    'epsilon': epsilon,
                    'delta': self.privacy_budget.delta if self.mechanism == 'gaussian' else 0.0,
                    'mechanism': self.mechanism
                }
            },
            metadata={'aggregation': 'mean', 'privacy_preserving': True}
        )


def compute_privacy_loss(
    mechanism: str,
    epsilon: float,
    delta: float,
    n_queries: int
) -> Tuple[float, float]:
    """
    Compute cumulative privacy loss from multiple queries.

    Uses basic composition theorems for differential privacy.

    Parameters
    ----------
    mechanism : str
        'laplace' or 'gaussian'.
    epsilon : float
        Privacy parameter per query.
    delta : float
        Failure probability per query.
    n_queries : int
        Number of queries.

    Returns
    -------
    total_epsilon : float
        Total epsilon.
    total_delta : float
        Total delta.

    Examples
    --------
    >>> total_eps, total_delta = compute_privacy_loss('gaussian', 0.1, 1e-5, 100)
    """
    if mechanism == 'laplace':
        # Pure DP: simple composition
        total_epsilon = epsilon * n_queries
        total_delta = 0.0
    else:  # gaussian
        # Advanced composition for (epsilon, delta)-DP
        # Using strong composition theorem
        total_epsilon = epsilon * np.sqrt(2 * n_queries * np.log(1 / delta))
        total_delta = delta * n_queries

    return total_epsilon, total_delta


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Privacy-Preserving XAI with Differential Privacy - Example")
    print("=" * 80)

    # Generate synthetic data
    np.random.seed(42)
    n_samples, n_features = 1000, 10

    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)

    # Simple model
    class SimpleModel:
        def predict(self, X):
            return (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)

        def predict_proba(self, X):
            pos_prob = 1 / (1 + np.exp(-(X[:, 0] + X[:, 1] - X[:, 2])))
            return np.column_stack([1 - pos_prob, pos_prob])

    model = SimpleModel()

    # Create base explainer (simplified SHAP-like)
    class SimpleShapExplainer(ExplainerBase):
        def explain(self, X, **kwargs):
            # Simplified feature importance (just coefficients)
            importance = np.array([1.0, 1.0, -1.0] + [0.0] * 7)

            return ExplanationResult(
                method='simple_shap',
                explanation_data={
                    'feature_importance': importance.tolist(),
                    'feature_names': [f'f{i}' for i in range(10)]
                },
                metadata={'simplified': True}
            )

    base_explainer = SimpleShapExplainer(model)

    # Test instance
    x_test = X[0]

    print("\n1. NON-PRIVATE EXPLANATION")
    print("-" * 80)
    base_exp = base_explainer.explain(x_test)
    print(f"Method: {base_exp.method}")
    print(f"Feature importance: {base_exp.explanation_data['feature_importance']}")

    print("\n2. DIFFERENTIALLY PRIVATE EXPLANATION (Laplace)")
    print("-" * 80)
    budget = PrivacyBudget(epsilon=1.0, delta=0.0)
    dp_explainer_laplace = DPFeatureImportanceExplainer(
        base_explainer,
        budget,
        mechanism='laplace',
        clip_threshold=2.0
    )

    dp_exp_laplace = dp_explainer_laplace.explain(x_test, epsilon_per_query=0.5)
    print(f"Method: {dp_exp_laplace.method}")
    print(f"Noisy feature importance: {dp_exp_laplace.explanation_data['feature_importance']}")
    print(f"Privacy: epsilon={dp_exp_laplace.explanation_data['privacy']['epsilon']}, "
          f"mechanism={dp_exp_laplace.explanation_data['privacy']['mechanism']}")
    print(f"Budget remaining: {budget.remaining():.4f}")

    print("\n3. DIFFERENTIALLY PRIVATE EXPLANATION (Gaussian)")
    print("-" * 80)
    budget_gaussian = PrivacyBudget(epsilon=1.0, delta=1e-5)
    dp_explainer_gaussian = DPFeatureImportanceExplainer(
        base_explainer,
        budget_gaussian,
        mechanism='gaussian',
        clip_threshold=2.0
    )

    dp_exp_gaussian = dp_explainer_gaussian.explain(x_test, epsilon_per_query=0.5)
    print(f"Method: {dp_exp_gaussian.method}")
    print(f"Noisy feature importance: {dp_exp_gaussian.explanation_data['feature_importance']}")
    print(f"Privacy: epsilon={dp_exp_gaussian.explanation_data['privacy']['epsilon']}, "
          f"delta={dp_exp_gaussian.explanation_data['privacy']['delta']}")

    print("\n4. AGGREGATED PRIVATE EXPLANATION")
    print("-" * 80)
    budget_agg = PrivacyBudget(epsilon=2.0, delta=1e-5)
    dp_agg = DPAggregatedExplainer(
        base_explainer,
        budget_agg,
        n_samples=100,
        mechanism='gaussian'
    )

    agg_exp = dp_agg.aggregate_explanations(X[:100], epsilon=1.0)
    print(f"Method: {agg_exp.method}")
    print(f"Aggregated importance: {agg_exp.explanation_data['feature_importance']}")
    print(f"Samples aggregated: {agg_exp.explanation_data['n_samples']}")
    print(f"Privacy: epsilon={agg_exp.explanation_data['privacy']['epsilon']}")

    print("\n5. PRIVACY LOSS ANALYSIS")
    print("-" * 80)
    n_queries = 10
    eps_total, delta_total = compute_privacy_loss('gaussian', 0.1, 1e-5, n_queries)
    print(f"After {n_queries} queries with epsilon=0.1, delta=1e-5 each:")
    print(f"Total epsilon: {eps_total:.4f}")
    print(f"Total delta: {delta_total:.6f}")

    print("\n6. COMPARISON: Privacy vs Utility")
    print("-" * 80)
    base_importance = np.array(base_exp.explanation_data['feature_importance'])
    dp_importance_laplace = np.array(dp_exp_laplace.explanation_data['feature_importance'])
    dp_importance_gaussian = np.array(dp_exp_gaussian.explanation_data['feature_importance'])

    error_laplace = np.linalg.norm(base_importance - dp_importance_laplace)
    error_gaussian = np.linalg.norm(base_importance - dp_importance_gaussian)

    print(f"L2 error (Laplace, eps=0.5): {error_laplace:.4f}")
    print(f"L2 error (Gaussian, eps=0.5, delta=1e-5): {error_gaussian:.4f}")
    print(f"\nTop features preserved:")
    top_3_base = np.argsort(np.abs(base_importance))[-3:][::-1]
    top_3_dp = np.argsort(np.abs(dp_importance_gaussian))[-3:][::-1]
    print(f"Base: {top_3_base}")
    print(f"DP:   {top_3_dp}")
    print(f"Overlap: {len(set(top_3_base) & set(top_3_dp))}/3")

    print("\n" + "=" * 80)
    print("Privacy-preserving XAI demonstration complete!")
    print("=" * 80)
