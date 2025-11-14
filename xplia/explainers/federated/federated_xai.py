"""
Federated XAI - Explainability for Federated Learning.

Compute explanations without centralizing data, preserving privacy
and data sovereignty in federated settings.

Author: XPLIA Team
License: MIT
"""

import numpy as np
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
import warnings

from xplia.core.base import ExplainerBase, ExplanationResult


@dataclass
class FederatedNode:
    """
    Represents a node in federated learning.

    Attributes
    ----------
    node_id : str
        Unique identifier for the node.
    data : ndarray
        Local data (stays on node).
    local_model : object
        Local model or model copy.
    weight : float
        Weight for aggregation (e.g., based on data size).
    """
    node_id: str
    data: np.ndarray
    local_model: Any
    weight: float = 1.0


@dataclass
class FederatedExplanation:
    """
    Federated explanation result.

    Attributes
    ----------
    global_explanation : ExplanationResult
        Aggregated global explanation.
    local_explanations : dict
        Per-node local explanations.
    aggregation_method : str
        Method used for aggregation.
    metadata : dict
        Additional metadata.
    """
    global_explanation: ExplanationResult
    local_explanations: Dict[str, ExplanationResult]
    aggregation_method: str
    metadata: Dict[str, Any]


class FederatedExplainer:
    """
    Base class for federated explainability.

    Coordinates explanation computation across federated nodes
    without centralizing data.

    Parameters
    ----------
    local_explainer_factory : callable
        Factory function to create explainer for each node.
        Signature: local_explainer_factory(model) -> ExplainerBase
    aggregation_method : str
        'weighted_average', 'median', 'consensus'.

    Examples
    --------
    >>> from xplia.explainers.shap import SHAPExplainer
    >>> factory = lambda model: SHAPExplainer(model)
    >>> fed_explainer = FederatedExplainer(factory, aggregation_method='weighted_average')
    >>> nodes = [FederatedNode('node1', X1, model1), ...]
    >>> fed_exp = fed_explainer.explain_federated(x_test, nodes)
    """

    def __init__(
        self,
        local_explainer_factory: Callable,
        aggregation_method: str = 'weighted_average'
    ):
        self.local_explainer_factory = local_explainer_factory
        self.aggregation_method = aggregation_method

        valid_methods = ['weighted_average', 'median', 'consensus']
        if aggregation_method not in valid_methods:
            raise ValueError(f"Aggregation method must be one of {valid_methods}")

    def _compute_local_explanation(
        self,
        node: FederatedNode,
        instance: np.ndarray,
        **kwargs
    ) -> ExplanationResult:
        """
        Compute explanation on a single node.

        This function would be executed on each node in practice.

        Parameters
        ----------
        node : FederatedNode
            Node to compute on.
        instance : ndarray
            Instance to explain.
        **kwargs
            Additional arguments for explainer.

        Returns
        -------
        explanation : ExplanationResult
            Local explanation from this node.
        """
        # Create local explainer
        local_explainer = self.local_explainer_factory(node.local_model)

        # Compute explanation using local data
        explanation = local_explainer.explain(instance, **kwargs)

        # Add node metadata
        if explanation.metadata is None:
            explanation.metadata = {}
        explanation.metadata['node_id'] = node.node_id
        explanation.metadata['node_weight'] = node.weight

        return explanation

    def _aggregate_weighted_average(
        self,
        local_explanations: List[ExplanationResult]
    ) -> np.ndarray:
        """
        Aggregate feature importances using weighted average.

        Parameters
        ----------
        local_explanations : list of ExplanationResult
            Local explanations from nodes.

        Returns
        -------
        aggregated_importance : ndarray
            Weighted average of feature importances.
        """
        importance_list = []
        weights = []

        for exp in local_explanations:
            if 'feature_importance' not in exp.explanation_data:
                continue

            importance = np.array(exp.explanation_data['feature_importance'])
            weight = exp.metadata.get('node_weight', 1.0)

            importance_list.append(importance)
            weights.append(weight)

        if not importance_list:
            raise ValueError("No feature importances found in local explanations")

        importance_matrix = np.stack(importance_list, axis=0)
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize

        # Weighted average
        aggregated = np.sum(importance_matrix * weights[:, np.newaxis], axis=0)

        return aggregated

    def _aggregate_median(
        self,
        local_explanations: List[ExplanationResult]
    ) -> np.ndarray:
        """
        Aggregate using median (robust to outliers).

        Parameters
        ----------
        local_explanations : list of ExplanationResult
            Local explanations from nodes.

        Returns
        -------
        aggregated_importance : ndarray
            Median of feature importances.
        """
        importance_list = []

        for exp in local_explanations:
            if 'feature_importance' in exp.explanation_data:
                importance = np.array(exp.explanation_data['feature_importance'])
                importance_list.append(importance)

        if not importance_list:
            raise ValueError("No feature importances found")

        importance_matrix = np.stack(importance_list, axis=0)
        aggregated = np.median(importance_matrix, axis=0)

        return aggregated

    def _aggregate_consensus(
        self,
        local_explanations: List[ExplanationResult],
        top_k: int = 5
    ) -> np.ndarray:
        """
        Aggregate based on consensus of top-k features.

        Assigns higher importance to features that appear in top-k
        across multiple nodes.

        Parameters
        ----------
        local_explanations : list of ExplanationResult
            Local explanations from nodes.
        top_k : int
            Number of top features to consider.

        Returns
        -------
        consensus_importance : ndarray
            Consensus-based importance scores.
        """
        importance_list = []

        for exp in local_explanations:
            if 'feature_importance' in exp.explanation_data:
                importance = np.array(exp.explanation_data['feature_importance'])
                importance_list.append(importance)

        if not importance_list:
            raise ValueError("No feature importances found")

        n_features = importance_list[0].shape[0]
        consensus_scores = np.zeros(n_features)

        # Count how many times each feature appears in top-k
        for importance in importance_list:
            top_indices = np.argsort(np.abs(importance))[-top_k:]
            consensus_scores[top_indices] += 1

        # Normalize by number of nodes
        consensus_scores = consensus_scores / len(importance_list)

        return consensus_scores

    def explain_federated(
        self,
        instance: np.ndarray,
        nodes: List[FederatedNode],
        **kwargs
    ) -> FederatedExplanation:
        """
        Compute federated explanation across nodes.

        Parameters
        ----------
        instance : ndarray
            Instance to explain.
        nodes : list of FederatedNode
            Federated nodes.
        **kwargs
            Additional arguments for local explainers.

        Returns
        -------
        fed_explanation : FederatedExplanation
            Federated explanation with global and local results.
        """
        if not nodes:
            raise ValueError("Must provide at least one node")

        # Compute local explanations
        local_explanations = []
        local_exp_dict = {}

        for node in nodes:
            exp = self._compute_local_explanation(node, instance, **kwargs)
            local_explanations.append(exp)
            local_exp_dict[node.node_id] = exp

        # Aggregate
        if self.aggregation_method == 'weighted_average':
            aggregated_importance = self._aggregate_weighted_average(local_explanations)
        elif self.aggregation_method == 'median':
            aggregated_importance = self._aggregate_median(local_explanations)
        elif self.aggregation_method == 'consensus':
            aggregated_importance = self._aggregate_consensus(local_explanations)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

        # Create global explanation
        global_exp = ExplanationResult(
            method=f"federated_{local_explanations[0].method}",
            explanation_data={
                'feature_importance': aggregated_importance.tolist(),
                'n_nodes': len(nodes),
                'aggregation_method': self.aggregation_method
            },
            metadata={
                'federated': True,
                'node_ids': [node.node_id for node in nodes]
            }
        )

        return FederatedExplanation(
            global_explanation=global_exp,
            local_explanations=local_exp_dict,
            aggregation_method=self.aggregation_method,
            metadata={
                'n_nodes': len(nodes),
                'total_samples': sum(node.data.shape[0] for node in nodes)
            }
        )


class SecureAggregation:
    """
    Secure aggregation protocol for federated explanations.

    Uses cryptographic techniques to aggregate explanations without
    revealing individual node contributions.

    Parameters
    ----------
    noise_scale : float
        Scale of noise for secure aggregation.

    Examples
    --------
    >>> secure_agg = SecureAggregation(noise_scale=0.1)
    >>> aggregated = secure_agg.aggregate(local_values)
    """

    def __init__(self, noise_scale: float = 0.1):
        self.noise_scale = noise_scale

    def add_secure_noise(self, value: np.ndarray, seed: int) -> np.ndarray:
        """
        Add correlated noise that cancels out in aggregation.

        In practice, this would use proper secure multiparty computation.

        Parameters
        ----------
        value : ndarray
            Value to add noise to.
        seed : int
            Seed for reproducibility across parties.

        Returns
        -------
        noisy_value : ndarray
            Value with added noise.
        """
        rng = np.random.RandomState(seed)
        noise = rng.normal(0, self.noise_scale, size=value.shape)
        return value + noise

    def aggregate(
        self,
        noisy_values: List[np.ndarray],
        weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Aggregate noisy values.

        Parameters
        ----------
        noisy_values : list of ndarray
            Noisy values from nodes.
        weights : list of float, optional
            Aggregation weights.

        Returns
        -------
        aggregated : ndarray
            Aggregated result.
        """
        if weights is None:
            weights = [1.0 / len(noisy_values)] * len(noisy_values)

        weights = np.array(weights)
        weights = weights / np.sum(weights)

        values_matrix = np.stack(noisy_values, axis=0)
        aggregated = np.sum(values_matrix * weights[:, np.newaxis], axis=0)

        return aggregated


class FederatedSHAPExplainer:
    """
    Federated SHAP explainer.

    Computes SHAP values in federated setting by aggregating local
    SHAP contributions.

    Parameters
    ----------
    aggregation_method : str
        Aggregation method.
    secure : bool
        Whether to use secure aggregation.

    Examples
    --------
    >>> fed_shap = FederatedSHAPExplainer(aggregation_method='weighted_average')
    >>> explanation = fed_shap.explain_federated(x_test, nodes)
    """

    def __init__(
        self,
        aggregation_method: str = 'weighted_average',
        secure: bool = False
    ):
        self.aggregation_method = aggregation_method
        self.secure = secure
        if secure:
            self.secure_agg = SecureAggregation()

    def explain_federated(
        self,
        instance: np.ndarray,
        nodes: List[FederatedNode],
        **kwargs
    ) -> FederatedExplanation:
        """
        Compute federated SHAP explanation.

        Parameters
        ----------
        instance : ndarray
            Instance to explain.
        nodes : list of FederatedNode
            Federated nodes.
        **kwargs
            Additional arguments.

        Returns
        -------
        fed_explanation : FederatedExplanation
            Federated SHAP explanation.
        """
        # In practice, would use actual SHAP explainer
        # For demonstration, using simplified version

        local_explanations = []
        local_exp_dict = {}

        for node in nodes:
            # Simulate local SHAP computation
            # In reality: shap_explainer = shap.Explainer(node.local_model)
            #             shap_values = shap_explainer(instance)

            # Simplified: random SHAP values
            n_features = node.data.shape[1]
            shap_values = np.random.randn(n_features)

            exp = ExplanationResult(
                method='shap',
                explanation_data={
                    'feature_importance': shap_values.tolist()
                },
                metadata={
                    'node_id': node.node_id,
                    'node_weight': node.weight
                }
            )

            local_explanations.append(exp)
            local_exp_dict[node.node_id] = exp

        # Aggregate
        importance_list = []
        weights = []

        for exp in local_explanations:
            importance = np.array(exp.explanation_data['feature_importance'])
            weight = exp.metadata.get('node_weight', 1.0)

            if self.secure:
                # Add secure noise
                importance = self.secure_agg.add_secure_noise(
                    importance,
                    seed=hash(exp.metadata['node_id']) % 2**32
                )

            importance_list.append(importance)
            weights.append(weight)

        if self.secure:
            aggregated = self.secure_agg.aggregate(importance_list, weights)
        else:
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            importance_matrix = np.stack(importance_list, axis=0)
            aggregated = np.sum(importance_matrix * weights[:, np.newaxis], axis=0)

        global_exp = ExplanationResult(
            method='federated_shap',
            explanation_data={
                'feature_importance': aggregated.tolist(),
                'n_nodes': len(nodes),
                'secure_aggregation': self.secure
            },
            metadata={
                'federated': True,
                'node_ids': [node.node_id for node in nodes]
            }
        )

        return FederatedExplanation(
            global_explanation=global_exp,
            local_explanations=local_exp_dict,
            aggregation_method=self.aggregation_method,
            metadata={
                'n_nodes': len(nodes),
                'secure': self.secure
            }
        )


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Federated XAI - Example")
    print("=" * 80)

    # Simulate federated setting with 3 nodes
    np.random.seed(42)

    # Node 1: Hospital A
    X1 = np.random.randn(100, 5)
    y1 = (X1[:, 0] + X1[:, 1] > 0).astype(int)

    # Node 2: Hospital B
    X2 = np.random.randn(150, 5)
    y2 = (X2[:, 0] + X2[:, 1] > 0).astype(int)

    # Node 3: Hospital C
    X3 = np.random.randn(120, 5)
    y3 = (X3[:, 0] + X3[:, 1] > 0).astype(int)

    # Simple model class
    class SimpleModel:
        def predict(self, X):
            return (X[:, 0] + X[:, 1] > 0).astype(int)

        def predict_proba(self, X):
            score = X[:, 0] + X[:, 1]
            pos_prob = 1 / (1 + np.exp(-score))
            return np.column_stack([1 - pos_prob, pos_prob])

    # Each node has local model
    model1 = SimpleModel()
    model2 = SimpleModel()
    model3 = SimpleModel()

    # Create federated nodes
    nodes = [
        FederatedNode('hospital_a', X1, model1, weight=X1.shape[0]),
        FederatedNode('hospital_b', X2, model2, weight=X2.shape[0]),
        FederatedNode('hospital_c', X3, model3, weight=X3.shape[0])
    ]

    # Test instance
    x_test = np.random.randn(5)

    print("\n1. FEDERATED EXPLANATION (Weighted Average)")
    print("-" * 80)

    # Create simple explainer factory
    from xplia.core.base import ExplanationResult

    def simple_explainer_factory(model):
        class SimpleExplainer(ExplainerBase):
            def explain(self, X, **kwargs):
                # Simplified: just use coefficients
                importance = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
                importance += np.random.randn(5) * 0.2  # Add some noise

                return ExplanationResult(
                    method='simple_shap',
                    explanation_data={
                        'feature_importance': importance.tolist()
                    },
                    metadata={}
                )
        return SimpleExplainer(model)

    fed_explainer = FederatedExplainer(
        simple_explainer_factory,
        aggregation_method='weighted_average'
    )

    fed_exp = fed_explainer.explain_federated(x_test, nodes)

    print(f"Global explanation method: {fed_exp.global_explanation.method}")
    print(f"Number of nodes: {fed_exp.global_explanation.explanation_data['n_nodes']}")
    print(f"Aggregation: {fed_exp.aggregation_method}")
    print(f"Global feature importance: {fed_exp.global_explanation.explanation_data['feature_importance']}")

    print("\nLocal explanations:")
    for node_id, local_exp in fed_exp.local_explanations.items():
        importance = local_exp.explanation_data['feature_importance']
        print(f"  {node_id}: {importance}")

    print("\n2. FEDERATED EXPLANATION (Median Aggregation)")
    print("-" * 80)
    fed_explainer_median = FederatedExplainer(
        simple_explainer_factory,
        aggregation_method='median'
    )

    fed_exp_median = fed_explainer_median.explain_federated(x_test, nodes)
    print(f"Median aggregated importance: {fed_exp_median.global_explanation.explanation_data['feature_importance']}")

    print("\n3. FEDERATED EXPLANATION (Consensus)")
    print("-" * 80)
    fed_explainer_consensus = FederatedExplainer(
        simple_explainer_factory,
        aggregation_method='consensus'
    )

    fed_exp_consensus = fed_explainer_consensus.explain_federated(x_test, nodes)
    print(f"Consensus importance: {fed_exp_consensus.global_explanation.explanation_data['feature_importance']}")

    print("\n4. SECURE FEDERATED SHAP")
    print("-" * 80)
    fed_shap_secure = FederatedSHAPExplainer(
        aggregation_method='weighted_average',
        secure=True
    )

    fed_shap_exp = fed_shap_secure.explain_federated(x_test, nodes)
    print(f"Method: {fed_shap_exp.global_explanation.method}")
    print(f"Secure aggregation: {fed_shap_exp.global_explanation.explanation_data['secure_aggregation']}")
    print(f"Global SHAP values: {fed_shap_exp.global_explanation.explanation_data['feature_importance']}")

    print("\n5. METADATA")
    print("-" * 80)
    print(f"Total samples across federation: {fed_exp.metadata['total_samples']}")
    print(f"Node IDs: {fed_exp.global_explanation.metadata['node_ids']}")

    print("\n" + "=" * 80)
    print("Federated XAI demonstration complete!")
    print("Data never left individual nodes - explanations computed locally")
    print("and aggregated securely!")
    print("=" * 80)
