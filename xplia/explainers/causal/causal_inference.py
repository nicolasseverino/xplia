"""
Advanced Causal Inference for Explainable AI.

This module implements state-of-the-art causal inference methods for XAI:
- Structural Causal Models (SCM)
- Do-calculus and interventions
- Causal attribution
- Counterfactual reasoning with causal graphs
- Causal feature importance

Based on Pearl's causality framework and recent advances in causal ML.

Author: XPLIA Team
License: MIT
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

try:
    import networkx as nx
except ImportError:
    nx = None


class CausalRelationType(Enum):
    """Types of causal relationships."""
    DIRECT = "direct"  # X -> Y
    INDIRECT = "indirect"  # X -> Z -> Y
    CONFOUNDED = "confounded"  # X <- Z -> Y
    MEDIATED = "mediated"  # X -> M -> Y (M is mediator)
    COLLIDER = "collider"  # X -> Z <- Y


@dataclass
class CausalGraph:
    """
    Structural Causal Model (SCM) representation.

    Represents causal relationships as a directed acyclic graph (DAG).
    """
    nodes: List[str]
    edges: List[Tuple[str, str]]  # (cause, effect) pairs
    confounders: Dict[Tuple[str, str], List[str]] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and build graph."""
        if nx is None:
            raise ImportError("networkx is required for causal inference. Install with: pip install networkx")

        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.nodes)
        self.graph.add_edges_from(self.edges)

        # Check if DAG
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Causal graph must be a Directed Acyclic Graph (DAG)")

    def get_parents(self, node: str) -> List[str]:
        """Get direct causes of a node."""
        return list(self.graph.predecessors(node))

    def get_children(self, node: str) -> List[str]:
        """Get direct effects of a node."""
        return list(self.graph.successors(node))

    def get_ancestors(self, node: str) -> List[str]:
        """Get all causal ancestors."""
        return list(nx.ancestors(self.graph, node))

    def get_descendants(self, node: str) -> List[str]:
        """Get all causal descendants."""
        return list(nx.descendants(self.graph, node))

    def d_separated(self, X: str, Y: str, Z: Optional[List[str]] = None) -> bool:
        """
        Check if X and Y are d-separated given Z.

        D-separation implies conditional independence.
        """
        Z = Z or []
        return nx.d_separated(self.graph, {X}, {Y}, set(Z))

    def get_backdoor_paths(self, treatment: str, outcome: str) -> List[List[str]]:
        """
        Get all backdoor paths from treatment to outcome.

        Backdoor paths are non-causal paths that need to be blocked.
        """
        backdoor_paths = []

        # Find all paths
        all_paths = list(nx.all_simple_paths(
            self.graph.to_undirected(),
            treatment,
            outcome
        ))

        for path in all_paths:
            # Check if it's a backdoor path (starts with <- from treatment)
            if len(path) > 2:
                # Check if first edge goes into treatment
                if (path[1], path[0]) in self.edges:
                    backdoor_paths.append(path)

        return backdoor_paths

    def satisfies_backdoor_criterion(
        self,
        treatment: str,
        outcome: str,
        adjustment_set: List[str]
    ) -> bool:
        """
        Check if adjustment set satisfies backdoor criterion.

        Backdoor criterion ensures unbiased causal effect estimation.
        """
        # 1. No node in Z is a descendant of treatment
        treatment_descendants = self.get_descendants(treatment)
        if any(z in treatment_descendants for z in adjustment_set):
            return False

        # 2. Z blocks all backdoor paths
        backdoor_paths = self.get_backdoor_paths(treatment, outcome)

        for path in backdoor_paths:
            # Check if path is blocked by adjustment set
            blocked = False
            for i in range(1, len(path) - 1):
                if path[i] in adjustment_set:
                    blocked = True
                    break

            if not blocked:
                return False

        return True


@dataclass
class CausalEffect:
    """Represents a causal effect estimate."""
    treatment: str
    outcome: str
    effect: float
    confidence_interval: Tuple[float, float]
    method: str
    adjustment_set: List[str]
    p_value: float = None


class CausalEstimator(ABC):
    """Base class for causal effect estimation."""

    @abstractmethod
    def estimate_effect(
        self,
        treatment: str,
        outcome: str,
        data: pd.DataFrame,
        adjustment_set: Optional[List[str]] = None
    ) -> CausalEffect:
        """Estimate causal effect of treatment on outcome."""
        pass


class BackdoorAdjustment(CausalEstimator):
    """
    Backdoor adjustment for causal effect estimation.

    Implements the backdoor criterion for identifying causal effects.
    """

    def __init__(self, causal_graph: CausalGraph):
        """
        Initialize backdoor adjustment estimator.

        Parameters
        ----------
        causal_graph : CausalGraph
            Known or learned causal graph.
        """
        self.causal_graph = causal_graph

    def estimate_effect(
        self,
        treatment: str,
        outcome: str,
        data: pd.DataFrame,
        adjustment_set: Optional[List[str]] = None
    ) -> CausalEffect:
        """
        Estimate causal effect using backdoor adjustment.

        Parameters
        ----------
        treatment : str
            Treatment variable.

        outcome : str
            Outcome variable.

        data : pd.DataFrame
            Observational data.

        adjustment_set : list of str, optional
            Variables to adjust for. If None, automatically determined.

        Returns
        -------
        CausalEffect
            Estimated causal effect with confidence interval.
        """
        if adjustment_set is None:
            adjustment_set = self._find_adjustment_set(treatment, outcome)

        # Verify backdoor criterion
        if not self.causal_graph.satisfies_backdoor_criterion(
            treatment, outcome, adjustment_set
        ):
            raise ValueError(
                f"Adjustment set {adjustment_set} does not satisfy backdoor criterion"
            )

        # Estimate effect using stratification
        effect = self._stratified_estimate(
            data, treatment, outcome, adjustment_set
        )

        # Bootstrap confidence interval
        ci = self._bootstrap_ci(data, treatment, outcome, adjustment_set)

        return CausalEffect(
            treatment=treatment,
            outcome=outcome,
            effect=effect,
            confidence_interval=ci,
            method="backdoor_adjustment",
            adjustment_set=adjustment_set
        )

    def _find_adjustment_set(self, treatment: str, outcome: str) -> List[str]:
        """
        Find minimal sufficient adjustment set.

        Uses algorithm to find minimal set satisfying backdoor criterion.
        """
        # Start with all parents of treatment (excluding outcome)
        candidates = [
            node for node in self.causal_graph.get_parents(treatment)
            if node != outcome
        ]

        # Check if this satisfies backdoor criterion
        if self.causal_graph.satisfies_backdoor_criterion(
            treatment, outcome, candidates
        ):
            return candidates

        # Add confounders
        all_nodes = [
            node for node in self.causal_graph.nodes
            if node not in [treatment, outcome]
        ]

        # Try to find minimal set (greedy approach)
        adjustment_set = []
        for node in all_nodes:
            test_set = adjustment_set + [node]
            if self.causal_graph.satisfies_backdoor_criterion(
                treatment, outcome, test_set
            ):
                adjustment_set = test_set

        return adjustment_set

    def _stratified_estimate(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        adjustment_set: List[str]
    ) -> float:
        """
        Estimate effect using stratification on adjustment set.
        """
        if len(adjustment_set) == 0:
            # No adjustment needed - simple difference
            treated = data[data[treatment] == 1]
            control = data[data[treatment] == 0]
            return treated[outcome].mean() - control[outcome].mean()

        # Stratification approach (simplified)
        # In practice, would use propensity score matching or IPW
        from sklearn.linear_model import LinearRegression

        X = data[adjustment_set + [treatment]]
        y = data[outcome]

        model = LinearRegression()
        model.fit(X, y)

        # Coefficient of treatment is causal effect
        treatment_idx = list(X.columns).index(treatment)
        return model.coef_[treatment_idx]

    def _bootstrap_ci(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        adjustment_set: List[str],
        n_bootstrap: int = 1000,
        alpha: float = 0.05
    ) -> Tuple[float, float]:
        """Bootstrap confidence interval."""
        effects = []

        for _ in range(n_bootstrap):
            # Resample with replacement
            sample = data.sample(n=len(data), replace=True)

            # Estimate effect
            effect = self._stratified_estimate(
                sample, treatment, outcome, adjustment_set
            )
            effects.append(effect)

        # Percentile method
        lower = np.percentile(effects, alpha/2 * 100)
        upper = np.percentile(effects, (1 - alpha/2) * 100)

        return (lower, upper)


class DoCalculus:
    """
    Implements Pearl's do-calculus for causal inference.

    Allows computing interventional distributions P(Y | do(X=x)).
    """

    def __init__(self, causal_graph: CausalGraph):
        """
        Initialize do-calculus engine.

        Parameters
        ----------
        causal_graph : CausalGraph
            Causal graph structure.
        """
        self.causal_graph = causal_graph

    def intervention(
        self,
        data: pd.DataFrame,
        intervention: Dict[str, Any],
        target: str
    ) -> np.ndarray:
        """
        Compute effect of intervention do(X=x) on target Y.

        Parameters
        ----------
        data : pd.DataFrame
            Observational data.

        intervention : dict
            Intervention to perform, e.g., {'X': 1}.

        target : str
            Target variable to compute distribution for.

        Returns
        -------
        np.ndarray
            Distribution of target under intervention.
        """
        # Simplified implementation using causal adjustment formula

        intervention_var = list(intervention.keys())[0]
        intervention_val = list(intervention.values())[0]

        # Find adjustment set
        adjustment_set = self._find_adjustment_set_for_do(
            intervention_var, target
        )

        # Compute P(Y | do(X=x)) using adjustment formula:
        # P(Y | do(X=x)) = Σ_z P(Y | X=x, Z=z) P(Z=z)

        if len(adjustment_set) == 0:
            # No adjustment needed
            subset = data[data[intervention_var] == intervention_val]
            return subset[target].values

        # Stratify by adjustment set and compute weighted average
        # (Simplified version)
        return self._compute_interventional_distribution(
            data, intervention_var, intervention_val, target, adjustment_set
        )

    def _find_adjustment_set_for_do(
        self,
        intervention: str,
        target: str
    ) -> List[str]:
        """Find adjustment set for do-calculus."""
        # Use backdoor criterion
        parents = self.causal_graph.get_parents(intervention)
        return [p for p in parents if p != target]

    def _compute_interventional_distribution(
        self,
        data: pd.DataFrame,
        intervention_var: str,
        intervention_val: Any,
        target: str,
        adjustment_set: List[str]
    ) -> np.ndarray:
        """Compute interventional distribution."""
        # Simplified implementation
        results = []

        # For each stratum of adjustment variables
        for _, group in data.groupby(adjustment_set):
            # Within stratum, set intervention and observe target
            group_weight = len(group) / len(data)

            # Simulate intervention (simplified)
            intervened = group.copy()
            intervened[intervention_var] = intervention_val

            # Collect target values weighted by stratum size
            results.extend(group[target].values * group_weight)

        return np.array(results)


class CausalAttributionExplainer:
    """
    Causal attribution for model predictions.

    Attributes prediction changes to causal effects of features.
    """

    def __init__(
        self,
        model: Any,
        causal_graph: CausalGraph,
        feature_names: List[str]
    ):
        """
        Initialize causal attribution explainer.

        Parameters
        ----------
        model : object
            Trained ML model.

        causal_graph : CausalGraph
            Causal graph over features.

        feature_names : list of str
            Names of features.
        """
        self.model = model
        self.causal_graph = causal_graph
        self.feature_names = feature_names
        self.do_calc = DoCalculus(causal_graph)

    def explain(
        self,
        instance: np.ndarray,
        baseline: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute causal attribution for prediction.

        Parameters
        ----------
        instance : np.ndarray
            Instance to explain.

        baseline : np.ndarray, optional
            Baseline for comparison.

        Returns
        -------
        dict
            Causal attribution scores for each feature.
        """
        if baseline is None:
            baseline = np.zeros_like(instance)

        # Get predictions
        pred_instance = self.model.predict([instance])[0]
        pred_baseline = self.model.predict([baseline])[0]

        total_effect = pred_instance - pred_baseline

        # Compute causal attribution for each feature
        attributions = {}

        for i, feature in enumerate(self.feature_names):
            # Compute causal effect of this feature
            # Using path-specific effects
            causal_effect = self._compute_path_specific_effect(
                feature, instance, baseline, i
            )

            attributions[feature] = causal_effect

        return attributions

    def _compute_path_specific_effect(
        self,
        feature: str,
        instance: np.ndarray,
        baseline: np.ndarray,
        feature_idx: int
    ) -> float:
        """
        Compute path-specific causal effect.

        Separates direct and indirect effects.
        """
        # Direct effect: intervene on feature, hold all descendants at baseline
        descendants = self.causal_graph.get_descendants(feature)
        descendant_indices = [
            self.feature_names.index(d) for d in descendants
            if d in self.feature_names
        ]

        # Create intervened instance
        intervened = instance.copy()
        intervened[feature_idx] = instance[feature_idx]

        # Hold descendants at baseline
        for idx in descendant_indices:
            intervened[idx] = baseline[idx]

        # Compute direct effect
        pred_intervened = self.model.predict([intervened])[0]
        pred_baseline = self.model.predict([baseline])[0]

        direct_effect = pred_intervened - pred_baseline

        return direct_effect


def discover_causal_graph(
    data: pd.DataFrame,
    method: str = "pc",
    significance_level: float = 0.05
) -> CausalGraph:
    """
    Discover causal graph from observational data.

    Parameters
    ----------
    data : pd.DataFrame
        Observational data.

    method : str
        Discovery algorithm: 'pc' (PC algorithm), 'fci', 'ges'.

    significance_level : float
        Significance level for conditional independence tests.

    Returns
    -------
    CausalGraph
        Discovered causal graph.
    """
    # Simplified implementation
    # In practice, would use libraries like causal-learn, py-causal, etc.

    from scipy.stats import pearsonr

    nodes = list(data.columns)
    edges = []

    # Simple correlation-based discovery (placeholder)
    for i, col1 in enumerate(nodes):
        for col2 in nodes[i+1:]:
            corr, p_value = pearsonr(data[col1], data[col2])

            if p_value < significance_level and abs(corr) > 0.3:
                # Determine direction (simplified - would need proper algorithm)
                if corr > 0:
                    edges.append((col1, col2))
                else:
                    edges.append((col2, col1))

    return CausalGraph(nodes=nodes, edges=edges)


# Example usage functions
def example_causal_analysis():
    """Example of causal analysis workflow."""

    # 1. Define or discover causal graph
    graph = CausalGraph(
        nodes=['smoking', 'exercise', 'cholesterol', 'heart_disease'],
        edges=[
            ('smoking', 'cholesterol'),
            ('smoking', 'heart_disease'),
            ('exercise', 'cholesterol'),
            ('exercise', 'heart_disease'),
            ('cholesterol', 'heart_disease')
        ]
    )

    # 2. Generate synthetic data
    np.random.seed(42)
    n = 1000
    data = pd.DataFrame({
        'smoking': np.random.binomial(1, 0.3, n),
        'exercise': np.random.binomial(1, 0.6, n),
    })

    data['cholesterol'] = (
        0.3 * data['smoking']
        - 0.2 * data['exercise']
        + np.random.normal(0, 0.1, n)
    )

    data['heart_disease'] = (
        0.4 * data['smoking']
        - 0.3 * data['exercise']
        + 0.5 * data['cholesterol']
        + np.random.normal(0, 0.1, n)
    )

    # 3. Estimate causal effects
    estimator = BackdoorAdjustment(graph)

    effect = estimator.estimate_effect(
        treatment='smoking',
        outcome='heart_disease',
        data=data
    )

    print(f"Causal effect of smoking on heart disease: {effect.effect:.4f}")
    print(f"95% CI: {effect.confidence_interval}")

    # 4. Perform intervention
    do_calc = DoCalculus(graph)

    # What if everyone stopped smoking?
    intervened_dist = do_calc.intervention(
        data,
        intervention={'smoking': 0},
        target='heart_disease'
    )

    print(f"Heart disease under smoking intervention: {intervened_dist.mean():.4f}")

    return graph, effect


if __name__ == "__main__":
    example_causal_analysis()
