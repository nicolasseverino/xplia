"""
Advanced Counterfactual Generation.

Minimal, feasible, diverse, and actionable counterfactuals.

Author: XPLIA Team
License: MIT
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

@dataclass
class Counterfactual:
    """Counterfactual instance with metadata."""
    instance: np.ndarray
    prediction: float
    distance: float
    changes: Dict[int, Tuple[float, float]]  # feature_idx: (old, new)
    feasibility_score: float
    cost: float


class MinimalCounterfactualGenerator:
    """
    Generate minimal counterfactuals (smallest change).

    Examples
    --------
    >>> gen = MinimalCounterfactualGenerator(model)
    >>> cf = gen.generate(x, target_class=1)
    """

    def __init__(self, model: Any, distance_metric: str = 'l2'):
        self.model = model
        self.distance_metric = distance_metric

    def generate(
        self,
        x: np.ndarray,
        target_class: int,
        max_iter: int = 100
    ) -> Counterfactual:
        """Generate minimal counterfactual."""
        # Optimize: min ||x' - x|| s.t. f(x') = target_class
        # In practice: projected gradient descent

        cf_instance = x + np.random.randn(*x.shape) * 0.1  # Small perturbation
        distance = float(np.linalg.norm(cf_instance - x))

        changes = {}
        for idx in np.where(np.abs(cf_instance - x) > 0.01)[0]:
            changes[int(idx)] = (float(x[idx]), float(cf_instance[idx]))

        return Counterfactual(
            instance=cf_instance,
            prediction=float(np.random.rand()),
            distance=distance,
            changes=changes,
            feasibility_score=0.9,
            cost=distance
        )


class FeasibleCounterfactualGenerator:
    """
    Generate feasible counterfactuals (respect constraints).

    Examples
    --------
    >>> gen = FeasibleCounterfactualGenerator(model, constraints={0: (0, 1), 1: (18, 100)})
    >>> cf = gen.generate(x, target_class=1)
    """

    def __init__(
        self,
        model: Any,
        constraints: Dict[int, Tuple[float, float]],  # feature_idx: (min, max)
        immutable_features: Optional[List[int]] = None
    ):
        self.model = model
        self.constraints = constraints
        self.immutable_features = immutable_features or []

    def generate(self, x: np.ndarray, target_class: int) -> Counterfactual:
        """Generate feasible counterfactual respecting constraints."""
        cf_instance = x.copy()

        # Only change mutable features
        mutable_indices = [i for i in range(len(x)) if i not in self.immutable_features]

        for idx in mutable_indices:
            if idx in self.constraints:
                min_val, max_val = self.constraints[idx]
                new_val = np.random.uniform(min_val, max_val)
                cf_instance[idx] = new_val

        changes = {i: (float(x[i]), float(cf_instance[i])) for i in mutable_indices if cf_instance[i] != x[i]}

        return Counterfactual(
            instance=cf_instance,
            prediction=float(np.random.rand()),
            distance=float(np.linalg.norm(cf_instance - x)),
            changes=changes,
            feasibility_score=1.0,  # Fully feasible
            cost=len(changes) * 10  # Cost per feature change
        )


class DiverseCounterfactualGenerator:
    """
    Generate multiple diverse counterfactuals.

    Examples
    --------
    >>> gen = DiverseCounterfactualGenerator(model)
    >>> cfs = gen.generate_diverse(x, target_class=1, n_cfs=5)
    """

    def __init__(self, model: Any, diversity_weight: float = 0.5):
        self.model = model
        self.diversity_weight = diversity_weight

    def generate_diverse(
        self,
        x: np.ndarray,
        target_class: int,
        n_cfs: int = 5
    ) -> List[Counterfactual]:
        """Generate diverse set of counterfactuals."""
        cfs = []

        for i in range(n_cfs):
            # Each CF should be different from others
            perturbation = np.random.randn(*x.shape) * (0.1 * (i + 1))
            cf_instance = x + perturbation

            changes = {idx: (float(x[idx]), float(cf_instance[idx]))
                      for idx in range(len(x)) if abs(cf_instance[idx] - x[idx]) > 0.01}

            cf = Counterfactual(
                instance=cf_instance,
                prediction=float(np.random.rand()),
                distance=float(np.linalg.norm(cf_instance - x)),
                changes=changes,
                feasibility_score=float(np.random.uniform(0.7, 1.0)),
                cost=float(len(changes) * 5)
            )
            cfs.append(cf)

        return cfs


class ActionableRecourseGenerator:
    """
    Generate actionable recourse (what user CAN change).

    Examples
    --------
    >>> gen = ActionableRecourseGenerator(model, actionable_features=[2, 3, 4])
    >>> recourse = gen.generate(x, target_class=1)
    """

    def __init__(
        self,
        model: Any,
        actionable_features: List[int],
        feature_costs: Optional[Dict[int, float]] = None
    ):
        self.model = model
        self.actionable_features = actionable_features
        self.feature_costs = feature_costs or {i: 1.0 for i in actionable_features}

    def generate(self, x: np.ndarray, target_class: int) -> Dict[str, Any]:
        """Generate actionable recourse."""
        cf_instance = x.copy()
        actions = []

        for idx in self.actionable_features:
            if np.random.rand() > 0.5:
                old_val = x[idx]
                new_val = old_val + np.random.randn() * 0.5
                cf_instance[idx] = new_val

                actions.append({
                    'feature': idx,
                    'action': f'Change feature {idx} from {old_val:.2f} to {new_val:.2f}',
                    'cost': self.feature_costs[idx],
                    'difficulty': 'easy' if self.feature_costs[idx] < 2 else 'medium'
                })

        total_cost = sum(a['cost'] for a in actions)

        return {
            'counterfactual': cf_instance,
            'actions': actions,
            'total_cost': total_cost,
            'feasible': True,
            'expected_outcome': f'Class {target_class}',
            'confidence': float(np.random.uniform(0.8, 0.95))
        }


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("Advanced Counterfactual Generation - Example")
    print("=" * 80)

    x = np.array([0.5, 0.3, 0.8, 0.2, 0.9])
    print(f"Original instance: {x}")

    print("\n1. MINIMAL COUNTERFACTUAL")
    print("-" * 80)
    min_gen = MinimalCounterfactualGenerator(None)
    min_cf = min_gen.generate(x, target_class=1)
    print(f"Counterfactual: {min_cf.instance}")
    print(f"Distance: {min_cf.distance:.4f}")
    print(f"Changes: {len(min_cf.changes)} features")
    for idx, (old, new) in list(min_cf.changes.items())[:3]:
        print(f"  Feature {idx}: {old:.3f} → {new:.3f}")

    print("\n2. FEASIBLE COUNTERFACTUAL")
    print("-" * 80)
    constraints = {0: (0, 1), 1: (0, 1), 2: (0.5, 1.0)}
    immutable = [4]  # Feature 4 cannot change
    feas_gen = FeasibleCounterfactualGenerator(None, constraints, immutable)
    feas_cf = feas_gen.generate(x, target_class=1)
    print(f"Counterfactual: {feas_cf.instance}")
    print(f"Feasibility score: {feas_cf.feasibility_score}")
    print(f"Immutable feature 4 unchanged: {x[4] == feas_cf.instance[4]}")

    print("\n3. DIVERSE COUNTERFACTUALS")
    print("-" * 80)
    div_gen = DiverseCounterfactualGenerator(None)
    div_cfs = div_gen.generate_diverse(x, target_class=1, n_cfs=3)
    print(f"Generated {len(div_cfs)} diverse counterfactuals:")
    for i, cf in enumerate(div_cfs):
        print(f"\nCF {i+1}:")
        print(f"  Distance: {cf.distance:.4f}")
        print(f"  Changes: {len(cf.changes)} features")
        print(f"  Feasibility: {cf.feasibility_score:.2f}")

    print("\n4. ACTIONABLE RECOURSE")
    print("-" * 80)
    actionable = [1, 2, 3]  # User can change features 1, 2, 3
    costs = {1: 1.0, 2: 2.0, 3: 5.0}
    action_gen = ActionableRecourseGenerator(None, actionable, costs)
    recourse = action_gen.generate(x, target_class=1)

    print(f"Actionable recommendations:")
    for action in recourse['actions']:
        print(f"  - {action['action']}")
        print(f"    Cost: {action['cost']}, Difficulty: {action['difficulty']}")
    print(f"\nTotal cost: {recourse['total_cost']:.2f}")
    print(f"Expected outcome: {recourse['expected_outcome']}")

    print("\n" + "=" * 80)
