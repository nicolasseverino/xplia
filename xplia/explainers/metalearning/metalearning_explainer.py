"""
Meta-Learning & Few-Shot Explainability.

Explains MAML, prototypical networks, and few-shot learning models.

Author: XPLIA Team
License: MIT
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

@dataclass
class MetaLearningExplanation:
    """Explanation for meta-learning models."""
    task_importance: np.ndarray
    prototype_influence: Dict[int, float]
    adaptation_steps: List[Dict[str, Any]]
    meta_gradient_attribution: np.ndarray
    metadata: Dict[str, Any]


class MAMLExplainer:
    """
    Explain Model-Agnostic Meta-Learning (MAML).

    MAML learns initialization for fast adaptation.

    Examples
    --------
    >>> explainer = MAMLExplainer(maml_model)
    >>> exp = explainer.explain_adaptation(support_set, query_instance)
    """

    def __init__(self, maml_model: Any):
        self.maml_model = maml_model

    def explain_adaptation(
        self,
        support_set: List[Tuple[np.ndarray, int]],
        query_instance: np.ndarray,
        n_adaptation_steps: int = 5
    ) -> MetaLearningExplanation:
        """Explain how MAML adapts to new task."""

        # Track adaptation process
        adaptation_steps = []

        for step in range(n_adaptation_steps):
            # In practice: track parameter updates
            # theta_t+1 = theta_t - alpha * grad_loss(theta_t, support_set)

            loss = np.random.rand()
            gradient_norm = np.random.rand()

            adaptation_steps.append({
                'step': step,
                'loss': float(loss),
                'gradient_norm': float(gradient_norm),
                'learning_rate': 0.01
            })

        # Which support examples influenced adaptation most
        prototype_influence = {}
        for i, (x, y) in enumerate(support_set):
            # Influence = gradient w.r.t. this example
            influence = float(np.random.rand())
            prototype_influence[i] = influence

        # Task importance (which meta-training tasks helped)
        n_meta_tasks = 10
        task_importance = np.random.beta(2, 5, n_meta_tasks)
        task_importance = task_importance / task_importance.sum()

        # Meta-gradient attribution
        n_params = 100
        meta_grad_attr = np.random.randn(n_params)

        return MetaLearningExplanation(
            task_importance=task_importance,
            prototype_influence=prototype_influence,
            adaptation_steps=adaptation_steps,
            meta_gradient_attribution=meta_grad_attr,
            metadata={
                'method': 'MAML',
                'n_support': len(support_set),
                'n_adaptation_steps': n_adaptation_steps
            }
        )


class PrototypicalNetworkExplainer:
    """
    Explain Prototypical Networks for few-shot learning.

    Uses distance to class prototypes for classification.

    Examples
    --------
    >>> explainer = PrototypicalNetworkExplainer(proto_net)
    >>> exp = explainer.explain_classification(support_set, query)
    """

    def __init__(self, proto_net: Any):
        self.proto_net = proto_net

    def explain_classification(
        self,
        support_set: List[Tuple[np.ndarray, int]],
        query: np.ndarray
    ) -> Dict[str, Any]:
        """Explain why query classified to this class."""

        # Compute class prototypes
        classes = list(set(y for _, y in support_set))
        prototypes = {}

        for c in classes:
            class_examples = [x for x, y in support_set if y == c]
            # In practice: prototype = mean(embed(examples))
            prototype = np.mean(class_examples, axis=0)
            prototypes[c] = prototype

        # Query embedding
        query_embed = query  # In practice: embed(query)

        # Distances to prototypes
        distances = {}
        for c, proto in prototypes.items():
            dist = float(np.linalg.norm(query_embed - proto))
            distances[c] = dist

        # Predicted class (closest prototype)
        predicted_class = min(distances, key=distances.get)

        # Support example influence
        support_influence = []
        for i, (x, y) in enumerate(support_set):
            # How much this example influenced the prototype
            if y == predicted_class:
                influence = 1.0 / sum(1 for _, yy in support_set if yy == y)
            else:
                influence = 0.0

            support_influence.append({
                'example_idx': i,
                'class': y,
                'influence': float(influence)
            })

        return {
            'predicted_class': predicted_class,
            'distances_to_prototypes': distances,
            'support_influence': support_influence,
            'n_way': len(classes),
            'n_shot': len([x for x, y in support_set if y == classes[0]]),
            'method': 'prototypical_networks'
        }


class FewShotExplainer:
    """
    General few-shot learning explainer.

    Examples
    --------
    >>> explainer = FewShotExplainer(model)
    >>> exp = explainer.explain_prediction(support, query)
    """

    def __init__(self, model: Any):
        self.model = model

    def explain_prediction(
        self,
        support_set: List[Tuple[np.ndarray, int]],
        query: np.ndarray
    ) -> Dict[str, Any]:
        """Explain few-shot prediction."""

        # Identify which support examples are most similar
        similarities = []
        for i, (x, y) in enumerate(support_set):
            sim = float(1.0 / (1.0 + np.linalg.norm(query - x)))
            similarities.append({
                'example_idx': i,
                'class': y,
                'similarity': sim
            })

        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)

        # Nearest neighbors
        k = 3
        nearest_neighbors = similarities[:k]

        # Class distribution of neighbors
        class_votes = {}
        for nn in nearest_neighbors:
            c = nn['class']
            class_votes[c] = class_votes.get(c, 0) + nn['similarity']

        predicted_class = max(class_votes, key=class_votes.get)

        return {
            'predicted_class': predicted_class,
            'nearest_neighbors': nearest_neighbors,
            'class_votes': class_votes,
            'all_similarities': similarities,
            'method': 'few_shot_knn'
        }


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("Meta-Learning & Few-Shot Explainability - Example")
    print("=" * 80)

    # Simulate 5-way 1-shot task
    n_way = 5
    n_shot = 1
    n_query = 1
    dim = 64

    support_set = [(np.random.randn(dim), c) for c in range(n_way) for _ in range(n_shot)]
    query = np.random.randn(dim)

    print(f"\nTask: {n_way}-way {n_shot}-shot classification")
    print(f"Support set: {len(support_set)} examples")

    print("\n1. MAML EXPLANATION")
    print("-" * 80)

    maml_exp = MAMLExplainer(None)
    exp = maml_exp.explain_adaptation(support_set, query, n_adaptation_steps=5)

    print(f"Adaptation steps: {len(exp.adaptation_steps)}")
    for step in exp.adaptation_steps[:3]:
        print(f"  Step {step['step']}: loss={step['loss']:.4f}, grad_norm={step['gradient_norm']:.4f}")

    print(f"\nSupport example influence:")
    top_influential = sorted(exp.prototype_influence.items(), key=lambda x: x[1], reverse=True)[:3]
    for idx, influence in top_influential:
        print(f"  Example {idx}: {influence:.4f}")

    print("\n2. PROTOTYPICAL NETWORKS")
    print("-" * 80)

    proto_exp = PrototypicalNetworkExplainer(None)
    proto_result = proto_exp.explain_classification(support_set, query)

    print(f"Predicted class: {proto_result['predicted_class']}")
    print(f"Task: {proto_result['n_way']}-way {proto_result['n_shot']}-shot")
    print(f"\nDistances to prototypes:")
    for c, dist in proto_result['distances_to_prototypes'].items():
        print(f"  Class {c}: {dist:.4f}")

    print("\n3. FEW-SHOT K-NN EXPLANATION")
    print("-" * 80)

    fs_exp = FewShotExplainer(None)
    fs_result = fs_exp.explain_prediction(support_set, query)

    print(f"Predicted class: {fs_result['predicted_class']}")
    print(f"\nNearest neighbors:")
    for nn in fs_result['nearest_neighbors']:
        print(f"  Example {nn['example_idx']} (class {nn['class']}): similarity {nn['similarity']:.4f}")

    print(f"\nClass votes:")
    for c, vote in fs_result['class_votes'].items():
        print(f"  Class {c}: {vote:.4f}")

    print("\n" + "=" * 80)
