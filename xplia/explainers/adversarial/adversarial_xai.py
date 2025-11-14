"""
Adversarial Attacks and Defenses for Explainable AI.

This module implements:
- Adversarial attacks on explanations
- Explanation manipulation detection
- Robust explanation methods
- Defense mechanisms

Based on recent research in adversarial ML and explainability security.

Author: XPLIA Team
License: MIT
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod


class AttackType(Enum):
    """Types of adversarial attacks on explanations."""
    FEATURE_MANIPULATION = "feature_manipulation"  # Change feature importance rankings
    SALIENCY_ATTACK = "saliency_attack"  # Target saliency maps
    BACKDOOR_EXPLANATION = "backdoor_explanation"  # Plant backdoors in explanations
    FAIRWASHING_ATTACK = "fairwashing_attack"  # Hide discriminatory features
    RANKING_ATTACK = "ranking_attack"  # Manipulate feature rankings


@dataclass
class AdversarialExample:
    """Adversarial example for explainability."""
    original_instance: np.ndarray
    adversarial_instance: np.ndarray
    perturbation: np.ndarray
    original_explanation: Dict[str, Any]
    adversarial_explanation: Dict[str, Any]
    attack_type: AttackType
    success: bool
    l_inf_norm: float
    l2_norm: float


class ExplanationAttack(ABC):
    """Base class for adversarial attacks on explanations."""

    @abstractmethod
    def attack(
        self,
        model: Any,
        explainer: Any,
        instance: np.ndarray,
        target_explanation: Optional[Dict] = None,
        epsilon: float = 0.1,
        max_iter: int = 100
    ) -> AdversarialExample:
        """
        Generate adversarial example to manipulate explanation.

        Parameters
        ----------
        model : object
            Target model.

        explainer : object
            Explanation method.

        instance : np.ndarray
            Original instance.

        target_explanation : dict, optional
            Target explanation to achieve.

        epsilon : float
            Maximum perturbation bound.

        max_iter : int
            Maximum optimization iterations.

        Returns
        -------
        AdversarialExample
            Adversarial example with manipulated explanation.
        """
        pass


class FeatureRankingAttack(ExplanationAttack):
    """
    Attack to manipulate feature importance rankings.

    Goal: Make an unimportant feature appear most important.
    """

    def attack(
        self,
        model: Any,
        explainer: Any,
        instance: np.ndarray,
        target_explanation: Optional[Dict] = None,
        epsilon: float = 0.1,
        max_iter: int = 100
    ) -> AdversarialExample:
        """
        Manipulate feature rankings via gradient-based optimization.

        Parameters
        ----------
        target_explanation : dict
            Should contain 'target_feature': index of feature to make top.
        """
        # Get original explanation
        orig_exp = explainer.explain(instance.reshape(1, -1))
        orig_importance = np.array(orig_exp.explanation_data['feature_importance'])

        # Target: make target_feature the most important
        if target_explanation and 'target_feature' in target_explanation:
            target_feature = target_explanation['target_feature']
        else:
            # Default: make least important feature most important
            target_feature = np.argmin(np.abs(orig_importance))

        # Initialize adversarial example
        adv_instance = instance.copy()
        best_adv = None
        best_success = False

        # Gradient-based optimization (simplified)
        learning_rate = epsilon / max_iter

        for iteration in range(max_iter):
            # Get current explanation
            current_exp = explainer.explain(adv_instance.reshape(1, -1))
            current_importance = np.array(
                current_exp.explanation_data['feature_importance']
            )

            # Check success: is target_feature top-ranked?
            top_feature = np.argmax(np.abs(current_importance))
            if top_feature == target_feature:
                best_success = True
                best_adv = adv_instance.copy()
                break

            # Compute gradient (finite differences approximation)
            gradient = np.zeros_like(instance)

            for i in range(len(instance)):
                # Perturb feature i
                perturbed = adv_instance.copy()
                perturbed[i] += 1e-4

                # Get explanation
                perturbed_exp = explainer.explain(perturbed.reshape(1, -1))
                perturbed_importance = np.array(
                    perturbed_exp.explanation_data['feature_importance']
                )

                # Gradient = change in target feature importance
                gradient[i] = (
                    perturbed_importance[target_feature] -
                    current_importance[target_feature]
                ) / 1e-4

            # Update adversarial example
            adv_instance += learning_rate * gradient

            # Project to epsilon ball
            perturbation = adv_instance - instance
            perturbation = np.clip(perturbation, -epsilon, epsilon)
            adv_instance = instance + perturbation

            # Keep track of best
            if best_adv is None or best_success:
                best_adv = adv_instance.copy()

        # Get final explanations
        adv_exp = explainer.explain(best_adv.reshape(1, -1))
        adv_importance = np.array(adv_exp.explanation_data['feature_importance'])

        perturbation = best_adv - instance

        return AdversarialExample(
            original_instance=instance,
            adversarial_instance=best_adv,
            perturbation=perturbation,
            original_explanation={'importance': orig_importance},
            adversarial_explanation={'importance': adv_importance},
            attack_type=AttackType.RANKING_ATTACK,
            success=best_success,
            l_inf_norm=np.max(np.abs(perturbation)),
            l2_norm=np.linalg.norm(perturbation)
        )


class FairwashingAttack(ExplanationAttack):
    """
    Attack to hide discriminatory features.

    Makes protected attributes appear unimportant in explanations.
    """

    def attack(
        self,
        model: Any,
        explainer: Any,
        instance: np.ndarray,
        target_explanation: Optional[Dict] = None,
        epsilon: float = 0.1,
        max_iter: int = 100
    ) -> AdversarialExample:
        """
        Hide protected attributes in explanation.

        Parameters
        ----------
        target_explanation : dict
            Should contain 'protected_features': list of protected feature indices.
        """
        # Get original explanation
        orig_exp = explainer.explain(instance.reshape(1, -1))
        orig_importance = np.array(orig_exp.explanation_data['feature_importance'])

        # Protected features to hide
        if target_explanation and 'protected_features' in target_explanation:
            protected_features = target_explanation['protected_features']
        else:
            # Default: hide top-2 features
            protected_features = np.argsort(np.abs(orig_importance))[-2:]

        # Objective: minimize importance of protected features
        adv_instance = instance.copy()
        learning_rate = epsilon / max_iter

        for iteration in range(max_iter):
            # Get current explanation
            current_exp = explainer.explain(adv_instance.reshape(1, -1))
            current_importance = np.array(
                current_exp.explanation_data['feature_importance']
            )

            # Loss: sum of protected feature importances
            protected_importance = np.sum(
                np.abs([current_importance[f] for f in protected_features])
            )

            # Check success
            if protected_importance < 0.1:  # Threshold
                break

            # Compute gradient (finite differences)
            gradient = np.zeros_like(instance)

            for i in range(len(instance)):
                perturbed = adv_instance.copy()
                perturbed[i] += 1e-4

                perturbed_exp = explainer.explain(perturbed.reshape(1, -1))
                perturbed_importance = np.array(
                    perturbed_exp.explanation_data['feature_importance']
                )

                # Gradient = change in protected importance
                perturbed_protected = np.sum(
                    np.abs([perturbed_importance[f] for f in protected_features])
                )

                gradient[i] = (perturbed_protected - protected_importance) / 1e-4

            # Gradient descent to minimize protected importance
            adv_instance -= learning_rate * gradient

            # Project to epsilon ball
            perturbation = adv_instance - instance
            perturbation = np.clip(perturbation, -epsilon, epsilon)
            adv_instance = instance + perturbation

        # Get final explanation
        adv_exp = explainer.explain(adv_instance.reshape(1, -1))
        adv_importance = np.array(adv_exp.explanation_data['feature_importance'])

        perturbation = adv_instance - instance

        # Check success
        final_protected_importance = np.sum(
            np.abs([adv_importance[f] for f in protected_features])
        )
        success = final_protected_importance < np.sum(
            np.abs([orig_importance[f] for f in protected_features])
        ) * 0.5

        return AdversarialExample(
            original_instance=instance,
            adversarial_instance=adv_instance,
            perturbation=perturbation,
            original_explanation={'importance': orig_importance},
            adversarial_explanation={'importance': adv_importance},
            attack_type=AttackType.FAIRWASHING_ATTACK,
            success=success,
            l_inf_norm=np.max(np.abs(perturbation)),
            l2_norm=np.linalg.norm(perturbation)
        )


class ExplanationDefense(ABC):
    """Base class for defense mechanisms against adversarial explanations."""

    @abstractmethod
    def defend(
        self,
        model: Any,
        explainer: Any,
        instance: np.ndarray
    ) -> Dict[str, Any]:
        """
        Generate robust explanation resistant to adversarial attacks.
        """
        pass


class EnsembleDefense(ExplanationDefense):
    """
    Ensemble defense: aggregate explanations from multiple methods.

    Robust to attacks targeting specific explanation methods.
    """

    def __init__(self, explainer_list: List[Any]):
        """
        Initialize ensemble defense.

        Parameters
        ----------
        explainer_list : list
            List of different explanation methods.
        """
        self.explainer_list = explainer_list

    def defend(
        self,
        model: Any,
        explainer: Any,
        instance: np.ndarray
    ) -> Dict[str, Any]:
        """
        Generate ensemble explanation.

        Parameters
        ----------
        Returns majority vote or median of explanations.
        """
        all_importances = []

        for exp in self.explainer_list:
            explanation = exp.explain(instance.reshape(1, -1))
            importance = np.array(explanation.explanation_data['feature_importance'])
            all_importances.append(importance)

        # Aggregate: median for robustness
        aggregated_importance = np.median(all_importances, axis=0)

        # Compute consensus (agreement across methods)
        rankings = [np.argsort(imp)[::-1] for imp in all_importances]
        top_3_sets = [set(rank[:3]) for rank in rankings]

        # Jaccard similarity of top-3 features
        consensus = []
        for i in range(len(top_3_sets)):
            for j in range(i+1, len(top_3_sets)):
                jaccard = len(top_3_sets[i] & top_3_sets[j]) / len(top_3_sets[i] | top_3_sets[j])
                consensus.append(jaccard)

        consensus_score = np.mean(consensus) if consensus else 0.0

        return {
            'importance': aggregated_importance,
            'consensus_score': consensus_score,
            'method': 'ensemble_defense',
            'robust': consensus_score > 0.7
        }


class SmoothDefense(ExplanationDefense):
    """
    Smooth defense: average explanations over neighborhood.

    Similar to randomized smoothing for certified robustness.
    """

    def __init__(self, noise_scale: float = 0.1, n_samples: int = 50):
        """
        Initialize smooth defense.

        Parameters
        ----------
        noise_scale : float
            Standard deviation of Gaussian noise.

        n_samples : int
            Number of samples for averaging.
        """
        self.noise_scale = noise_scale
        self.n_samples = n_samples

    def defend(
        self,
        model: Any,
        explainer: Any,
        instance: np.ndarray
    ) -> Dict[str, Any]:
        """
        Generate smoothed explanation by averaging over noisy samples.
        """
        all_importances = []

        for _ in range(self.n_samples):
            # Add Gaussian noise
            noisy_instance = instance + np.random.randn(*instance.shape) * self.noise_scale

            # Get explanation
            explanation = explainer.explain(noisy_instance.reshape(1, -1))
            importance = np.array(explanation.explanation_data['feature_importance'])
            all_importances.append(importance)

        # Average explanations
        smoothed_importance = np.mean(all_importances, axis=0)

        # Compute stability (variance)
        variance = np.var(all_importances, axis=0)
        stability_score = 1.0 / (1.0 + variance.mean())

        return {
            'importance': smoothed_importance,
            'variance': variance,
            'stability_score': stability_score,
            'method': 'smooth_defense',
            'robust': stability_score > 0.8
        }


class AdversarialDetector:
    """
    Detects adversarial manipulation of explanations.

    Uses statistical tests and consistency checks.
    """

    def __init__(self, model: Any, explainer: Any):
        """Initialize detector."""
        self.model = model
        self.explainer = explainer

    def detect(
        self,
        instance: np.ndarray,
        explanation: Dict[str, Any],
        detection_method: str = "consistency"
    ) -> Tuple[bool, float]:
        """
        Detect if explanation is adversarially manipulated.

        Parameters
        ----------
        instance : np.ndarray
            Instance being explained.

        explanation : dict
            Explanation to verify.

        detection_method : str
            Detection method: 'consistency', 'statistical', 'ensemble'.

        Returns
        -------
        tuple
            (is_adversarial, confidence)
        """
        if detection_method == "consistency":
            return self._consistency_check(instance, explanation)
        elif detection_method == "statistical":
            return self._statistical_test(instance, explanation)
        elif detection_method == "ensemble":
            return self._ensemble_check(instance, explanation)
        else:
            raise ValueError(f"Unknown detection method: {detection_method}")

    def _consistency_check(
        self,
        instance: np.ndarray,
        explanation: Dict[str, Any]
    ) -> Tuple[bool, float]:
        """
        Check consistency between explanation and model behavior.

        Inconsistent explanations may indicate adversarial manipulation.
        """
        importance = explanation.get('importance', explanation.get('feature_importance'))

        # Get top-3 features according to explanation
        top_features = np.argsort(np.abs(importance))[-3:]

        # Verify by perturbation
        base_pred = self.model.predict([instance])[0]
        consistency_scores = []

        for feature_idx in top_features:
            # Perturb feature
            perturbed = instance.copy()
            perturbed[feature_idx] += 0.5  # Increase feature

            # Check prediction change
            perturbed_pred = self.model.predict([perturbed])[0]
            pred_change = abs(perturbed_pred - base_pred)

            # High importance should lead to high prediction change
            expected_importance = importance[feature_idx]
            consistency = pred_change / (abs(expected_importance) + 1e-10)
            consistency_scores.append(consistency)

        # Aggregate consistency
        avg_consistency = np.mean(consistency_scores)

        # Low consistency suggests adversarial manipulation
        is_adversarial = avg_consistency < 0.5
        confidence = 1.0 - avg_consistency

        return is_adversarial, confidence

    def _statistical_test(
        self,
        instance: np.ndarray,
        explanation: Dict[str, Any]
    ) -> Tuple[bool, float]:
        """
        Statistical test for adversarial explanations.

        Uses distribution of explanations over similar instances.
        """
        importance = explanation.get('importance', explanation.get('feature_importance'))

        # Generate similar instances
        n_similar = 50
        similar_instances = []

        for _ in range(n_similar):
            noise = np.random.randn(*instance.shape) * 0.1
            similar = instance + noise
            similar_instances.append(similar)

        # Get explanations for similar instances
        similar_importances = []

        for similar in similar_instances:
            exp = self.explainer.explain(similar.reshape(1, -1))
            sim_importance = np.array(exp.explanation_data['feature_importance'])
            similar_importances.append(sim_importance)

        similar_importances = np.array(similar_importances)

        # Statistical test: is given explanation an outlier?
        mean_importance = similar_importances.mean(axis=0)
        std_importance = similar_importances.std(axis=0) + 1e-10

        # Z-score
        z_scores = np.abs((importance - mean_importance) / std_importance)

        # High z-scores suggest outlier/adversarial
        max_z = np.max(z_scores)
        is_adversarial = max_z > 3.0  # 3-sigma threshold

        confidence = min(max_z / 3.0, 1.0)

        return is_adversarial, confidence

    def _ensemble_check(
        self,
        instance: np.ndarray,
        explanation: Dict[str, Any]
    ) -> Tuple[bool, float]:
        """
        Ensemble-based detection using multiple explanation methods.
        """
        # This would require multiple explainers
        # Simplified: check against single explainer multiple times
        importance = explanation.get('importance', explanation.get('feature_importance'))

        # Get multiple explanations
        n_runs = 10
        importances = []

        for _ in range(n_runs):
            exp = self.explainer.explain(instance.reshape(1, -1))
            imp = np.array(exp.explanation_data['feature_importance'])
            importances.append(imp)

        importances = np.array(importances)

        # Check if given explanation deviates from average
        mean_importance = importances.mean(axis=0)
        deviation = np.linalg.norm(importance - mean_importance)

        # High deviation suggests adversarial
        is_adversarial = deviation > 1.0
        confidence = min(deviation, 1.0)

        return is_adversarial, confidence


def example_adversarial_xai():
    """Example of adversarial XAI workflow."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification

    # Train model
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    # Simple explainer
    class SimpleExplainer:
        def __init__(self, model):
            self.model = model

        def explain(self, X):
            importance = self.model.feature_importances_
            class Result:
                def __init__(self, importance):
                    self.explanation_data = {'feature_importance': importance}
            return Result(importance)

    explainer = SimpleExplainer(model)

    # Original explanation
    instance = X[0]
    orig_exp = explainer.explain(instance.reshape(1, -1))
    orig_importance = orig_exp.explanation_data['feature_importance']

    print("Original explanation:")
    print(f"Top feature: {np.argmax(orig_importance)}")
    print(f"Importance: {orig_importance}")

    # Attack: make feature 0 appear most important
    attacker = FeatureRankingAttack()
    adv_example = attacker.attack(
        model, explainer, instance,
        target_explanation={'target_feature': 0},
        epsilon=0.3
    )

    print(f"\nAttack success: {adv_example.success}")
    print(f"Perturbation L∞: {adv_example.l_inf_norm:.4f}")
    print(f"Adversarial top feature: {np.argmax(adv_example.adversarial_explanation['importance'])}")

    # Defense: smooth explanation
    defense = SmoothDefense(noise_scale=0.1, n_samples=30)
    robust_exp = defense.defend(model, explainer, adv_example.adversarial_instance)

    print(f"\nRobust explanation (defense):")
    print(f"Stability score: {robust_exp['stability_score']:.4f}")
    print(f"Robust: {robust_exp['robust']}")

    # Detection
    detector = AdversarialDetector(model, explainer)
    is_adv, confidence = detector.detect(
        adv_example.adversarial_instance,
        {'importance': adv_example.adversarial_explanation['importance']},
        detection_method='consistency'
    )

    print(f"\nDetection results:")
    print(f"Is adversarial: {is_adv}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    example_adversarial_xai()
