"""
Tests for TIER 2 - Research Excellence Modules
Tests for Meta-Learning, Neuro-Symbolic, Continual, Bayesian, MoE, and RecSys
"""

import pytest
import numpy as np
from typing import Dict, List, Any

# Meta-Learning
from xplia.explainers.metalearning.metalearning_explainer import (
    MAMLExplainer,
    PrototypicalNetworkExplainer,
    FewShotExplainer,
)

# Neuro-Symbolic
from xplia.explainers.neurosymbolic.neurosymbolic_explainer import (
    RuleExtractor,
    LogicExplainer,
)

# Continual Learning
from xplia.explainers.continual.continual_explainer import (
    ExplanationEvolutionTracker,
    CatastrophicForgettingDetector,
)

# Bayesian
from xplia.explainers.bayesian.bayesian_explainer import (
    UncertaintyDecomposer,
    BayesianFeatureImportance,
)

# Mixture of Experts
from xplia.explainers.moe.moe_explainer import (
    ExpertRoutingExplainer,
    ExpertSpecializationAnalyzer,
)

# Recommender Systems
from xplia.explainers.recommender.recsys_explainer import (
    CollaborativeFilteringExplainer,
    MatrixFactorizationExplainer,
)


# ===================
# META-LEARNING
# ===================

class TestMAMLExplainer:
    """Test MAML Explainer."""

    def test_initialization(self):
        """Test initialization."""
        explainer = MAMLExplainer()
        assert explainer is not None

    def test_explain_adaptation(self):
        """Test adaptation explanation."""
        explainer = MAMLExplainer()

        support_set = (np.random.rand(5, 10), np.random.randint(0, 2, 5))
        query_instance = np.random.rand(10)

        explanation = explainer.explain_adaptation(
            support_set,
            query_instance,
            n_adaptation_steps=5
        )

        assert explanation is not None
        assert hasattr(explanation, 'task_importance')
        assert hasattr(explanation, 'adaptation_trajectory')


class TestPrototypicalNetworkExplainer:
    """Test Prototypical Network Explainer."""

    def test_initialization(self):
        """Test initialization."""
        explainer = PrototypicalNetworkExplainer()
        assert explainer is not None

    def test_explain_prototype_distance(self):
        """Test prototype distance explanation."""
        explainer = PrototypicalNetworkExplainer()

        support_set = (np.random.rand(10, 20), np.random.randint(0, 3, 10))
        query_instance = np.random.rand(20)

        explanation = explainer.explain_prototype_distance(support_set, query_instance)

        assert explanation is not None
        assert hasattr(explanation, 'prototype_distances')
        assert hasattr(explanation, 'predicted_class')


class TestFewShotExplainer:
    """Test Few-Shot Explainer."""

    def test_explain_few_shot(self):
        """Test few-shot explanation."""
        explainer = FewShotExplainer()

        support_set = (np.random.rand(5, 15), np.array([0, 0, 1, 1, 1]))
        query = np.random.rand(15)

        explanation = explainer.explain(support_set, query)

        assert explanation is not None


# ===================
# NEURO-SYMBOLIC
# ===================

class TestRuleExtractor:
    """Test Rule Extractor."""

    def test_initialization(self):
        """Test initialization."""
        extractor = RuleExtractor()
        assert extractor is not None

    def test_extract_decision_rules(self):
        """Test decision rule extraction."""
        extractor = RuleExtractor()

        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)

        rules = extractor.extract_decision_rules(X, y)

        assert isinstance(rules, list)
        assert len(rules) > 0
        # Each rule should have condition, confidence, support
        for rule in rules:
            assert 'condition' in rule
            assert 'confidence' in rule


class TestLogicExplainer:
    """Test Logic Explainer."""

    def test_initialization(self):
        """Test initialization."""
        explainer = LogicExplainer()
        assert explainer is not None

    def test_explain_logic(self):
        """Test logic-based explanation."""
        explainer = LogicExplainer()

        instance = np.array([1.0, 0.5, 0.3, 0.8])
        prediction = 1

        explanation = explainer.explain(instance, prediction)

        assert explanation is not None
        assert hasattr(explanation, 'logic_rules')


# ===================
# CONTINUAL LEARNING
# ===================

class TestExplanationEvolutionTracker:
    """Test Explanation Evolution Tracker."""

    def test_initialization(self):
        """Test initialization."""
        tracker = ExplanationEvolutionTracker()
        assert tracker is not None

    def test_track_evolution(self):
        """Test explanation evolution tracking."""
        tracker = ExplanationEvolutionTracker()

        # Simulate multiple tasks
        for task_id in range(3):
            explanation = {
                'task_id': task_id,
                'feature_importance': np.random.rand(10),
                'accuracy': 0.8 + np.random.rand() * 0.1
            }
            tracker.add_explanation(task_id, explanation)

        evolution = tracker.get_evolution_trajectory()

        assert evolution is not None
        assert len(evolution) == 3


class TestCatastrophicForgettingDetector:
    """Test Catastrophic Forgetting Detector."""

    def test_initialization(self):
        """Test initialization."""
        detector = CatastrophicForgettingDetector()
        assert detector is not None

    def test_detect_forgetting(self):
        """Test forgetting detection."""
        detector = CatastrophicForgettingDetector()

        # Simulate task performances
        task_performances = {
            'task_0': [0.9, 0.85, 0.7, 0.6],  # Forgetting
            'task_1': [0.85, 0.88, 0.9, 0.87],  # Stable
        }

        result = detector.detect(task_performances)

        assert result is not None
        assert hasattr(result, 'forgetting_detected')
        assert hasattr(result, 'affected_tasks')


# ===================
# BAYESIAN
# ===================

class TestUncertaintyDecomposer:
    """Test Uncertainty Decomposer."""

    def test_initialization(self):
        """Test initialization."""
        decomposer = UncertaintyDecomposer()
        assert decomposer is not None

    def test_decompose_uncertainty(self):
        """Test uncertainty decomposition."""
        decomposer = UncertaintyDecomposer()

        x = np.random.rand(1, 10)

        result = decomposer.decompose_uncertainty(x, n_samples=10)

        assert result is not None
        assert hasattr(result, 'epistemic_uncertainty')
        assert hasattr(result, 'aleatoric_uncertainty')
        assert hasattr(result, 'total_uncertainty')


class TestBayesianFeatureImportance:
    """Test Bayesian Feature Importance."""

    def test_initialization(self):
        """Test initialization."""
        explainer = BayesianFeatureImportance()
        assert explainer is not None

    def test_explain_with_credible_intervals(self):
        """Test feature importance with credible intervals."""
        explainer = BayesianFeatureImportance()

        X = np.random.rand(50, 8)
        y = np.random.randint(0, 2, 50)

        explanation = explainer.explain(X, y, credible_level=0.95)

        assert explanation is not None
        assert hasattr(explanation, 'feature_importance_mean')
        assert hasattr(explanation, 'credible_intervals')


# ===================
# MIXTURE OF EXPERTS
# ===================

class TestExpertRoutingExplainer:
    """Test Expert Routing Explainer."""

    def test_initialization(self):
        """Test initialization."""
        explainer = ExpertRoutingExplainer(num_experts=4)
        assert explainer.num_experts == 4

    def test_explain_routing(self):
        """Test routing explanation."""
        explainer = ExpertRoutingExplainer(num_experts=4)

        instance = np.random.rand(10)

        explanation = explainer.explain_routing(instance)

        assert explanation is not None
        assert hasattr(explanation, 'expert_weights')
        assert hasattr(explanation, 'selected_experts')
        assert len(explanation.expert_weights) == 4


class TestExpertSpecializationAnalyzer:
    """Test Expert Specialization Analyzer."""

    def test_initialization(self):
        """Test initialization."""
        analyzer = ExpertSpecializationAnalyzer(num_experts=3)
        assert analyzer.num_experts == 3

    def test_analyze_specialization(self):
        """Test specialization analysis."""
        analyzer = ExpertSpecializationAnalyzer(num_experts=3)

        # Simulate data and expert assignments
        X = np.random.rand(100, 10)
        expert_assignments = np.random.randint(0, 3, 100)

        analysis = analyzer.analyze_specialization(X, expert_assignments)

        assert analysis is not None
        assert hasattr(analysis, 'expert_clusters')
        assert hasattr(analysis, 'specialization_scores')


# ===================
# RECOMMENDER SYSTEMS
# ===================

class TestCollaborativeFilteringExplainer:
    """Test Collaborative Filtering Explainer."""

    def test_initialization(self):
        """Test initialization."""
        explainer = CollaborativeFilteringExplainer(
            user_item_matrix=np.random.rand(50, 30),
            similarity_metric='cosine'
        )
        assert explainer is not None

    def test_explain_recommendation(self):
        """Test recommendation explanation."""
        user_item_matrix = np.random.rand(50, 30)
        explainer = CollaborativeFilteringExplainer(
            user_item_matrix=user_item_matrix,
            similarity_metric='cosine'
        )

        user_id = 0
        item_id = 5

        explanation = explainer.explain_recommendation(user_id, item_id)

        assert explanation is not None
        assert hasattr(explanation, 'similar_users')
        assert hasattr(explanation, 'similar_items_liked')


class TestMatrixFactorizationExplainer:
    """Test Matrix Factorization Explainer."""

    def test_initialization(self):
        """Test initialization."""
        explainer = MatrixFactorizationExplainer(n_factors=10)
        assert explainer.n_factors == 10

    def test_fit_and_explain(self):
        """Test fitting and explanation."""
        explainer = MatrixFactorizationExplainer(n_factors=10)

        user_item_matrix = np.random.rand(30, 20)
        explainer.fit(user_item_matrix)

        user_id = 0
        item_id = 5

        explanation = explainer.explain_recommendation(user_id, item_id)

        assert explanation is not None
        assert hasattr(explanation, 'factor_importance')
        assert hasattr(explanation, 'latent_factors')


# Integration Tests
class TestTier2Integration:
    """Integration tests for TIER 2 modules."""

    def test_meta_learning_with_continual(self):
        """Test meta-learning with continual learning tracking."""
        maml_explainer = MAMLExplainer()
        evolution_tracker = ExplanationEvolutionTracker()

        # Simulate few-shot learning on multiple tasks
        for task_id in range(3):
            support_set = (np.random.rand(5, 10), np.random.randint(0, 2, 5))
            query = np.random.rand(10)

            explanation = maml_explainer.explain_adaptation(support_set, query)

            # Track evolution
            evolution_tracker.add_explanation(task_id, {
                'task_id': task_id,
                'explanation': explanation
            })

        evolution = evolution_tracker.get_evolution_trajectory()
        assert len(evolution) == 3

    def test_bayesian_with_moe(self):
        """Test Bayesian uncertainty with MoE."""
        uncertainty_decomposer = UncertaintyDecomposer()
        moe_explainer = ExpertRoutingExplainer(num_experts=3)

        instance = np.random.rand(1, 10)

        # Decompose uncertainty
        uncertainty = uncertainty_decomposer.decompose_uncertainty(instance)

        # Explain expert routing
        routing = moe_explainer.explain_routing(instance[0])

        assert uncertainty is not None
        assert routing is not None

    def test_neuro_symbolic_with_recsys(self):
        """Test rule extraction with recommender explanations."""
        rule_extractor = RuleExtractor()
        cf_explainer = CollaborativeFilteringExplainer(
            user_item_matrix=np.random.rand(50, 30)
        )

        # Extract rules from user preferences
        X = np.random.rand(50, 5)  # User features
        y = np.random.randint(0, 2, 50)  # Liked/disliked

        rules = rule_extractor.extract_decision_rules(X, y)

        # Explain recommendation
        rec_exp = cf_explainer.explain_recommendation(user_id=0, item_id=5)

        assert len(rules) > 0
        assert rec_exp is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
