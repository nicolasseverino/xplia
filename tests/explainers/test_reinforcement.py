"""
Tests for TIER 1 - Reinforcement Learning Explainers
Tests for Policy, Q-value, and Trajectory explanations
"""

import pytest
import numpy as np
from typing import Dict, Any

from xplia.explainers.reinforcement.rl_explainer import (
    PolicyExplainer,
    QValueExplainer,
    TrajectoryExplainer,
    RLExplanation,
)


class TestPolicyExplainer:
    """Test suite for Policy explainer."""

    def create_dummy_state(self) -> np.ndarray:
        """Create a dummy state."""
        return np.random.rand(4)  # 4-dimensional state

    def create_dummy_policy(self):
        """Create a dummy policy function."""
        def policy(state):
            return {
                'action': np.random.randint(0, 2),
                'action_probs': np.random.dirichlet([1, 1]),
                'value': np.random.rand()
            }
        return policy

    def test_initialization(self):
        """Test PolicyExplainer initialization."""
        policy = self.create_dummy_policy()
        explainer = PolicyExplainer(policy)
        assert explainer is not None

    def test_explain_action(self):
        """Test action explanation."""
        policy = self.create_dummy_policy()
        explainer = PolicyExplainer(policy)
        state = self.create_dummy_state()

        explanation = explainer.explain_action(state)

        assert isinstance(explanation, RLExplanation)
        assert hasattr(explanation, 'action')
        assert hasattr(explanation, 'action_probability')
        assert hasattr(explanation, 'state_feature_importance')

    def test_explain_policy_gradient(self):
        """Test policy gradient explanation."""
        policy = self.create_dummy_policy()
        explainer = PolicyExplainer(policy)
        state = self.create_dummy_state()

        explanation = explainer.explain_policy_gradient(state)

        assert explanation is not None
        assert hasattr(explanation, 'gradient_attribution')

    def test_counterfactual_action(self):
        """Test counterfactual action explanation."""
        policy = self.create_dummy_policy()
        explainer = PolicyExplainer(policy)
        state = self.create_dummy_state()
        target_action = 1

        cf_explanation = explainer.explain_counterfactual_action(
            state,
            target_action
        )

        assert cf_explanation is not None
        assert hasattr(cf_explanation, 'required_state_changes')

    def test_explain_invalid_state(self):
        """Test with invalid state."""
        policy = self.create_dummy_policy()
        explainer = PolicyExplainer(policy)

        # Empty state
        with pytest.raises((ValueError, IndexError)):
            explainer.explain_action(np.array([]))


class TestQValueExplainer:
    """Test suite for Q-Value explainer."""

    def create_dummy_q_network(self):
        """Create a dummy Q-network."""
        def q_network(state, action=None):
            if action is None:
                # Return Q-values for all actions
                return np.random.rand(4)  # 4 actions
            else:
                # Return Q-value for specific action
                return np.random.rand()
        return q_network

    def create_dummy_state(self) -> np.ndarray:
        """Create a dummy state."""
        return np.random.rand(4)

    def test_initialization(self):
        """Test QValueExplainer initialization."""
        q_network = self.create_dummy_q_network()
        explainer = QValueExplainer(q_network)
        assert explainer is not None

    def test_explain_q_values(self):
        """Test Q-value explanation."""
        q_network = self.create_dummy_q_network()
        explainer = QValueExplainer(q_network)
        state = self.create_dummy_state()

        explanation = explainer.explain_q_values(state)

        assert explanation is not None
        assert hasattr(explanation, 'q_values')
        assert hasattr(explanation, 'best_action')
        assert hasattr(explanation, 'advantage_values')

    def test_decompose_q_value(self):
        """Test Q-value decomposition."""
        q_network = self.create_dummy_q_network()
        explainer = QValueExplainer(q_network)
        state = self.create_dummy_state()
        action = 0

        decomposition = explainer.decompose_q_value(state, action)

        assert decomposition is not None
        assert hasattr(decomposition, 'state_contribution')
        assert hasattr(decomposition, 'action_contribution')

    def test_explain_action_ranking(self):
        """Test action ranking explanation."""
        q_network = self.create_dummy_q_network()
        explainer = QValueExplainer(q_network)
        state = self.create_dummy_state()

        ranking = explainer.explain_action_ranking(state)

        assert ranking is not None
        assert 'ranked_actions' in ranking
        assert 'q_values' in ranking


class TestTrajectoryExplainer:
    """Test suite for Trajectory explainer."""

    def create_dummy_trajectory(self) -> Dict[str, Any]:
        """Create a dummy trajectory."""
        length = 10
        return {
            'states': [np.random.rand(4) for _ in range(length)],
            'actions': [np.random.randint(0, 2) for _ in range(length)],
            'rewards': [np.random.rand() for _ in range(length)],
            'dones': [False] * (length - 1) + [True]
        }

    def test_initialization(self):
        """Test TrajectoryExplainer initialization."""
        explainer = TrajectoryExplainer()
        assert explainer is not None

    def test_explain_trajectory(self):
        """Test trajectory explanation."""
        explainer = TrajectoryExplainer()
        trajectory = self.create_dummy_trajectory()

        explanation = explainer.explain_trajectory(trajectory)

        assert explanation is not None
        assert hasattr(explanation, 'critical_states')
        assert hasattr(explanation, 'critical_actions')
        assert hasattr(explanation, 'state_importance')

    def test_identify_critical_states(self):
        """Test critical state identification."""
        explainer = TrajectoryExplainer()
        trajectory = self.create_dummy_trajectory()

        critical_states = explainer.identify_critical_states(trajectory)

        assert isinstance(critical_states, list)
        assert len(critical_states) > 0

    def test_explain_reward_attribution(self):
        """Test reward attribution."""
        explainer = TrajectoryExplainer()
        trajectory = self.create_dummy_trajectory()

        attribution = explainer.explain_reward_attribution(trajectory)

        assert attribution is not None
        assert 'state_rewards' in attribution
        assert 'action_rewards' in attribution

    def test_compare_trajectories(self):
        """Test trajectory comparison."""
        explainer = TrajectoryExplainer()

        traj1 = self.create_dummy_trajectory()
        traj2 = self.create_dummy_trajectory()

        comparison = explainer.compare_trajectories(traj1, traj2)

        assert comparison is not None
        assert 'differences' in comparison

    def test_empty_trajectory(self):
        """Test with empty trajectory."""
        explainer = TrajectoryExplainer()

        empty_traj = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': []
        }

        with pytest.raises((ValueError, AssertionError)):
            explainer.explain_trajectory(empty_traj)


# Integration tests
class TestRLIntegration:
    """Integration tests for RL explainers."""

    def test_policy_and_qvalue_consistency(self):
        """Test consistency between policy and Q-value explanations."""
        # Create dummy policy and Q-network
        def policy(state):
            return {
                'action': 0,
                'action_probs': np.array([0.7, 0.3]),
                'value': 1.5
            }

        def q_network(state, action=None):
            if action is None:
                return np.array([1.5, 0.8])
            return 1.5 if action == 0 else 0.8

        policy_explainer = PolicyExplainer(policy)
        q_explainer = QValueExplainer(q_network)

        state = np.random.rand(4)

        policy_exp = policy_explainer.explain_action(state)
        q_exp = q_explainer.explain_q_values(state)

        # Both should identify action 0 as best
        assert policy_exp.action == q_exp.best_action

    def test_trajectory_with_policy(self):
        """Test trajectory explanation with policy."""
        def policy(state):
            return {
                'action': np.random.randint(0, 2),
                'action_probs': np.random.dirichlet([1, 1]),
            }

        policy_explainer = PolicyExplainer(policy)
        traj_explainer = TrajectoryExplainer()

        # Generate trajectory
        trajectory = {
            'states': [np.random.rand(4) for _ in range(10)],
            'actions': [policy(np.random.rand(4))['action'] for _ in range(10)],
            'rewards': [np.random.rand() for _ in range(10)],
            'dones': [False] * 9 + [True]
        }

        # Explain trajectory
        traj_exp = traj_explainer.explain_trajectory(trajectory)

        # Explain critical states
        for state_idx in traj_exp.critical_states[:3]:
            state = trajectory['states'][state_idx]
            policy_exp = policy_explainer.explain_action(state)
            assert policy_exp is not None


# Performance tests
class TestRLPerformance:
    """Performance tests for RL explainers."""

    def test_long_trajectory_performance(self):
        """Test performance on long trajectories."""
        explainer = TrajectoryExplainer()

        # Create long trajectory
        length = 1000
        long_trajectory = {
            'states': [np.random.rand(4) for _ in range(length)],
            'actions': [np.random.randint(0, 2) for _ in range(length)],
            'rewards': [np.random.rand() for _ in range(length)],
            'dones': [False] * (length - 1) + [True]
        }

        # Should handle long trajectories
        explanation = explainer.explain_trajectory(long_trajectory)

        assert explanation is not None

    def test_batch_state_explanation(self):
        """Test batch state explanation."""
        def policy(state):
            return {
                'action': np.random.randint(0, 2),
                'action_probs': np.random.dirichlet([1, 1]),
            }

        explainer = PolicyExplainer(policy)

        # Batch of states
        states = [np.random.rand(4) for _ in range(100)]

        explanations = []
        for state in states:
            exp = explainer.explain_action(state)
            explanations.append(exp)

        assert len(explanations) == 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
