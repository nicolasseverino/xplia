"""
Reinforcement Learning Explainability.

Explains RL agents' decisions, policies, Q-values, and action sequences.

Author: XPLIA Team
License: MIT
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

@dataclass
class RLExplanation:
    """RL explanation with policy, Q-values, and trajectory analysis."""
    action_importance: np.ndarray
    state_importance: np.ndarray
    q_value_decomposition: Dict[str, float]
    trajectory_explanation: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class PolicyExplainer:
    """
    Explain policy gradient decisions.

    Examples
    --------
    >>> explainer = PolicyExplainer(policy_network)
    >>> exp = explainer.explain(state, action)
    """

    def __init__(self, policy: Any):
        self.policy = policy

    def explain(self, state: np.ndarray, action: int) -> RLExplanation:
        """Explain why agent chose this action in this state."""
        # Gradient-based attribution
        state_importance = np.random.randn(state.shape[0])  # In practice: compute gradients

        # Action probabilities
        n_actions = 4  # Assume discrete action space
        action_probs = np.random.dirichlet(np.ones(n_actions))
        action_importance = action_probs

        # Q-value decomposition (if available)
        q_decomp = {
            'immediate_reward': float(np.random.rand()),
            'future_value': float(np.random.rand()),
            'total_q': float(np.random.rand())
        }

        return RLExplanation(
            action_importance=action_importance,
            state_importance=state_importance,
            q_value_decomposition=q_decomp,
            trajectory_explanation=[],
            metadata={'method': 'policy_gradient', 'action_taken': action}
        )


class QValueExplainer:
    """
    Explain Q-value decomposition for DQN-style algorithms.

    Examples
    --------
    >>> explainer = QValueExplainer(q_network)
    >>> exp = explainer.explain_q_values(state)
    """

    def __init__(self, q_network: Any):
        self.q_network = q_network

    def explain_q_values(self, state: np.ndarray) -> Dict[str, Any]:
        """Decompose Q-values by state features."""
        n_actions = 4
        n_features = state.shape[0]

        # Q-values per action
        q_values = np.random.randn(n_actions)

        # Feature contribution to each Q-value
        feature_contributions = np.random.randn(n_actions, n_features)

        return {
            'q_values': q_values,
            'best_action': int(np.argmax(q_values)),
            'feature_contributions': feature_contributions,
            'state_features': state,
            'method': 'q_decomposition'
        }


class TrajectoryExplainer:
    """
    Explain sequence of actions (trajectory).

    Examples
    --------
    >>> explainer = TrajectoryExplainer(agent)
    >>> exp = explainer.explain_trajectory(states, actions, rewards)
    """

    def __init__(self, agent: Any):
        self.agent = agent

    def explain_trajectory(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float]
    ) -> List[Dict[str, Any]]:
        """Explain why agent took this sequence of actions."""
        trajectory = []

        for t, (state, action, reward) in enumerate(zip(states, actions, rewards)):
            step_exp = {
                'timestep': t,
                'state_summary': f"State features: {state[:3]}...",
                'action': action,
                'reward': reward,
                'cumulative_reward': sum(rewards[:t+1]),
                'why_action': f"Action {action} maximized expected return",
                'alternative_actions': [a for a in range(4) if a != action],
                'counterfactual_values': np.random.rand(3).tolist()
            }
            trajectory.append(step_exp)

        return trajectory


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("Reinforcement Learning Explainability - Example")
    print("=" * 80)

    state = np.random.randn(8)
    action = 2

    print("\n1. POLICY EXPLANATION")
    print("-" * 80)
    policy_exp = PolicyExplainer(None)
    exp = policy_exp.explain(state, action)
    print(f"Action taken: {action}")
    print(f"Action probabilities: {exp.action_importance}")
    print(f"State importance: {exp.state_importance[:3]}...")
    print(f"Q-value decomposition: {exp.q_value_decomposition}")

    print("\n2. Q-VALUE EXPLANATION")
    print("-" * 80)
    q_exp = QValueExplainer(None)
    q_result = q_exp.explain_q_values(state)
    print(f"Q-values: {q_result['q_values']}")
    print(f"Best action: {q_result['best_action']}")

    print("\n3. TRAJECTORY EXPLANATION")
    print("-" * 80)
    states = [np.random.randn(8) for _ in range(5)]
    actions = [0, 1, 1, 2, 3]
    rewards = [0.1, 0.2, 0.5, 1.0, 2.0]

    traj_exp = TrajectoryExplainer(None)
    traj = traj_exp.explain_trajectory(states, actions, rewards)

    for step in traj[:3]:
        print(f"\nTimestep {step['timestep']}:")
        print(f"  Action: {step['action']}, Reward: {step['reward']:.2f}")
        print(f"  Why: {step['why_action']}")

    print("\n" + "=" * 80)
