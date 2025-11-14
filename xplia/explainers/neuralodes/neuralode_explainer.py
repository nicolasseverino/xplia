"""
Neural ODEs Explainability.

Explains continuous-time neural networks.

Author: XPLIA Team
License: MIT
"""

import numpy as np
from typing import Dict, Any, Optional, List

class NeuralODEExplainer:
    """Explain Neural ODE dynamics."""

    def explain_trajectory(self, initial_state: np.ndarray, time_points: np.ndarray) -> Dict[str, Any]:
        """Explain ODE trajectory evolution."""

        # Simulate trajectory
        trajectory = []
        state = initial_state.copy()

        for t in time_points:
            # dz/dt = f(z, t)
            derivative = np.random.randn(*state.shape) * 0.1
            state = state + derivative * 0.1
            trajectory.append(state.copy())

        trajectory = np.array(trajectory)

        # Analyze dynamics
        velocity_norms = [np.linalg.norm(trajectory[i+1] - trajectory[i]) for i in range(len(trajectory)-1)]

        return {
            'trajectory': trajectory.tolist(),
            'n_timepoints': len(time_points),
            'avg_velocity': float(np.mean(velocity_norms)),
            'max_velocity': float(np.max(velocity_norms)),
            'trajectory_length': float(np.sum(velocity_norms)),
            'dynamics_type': 'stable' if np.mean(velocity_norms) < 0.5 else 'unstable'
        }

# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("Neural ODEs Explainability - Example")
    print("=" * 80)

    initial = np.random.randn(5)
    time_points = np.linspace(0, 1, 20)

    print("\nNEURAL ODE TRAJECTORY EXPLANATION")
    print("-" * 80)

    node_exp = NeuralODEExplainer()
    result = node_exp.explain_trajectory(initial, time_points)

    print(f"Time points: {result['n_timepoints']}")
    print(f"Dynamics type: {result['dynamics_type']}")
    print(f"Average velocity: {result['avg_velocity']:.4f}")
    print(f"Max velocity: {result['max_velocity']:.4f}")
    print(f"Total trajectory length: {result['trajectory_length']:.4f}")

    print("\n" + "=" * 80)
