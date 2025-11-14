"""
Continual/Lifelong Learning Explainability.

Explains how models evolve over time and detect catastrophic forgetting.

Author: XPLIA Team
License: MIT
"""

import numpy as np
from typing import Dict, Any, Optional, List

class ExplanationEvolutionTracker:
    """Track how explanations change over time."""

    def __init__(self):
        self.explanation_history = []

    def track_explanation(self, task_id: int, explanation: np.ndarray):
        """Track explanation for current task."""
        self.explanation_history.append({
            'task_id': task_id,
            'explanation': explanation,
            'timestamp': len(self.explanation_history)
        })

    def detect_drift(self) -> Dict[str, Any]:
        """Detect if explanations are drifting."""
        if len(self.explanation_history) < 2:
            return {'drift_detected': False}

        recent = self.explanation_history[-1]['explanation']
        previous = self.explanation_history[-2]['explanation']

        drift_magnitude = float(np.linalg.norm(recent - previous))

        return {
            'drift_detected': drift_magnitude > 0.5,
            'drift_magnitude': drift_magnitude,
            'n_tasks_tracked': len(self.explanation_history)
        }

class CatastrophicForgettingDetector:
    """Detect catastrophic forgetting in continual learning."""

    def detect_forgetting(
        self,
        task_performances: Dict[int, float],
        current_task: int
    ) -> Dict[str, Any]:
        """Detect if model forgot previous tasks."""

        forgetting_scores = {}
        for task_id, perf in task_performances.items():
            if task_id < current_task:
                # Performance drop indicates forgetting
                forgetting = max(0, 1.0 - perf)  # Assume perf in [0, 1]
                forgetting_scores[task_id] = forgetting

        avg_forgetting = np.mean(list(forgetting_scores.values())) if forgetting_scores else 0

        return {
            'forgetting_detected': avg_forgetting > 0.3,
            'average_forgetting': float(avg_forgetting),
            'per_task_forgetting': forgetting_scores,
            'current_task': current_task
        }

# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("Continual Learning Explainability - Example")
    print("=" * 80)

    print("\n1. EXPLANATION EVOLUTION TRACKING")
    print("-" * 80)
    tracker = ExplanationEvolutionTracker()

    for task in range(5):
        exp = np.random.randn(10)
        tracker.track_explanation(task, exp)

    drift = tracker.detect_drift()
    print(f"Drift detected: {drift['drift_detected']}")
    print(f"Drift magnitude: {drift['drift_magnitude']:.4f}")
    print(f"Tasks tracked: {drift['n_tasks_tracked']}")

    print("\n2. CATASTROPHIC FORGETTING DETECTION")
    print("-" * 80)
    task_perfs = {0: 0.95, 1: 0.90, 2: 0.60, 3: 0.55}  # Perf degraded
    current = 4

    detector = CatastrophicForgettingDetector()
    result = detector.detect_forgetting(task_perfs, current)

    print(f"Forgetting detected: {result['forgetting_detected']}")
    print(f"Average forgetting: {result['average_forgetting']:.2%}")
    print(f"\nPer-task forgetting:")
    for task, forget in result['per_task_forgetting'].items():
        print(f"  Task {task}: {forget:.2%}")

    print("\n" + "=" * 80)
