"""
Real-Time Streaming XAI.

Explainability for streaming data and real-time systems.

Author: XPLIA Team
License: MIT
"""

import numpy as np
from typing import Dict, Any, Optional, List, Deque, Callable
from dataclasses import dataclass
from collections import deque
import time
import warnings

from xplia.core.base import ExplainerBase, ExplanationResult


@dataclass
class StreamingExplanation:
    """
    Explanation for streaming data point.

    Attributes
    ----------
    timestamp : float
        When explanation was generated.
    explanation : ExplanationResult
        Explanation for current instance.
    drift_detected : bool
        Whether concept drift was detected.
    latency_ms : float
        Explanation generation latency.
    metadata : dict
        Additional metadata.
    """
    timestamp: float
    explanation: ExplanationResult
    drift_detected: bool
    latency_ms: float
    metadata: Dict[str, Any]


class IncrementalExplainer:
    """
    Incremental explainer for streaming data.

    Updates explanations efficiently as new data arrives without
    recomputing from scratch.

    Parameters
    ----------
    base_explainer : ExplainerBase
        Base explainer to use.
    window_size : int
        Size of sliding window for statistics.
    update_frequency : int
        Update model every N samples.

    Examples
    --------
    >>> explainer = IncrementalExplainer(shap_explainer, window_size=100)
    >>> for x in data_stream:
    ...     explanation = explainer.explain_incremental(x)
    """

    def __init__(
        self,
        base_explainer: ExplainerBase,
        window_size: int = 100,
        update_frequency: int = 10
    ):
        self.base_explainer = base_explainer
        self.window_size = window_size
        self.update_frequency = update_frequency

        # Sliding window of recent samples
        self.window: Deque[np.ndarray] = deque(maxlen=window_size)

        # Running statistics
        self.mean = None
        self.std = None
        self.n_samples = 0

    def _update_statistics(self, x: np.ndarray):
        """Update running statistics with new sample."""
        self.window.append(x)
        self.n_samples += 1

        if len(self.window) > 0:
            window_array = np.array(self.window)
            self.mean = np.mean(window_array, axis=0)
            self.std = np.std(window_array, axis=0)

    def explain_incremental(
        self,
        x: np.ndarray,
        **kwargs
    ) -> StreamingExplanation:
        """
        Generate explanation incrementally.

        Parameters
        ----------
        x : ndarray
            New streaming instance.
        **kwargs
            Additional arguments for explainer.

        Returns
        -------
        streaming_exp : StreamingExplanation
            Streaming explanation.
        """
        start_time = time.time()

        # Update statistics
        self._update_statistics(x)

        # Detect drift (simple z-score based)
        drift_detected = False
        if self.mean is not None and self.std is not None:
            z_scores = np.abs((x - self.mean) / (self.std + 1e-8))
            drift_detected = np.any(z_scores > 3.0)

        # Generate explanation
        explanation = self.base_explainer.explain(x, **kwargs)

        # Compute latency
        latency_ms = (time.time() - start_time) * 1000

        return StreamingExplanation(
            timestamp=time.time(),
            explanation=explanation,
            drift_detected=drift_detected,
            latency_ms=latency_ms,
            metadata={
                'n_samples_seen': self.n_samples,
                'window_size': len(self.window)
            }
        )


class ApproximateExplainer:
    """
    Approximate explainer for low-latency requirements.

    Uses approximations to generate explanations faster at the cost
    of some accuracy.

    Parameters
    ----------
    base_explainer : ExplainerBase
        Base explainer.
    approximation_level : str
        'low' (high accuracy, slow), 'medium', 'high' (low accuracy, fast).

    Examples
    --------
    >>> explainer = ApproximateExplainer(shap_explainer, approximation_level='high')
    >>> explanation = explainer.explain(x)  # Fast but approximate
    """

    def __init__(
        self,
        base_explainer: ExplainerBase,
        approximation_level: str = 'medium'
    ):
        self.base_explainer = base_explainer
        self.approximation_level = approximation_level

        valid_levels = ['low', 'medium', 'high']
        if approximation_level not in valid_levels:
            raise ValueError(f"Approximation level must be one of {valid_levels}")

        # Map approximation level to parameters
        self.params = {
            'low': {'n_samples': 1000, 'max_iter': 100},
            'medium': {'n_samples': 100, 'max_iter': 20},
            'high': {'n_samples': 10, 'max_iter': 5}
        }[approximation_level]

    def explain(
        self,
        x: np.ndarray,
        **kwargs
    ) -> ExplanationResult:
        """
        Generate approximate explanation.

        Parameters
        ----------
        x : ndarray
            Instance to explain.
        **kwargs
            Additional arguments.

        Returns
        -------
        explanation : ExplanationResult
            Approximate explanation.
        """
        # Merge approximation params with kwargs
        kwargs.update(self.params)

        # Generate explanation with reduced computation
        explanation = self.base_explainer.explain(x, **kwargs)

        # Mark as approximate
        if explanation.metadata is None:
            explanation.metadata = {}
        explanation.metadata['approximate'] = True
        explanation.metadata['approximation_level'] = self.approximation_level

        return explanation


class DriftDetector:
    """
    Concept drift detector for streaming explanations.

    Detects when explanation patterns change over time, indicating
    concept drift.

    Parameters
    ----------
    window_size : int
        Window size for drift detection.
    threshold : float
        Drift detection threshold.

    Examples
    --------
    >>> detector = DriftDetector(window_size=50, threshold=0.1)
    >>> for explanation in stream:
    ...     drift = detector.detect(explanation)
    ...     if drift:
    ...         print("Concept drift detected!")
    """

    def __init__(
        self,
        window_size: int = 50,
        threshold: float = 0.1
    ):
        self.window_size = window_size
        self.threshold = threshold

        # Store recent explanations
        self.explanation_history: Deque[np.ndarray] = deque(maxlen=window_size)

        # Reference distribution (first window)
        self.reference_distribution = None

    def _extract_features(self, explanation: ExplanationResult) -> np.ndarray:
        """
        Extract features from explanation for drift detection.

        Parameters
        ----------
        explanation : ExplanationResult
            Explanation to extract features from.

        Returns
        -------
        features : ndarray
            Feature vector for drift detection.
        """
        if 'feature_importance' in explanation.explanation_data:
            return np.array(explanation.explanation_data['feature_importance'])
        else:
            # Fallback: use prediction if available
            return np.array([0.0])

    def detect(self, explanation: ExplanationResult) -> bool:
        """
        Detect if drift occurred.

        Parameters
        ----------
        explanation : ExplanationResult
            New explanation.

        Returns
        -------
        drift_detected : bool
            True if drift detected.
        """
        features = self._extract_features(explanation)
        self.explanation_history.append(features)

        # Need full window for detection
        if len(self.explanation_history) < self.window_size:
            return False

        # Set reference on first full window
        if self.reference_distribution is None:
            self.reference_distribution = np.mean(
                list(self.explanation_history)[:self.window_size // 2],
                axis=0
            )
            return False

        # Compare current window to reference
        current_distribution = np.mean(
            list(self.explanation_history)[self.window_size // 2:],
            axis=0
        )

        # Compute distributional distance
        distance = np.linalg.norm(current_distribution - self.reference_distribution)

        # Detect drift
        if distance > self.threshold:
            # Update reference
            self.reference_distribution = current_distribution
            return True

        return False


class StreamingAggregator:
    """
    Aggregate explanations over streaming windows.

    Computes aggregate statistics (mean, variance, trends) over
    streaming explanations.

    Parameters
    ----------
    window_size : int
        Aggregation window size.
    aggregation_fn : callable, optional
        Custom aggregation function.

    Examples
    --------
    >>> aggregator = StreamingAggregator(window_size=100)
    >>> for explanation in stream:
    ...     agg_exp = aggregator.aggregate(explanation)
    ...     print(f"Mean importance: {agg_exp['mean_importance']}")
    """

    def __init__(
        self,
        window_size: int = 100,
        aggregation_fn: Optional[Callable] = None
    ):
        self.window_size = window_size
        self.aggregation_fn = aggregation_fn or self._default_aggregation

        self.explanation_window: Deque[ExplanationResult] = deque(maxlen=window_size)

    def _default_aggregation(
        self,
        explanations: List[ExplanationResult]
    ) -> Dict[str, Any]:
        """
        Default aggregation: mean and std of feature importances.

        Parameters
        ----------
        explanations : list of ExplanationResult
            Explanations to aggregate.

        Returns
        -------
        aggregated : dict
            Aggregated statistics.
        """
        importance_list = []

        for exp in explanations:
            if 'feature_importance' in exp.explanation_data:
                importance_list.append(np.array(exp.explanation_data['feature_importance']))

        if not importance_list:
            return {}

        importance_matrix = np.stack(importance_list, axis=0)

        return {
            'mean_importance': np.mean(importance_matrix, axis=0).tolist(),
            'std_importance': np.std(importance_matrix, axis=0).tolist(),
            'n_explanations': len(importance_list)
        }

    def aggregate(
        self,
        explanation: ExplanationResult
    ) -> Dict[str, Any]:
        """
        Add explanation and return aggregated statistics.

        Parameters
        ----------
        explanation : ExplanationResult
            New explanation to add.

        Returns
        -------
        aggregated : dict
            Aggregated statistics over window.
        """
        self.explanation_window.append(explanation)

        # Aggregate
        aggregated = self.aggregation_fn(list(self.explanation_window))

        return aggregated


class RealTimeExplainerPipeline:
    """
    Complete pipeline for real-time explainability.

    Combines incremental explanation, drift detection, and aggregation
    for production streaming systems.

    Parameters
    ----------
    base_explainer : ExplainerBase
        Base explainer.
    window_size : int
        Window size for statistics.
    enable_drift_detection : bool
        Enable drift detection.
    enable_aggregation : bool
        Enable aggregation.

    Examples
    --------
    >>> pipeline = RealTimeExplainerPipeline(
    ...     shap_explainer,
    ...     window_size=100,
    ...     enable_drift_detection=True
    ... )
    >>> for x in data_stream:
    ...     result = pipeline.process(x)
    ...     if result['drift_detected']:
    ...         print("Drift detected - retrain model!")
    """

    def __init__(
        self,
        base_explainer: ExplainerBase,
        window_size: int = 100,
        enable_drift_detection: bool = True,
        enable_aggregation: bool = True,
        latency_threshold_ms: float = 100.0
    ):
        self.base_explainer = base_explainer
        self.window_size = window_size
        self.latency_threshold_ms = latency_threshold_ms

        # Components
        self.incremental_explainer = IncrementalExplainer(base_explainer, window_size)

        self.drift_detector = None
        if enable_drift_detection:
            self.drift_detector = DriftDetector(window_size)

        self.aggregator = None
        if enable_aggregation:
            self.aggregator = StreamingAggregator(window_size)

        # Monitoring
        self.total_processed = 0
        self.total_drift_events = 0
        self.latency_violations = 0

    def process(
        self,
        x: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process streaming instance.

        Parameters
        ----------
        x : ndarray
            Streaming instance.
        **kwargs
            Additional arguments.

        Returns
        -------
        result : dict
            Processing result with explanation and metadata.
        """
        # Generate explanation
        streaming_exp = self.incremental_explainer.explain_incremental(x, **kwargs)

        # Detect drift
        drift_detected = streaming_exp.drift_detected
        if self.drift_detector:
            drift_from_explanations = self.drift_detector.detect(streaming_exp.explanation)
            drift_detected = drift_detected or drift_from_explanations

        if drift_detected:
            self.total_drift_events += 1

        # Aggregate
        aggregated = None
        if self.aggregator:
            aggregated = self.aggregator.aggregate(streaming_exp.explanation)

        # Monitor latency
        if streaming_exp.latency_ms > self.latency_threshold_ms:
            self.latency_violations += 1
            warnings.warn(
                f"Latency violation: {streaming_exp.latency_ms:.2f}ms > "
                f"{self.latency_threshold_ms}ms"
            )

        self.total_processed += 1

        return {
            'explanation': streaming_exp.explanation,
            'drift_detected': drift_detected,
            'latency_ms': streaming_exp.latency_ms,
            'aggregated_stats': aggregated,
            'timestamp': streaming_exp.timestamp,
            'monitoring': {
                'total_processed': self.total_processed,
                'total_drift_events': self.total_drift_events,
                'latency_violations': self.latency_violations
            }
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.

        Returns
        -------
        stats : dict
            Pipeline statistics.
        """
        return {
            'total_processed': self.total_processed,
            'total_drift_events': self.total_drift_events,
            'drift_rate': self.total_drift_events / max(self.total_processed, 1),
            'latency_violations': self.latency_violations,
            'latency_violation_rate': self.latency_violations / max(self.total_processed, 1)
        }


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Real-Time Streaming XAI - Example")
    print("=" * 80)

    # Generate streaming data
    np.random.seed(42)

    # Simulate data stream with concept drift
    def generate_stream(n_samples=500):
        """Generate synthetic streaming data with drift."""
        for i in range(n_samples):
            if i < 250:
                # Distribution 1
                x = np.random.randn(5) + np.array([0, 0, 0, 0, 0])
            else:
                # Distribution 2 (concept drift)
                x = np.random.randn(5) + np.array([2, 2, 0, 0, 0])

            yield x

    # Simple model
    class SimpleModel:
        def predict(self, X):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            return (X[:, 0] + X[:, 1] > 0).astype(int)

        def predict_proba(self, X):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            score = X[:, 0] + X[:, 1]
            pos_prob = 1 / (1 + np.exp(-score))
            return np.column_stack([1 - pos_prob, pos_prob])

    model = SimpleModel()

    # Base explainer
    class SimpleExplainer(ExplainerBase):
        def explain(self, X, **kwargs):
            if X.ndim == 1:
                X = X.reshape(1, -1)

            # Simplified feature importance
            importance = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
            importance += np.random.randn(5) * 0.1

            return ExplanationResult(
                method='simple_shap',
                explanation_data={
                    'feature_importance': importance.tolist()
                },
                metadata={}
            )

    base_explainer = SimpleExplainer(model)

    print("\n1. INCREMENTAL EXPLAINER")
    print("-" * 80)
    inc_explainer = IncrementalExplainer(base_explainer, window_size=50)

    stream = generate_stream(n_samples=100)
    for i, x in enumerate(stream):
        streaming_exp = inc_explainer.explain_incremental(x)

        if i % 25 == 0:
            print(f"Sample {i}:")
            print(f"  Latency: {streaming_exp.latency_ms:.2f}ms")
            print(f"  Drift detected: {streaming_exp.drift_detected}")
            print(f"  Feature importance: {streaming_exp.explanation.explanation_data['feature_importance']}")

    print("\n2. APPROXIMATE EXPLAINER (Low Latency)")
    print("-" * 80)
    approx_explainer = ApproximateExplainer(base_explainer, approximation_level='high')

    x_test = np.random.randn(5)
    start = time.time()
    approx_exp = approx_explainer.explain(x_test)
    latency = (time.time() - start) * 1000

    print(f"Latency: {latency:.2f}ms")
    print(f"Approximation level: {approx_exp.metadata['approximation_level']}")
    print(f"Feature importance: {approx_exp.explanation_data['feature_importance']}")

    print("\n3. DRIFT DETECTOR")
    print("-" * 80)
    drift_detector = DriftDetector(window_size=50, threshold=0.5)

    stream = generate_stream(n_samples=300)
    drift_points = []

    for i, x in enumerate(stream):
        exp = base_explainer.explain(x)
        drift = drift_detector.detect(exp)

        if drift:
            drift_points.append(i)
            print(f"Concept drift detected at sample {i}")

    print(f"Total drift events: {len(drift_points)}")

    print("\n4. STREAMING AGGREGATOR")
    print("-" * 80)
    aggregator = StreamingAggregator(window_size=100)

    stream = generate_stream(n_samples=150)
    for i, x in enumerate(stream):
        exp = base_explainer.explain(x)
        agg_stats = aggregator.aggregate(exp)

        if i % 50 == 49:
            print(f"\nAfter {i+1} samples:")
            print(f"  Mean importance: {agg_stats['mean_importance']}")
            print(f"  Std importance: {agg_stats['std_importance']}")

    print("\n5. REAL-TIME PIPELINE (Full System)")
    print("-" * 80)
    pipeline = RealTimeExplainerPipeline(
        base_explainer,
        window_size=50,
        enable_drift_detection=True,
        enable_aggregation=True,
        latency_threshold_ms=50.0
    )

    stream = generate_stream(n_samples=300)
    for i, x in enumerate(stream):
        result = pipeline.process(x)

        if i % 100 == 99:
            print(f"\nAfter {i+1} samples:")
            print(f"  Drift detected this sample: {result['drift_detected']}")
            print(f"  Latency: {result['latency_ms']:.2f}ms")
            if result['aggregated_stats']:
                print(f"  Aggregated mean importance: {result['aggregated_stats']['mean_importance']}")
            print(f"  Monitoring stats: {result['monitoring']}")

    print("\n6. PIPELINE STATISTICS")
    print("-" * 80)
    stats = pipeline.get_statistics()
    print(f"Total processed: {stats['total_processed']}")
    print(f"Total drift events: {stats['total_drift_events']}")
    print(f"Drift rate: {stats['drift_rate']:.2%}")
    print(f"Latency violations: {stats['latency_violations']}")
    print(f"Latency violation rate: {stats['latency_violation_rate']:.2%}")

    print("\n" + "=" * 80)
    print("Real-time streaming XAI demonstration complete!")
    print("=" * 80)
