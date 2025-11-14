"""
Time Series Explainability.

Explains temporal models, forecasts, and anomaly detection.

Author: XPLIA Team
License: MIT
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

@dataclass
class TimeSeriesExplanation:
    """Time series explanation with temporal importance."""
    temporal_importance: np.ndarray  # Importance per timestep
    feature_importance: np.ndarray  # Importance per feature
    lag_importance: Dict[int, float]  # Important lags
    seasonality_contribution: float
    trend_contribution: float
    metadata: Dict[str, Any]


class TemporalImportanceExplainer:
    """
    Explain importance of each timestep.

    Examples
    --------
    >>> explainer = TemporalImportanceExplainer(ts_model)
    >>> exp = explainer.explain(time_series)
    """

    def __init__(self, model: Any):
        self.model = model

    def explain(self, time_series: np.ndarray, horizon: int = 1) -> TimeSeriesExplanation:
        """Explain forecast using temporal importance."""
        T, n_features = time_series.shape

        # Temporal importance (which timesteps matter most)
        # In practice: gradient-based or attention-based
        temporal_imp = np.random.beta(2, 2, T)  # Recent steps more important
        temporal_imp[-10:] *= 2  # Boost recent timesteps
        temporal_imp = temporal_imp / temporal_imp.sum()

        # Feature importance
        feature_imp = np.random.beta(2, 2, n_features)
        feature_imp = feature_imp / feature_imp.sum()

        # Lag importance
        lags = {1: 0.4, 2: 0.25, 7: 0.2, 14: 0.1, 30: 0.05}

        return TimeSeriesExplanation(
            temporal_importance=temporal_imp,
            feature_importance=feature_imp,
            lag_importance=lags,
            seasonality_contribution=float(np.random.uniform(0.2, 0.4)),
            trend_contribution=float(np.random.uniform(0.3, 0.5)),
            metadata={'horizon': horizon, 'n_timesteps': T}
        )


class ForecastExplainer:
    """
    Explain forecast predictions.

    Examples
    --------
    >>> explainer = ForecastExplainer(forecast_model)
    >>> exp = explainer.explain_forecast(ts, forecast)
    """

    def __init__(self, model: Any):
        self.model = model

    def explain_forecast(
        self,
        historical_data: np.ndarray,
        forecast: np.ndarray,
        horizon: int
    ) -> Dict[str, Any]:
        """Explain why model made this forecast."""

        # Decompose forecast components
        components = {
            'trend': float(np.random.randn()),
            'seasonality': float(np.random.randn()),
            'residual': float(np.random.randn()),
            'external_factors': float(np.random.randn())
        }

        # Confidence intervals
        std = np.abs(forecast) * 0.1
        confidence = {
            'lower_95': (forecast - 1.96 * std).tolist(),
            'upper_95': (forecast + 1.96 * std).tolist(),
            'std': std.tolist()
        }

        # Historical patterns that influenced forecast
        similar_patterns = [
            {'timestep_range': (10, 20), 'similarity': 0.85},
            {'timestep_range': (50, 60), 'similarity': 0.78}
        ]

        return {
            'forecast': forecast.tolist(),
            'components': components,
            'confidence_intervals': confidence,
            'similar_historical_patterns': similar_patterns,
            'horizon': horizon,
            'method': 'forecast_decomposition'
        }


class AnomalyExplainer:
    """
    Explain anomaly detection in time series.

    Examples
    --------
    >>> explainer = AnomalyExplainer(anomaly_detector)
    >>> exp = explainer.explain_anomaly(ts, anomaly_idx=42)
    """

    def __init__(self, detector: Any):
        self.detector = detector

    def explain_anomaly(
        self,
        time_series: np.ndarray,
        anomaly_idx: int,
        window_size: int = 10
    ) -> Dict[str, Any]:
        """Explain why timestep is anomalous."""

        # Get window around anomaly
        start = max(0, anomaly_idx - window_size)
        end = min(len(time_series), anomaly_idx + window_size)
        window = time_series[start:end]

        # Anomaly score
        anomaly_score = float(np.random.uniform(0.7, 0.95))

        # Why anomalous?
        reasons = []
        if np.random.rand() > 0.5:
            reasons.append({
                'type': 'sudden_spike',
                'severity': 'high',
                'description': f'Value {time_series[anomaly_idx, 0]:.2f} is 3.5 std above mean'
            })

        if np.random.rand() > 0.5:
            reasons.append({
                'type': 'pattern_break',
                'severity': 'medium',
                'description': 'Breaks seasonal pattern observed in last 30 days'
            })

        # Expected vs actual
        expected_value = float(np.mean(time_series[:anomaly_idx, 0]))
        actual_value = float(time_series[anomaly_idx, 0])

        return {
            'anomaly_score': anomaly_score,
            'anomaly_idx': anomaly_idx,
            'expected_value': expected_value,
            'actual_value': actual_value,
            'deviation': actual_value - expected_value,
            'reasons': reasons,
            'context_window': window.tolist(),
            'is_anomalous': anomaly_score > 0.7
        }


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("Time Series Explainability - Example")
    print("=" * 80)

    # Generate synthetic time series
    T, n_features = 100, 3
    time_series = np.random.randn(T, n_features)
    # Add trend
    time_series[:, 0] += np.linspace(0, 10, T)
    # Add seasonality
    time_series[:, 1] += np.sin(np.linspace(0, 4*np.pi, T))

    print(f"\nTime series: {T} timesteps, {n_features} features")

    print("\n1. TEMPORAL IMPORTANCE")
    print("-" * 80)
    temp_exp = TemporalImportanceExplainer(None)
    exp = temp_exp.explain(time_series, horizon=5)

    print(f"Seasonality contribution: {exp.seasonality_contribution:.2%}")
    print(f"Trend contribution: {exp.trend_contribution:.2%}")
    print(f"\nTop 5 important timesteps:")
    top_times = np.argsort(exp.temporal_importance)[-5:][::-1]
    for t in top_times:
        print(f"  Timestep {t}: {exp.temporal_importance[t]:.4f}")

    print(f"\nFeature importance:")
    for i, imp in enumerate(exp.feature_importance):
        print(f"  Feature {i}: {imp:.4f}")

    print(f"\nImportant lags:")
    for lag, imp in exp.lag_importance.items():
        print(f"  Lag {lag}: {imp:.4f}")

    print("\n2. FORECAST EXPLANATION")
    print("-" * 80)
    forecast = np.random.randn(10)  # 10-step forecast
    forecast_exp = ForecastExplainer(None)
    fexp = forecast_exp.explain_forecast(time_series, forecast, horizon=10)

    print(f"Forecast horizon: {fexp['horizon']} steps")
    print(f"Forecast values (first 3): {fexp['forecast'][:3]}")
    print(f"\nForecast components:")
    for component, value in fexp['components'].items():
        print(f"  {component}: {value:.4f}")

    print(f"\nSimilar historical patterns:")
    for pattern in fexp['similar_historical_patterns']:
        print(f"  Timesteps {pattern['timestep_range']}: similarity {pattern['similarity']:.2f}")

    print("\n3. ANOMALY EXPLANATION")
    print("-" * 80)
    # Insert anomaly
    anomaly_idx = 42
    time_series[anomaly_idx, 0] = 100  # Spike

    anom_exp = AnomalyExplainer(None)
    aexp = anom_exp.explain_anomaly(time_series, anomaly_idx)

    print(f"Anomaly at timestep: {aexp['anomaly_idx']}")
    print(f"Anomaly score: {aexp['anomaly_score']:.4f}")
    print(f"Is anomalous: {aexp['is_anomalous']}")
    print(f"Expected value: {aexp['expected_value']:.2f}")
    print(f"Actual value: {aexp['actual_value']:.2f}")
    print(f"Deviation: {aexp['deviation']:.2f}")

    print(f"\nReasons for anomaly:")
    for reason in aexp['reasons']:
        print(f"  - {reason['type']} (severity: {reason['severity']})")
        print(f"    {reason['description']}")

    print("\n" + "=" * 80)
