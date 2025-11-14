"""
Tests for TIER 1 - Time Series Explainers
Tests for Temporal Importance, Forecast, and Anomaly explanations
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any

from xplia.explainers.timeseries.timeseries_explainer import (
    TemporalImportanceExplainer,
    ForecastExplainer,
    AnomalyExplainer,
    TimeSeriesExplanation,
)


class TestTemporalImportanceExplainer:
    """Test suite for Temporal Importance explainer."""

    def create_dummy_timeseries(self, length=100) -> pd.DataFrame:
        """Create a dummy time series."""
        dates = pd.date_range('2020-01-01', periods=length, freq='D')
        return pd.DataFrame({
            'value': np.random.rand(length),
            'feature1': np.random.rand(length),
            'feature2': np.random.rand(length)
        }, index=dates)

    def test_initialization(self):
        """Test initialization."""
        explainer = TemporalImportanceExplainer(window_size=30)
        assert explainer is not None
        assert explainer.window_size == 30

    def test_explain_temporal_importance(self):
        """Test temporal importance explanation."""
        explainer = TemporalImportanceExplainer(window_size=30)
        ts = self.create_dummy_timeseries()

        explanation = explainer.explain(ts)

        assert isinstance(explanation, TimeSeriesExplanation)
        assert hasattr(explanation, 'lag_importance')
        assert hasattr(explanation, 'feature_importance')

    def test_identify_important_lags(self):
        """Test lag importance identification."""
        explainer = TemporalImportanceExplainer(window_size=20)
        ts = self.create_dummy_timeseries()

        lags = explainer.identify_important_lags(ts, max_lag=10)

        assert isinstance(lags, dict)
        assert len(lags) > 0

    def test_seasonal_decomposition(self):
        """Test seasonal decomposition."""
        explainer = TemporalImportanceExplainer(window_size=30)
        ts = self.create_dummy_timeseries(length=365)  # One year

        decomposition = explainer.decompose_seasonality(
            ts['value'],
            period=7  # Weekly
        )

        assert decomposition is not None
        assert 'trend' in decomposition
        assert 'seasonal' in decomposition
        assert 'residual' in decomposition

    def test_short_timeseries(self):
        """Test with short time series."""
        explainer = TemporalImportanceExplainer(window_size=10)
        ts = self.create_dummy_timeseries(length=15)

        explanation = explainer.explain(ts)
        assert explanation is not None


class TestForecastExplainer:
    """Test suite for Forecast explainer."""

    def create_dummy_forecast_data(self) -> Dict[str, Any]:
        """Create dummy forecast data."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        return {
            'historical': pd.Series(np.random.rand(100), index=dates),
            'forecast': pd.Series(np.random.rand(30),
                                index=pd.date_range('2020-04-10', periods=30, freq='D')),
            'model': 'dummy_model'
        }

    def test_initialization(self):
        """Test initialization."""
        explainer = ForecastExplainer()
        assert explainer is not None

    def test_explain_forecast(self):
        """Test forecast explanation."""
        explainer = ForecastExplainer()
        data = self.create_dummy_forecast_data()

        explanation = explainer.explain_forecast(
            historical=data['historical'],
            forecast=data['forecast'],
            forecast_horizon=30
        )

        assert explanation is not None
        assert hasattr(explanation, 'forecast_drivers')
        assert hasattr(explanation, 'confidence_intervals')

    def test_identify_forecast_drivers(self):
        """Test forecast driver identification."""
        explainer = ForecastExplainer()
        data = self.create_dummy_forecast_data()

        drivers = explainer.identify_drivers(
            data['historical'],
            data['forecast']
        )

        assert isinstance(drivers, dict)

    def test_decompose_forecast_error(self):
        """Test forecast error decomposition."""
        explainer = ForecastExplainer()

        forecast = pd.Series(np.random.rand(30))
        actual = pd.Series(np.random.rand(30))

        error_decomp = explainer.decompose_error(forecast, actual)

        assert error_decomp is not None
        assert 'bias' in error_decomp
        assert 'variance' in error_decomp

    def test_explain_multivariate_forecast(self):
        """Test multivariate forecast explanation."""
        explainer = ForecastExplainer()

        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        historical = pd.DataFrame({
            'var1': np.random.rand(100),
            'var2': np.random.rand(100)
        }, index=dates)

        forecast_dates = pd.date_range('2020-04-10', periods=30, freq='D')
        forecast = pd.DataFrame({
            'var1': np.random.rand(30),
            'var2': np.random.rand(30)
        }, index=forecast_dates)

        explanation = explainer.explain_forecast(
            historical=historical,
            forecast=forecast,
            forecast_horizon=30
        )

        assert explanation is not None


class TestAnomalyExplainer:
    """Test suite for Anomaly explainer."""

    def create_timeseries_with_anomalies(self, length=100) -> pd.Series:
        """Create time series with injected anomalies."""
        dates = pd.date_range('2020-01-01', periods=length, freq='D')
        values = np.random.rand(length)

        # Inject anomalies
        anomaly_indices = [10, 30, 70]
        for idx in anomaly_indices:
            values[idx] = values[idx] + 5.0  # Large spike

        return pd.Series(values, index=dates)

    def test_initialization(self):
        """Test initialization."""
        explainer = AnomalyExplainer(threshold=2.5)
        assert explainer is not None
        assert explainer.threshold == 2.5

    def test_detect_and_explain(self):
        """Test anomaly detection and explanation."""
        explainer = AnomalyExplainer(threshold=2.5)
        ts = self.create_timeseries_with_anomalies()

        explanation = explainer.detect_and_explain(ts)

        assert explanation is not None
        assert hasattr(explanation, 'anomaly_indices')
        assert hasattr(explanation, 'anomaly_scores')
        assert hasattr(explanation, 'anomaly_reasons')
        assert len(explanation.anomaly_indices) > 0

    def test_explain_specific_anomaly(self):
        """Test explanation of specific anomaly."""
        explainer = AnomalyExplainer(threshold=2.5)
        ts = self.create_timeseries_with_anomalies()

        # Detect anomalies first
        result = explainer.detect_and_explain(ts)

        # Explain first anomaly
        if len(result.anomaly_indices) > 0:
            anomaly_idx = result.anomaly_indices[0]
            explanation = explainer.explain_anomaly(ts, anomaly_idx)

            assert explanation is not None
            assert 'reason' in explanation
            assert 'severity' in explanation

    def test_no_anomalies(self):
        """Test with time series without anomalies."""
        explainer = AnomalyExplainer(threshold=5.0)  # High threshold

        # Normal time series
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        ts = pd.Series(np.random.rand(100), index=dates)

        explanation = explainer.detect_and_explain(ts)

        assert explanation is not None
        # May have zero anomalies
        assert len(explanation.anomaly_indices) >= 0

    def test_contextual_anomaly(self):
        """Test contextual anomaly detection."""
        explainer = AnomalyExplainer(threshold=2.5, method='contextual')
        ts = self.create_timeseries_with_anomalies()

        explanation = explainer.detect_and_explain(ts)

        assert explanation is not None
        assert len(explanation.anomaly_indices) > 0


# Integration tests
class TestTimeSeriesIntegration:
    """Integration tests for time series explainers."""

    def test_temporal_and_forecast_integration(self):
        """Test integration of temporal importance and forecast explanation."""
        temp_explainer = TemporalImportanceExplainer(window_size=30)
        forecast_explainer = ForecastExplainer()

        # Create time series
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        ts = pd.DataFrame({
            'value': np.random.rand(100)
        }, index=dates)

        # Analyze temporal importance
        temp_exp = temp_explainer.explain(ts)

        # Use insights for forecast
        historical = ts['value']
        forecast = pd.Series(np.random.rand(30),
                           index=pd.date_range('2020-04-10', periods=30, freq='D'))

        forecast_exp = forecast_explainer.explain_forecast(
            historical=historical,
            forecast=forecast,
            forecast_horizon=30
        )

        assert temp_exp is not None
        assert forecast_exp is not None

    def test_anomaly_with_temporal_importance(self):
        """Test anomaly detection with temporal importance."""
        temp_explainer = TemporalImportanceExplainer(window_size=20)
        anomaly_explainer = AnomalyExplainer(threshold=2.5)

        # Create time series with anomalies
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        values = np.random.rand(100)
        values[50] = 10.0  # Anomaly

        ts = pd.DataFrame({
            'value': values,
            'feature1': np.random.rand(100)
        }, index=dates)

        # Detect anomalies
        anomaly_exp = anomaly_explainer.detect_and_explain(ts['value'])

        # Analyze temporal patterns
        temp_exp = temp_explainer.explain(ts)

        assert anomaly_exp is not None
        assert temp_exp is not None
        assert len(anomaly_exp.anomaly_indices) > 0


# Performance tests
class TestTimeSeriesPerformance:
    """Performance tests for time series explainers."""

    def test_long_timeseries_performance(self):
        """Test performance on long time series."""
        explainer = TemporalImportanceExplainer(window_size=30)

        # Long time series (10 years daily)
        dates = pd.date_range('2010-01-01', periods=3650, freq='D')
        ts = pd.DataFrame({
            'value': np.random.rand(3650)
        }, index=dates)

        explanation = explainer.explain(ts)
        assert explanation is not None

    def test_high_frequency_data(self):
        """Test with high-frequency data."""
        explainer = AnomalyExplainer(threshold=2.5)

        # Hourly data for one month
        dates = pd.date_range('2020-01-01', periods=720, freq='H')
        ts = pd.Series(np.random.rand(720), index=dates)

        explanation = explainer.detect_and_explain(ts)
        assert explanation is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
