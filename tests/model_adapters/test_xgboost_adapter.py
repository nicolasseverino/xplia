"""
Comprehensive tests for XGBoost model adapter.
"""

import pytest
import numpy as np
import pandas as pd

pytest.importorskip("xgboost")

import xgboost as xgb
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

try:
    from xplia.core.model_adapters.xgboost_adapter import XGBoostAdapter
except ImportError:
    XGBoostAdapter = None
    pytest.skip("XGBoost adapter not found", allow_module_level=True)


class TestXGBoostAdapterClassification:
    """Test XGBoost adapter with classification models."""

    @pytest.fixture
    def classification_model_and_data(self):
        """Create and train XGBoost classifier."""
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = xgb.XGBClassifier(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        return model, X_test, [f'feature_{i}' for i in range(10)]

    def test_xgboost_prediction(self, classification_model_and_data):
        """Test XGBoost prediction."""
        model, X_test, feature_names = classification_model_and_data

        adapter = XGBoostAdapter(model, feature_names=feature_names)
        predictions = adapter.predict(X_test)

        assert predictions.shape[0] == X_test.shape[0]
        assert len(np.unique(predictions)) <= 2

    def test_xgboost_predict_proba(self, classification_model_and_data):
        """Test XGBoost probability prediction."""
        model, X_test, feature_names = classification_model_and_data

        adapter = XGBoostAdapter(model, feature_names=feature_names)
        probabilities = adapter.predict_proba(X_test)

        assert probabilities.shape[0] == X_test.shape[0]
        assert probabilities.shape[1] == 2
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_xgboost_feature_importance(self, classification_model_and_data):
        """Test feature importance extraction."""
        model, X_test, feature_names = classification_model_and_data

        adapter = XGBoostAdapter(model, feature_names=feature_names)
        importance = adapter.get_feature_importance()

        assert len(importance) == 10
        assert all(val >= 0 for val in importance)

    def test_xgboost_pandas_input(self, classification_model_and_data):
        """Test with pandas DataFrame input."""
        model, X_test, feature_names = classification_model_and_data

        adapter = XGBoostAdapter(model, feature_names=feature_names)
        df = pd.DataFrame(X_test, columns=feature_names)
        predictions = adapter.predict(df)

        assert predictions.shape[0] == X_test.shape[0]


class TestXGBoostAdapterRegression:
    """Test XGBoost adapter with regression models."""

    @pytest.fixture
    def regression_model_and_data(self):
        """Create and train XGBoost regressor."""
        X, y = make_regression(n_samples=500, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = xgb.XGBRegressor(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        return model, X_test, [f'feature_{i}' for i in range(10)]

    def test_xgboost_regression_prediction(self, regression_model_and_data):
        """Test XGBoost regression prediction."""
        model, X_test, feature_names = regression_model_and_data

        adapter = XGBoostAdapter(model, feature_names=feature_names)
        predictions = adapter.predict(X_test)

        assert predictions.shape[0] == X_test.shape[0]
        assert isinstance(predictions[0], (int, float, np.number))


class TestXGBoostAdapterMetadata:
    """Test metadata extraction."""

    def test_metadata_extraction(self, classification_model_and_data):
        """Test metadata extraction."""
        model, X_test, feature_names = classification_model_and_data

        adapter = XGBoostAdapter(model, feature_names=feature_names)
        metadata = adapter.get_metadata()

        assert isinstance(metadata, dict)

    def test_feature_names_extraction(self, classification_model_and_data):
        """Test feature names extraction."""
        model, X_test, feature_names = classification_model_and_data

        adapter = XGBoostAdapter(model, feature_names=feature_names)
        names = adapter.get_feature_names()

        assert len(names) == 10


class TestXGBoostAdapterMulticlass:
    """Test XGBoost adapter with multiclass classification."""

    @pytest.fixture
    def multiclass_model_and_data(self):
        """Create multiclass XGBoost model."""
        X, y = make_classification(
            n_samples=500,
            n_features=10,
            n_classes=3,
            n_informative=8,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = xgb.XGBClassifier(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        return model, X_test, [f'feature_{i}' for i in range(10)]

    def test_multiclass_prediction(self, multiclass_model_and_data):
        """Test multiclass prediction."""
        model, X_test, feature_names = multiclass_model_and_data

        adapter = XGBoostAdapter(model, feature_names=feature_names)
        predictions = adapter.predict(X_test)

        assert predictions.shape[0] == X_test.shape[0]
        assert len(np.unique(predictions)) <= 3

    def test_multiclass_probabilities(self, multiclass_model_and_data):
        """Test multiclass probability prediction."""
        model, X_test, feature_names = multiclass_model_and_data

        adapter = XGBoostAdapter(model, feature_names=feature_names)
        probabilities = adapter.predict_proba(X_test)

        assert probabilities.shape[0] == X_test.shape[0]
        assert probabilities.shape[1] == 3
        assert np.allclose(probabilities.sum(axis=1), 1.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
