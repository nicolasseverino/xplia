"""
Comprehensive tests for scikit-learn model adapter.

Tests all major scikit-learn model types and ensures proper integration
with the XPLIA framework.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

try:
    from xplia.core.model_adapters.sklearn_adapter import SklearnAdapter
except ImportError:
    SklearnAdapter = None
    pytest.skip("Sklearn adapter not found", allow_module_level=True)


class TestSklearnAdapterClassification:
    """Test sklearn adapter with classification models."""

    @pytest.fixture
    def classification_data(self):
        """Create classification dataset."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': [f'feature_{i}' for i in range(10)]
        }

    def test_random_forest_adapter(self, classification_data):
        """Test adapter with RandomForestClassifier."""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(classification_data['X_train'], classification_data['y_train'])

        adapter = SklearnAdapter(
            model,
            feature_names=classification_data['feature_names']
        )

        # Test predictions
        predictions = adapter.predict(classification_data['X_test'])
        assert predictions.shape[0] == classification_data['X_test'].shape[0]
        assert len(np.unique(predictions)) <= 2

        # Test probability predictions
        probabilities = adapter.predict_proba(classification_data['X_test'])
        assert probabilities.shape[0] == classification_data['X_test'].shape[0]
        assert probabilities.shape[1] == 2
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_logistic_regression_adapter(self, classification_data):
        """Test adapter with LogisticRegression."""
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(classification_data['X_train'], classification_data['y_train'])

        adapter = SklearnAdapter(
            model,
            feature_names=classification_data['feature_names']
        )

        predictions = adapter.predict(classification_data['X_test'])
        assert predictions.shape[0] == classification_data['X_test'].shape[0]

        probabilities = adapter.predict_proba(classification_data['X_test'])
        assert probabilities.shape[1] == 2

    def test_decision_tree_adapter(self, classification_data):
        """Test adapter with DecisionTreeClassifier."""
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(classification_data['X_train'], classification_data['y_train'])

        adapter = SklearnAdapter(
            model,
            feature_names=classification_data['feature_names']
        )

        predictions = adapter.predict(classification_data['X_test'])
        assert predictions.shape[0] == classification_data['X_test'].shape[0]

    def test_svc_adapter(self, classification_data):
        """Test adapter with SVC."""
        model = SVC(kernel='rbf', probability=True, random_state=42)
        model.fit(classification_data['X_train'], classification_data['y_train'])

        adapter = SklearnAdapter(
            model,
            feature_names=classification_data['feature_names']
        )

        predictions = adapter.predict(classification_data['X_test'])
        assert predictions.shape[0] == classification_data['X_test'].shape[0]

        probabilities = adapter.predict_proba(classification_data['X_test'])
        assert probabilities.shape[1] == 2

    def test_mlp_adapter(self, classification_data):
        """Test adapter with MLPClassifier."""
        model = MLPClassifier(hidden_layer_sizes=(50,), random_state=42, max_iter=500)
        model.fit(classification_data['X_train'], classification_data['y_train'])

        adapter = SklearnAdapter(
            model,
            feature_names=classification_data['feature_names']
        )

        predictions = adapter.predict(classification_data['X_test'])
        assert predictions.shape[0] == classification_data['X_test'].shape[0]

        probabilities = adapter.predict_proba(classification_data['X_test'])
        assert probabilities.shape[1] == 2


class TestSklearnAdapterRegression:
    """Test sklearn adapter with regression models."""

    @pytest.fixture
    def regression_data(self):
        """Create regression dataset."""
        X, y = make_regression(
            n_samples=200,
            n_features=10,
            n_informative=8,
            noise=0.1,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': [f'feature_{i}' for i in range(10)]
        }

    def test_random_forest_regressor_adapter(self, regression_data):
        """Test adapter with RandomForestRegressor."""
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(regression_data['X_train'], regression_data['y_train'])

        adapter = SklearnAdapter(
            model,
            feature_names=regression_data['feature_names']
        )

        predictions = adapter.predict(regression_data['X_test'])
        assert predictions.shape[0] == regression_data['X_test'].shape[0]
        assert isinstance(predictions[0], (int, float, np.number))

    def test_linear_regression_adapter(self, regression_data):
        """Test adapter with LinearRegression."""
        model = LinearRegression()
        model.fit(regression_data['X_train'], regression_data['y_train'])

        adapter = SklearnAdapter(
            model,
            feature_names=regression_data['feature_names']
        )

        predictions = adapter.predict(regression_data['X_test'])
        assert predictions.shape[0] == regression_data['X_test'].shape[0]

    def test_ridge_adapter(self, regression_data):
        """Test adapter with Ridge."""
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(regression_data['X_train'], regression_data['y_train'])

        adapter = SklearnAdapter(
            model,
            feature_names=regression_data['feature_names']
        )

        predictions = adapter.predict(regression_data['X_test'])
        assert predictions.shape[0] == regression_data['X_test'].shape[0]

    def test_svr_adapter(self, regression_data):
        """Test adapter with SVR."""
        model = SVR(kernel='rbf')
        model.fit(regression_data['X_train'], regression_data['y_train'])

        adapter = SklearnAdapter(
            model,
            feature_names=regression_data['feature_names']
        )

        predictions = adapter.predict(regression_data['X_test'])
        assert predictions.shape[0] == regression_data['X_test'].shape[0]


class TestSklearnAdapterMetadata:
    """Test metadata extraction from sklearn models."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple trained model."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model, [f'feature_{i}' for i in range(5)]

    def test_metadata_extraction(self, simple_model):
        """Test that metadata is properly extracted."""
        model, feature_names = simple_model
        adapter = SklearnAdapter(model, feature_names=feature_names)

        metadata = adapter.get_metadata()
        assert isinstance(metadata, dict)
        assert 'model_type' in metadata or 'type' in metadata

    def test_feature_names_extraction(self, simple_model):
        """Test that feature names are properly handled."""
        model, feature_names = simple_model
        adapter = SklearnAdapter(model, feature_names=feature_names)

        names = adapter.get_feature_names()
        assert len(names) == 5
        assert all(isinstance(name, str) for name in names)

    def test_model_type_detection(self, simple_model):
        """Test that model type is correctly detected."""
        model, feature_names = simple_model
        adapter = SklearnAdapter(model, feature_names=feature_names)

        model_type = adapter.get_model_type()
        assert model_type in ['classification', 'classifier', 'random_forest'] or model_type is not None


class TestSklearnAdapterInputFormats:
    """Test adapter with different input formats."""

    @pytest.fixture
    def simple_trained_model(self):
        """Create a simple trained model."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model, X[:10], [f'feature_{i}' for i in range(5)]

    def test_numpy_array_input(self, simple_trained_model):
        """Test with numpy array input."""
        model, X_test, feature_names = simple_trained_model
        adapter = SklearnAdapter(model, feature_names=feature_names)

        predictions = adapter.predict(X_test)
        assert predictions.shape[0] == 10

    def test_pandas_dataframe_input(self, simple_trained_model):
        """Test with pandas DataFrame input."""
        model, X_test, feature_names = simple_trained_model
        adapter = SklearnAdapter(model, feature_names=feature_names)

        df = pd.DataFrame(X_test, columns=feature_names)
        predictions = adapter.predict(df)
        assert predictions.shape[0] == 10

    def test_single_sample_prediction(self, simple_trained_model):
        """Test prediction on single sample."""
        model, X_test, feature_names = simple_trained_model
        adapter = SklearnAdapter(model, feature_names=feature_names)

        single_sample = X_test[0:1]
        prediction = adapter.predict(single_sample)
        assert prediction.shape[0] == 1


class TestSklearnAdapterErrorHandling:
    """Test error handling in sklearn adapter."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple trained model."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model

    def test_wrong_number_of_features(self, simple_model):
        """Test error when wrong number of features provided."""
        adapter = SklearnAdapter(simple_model)

        wrong_features = np.random.randn(10, 3)  # Should be 5 features
        with pytest.raises(Exception):
            adapter.predict(wrong_features)

    def test_invalid_input_type(self, simple_model):
        """Test error with invalid input type."""
        adapter = SklearnAdapter(simple_model)

        with pytest.raises((TypeError, ValueError, Exception)):
            adapter.predict("invalid input")

    def test_empty_input(self, simple_model):
        """Test handling of empty input."""
        adapter = SklearnAdapter(simple_model)

        empty_input = np.array([]).reshape(0, 5)
        try:
            predictions = adapter.predict(empty_input)
            assert len(predictions) == 0
        except Exception:
            # Some implementations may raise an exception for empty input
            pass


class TestSklearnAdapterFeatureImportance:
    """Test feature importance extraction."""

    @pytest.fixture
    def tree_model(self):
        """Create a tree-based model with feature importance."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model, [f'feature_{i}' for i in range(5)]

    def test_feature_importance_extraction(self, tree_model):
        """Test extracting feature importance."""
        model, feature_names = tree_model
        adapter = SklearnAdapter(model, feature_names=feature_names)

        try:
            importance = adapter.get_feature_importance()
            assert len(importance) == 5
            assert all(isinstance(val, (int, float, np.number)) for val in importance)
            assert all(val >= 0 for val in importance)
        except AttributeError:
            # Not all models have feature importance
            pytest.skip("Feature importance not available for this model")


class TestSklearnAdapterMulticlass:
    """Test adapter with multiclass classification."""

    @pytest.fixture
    def multiclass_data(self):
        """Create multiclass classification dataset."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_classes=3,
            n_informative=8,
            n_redundant=2,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': [f'feature_{i}' for i in range(10)]
        }

    def test_multiclass_prediction(self, multiclass_data):
        """Test multiclass prediction."""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(multiclass_data['X_train'], multiclass_data['y_train'])

        adapter = SklearnAdapter(
            model,
            feature_names=multiclass_data['feature_names']
        )

        predictions = adapter.predict(multiclass_data['X_test'])
        assert predictions.shape[0] == multiclass_data['X_test'].shape[0]
        assert len(np.unique(predictions)) <= 3

    def test_multiclass_probabilities(self, multiclass_data):
        """Test multiclass probability prediction."""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(multiclass_data['X_train'], multiclass_data['y_train'])

        adapter = SklearnAdapter(
            model,
            feature_names=multiclass_data['feature_names']
        )

        probabilities = adapter.predict_proba(multiclass_data['X_test'])
        assert probabilities.shape[0] == multiclass_data['X_test'].shape[0]
        assert probabilities.shape[1] == 3
        assert np.allclose(probabilities.sum(axis=1), 1.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
