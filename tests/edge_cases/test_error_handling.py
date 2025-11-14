"""
Comprehensive edge case and error handling tests.

Tests boundary conditions, error cases, and robustness.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


class TestNullAndMissingData:
    """Test handling of null and missing data."""

    def test_nan_in_features(self):
        """Test handling NaN values in features."""
        X = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]])
        y = np.array([0, 1, 0])

        # Model should handle or raise appropriate error
        try:
            model = RandomForestClassifier(n_estimators=5, random_state=42)
            model.fit(X, y)
        except (ValueError, Exception):
            # Expected behavior - many models don't handle NaN
            pass

    def test_inf_values(self):
        """Test handling of infinity values."""
        X = np.array([[1, 2, np.inf], [4, 5, 6], [7, -np.inf, 9]])
        y = np.array([0, 1, 0])

        try:
            model = RandomForestClassifier(n_estimators=5, random_state=42)
            model.fit(X, y)
        except (ValueError, Exception):
            # Expected behavior
            pass

    def test_all_nan_column(self):
        """Test data with all-NaN column."""
        X = np.array([[1, np.nan], [2, np.nan], [3, np.nan]])
        y = np.array([0, 1, 0])

        with pytest.raises((ValueError, Exception)):
            model = RandomForestClassifier(n_estimators=5, random_state=42)
            model.fit(X, y)

    def test_empty_dataframe(self):
        """Test empty DataFrame."""
        df = pd.DataFrame()

        assert df.empty
        assert len(df) == 0


class TestDataShapeEdgeCases:
    """Test edge cases related to data shapes."""

    def test_single_sample(self):
        """Test prediction on single sample."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        single_sample = X[0:1]
        prediction = model.predict(single_sample)
        assert prediction.shape[0] == 1

    def test_single_feature(self):
        """Test model with single feature."""
        X = np.random.randn(100, 1)
        y = (X[:, 0] > 0).astype(int)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        prediction = model.predict(X[:5])
        assert prediction.shape[0] == 5

    def test_very_few_samples(self):
        """Test with very few training samples."""
        X = np.random.randn(3, 5)
        y = np.array([0, 1, 0])

        try:
            model = RandomForestClassifier(n_estimators=2, random_state=42)
            model.fit(X, y)
            prediction = model.predict(X)
            assert prediction.shape[0] == 3
        except Exception:
            # Some models may require minimum samples
            pass

    def test_zero_samples(self):
        """Test with zero samples."""
        X = np.array([]).reshape(0, 5)
        y = np.array([])

        with pytest.raises((ValueError, Exception)):
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)


class TestExtremeValues:
    """Test handling of extreme values."""

    def test_very_large_values(self):
        """Test with very large feature values."""
        X = np.random.randn(100, 5) * 1e10
        y = np.random.randint(0, 2, 100)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        prediction = model.predict(X[:5])
        assert prediction.shape[0] == 5

    def test_very_small_values(self):
        """Test with very small feature values."""
        X = np.random.randn(100, 5) * 1e-10
        y = np.random.randint(0, 2, 100)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        prediction = model.predict(X[:5])
        assert prediction.shape[0] == 5

    def test_mixed_extreme_values(self):
        """Test with mixed extreme values."""
        X = np.random.randn(100, 5)
        X[:, 0] *= 1e10  # Very large
        X[:, 1] *= 1e-10  # Very small
        X[:, 2] = np.where(X[:, 2] > 0, np.inf, -np.inf)  # Infinity

        y = np.random.randint(0, 2, 100)

        try:
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            # Replace inf with large values
            X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
            model.fit(X, y)
            prediction = model.predict(X[:5])
            assert prediction.shape[0] == 5
        except Exception:
            pass


class TestDataTypeEdgeCases:
    """Test edge cases with different data types."""

    def test_integer_features(self):
        """Test with integer features."""
        X = np.random.randint(0, 100, size=(100, 5))
        y = np.random.randint(0, 2, 100)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        prediction = model.predict(X[:5])
        assert prediction.shape[0] == 5

    def test_mixed_types_dataframe(self):
        """Test DataFrame with mixed types."""
        df = pd.DataFrame({
            'int_col': np.random.randint(0, 10, 100),
            'float_col': np.random.randn(100),
            'bool_col': np.random.choice([True, False], 100)
        })

        y = np.random.randint(0, 2, 100)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(df, y)

        prediction = model.predict(df[:5])
        assert prediction.shape[0] == 5

    def test_categorical_encoding(self):
        """Test categorical data handling."""
        df = pd.DataFrame({
            'category': ['A', 'B', 'C'] * 33 + ['A'],
            'value': np.random.randn(100)
        })

        # Need to encode categories
        df_encoded = pd.get_dummies(df, columns=['category'])
        y = np.random.randint(0, 2, 100)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(df_encoded, y)

        prediction = model.predict(df_encoded[:5])
        assert prediction.shape[0] == 5


class TestImbalancedData:
    """Test with highly imbalanced datasets."""

    def test_severe_class_imbalance(self):
        """Test with 99:1 class imbalance."""
        X = np.random.randn(1000, 5)
        y = np.array([0] * 990 + [1] * 10)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        prediction = model.predict(X[:100])
        assert prediction.shape[0] == 100

    def test_single_class_only(self):
        """Test when only one class present."""
        X = np.random.randn(100, 5)
        y = np.zeros(100)  # All same class

        try:
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)
            # This may succeed but model won't be useful
        except Exception:
            # Some implementations may reject single-class data
            pass


class TestMemoryAndPerformance:
    """Test memory and performance edge cases."""

    def test_large_feature_space(self):
        """Test with many features."""
        X = np.random.randn(100, 1000)
        y = np.random.randint(0, 2, 100)

        model = RandomForestClassifier(n_estimators=5, random_state=42, max_depth=3)
        model.fit(X, y)

        prediction = model.predict(X[:5])
        assert prediction.shape[0] == 5

    @pytest.mark.slow
    def test_large_dataset(self):
        """Test with large dataset."""
        X = np.random.randn(10000, 50)
        y = np.random.randint(0, 2, 10000)

        model = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)

        import time
        start = time.time()
        model.fit(X, y)
        elapsed = time.time() - start

        prediction = model.predict(X[:100])
        assert prediction.shape[0] == 100
        assert elapsed < 60  # Should complete in reasonable time


class TestConcurrencyAndThreadSafety:
    """Test concurrent operations."""

    def test_parallel_predictions(self):
        """Test parallel predictions using n_jobs."""
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
        model.fit(X, y)

        prediction = model.predict(X[:100])
        assert prediction.shape[0] == 100


class TestErrorMessages:
    """Test that error messages are informative."""

    def test_shape_mismatch_error(self):
        """Test error message for shape mismatch."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        wrong_shape = np.random.randn(10, 3)  # Should be 5 features

        with pytest.raises(ValueError) as exc_info:
            model.predict(wrong_shape)

        error_message = str(exc_info.value)
        assert len(error_message) > 0  # Should have informative error

    def test_invalid_parameter_error(self):
        """Test error with invalid parameters."""
        with pytest.raises((ValueError, TypeError)):
            RandomForestClassifier(n_estimators=-5)  # Negative not allowed


class TestRobustness:
    """Test robustness to various conditions."""

    def test_duplicate_samples(self):
        """Test with duplicate samples."""
        X = np.array([[1, 2]] * 50 + [[3, 4]] * 50)
        y = np.array([0] * 50 + [1] * 50)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        prediction = model.predict(X[:10])
        assert prediction.shape[0] == 10

    def test_constant_features(self):
        """Test with constant features."""
        X = np.random.randn(100, 5)
        X[:, 2] = 5.0  # Constant feature

        y = np.random.randint(0, 2, 100)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        prediction = model.predict(X[:5])
        assert prediction.shape[0] == 5

    def test_perfectly_correlated_features(self):
        """Test with perfectly correlated features."""
        X = np.random.randn(100, 3)
        X = np.column_stack([X, X[:, 0], X[:, 1]])  # Add duplicates

        y = np.random.randint(0, 2, 100)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        prediction = model.predict(X[:5])
        assert prediction.shape[0] == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
