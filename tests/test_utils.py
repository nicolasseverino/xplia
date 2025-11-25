"""
Tests unitaires pour les utilitaires de XPLIA
=============================================
"""

import pytest
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier

from xplia.utils.performance import Timer, MemoryTracker, measure_performance
from xplia.utils.validation import validate_input, validate_model, validate_feature_names


class TestTimer:
    """Tests pour Timer."""
    
    def test_timer_basic(self):
        """Test utilisation basique du Timer."""
        with Timer("Test operation", verbose=False) as timer:
            time.sleep(0.1)
        
        assert timer.elapsed is not None
        assert timer.elapsed >= 0.1
    
    def test_timer_get_elapsed(self):
        """Test récupération temps écoulé."""
        timer = Timer("Test", verbose=False)
        timer.__enter__()
        time.sleep(0.05)
        timer.__exit__()
        
        elapsed = timer.get_elapsed()
        assert elapsed is not None
        assert elapsed >= 0.05
    
    def test_timer_verbose(self, capsys):
        """Test mode verbose."""
        with Timer("Verbose test", verbose=True):
            time.sleep(0.01)
        
        # Note: Le test du verbose nécessiterait de capturer les logs


class TestMemoryTracker:
    """Tests pour MemoryTracker."""
    
    def test_memory_tracker_basic(self):
        """Test utilisation basique du MemoryTracker."""
        with MemoryTracker("Test operation", verbose=False) as tracker:
            # Allouer de la mémoire
            data = [i for i in range(10000)]
        
        assert tracker.memory_used is not None
    
    def test_memory_tracker_get_memory(self):
        """Test récupération mémoire utilisée."""
        tracker = MemoryTracker("Test", verbose=False)
        tracker.__enter__()
        data = [i for i in range(10000)]
        tracker.__exit__()
        
        memory = tracker.get_memory_used()
        assert memory is not None


class TestMeasurePerformance:
    """Tests pour measure_performance."""
    
    def test_measure_performance_basic(self):
        """Test mesure de performance basique."""
        with measure_performance("Test operation") as metrics:
            time.sleep(0.05)
        
        assert 'elapsed_time' in metrics
        assert metrics['elapsed_time'] >= 0.05
    
    def test_measure_performance_with_memory(self):
        """Test mesure avec tracking mémoire."""
        with measure_performance("Test with memory", track_memory=True) as metrics:
            data = [i for i in range(10000)]
        
        assert 'elapsed_time' in metrics
        assert 'memory_used' in metrics
    
    def test_measure_performance_without_memory(self):
        """Test mesure sans tracking mémoire."""
        with measure_performance("Test without memory", track_memory=False) as metrics:
            time.sleep(0.01)
        
        assert 'elapsed_time' in metrics
        assert 'memory_used' not in metrics


class TestValidateInput:
    """Tests pour validate_input."""
    
    def test_validate_numpy_array(self):
        """Test validation array numpy."""
        X = np.array([[1, 2], [3, 4]])
        result = validate_input(X)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
    
    def test_validate_pandas_dataframe(self):
        """Test validation DataFrame pandas."""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        result = validate_input(df)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
    
    def test_validate_list(self):
        """Test validation liste."""
        X = [[1, 2], [3, 4]]
        result = validate_input(X)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
    
    def test_validate_1d_array(self):
        """Test validation array 1D."""
        X = np.array([1, 2, 3, 4])
        result = validate_input(X, allow_1d=False)
        assert result.shape == (1, 4)
    
    def test_validate_1d_allowed(self):
        """Test validation array 1D autorisé."""
        X = np.array([1, 2, 3, 4])
        result = validate_input(X, allow_1d=True)
        assert result.shape == (4,)
    
    def test_validate_expected_shape(self):
        """Test validation avec forme attendue."""
        X = np.array([[1, 2], [3, 4]])
        result = validate_input(X, expected_shape=(2, 2))
        assert result.shape == (2, 2)
    
    def test_validate_invalid_type(self):
        """Test validation type invalide."""
        with pytest.raises(ValueError):
            validate_input("not an array")
    
    def test_validate_with_nan(self):
        """Test validation avec NaN."""
        X = np.array([[1, 2], [3, np.nan]])
        # Should not raise, just warn
        result = validate_input(X)
        assert result.shape == (2, 2)


class TestValidateModel:
    """Tests pour validate_model."""
    
    def test_validate_sklearn_model(self):
        """Test validation modèle sklearn."""
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        result = validate_model(model, required_methods=['fit', 'predict'])
        assert result is True
    
    def test_validate_default_methods(self):
        """Test validation avec méthodes par défaut."""
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        result = validate_model(model)
        assert result is True
    
    def test_validate_missing_method(self):
        """Test validation avec méthode manquante."""
        class FakeModel:
            def predict(self, X):
                return X
        
        model = FakeModel()
        with pytest.raises(ValueError) as exc_info:
            validate_model(model, required_methods=['predict', 'fit'])
        assert 'fit' in str(exc_info.value)
    
    def test_validate_custom_methods(self):
        """Test validation avec méthodes personnalisées."""
        class CustomModel:
            def predict(self, X):
                return X
            def predict_proba(self, X):
                return X
        
        model = CustomModel()
        result = validate_model(model, required_methods=['predict', 'predict_proba'])
        assert result is True


class TestValidateFeatureNames:
    """Tests pour validate_feature_names."""
    
    def test_validate_feature_names_list(self):
        """Test validation liste de noms."""
        names = ['feature1', 'feature2', 'feature3']
        result = validate_feature_names(names, n_features=3)
        assert result == names
    
    def test_validate_feature_names_array(self):
        """Test validation array de noms."""
        names = np.array(['f1', 'f2'])
        result = validate_feature_names(names, n_features=2)
        assert isinstance(result, list)
        assert len(result) == 2
    
    def test_validate_feature_names_none(self):
        """Test validation avec None."""
        result = validate_feature_names(None, n_features=3)
        assert len(result) == 3
        assert result[0] == 'feature_0'
        assert result[1] == 'feature_1'
    
    def test_validate_feature_names_mismatch(self):
        """Test validation avec nombre incorrect."""
        names = ['f1', 'f2']
        with pytest.raises(ValueError) as exc_info:
            validate_feature_names(names, n_features=3)
        assert 'différent' in str(exc_info.value).lower()


class TestUtilsIntegration:
    """Tests d'intégration des utilitaires."""
    
    def test_workflow_complete(self):
        """Test workflow complet avec tous les utils."""
        # 1. Validation des données
        X = np.array([[1, 2], [3, 4], [5, 6]])
        X_valid = validate_input(X, expected_shape=(3, 2))
        
        # 2. Validation du modèle
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        validate_model(model, required_methods=['fit', 'predict'])
        
        # 3. Validation des feature names
        feature_names = validate_feature_names(['f1', 'f2'], n_features=2)
        
        # 4. Mesure de performance
        with measure_performance("Training", track_memory=True) as metrics:
            y = np.array([0, 1, 0])
            model.fit(X_valid, y)
        
        assert 'elapsed_time' in metrics
        assert 'memory_used' in metrics
        assert metrics['elapsed_time'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
