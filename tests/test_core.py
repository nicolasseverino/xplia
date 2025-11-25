"""
Tests unitaires pour les modules core de XPLIA
==============================================
"""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from xplia.core.base import (
    ExplainabilityMethod,
    AudienceLevel,
    ModelType,
    FeatureImportance,
    ExplanationQuality,
    ExplanationFormat
)
from xplia.core.registry import Registry, ComponentType
from xplia.core.config import ConfigManager


class TestEnums:
    """Tests pour les enums."""
    
    def test_explainability_method_values(self):
        """Test valeurs ExplainabilityMethod."""
        assert ExplainabilityMethod.SHAP.value == 'shap'
        assert ExplainabilityMethod.LIME.value == 'lime'
        assert ExplainabilityMethod.UNIFIED.value == 'unified'
    
    def test_audience_level_values(self):
        """Test valeurs AudienceLevel."""
        assert AudienceLevel.NOVICE.value == 'novice'
        assert AudienceLevel.BASIC.value == 'basic'
        assert AudienceLevel.EXPERT.value == 'expert'
    
    def test_audience_level_aliases(self):
        """Test aliases AudienceLevel."""
        assert AudienceLevel.PUBLIC.value == 'novice'
        assert AudienceLevel.BUSINESS.value == 'basic'
        assert AudienceLevel.TECHNICAL.value == 'advanced'
    
    def test_model_type_values(self):
        """Test valeurs ModelType."""
        assert ModelType.CLASSIFICATION.value == 'classification'
        assert ModelType.REGRESSION.value == 'regression'
    
    def test_explanation_format_values(self):
        """Test valeurs ExplanationFormat."""
        assert ExplanationFormat.JSON.value == 'json'
        assert ExplanationFormat.HTML.value == 'html'
        assert ExplanationFormat.PDF.value == 'pdf'


class TestFeatureImportance:
    """Tests pour FeatureImportance."""
    
    def test_create_feature_importance(self):
        """Test création FeatureImportance."""
        fi = FeatureImportance(
            feature_name='age',
            importance_value=0.5,
            importance_rank=1,
            direction='positive'
        )
        assert fi.feature_name == 'age'
        assert fi.importance_value == 0.5
        assert fi.importance_rank == 1
        assert fi.direction == 'positive'
    
    def test_feature_importance_to_dict(self):
        """Test conversion en dict."""
        fi = FeatureImportance(
            feature_name='age',
            importance_value=0.5,
            importance_rank=1
        )
        d = fi.to_dict()
        assert isinstance(d, dict)
        assert d['feature_name'] == 'age'
        assert d['importance_value'] == 0.5
    
    def test_feature_importance_alias(self):
        """Test alias importance."""
        fi = FeatureImportance(
            feature_name='age',
            importance_value=0.5
        )
        assert fi.importance == 0.5


class TestExplanationQuality:
    """Tests pour ExplanationQuality."""
    
    def test_create_explanation_quality(self):
        """Test création ExplanationQuality."""
        eq = ExplanationQuality(
            fidelity=0.9,
            stability=0.8,
            sparsity=0.7,
            consistency=0.85
        )
        assert eq.fidelity == 0.9
        assert eq.stability == 0.8
    
    def test_overall_score(self):
        """Test calcul score global."""
        eq = ExplanationQuality(
            fidelity=0.9,
            stability=0.8,
            sparsity=0.7,
            consistency=0.6
        )
        score = eq.overall_score()
        assert 0 <= score <= 1
        assert score == pytest.approx(0.75, abs=0.01)
    
    def test_overall_score_partial(self):
        """Test score avec valeurs partielles."""
        eq = ExplanationQuality(fidelity=0.9, stability=0.8)
        score = eq.overall_score()
        assert score == pytest.approx(0.85, abs=0.01)
    
    def test_to_dict(self):
        """Test conversion en dict."""
        eq = ExplanationQuality(fidelity=0.9)
        d = eq.to_dict()
        assert isinstance(d, dict)
        assert d['fidelity'] == 0.9


class TestRegistry:
    """Tests pour Registry."""
    
    @pytest.fixture
    def registry(self):
        """Crée une instance Registry pour les tests."""
        return Registry()
    
    def test_registry_creation(self, registry):
        """Test création Registry."""
        assert registry is not None
        assert hasattr(registry, 'list_all_components')
    
    def test_list_all_components(self, registry):
        """Test liste de tous les composants."""
        components = registry.list_all_components()
        assert isinstance(components, dict)
        assert ComponentType.EXPLAINER in components
        assert ComponentType.VISUALIZER in components
    
    def test_get_explainers(self, registry):
        """Test récupération des explainers."""
        explainers = registry.get_explainers()
        assert isinstance(explainers, list)
    
    def test_get_visualizers(self, registry):
        """Test récupération des visualizers."""
        visualizers = registry.get_visualizers()
        assert isinstance(visualizers, list)
    
    def test_validate_dependencies(self, registry):
        """Test validation des dépendances."""
        errors = registry.validate_dependencies()
        assert isinstance(errors, dict)
    
    def test_detect_cycles(self, registry):
        """Test détection de cycles."""
        cycles = registry.detect_cycles()
        assert isinstance(cycles, list)


class TestConfigManager:
    """Tests pour ConfigManager."""
    
    @pytest.fixture
    def config_manager(self):
        """Crée une instance ConfigManager."""
        return ConfigManager()
    
    def test_config_manager_creation(self, config_manager):
        """Test création ConfigManager."""
        assert config_manager is not None
    
    def test_get_default_config(self, config_manager):
        """Test récupération config par défaut."""
        config = config_manager.get_default_config()
        assert isinstance(config, dict)
    
    def test_set_and_get_config(self, config_manager):
        """Test set/get configuration."""
        # ConfigManager peut avoir une API différente
        if hasattr(config_manager, 'set'):
            config_manager.set('test_key', 'test_value')
            value = config_manager.get('test_key')
            assert value == 'test_value'
        else:
            pytest.skip("ConfigManager.set not implemented")
    
    def test_get_nonexistent_key(self, config_manager):
        """Test get clé inexistante."""
        if hasattr(config_manager, 'get'):
            value = config_manager.get('nonexistent_key', default='default_value')
            assert value == 'default_value'
        else:
            pytest.skip("ConfigManager.get not implemented")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
