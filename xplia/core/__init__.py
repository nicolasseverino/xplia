"""
Module central (Core) de XPLIA
=================================

Ce module fournit les composants fondamentaux et l'architecture 
de base pour le système d'explicabilité XPLIA.
"""

from .base import (
    ExplainerBase,
    ExplanationResult,
    FeatureImportance,
    ModelMetadata,
    ExplanationQuality,
    ExplanationFormat,
    ExplainabilityMethod,
    AudienceLevel,
    ModelType,
    ConfigurableMixin as ConfigMixin,  # Alias pour compatibilité
    AuditableMixin as AuditMixin,  # Alias pour compatibilité
)

# Importer ModelAdapterBase depuis le bon module
from .model_adapters.base import ModelAdapterBase

from .factory import ModelFactory, ExplainerFactory, VisualizerFactory, load_model, create_explainer
from .registry import Registry
from .config import ConfigManager

# Initialisation du registre global
registry = Registry()

# Initialisation du gestionnaire de configuration
config_manager = ConfigManager()

# Niveaux d'audience supportés
AUDIENCE_LEVELS = {
    'novice': AudienceLevel.NOVICE,
    'basic': AudienceLevel.BASIC,
    'intermediate': AudienceLevel.INTERMEDIATE,
    'advanced': AudienceLevel.ADVANCED,
    'expert': AudienceLevel.EXPERT,
    # Aliases
    'public': AudienceLevel.PUBLIC,
    'business': AudienceLevel.BUSINESS,
    'technical': AudienceLevel.TECHNICAL
}

# Méthodes d'explicabilité supportées
EXPLAINABILITY_METHODS = {
    'shap': ExplainabilityMethod.SHAP,
    'lime': ExplainabilityMethod.LIME,
    'counterfactual': ExplainabilityMethod.COUNTERFACTUAL,
    'unified': ExplainabilityMethod.UNIFIED,
    'gradient': ExplainabilityMethod.GRADIENT,
    'integrated_gradients': ExplainabilityMethod.INTEGRATED_GRADIENTS
}

__all__ = [
    # Classes de base
    'ExplainerBase',
    'ExplanationResult',
    'FeatureImportance',
    'ModelMetadata',
    'ExplanationQuality',
    'ExplanationFormat',
    'ModelAdapterBase',
    
    # Enums
    'ExplainabilityMethod',
    'AudienceLevel',
    'ModelType',
    
    # Mixins
    'ConfigMixin',
    'AuditMixin',
    
    # Factories
    'ModelFactory',
    'ExplainerFactory',
    'VisualizerFactory',
    
    # Factory functions
    'load_model',
    'create_explainer',
    
    # Utilitaires
    'Registry',
    'ConfigManager',
    
    # Constantes
    'AUDIENCE_LEVELS',
    'EXPLAINABILITY_METHODS',
    
    # Instances
    'registry',
    'config_manager'
]
