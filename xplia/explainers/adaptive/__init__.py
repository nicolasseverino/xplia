"""
XPLIA Adaptive Meta-Explainers
============================

Ce module implémente un système avancé de méta-explication adaptative qui:
1. Sélectionne automatiquement le ou les meilleurs explainers pour un contexte donné
2. Combine intelligemment plusieurs techniques d'explication
3. Optimise la qualité des explications selon les caractéristiques du modèle et des données

Cette approche révolutionnaire permet d'obtenir des explications plus robustes et précises
en tirant parti des forces de chaque technique d'explication et en compensant leurs faiblesses.
"""

from .meta_explainer import AdaptiveMetaExplainer
from .explainer_selector import ExplainerSelector
from .explanation_quality import QualityEstimator
from .fusion_strategies import ExplanationFusionStrategy

__all__ = [
    'AdaptiveMetaExplainer', 
    'ExplainerSelector',
    'QualityEstimator',
    'ExplanationFusionStrategy'
]
