"""
Module de visualisations pour XPLIA
==================================

Ce module fournit des composants de visualisation adaptés à chaque type d'explication
générée par les explainers de XPLIA, avec support pour différents niveaux d'audience.
"""

from .base import VisualizerBase
from .registry import register_visualizer
from .feature_importance_viz import FeatureImportanceVisualizer
from .attention_viz import AttentionVisualizer
from .gradient_viz import GradientVisualizer
from .shap_viz import ShapVisualizer
from .lime_viz import LimeVisualizer
from .pdp_viz import PDPVisualizer
from .counterfactual_viz import CounterfactualVisualizer
from .anchor_viz import AnchorVisualizer

__all__ = [
    "VisualizerBase",
    "register_visualizer",
    "FeatureImportanceVisualizer",
    "AttentionVisualizer",
    "GradientVisualizer",
    "ShapVisualizer",
    "LimeVisualizer",
    "PDPVisualizer",
    "CounterfactualVisualizer",
    "AnchorVisualizer",
]
