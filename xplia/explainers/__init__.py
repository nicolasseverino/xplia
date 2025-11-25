"""
Module des Explainers de LUMIA
=================================

Ce module fournit des implémentations de diverses méthodes d'explicabilité
de modèles d'IA, avec une interface unifiée et cohérente.
"""

# Import conditionnel pour éviter les erreurs de syntaxe
try:
    from .shap_explainer import ShapExplainer
except (ImportError, SyntaxError) as e:
    import warnings
    warnings.warn(f"Could not import ShapExplainer: {e}")
    ShapExplainer = None

try:
    from .lime_explainer import LimeExplainer
except (ImportError, SyntaxError) as e:
    import warnings
    warnings.warn(f"Could not import LimeExplainer: {e}")
    LimeExplainer = None

# Import conditionnel des autres explainers
try:
    from .unified_explainer import UnifiedExplainer
except (ImportError, SyntaxError):
    UnifiedExplainer = None

try:
    from .counterfactual_explainer import CounterfactualExplainer
except (ImportError, SyntaxError):
    CounterfactualExplainer = None

try:
    from .feature_importance_explainer import FeatureImportanceExplainer
except (ImportError, SyntaxError):
    FeatureImportanceExplainer = None

try:
    from .partial_dependence_explainer import PartialDependenceExplainer
except (ImportError, SyntaxError):
    PartialDependenceExplainer = None

try:
    from .attention_explainer import AttentionExplainer
except (ImportError, SyntaxError):
    AttentionExplainer = None

try:
    from .gradient_explainer import GradientExplainer
except (ImportError, SyntaxError):
    GradientExplainer = None

try:
    from .anchor_explainer import AnchorExplainer
except (ImportError, SyntaxError):
    AnchorExplainer = None

__all__ = [
    "ShapExplainer",
    "LimeExplainer",
    "UnifiedExplainer",
    "CounterfactualExplainer",
    "FeatureImportanceExplainer",
    "PartialDependenceExplainer",
    "AttentionExplainer",
    "GradientExplainer",
    "AnchorExplainer",
]
