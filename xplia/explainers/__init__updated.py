"""
Module des Explainers de XPLIA
=================================

Ce module fournit des implémentations de diverses méthodes d'explicabilité
de modèles d'IA, avec une interface unifiée et cohérente.

Il inclut également des modules avancés pour l'évaluation de la fiabilité
des explications, la détection de fairwashing, et la génération de rapports
de confiance.
"""

# Explainers de base
from .shap_explainer import ShapExplainer
from .lime_explainer import LimeExplainer
from .unified_explainer import UnifiedExplainer
from .counterfactual_explainer import CounterfactualExplainer
from .feature_importance_explainer import FeatureImportanceExplainer
from .pdp_explainer import PartialDependenceExplainer
from .attention_explainer import AttentionExplainer
from .gradient_explainer import GradientExplainer
from .anchor_explainer import AnchorExplainer

# Modules d'évaluation de confiance
from .trust.uncertainty import UncertaintyQuantifier, UncertaintyMetrics, UncertaintyType
from .trust.fairwashing import FairwashingDetector, FairwashingAudit, FairwashingType
from .trust.confidence_report import ConfidenceReport, TrustScore, TrustLevel

# Modules de calibration et d'adaptation à l'audience
from .calibration.calibrator import ExplanationCalibrator
from .calibration.metrics import CalibrationMetrics
from .calibration.audience_profiles import AudienceProfile, AudienceAdapter

__all__ = [
    # Explainers de base
    "ShapExplainer",
    "LimeExplainer",
    "UnifiedExplainer",
    "CounterfactualExplainer",
    "FeatureImportanceExplainer",
    "PartialDependenceExplainer",
    "AttentionExplainer",
    "GradientExplainer",
    "AnchorExplainer",
    
    # Modules d'évaluation de confiance
    "UncertaintyQuantifier",
    "UncertaintyMetrics",
    "UncertaintyType",
    "FairwashingDetector",
    "FairwashingAudit",
    "FairwashingType",
    "ConfidenceReport",
    "TrustScore",
    "TrustLevel",
    
    # Modules de calibration et d'adaptation à l'audience
    "ExplanationCalibrator",
    "CalibrationMetrics",
    "AudienceProfile",
    "AudienceAdapter",
]
