"""
XPLIA Évaluation de Confiance Avancée
==================================

Ce module implémente des mécanismes avancés pour quantifier l'incertitude
des explications et détecter les explications potentiellement trompeuses.

Il comprend deux composants principaux:
1. Quantification d'incertitude: Métriques pour évaluer la fiabilité des explications
2. Détection de fairwashing: Identification des explications manipulées pour masquer des biais
"""

from .uncertainty import UncertaintyQuantifier, UncertaintyMetrics
from .fairwashing import FairwashingDetector, FairwashingAudit
from .confidence_report import ConfidenceReport, TrustScore

__all__ = [
    'UncertaintyQuantifier',
    'UncertaintyMetrics',
    'FairwashingDetector',
    'FairwashingAudit',
    'ConfidenceReport',
    'TrustScore'
]
