"""
XPLIA Auto-Calibration et Adaptation d'Audience
============================================

Ce module implémente un système sophistiqué d'auto-calibration et d'adaptation 
d'audience qui permet:
1. L'ajustement automatique du niveau de détail des explications
2. L'adaptation du style et du contenu selon le profil de l'utilisateur
3. L'optimisation dynamique des paramètres d'explication selon le contexte

Cette approche avancée améliore considérablement la pertinence et
l'accessibilité des explications pour différents profils d'utilisateurs.
"""

from .audience_adapter import AudienceAdapter
from .explanation_calibrator import ExplanationCalibrator
from .audience_profiles import UserProfile, AudienceProfileManager
from .calibration_metrics import CalibrationMetrics

__all__ = [
    'AudienceAdapter',
    'ExplanationCalibrator',
    'UserProfile',
    'AudienceProfileManager',
    'CalibrationMetrics'
]
