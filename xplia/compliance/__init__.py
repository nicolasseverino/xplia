"""
Module de conformité réglementaire XPLIA
========================================

Ce package fournit les composants pour la conformité RGPD (droit à l'explication),
le support de l'AI Act européen et la génération automatique de rapports de conformité.
"""

# Import conditionnel pour éviter les erreurs si les modules ne sont pas encore chargés
try:
    from .compliance_checker import ComplianceChecker
except ImportError:
    # Créer une classe stub si le module n'existe pas encore
    class ComplianceChecker:
        """Stub pour ComplianceChecker."""
        def __init__(self, *args, **kwargs):
            pass

try:
    from .gdpr import GDPRCompliance
except ImportError:
    GDPRCompliance = None

try:
    from .ai_act import AIActCompliance
except ImportError:
    AIActCompliance = None

try:
    from .hipaa import HIPAACompliance
except ImportError:
    HIPAACompliance = None

__all__ = [
    'ComplianceChecker',
    'GDPRCompliance',
    'AIActCompliance',
    'HIPAACompliance',
]
