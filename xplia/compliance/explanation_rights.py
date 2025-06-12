"""
Gestion du droit à l'explication (RGPD) pour XPLIA
==================================================

Ce module fournit les outils pour répondre automatiquement et auditablement
aux demandes d'explication d'un utilisateur final, conformément au RGPD.
"""

from typing import Any, Dict
import datetime

class ExplanationRequestLog:
    """
    Journalise toutes les demandes d'explication pour auditabilité RGPD.
    """
    def __init__(self):
        self._log = []
    def add(self, user_id: str, input_data: Any, explanation: Any):
        self._log.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "user_id": user_id,
            "input": input_data,
            "explanation": explanation
        })
    def export(self) -> list:
        return self._log

class GDPRComplianceManager:
    """
    Gère le droit à l'explication (article 22 RGPD) pour XPLIA.
    """
    def __init__(self):
        self.request_log = ExplanationRequestLog()
    def request_explanation(self, user_id: str, model, input_data: Any, explainer, **kwargs) -> Dict:
        # Génère une explication, journalise la demande et renvoie la réponse conforme RGPD
        explanation = explainer.explain_instance(input_data, **kwargs)
        self.request_log.add(user_id, input_data, explanation)
        return {
            "user_id": user_id,
            "input": input_data,
            "explanation": explanation,
            "compliance": "RGPD Article 22 - Droit à l'explication satisfait"
        }
    def export_audit_trail(self) -> list:
        return self.request_log.export()
