"""
Rapport de Confiance pour Explications
==================================

Ce module implémente un système de génération de rapports de confiance
qui intègre les métriques d'incertitude et les résultats de détection
de fairwashing pour fournir une évaluation globale de la fiabilité des explications.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import json

from ...core.base import ExplanationResult
from .uncertainty import UncertaintyMetrics
from .fairwashing import FairwashingAudit


class TrustLevel(Enum):
    """Niveaux de confiance pour les explications."""
    VERY_LOW = "very_low"       # Confiance très faible
    LOW = "low"                 # Confiance faible
    MODERATE = "moderate"       # Confiance modérée
    HIGH = "high"               # Confiance élevée
    VERY_HIGH = "very_high"     # Confiance très élevée


@dataclass
class TrustScore:
    """
    Score de confiance pour une explication.
    """
    # Score global de confiance (0 = confiance minimale, 1 = confiance maximale)
    global_trust: float = 0.0
    
    # Niveau de confiance
    trust_level: TrustLevel = TrustLevel.MODERATE
    
    # Scores par dimension
    uncertainty_trust: float = 0.0
    fairwashing_trust: float = 0.0
    consistency_trust: float = 0.0
    robustness_trust: float = 0.0
    
    # Facteurs influençant le score
    trust_factors: Dict[str, float] = field(default_factory=dict)
    
    # Métadonnées
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit le score en dictionnaire."""
        return {
            "global_trust": self.global_trust,
            "trust_level": self.trust_level.value,
            "uncertainty_trust": self.uncertainty_trust,
            "fairwashing_trust": self.fairwashing_trust,
            "consistency_trust": self.consistency_trust,
            "robustness_trust": self.robustness_trust,
            "trust_factors": self.trust_factors,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrustScore':
        """Crée un score à partir d'un dictionnaire."""
        try:
            trust_level = TrustLevel(data.get("trust_level", TrustLevel.MODERATE.value))
        except ValueError:
            logging.warning(f"Niveau de confiance inconnu, utilisation de MODERATE par défaut")
            trust_level = TrustLevel.MODERATE
        
        return cls(
            global_trust=data.get("global_trust", 0.0),
            trust_level=trust_level,
            uncertainty_trust=data.get("uncertainty_trust", 0.0),
            fairwashing_trust=data.get("fairwashing_trust", 0.0),
            consistency_trust=data.get("consistency_trust", 0.0),
            robustness_trust=data.get("robustness_trust", 0.0),
            trust_factors=data.get("trust_factors", {}),
            timestamp=data.get("timestamp", time.time())
        )


class ConfidenceReport:
    """
    Générateur de rapports de confiance pour les explications.
    """
    
    def __init__(self, 
                 uncertainty_weight: float = 0.4,
                 fairwashing_weight: float = 0.3,
                 consistency_weight: float = 0.2,
                 robustness_weight: float = 0.1):
        """
        Initialise le générateur de rapports.
        
        Args:
            uncertainty_weight: Poids de l'incertitude dans le score global
            fairwashing_weight: Poids du fairwashing dans le score global
            consistency_weight: Poids de la cohérence dans le score global
            robustness_weight: Poids de la robustesse dans le score global
        """
        self.uncertainty_weight = uncertainty_weight
        self.fairwashing_weight = fairwashing_weight
        self.consistency_weight = consistency_weight
        self.robustness_weight = robustness_weight
    
    def generate_report(self, 
                      explanation: ExplanationResult,
                      uncertainty_metrics: Optional[UncertaintyMetrics] = None,
                      fairwashing_audit: Optional[FairwashingAudit] = None,
                      additional_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Génère un rapport de confiance complet.
        
        Args:
            explanation: L'explication à évaluer
            uncertainty_metrics: Métriques d'incertitude (optionnel)
            fairwashing_audit: Audit de fairwashing (optionnel)
            additional_metrics: Métriques additionnelles (optionnel)
            
        Returns:
            Rapport de confiance
        """
        # Calculer le score de confiance
        trust_score = self.calculate_trust_score(
            explanation, uncertainty_metrics, fairwashing_audit, additional_metrics
        )
        
        # Générer le rapport complet
        report = {
            "trust_score": trust_score.to_dict(),
            "summary": self._generate_summary(trust_score),
            "recommendations": self._generate_recommendations(trust_score, explanation),
            "detailed_metrics": {}
        }
        
        # Ajouter les métriques d'incertitude si disponibles
        if uncertainty_metrics:
            if isinstance(uncertainty_metrics, dict):
                report["detailed_metrics"]["uncertainty"] = uncertainty_metrics
            else:
                report["detailed_metrics"]["uncertainty"] = uncertainty_metrics.to_dict()
        
        # Ajouter les résultats d'audit de fairwashing si disponibles
        if fairwashing_audit:
            if isinstance(fairwashing_audit, dict):
                report["detailed_metrics"]["fairwashing"] = fairwashing_audit
            else:
                report["detailed_metrics"]["fairwashing"] = fairwashing_audit.to_dict()
        
        # Ajouter les métriques additionnelles si disponibles
        if additional_metrics:
            report["detailed_metrics"].update(additional_metrics)
        
        return report
    
    def calculate_trust_score(self, 
                            explanation: ExplanationResult,
                            uncertainty_metrics: Optional[UncertaintyMetrics] = None,
                            fairwashing_audit: Optional[FairwashingAudit] = None,
                            additional_metrics: Optional[Dict[str, Any]] = None) -> TrustScore:
        """
        Calcule un score de confiance global.
        
        Args:
            explanation: L'explication à évaluer
            uncertainty_metrics: Métriques d'incertitude (optionnel)
            fairwashing_audit: Audit de fairwashing (optionnel)
            additional_metrics: Métriques additionnelles (optionnel)
            
        Returns:
            Score de confiance
        """
        # Initialiser le score de confiance
        trust_score = TrustScore()
        
        # Extraire les métriques d'incertitude des métadonnées si non fournies
        if uncertainty_metrics is None and hasattr(explanation.metadata, "uncertainty_metrics"):
            uncertainty_metrics = explanation.metadata.uncertainty_metrics
        
        # Extraire l'audit de fairwashing des métadonnées si non fourni
        if fairwashing_audit is None and hasattr(explanation.metadata, "fairwashing_audit"):
            fairwashing_audit = explanation.metadata.fairwashing_audit
        
        # Calculer le score de confiance lié à l'incertitude
        if uncertainty_metrics:
            if isinstance(uncertainty_metrics, dict):
                global_uncertainty = uncertainty_metrics.get("global_uncertainty", 0.0)
            else:
                global_uncertainty = uncertainty_metrics.global_uncertainty
            
            # Convertir l'incertitude en confiance (1 - incertitude)
            trust_score.uncertainty_trust = 1.0 - global_uncertainty
            
            # Enregistrer les facteurs d'incertitude
            trust_score.trust_factors["global_uncertainty"] = global_uncertainty
        else:
            # Valeur par défaut si aucune métrique d'incertitude n'est disponible
            trust_score.uncertainty_trust = 0.5
        
        # Calculer le score de confiance lié au fairwashing
        if fairwashing_audit:
            if isinstance(fairwashing_audit, dict):
                fairwashing_score = fairwashing_audit.get("fairwashing_score", 0.0)
            else:
                fairwashing_score = fairwashing_audit.fairwashing_score
            
            # Convertir le score de fairwashing en confiance (1 - fairwashing)
            trust_score.fairwashing_trust = 1.0 - fairwashing_score
            
            # Enregistrer les facteurs de fairwashing
            trust_score.trust_factors["fairwashing_score"] = fairwashing_score
        else:
            # Valeur par défaut si aucun audit de fairwashing n'est disponible
            trust_score.fairwashing_trust = 0.7
        
        # Calculer les scores de cohérence et de robustesse
        # (dans une implémentation réelle, ces scores seraient calculés
        # à partir de métriques spécifiques)
        trust_score.consistency_trust = 0.8  # Valeur par défaut
        trust_score.robustness_trust = 0.7   # Valeur par défaut
        
        if additional_metrics:
            if "consistency_score" in additional_metrics:
                trust_score.consistency_trust = additional_metrics["consistency_score"]
                trust_score.trust_factors["consistency_score"] = additional_metrics["consistency_score"]
            
            if "robustness_score" in additional_metrics:
                trust_score.robustness_trust = additional_metrics["robustness_score"]
                trust_score.trust_factors["robustness_score"] = additional_metrics["robustness_score"]
        
        # Calculer le score global de confiance (moyenne pondérée)
        trust_score.global_trust = (
            self.uncertainty_weight * trust_score.uncertainty_trust +
            self.fairwashing_weight * trust_score.fairwashing_trust +
            self.consistency_weight * trust_score.consistency_trust +
            self.robustness_weight * trust_score.robustness_trust
        )
        
        # Déterminer le niveau de confiance
        trust_score.trust_level = self._determine_trust_level(trust_score.global_trust)
        
        return trust_score
    
    def _determine_trust_level(self, trust_score: float) -> TrustLevel:
        """
        Détermine le niveau de confiance à partir du score.
        
        Args:
            trust_score: Score de confiance entre 0 et 1
            
        Returns:
            Niveau de confiance
        """
        if trust_score < 0.2:
            return TrustLevel.VERY_LOW
        elif trust_score < 0.4:
            return TrustLevel.LOW
        elif trust_score < 0.7:
            return TrustLevel.MODERATE
        elif trust_score < 0.9:
            return TrustLevel.HIGH
        else:
            return TrustLevel.VERY_HIGH
    
    def _generate_summary(self, trust_score: TrustScore) -> str:
        """
        Génère un résumé textuel du score de confiance.
        
        Args:
            trust_score: Score de confiance
            
        Returns:
            Résumé textuel
        """
        # Phrases selon le niveau de confiance
        trust_phrases = {
            TrustLevel.VERY_LOW: "L'explication présente un niveau de confiance très faible et ne devrait pas être utilisée pour des décisions importantes.",
            TrustLevel.LOW: "L'explication présente un niveau de confiance faible et devrait être utilisée avec une grande prudence.",
            TrustLevel.MODERATE: "L'explication présente un niveau de confiance modéré et peut être utilisée avec une certaine prudence.",
            TrustLevel.HIGH: "L'explication présente un niveau de confiance élevé et peut être utilisée avec confiance.",
            TrustLevel.VERY_HIGH: "L'explication présente un niveau de confiance très élevé et peut être utilisée avec une grande confiance."
        }
        
        # Phrase de base selon le niveau
        summary = trust_phrases[trust_score.trust_level]
        
        # Ajouter des détails sur les facteurs principaux
        factors = []
        
        if trust_score.uncertainty_trust < 0.5:
            factors.append("une incertitude élevée")
        
        if trust_score.fairwashing_trust < 0.6:
            factors.append("des signes potentiels de fairwashing")
        
        if trust_score.consistency_trust < 0.6:
            factors.append("un manque de cohérence")
        
        if trust_score.robustness_trust < 0.6:
            factors.append("un manque de robustesse")
        
        if factors:
            summary += f" Les principaux facteurs affectant la confiance sont : {', '.join(factors)}."
        
        return summary
    
    def _generate_recommendations(self, 
                                trust_score: TrustScore,
                                explanation: ExplanationResult) -> List[str]:
        """
        Génère des recommandations basées sur le score de confiance.
        
        Args:
            trust_score: Score de confiance
            explanation: L'explication évaluée
            
        Returns:
            Liste de recommandations
        """
        recommendations = []
        
        # Recommandations générales selon le niveau de confiance
        if trust_score.trust_level in [TrustLevel.VERY_LOW, TrustLevel.LOW]:
            recommendations.append("Envisagez d'utiliser un autre explainer ou une autre méthode d'explication.")
            recommendations.append("Complétez cette explication avec d'autres méthodes pour une analyse plus fiable.")
        
        # Recommandations spécifiques selon les facteurs
        if trust_score.uncertainty_trust < 0.5:
            recommendations.append("Augmentez la taille de l'échantillon ou utilisez des méthodes d'estimation d'incertitude plus robustes.")
        
        if trust_score.fairwashing_trust < 0.6:
            recommendations.append("Vérifiez si des biais sont masqués dans l'explication et examinez les features sensibles.")
        
        if trust_score.consistency_trust < 0.6:
            recommendations.append("Testez la cohérence de l'explication avec différentes initialisations ou paramètres.")
        
        if trust_score.robustness_trust < 0.6:
            recommendations.append("Évaluez la robustesse de l'explication face à des perturbations des données d'entrée.")
        
        return recommendations
    
    def apply_to_explanation(self, 
                           explanation: ExplanationResult,
                           report: Dict[str, Any]) -> ExplanationResult:
        """
        Applique le rapport de confiance à une explication.
        
        Args:
            explanation: L'explication à enrichir
            report: Rapport de confiance
            
        Returns:
            Explication enrichie avec le rapport de confiance
        """
        # Ajouter le rapport de confiance aux métadonnées
        explanation.metadata.confidence_report = report
        
        # Ajouter le score de confiance comme attribut direct
        if "trust_score" in report:
            explanation.trust_score = report["trust_score"]
        
        return explanation
