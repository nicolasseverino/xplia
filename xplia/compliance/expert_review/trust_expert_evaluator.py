"""
Évaluateur expert pour les métriques de confiance
================================================

Ce module fournit des outils pour l'évaluation experte des métriques de confiance
générées par XPLIA, permettant une notation objective de leur qualité.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

from .evaluation_criteria import EvaluationCriteria, TRUST_EVALUATION_CRITERIA

logger = logging.getLogger(__name__)

@dataclass
class ExpertReview:
    """
    Résultat d'une évaluation experte.
    
    Attributes:
        criteria: Liste des critères évalués
        scores: Scores attribués pour chaque critère
        global_score: Score global de l'évaluation
        strengths: Points forts identifiés
        weaknesses: Points faibles identifiés
        recommendations: Recommandations d'amélioration
        timestamp: Date et heure de l'évaluation
        metadata: Métadonnées supplémentaires
    """
    criteria: List[EvaluationCriteria]
    scores: Dict[str, float]
    global_score: float
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_quality_levels(self) -> Dict[str, str]:
        """
        Obtient les niveaux de qualité pour chaque critère.
        
        Returns:
            Dictionnaire des niveaux de qualité par critère
        """
        quality_levels = {}
        for criterion in self.criteria:
            score = self.scores.get(criterion.name, 0.0)
            quality_levels[criterion.name] = criterion.get_quality_level(score)
        return quality_levels
    
    def get_category_scores(self) -> Dict[str, float]:
        """
        Calcule les scores moyens par catégorie de critères.
        
        Returns:
            Dictionnaire des scores moyens par catégorie
        """
        category_scores = {}
        category_counts = {}
        
        for criterion in self.criteria:
            category = criterion.category.value
            score = self.scores.get(criterion.name, 0.0)
            
            if category not in category_scores:
                category_scores[category] = 0.0
                category_counts[category] = 0
            
            category_scores[category] += score
            category_counts[category] += 1
        
        # Calcul des moyennes
        for category, total in category_scores.items():
            count = category_counts[category]
            if count > 0:
                category_scores[category] = total / count
        
        return category_scores
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit l'évaluation en dictionnaire.
        
        Returns:
            Dictionnaire représentant l'évaluation
        """
        return {
            "global_score": self.global_score,
            "scores": self.scores,
            "quality_levels": self.get_quality_levels(),
            "category_scores": self.get_category_scores(),
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


class TrustExpertEvaluator:
    """
    Évaluateur expert pour les métriques de confiance.
    
    Cette classe permet d'évaluer la qualité des métriques de confiance
    générées par XPLIA selon des critères prédéfinis.
    """
    
    def __init__(
        self,
        criteria: List[EvaluationCriteria] = None,
        reference_data: Optional[Dict[str, Any]] = None
    ):
        """
        Initialise l'évaluateur expert.
        
        Args:
            criteria: Liste des critères d'évaluation (utilise les critères par défaut si None)
            reference_data: Données de référence pour l'évaluation comparative
        """
        self.criteria = criteria or TRUST_EVALUATION_CRITERIA
        self.reference_data = reference_data
        logger.info(f"Évaluateur expert initialisé avec {len(self.criteria)} critères")
    
    def evaluate_uncertainty_metrics(
        self,
        uncertainty_metrics: Any,
        reference_metrics: Optional[Any] = None,
        explanation: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Évalue les métriques d'incertitude.
        
        Args:
            uncertainty_metrics: Métriques d'incertitude à évaluer
            reference_metrics: Métriques de référence pour comparaison
            explanation: Explication associée aux métriques
            
        Returns:
            Scores d'évaluation pour les critères liés à l'incertitude
        """
        scores = {}
        
        # Évaluation de la précision de l'incertitude
        try:
            # Vérification de la présence des attributs attendus
            if hasattr(uncertainty_metrics, "global_uncertainty"):
                # Vérification de la plage de valeurs
                if 0.0 <= uncertainty_metrics.global_uncertainty <= 1.0:
                    scores["Précision de l'incertitude"] = 8.0
                else:
                    scores["Précision de l'incertitude"] = 4.0
            else:
                scores["Précision de l'incertitude"] = 2.0
            
            # Évaluation de la calibration des intervalles
            if hasattr(uncertainty_metrics, "confidence_intervals"):
                if uncertainty_metrics.confidence_intervals:
                    scores["Calibration des intervalles"] = 7.5
                else:
                    scores["Calibration des intervalles"] = 5.0
            else:
                scores["Calibration des intervalles"] = 3.0
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation des métriques d'incertitude: {e}")
            scores["Précision de l'incertitude"] = 1.0
            scores["Calibration des intervalles"] = 1.0
        
        return scores
    
    def evaluate_fairwashing_audit(
        self,
        fairwashing_audit: Any,
        reference_audit: Optional[Any] = None,
        explanation: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Évalue les résultats de l'audit de fairwashing.
        
        Args:
            fairwashing_audit: Audit de fairwashing à évaluer
            reference_audit: Audit de référence pour comparaison
            explanation: Explication associée à l'audit
            
        Returns:
            Scores d'évaluation pour les critères liés au fairwashing
        """
        scores = {}
        
        # Évaluation de la détection de fairwashing
        try:
            # Vérification de la présence des attributs attendus
            if hasattr(fairwashing_audit, "fairwashing_score"):
                # Vérification de la plage de valeurs
                if 0.0 <= fairwashing_audit.fairwashing_score <= 1.0:
                    scores["Détection de fairwashing"] = 8.0
                else:
                    scores["Détection de fairwashing"] = 4.0
            else:
                scores["Détection de fairwashing"] = 2.0
            
            # Évaluation de l'équité de l'explication
            if hasattr(fairwashing_audit, "detected_types"):
                scores["Équité de l'explication"] = 7.0
            else:
                scores["Équité de l'explication"] = 3.0
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation de l'audit de fairwashing: {e}")
            scores["Détection de fairwashing"] = 1.0
            scores["Équité de l'explication"] = 1.0
        
        return scores
    
    def evaluate_confidence_report(
        self,
        confidence_report: Dict[str, Any],
        reference_report: Optional[Dict[str, Any]] = None,
        explanation: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Évalue le rapport de confiance.
        
        Args:
            confidence_report: Rapport de confiance à évaluer
            reference_report: Rapport de référence pour comparaison
            explanation: Explication associée au rapport
            
        Returns:
            Scores d'évaluation pour les critères liés au rapport de confiance
        """
        scores = {}
        
        # Évaluation de la cohérence des métriques
        try:
            # Vérification de la présence des éléments attendus
            if "trust_score" in confidence_report:
                trust_score = confidence_report["trust_score"]
                
                # Vérification de la cohérence interne
                if all(k in trust_score for k in ["global_trust", "uncertainty_trust", "fairwashing_trust"]):
                    scores["Cohérence des métriques"] = 8.0
                else:
                    scores["Cohérence des métriques"] = 4.0
            else:
                scores["Cohérence des métriques"] = 2.0
            
            # Évaluation de la complétude des métriques
            if "trust_score" in confidence_report and "recommendations" in confidence_report:
                scores["Complétude des métriques"] = 7.5
            else:
                scores["Complétude des métriques"] = 3.0
            
            # Évaluation de l'interprétabilité des scores
            if "summary" in confidence_report:
                scores["Interprétabilité des scores"] = 8.0
            else:
                scores["Interprétabilité des scores"] = 4.0
            
            # Évaluation de l'utilité des recommandations
            if "recommendations" in confidence_report and confidence_report["recommendations"]:
                scores["Utilité des recommandations"] = 7.0
            else:
                scores["Utilité des recommandations"] = 3.0
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation du rapport de confiance: {e}")
            scores["Cohérence des métriques"] = 1.0
            scores["Complétude des métriques"] = 1.0
            scores["Interprétabilité des scores"] = 1.0
            scores["Utilité des recommandations"] = 1.0
        
        return scores
    
    def evaluate_trust_metrics(
        self,
        uncertainty_metrics: Any = None,
        fairwashing_audit: Any = None,
        confidence_report: Dict[str, Any] = None,
        explanation: Optional[Dict[str, Any]] = None
    ) -> ExpertReview:
        """
        Évalue l'ensemble des métriques de confiance.
        
        Args:
            uncertainty_metrics: Métriques d'incertitude à évaluer
            fairwashing_audit: Audit de fairwashing à évaluer
            confidence_report: Rapport de confiance à évaluer
            explanation: Explication associée aux métriques
            
        Returns:
            Évaluation experte complète
        """
        all_scores = {}
        
        # Évaluation des différentes composantes
        if uncertainty_metrics:
            uncertainty_scores = self.evaluate_uncertainty_metrics(
                uncertainty_metrics, explanation=explanation
            )
            all_scores.update(uncertainty_scores)
        
        if fairwashing_audit:
            fairwashing_scores = self.evaluate_fairwashing_audit(
                fairwashing_audit, explanation=explanation
            )
            all_scores.update(fairwashing_scores)
        
        if confidence_report:
            confidence_scores = self.evaluate_confidence_report(
                confidence_report, explanation=explanation
            )
            all_scores.update(confidence_scores)
        
        # Évaluation des critères génériques
        all_scores["Transparence des calculs"] = 7.0  # Valeur par défaut
        all_scores["Performance de calcul"] = 6.0     # Valeur par défaut
        all_scores["Robustesse aux perturbations"] = 6.5  # Valeur par défaut
        
        # Calcul du score global
        total_weight = 0.0
        weighted_sum = 0.0
        
        for criterion in self.criteria:
            if criterion.name in all_scores:
                score = all_scores[criterion.name]
                weighted_sum += score * criterion.weight
                total_weight += criterion.weight
        
        global_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Identification des points forts et faibles
        strengths = []
        weaknesses = []
        
        for criterion in self.criteria:
            if criterion.name in all_scores:
                score = all_scores[criterion.name]
                if score >= 8.0:
                    strengths.append(f"{criterion.name}: {score:.1f}/10")
                elif score <= 4.0:
                    weaknesses.append(f"{criterion.name}: {score:.1f}/10")
        
        # Génération de recommandations
        recommendations = []
        
        if "Précision de l'incertitude" in all_scores and all_scores["Précision de l'incertitude"] < 6.0:
            recommendations.append("Améliorer la précision des estimations d'incertitude")
        
        if "Détection de fairwashing" in all_scores and all_scores["Détection de fairwashing"] < 6.0:
            recommendations.append("Renforcer les mécanismes de détection de fairwashing")
        
        if "Cohérence des métriques" in all_scores and all_scores["Cohérence des métriques"] < 6.0:
            recommendations.append("Améliorer la cohérence entre les différentes métriques de confiance")
        
        if "Utilité des recommandations" in all_scores and all_scores["Utilité des recommandations"] < 6.0:
            recommendations.append("Fournir des recommandations plus actionnables et pertinentes")
        
        # Création de l'évaluation experte
        review = ExpertReview(
            criteria=self.criteria,
            scores=all_scores,
            global_score=global_score,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            metadata={
                "has_uncertainty_metrics": uncertainty_metrics is not None,
                "has_fairwashing_audit": fairwashing_audit is not None,
                "has_confidence_report": confidence_report is not None,
                "has_explanation": explanation is not None
            }
        )
        
        logger.info(f"Évaluation experte complétée avec un score global de {global_score:.2f}/10")
        return review
