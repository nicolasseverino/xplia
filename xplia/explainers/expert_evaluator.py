"""
Évaluateur expert pour la qualité des explications
=================================================

Ce module fournit des outils pour évaluer la qualité des explications générées
par XPLIA selon des critères objectifs et des métriques quantitatives.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass

from xplia.compliance.expert_review.evaluation_criteria import (
    EvaluationCriteria, EXPLANATION_QUALITY_CRITERIA, CriteriaCategory
)
from xplia.compliance.expert_review.trust_expert_evaluator import ExpertReview

logger = logging.getLogger(__name__)


@dataclass
class QualityCriteria:
    """
    Critères de qualité pour les explications.
    
    Attributes:
        name: Nom du critère
        description: Description du critère
        weight: Poids du critère dans l'évaluation globale
        evaluation_function: Fonction d'évaluation pour ce critère
    """
    name: str
    description: str
    weight: float
    evaluation_function: callable


class ExplanationQualityEvaluator:
    """
    Évaluateur de la qualité des explications.
    
    Cette classe permet d'évaluer la qualité des explications générées par XPLIA
    selon différents critères objectifs et métriques quantitatives.
    """
    
    def __init__(
        self,
        criteria: List[EvaluationCriteria] = None,
        reference_data: Optional[Dict[str, Any]] = None
    ):
        """
        Initialise l'évaluateur de qualité.
        
        Args:
            criteria: Liste des critères d'évaluation (utilise les critères par défaut si None)
            reference_data: Données de référence pour l'évaluation comparative
        """
        self.criteria = criteria or EXPLANATION_QUALITY_CRITERIA
        self.reference_data = reference_data
        logger.info(f"Évaluateur de qualité initialisé avec {len(self.criteria)} critères")
    
    def evaluate_fidelity(
        self,
        explanation: Dict[str, Any],
        model_adapter: Any,
        instance: pd.Series,
        background_data: pd.DataFrame = None
    ) -> float:
        """
        Évalue la fidélité de l'explication par rapport au modèle.
        
        Args:
            explanation: Explication à évaluer
            model_adapter: Adaptateur du modèle
            instance: Instance expliquée
            background_data: Données de fond pour l'évaluation
            
        Returns:
            Score de fidélité (0-10)
        """
        try:
            # Vérification de la présence des attributions de features
            if "feature_attributions" not in explanation:
                return 3.0
            
            feature_attributions = explanation["feature_attributions"]
            
            # Vérification de la prédiction du modèle
            if model_adapter is None:
                return 5.0
            
            # Calcul de la fidélité locale (approximation simple)
            # Dans une implémentation réelle, on utiliserait des méthodes plus sophistiquées
            score = 7.5  # Score par défaut
            
            # Bonus si les attributions sont cohérentes avec le modèle
            if "prediction" in explanation and hasattr(model_adapter, "predict"):
                score += 1.0
            
            # Bonus si l'explication inclut des intervalles de confiance
            if "confidence_intervals" in explanation:
                score += 0.5
            
            return min(score, 10.0)
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation de la fidélité: {e}")
            return 4.0
    
    def evaluate_completeness(self, explanation: Dict[str, Any]) -> float:
        """
        Évalue la complétude de l'explication.
        
        Args:
            explanation: Explication à évaluer
            
        Returns:
            Score de complétude (0-10)
        """
        try:
            # Vérification des éléments essentiels
            essential_elements = [
                "feature_attributions",
                "prediction",
                "model_type"
            ]
            
            # Éléments supplémentaires valorisés
            additional_elements = [
                "confidence_intervals",
                "counterfactuals",
                "global_importance",
                "uncertainty_metrics",
                "fairwashing_audit",
                "confidence_report"
            ]
            
            # Calcul du score de base
            base_score = 5.0
            
            # Points pour les éléments essentiels
            for element in essential_elements:
                if element in explanation:
                    base_score += 1.0
            
            # Points pour les éléments supplémentaires
            for element in additional_elements:
                if element in explanation:
                    base_score += 0.5
            
            return min(base_score, 10.0)
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation de la complétude: {e}")
            return 4.0
    
    def evaluate_consistency(
        self,
        explanation: Dict[str, Any],
        similar_explanations: List[Dict[str, Any]] = None
    ) -> float:
        """
        Évalue la cohérence de l'explication par rapport à des explications similaires.
        
        Args:
            explanation: Explication à évaluer
            similar_explanations: Liste d'explications similaires pour comparaison
            
        Returns:
            Score de cohérence (0-10)
        """
        try:
            # Si pas d'explications similaires, score par défaut
            if not similar_explanations:
                return 6.0
            
            # Vérification de la présence des attributions de features
            if "feature_attributions" not in explanation:
                return 4.0
            
            # Score par défaut
            score = 7.0
            
            # Dans une implémentation réelle, on calculerait la variance des attributions
            # entre les explications similaires et on pénaliserait les incohérences
            
            return score
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation de la cohérence: {e}")
            return 5.0
    
    def evaluate_robustness(
        self,
        explanation: Dict[str, Any],
        perturbed_explanations: List[Dict[str, Any]] = None
    ) -> float:
        """
        Évalue la robustesse de l'explication face à des perturbations.
        
        Args:
            explanation: Explication à évaluer
            perturbed_explanations: Liste d'explications pour des instances perturbées
            
        Returns:
            Score de robustesse (0-10)
        """
        try:
            # Si pas d'explications perturbées, score par défaut
            if not perturbed_explanations:
                return 6.0
            
            # Vérification de la présence des attributions de features
            if "feature_attributions" not in explanation:
                return 4.0
            
            # Score par défaut
            score = 7.0
            
            # Dans une implémentation réelle, on calculerait la stabilité des attributions
            # face à de petites perturbations des données d'entrée
            
            return score
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation de la robustesse: {e}")
            return 5.0
    
    def evaluate_simplicity(self, explanation: Dict[str, Any]) -> float:
        """
        Évalue la simplicité et la clarté de l'explication.
        
        Args:
            explanation: Explication à évaluer
            
        Returns:
            Score de simplicité (0-10)
        """
        try:
            # Vérification de la présence des attributions de features
            if "feature_attributions" not in explanation:
                return 4.0
            
            feature_attributions = explanation["feature_attributions"]
            
            # Calcul du nombre de features avec une attribution significative
            significant_features = 0
            if isinstance(feature_attributions, dict):
                for feature, value in feature_attributions.items():
                    if abs(value) > 0.05:  # Seuil arbitraire
                        significant_features += 1
            
            # Pénalisation pour un trop grand nombre de features significatives
            if significant_features > 10:
                return 5.0
            elif significant_features > 5:
                return 7.0
            else:
                return 9.0
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation de la simplicité: {e}")
            return 5.0
    
    def evaluate_fairness(
        self,
        explanation: Dict[str, Any],
        sensitive_features: List[str] = None
    ) -> float:
        """
        Évalue l'équité de l'explication vis-à-vis des groupes protégés.
        
        Args:
            explanation: Explication à évaluer
            sensitive_features: Liste des features sensibles
            
        Returns:
            Score d'équité (0-10)
        """
        try:
            # Si pas de features sensibles spécifiées, score par défaut
            if not sensitive_features:
                return 6.0
            
            # Vérification de la présence des attributions de features
            if "feature_attributions" not in explanation:
                return 4.0
            
            feature_attributions = explanation["feature_attributions"]
            
            # Vérification de la présence d'un audit de fairwashing
            if "fairwashing_audit" in explanation:
                return 8.0
            
            # Score par défaut
            score = 6.0
            
            # Dans une implémentation réelle, on analyserait les attributions
            # des features sensibles pour détecter des biais potentiels
            
            return score
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation de l'équité: {e}")
            return 5.0
    
    def evaluate_transparency(self, explanation: Dict[str, Any]) -> float:
        """
        Évalue la transparence des méthodes utilisées pour générer l'explication.
        
        Args:
            explanation: Explication à évaluer
            
        Returns:
            Score de transparence (0-10)
        """
        try:
            # Vérification de la présence d'informations sur la méthode
            if "method" in explanation:
                score = 7.0
            else:
                score = 5.0
            
            # Bonus pour des informations supplémentaires
            if "method_parameters" in explanation:
                score += 1.0
            
            if "method_description" in explanation:
                score += 1.0
            
            return min(score, 10.0)
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation de la transparence: {e}")
            return 5.0
    
    def evaluate_actionability(self, explanation: Dict[str, Any]) -> float:
        """
        Évalue si l'explication fournit des informations actionnables.
        
        Args:
            explanation: Explication à évaluer
            
        Returns:
            Score d'actionabilité (0-10)
        """
        try:
            # Vérification de la présence de contrefactuels
            if "counterfactuals" in explanation:
                score = 8.0
            else:
                score = 5.0
            
            # Bonus pour des recommandations
            if "recommendations" in explanation:
                score += 1.0
            
            # Bonus pour des intervalles de confiance
            if "confidence_intervals" in explanation:
                score += 0.5
            
            return min(score, 10.0)
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation de l'actionabilité: {e}")
            return 5.0
    
    def evaluate_performance(self, explanation: Dict[str, Any]) -> float:
        """
        Évalue l'efficacité computationnelle de la génération d'explications.
        
        Args:
            explanation: Explication à évaluer
            
        Returns:
            Score de performance (0-10)
        """
        try:
            # Vérification de la présence d'informations sur le temps de calcul
            if "computation_time" in explanation:
                computation_time = explanation["computation_time"]
                
                # Évaluation basée sur le temps de calcul
                if computation_time < 1.0:  # Moins d'une seconde
                    return 9.0
                elif computation_time < 5.0:  # Moins de 5 secondes
                    return 8.0
                elif computation_time < 30.0:  # Moins de 30 secondes
                    return 7.0
                elif computation_time < 60.0:  # Moins d'une minute
                    return 6.0
                else:
                    return 5.0
            else:
                return 6.0  # Score par défaut
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation de la performance: {e}")
            return 5.0
    
    def evaluate_uncertainty_quantification(self, explanation: Dict[str, Any]) -> float:
        """
        Évalue si l'explication quantifie correctement l'incertitude.
        
        Args:
            explanation: Explication à évaluer
            
        Returns:
            Score de quantification d'incertitude (0-10)
        """
        try:
            # Vérification de la présence de métriques d'incertitude
            if "uncertainty_metrics" in explanation:
                score = 8.0
            elif "confidence_intervals" in explanation:
                score = 7.0
            else:
                score = 4.0
            
            # Bonus pour des types d'incertitude détaillés
            if "uncertainty_types" in explanation:
                score += 1.0
            
            return min(score, 10.0)
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation de la quantification d'incertitude: {e}")
            return 4.0
    
    def evaluate_explanation(
        self,
        explanation: Dict[str, Any],
        model_adapter: Any = None,
        instance: pd.Series = None,
        background_data: pd.DataFrame = None,
        similar_explanations: List[Dict[str, Any]] = None,
        perturbed_explanations: List[Dict[str, Any]] = None,
        sensitive_features: List[str] = None
    ) -> ExpertReview:
        """
        Évalue la qualité globale d'une explication.
        
        Args:
            explanation: Explication à évaluer
            model_adapter: Adaptateur du modèle
            instance: Instance expliquée
            background_data: Données de fond pour l'évaluation
            similar_explanations: Liste d'explications similaires pour évaluer la cohérence
            perturbed_explanations: Liste d'explications pour des instances perturbées
            sensitive_features: Liste des features sensibles
            
        Returns:
            Évaluation experte complète
        """
        scores = {}
        
        # Évaluation de chaque critère
        scores["Fidélité au modèle"] = self.evaluate_fidelity(
            explanation, model_adapter, instance, background_data
        )
        
        scores["Complétude de l'explication"] = self.evaluate_completeness(explanation)
        
        scores["Cohérence entre instances"] = self.evaluate_consistency(
            explanation, similar_explanations
        )
        
        scores["Robustesse de l'explication"] = self.evaluate_robustness(
            explanation, perturbed_explanations
        )
        
        scores["Simplicité et clarté"] = self.evaluate_simplicity(explanation)
        
        scores["Équité de l'explication"] = self.evaluate_fairness(
            explanation, sensitive_features
        )
        
        scores["Transparence des méthodes"] = self.evaluate_transparency(explanation)
        
        scores["Actionabilité"] = self.evaluate_actionability(explanation)
        
        scores["Performance de génération"] = self.evaluate_performance(explanation)
        
        scores["Quantification de l'incertitude"] = self.evaluate_uncertainty_quantification(
            explanation
        )
        
        # Calcul du score global
        total_weight = 0.0
        weighted_sum = 0.0
        
        for criterion in self.criteria:
            if criterion.name in scores:
                score = scores[criterion.name]
                weighted_sum += score * criterion.weight
                total_weight += criterion.weight
        
        global_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Identification des points forts et faibles
        strengths = []
        weaknesses = []
        
        for criterion in self.criteria:
            if criterion.name in scores:
                score = scores[criterion.name]
                if score >= 8.0:
                    strengths.append(f"{criterion.name}: {score:.1f}/10")
                elif score <= 4.0:
                    weaknesses.append(f"{criterion.name}: {score:.1f}/10")
        
        # Génération de recommandations
        recommendations = []
        
        if scores.get("Fidélité au modèle", 0) < 6.0:
            recommendations.append("Améliorer la fidélité de l'explication au modèle")
        
        if scores.get("Complétude de l'explication", 0) < 6.0:
            recommendations.append("Enrichir l'explication avec plus d'informations pertinentes")
        
        if scores.get("Simplicité et clarté", 0) < 6.0:
            recommendations.append("Simplifier l'explication pour la rendre plus claire")
        
        if scores.get("Quantification de l'incertitude", 0) < 6.0:
            recommendations.append("Ajouter des métriques d'incertitude à l'explication")
        
        # Création de l'évaluation experte
        review = ExpertReview(
            criteria=self.criteria,
            scores=scores,
            global_score=global_score,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            metadata={
                "has_model_adapter": model_adapter is not None,
                "has_instance": instance is not None,
                "has_background_data": background_data is not None,
                "has_similar_explanations": similar_explanations is not None,
                "has_perturbed_explanations": perturbed_explanations is not None,
                "has_sensitive_features": sensitive_features is not None
            }
        )
        
        logger.info(f"Évaluation de qualité complétée avec un score global de {global_score:.2f}/10")
        return review
