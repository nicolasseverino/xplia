"""
Estimateur de Qualité d'Explication
=================================

Ce module implémente un système sophistiqué pour évaluer la qualité
des explications selon plusieurs dimensions cruciales telles que la 
fidélité, l'intelligibilité, la consistance et la robustesse.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass

from ...core.base import ExplanationResult, ExplanationQuality


@dataclass
class QualityMetrics:
    """
    Métriques multidimensionnelles de qualité pour une explication.
    """
    fidelity: float = 0.0          # Fidélité au modèle original
    intelligibility: float = 0.0   # Facilité de compréhension
    consistency: float = 0.0       # Cohérence interne de l'explication
    robustness: float = 0.0        # Robustesse aux perturbations
    completeness: float = 0.0      # Couverture des aspects importants
    parsimony: float = 0.0         # Simplicité/concision de l'explication
    
    def overall_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calcule le score global de qualité en fonction des poids spécifiés.
        
        Args:
            weights: Dictionnaire des poids à appliquer à chaque dimension
                    (par défaut, toutes les dimensions ont le même poids)
                    
        Returns:
            Score global de qualité entre 0 et 1
        """
        default_weights = {
            "fidelity": 0.25,
            "intelligibility": 0.2,
            "consistency": 0.15,
            "robustness": 0.15,
            "completeness": 0.15,
            "parsimony": 0.1
        }
        
        weights = weights or default_weights
        
        # Normaliser les poids pour qu'ils somment à 1
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        
        # Calculer le score pondéré
        score = 0.0
        for metric, weight in normalized_weights.items():
            if hasattr(self, metric):
                score += getattr(self, metric) * weight
        
        return score


class QualityEstimator:
    """
    Système d'estimation de la qualité des explications selon
    plusieurs dimensions et critères.
    """
    
    def __init__(self, 
                 reference_data: Optional[Any] = None,
                 reference_model: Optional[Any] = None,
                 custom_metrics: Optional[Dict[str, callable]] = None,
                 **kwargs):
        """
        Initialise l'estimateur de qualité.
        
        Args:
            reference_data: Données de référence pour l'évaluation
            reference_model: Modèle de référence pour l'évaluation
            custom_metrics: Métriques personnalisées supplémentaires
            **kwargs: Paramètres additionnels
        """
        self.reference_data = reference_data
        self.reference_model = reference_model
        self.custom_metrics = custom_metrics or {}
        
        # Historique des évaluations pour l'étalonnage
        self.evaluation_history = []
    
    def estimate_quality(self, 
                        explanation: ExplanationResult,
                        model: Any,
                        X: Any,
                        context: Optional[Dict[str, Any]] = None) -> float:
        """
        Estime la qualité globale d'une explication.
        
        Args:
            explanation: Résultat d'explication à évaluer
            model: Modèle expliqué
            X: Données expliquées
            context: Contexte d'explication
            
        Returns:
            Score global de qualité entre 0 et 1
        """
        # Calculer les métriques de qualité
        metrics = self._compute_quality_metrics(
            explanation=explanation,
            model=model,
            X=X,
            context=context
        )
        
        # Ajuster les poids selon le contexte
        weights = self._get_context_adapted_weights(context)
        
        # Calculer le score global
        overall_score = metrics.overall_score(weights)
        
        # Sauvegarder l'évaluation pour l'étalonnage futur
        self.evaluation_history.append((metrics, overall_score, context))
        
        return overall_score
    
    def _compute_quality_metrics(self, 
                               explanation: ExplanationResult,
                               model: Any,
                               X: Any,
                               context: Optional[Dict[str, Any]] = None) -> QualityMetrics:
        """
        Calcule les métriques de qualité multidimensionnelles pour une explication.
        
        Args:
            explanation: Résultat d'explication à évaluer
            model: Modèle expliqué
            X: Données expliquées
            context: Contexte d'explication
            
        Returns:
            Métriques de qualité calculées
        """
        metrics = QualityMetrics()
        
        # Récupérer les métadonnées de qualité si disponibles
        quality_metadata = getattr(explanation.metadata, "quality", None)
        if quality_metadata and isinstance(quality_metadata, ExplanationQuality):
            # Utiliser les métriques pré-calculées si disponibles
            metrics.fidelity = getattr(quality_metadata, "fidelity", 0.0)
            metrics.intelligibility = getattr(quality_metadata, "intelligibility", 0.0)
            metrics.consistency = getattr(quality_metadata, "consistency", 0.0)
            metrics.robustness = getattr(quality_metadata, "robustness", 0.0)
            metrics.completeness = getattr(quality_metadata, "completeness", 0.0)
            metrics.parsimony = getattr(quality_metadata, "parsimony", 0.0)
        else:
            # Calculer les métriques
            metrics.fidelity = self._evaluate_fidelity(explanation, model, X)
            metrics.intelligibility = self._evaluate_intelligibility(explanation, context)
            metrics.consistency = self._evaluate_consistency(explanation)
            metrics.robustness = self._evaluate_robustness(explanation, model, X)
            metrics.completeness = self._evaluate_completeness(explanation, model, X)
            metrics.parsimony = self._evaluate_parsimony(explanation)
            
            # Appliquer des métriques personnalisées si disponibles
            for name, metric_fn in self.custom_metrics.items():
                try:
                    value = metric_fn(explanation, model, X, context)
                    if hasattr(metrics, name):
                        setattr(metrics, name, value)
                    else:
                        logging.warning(f"Métrique personnalisée '{name}' ignorée: attribut non défini dans QualityMetrics")
                except Exception as e:
                    logging.error(f"Échec du calcul de la métrique personnalisée '{name}': {str(e)}")
        
        return metrics
    
    def _get_context_adapted_weights(self, 
                                    context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Détermine les poids optimaux pour les métriques en fonction du contexte.
        
        Args:
            context: Contexte d'explication
            
        Returns:
            Dictionnaire des poids adaptés au contexte
        """
        if not context:
            return {
                "fidelity": 0.25,
                "intelligibility": 0.2,
                "consistency": 0.15,
                "robustness": 0.15,
                "completeness": 0.15,
                "parsimony": 0.1
            }
        
        # Adapter les poids selon l'audience
        audience = context.get("audience_level", "technical")
        if audience == "non_technical":
            return {
                "fidelity": 0.15,
                "intelligibility": 0.35,  # Accent sur l'intelligibilité
                "consistency": 0.15,
                "robustness": 0.1,
                "completeness": 0.1,
                "parsimony": 0.15       # Plus d'importance à la parcimonie
            }
        elif audience == "regulatory":
            return {
                "fidelity": 0.3,        # Accent sur la fidélité
                "intelligibility": 0.15,
                "consistency": 0.2,      # Plus d'importance à la cohérence
                "robustness": 0.2,       # Plus d'importance à la robustesse
                "completeness": 0.1,
                "parsimony": 0.05
            }
        
        # Adapter selon la criticité
        criticality = context.get("criticality", "medium")
        if criticality == "high":
            return {
                "fidelity": 0.3,
                "intelligibility": 0.15,
                "consistency": 0.15,
                "robustness": 0.25,     # Plus d'importance à la robustesse
                "completeness": 0.1,
                "parsimony": 0.05
            }
        
        # Par défaut, poids standards
        return {
            "fidelity": 0.25,
            "intelligibility": 0.2,
            "consistency": 0.15,
            "robustness": 0.15,
            "completeness": 0.15,
            "parsimony": 0.1
        }
    
    # Méthodes d'évaluation des différentes dimensions de qualité
    
    def _evaluate_fidelity(self, 
                          explanation: ExplanationResult,
                          model: Any,
                          X: Any) -> float:
        """
        Évalue la fidélité de l'explication par rapport au modèle.
        
        Args:
            explanation: Explication à évaluer
            model: Modèle expliqué
            X: Données expliquées
            
        Returns:
            Score de fidélité entre 0 et 1
        """
        # Dans une implémentation complète, cette méthode pourrait:
        # 1. Générer des prédictions avec le modèle original
        # 2. Générer des prédictions avec un modèle proxy basé sur l'explication
        # 3. Comparer les deux ensembles de prédictions
        
        # Version simplifiée pour l'exemple
        if hasattr(explanation, "fidelity_score"):
            return min(1.0, max(0.0, explanation.fidelity_score))
        
        # Valeur par défaut modérée
        return 0.7
    
    def _evaluate_intelligibility(self,
                                explanation: ExplanationResult,
                                context: Optional[Dict[str, Any]] = None) -> float:
        """
        Évalue l'intelligibilité de l'explication.
        
        Args:
            explanation: Explication à évaluer
            context: Contexte d'explication
            
        Returns:
            Score d'intelligibilité entre 0 et 1
        """
        # Dans une implémentation complète, cette méthode pourrait:
        # 1. Analyser la complexité de l'explication (nombre de règles, profondeur d'arbre, etc.)
        # 2. Évaluer l'adéquation au niveau d'audience ciblé
        # 3. Vérifier la présence d'éléments visuels et textuels complémentaires
        
        # Version simplifiée pour l'exemple
        if hasattr(explanation.metadata, "complexity"):
            complexity = explanation.metadata.complexity
            return 1.0 - min(1.0, complexity / 10.0)  # Plus c'est complexe, moins c'est intelligible
        
        # Valeur par défaut modérée
        return 0.6
    
    def _evaluate_consistency(self, explanation: ExplanationResult) -> float:
        """
        Évalue la cohérence interne de l'explication.
        
        Args:
            explanation: Explication à évaluer
            
        Returns:
            Score de cohérence entre 0 et 1
        """
        # Version simplifiée pour l'exemple
        if hasattr(explanation.metadata, "consistency"):
            return min(1.0, max(0.0, explanation.metadata.consistency))
        
        # Valeur par défaut modérée
        return 0.8
    
    def _evaluate_robustness(self, 
                           explanation: ExplanationResult,
                           model: Any,
                           X: Any) -> float:
        """
        Évalue la robustesse de l'explication face aux perturbations.
        
        Args:
            explanation: Explication à évaluer
            model: Modèle expliqué
            X: Données expliquées
            
        Returns:
            Score de robustesse entre 0 et 1
        """
        # Version simplifiée pour l'exemple
        if hasattr(explanation.metadata, "robustness"):
            return min(1.0, max(0.0, explanation.metadata.robustness))
        
        # Valeur par défaut modérée
        return 0.7
    
    def _evaluate_completeness(self, 
                             explanation: ExplanationResult,
                             model: Any,
                             X: Any) -> float:
        """
        Évalue la complétude de l'explication (couverture des aspects importants).
        
        Args:
            explanation: Explication à évaluer
            model: Modèle expliqué
            X: Données expliquées
            
        Returns:
            Score de complétude entre 0 et 1
        """
        # Version simplifiée pour l'exemple
        if hasattr(explanation.metadata, "completeness"):
            return min(1.0, max(0.0, explanation.metadata.completeness))
        
        # Valeur par défaut modérée
        return 0.75
    
    def _evaluate_parsimony(self, explanation: ExplanationResult) -> float:
        """
        Évalue la parcimonie (simplicité/concision) de l'explication.
        
        Args:
            explanation: Explication à évaluer
            
        Returns:
            Score de parcimonie entre 0 et 1
        """
        # Version simplifiée pour l'exemple
        if hasattr(explanation.metadata, "complexity"):
            complexity = explanation.metadata.complexity
            return max(0.0, 1.0 - min(1.0, complexity / 5.0))
        
        # Valeur par défaut modérée
        return 0.65
    
    def calibrate(self, 
                explanations: List[ExplanationResult],
                models: List[Any],
                X_list: List[Any],
                ground_truth_scores: Optional[List[float]] = None,
                **kwargs) -> None:
        """
        Calibre l'estimateur de qualité sur un ensemble d'explications de référence.
        
        Args:
            explanations: Liste d'explications de référence
            models: Liste des modèles correspondants
            X_list: Liste des données correspondantes
            ground_truth_scores: Scores de qualité de référence (optionnel)
            **kwargs: Paramètres additionnels
        """
        # Cette méthode pourrait être implémentée pour ajuster les paramètres
        # internes de l'estimateur en fonction d'un ensemble de référence
        pass
