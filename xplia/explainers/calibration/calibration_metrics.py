"""
Métriques de Calibration
=====================

Ce module implémente diverses métriques pour évaluer la qualité
des explications et guider le processus d'auto-calibration.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import time

from ...core.base import ExplanationResult


class CalibrationMetrics:
    """
    Métriques pour évaluer et optimiser la calibration des explications.
    """
    
    def __init__(self):
        """Initialise les métriques de calibration."""
        pass
    
    def evaluate_explanation(self, 
                           explanation: ExplanationResult,
                           ground_truth: Optional[Any] = None,
                           user_feedback: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Évalue une explication selon diverses métriques.
        
        Args:
            explanation: L'explication à évaluer
            ground_truth: Vérité terrain (optionnel)
            user_feedback: Feedback utilisateur (optionnel)
            
        Returns:
            Dictionnaire des scores de métriques
        """
        metrics = {}
        
        # Évaluer la complexité
        metrics["complexity_score"] = self.measure_complexity(explanation)
        
        # Évaluer la concision
        metrics["conciseness_score"] = self.measure_conciseness(explanation)
        
        # Évaluer la cohérence
        metrics["coherence_score"] = self.measure_coherence(explanation)
        
        # Si une vérité terrain est disponible, évaluer la fidélité
        if ground_truth is not None:
            metrics["fidelity_score"] = self.measure_fidelity(explanation, ground_truth)
        
        # Si un feedback utilisateur est disponible, évaluer la satisfaction
        if user_feedback is not None:
            metrics["user_satisfaction"] = self.measure_user_satisfaction(user_feedback)
        
        # Calculer un score global
        available_scores = [score for score in metrics.values() if isinstance(score, (int, float))]
        if available_scores:
            metrics["overall_score"] = sum(available_scores) / len(available_scores)
        
        return metrics
    
    def measure_complexity(self, explanation: ExplanationResult) -> float:
        """
        Mesure la complexité d'une explication.
        
        Args:
            explanation: L'explication à évaluer
            
        Returns:
            Score de complexité entre 0 et 1
        """
        # Dans une implémentation réelle, cette méthode pourrait:
        # 1. Analyser la complexité linguistique du texte
        # 2. Compter le nombre de termes techniques
        # 3. Évaluer la complexité des visualisations
        
        # Pour l'exemple, nous retournons une valeur simulée
        # basée sur des métadonnées d'explication
        
        complexity = 0.5  # Valeur par défaut
        
        # Si des métadonnées de complexité sont disponibles
        if hasattr(explanation.metadata, "complexity_level"):
            complexity = explanation.metadata.complexity_level
        
        # Normaliser entre 0 et 1
        return min(1.0, max(0.0, complexity))
    
    def measure_conciseness(self, explanation: ExplanationResult) -> float:
        """
        Mesure la concision d'une explication.
        
        Args:
            explanation: L'explication à évaluer
            
        Returns:
            Score de concision entre 0 et 1
        """
        # Dans une implémentation réelle, cette méthode pourrait:
        # 1. Évaluer la longueur du texte par rapport à l'information transmise
        # 2. Détecter les redondances ou répétitions
        # 3. Mesurer la densité d'information
        
        # Pour l'exemple, nous retournons une valeur simulée
        conciseness = 0.5  # Valeur par défaut
        
        # Si des métadonnées de concision sont disponibles
        if hasattr(explanation.metadata, "conciseness_level"):
            conciseness = explanation.metadata.conciseness_level
        
        # Normaliser entre 0 et 1
        return min(1.0, max(0.0, conciseness))
    
    def measure_coherence(self, explanation: ExplanationResult) -> float:
        """
        Mesure la cohérence d'une explication.
        
        Args:
            explanation: L'explication à évaluer
            
        Returns:
            Score de cohérence entre 0 et 1
        """
        # Dans une implémentation réelle, cette méthode pourrait:
        # 1. Vérifier la cohérence interne de l'explication
        # 2. Détecter les contradictions
        # 3. Évaluer la structure logique
        
        # Pour l'exemple, nous retournons une valeur simulée
        coherence = 0.7  # Valeur par défaut
        
        # Si des métadonnées de cohérence sont disponibles
        if hasattr(explanation.metadata, "coherence_level"):
            coherence = explanation.metadata.coherence_level
        
        # Normaliser entre 0 et 1
        return min(1.0, max(0.0, coherence))
    
    def measure_fidelity(self, 
                        explanation: ExplanationResult,
                        ground_truth: Any) -> float:
        """
        Mesure la fidélité d'une explication par rapport à une vérité terrain.
        
        Args:
            explanation: L'explication à évaluer
            ground_truth: Vérité terrain
            
        Returns:
            Score de fidélité entre 0 et 1
        """
        # Dans une implémentation réelle, cette méthode pourrait:
        # 1. Comparer les attributions de features avec la vérité terrain
        # 2. Vérifier la cohérence des prédictions du modèle et de l'explication
        # 3. Évaluer la précision des contrefactuels
        
        # Pour l'exemple, nous retournons une valeur simulée
        fidelity = 0.8  # Valeur par défaut
        
        # Si des métadonnées de fidélité sont disponibles
        if hasattr(explanation.metadata, "fidelity_score"):
            fidelity = explanation.metadata.fidelity_score
        
        # Normaliser entre 0 et 1
        return min(1.0, max(0.0, fidelity))
    
    def measure_user_satisfaction(self, user_feedback: Dict[str, Any]) -> float:
        """
        Mesure la satisfaction de l'utilisateur à partir de son feedback.
        
        Args:
            user_feedback: Feedback utilisateur
            
        Returns:
            Score de satisfaction entre 0 et 1
        """
        # Exemple simple de calcul de score
        satisfaction = 0.5  # Score neutre par défaut
        
        if "rating" in user_feedback:
            # Normaliser une note sur 5 ou 10 à une échelle de 0 à 1
            rating = user_feedback["rating"]
            if isinstance(rating, (int, float)):
                max_rating = 5.0  # Par défaut, supposer une échelle de 5
                if rating > 5:
                    max_rating = 10.0  # Ajuster pour une échelle de 10
                
                satisfaction = min(1.0, max(0.0, rating / max_rating))
        
        if "helpful" in user_feedback:
            helpful = user_feedback["helpful"]
            if isinstance(helpful, bool):
                satisfaction = 0.8 if helpful else 0.2
        
        if "clarity" in user_feedback and "relevance" in user_feedback:
            # Moyenne pondérée de la clarté et de la pertinence
            clarity = min(1.0, max(0.0, user_feedback["clarity"]))
            relevance = min(1.0, max(0.0, user_feedback["relevance"]))
            satisfaction = 0.6 * clarity + 0.4 * relevance
        
        return satisfaction
    
    def compare_explanations(self, 
                           explanations: List[ExplanationResult],
                           ground_truth: Optional[Any] = None) -> Dict[str, List[float]]:
        """
        Compare plusieurs explications selon diverses métriques.
        
        Args:
            explanations: Liste d'explications à comparer
            ground_truth: Vérité terrain (optionnel)
            
        Returns:
            Dictionnaire des scores de métriques pour chaque explication
        """
        comparison = {
            "complexity_scores": [],
            "conciseness_scores": [],
            "coherence_scores": [],
            "overall_scores": []
        }
        
        if ground_truth is not None:
            comparison["fidelity_scores"] = []
        
        # Évaluer chaque explication
        for explanation in explanations:
            metrics = self.evaluate_explanation(explanation, ground_truth)
            
            comparison["complexity_scores"].append(metrics.get("complexity_score", 0.0))
            comparison["conciseness_scores"].append(metrics.get("conciseness_score", 0.0))
            comparison["coherence_scores"].append(metrics.get("coherence_score", 0.0))
            comparison["overall_scores"].append(metrics.get("overall_score", 0.0))
            
            if ground_truth is not None:
                comparison["fidelity_scores"].append(metrics.get("fidelity_score", 0.0))
        
        return comparison
    
    def find_optimal_explanation(self, 
                               explanations: List[ExplanationResult],
                               ground_truth: Optional[Any] = None,
                               criteria: Optional[Dict[str, float]] = None) -> Tuple[int, float]:
        """
        Trouve l'explication optimale selon des critères pondérés.
        
        Args:
            explanations: Liste d'explications à comparer
            ground_truth: Vérité terrain (optionnel)
            criteria: Critères pondérés (optionnel)
            
        Returns:
            Tuple (indice de la meilleure explication, score)
        """
        if not explanations:
            return (-1, 0.0)
        
        # Critères par défaut
        default_criteria = {
            "complexity_score": 0.2,
            "conciseness_score": 0.3,
            "coherence_score": 0.3,
            "fidelity_score": 0.2
        }
        
        # Utiliser les critères fournis ou les critères par défaut
        criteria = criteria or default_criteria
        
        # Comparer les explications
        comparison = self.compare_explanations(explanations, ground_truth)
        
        # Calculer les scores pondérés
        weighted_scores = []
        for i in range(len(explanations)):
            score = 0.0
            total_weight = 0.0
            
            for metric, weight in criteria.items():
                if metric in comparison and i < len(comparison[metric]):
                    score += weight * comparison[metric][i]
                    total_weight += weight
            
            if total_weight > 0:
                weighted_scores.append(score / total_weight)
            else:
                weighted_scores.append(0.0)
        
        # Trouver l'indice de l'explication avec le meilleur score
        if weighted_scores:
            best_index = np.argmax(weighted_scores)
            return (best_index, weighted_scores[best_index])
        
        return (0, 0.0)  # Par défaut, retourner la première explication
