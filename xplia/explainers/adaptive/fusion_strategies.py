"""
Stratégies de Fusion d'Explications
=================================

Ce module implémente différentes stratégies pour fusionner les résultats
de plusieurs explainers afin d'obtenir une explication plus complète,
robuste et fiable.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from abc import ABC, abstractmethod

from ...core.base import ExplanationResult, AudienceLevel


class ExplanationFusionStrategy(ABC):
    """
    Classe abstraite pour les stratégies de fusion d'explications.
    """
    
    @abstractmethod
    def fuse_explanations(self,
                        explanations: List[ExplanationResult],
                        qualities: List[float],
                        model: Any,
                        audience_level: AudienceLevel = AudienceLevel.TECHNICAL,
                        **kwargs) -> ExplanationResult:
        """
        Fusionne plusieurs explications en une seule.
        
        Args:
            explanations: Liste des explications à fusionner
            qualities: Liste des scores de qualité correspondants
            model: Le modèle expliqué
            audience_level: Niveau d'audience ciblé
            **kwargs: Paramètres additionnels
            
        Returns:
            Explication fusionnée
        """
        pass
    
    @staticmethod
    def create(strategy_name: str, **kwargs) -> 'ExplanationFusionStrategy':
        """
        Crée une instance de stratégie de fusion selon le nom spécifié.
        
        Args:
            strategy_name: Nom de la stratégie
            **kwargs: Paramètres spécifiques à la stratégie
            
        Returns:
            Instance de stratégie de fusion
        """
        strategies = {
            "weighted_ensemble": WeightedEnsembleFusionStrategy,
            "hierarchical": HierarchicalFusionStrategy,
            "best_quality": BestQualityFusionStrategy,
            "complementary": ComplementaryFusionStrategy,
            "modality_specific": ModalitySpecificFusionStrategy
        }
        
        if strategy_name not in strategies:
            logging.warning(f"Stratégie de fusion '{strategy_name}' inconnue, utilisation par défaut.")
            return WeightedEnsembleFusionStrategy(**kwargs)
        
        return strategies[strategy_name](**kwargs)


class WeightedEnsembleFusionStrategy(ExplanationFusionStrategy):
    """
    Stratégie de fusion basée sur un ensemble pondéré des explications.
    
    Cette stratégie fusionne les explications en les pondérant par leur
    score de qualité, privilégiant ainsi les explications les plus fiables.
    """
    
    def __init__(self, min_quality_threshold: float = 0.3, **kwargs):
        """
        Initialise la stratégie de fusion par ensemble pondéré.
        
        Args:
            min_quality_threshold: Seuil minimal de qualité pour inclure une explication
            **kwargs: Paramètres additionnels
        """
        self.min_quality_threshold = min_quality_threshold
    
    def fuse_explanations(self,
                        explanations: List[ExplanationResult],
                        qualities: List[float],
                        model: Any,
                        audience_level: AudienceLevel = AudienceLevel.TECHNICAL,
                        **kwargs) -> ExplanationResult:
        """
        Fusionne plusieurs explications en utilisant un ensemble pondéré.
        
        Args:
            explanations: Liste des explications à fusionner
            qualities: Liste des scores de qualité correspondants
            model: Le modèle expliqué
            audience_level: Niveau d'audience ciblé
            **kwargs: Paramètres additionnels
            
        Returns:
            Explication fusionnée
        """
        if not explanations:
            raise ValueError("Aucune explication à fusionner")
        
        if len(explanations) == 1:
            return explanations[0]
        
        # Filtrer les explications de qualité insuffisante
        valid_pairs = [
            (exp, qual) for exp, qual in zip(explanations, qualities)
            if qual >= self.min_quality_threshold
        ]
        
        if not valid_pairs:
            # Si aucune explication ne dépasse le seuil, prendre la meilleure
            best_idx = np.argmax(qualities)
            return explanations[best_idx]
        
        valid_explanations, valid_qualities = zip(*valid_pairs)
        
        # Normaliser les poids
        total_quality = sum(valid_qualities)
        if total_quality == 0:
            weights = [1.0 / len(valid_qualities)] * len(valid_qualities)
        else:
            weights = [q / total_quality for q in valid_qualities]
        
        # Fusionner les explications pondérées
        # (Dans une implémentation réelle, cette fusion dépendrait du type d'explication)
        base_explanation = self._get_best_base_explanation(valid_explanations, valid_qualities)
        fused_explanation = self._combine_explanations(
            base_explanation, valid_explanations, weights, audience_level
        )
        
        return fused_explanation
    
    def _get_best_base_explanation(self, 
                                 explanations: List[ExplanationResult],
                                 qualities: List[float]) -> ExplanationResult:
        """
        Sélectionne la meilleure explication comme base pour la fusion.
        
        Args:
            explanations: Liste des explications
            qualities: Liste des scores de qualité
            
        Returns:
            Meilleure explication de base
        """
        best_idx = np.argmax(qualities)
        return explanations[best_idx]
    
    def _combine_explanations(self,
                            base: ExplanationResult,
                            explanations: List[ExplanationResult],
                            weights: List[float],
                            audience_level: AudienceLevel) -> ExplanationResult:
        """
        Combine plusieurs explications en une seule.
        
        Args:
            base: Explication de base à enrichir
            explanations: Liste des explications à combiner
            weights: Poids des explications
            audience_level: Niveau d'audience ciblé
            
        Returns:
            Explication combinée
        """
        # Dans une implémentation réelle, cette méthode serait plus sophistiquée
        # et adapterait la fusion selon le type d'explication et de données
        
        # Pour l'exemple, nous retournons simplement l'explication de base
        # enrichie de métadonnées sur la fusion
        base.metadata.fusion_info = {
            "strategy": "weighted_ensemble",
            "num_explanations": len(explanations),
            "weights": weights,
            "audience_level": str(audience_level)
        }
        
        return base


class BestQualityFusionStrategy(ExplanationFusionStrategy):
    """
    Stratégie de fusion qui sélectionne simplement la meilleure explication.
    """
    
    def fuse_explanations(self,
                        explanations: List[ExplanationResult],
                        qualities: List[float],
                        model: Any,
                        audience_level: AudienceLevel = AudienceLevel.TECHNICAL,
                        **kwargs) -> ExplanationResult:
        """
        Sélectionne l'explication de meilleure qualité.
        
        Args:
            explanations: Liste des explications à évaluer
            qualities: Liste des scores de qualité correspondants
            model: Le modèle expliqué
            audience_level: Niveau d'audience ciblé
            **kwargs: Paramètres additionnels
            
        Returns:
            Meilleure explication
        """
        if not explanations:
            raise ValueError("Aucune explication à fusionner")
        
        best_idx = np.argmax(qualities)
        best_explanation = explanations[best_idx]
        
        # Ajouter des métadonnées sur la sélection
        best_explanation.metadata.fusion_info = {
            "strategy": "best_quality",
            "num_candidates": len(explanations),
            "selected_quality": qualities[best_idx],
            "audience_level": str(audience_level)
        }
        
        return best_explanation


# Classes supplémentaires à implémenter dans une version future
class HierarchicalFusionStrategy(ExplanationFusionStrategy):
    """
    Stratégie de fusion hiérarchique qui organise les explications en niveaux.
    """
    
    def fuse_explanations(self,
                        explanations: List[ExplanationResult],
                        qualities: List[float],
                        model: Any,
                        audience_level: AudienceLevel = AudienceLevel.TECHNICAL,
                        **kwargs) -> ExplanationResult:
        """
        Fusionne les explications de manière hiérarchique.
        """
        # Implémentation à venir
        # Pour l'instant, utiliser la stratégie par défaut
        fallback = WeightedEnsembleFusionStrategy()
        return fallback.fuse_explanations(
            explanations, qualities, model, audience_level, **kwargs
        )


class ComplementaryFusionStrategy(ExplanationFusionStrategy):
    """
    Stratégie de fusion qui sélectionne des explications complémentaires.
    """
    
    def fuse_explanations(self,
                        explanations: List[ExplanationResult],
                        qualities: List[float],
                        model: Any,
                        audience_level: AudienceLevel = AudienceLevel.TECHNICAL,
                        **kwargs) -> ExplanationResult:
        """
        Fusionne les explications en privilégiant la complémentarité.
        """
        # Implémentation à venir
        # Pour l'instant, utiliser la stratégie par défaut
        fallback = WeightedEnsembleFusionStrategy()
        return fallback.fuse_explanations(
            explanations, qualities, model, audience_level, **kwargs
        )


class ModalitySpecificFusionStrategy(ExplanationFusionStrategy):
    """
    Stratégie de fusion adaptée aux explications multimodales.
    """
    
    def fuse_explanations(self,
                        explanations: List[ExplanationResult],
                        qualities: List[float],
                        model: Any,
                        audience_level: AudienceLevel = AudienceLevel.TECHNICAL,
                        **kwargs) -> ExplanationResult:
        """
        Fusionne les explications en tenant compte des spécificités des modalités.
        """
        # Implémentation à venir
        # Pour l'instant, utiliser la stratégie par défaut
        fallback = WeightedEnsembleFusionStrategy()
        return fallback.fuse_explanations(
            explanations, qualities, model, audience_level, **kwargs
        )
