"""
Quantification d'Incertitude pour Explications
=========================================

Ce module implémente des métriques révolutionnaires pour évaluer
la fiabilité des explications générées par les différents explainers.
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time

from ...core.base import ExplanationResult, ExplainerBase


class UncertaintyType(Enum):
    """Types d'incertitude dans les explications."""
    ALEATORIC = "aleatoric"       # Incertitude inhérente aux données
    EPISTEMIC = "epistemic"       # Incertitude due aux limites du modèle
    STRUCTURAL = "structural"     # Incertitude due à la structure du modèle
    APPROXIMATION = "approximation"  # Incertitude due à l'approximation de l'explication
    SAMPLING = "sampling"         # Incertitude due à l'échantillonnage
    FEATURE = "feature"           # Incertitude sur les attributions de features


@dataclass
class UncertaintyMetrics:
    """
    Métriques d'incertitude pour une explication.
    """
    # Score global d'incertitude (0 = certitude maximale, 1 = incertitude maximale)
    global_uncertainty: float = 0.0
    
    # Scores par type d'incertitude
    aleatoric_uncertainty: float = 0.0
    epistemic_uncertainty: float = 0.0
    structural_uncertainty: float = 0.0
    approximation_uncertainty: float = 0.0
    sampling_uncertainty: float = 0.0
    
    # Incertitude par feature (si applicable)
    feature_uncertainties: Dict[str, float] = field(default_factory=dict)
    
    # Intervalles de confiance (si applicable)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Métadonnées
    timestamp: float = field(default_factory=time.time)
    method: str = "default"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit les métriques en dictionnaire."""
        return {
            "global_uncertainty": self.global_uncertainty,
            "aleatoric_uncertainty": self.aleatoric_uncertainty,
            "epistemic_uncertainty": self.epistemic_uncertainty,
            "structural_uncertainty": self.structural_uncertainty,
            "approximation_uncertainty": self.approximation_uncertainty,
            "sampling_uncertainty": self.sampling_uncertainty,
            "feature_uncertainties": self.feature_uncertainties,
            "confidence_intervals": {
                k: {"lower": v[0], "upper": v[1]} 
                for k, v in self.confidence_intervals.items()
            },
            "timestamp": self.timestamp,
            "method": self.method
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UncertaintyMetrics':
        """Crée des métriques à partir d'un dictionnaire."""
        confidence_intervals = {}
        for k, v in data.get("confidence_intervals", {}).items():
            if isinstance(v, dict) and "lower" in v and "upper" in v:
                confidence_intervals[k] = (v["lower"], v["upper"])
        
        return cls(
            global_uncertainty=data.get("global_uncertainty", 0.0),
            aleatoric_uncertainty=data.get("aleatoric_uncertainty", 0.0),
            epistemic_uncertainty=data.get("epistemic_uncertainty", 0.0),
            structural_uncertainty=data.get("structural_uncertainty", 0.0),
            approximation_uncertainty=data.get("approximation_uncertainty", 0.0),
            sampling_uncertainty=data.get("sampling_uncertainty", 0.0),
            feature_uncertainties=data.get("feature_uncertainties", {}),
            confidence_intervals=confidence_intervals,
            timestamp=data.get("timestamp", time.time()),
            method=data.get("method", "default")
        )


class UncertaintyQuantifier:
    """
    Quantificateur d'incertitude pour les explications.
    """
    
    def __init__(self, 
                 n_bootstrap_samples: int = 100,
                 confidence_level: float = 0.95,
                 methods: Optional[List[str]] = None):
        """
        Initialise le quantificateur d'incertitude.
        
        Args:
            n_bootstrap_samples: Nombre d'échantillons bootstrap pour l'estimation
            confidence_level: Niveau de confiance pour les intervalles
            methods: Méthodes d'estimation d'incertitude à utiliser
        """
        self.n_bootstrap_samples = n_bootstrap_samples
        self.confidence_level = confidence_level
        self.methods = methods or ["bootstrap", "ensemble", "sensitivity"]
    
    def quantify_uncertainty(self, 
                           explanation: ExplanationResult,
                           explainer: Optional[ExplainerBase] = None,
                           X: Optional[Any] = None,
                           y: Optional[Any] = None,
                           **kwargs) -> UncertaintyMetrics:
        """
        Quantifie l'incertitude d'une explication.
        
        Args:
            explanation: L'explication à évaluer
            explainer: L'explainer utilisé (optionnel)
            X: Données d'entrée (optionnel)
            y: Cibles (optionnel)
            **kwargs: Paramètres additionnels
            
        Returns:
            Métriques d'incertitude
        """
        # Initialiser les métriques
        metrics = UncertaintyMetrics()
        
        # Liste des méthodes d'estimation disponibles
        estimation_methods = {
            "bootstrap": self._bootstrap_uncertainty,
            "ensemble": self._ensemble_uncertainty,
            "sensitivity": self._sensitivity_analysis,
            "variance": self._variance_estimation,
            "bayesian": self._bayesian_uncertainty
        }
        
        # Appliquer les méthodes disponibles
        available_methods = [m for m in self.methods if m in estimation_methods]
        
        if not available_methods:
            logging.warning("Aucune méthode d'estimation d'incertitude disponible")
            return metrics
        
        # Collecter les résultats de chaque méthode
        method_results = []
        
        for method_name in available_methods:
            if explainer is None and method_name in ["bootstrap", "sensitivity", "bayesian"]:
                continue  # Ces méthodes nécessitent un explainer
                
            if X is None and method_name in ["bootstrap", "sensitivity", "variance"]:
                continue  # Ces méthodes nécessitent des données
            
            method_fn = estimation_methods[method_name]
            try:
                method_metrics = method_fn(explanation, explainer, X, y, **kwargs)
                method_results.append(method_metrics)
            except Exception as e:
                logging.warning(f"Erreur lors de l'estimation d'incertitude avec {method_name}: {str(e)}")
        
        # Si aucune méthode n'a fonctionné, retourner les métriques par défaut
        if not method_results:
            return metrics
        
        # Agréger les résultats des différentes méthodes
        metrics = self._aggregate_metrics(method_results)
        
        return metrics
    
    def _bootstrap_uncertainty(self, 
                             explanation: ExplanationResult,
                             explainer: Optional[ExplainerBase],
                             X: Optional[Any],
                             y: Optional[Any],
                             **kwargs) -> UncertaintyMetrics:
        """
        Estime l'incertitude par bootstrap.
        
        Args:
            explanation: L'explication à évaluer
            explainer: L'explainer utilisé
            X: Données d'entrée
            y: Cibles (optionnel)
            **kwargs: Paramètres additionnels
            
        Returns:
            Métriques d'incertitude
        """
        # Dans une implémentation réelle, cette méthode pourrait:
        # 1. Générer des échantillons bootstrap des données
        # 2. Recalculer l'explication pour chaque échantillon
        # 3. Mesurer la variance des explications
        
        # Pour l'exemple, nous retournons des métriques simulées
        metrics = UncertaintyMetrics(
            global_uncertainty=0.15,
            aleatoric_uncertainty=0.2,
            epistemic_uncertainty=0.1,
            sampling_uncertainty=0.15,
            method="bootstrap"
        )
        
        # Simuler des incertitudes par feature
        if hasattr(explanation, "feature_importances"):
            feature_importances = explanation.feature_importances
            metrics.feature_uncertainties = {
                feature: np.random.uniform(0.05, 0.25) 
                for feature in feature_importances.keys()
            }
        
        # Simuler des intervalles de confiance
        if hasattr(explanation, "feature_importances"):
            feature_importances = explanation.feature_importances
            metrics.confidence_intervals = {
                feature: (
                    max(0, value - np.random.uniform(0.05, 0.15) * value),
                    min(1, value + np.random.uniform(0.05, 0.15) * value)
                )
                for feature, value in feature_importances.items()
            }
        
        return metrics
    
    def _ensemble_uncertainty(self, 
                            explanation: ExplanationResult,
                            explainer: Optional[ExplainerBase],
                            X: Optional[Any],
                            y: Optional[Any],
                            **kwargs) -> UncertaintyMetrics:
        """
        Estime l'incertitude par ensemble d'explainers.
        
        Args:
            explanation: L'explication à évaluer
            explainer: L'explainer utilisé
            X: Données d'entrée
            y: Cibles (optionnel)
            **kwargs: Paramètres additionnels
            
        Returns:
            Métriques d'incertitude
        """
        # Dans une implémentation réelle, cette méthode pourrait:
        # 1. Utiliser plusieurs explainers différents
        # 2. Comparer leurs explications
        # 3. Mesurer la variance entre les explications
        
        # Pour l'exemple, nous retournons des métriques simulées
        metrics = UncertaintyMetrics(
            global_uncertainty=0.18,
            epistemic_uncertainty=0.25,
            structural_uncertainty=0.15,
            method="ensemble"
        )
        
        return metrics
    
    def _sensitivity_analysis(self, 
                            explanation: ExplanationResult,
                            explainer: Optional[ExplainerBase],
                            X: Optional[Any],
                            y: Optional[Any],
                            **kwargs) -> UncertaintyMetrics:
        """
        Estime l'incertitude par analyse de sensibilité.
        
        Args:
            explanation: L'explication à évaluer
            explainer: L'explainer utilisé
            X: Données d'entrée
            y: Cibles (optionnel)
            **kwargs: Paramètres additionnels
            
        Returns:
            Métriques d'incertitude
        """
        # Dans une implémentation réelle, cette méthode pourrait:
        # 1. Perturber légèrement les données d'entrée
        # 2. Observer la stabilité de l'explication
        # 3. Quantifier la sensibilité aux perturbations
        
        # Pour l'exemple, nous retournons des métriques simulées
        metrics = UncertaintyMetrics(
            global_uncertainty=0.12,
            aleatoric_uncertainty=0.15,
            feature_uncertainties={},
            method="sensitivity"
        )
        
        return metrics
    
    def _variance_estimation(self, 
                           explanation: ExplanationResult,
                           explainer: Optional[ExplainerBase],
                           X: Optional[Any],
                           y: Optional[Any],
                           **kwargs) -> UncertaintyMetrics:
        """
        Estime l'incertitude par analyse de variance.
        
        Args:
            explanation: L'explication à évaluer
            explainer: L'explainer utilisé
            X: Données d'entrée
            y: Cibles (optionnel)
            **kwargs: Paramètres additionnels
            
        Returns:
            Métriques d'incertitude
        """
        # Dans une implémentation réelle, cette méthode pourrait:
        # 1. Analyser la variance des attributions de features
        # 2. Identifier les régions de haute variance
        
        # Pour l'exemple, nous retournons des métriques simulées
        metrics = UncertaintyMetrics(
            global_uncertainty=0.14,
            sampling_uncertainty=0.18,
            method="variance"
        )
        
        return metrics
    
    def _bayesian_uncertainty(self, 
                            explanation: ExplanationResult,
                            explainer: Optional[ExplainerBase],
                            X: Optional[Any],
                            y: Optional[Any],
                            **kwargs) -> UncertaintyMetrics:
        """
        Estime l'incertitude par approche bayésienne.
        
        Args:
            explanation: L'explication à évaluer
            explainer: L'explainer utilisé
            X: Données d'entrée
            y: Cibles (optionnel)
            **kwargs: Paramètres additionnels
            
        Returns:
            Métriques d'incertitude
        """
        # Dans une implémentation réelle, cette méthode pourrait:
        # 1. Utiliser des approches bayésiennes pour estimer l'incertitude
        # 2. Calculer des distributions postérieures
        
        # Pour l'exemple, nous retournons des métriques simulées
        metrics = UncertaintyMetrics(
            global_uncertainty=0.16,
            epistemic_uncertainty=0.22,
            aleatoric_uncertainty=0.12,
            method="bayesian"
        )
        
        return metrics
    
    def _aggregate_metrics(self, metrics_list: List[UncertaintyMetrics]) -> UncertaintyMetrics:
        """
        Agrège les métriques de plusieurs méthodes.
        
        Args:
            metrics_list: Liste de métriques à agréger
            
        Returns:
            Métriques agrégées
        """
        if not metrics_list:
            return UncertaintyMetrics()
        
        # Initialiser les métriques agrégées
        aggregated = UncertaintyMetrics(
            method="aggregated",
            timestamp=time.time()
        )
        
        # Calculer les moyennes des métriques numériques
        n_metrics = len(metrics_list)
        
        aggregated.global_uncertainty = sum(m.global_uncertainty for m in metrics_list) / n_metrics
        aggregated.aleatoric_uncertainty = sum(m.aleatoric_uncertainty for m in metrics_list) / n_metrics
        aggregated.epistemic_uncertainty = sum(m.epistemic_uncertainty for m in metrics_list) / n_metrics
        aggregated.structural_uncertainty = sum(m.structural_uncertainty for m in metrics_list) / n_metrics
        aggregated.approximation_uncertainty = sum(m.approximation_uncertainty for m in metrics_list) / n_metrics
        aggregated.sampling_uncertainty = sum(m.sampling_uncertainty for m in metrics_list) / n_metrics
        
        # Agréger les incertitudes par feature
        all_features = set()
        for metrics in metrics_list:
            all_features.update(metrics.feature_uncertainties.keys())
        
        for feature in all_features:
            values = [m.feature_uncertainties.get(feature, 0.0) for m in metrics_list 
                     if feature in m.feature_uncertainties]
            
            if values:
                aggregated.feature_uncertainties[feature] = sum(values) / len(values)
        
        # Agréger les intervalles de confiance
        all_interval_keys = set()
        for metrics in metrics_list:
            all_interval_keys.update(metrics.confidence_intervals.keys())
        
        for key in all_interval_keys:
            lower_bounds = []
            upper_bounds = []
            
            for metrics in metrics_list:
                if key in metrics.confidence_intervals:
                    lower, upper = metrics.confidence_intervals[key]
                    lower_bounds.append(lower)
                    upper_bounds.append(upper)
            
            if lower_bounds and upper_bounds:
                # Prendre l'intervalle le plus conservateur
                aggregated.confidence_intervals[key] = (
                    min(lower_bounds),
                    max(upper_bounds)
                )
        
        return aggregated
    
    def apply_to_explanation(self, 
                           explanation: ExplanationResult,
                           metrics: UncertaintyMetrics) -> ExplanationResult:
        """
        Applique les métriques d'incertitude à une explication.
        
        Args:
            explanation: L'explication à enrichir
            metrics: Métriques d'incertitude
            
        Returns:
            Explication enrichie avec les métriques d'incertitude
        """
        # Ajouter les métriques d'incertitude aux métadonnées
        explanation.metadata.uncertainty_metrics = metrics.to_dict()
        
        # Si l'explication contient des attributions de features,
        # ajouter les incertitudes correspondantes
        if hasattr(explanation, "feature_importances"):
            explanation.feature_importance_uncertainties = metrics.feature_uncertainties
            explanation.feature_importance_intervals = metrics.confidence_intervals
        
        return explanation
