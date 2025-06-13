"""
Calibrateur d'Explication
======================

Ce module implémente un système d'auto-calibration des explications
qui ajuste automatiquement le niveau de détail, la complexité et
d'autres paramètres pour optimiser la qualité des explications.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass, field
import time

from ...core.base import ExplanationResult, ExplainerBase, AudienceLevel
from .audience_profiles import UserProfile
from .audience_adapter import AudienceAdapter
from .calibration_metrics import CalibrationMetrics


@dataclass
class CalibrationParameters:
    """
    Paramètres de calibration pour un explainer.
    """
    # Paramètres généraux
    detail_level: float = 0.5       # 0.0 (minimal) à 1.0 (maximal)
    complexity_level: float = 0.5   # 0.0 (simple) à 1.0 (complexe)
    visualization_ratio: float = 0.5  # 0.0 (texte uniquement) à 1.0 (visuel maximal)
    
    # Paramètres spécifiques à l'explainer
    specific_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit les paramètres en dictionnaire."""
        return {
            "detail_level": self.detail_level,
            "complexity_level": self.complexity_level,
            "visualization_ratio": self.visualization_ratio,
            **self.specific_params
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalibrationParameters':
        """Crée des paramètres à partir d'un dictionnaire."""
        specific_params = {k: v for k, v in data.items() 
                         if k not in ["detail_level", "complexity_level", "visualization_ratio"]}
        
        return cls(
            detail_level=data.get("detail_level", 0.5),
            complexity_level=data.get("complexity_level", 0.5),
            visualization_ratio=data.get("visualization_ratio", 0.5),
            specific_params=specific_params
        )


class ExplanationCalibrator:
    """
    Système d'auto-calibration des explications qui optimise
    les paramètres des explainers selon le contexte et le feedback.
    """
    
    def __init__(self, 
                 metrics: Optional[CalibrationMetrics] = None,
                 audience_adapter: Optional[AudienceAdapter] = None,
                 learning_rate: float = 0.1,
                 calibration_history_size: int = 100):
        """
        Initialise le calibrateur d'explication.
        
        Args:
            metrics: Métriques de calibration
            audience_adapter: Adaptateur d'audience
            learning_rate: Taux d'apprentissage pour les ajustements
            calibration_history_size: Taille de l'historique de calibration
        """
        self.metrics = metrics or CalibrationMetrics()
        self.audience_adapter = audience_adapter or AudienceAdapter()
        self.learning_rate = learning_rate
        self.calibration_history_size = calibration_history_size
        
        # Historique des calibrations par explainer et contexte
        self.calibration_history = []
        
        # Cache des paramètres optimaux par signature de contexte
        self.optimal_params_cache = {}
    
    def calibrate_explainer(self, 
                          explainer: ExplainerBase,
                          X: Any,
                          y: Optional[Any] = None,
                          user_profile: Optional[UserProfile] = None,
                          context: Optional[Dict[str, Any]] = None,
                          **kwargs) -> ExplainerBase:
        """
        Calibre un explainer selon le contexte et les données.
        
        Args:
            explainer: L'explainer à calibrer
            X: Données d'entrée
            y: Cibles (optionnel)
            user_profile: Profil utilisateur (optionnel)
            context: Contexte d'explication (optionnel)
            **kwargs: Paramètres additionnels
            
        Returns:
            Explainer calibré
        """
        # Créer une signature du contexte
        context_sig = self._get_context_signature(explainer, context, user_profile)
        
        # Vérifier le cache pour des paramètres optimaux
        if context_sig in self.optimal_params_cache:
            optimal_params = self.optimal_params_cache[context_sig]
            self._apply_parameters(explainer, optimal_params)
            return explainer
        
        # Déterminer les paramètres initiaux
        initial_params = self._get_initial_parameters(explainer, context, user_profile)
        
        # Si aucune donnée n'est fournie, utiliser les paramètres initiaux
        if X is None:
            self._apply_parameters(explainer, initial_params)
            return explainer
        
        # Optimiser les paramètres par exploration
        optimal_params = self._optimize_parameters(
            explainer, X, y, initial_params, context, user_profile, **kwargs
        )
        
        # Mettre en cache les paramètres optimaux
        self.optimal_params_cache[context_sig] = optimal_params
        
        # Appliquer les paramètres optimaux
        self._apply_parameters(explainer, optimal_params)
        
        return explainer
    
    def update_from_feedback(self, 
                           explanation: ExplanationResult,
                           feedback: Dict[str, Any],
                           explainer: ExplainerBase,
                           context: Optional[Dict[str, Any]] = None,
                           user_profile: Optional[UserProfile] = None) -> None:
        """
        Met à jour la calibration en fonction du feedback utilisateur.
        
        Args:
            explanation: L'explication évaluée
            feedback: Feedback utilisateur
            explainer: L'explainer utilisé
            context: Contexte d'explication
            user_profile: Profil utilisateur
        """
        # Extraire les paramètres utilisés pour cette explication
        if not hasattr(explanation.metadata, "calibration_params"):
            logging.warning("Impossible de mettre à jour la calibration: paramètres manquants")
            return
        
        params = explanation.metadata.calibration_params
        
        # Calculer un score de satisfaction basé sur le feedback
        satisfaction_score = self._calculate_satisfaction_score(feedback)
        
        # Créer une signature du contexte
        context_sig = self._get_context_signature(explainer, context, user_profile)
        
        # Mettre à jour l'historique de calibration
        self.calibration_history.append({
            "timestamp": time.time(),
            "explainer_type": explainer.__class__.__name__,
            "context_signature": context_sig,
            "parameters": params.to_dict() if isinstance(params, CalibrationParameters) else params,
            "satisfaction_score": satisfaction_score,
            "feedback": feedback
        })
        
        # Limiter la taille de l'historique
        if len(self.calibration_history) > self.calibration_history_size:
            self.calibration_history = self.calibration_history[-self.calibration_history_size:]
        
        # Mettre à jour les paramètres optimaux si nécessaire
        if context_sig in self.optimal_params_cache:
            current_optimal = self.optimal_params_cache[context_sig]
            
            # Si le feedback est meilleur que les paramètres optimaux actuels,
            # mettre à jour les paramètres optimaux
            if satisfaction_score > 0.7:  # Seuil arbitraire de "bon" feedback
                if isinstance(params, dict):
                    params = CalibrationParameters.from_dict(params)
                
                # Fusion pondérée des paramètres
                updated_params = self._blend_parameters(
                    current_optimal, params, weight=self.learning_rate
                )
                
                self.optimal_params_cache[context_sig] = updated_params
    
    def _get_initial_parameters(self, 
                              explainer: ExplainerBase,
                              context: Optional[Dict[str, Any]],
                              user_profile: Optional[UserProfile]) -> CalibrationParameters:
        """
        Détermine les paramètres initiaux pour un explainer.
        
        Args:
            explainer: L'explainer à calibrer
            context: Contexte d'explication
            user_profile: Profil utilisateur
            
        Returns:
            Paramètres initiaux
        """
        # Paramètres par défaut
        params = CalibrationParameters()
        
        # Ajuster selon le niveau d'audience
        audience_level = None
        if user_profile:
            audience_level = user_profile.audience_level
        elif context and "audience_level" in context:
            audience_level = context["audience_level"]
        
        if audience_level:
            if audience_level == AudienceLevel.NON_TECHNICAL:
                params.detail_level = 0.3
                params.complexity_level = 0.2
                params.visualization_ratio = 0.8
            elif audience_level == AudienceLevel.TECHNICAL:
                params.detail_level = 0.7
                params.complexity_level = 0.7
                params.visualization_ratio = 0.5
            elif audience_level == AudienceLevel.REGULATORY:
                params.detail_level = 0.9
                params.complexity_level = 0.6
                params.visualization_ratio = 0.4
        
        # Ajuster selon le type d'explainer
        explainer_name = explainer.__class__.__name__.lower()
        
        if "shap" in explainer_name:
            params.specific_params["n_samples"] = 1000
            params.specific_params["feature_perturbation"] = "interventional"
        elif "lime" in explainer_name:
            params.specific_params["num_features"] = 10
            params.specific_params["num_samples"] = 5000
        elif "counterfactual" in explainer_name:
            params.specific_params["proximity_weight"] = 0.5
            params.specific_params["sparsity_weight"] = 0.5
        elif "gradient" in explainer_name:
            params.specific_params["smooth_grad"] = True
            params.specific_params["smooth_samples"] = 50
        
        # Ajuster selon le contexte spécifique
        if context:
            if context.get("time_constraint") == "limited":
                params.detail_level = min(params.detail_level, 0.4)
                params.specific_params["max_features"] = 5
            
            if context.get("explanation_purpose") == "debugging":
                params.detail_level = 0.9
                params.complexity_level = 0.8
            
            if context.get("explanation_purpose") == "overview":
                params.detail_level = 0.4
                params.complexity_level = 0.3
        
        return params
    
    def _optimize_parameters(self, 
                           explainer: ExplainerBase,
                           X: Any,
                           y: Optional[Any],
                           initial_params: CalibrationParameters,
                           context: Optional[Dict[str, Any]],
                           user_profile: Optional[UserProfile],
                           **kwargs) -> CalibrationParameters:
        """
        Optimise les paramètres par exploration.
        
        Args:
            explainer: L'explainer à calibrer
            X: Données d'entrée
            y: Cibles (optionnel)
            initial_params: Paramètres initiaux
            context: Contexte d'explication
            user_profile: Profil utilisateur
            **kwargs: Paramètres additionnels
            
        Returns:
            Paramètres optimisés
        """
        # Dans une implémentation réelle, cette méthode pourrait:
        # 1. Explorer l'espace des paramètres autour des paramètres initiaux
        # 2. Évaluer la qualité des explications pour chaque jeu de paramètres
        # 3. Sélectionner les paramètres optimaux
        
        # Pour l'exemple, nous nous contentons de retourner les paramètres initiaux
        # avec une légère perturbation aléatoire pour simuler l'optimisation
        optimized_params = CalibrationParameters(
            detail_level=min(1.0, max(0.0, initial_params.detail_level + np.random.uniform(-0.1, 0.1))),
            complexity_level=min(1.0, max(0.0, initial_params.complexity_level + np.random.uniform(-0.1, 0.1))),
            visualization_ratio=min(1.0, max(0.0, initial_params.visualization_ratio + np.random.uniform(-0.1, 0.1))),
            specific_params=initial_params.specific_params.copy()
        )
        
        return optimized_params
    
    def _apply_parameters(self, 
                        explainer: ExplainerBase,
                        params: CalibrationParameters) -> None:
        """
        Applique les paramètres à un explainer.
        
        Args:
            explainer: L'explainer à configurer
            params: Paramètres à appliquer
        """
        # Appliquer les paramètres généraux
        if hasattr(explainer, "detail_level"):
            explainer.detail_level = params.detail_level
        
        if hasattr(explainer, "complexity_level"):
            explainer.complexity_level = params.complexity_level
        
        if hasattr(explainer, "visualization_ratio"):
            explainer.visualization_ratio = params.visualization_ratio
        
        # Appliquer les paramètres spécifiques
        for param_name, param_value in params.specific_params.items():
            if hasattr(explainer, param_name):
                setattr(explainer, param_name, param_value)
        
        # Stocker les paramètres de calibration dans l'explainer
        explainer.calibration_params = params
    
    def _get_context_signature(self, 
                             explainer: ExplainerBase,
                             context: Optional[Dict[str, Any]],
                             user_profile: Optional[UserProfile]) -> str:
        """
        Génère une signature unique pour un contexte d'explication.
        
        Args:
            explainer: L'explainer concerné
            context: Contexte d'explication
            user_profile: Profil utilisateur
            
        Returns:
            Signature du contexte
        """
        # Extraire les éléments clés du contexte
        key_elements = [explainer.__class__.__name__]
        
        if user_profile:
            key_elements.append(f"audience:{user_profile.audience_level.name}")
            key_elements.append(f"language:{user_profile.language}")
        
        if context:
            for key in sorted(context.keys()):
                value = context[key]
                if isinstance(value, (str, int, float, bool)):
                    key_elements.append(f"{key}:{value}")
        
        return "|".join(key_elements)
    
    def _calculate_satisfaction_score(self, feedback: Dict[str, Any]) -> float:
        """
        Calcule un score de satisfaction basé sur le feedback utilisateur.
        
        Args:
            feedback: Feedback utilisateur
            
        Returns:
            Score de satisfaction entre 0 et 1
        """
        # Exemple simple de calcul de score
        score = 0.5  # Score neutre par défaut
        
        if "rating" in feedback:
            # Normaliser une note sur 5 ou 10 à une échelle de 0 à 1
            rating = feedback["rating"]
            if isinstance(rating, (int, float)):
                max_rating = 5.0  # Par défaut, supposer une échelle de 5
                if rating > 5:
                    max_rating = 10.0  # Ajuster pour une échelle de 10
                
                score = min(1.0, max(0.0, rating / max_rating))
        
        if "helpful" in feedback:
            helpful = feedback["helpful"]
            if isinstance(helpful, bool):
                score = 0.8 if helpful else 0.2
        
        if "clarity" in feedback and "relevance" in feedback:
            # Moyenne pondérée de la clarté et de la pertinence
            clarity = min(1.0, max(0.0, feedback["clarity"]))
            relevance = min(1.0, max(0.0, feedback["relevance"]))
            score = 0.6 * clarity + 0.4 * relevance
        
        return score
    
    def _blend_parameters(self, 
                        params1: CalibrationParameters,
                        params2: CalibrationParameters,
                        weight: float = 0.5) -> CalibrationParameters:
        """
        Fusionne deux jeux de paramètres avec une pondération.
        
        Args:
            params1: Premier jeu de paramètres
            params2: Deuxième jeu de paramètres
            weight: Poids du deuxième jeu (entre 0 et 1)
            
        Returns:
            Paramètres fusionnés
        """
        # Limiter le poids entre 0 et 1
        weight = min(1.0, max(0.0, weight))
        
        # Fusionner les paramètres généraux
        blended_params = CalibrationParameters(
            detail_level=(1 - weight) * params1.detail_level + weight * params2.detail_level,
            complexity_level=(1 - weight) * params1.complexity_level + weight * params2.complexity_level,
            visualization_ratio=(1 - weight) * params1.visualization_ratio + weight * params2.visualization_ratio
        )
        
        # Fusionner les paramètres spécifiques
        # (uniquement ceux présents dans les deux jeux)
        for param_name, param1_value in params1.specific_params.items():
            if param_name in params2.specific_params:
                param2_value = params2.specific_params[param_name]
                
                # Fusionner uniquement les valeurs numériques
                if isinstance(param1_value, (int, float)) and isinstance(param2_value, (int, float)):
                    blended_value = (1 - weight) * param1_value + weight * param2_value
                    
                    # Arrondir si la valeur d'origine était un entier
                    if isinstance(param1_value, int) and isinstance(param2_value, int):
                        blended_value = int(round(blended_value))
                    
                    blended_params.specific_params[param_name] = blended_value
                else:
                    # Pour les valeurs non numériques, conserver la valeur d'origine
                    # ou la nouvelle selon le poids
                    if weight > 0.5:
                        blended_params.specific_params[param_name] = param2_value
                    else:
                        blended_params.specific_params[param_name] = param1_value
            else:
                # Conserver les paramètres uniques du premier jeu
                blended_params.specific_params[param_name] = param1_value
        
        # Ajouter les paramètres uniques du deuxième jeu
        for param_name, param2_value in params2.specific_params.items():
            if param_name not in params1.specific_params:
                blended_params.specific_params[param_name] = param2_value
        
        return blended_params
