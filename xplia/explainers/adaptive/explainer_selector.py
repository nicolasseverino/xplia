"""
Sélecteur d'Explainers Adaptatif
===============================

Ce module implémente un système intelligent de sélection d'explainers
qui choisit les techniques d'explication les plus appropriées en fonction
du contexte, du modèle, des données et des besoins de l'utilisateur.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Set, Type
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from ...core.base import ExplainerBase, AudienceLevel


class ExplainerProfile:
    """
    Profil d'un explainer décrivant ses capacités et domaines d'expertise.
    """
    
    def __init__(self, 
                 explainer: ExplainerBase,
                 strengths: Dict[str, float] = None,
                 weaknesses: Dict[str, float] = None,
                 supported_data_types: Set[str] = None,
                 supported_model_types: Set[str] = None,
                 min_samples: int = 0,
                 max_features: Optional[int] = None,
                 computational_cost: float = 0.5,  # 0-1 scale
                 interpretability: float = 0.5,    # 0-1 scale
                 robustness: float = 0.5           # 0-1 scale
                 ):
        """
        Initialise un profil d'explainer.
        
        Args:
            explainer: L'instance de l'explainer
            strengths: Dictionnaire des points forts avec scores (0-1)
            weaknesses: Dictionnaire des points faibles avec scores (0-1)
            supported_data_types: Ensemble des types de données supportés
            supported_model_types: Ensemble des types de modèles supportés
            min_samples: Nombre minimum d'échantillons requis
            max_features: Nombre maximum de features supporté
            computational_cost: Coût computationnel (0=faible, 1=élevé)
            interpretability: Niveau d'interprétabilité (0=faible, 1=élevé)
            robustness: Niveau de robustesse (0=faible, 1=élevé)
        """
        self.explainer = explainer
        self.strengths = strengths or {}
        self.weaknesses = weaknesses or {}
        self.supported_data_types = supported_data_types or {"tabular"}
        self.supported_model_types = supported_model_types or {"any"}
        self.min_samples = min_samples
        self.max_features = max_features
        self.computational_cost = computational_cost
        self.interpretability = interpretability
        self.robustness = robustness
        
    def compatibility_score(self, context: Dict[str, Any]) -> float:
        """
        Calcule un score de compatibilité entre l'explainer et le contexte.
        
        Args:
            context: Dictionnaire décrivant le contexte d'explication
            
        Returns:
            Score de compatibilité entre 0 et 1
        """
        score = 1.0
        
        # Vérifier la compatibilité avec le type de données
        data_type = context.get("data_type", "unknown")
        if data_type not in self.supported_data_types and "any" not in self.supported_data_types:
            return 0.0  # Incompatible
        
        # Vérifier la compatibilité avec le type de modèle
        model_type = context.get("model_type", "unknown")
        if (model_type not in self.supported_model_types and 
            "any" not in self.supported_model_types):
            return 0.0  # Incompatible
        
        # Vérifier les contraintes de taille
        feature_count = context.get("feature_count", 0)
        if self.max_features and feature_count > self.max_features:
            return 0.0  # Trop de features
        
        # Ajuster le score selon les points forts et les caractéristiques du contexte
        for strength, value in self.strengths.items():
            if strength in context:
                score += value * 0.2  # Bonus pour les points forts pertinents
        
        # Pénaliser pour les points faibles pertinents
        for weakness, value in self.weaknesses.items():
            if weakness in context:
                score -= value * 0.2  # Malus pour les points faibles pertinents
        
        # Ajuster selon le niveau d'audience
        audience_level = context.get("audience_level", AudienceLevel.TECHNICAL)
        if audience_level == AudienceLevel.TECHNICAL and self.interpretability < 0.3:
            score -= 0.2  # Pénalité pour explications techniques peu interprétables
        elif audience_level == AudienceLevel.NON_TECHNICAL and self.interpretability < 0.7:
            score -= 0.3  # Pénalité plus importante pour un public non technique
        
        # Normaliser le score final entre 0 et 1
        return max(0.0, min(1.0, score))


class ExplainerSelector:
    """
    Système de sélection intelligente d'explainers basé sur le contexte.
    """
    
    def __init__(self, 
                 available_explainers: List[ExplainerBase],
                 strategy: str = "auto",
                 max_explainers: int = 3,
                 **kwargs):
        """
        Initialise le sélecteur d'explainers.
        
        Args:
            available_explainers: Liste des explainers disponibles
            strategy: Stratégie de sélection ('auto', 'all', 'best_k')
            max_explainers: Nombre maximum d'explainers à sélectionner
            **kwargs: Paramètres additionnels
        """
        self.available_explainers = available_explainers
        self.strategy = strategy
        self.max_explainers = max_explainers
        
        # Créer les profils pour tous les explainers disponibles
        self.explainer_profiles = self._build_explainer_profiles()
        
        # Cache des sélections précédentes par signature de contexte
        self.selection_cache = {}
    
    def _build_explainer_profiles(self) -> Dict[ExplainerBase, ExplainerProfile]:
        """
        Construit les profils pour tous les explainers disponibles.
        
        Returns:
            Dictionnaire associant chaque explainer à son profil
        """
        profiles = {}
        
        for explainer in self.available_explainers:
            # Déterminer les caractéristiques du profil selon le type d'explainer
            profile = self._create_profile_for_explainer(explainer)
            profiles[explainer] = profile
        
        return profiles
    
    def _create_profile_for_explainer(self, explainer: ExplainerBase) -> ExplainerProfile:
        """
        Crée un profil pour un explainer spécifique.
        
        Args:
            explainer: L'explainer à profiler
            
        Returns:
            Profil de l'explainer
        """
        # Cette méthode serait implémentée avec une logique plus sophistiquée
        # pour détecter automatiquement les caractéristiques de chaque explainer
        
        # Pour l'instant, utilisons des valeurs par défaut basées sur le nom
        explainer_name = explainer.__class__.__name__.lower()
        
        if "shap" in explainer_name:
            return ExplainerProfile(
                explainer=explainer,
                strengths={"feature_importance": 0.9, "global_local": 0.8},
                weaknesses={"runtime": 0.7},
                supported_data_types={"tabular", "text", "image"},
                computational_cost=0.7,
                interpretability=0.8,
                robustness=0.7
            )
        elif "lime" in explainer_name:
            return ExplainerProfile(
                explainer=explainer,
                strengths={"local_explanation": 0.9, "intuitive": 0.8},
                weaknesses={"stability": 0.6},
                supported_data_types={"tabular", "text", "image"},
                computational_cost=0.5,
                interpretability=0.9,
                robustness=0.6
            )
        elif "counterfactual" in explainer_name:
            return ExplainerProfile(
                explainer=explainer,
                strengths={"actionable": 0.9, "intuitive": 0.9},
                weaknesses={"runtime": 0.8, "stability": 0.6},
                supported_data_types={"tabular"},
                computational_cost=0.8,
                interpretability=0.9,
                robustness=0.5
            )
        elif "gradient" in explainer_name:
            return ExplainerProfile(
                explainer=explainer,
                strengths={"deep_learning": 0.9, "efficiency": 0.7},
                weaknesses={"noise": 0.7},
                supported_data_types={"image", "text"},
                supported_model_types={"neural_network", "deep_learning"},
                computational_cost=0.4,
                interpretability=0.6,
                robustness=0.6
            )
        elif "attention" in explainer_name:
            return ExplainerProfile(
                explainer=explainer,
                strengths={"transformers": 0.9, "text": 0.9},
                weaknesses={"non_attention_models": 0.9},
                supported_data_types={"text", "image"},
                supported_model_types={"transformer", "attention_based"},
                computational_cost=0.3,
                interpretability=0.7,
                robustness=0.7
            )
        else:
            # Profil générique par défaut
            return ExplainerProfile(
                explainer=explainer,
                supported_data_types={"tabular", "text", "image"},
                computational_cost=0.5,
                interpretability=0.5,
                robustness=0.5
            )
    
    def select_explainers(self, 
                         context: Dict[str, Any],
                         model: Any,
                         data: Any,
                         **kwargs) -> List[ExplainerBase]:
        """
        Sélectionne les explainers les plus appropriés pour le contexte donné.
        
        Args:
            context: Dictionnaire décrivant le contexte d'explication
            model: Le modèle à expliquer
            data: Les données à expliquer
            **kwargs: Paramètres additionnels
            
        Returns:
            Liste des explainers sélectionnés
        """
        # Créer une signature du contexte pour le cache
        context_sig = self._get_context_signature(context)
        
        # Vérifier le cache
        if context_sig in self.selection_cache:
            return self.selection_cache[context_sig]
        
        # Appliquer la stratégie de sélection appropriée
        if self.strategy == "all":
            selected = self.available_explainers
        elif self.strategy == "best_k":
            selected = self._select_best_k(context, self.max_explainers)
        else:  # "auto" ou autre
            selected = self._select_auto(context, model, data, **kwargs)
        
        # Mettre en cache la sélection
        self.selection_cache[context_sig] = selected
        
        return selected
    
    def _select_best_k(self, 
                       context: Dict[str, Any],
                       k: int) -> List[ExplainerBase]:
        """
        Sélectionne les k meilleurs explainers selon leur score de compatibilité.
        
        Args:
            context: Dictionnaire décrivant le contexte d'explication
            k: Nombre d'explainers à sélectionner
            
        Returns:
            Liste des k explainers les plus compatibles
        """
        # Calculer les scores de compatibilité pour tous les explainers
        scores = []
        for explainer, profile in self.explainer_profiles.items():
            score = profile.compatibility_score(context)
            scores.append((explainer, score))
        
        # Trier par score décroissant et prendre les k premiers
        scores.sort(key=lambda x: x[1], reverse=True)
        return [explainer for explainer, _ in scores[:k]]
    
    def _select_auto(self, 
                    context: Dict[str, Any],
                    model: Any,
                    data: Any,
                    **kwargs) -> List[ExplainerBase]:
        """
        Sélectionne automatiquement les explainers les plus appropriés
        en utilisant une stratégie adaptative sophistiquée.
        
        Args:
            context: Dictionnaire décrivant le contexte d'explication
            model: Le modèle à expliquer
            data: Les données à expliquer
            **kwargs: Paramètres additionnels
            
        Returns:
            Liste des explainers sélectionnés automatiquement
        """
        # Calculer les scores de compatibilité pour tous les explainers
        scores = []
        for explainer, profile in self.explainer_profiles.items():
            score = profile.compatibility_score(context)
            
            # Appliquer des ajustements spécifiques au contexte
            if context.get("computational_budget") == "low" and profile.computational_cost > 0.7:
                score *= 0.5  # Réduire le score pour les explainers coûteux
            
            # Favoriser la diversité des approches d'explication
            explainer_type = self._get_explainer_type(explainer)
            scores.append((explainer, score, explainer_type))
        
        # Trier par score décroissant
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Sélectionner les meilleurs explainers tout en assurant la diversité
        selected = []
        selected_types = set()
        
        # D'abord inclure le meilleur explainer global
        for explainer, score, explainer_type in scores:
            if score > 0.5:  # Seuil minimal de compatibilité
                selected.append(explainer)
                selected_types.add(explainer_type)
                break
        
        # Ensuite ajouter les meilleurs de chaque type
        remaining_slots = min(self.max_explainers - 1, 2)  # Garder au moins 1-2 slots
        for explainer, score, explainer_type in scores:
            if len(selected) >= self.max_explainers:
                break
            if explainer not in selected and explainer_type not in selected_types and score > 0.3:
                selected.append(explainer)
                selected_types.add(explainer_type)
                remaining_slots -= 1
                if remaining_slots <= 0:
                    break
        
        # Compléter avec les meilleurs scores restants si nécessaire
        for explainer, score, _ in scores:
            if len(selected) >= self.max_explainers:
                break
            if explainer not in selected and score > 0.3:
                selected.append(explainer)
        
        return selected
    
    def _get_context_signature(self, context: Dict[str, Any]) -> str:
        """
        Génère une signature unique pour un contexte donné.
        
        Args:
            context: Dictionnaire du contexte
            
        Returns:
            Chaîne de signature unique
        """
        # Simplification pour l'exemple
        key_elements = []
        for key in sorted(context.keys()):
            value = context[key]
            if isinstance(value, (str, int, float, bool)):
                key_elements.append(f"{key}:{value}")
            elif isinstance(value, (list, tuple, set)):
                key_elements.append(f"{key}:{len(value)}")
        
        return ";".join(key_elements)
    
    def _get_explainer_type(self, explainer: ExplainerBase) -> str:
        """
        Détermine le type conceptuel d'un explainer.
        
        Args:
            explainer: L'explainer à analyser
            
        Returns:
            Type d'explainer ("feature_attribution", "counterfactual", etc.)
        """
        name = explainer.__class__.__name__.lower()
        
        if any(x in name for x in ["shap", "lime", "feature", "importance"]):
            return "feature_attribution"
        elif any(x in name for x in ["counterfactual", "example"]):
            return "counterfactual"
        elif any(x in name for x in ["anchor", "rule"]):
            return "rule_based"
        elif any(x in name for x in ["partial", "pdp", "ice"]):
            return "partial_dependence"
        elif any(x in name for x in ["gradient", "backprop"]):
            return "gradient_based"
        elif any(x in name for x in ["attention"]):
            return "attention_based"
        else:
            return "other"
