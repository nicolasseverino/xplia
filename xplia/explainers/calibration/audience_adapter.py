"""
Adaptateur d'Audience pour Explications
====================================

Ce module implémente un système d'adaptation des explications
selon le profil de l'utilisateur et le contexte d'utilisation.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

from ...core.base import ExplanationResult, AudienceLevel
from .audience_profiles import UserProfile, ExplanationPreference, DomainExpertise


class AdaptationStrategy(Enum):
    """Stratégies d'adaptation des explications."""
    SIMPLIFY = "simplify"               # Simplifier l'explication
    ENRICH = "enrich"                   # Enrichir avec plus de détails
    VISUALIZE = "visualize"             # Ajouter des éléments visuels
    CONTEXTUALIZE = "contextualize"     # Ajouter du contexte spécifique au domaine
    SUMMARIZE = "summarize"             # Résumer l'explication
    TRANSLATE = "translate"             # Traduire dans une autre langue
    INTERACTIVE = "interactive"         # Rendre l'explication interactive


class AudienceAdapter:
    """
    Adaptateur d'explications selon le profil de l'audience.
    """
    
    def __init__(self, 
                 default_audience_level: AudienceLevel = AudienceLevel.TECHNICAL,
                 default_language: str = "fr",
                 adaptation_strategies: Optional[List[AdaptationStrategy]] = None):
        """
        Initialise l'adaptateur d'audience.
        
        Args:
            default_audience_level: Niveau d'audience par défaut
            default_language: Langue par défaut
            adaptation_strategies: Stratégies d'adaptation disponibles
        """
        self.default_audience_level = default_audience_level
        self.default_language = default_language
        self.adaptation_strategies = adaptation_strategies or list(AdaptationStrategy)
        
        # Dictionnaire des adaptateurs spécifiques par stratégie
        self._strategy_adapters = {
            AdaptationStrategy.SIMPLIFY: self._simplify_explanation,
            AdaptationStrategy.ENRICH: self._enrich_explanation,
            AdaptationStrategy.VISUALIZE: self._visualize_explanation,
            AdaptationStrategy.CONTEXTUALIZE: self._contextualize_explanation,
            AdaptationStrategy.SUMMARIZE: self._summarize_explanation,
            AdaptationStrategy.TRANSLATE: self._translate_explanation,
            AdaptationStrategy.INTERACTIVE: self._make_interactive
        }
    
    def adapt_explanation(self, 
                         explanation: ExplanationResult,
                         user_profile: Optional[UserProfile] = None,
                         context: Optional[Dict[str, Any]] = None) -> ExplanationResult:
        """
        Adapte une explication selon le profil utilisateur et le contexte.
        
        Args:
            explanation: L'explication à adapter
            user_profile: Profil de l'utilisateur (optionnel)
            context: Contexte d'explication (optionnel)
            
        Returns:
            Explication adaptée
        """
        if not user_profile and not context:
            return explanation  # Aucune adaptation nécessaire
        
        # Déterminer les stratégies d'adaptation à appliquer
        strategies = self._determine_adaptation_strategies(
            explanation, user_profile, context
        )
        
        # Appliquer les stratégies d'adaptation
        adapted_explanation = explanation
        for strategy in strategies:
            if strategy in self._strategy_adapters:
                adapter_fn = self._strategy_adapters[strategy]
                adapted_explanation = adapter_fn(adapted_explanation, user_profile, context)
        
        # Mettre à jour les métadonnées de l'explication
        if not hasattr(adapted_explanation.metadata, "adaptation_info"):
            adapted_explanation.metadata.adaptation_info = {}
        
        adapted_explanation.metadata.adaptation_info.update({
            "adapted_for_user": user_profile.user_id if user_profile else None,
            "audience_level": user_profile.audience_level.name if user_profile else self.default_audience_level.name,
            "applied_strategies": [s.value for s in strategies]
        })
        
        return adapted_explanation
    
    def _determine_adaptation_strategies(self,
                                       explanation: ExplanationResult,
                                       user_profile: Optional[UserProfile],
                                       context: Optional[Dict[str, Any]]) -> List[AdaptationStrategy]:
        """
        Détermine les stratégies d'adaptation à appliquer.
        
        Args:
            explanation: L'explication à adapter
            user_profile: Profil de l'utilisateur
            context: Contexte d'explication
            
        Returns:
            Liste des stratégies à appliquer
        """
        strategies = []
        
        # Si aucun profil utilisateur, utiliser uniquement le contexte
        if not user_profile:
            if context and "audience_level" in context:
                if context["audience_level"] == AudienceLevel.NON_TECHNICAL:
                    strategies.append(AdaptationStrategy.SIMPLIFY)
                elif context["audience_level"] == AudienceLevel.REGULATORY:
                    strategies.append(AdaptationStrategy.ENRICH)
            
            if context and "language" in context and context["language"] != self.default_language:
                strategies.append(AdaptationStrategy.TRANSLATE)
            
            return strategies
        
        # Adapter selon les préférences de l'utilisateur
        if ExplanationPreference.VISUAL in user_profile.preferences:
            strategies.append(AdaptationStrategy.VISUALIZE)
        
        if ExplanationPreference.TECHNICAL in user_profile.preferences:
            strategies.append(AdaptationStrategy.ENRICH)
        
        if ExplanationPreference.SIMPLIFIED in user_profile.preferences:
            strategies.append(AdaptationStrategy.SIMPLIFY)
        
        if ExplanationPreference.COMPREHENSIVE in user_profile.preferences:
            strategies.append(AdaptationStrategy.ENRICH)
            strategies.append(AdaptationStrategy.CONTEXTUALIZE)
        
        if ExplanationPreference.CONCISE in user_profile.preferences:
            strategies.append(AdaptationStrategy.SUMMARIZE)
        
        if ExplanationPreference.INTERACTIVE in user_profile.preferences:
            strategies.append(AdaptationStrategy.INTERACTIVE)
        
        # Adapter selon le niveau d'audience
        if user_profile.audience_level == AudienceLevel.NON_TECHNICAL:
            if AdaptationStrategy.SIMPLIFY not in strategies:
                strategies.append(AdaptationStrategy.SIMPLIFY)
            
            if AdaptationStrategy.VISUALIZE not in strategies:
                strategies.append(AdaptationStrategy.VISUALIZE)
        
        elif user_profile.audience_level == AudienceLevel.REGULATORY:
            if AdaptationStrategy.ENRICH not in strategies:
                strategies.append(AdaptationStrategy.ENRICH)
            
            if AdaptationStrategy.CONTEXTUALIZE not in strategies:
                strategies.append(AdaptationStrategy.CONTEXTUALIZE)
        
        # Adapter selon la langue
        if user_profile.language != self.default_language:
            strategies.append(AdaptationStrategy.TRANSLATE)
        
        # Prendre en compte le contexte spécifique
        if context:
            if context.get("time_constraint") == "limited":
                # Privilégier la concision en cas de contrainte de temps
                if AdaptationStrategy.SUMMARIZE not in strategies:
                    strategies.append(AdaptationStrategy.SUMMARIZE)
                
                # Retirer les stratégies qui augmentent la complexité
                if AdaptationStrategy.ENRICH in strategies:
                    strategies.remove(AdaptationStrategy.ENRICH)
            
            if context.get("domain") and context.get("domain") in user_profile.domain_expertise:
                expertise = user_profile.domain_expertise[context["domain"]]
                if expertise >= DomainExpertise.ADVANCED:
                    # Pour les experts, enrichir avec des détails techniques
                    if AdaptationStrategy.ENRICH not in strategies:
                        strategies.append(AdaptationStrategy.ENRICH)
                    
                    # Retirer la simplification pour les experts
                    if AdaptationStrategy.SIMPLIFY in strategies:
                        strategies.remove(AdaptationStrategy.SIMPLIFY)
        
        return strategies
    
    # Méthodes d'adaptation spécifiques
    
    def _simplify_explanation(self,
                            explanation: ExplanationResult,
                            user_profile: Optional[UserProfile],
                            context: Optional[Dict[str, Any]]) -> ExplanationResult:
        """
        Simplifie une explication pour la rendre plus accessible.
        
        Args:
            explanation: L'explication à simplifier
            user_profile: Profil de l'utilisateur
            context: Contexte d'explication
            
        Returns:
            Explication simplifiée
        """
        # Dans une implémentation réelle, cette méthode pourrait:
        # 1. Réduire le jargon technique
        # 2. Simplifier les concepts complexes
        # 3. Ajouter des analogies ou exemples simples
        # 4. Réduire la quantité de détails techniques
        
        # Pour l'exemple, nous nous contentons de marquer l'explication comme simplifiée
        explanation.metadata.simplification_applied = True
        
        return explanation
    
    def _enrich_explanation(self,
                          explanation: ExplanationResult,
                          user_profile: Optional[UserProfile],
                          context: Optional[Dict[str, Any]]) -> ExplanationResult:
        """
        Enrichit une explication avec des détails supplémentaires.
        
        Args:
            explanation: L'explication à enrichir
            user_profile: Profil de l'utilisateur
            context: Contexte d'explication
            
        Returns:
            Explication enrichie
        """
        # Dans une implémentation réelle, cette méthode pourrait:
        # 1. Ajouter des détails techniques supplémentaires
        # 2. Inclure des références à des articles ou publications
        # 3. Ajouter des formules mathématiques ou algorithmes
        # 4. Inclure des analyses statistiques plus poussées
        
        # Pour l'exemple, nous nous contentons de marquer l'explication comme enrichie
        explanation.metadata.enrichment_applied = True
        
        return explanation
    
    def _visualize_explanation(self,
                             explanation: ExplanationResult,
                             user_profile: Optional[UserProfile],
                             context: Optional[Dict[str, Any]]) -> ExplanationResult:
        """
        Ajoute des éléments visuels à une explication.
        
        Args:
            explanation: L'explication à visualiser
            user_profile: Profil de l'utilisateur
            context: Contexte d'explication
            
        Returns:
            Explication avec éléments visuels
        """
        # Dans une implémentation réelle, cette méthode pourrait:
        # 1. Ajouter des graphiques ou diagrammes
        # 2. Convertir des tableaux en visualisations
        # 3. Ajouter des codes couleur ou mise en évidence
        
        # Pour l'exemple, nous nous contentons de marquer l'explication comme visualisée
        explanation.metadata.visualization_applied = True
        
        return explanation
    
    def _contextualize_explanation(self,
                                 explanation: ExplanationResult,
                                 user_profile: Optional[UserProfile],
                                 context: Optional[Dict[str, Any]]) -> ExplanationResult:
        """
        Contextualise une explication selon le domaine d'application.
        
        Args:
            explanation: L'explication à contextualiser
            user_profile: Profil de l'utilisateur
            context: Contexte d'explication
            
        Returns:
            Explication contextualisée
        """
        # Dans une implémentation réelle, cette méthode pourrait:
        # 1. Ajouter des exemples spécifiques au domaine
        # 2. Faire référence à des cas d'usage pertinents
        # 3. Adapter la terminologie au domaine
        
        # Pour l'exemple, nous nous contentons de marquer l'explication comme contextualisée
        explanation.metadata.contextualization_applied = True
        domain = context.get("domain") if context else None
        explanation.metadata.contextualization_domain = domain
        
        return explanation
    
    def _summarize_explanation(self,
                             explanation: ExplanationResult,
                             user_profile: Optional[UserProfile],
                             context: Optional[Dict[str, Any]]) -> ExplanationResult:
        """
        Résume une explication pour la rendre plus concise.
        
        Args:
            explanation: L'explication à résumer
            user_profile: Profil de l'utilisateur
            context: Contexte d'explication
            
        Returns:
            Explication résumée
        """
        # Dans une implémentation réelle, cette méthode pourrait:
        # 1. Extraire les points clés de l'explication
        # 2. Réduire la verbosité
        # 3. Créer une version condensée avec les informations essentielles
        
        # Pour l'exemple, nous nous contentons de marquer l'explication comme résumée
        explanation.metadata.summarization_applied = True
        
        return explanation
    
    def _translate_explanation(self,
                             explanation: ExplanationResult,
                             user_profile: Optional[UserProfile],
                             context: Optional[Dict[str, Any]]) -> ExplanationResult:
        """
        Traduit une explication dans la langue de l'utilisateur.
        
        Args:
            explanation: L'explication à traduire
            user_profile: Profil de l'utilisateur
            context: Contexte d'explication
            
        Returns:
            Explication traduite
        """
        # Dans une implémentation réelle, cette méthode pourrait:
        # 1. Utiliser un service de traduction
        # 2. Adapter les expressions idiomatiques
        # 3. Préserver la terminologie technique spécifique
        
        # Pour l'exemple, nous nous contentons de marquer l'explication comme traduite
        target_language = user_profile.language if user_profile else context.get("language", self.default_language)
        explanation.metadata.translation_applied = True
        explanation.metadata.target_language = target_language
        
        return explanation
    
    def _make_interactive(self,
                        explanation: ExplanationResult,
                        user_profile: Optional[UserProfile],
                        context: Optional[Dict[str, Any]]) -> ExplanationResult:
        """
        Rend une explication interactive.
        
        Args:
            explanation: L'explication à rendre interactive
            user_profile: Profil de l'utilisateur
            context: Contexte d'explication
            
        Returns:
            Explication interactive
        """
        # Dans une implémentation réelle, cette méthode pourrait:
        # 1. Ajouter des éléments interactifs (sliders, boutons, etc.)
        # 2. Permettre l'exploration de différents scénarios
        # 3. Ajouter des tooltips ou infobulles
        
        # Pour l'exemple, nous nous contentons de marquer l'explication comme interactive
        explanation.metadata.interactivity_applied = True
        
        return explanation
