"""
Module de base pour l'explicabilité multimodale
==============================================

Ce module définit les classes abstraites et les interfaces pour tous les explainers multimodaux.
Il fournit l'architecture fondamentale permettant d'analyser et d'expliquer les modèles
qui combinent plusieurs types de données (texte, image, audio, séries temporelles, etc.).
"""

import abc
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union, Set, Type, ClassVar

import numpy as np
import pandas as pd

from ...core.base import (
    ExplainerBase, ExplanationResult, ExplainabilityMethod,
    AudienceLevel, ExplanationQuality, ExplanationFormat, ModelMetadata
)
from ...core.registry import register_explainer
from ...core.optimizations import (
    optimize, parallel_map, cached_call, optimize_memory
)
from ...core.performance import (
    ParallelExecutor, cached_result, memory_efficient, 
    process_in_chunks, optimize_explanations
)


class DataModality(Enum):
    """Types de modalités de données supportées par les explainers multimodaux."""
    TEXT = auto()
    IMAGE = auto()
    AUDIO = auto()
    TIMESERIES = auto()
    TABULAR = auto()
    VIDEO = auto()
    GRAPH = auto()
    EMBEDDING = auto()
    MIXED = auto()


@dataclass
class ModalityExplanation:
    """Explication pour une modalité spécifique."""
    modality: DataModality
    explanation: ExplanationResult
    confidence: float
    contribution: float  # Contribution relative de cette modalité à la décision finale


@dataclass
class MultimodalExplanationResult(ExplanationResult):
    """Résultat d'explication pour un modèle multimodal."""
    modality_explanations: List[ModalityExplanation] = field(default_factory=list)
    cross_modal_interactions: Dict[Tuple[DataModality, DataModality], float] = field(default_factory=dict)
    fusion_explanation: Optional[Dict[str, Any]] = None  # Explication du mécanisme de fusion des modalités
    
    def get_explanation_by_modality(self, modality: DataModality) -> Optional[ModalityExplanation]:
        """Récupère l'explication pour une modalité spécifique."""
        for exp in self.modality_explanations:
            if exp.modality == modality:
                return exp
        return None


class MultimodalExplainerBase(ExplainerBase, metaclass=abc.ABCMeta):
    """
    Classe de base pour tous les explainers multimodaux.
    
    Cette classe définit l'interface commune que doivent implémenter tous les explainers 
    capables de fournir des explications pour des modèles traitant plusieurs types de données.
    """
    
    supported_modalities: ClassVar[Set[DataModality]] = {
        DataModality.TEXT, DataModality.IMAGE, DataModality.AUDIO, 
        DataModality.TIMESERIES, DataModality.TABULAR
    }
    
    def __init__(self, 
                 model: Any, 
                 modalities: Set[DataModality] = None,
                 explanation_method: ExplainabilityMethod = ExplainabilityMethod.AGNOSTIC,
                 audience_level: AudienceLevel = AudienceLevel.TECHNICAL,
                 **kwargs):
        """
        Initialise un explainer multimodal.
        
        Args:
            model: Le modèle à expliquer
            modalities: Ensemble des modalités traitées par le modèle
            explanation_method: Méthode d'explicabilité à utiliser
            audience_level: Niveau d'audience ciblé pour les explications
            **kwargs: Paramètres additionnels spécifiques à l'explainer
        """
        super().__init__(model=model, 
                         explanation_method=explanation_method,
                         audience_level=audience_level, 
                         **kwargs)
        
        self.modalities = modalities or self.detect_modalities(model)
        self.modality_explainers = {}  # Sera rempli avec des explainers spécifiques à chaque modalité
        self.initialize_modality_explainers()
        
    @classmethod
    def detect_modalities(cls, model: Any) -> Set[DataModality]:
        """
        Détecte automatiquement les modalités traitées par le modèle.
        
        Args:
            model: Le modèle à analyser
            
        Returns:
            Ensemble des modalités détectées
        """
        # Implémentation à surcharger dans les classes dérivées
        return {DataModality.MIXED}
    
    @abc.abstractmethod
    def initialize_modality_explainers(self) -> None:
        """
        Initialise les explainers spécifiques pour chaque modalité supportée.
        """
        pass
    
    @abc.abstractmethod
    def explain_modality(self, 
                         modality: DataModality, 
                         inputs: Any, 
                         **kwargs) -> ModalityExplanation:
        """
        Génère une explication pour une modalité spécifique.
        
        Args:
            modality: La modalité à expliquer
            inputs: Les données d'entrée pour cette modalité
            **kwargs: Paramètres additionnels
            
        Returns:
            Une explication pour la modalité spécifiée
        """
        pass
    
    @optimize_explanations
    @memory_efficient
    def explain(self, 
                inputs: Dict[DataModality, Any], 
                reference_inputs: Optional[Dict[DataModality, Any]] = None,
                **kwargs) -> MultimodalExplanationResult:
        """
        Génère des explications pour toutes les modalités d'un modèle multimodal.
        
        Args:
            inputs: Dictionnaire associant chaque modalité à ses données d'entrée
            reference_inputs: Données de référence pour les explications contrastives
            **kwargs: Paramètres additionnels
            
        Returns:
            Un résultat d'explication multimodale complet
        """
        # Vérification des modalités fournies
        input_modalities = set(inputs.keys())
        if not input_modalities.issubset(self.modalities):
            unsupported = input_modalities - self.modalities
            raise ValueError(f"Modalités non supportées: {unsupported}")
        
        # Génération parallèle des explications pour chaque modalité
        modality_explanations = parallel_map(
            lambda mod: self.explain_modality(mod, inputs[mod], **kwargs),
            list(inputs.keys())
        )
        
        # Analyse des interactions entre modalités
        cross_modal_interactions = self._analyze_cross_modal_interactions(
            inputs, modality_explanations
        )
        
        # Génération de l'explication du mécanisme de fusion
        fusion_explanation = self._explain_fusion_mechanism(
            inputs, modality_explanations, **kwargs
        )
        
        # Construction du résultat d'explication multimodale
        return MultimodalExplanationResult(
            model_name=str(self.model),
            explanation_method=self.explanation_method,
            audience_level=self.audience_level,
            modality_explanations=modality_explanations,
            cross_modal_interactions=cross_modal_interactions,
            fusion_explanation=fusion_explanation,
            metadata=self._generate_metadata(**kwargs)
        )
    
    @abc.abstractmethod
    def _analyze_cross_modal_interactions(self, 
                                         inputs: Dict[DataModality, Any],
                                         explanations: List[ModalityExplanation]
                                         ) -> Dict[Tuple[DataModality, DataModality], float]:
        """
        Analyse les interactions entre les différentes modalités.
        
        Args:
            inputs: Dictionnaire des données d'entrée par modalité
            explanations: Liste des explications par modalité
            
        Returns:
            Dictionnaire associant des paires de modalités à leurs scores d'interaction
        """
        pass
    
    @abc.abstractmethod
    def _explain_fusion_mechanism(self,
                                 inputs: Dict[DataModality, Any],
                                 explanations: List[ModalityExplanation],
                                 **kwargs) -> Dict[str, Any]:
        """
        Explique le mécanisme de fusion des différentes modalités.
        
        Args:
            inputs: Dictionnaire des données d'entrée par modalité
            explanations: Liste des explications par modalité
            **kwargs: Paramètres additionnels
            
        Returns:
            Dictionnaire contenant l'explication du mécanisme de fusion
        """
        pass
    
    def _generate_metadata(self, **kwargs) -> ModelMetadata:
        """
        Génère les métadonnées pour l'explication.
        
        Args:
            **kwargs: Paramètres additionnels
            
        Returns:
            Métadonnées du modèle et de l'explication
        """
        modality_info = {str(modality): True for modality in self.modalities}
        
        return ModelMetadata(
            model_type="multimodal",
            model_class=type(self.model).__name__,
            modalities=modality_info,
            explanation_quality=self._estimate_explanation_quality(),
            explanation_time_ms=self.last_explanation_time_ms if hasattr(self, "last_explanation_time_ms") else None,
            explanation_version="1.0.0",
            additional_info=kwargs.get("metadata_info", {})
        )
    
    def _estimate_explanation_quality(self) -> ExplanationQuality:
        """
        Estime la qualité de l'explication multimodale.
        
        Returns:
            Niveau estimé de qualité de l'explication
        """
        # Cette méthode peut être surchargée par les classes dérivées
        # pour fournir une estimation plus précise
        return ExplanationQuality.HIGH
