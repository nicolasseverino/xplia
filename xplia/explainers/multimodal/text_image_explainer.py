"""
Explainer multimodal pour les modèles combinant texte et image
============================================================

Ce module implémente un explainer spécialisé pour les modèles qui travaillent
simultanément avec des données textuelles et des images, comme CLIP, VisualBERT, etc.
"""

import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import torch
import cv2

from .base import (
    MultimodalExplainerBase, DataModality, ModalityExplanation, 
    MultimodalExplanationResult
)
from .registry import register_multimodal_explainer
from ...core.base import ExplainabilityMethod, AudienceLevel, ExplanationQuality
from ...core.optimizations import optimize, cached_call, parallel_map
from ...explainers.lime_explainer import LIMEExplainer
from ...explainers.gradient_explainer import GradientExplainer
from ...explainers.shap_explainer import SHAPExplainer
from ...explainers.attention_explainer import AttentionExplainer


logger = logging.getLogger(__name__)


@register_multimodal_explainer("text_image")
class TextImageExplainer(MultimodalExplainerBase):
    """
    Explainer spécialisé pour les modèles combinant texte et image.
    
    Cet explainer est capable de générer des explications pour des modèles
    multimodaux qui traitent simultanément des données textuelles et des images,
    comme CLIP, VisualBERT, ou d'autres architectures vision-langage.
    """
    
    supported_modalities = {DataModality.TEXT, DataModality.IMAGE}
    
    def __init__(self, 
                 model: Any,
                 model_type: str = "generic",
                 text_tokenizer = None,
                 image_preprocessor = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 explanation_method: ExplainabilityMethod = ExplainabilityMethod.MODEL_SPECIFIC,
                 audience_level: AudienceLevel = AudienceLevel.TECHNICAL,
                 **kwargs):
        """
        Initialise l'explainer texte-image.
        
        Args:
            model: Le modèle multimodal à expliquer
            model_type: Type spécifique du modèle ("clip", "visualbert", "vilt", etc.)
            text_tokenizer: Tokenizer pour les entrées textuelles
            image_preprocessor: Préprocesseur pour les images
            device: Appareil d'exécution ("cpu", "cuda", etc.)
            explanation_method: Méthode d'explicabilité à utiliser
            audience_level: Niveau d'audience ciblé pour les explications
            **kwargs: Paramètres additionnels
        """
        super().__init__(
            model=model,
            modalities={DataModality.TEXT, DataModality.IMAGE},
            explanation_method=explanation_method,
            audience_level=audience_level,
            **kwargs
        )
        
        self.model_type = model_type.lower()
        self.text_tokenizer = text_tokenizer
        self.image_preprocessor = image_preprocessor
        self.device = device
        self.attention_maps = {}
        
        # Configuration spécifique selon le type de modèle
        self._configure_for_model_type()
        
    def _configure_for_model_type(self):
        """Configure l'explainer selon le type spécifique du modèle."""
        if self.model_type == "clip":
            self.extract_attention = self._extract_clip_attention
            self.text_feature_names = self._get_clip_text_feature_names
            self.image_feature_names = self._get_clip_image_feature_names
        elif self.model_type == "visualbert":
            self.extract_attention = self._extract_visualbert_attention
            self.text_feature_names = self._get_visualbert_text_feature_names
            self.image_feature_names = self._get_visualbert_image_feature_names
        elif self.model_type == "vilt":
            self.extract_attention = self._extract_vilt_attention
            self.text_feature_names = self._get_vilt_text_feature_names
            self.image_feature_names = self._get_vilt_image_feature_names
        else:
            # Comportement par défaut pour les modèles génériques
            self.extract_attention = self._extract_generic_attention
            self.text_feature_names = self._get_generic_text_feature_names
            self.image_feature_names = self._get_generic_image_feature_names
    
    def initialize_modality_explainers(self):
        """Initialise les explainers spécifiques pour chaque modalité."""
        
        # Explainer pour les données textuelles
        self.modality_explainers[DataModality.TEXT] = {
            "lime": LIMEExplainer(self.model),
            "attention": AttentionExplainer(self.model)
        }
        
        # Explainer pour les images
        self.modality_explainers[DataModality.IMAGE] = {
            "gradient": GradientExplainer(self.model),
            "shap": SHAPExplainer(self.model)
        }
    
    @cached_call
    def explain_modality(self, 
                         modality: DataModality,
                         inputs: Any,
                         **kwargs) -> ModalityExplanation:
        """
        Génère une explication pour une modalité spécifique.
        
        Args:
            modality: La modalité à expliquer (TEXT ou IMAGE)
            inputs: Les données d'entrée pour cette modalité
            **kwargs: Paramètres additionnels
            
        Returns:
            Une explication pour la modalité spécifiée
        """
        start_time = time.time()
        
        if modality == DataModality.TEXT:
            explanation = self._explain_text_modality(inputs, **kwargs)
            contribution = self._estimate_text_contribution(inputs, **kwargs)
        elif modality == DataModality.IMAGE:
            explanation = self._explain_image_modality(inputs, **kwargs)
            contribution = self._estimate_image_contribution(inputs, **kwargs)
        else:
            raise ValueError(f"Modalité non supportée: {modality}")
        
        # Estimation de la confiance dans l'explication
        confidence = self._estimate_explanation_confidence(modality, explanation)
        
        self.last_explanation_time_ms = int((time.time() - start_time) * 1000)
        
        return ModalityExplanation(
            modality=modality,
            explanation=explanation,
            confidence=confidence,
            contribution=contribution
        )
    
    def _explain_text_modality(self, text_input: str, **kwargs):
        """
        Génère une explication pour la modalité texte.
        
        Args:
            text_input: Texte d'entrée à expliquer
            **kwargs: Paramètres additionnels
            
        Returns:
            Objet ExplanationResult pour la modalité texte
        """
        # Choisir la meilleure méthode d'explication selon le contexte
        if self.model_type in ["visualbert", "vilt", "clip"]:
            # Utiliser l'attention pour ces modèles
            explainer = self.modality_explainers[DataModality.TEXT]["attention"]
            return explainer.explain(text_input, **kwargs)
        else:
            # Utiliser LIME pour les modèles génériques ou inconnus
            explainer = self.modality_explainers[DataModality.TEXT]["lime"]
            return explainer.explain(text_input, **kwargs)
    
    def _explain_image_modality(self, image_input: Any, **kwargs):
        """
        Génère une explication pour la modalité image.
        
        Args:
            image_input: Image d'entrée à expliquer
            **kwargs: Paramètres additionnels
            
        Returns:
            Objet ExplanationResult pour la modalité image
        """
        # Choisir la meilleure méthode d'explication selon le contexte
        if hasattr(self.model, "get_attention") or self.model_type in ["visualbert", "vilt"]:
            # Générer une carte d'attention visuelle
            attention_map = self.extract_attention(image_input, **kwargs)
            explanation = self._generate_visual_explanation_from_attention(
                image_input, attention_map
            )
            return explanation
        else:
            # Utiliser la méthode du gradient pour les modèles génériques
            explainer = self.modality_explainers[DataModality.IMAGE]["gradient"]
            return explainer.explain(image_input, **kwargs)
    
    def _analyze_cross_modal_interactions(self,
                                         inputs: Dict[DataModality, Any],
                                         explanations: List[ModalityExplanation]
                                         ) -> Dict[Tuple[DataModality, DataModality], float]:
        """
        Analyse les interactions entre texte et image.
        
        Args:
            inputs: Dictionnaire des entrées par modalité
            explanations: Liste des explications par modalité
            
        Returns:
            Dictionnaire des scores d'interaction entre modalités
        """
        # Pour les modèles texte-image, nous n'avons qu'une interaction à analyser
        text_input = inputs.get(DataModality.TEXT)
        image_input = inputs.get(DataModality.IMAGE)
        
        if text_input is None or image_input is None:
            return {}
        
        # Calculer le score d'interaction texte-image
        interaction_score = self._compute_text_image_interaction(
            text_input, image_input
        )
        
        return {(DataModality.TEXT, DataModality.IMAGE): interaction_score}
    
    def _explain_fusion_mechanism(self,
                                inputs: Dict[DataModality, Any],
                                explanations: List[ModalityExplanation],
                                **kwargs) -> Dict[str, Any]:
        """
        Explique le mécanisme de fusion texte-image.
        
        Args:
            inputs: Dictionnaire des entrées par modalité
            explanations: Liste des explications par modalité
            **kwargs: Paramètres additionnels
            
        Returns:
            Dictionnaire décrivant le mécanisme de fusion
        """
        # Analyse du mécanisme de fusion selon le type de modèle
        if self.model_type == "clip":
            return self._explain_clip_fusion(inputs, explanations, **kwargs)
        elif self.model_type == "visualbert":
            return self._explain_visualbert_fusion(inputs, explanations, **kwargs)
        else:
            # Mécanisme de fusion générique
            return self._explain_generic_fusion(inputs, explanations, **kwargs)
    
    # Méthodes utilitaires pour CLIP
    
    def _extract_clip_attention(self, inputs, **kwargs):
        """Extrait les cartes d'attention d'un modèle CLIP."""
        # Implémentation spécifique à CLIP
        pass
    
    def _get_clip_text_feature_names(self, text_input):
        """Obtient les noms des features textuelles pour CLIP."""
        # Implémentation spécifique à CLIP
        pass
    
    def _get_clip_image_feature_names(self, image_input):
        """Obtient les noms des features visuelles pour CLIP."""
        # Implémentation spécifique à CLIP
        pass
    
    def _explain_clip_fusion(self, inputs, explanations, **kwargs):
        """Explique le mécanisme de fusion spécifique à CLIP."""
        # Implémentation spécifique à CLIP
        fusion_explanation = {
            "mechanism": "cosine_similarity",
            "description": "CLIP calcule la similarité cosinus entre les embeddings texte et image",
            "architecture": "dual_encoder",
            "fusion_point": "late_fusion"
        }
        return fusion_explanation
    
    # Méthodes utilitaires génériques
    
    def _extract_generic_attention(self, inputs, **kwargs):
        """Méthode générique pour extraire l'attention."""
        # Implémentation générique
        pass
    
    def _get_generic_text_feature_names(self, text_input):
        """Méthode générique pour nommer les features textuelles."""
        # Implémentation générique
        pass
    
    def _get_generic_image_feature_names(self, image_input):
        """Méthode générique pour nommer les features visuelles."""
        # Implémentation générique
        pass
    
    def _explain_generic_fusion(self, inputs, explanations, **kwargs):
        """Explique un mécanisme de fusion générique."""
        # Implémentation générique
        fusion_explanation = {
            "mechanism": "unknown",
            "description": "Mécanisme de fusion non spécifié",
            "architecture": "multimodal_fusion",
            "fusion_point": "unknown"
        }
        return fusion_explanation
    
    def _generate_visual_explanation_from_attention(self, image_input, attention_map):
        """
        Génère une explication visuelle à partir d'une carte d'attention.
        
        Args:
            image_input: Image originale
            attention_map: Carte d'attention
            
        Returns:
            ExplanationResult contenant l'explication visuelle
        """
        # Implémentation générique pour visualiser une carte d'attention
        pass
    
    def _compute_text_image_interaction(self, text_input, image_input):
        """
        Calcule le score d'interaction entre texte et image.
        
        Args:
            text_input: Entrée textuelle
            image_input: Entrée visuelle
            
        Returns:
            Score d'interaction entre 0 et 1
        """
        # Implémentation générique
        return 0.5  # Valeur par défaut
    
    def _estimate_text_contribution(self, text_input, **kwargs):
        """
        Estime la contribution de la modalité texte à la décision finale.
        
        Args:
            text_input: Entrée textuelle
            **kwargs: Paramètres additionnels
            
        Returns:
            Score de contribution entre 0 et 1
        """
        # Implémentation générique
        return 0.5  # Valeur par défaut
    
    def _estimate_image_contribution(self, image_input, **kwargs):
        """
        Estime la contribution de la modalité image à la décision finale.
        
        Args:
            image_input: Entrée visuelle
            **kwargs: Paramètres additionnels
            
        Returns:
            Score de contribution entre 0 et 1
        """
        # Implémentation générique
        return 0.5  # Valeur par défaut
    
    def _estimate_explanation_confidence(self, modality, explanation):
        """
        Estime la confiance dans l'explication d'une modalité.
        
        Args:
            modality: Modalité concernée
            explanation: Explication générée
            
        Returns:
            Score de confiance entre 0 et 1
        """
        # Implémentation générique
        return 0.8  # Valeur par défaut
