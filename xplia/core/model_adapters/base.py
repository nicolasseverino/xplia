"""
Base pour les adaptateurs de modèles XPLIA
========================================

Ce module fournit la classe de base abstraite pour tous les adaptateurs
de modèles dans XPLIA.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from ...core import ModelMetadata, ModelType  # Import depuis le package principal

class ModelAdapterBase(ABC):
    """
    Classe de base abstraite pour tous les adaptateurs de modèles XPLIA.
    
    Cette classe définit l'interface commune que tous les adapteurs de modèles
    doivent implémenter pour être compatibles avec XPLIA.
    """
    
    def __init__(self, model: Any, **kwargs):
        """
        Initialise l'adaptateur de modèle.
        
        Args:
            model: Le modèle à adapter
            **kwargs: Arguments additionnels spécifiques à l'implémentation
        """
        self.model = model
        self._metadata = self._extract_metadata(**kwargs)
        self._feature_names = kwargs.get('feature_names', None)
        self._output_names = kwargs.get('output_names', None)
        self._model_type = None
        self._framework = "unknown"
        
    @property
    def metadata(self) -> ModelMetadata:
        """Retourne les métadonnées du modèle."""
        return self._metadata
        
    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame], **kwargs) -> np.ndarray:
        """
        Effectue une prédiction avec le modèle.
        
        Args:
            X: Données d'entrée
            **kwargs: Arguments additionnels pour la prédiction
            
        Returns:
            Prédictions du modèle
        """
        pass
        
    @abstractmethod
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame], **kwargs) -> np.ndarray:
        """
        Retourne les probabilités de prédiction.
        
        Args:
            X: Données d'entrée
            **kwargs: Arguments additionnels pour la prédiction
            
        Returns:
            Probabilités de prédiction (shape: n_samples, n_classes)
        """
        pass
        
    @abstractmethod
    def _extract_metadata(self, **kwargs) -> ModelMetadata:
        """
        Extrait les métadonnées du modèle.
        
        Returns:
            ModelMetadata: Métadonnées du modèle
        """
        pass
        
    def get_feature_names(self) -> List[str]:
        """
        Retourne les noms des caractéristiques du modèle.
        
        Returns:
            Liste des noms de caractéristiques
        """
        return self._feature_names
        
    def set_feature_names(self, feature_names: List[str]):
        """
        Définit les noms des caractéristiques du modèle.
        
        Args:
            feature_names: Liste des noms de caractéristiques
        """
        self._feature_names = feature_names
        
    def get_output_names(self) -> List[str]:
        """
        Retourne les noms des sorties du modèle.
        
        Returns:
            Liste des noms de sorties
        """
        return self._output_names
        
    def set_output_names(self, output_names: List[str]):
        """
        Définit les noms des sorties du modèle.
        
        Args:
            output_names: Liste des noms de sorties
        """
        self._output_names = output_names
        
    def supports_predict_proba(self) -> bool:
        """
        Vérifie si le modèle supporte les probabilités de prédiction.
        
        Returns:
            bool: True si le modèle supporte predict_proba, False sinon
        """
        return hasattr(self.model, 'predict_proba') or hasattr(self.model, 'predict_probs')
        
    def get_model_type(self) -> ModelType:
        """
        Retourne le type du modèle.
        
        Returns:
            ModelType: Type du modèle (classification, regression, etc.)
        """
        return self._model_type
        
    def get_framework(self) -> str:
        """
        Retourne le framework du modèle.
        
        Returns:
            str: Nom du framework (sklearn, tensorflow, pytorch, etc.)
        """
        return self._framework
        
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Retourne un résumé des caractéristiques du modèle.
        
        Returns:
            dict: Dictionnaire contenant les informations du modèle
        """
        return {
            'framework': self.get_framework(),
            'model_type': self.get_model_type().name if self.get_model_type() else "unknown",
            'supports_proba': self.supports_predict_proba(),
            'feature_names': self.get_feature_names(),
            'output_names': self.get_output_names(),
            'model_class': self.model.__class__.__name__,
            'model_module': self.model.__class__.__module__
        }
