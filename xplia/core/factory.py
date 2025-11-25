"""
Factory pour la création des explainers et le chargement des modèles de XPLIA
===================================================================

Ce module fournit les fonctionnalités pour instancier dynamiquement
les explainers appropriés et charger différents types de modèles.
"""

import importlib
import inspect
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

import joblib
import numpy as np

from .base import ExplainerBase, ExplainabilityMethod
from .registry import get_registered_explainers


def load_model(model_path: Union[str, Path], model_type: Optional[str] = None, **kwargs) -> Any:
    """
    Charge un modèle à partir d'un chemin de fichier.

    Cette fonction détecte automatiquement le type de modèle et utilise
    le chargeur approprié (pickle, joblib, keras, pytorch, etc.).

    Args:
        model_path: Chemin vers le fichier du modèle
        model_type: Type de modèle (optionnel, pour forcer un type spécifique)
        **kwargs: Arguments additionnels pour le chargeur spécifique

    Returns:
        Any: Modèle chargé

    Raises:
        ValueError: Si le type de modèle n'est pas supporté ou ne peut pas être détecté
    """
    model_path = Path(model_path)
    
    # Détection automatique du type si non spécifié
    if model_type is None:
        suffix = model_path.suffix.lower()
        if suffix in ['.pkl', '.pickle']:
            model_type = 'pickle'
        elif suffix in ['.joblib']:
            model_type = 'joblib'
        elif suffix in ['.h5', '.keras', '.tf']:
            model_type = 'tensorflow'
        elif suffix in ['.pt', '.pth']:
            model_type = 'pytorch'
        else:
            raise ValueError(f"Impossible de détecter le type de modèle pour {model_path}")

    # Chargement basé sur le type
    if model_type == 'pickle':
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    elif model_type == 'joblib':
        model = joblib.load(model_path)
    elif model_type == 'tensorflow':
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path, **kwargs)
        except ImportError:
            raise ImportError("TensorFlow est requis pour charger ce modèle. Installez-le avec 'pip install tensorflow'.")
    elif model_type == 'pytorch':
        try:
            import torch
            model = torch.load(model_path, **kwargs)
        except ImportError:
            raise ImportError("PyTorch est requis pour charger ce modèle. Installez-le avec 'pip install torch'.")
    else:
        raise ValueError(f"Type de modèle non supporté: {model_type}")
    
    return model


def create_explainer(model: Any, 
                    method: Union[str, ExplainabilityMethod] = "unified", 
                    explainer_cls: Optional[Type[ExplainerBase]] = None,
                    **kwargs) -> ExplainerBase:
    """
    Crée un explainer pour le modèle donné.

    Cette fonction sert de factory pour instancier l'explainer approprié
    en fonction du modèle et de la méthode d'explicabilité souhaitée.

    Args:
        model: Modèle à expliquer
        method: Méthode d'explicabilité à utiliser
        explainer_cls: Classe d'explainer à utiliser (optionnel)
        **kwargs: Arguments additionnels pour l'explainer

    Returns:
        ExplainerBase: Instance d'explainer appropriée

    Raises:
        ValueError: Si aucun explainer approprié n'est trouvé
    """
    
    # Conversion du string en enum si nécessaire
    if isinstance(method, str):
        try:
            method = ExplainabilityMethod(method.lower())
        except ValueError:
            raise ValueError(f"Méthode d'explicabilité non reconnue: {method}")
    
    # Si la classe est explicitement fournie, l'utiliser
    if explainer_cls is not None:
        if not issubclass(explainer_cls, ExplainerBase):
            raise TypeError(f"La classe fournie {explainer_cls.__name__} n'est pas un sous-type d'ExplainerBase")
        return explainer_cls(model, **kwargs)
    
    # Recherche dans le registre des explainers
    registered_explainers = get_registered_explainers()
    model_type = type(model).__name__
    
    # Recherche par méthode et compatibilité du modèle
    compatible_explainers = []
    for explainer_class in registered_explainers:
        explainer_instance = explainer_class(model, **kwargs)
        if (explainer_instance.method == method and 
            explainer_instance.supports_model(model)):
            compatible_explainers.append(explainer_class)
    
    if not compatible_explainers:
        # Essayer l'explainer unifié en dernier recours
        if method != ExplainabilityMethod.UNIFIED:
            return create_explainer(model, ExplainabilityMethod.UNIFIED, **kwargs)
        raise ValueError(f"Aucun explainer compatible trouvé pour la méthode {method} "
                        f"et le type de modèle {model_type}")
    
    # Prendre le premier explainer compatible
    # TODO: Stratégie plus sophistiquée de sélection si plusieurs candidats
    selected_explainer_class = compatible_explainers[0]
    
    # Instancier et retourner l'explainer
    return selected_explainer_class(model, **kwargs)


def auto_detect_model_type(model: Any) -> str:
    """
    Détecte automatiquement le type du modèle.

    Args:
        model: Modèle à analyser

    Returns:
        str: Type de modèle détecté
    """
    # Vérification des types communs
    model_module = model.__class__.__module__
    model_name = model.__class__.__name__
    
    if 'sklearn' in model_module:
        return 'sklearn'
    elif 'xgboost' in model_module:
        return 'xgboost'
    elif 'lightgbm' in model_module:
        return 'lightgbm'
    elif 'keras' in model_module or 'tensorflow' in model_module:
        return 'tensorflow'
    elif 'torch' in model_module:
        return 'pytorch'
    elif 'catboost' in model_module:
        return 'catboost'
    
    # Détection par interface (vérification des méthodes)
    if hasattr(model, 'predict_proba') and hasattr(model, 'fit'):
        return 'sklearn-like'
    
    # Si aucun type n'a été détecté
    return 'unknown'


class ModelFactory:
    """
    Factory pour créer des adaptateurs de modèles.
    
    Cette classe fournit des méthodes pour charger et adapter différents
    types de modèles ML/DL de manière uniforme.
    """
    
    @staticmethod
    def load_model(model_path: Union[str, Path], model_type: Optional[str] = None, **kwargs) -> Any:
        """
        Charge un modèle depuis un fichier.
        
        Args:
            model_path: Chemin vers le fichier du modèle
            model_type: Type de modèle (optionnel, auto-détecté si non fourni)
            **kwargs: Arguments additionnels pour le chargeur
            
        Returns:
            Modèle chargé
        """
        return load_model(model_path, model_type, **kwargs)
    
    @staticmethod
    def create_adapter(model: Any, **kwargs):
        """
        Crée un adaptateur pour le modèle donné.
        
        Args:
            model: Modèle à adapter
            **kwargs: Arguments additionnels
            
        Returns:
            ModelAdapterBase: Adaptateur approprié pour le modèle
        """
        from .model_adapters.base import ModelAdapterBase
        from .model_adapters.sklearn_adapter import SklearnModelAdapter
        
        model_type = auto_detect_model_type(model)
        
        # Mapping des types aux adaptateurs
        adapters = {
            'sklearn': SklearnModelAdapter,
            'sklearn-like': SklearnModelAdapter,
        }
        
        # Import conditionnel des adaptateurs optionnels
        try:
            from .model_adapters.pytorch_adapter import PyTorchModelAdapter
            adapters['pytorch'] = PyTorchModelAdapter
        except ImportError:
            pass
            
        try:
            from .model_adapters.tensorflow_adapter import TensorFlowModelAdapter
            adapters['tensorflow'] = TensorFlowModelAdapter
        except ImportError:
            pass
            
        try:
            from .model_adapters.xgboost_adapter import XGBoostModelAdapter
            adapters['xgboost'] = XGBoostModelAdapter
        except ImportError:
            pass
        
        adapter_class = adapters.get(model_type)
        if adapter_class is None:
            raise ValueError(f"Aucun adaptateur trouvé pour le type de modèle: {model_type}")
        
        return adapter_class(model, **kwargs)
    
    @staticmethod
    def detect_model_type(model: Any) -> str:
        """
        Détecte le type d'un modèle.
        
        Args:
            model: Modèle à analyser
            
        Returns:
            str: Type de modèle détecté
        """
        return auto_detect_model_type(model)


class ExplainerFactory:
    """
    Factory pour créer des explainers.
    
    Cette classe centralise la création d'explainers en fonction
    du modèle et de la méthode d'explicabilité souhaitée.
    """
    
    @staticmethod
    def create(model: Any, 
               method: Union[str, ExplainabilityMethod] = "unified",
               explainer_cls: Optional[Type[ExplainerBase]] = None,
               **kwargs) -> ExplainerBase:
        """
        Crée un explainer pour le modèle donné.
        
        Args:
            model: Modèle à expliquer
            method: Méthode d'explicabilité ('shap', 'lime', 'unified', etc.)
            explainer_cls: Classe d'explainer spécifique (optionnel)
            **kwargs: Arguments additionnels pour l'explainer
            
        Returns:
            ExplainerBase: Instance d'explainer appropriée
            
        Raises:
            ValueError: Si la méthode n'est pas reconnue
            TypeError: Si explainer_cls n'est pas un ExplainerBase
        """
        return create_explainer(model, method, explainer_cls, **kwargs)
    
    @staticmethod
    def list_available_methods() -> list:
        """
        Liste toutes les méthodes d'explicabilité disponibles.
        
        Returns:
            list: Liste des méthodes disponibles
        """
        return [method.value for method in ExplainabilityMethod]
    
    @staticmethod
    def get_recommended_method(model: Any) -> str:
        """
        Recommande la meilleure méthode d'explicabilité pour un modèle.
        
        Args:
            model: Modèle à analyser
            
        Returns:
            str: Méthode recommandée
        """
        model_type = auto_detect_model_type(model)
        
        # Recommandations basées sur le type de modèle
        recommendations = {
            'sklearn': 'shap',  # SHAP excellent pour sklearn
            'xgboost': 'shap',  # SHAP optimisé pour XGBoost
            'lightgbm': 'shap',
            'catboost': 'shap',
            'tensorflow': 'gradient',  # Gradients pour deep learning
            'pytorch': 'gradient',
            'sklearn-like': 'lime',  # LIME pour modèles génériques
            'unknown': 'unified'  # Unified en dernier recours
        }
        
        return recommendations.get(model_type, 'unified')


class VisualizerFactory:
    """
    Factory pour créer des visualiseurs.
    
    Cette classe gère la création de différents types de visualisations
    pour les explications.
    """
    
    @staticmethod
    def create(chart_type: str, **kwargs):
        """
        Crée un visualiseur pour le type de graphique spécifié.
        
        Args:
            chart_type: Type de graphique ('bar', 'line', 'heatmap', etc.)
            **kwargs: Arguments de configuration du graphique
            
        Returns:
            Visualiseur approprié
            
        Raises:
            ValueError: Si le type de graphique n'est pas supporté
        """
        from ..visualizations import ChartGenerator
        
        generator = ChartGenerator()
        
        # Validation du type de graphique
        valid_types = [
            'bar', 'line', 'pie', 'scatter', 'heatmap', 
            'radar', 'sankey', 'waterfall', 'boxplot', 
            'histogram', 'treemap', 'gauge', 'table'
        ]
        
        if chart_type not in valid_types:
            raise ValueError(
                f"Type de graphique non supporté: {chart_type}. "
                f"Types valides: {', '.join(valid_types)}"
            )
        
        return generator
    
    @staticmethod
    def list_available_charts() -> list:
        """
        Liste tous les types de graphiques disponibles.
        
        Returns:
            list: Liste des types de graphiques
        """
        return [
            'bar', 'line', 'pie', 'scatter', 'heatmap',
            'radar', 'sankey', 'waterfall', 'boxplot',
            'histogram', 'treemap', 'gauge', 'table'
        ]
    
    @staticmethod
    def get_recommended_chart(explanation_type: str) -> str:
        """
        Recommande le meilleur type de graphique pour un type d'explication.
        
        Args:
            explanation_type: Type d'explication ('feature_importance', 'shap_values', etc.)
            
        Returns:
            str: Type de graphique recommandé
        """
        recommendations = {
            'feature_importance': 'bar',
            'shap_values': 'waterfall',
            'lime_weights': 'bar',
            'partial_dependence': 'line',
            'interaction': 'heatmap',
            'counterfactual': 'radar',
            'uncertainty': 'boxplot',
            'distribution': 'histogram',
            'comparison': 'radar',
            'hierarchy': 'treemap',
            'flow': 'sankey',
            'metric': 'gauge',
            'data': 'table'
        }
        
        return recommendations.get(explanation_type, 'bar')
