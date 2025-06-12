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
