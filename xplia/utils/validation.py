"""
Fonctions de validation
========================

Utilitaires pour valider les entrées et modèles.
"""

import numpy as np
import pandas as pd
from typing import Any, Union, List, Optional
import logging

logger = logging.getLogger(__name__)


def validate_input(X: Any, expected_shape: Optional[tuple] = None, 
                   allow_1d: bool = False) -> np.ndarray:
    """
    Valide et convertit les données d'entrée en format numpy.
    
    Args:
        X: Données d'entrée
        expected_shape: Forme attendue (optionnel)
        allow_1d: Si True, autorise les tableaux 1D
        
    Returns:
        np.ndarray: Données validées
        
    Raises:
        ValueError: Si les données ne sont pas valides
    """
    # Conversion en numpy array
    if isinstance(X, pd.DataFrame):
        X = X.values
    elif isinstance(X, list):
        X = np.array(X)
    elif not isinstance(X, np.ndarray):
        raise ValueError(f"Type de données non supporté: {type(X)}")
    
    # Validation de la forme
    if not allow_1d and X.ndim == 1:
        X = X.reshape(1, -1)
    
    if expected_shape is not None:
        if X.shape != expected_shape:
            logger.warning(f"Forme des données ({X.shape}) différente de la forme attendue ({expected_shape})")
    
    # Vérification des valeurs manquantes
    if np.any(np.isnan(X)):
        logger.warning("Données contenant des valeurs manquantes (NaN)")
    
    return X


def validate_model(model: Any, required_methods: Optional[List[str]] = None) -> bool:
    """
    Valide qu'un modèle possède les méthodes requises.
    
    Args:
        model: Modèle à valider
        required_methods: Liste des méthodes requises
        
    Returns:
        bool: True si le modèle est valide
        
    Raises:
        ValueError: Si le modèle ne possède pas les méthodes requises
    """
    if required_methods is None:
        required_methods = ['predict']
    
    missing_methods = []
    for method in required_methods:
        if not hasattr(model, method):
            missing_methods.append(method)
    
    if missing_methods:
        raise ValueError(
            f"Le modèle ne possède pas les méthodes requises: {', '.join(missing_methods)}"
        )
    
    return True


def validate_feature_names(feature_names: Union[List[str], np.ndarray], 
                           n_features: int) -> List[str]:
    """
    Valide et normalise les noms de features.
    
    Args:
        feature_names: Noms des features
        n_features: Nombre de features attendu
        
    Returns:
        List[str]: Noms de features validés
        
    Raises:
        ValueError: Si le nombre de noms ne correspond pas
    """
    if feature_names is None:
        return [f"feature_{i}" for i in range(n_features)]
    
    if len(feature_names) != n_features:
        raise ValueError(
            f"Nombre de noms de features ({len(feature_names)}) "
            f"différent du nombre de features ({n_features})"
        )
    
    return list(feature_names)
