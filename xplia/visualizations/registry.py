"""
Registre des visualiseurs XPLIA
==============================

Ce module fournit un registre pour les visualiseurs XPLIA, permettant
l'enregistrement et la récupération des différentes classes de visualisation.
"""

import logging
from typing import Dict, Type, Any, Optional
from functools import wraps

from ..core.enums import ExplainabilityMethod
from .base import VisualizerBase

# Registre global des visualiseurs
_VISUALIZER_REGISTRY: Dict[str, Type[VisualizerBase]] = {}
logger = logging.getLogger(__name__)


def register_visualizer(explainability_method: ExplainabilityMethod):
    """
    Décorateur pour enregistrer un visualiseur dans le registre global.
    
    Args:
        explainability_method: Méthode d'explicabilité associée au visualiseur
        
    Returns:
        Fonction décorateur
    """
    def decorator(cls):
        if not issubclass(cls, VisualizerBase):
            raise TypeError(f"La classe {cls.__name__} doit hériter de VisualizerBase")
        
        method_name = explainability_method.value
        _VISUALIZER_REGISTRY[method_name] = cls
        logger.info(f"Visxplialiseur {cls.__name__} enregistré pour la méthode {method_name}")
        
        @wraps(cls)
        def wrapper(*args, **kwargs):
            return cls(*args, **kwargs)
        
        return wrapper
    
    return decorator


def get_visualizer(explainability_method: ExplainabilityMethod, **kwargs) -> Optional[VisualizerBase]:
    """
    Récupère une instance de visualiseur pour la méthode d'explicabilité spécifiée.
    
    Args:
        explainability_method: Méthode d'explicabilité pour laquelle récupérer un visualiseur
        **kwargs: Paramètres à passer au constructeur du visualiseur
        
    Returns:
        Instance de visualiseur ou None si aucun visualiseur n'est enregistré
    """
    method_name = explainability_method.value
    
    if method_name in _VISUALIZER_REGISTRY:
        visxplializer_cls = _VISUALIZER_REGISTRY[method_name]
        return visxplializer_cls(**kwargs)
    else:
        logger.warning(f"Aucun visualiseur enregistré pour la méthode {method_name}")
        return None


def list_available_visualizers() -> Dict[str, str]:
    """
    Liste tous les visualiseurs disponibles.
    
    Returns:
        Dictionnaire des méthodes d'explicabilité et noms de classes de visualiseurs
    """
    return {method: cls.__name__ for method, cls in _VISUALIZER_REGISTRY.items()}
