"""
Registre des explainers multimodaux
===================================

Ce module fournit un mécanisme de registre pour découvrir et 
enregistrer dynamiquement les explainers multimodaux.
"""

import inspect
from typing import Any, Callable, Dict, List, Type, Set, Optional

from .base import MultimodalExplainerBase, DataModality


# Registre global des explainers multimodaux
_MULTIMODAL_EXPLAINER_REGISTRY: Dict[str, Type[MultimodalExplainerBase]] = {}
_MULTIMODAL_EXPLAINER_BY_MODALITY: Dict[DataModality, List[Type[MultimodalExplainerBase]]] = {
    modality: [] for modality in DataModality
}


def register_multimodal_explainer(name: Optional[str] = None) -> Callable:
    """
    Décorateur pour enregistrer un explainer multimodal dans le registre global.
    
    Args:
        name: Nom d'enregistrement optionnel (par défaut: nom de la classe)
        
    Returns:
        Décorateur d'enregistrement
    """
    def decorator(cls: Type[MultimodalExplainerBase]) -> Type[MultimodalExplainerBase]:
        # Vérifier que la classe hérite bien de MultimodalExplainerBase
        if not inspect.isclass(cls) or not issubclass(cls, MultimodalExplainerBase):
            raise TypeError(
                f"Le décorateur register_multimodal_explainer ne peut être appliqué qu'à "
                f"des classes dérivant de MultimodalExplainerBase, pas à {cls.__name__}"
            )
        
        # Déterminer le nom d'enregistrement
        registry_name = name or cls.__name__
        
        # Enregistrer l'explainer dans le registre global
        if registry_name in _MULTIMODAL_EXPLAINER_REGISTRY:
            raise ValueError(
                f"Un explainer multimodal nommé '{registry_name}' existe déjà dans le registre"
            )
        
        _MULTIMODAL_EXPLAINER_REGISTRY[registry_name] = cls
        
        # Enregistrer l'explainer par modalité
        for modality in cls.supported_modalities:
            _MULTIMODAL_EXPLAINER_BY_MODALITY[modality].append(cls)
        
        # Ajouter un attribut à la classe pour faciliter l'introspection
        cls._registry_name = registry_name
        
        return cls
    
    return decorator


def get_multimodal_explainer(name: str) -> Type[MultimodalExplainerBase]:
    """
    Récupère un explainer multimodal par son nom.
    
    Args:
        name: Nom de l'explainer à récupérer
        
    Returns:
        Classe d'explainer multimodal
        
    Raises:
        KeyError: Si aucun explainer avec ce nom n'est enregistré
    """
    if name not in _MULTIMODAL_EXPLAINER_REGISTRY:
        raise KeyError(f"Aucun explainer multimodal nommé '{name}' n'est enregistré")
    
    return _MULTIMODAL_EXPLAINER_REGISTRY[name]


def list_multimodal_explainers() -> List[str]:
    """
    Liste tous les explainers multimodaux enregistrés.
    
    Returns:
        Liste des noms d'explainers multimodaux enregistrés
    """
    return list(_MULTIMODAL_EXPLAINER_REGISTRY.keys())


def find_explainers_for_modalities(modalities: Set[DataModality]) -> List[Type[MultimodalExplainerBase]]:
    """
    Trouve tous les explainers supportant un ensemble de modalités.
    
    Args:
        modalities: Ensemble des modalités requises
        
    Returns:
        Liste des classes d'explainers supportant toutes les modalités spécifiées
    """
    candidates = []
    
    for explainer_cls in _MULTIMODAL_EXPLAINER_REGISTRY.values():
        if modalities.issubset(explainer_cls.supported_modalities):
            candidates.append(explainer_cls)
    
    return candidates


def create_multimodal_explainer(modalities: Set[DataModality], 
                               model: Any, 
                               **kwargs) -> MultimodalExplainerBase:
    """
    Crée automatiquement un explainer multimodal adapté aux modalités spécifiées.
    
    Args:
        modalities: Ensemble des modalités à expliquer
        model: Modèle à expliquer
        **kwargs: Arguments additionnels à passer au constructeur de l'explainer
        
    Returns:
        Instance d'explainer multimodal
        
    Raises:
        ValueError: Si aucun explainer ne supporte les modalités spécifiées
    """
    candidates = find_explainers_for_modalities(modalities)
    
    if not candidates:
        raise ValueError(
            f"Aucun explainer multimodal ne supporte toutes les modalités: {modalities}"
        )
    
    # Choisir le premier explainer compatible
    # Note: Une logique plus sophistiquée pourrait être implémentée ici
    explainer_cls = candidates[0]
    
    return explainer_cls(model=model, modalities=modalities, **kwargs)
