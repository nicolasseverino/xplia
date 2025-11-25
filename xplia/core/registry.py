"""
Système avancé de registre pour XPLIA
==================================

Ce module implémente un système de registre hautement extensible permettant 
d'enregistrer et de découvrir dynamiquement les explainers, visualiseurs et autres
composants de la librairie XPLIA.

Fonctionnalités avancées:
- Enregistrement transparent et automatique des composants
- Support pour métadonnées riches et annotations
- Gestion des dépendances et des incompatibilités entre composants
- Recherche sémantique et filtrage avancé des composants
- Versionning et compatibilité entre composants du framework
- Audit automatique et surveillance des performances
- Intégration fluide avec les systèmes d'observabilité
- Support pour la découverte à chaud de nouveaux plugins
- Mécanisme de substitution conditionnelle des composants
- Documentation automatique des interfaces et capacités

Ce système de registre forme la colonne vertébrale architecturale du framework XPLIA,
permettant son évolution et son extensibilité sans compromettre la stabilité
ou la cohérence de l'ensemble du système.
"""

import inspect
import sys
import logging
import importlib
import json
import threading
import time
import datetime
from functools import wraps, lru_cache
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union, Tuple, NamedTuple, Protocol
import pkg_resources

# Logger dédié au système de registre
logger = logging.getLogger(__name__)

# Type générique pour nos décorateurs
T = TypeVar('T')


# Classe pour représenter la version des composants
class Version:
    """Gestion sémantique des versions pour les composants XPLIA."""
    
    def __init__(self, version_str: str):
        parts = version_str.split('.')
        self.major = int(parts[0]) if len(parts) > 0 else 0
        self.minor = int(parts[1]) if len(parts) > 1 else 0
        self.patch = int(parts[2]) if len(parts) > 2 else 0
    
    def __str__(self):
        return f"{self.major}.{self.minor}.{self.patch}"
    
    def __eq__(self, other):
        if not isinstance(other, Version):
            return False
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)
    
    def __lt__(self, other):
        if not isinstance(other, Version):
            return NotImplemented
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)


# Types de composants disponibles dans XPLIA
class ComponentType(Enum):
    EXPLAINER = "explainer"
    VISUALIZER = "visualizer"
    MODEL_ADAPTER = "model_adapter"
    COMPLIANCE_CHECKER = "compliance_checker"
    FEATURE_EXTRACTOR = "feature_extractor"
    REPORT_GENERATOR = "report_generator"
    DATA_CONNECTOR = "data_connector"
    PLUGIN = "plugin"
    

@dataclass
class ComponentMetadata:
    """Métadonnées riches pour les composants du registre."""
    name: str
    component_type: ComponentType
    version: Version
    description: str = ""
    author: str = ""
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    dependencies: Dict[str, Version] = field(default_factory=dict)
    incompatibilities: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    capabilities: Dict[str, Any] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    configuration_schema: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    is_deprecated: bool = False
    replacement: Optional[str] = None
    priority: int = 0  # Priorité pour la résolution des conflits
    extra_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convertit les métadonnées en dictionnaire."""
        result = {
            "name": self.name,
            "component_type": self.component_type.value,
            "version": str(self.version),
            "description": self.description,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "dependencies": {k: str(v) for k, v in self.dependencies.items()},
            "incompatibilities": self.incompatibilities,
            "tags": self.tags,
            "enabled": self.enabled,
            "is_deprecated": self.is_deprecated,
            "priority": self.priority
        }
        
        # Inclure les champs non vides optionnels
        if self.capabilities:
            result["capabilities"] = self.capabilities
        if self.examples:
            result["examples"] = self.examples
        if self.performance_metrics:
            result["performance_metrics"] = self.performance_metrics
        if self.configuration_schema:
            result["configuration_schema"] = self.configuration_schema
        if self.replacement:
            result["replacement"] = self.replacement
        if self.extra_data:
            result["extra_data"] = self.extra_data
            
        return result


@dataclass
class ComponentRegistry:
    """Registre avancé pour un type spécifique de composants."""
    component_type: ComponentType
    components: Dict[str, Tuple[Type, ComponentMetadata]] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)
    
    def register(self, component_class: Type, metadata: ComponentMetadata) -> None:
        """Enregistre un composant avec ses métadonnées."""
        with self._lock:
            self.components[metadata.name] = (component_class, metadata)
            self._update_dependency_graph()
            logger.debug(f"Component registered: {metadata.name} ({metadata.component_type.value})")
    
    def unregister(self, name: str) -> bool:
        """Désenregistre un composant par son nom."""
        with self._lock:
            if name in self.components:
                del self.components[name]
                self._update_dependency_graph()
                logger.debug(f"Component unregistered: {name}")
                return True
            return False
    
    def get(self, name: str) -> Optional[Tuple[Type, ComponentMetadata]]:
        """Récupère un composant par son nom."""
        with self._lock:
            return self.components.get(name)
    
    def get_all(self) -> List[Tuple[Type, ComponentMetadata]]:
        """Récupère tous les composants enregistrés."""
        with self._lock:
            return list(self.components.values())
    
    def find(self, **criteria) -> List[Tuple[Type, ComponentMetadata]]:
        """Recherche des composants selon des critères spécifiques."""
        results = []
        with self._lock:
            for component_class, metadata in self.components.values():
                match = True
                for key, value in criteria.items():
                    # Vérification spéciale pour les tags (liste)
                    if key == "tags" and isinstance(value, list):
                        if not all(tag in metadata.tags for tag in value):
                            match = False
                            break
                    # Vérification spéciale pour les capacités (dict)
                    elif key == "capabilities" and isinstance(value, dict):
                        for cap_key, cap_value in value.items():
                            if cap_key not in metadata.capabilities or metadata.capabilities[cap_key] != cap_value:
                                match = False
                                break
                    # Vérification standard pour les autres attributs
                    elif not hasattr(metadata, key) or getattr(metadata, key) != value:
                        match = False
                        break
                if match:
                    results.append((component_class, metadata))
        return results
    
    def _update_dependency_graph(self) -> None:
        """Met à jour le graphe des dépendances entre composants."""
        # Cette méthode serait implémentée pour gérer les relations complexes
        # entre les composants et vérifier les conflits potentiels
        pass


# Structure de registre global avec typage fort
_REGISTRY = {
    ComponentType.EXPLAINER: ComponentRegistry(ComponentType.EXPLAINER),
    ComponentType.VISUALIZER: ComponentRegistry(ComponentType.VISUALIZER),
    ComponentType.MODEL_ADAPTER: ComponentRegistry(ComponentType.MODEL_ADAPTER),
    ComponentType.COMPLIANCE_CHECKER: ComponentRegistry(ComponentType.COMPLIANCE_CHECKER),
    ComponentType.FEATURE_EXTRACTOR: ComponentRegistry(ComponentType.FEATURE_EXTRACTOR),
    ComponentType.REPORT_GENERATOR: ComponentRegistry(ComponentType.REPORT_GENERATOR),
    ComponentType.DATA_CONNECTOR: ComponentRegistry(ComponentType.DATA_CONNECTOR),
    ComponentType.PLUGIN: ComponentRegistry(ComponentType.PLUGIN),
}


# Pour la compatibilité avec le code existant
_EXPLAINER_REGISTRY: Set[Type] = set()
_VISUALIZER_REGISTRY: Set[Type] = set()
_MODEL_ADAPTER_REGISTRY: Set[Type] = set()
_COMPLIANCE_CHECKER_REGISTRY: Set[Type] = set()


def register_explainer(cls: Optional[Type] = None, *, version: str = "1.0.0", description: str = "",
                    author: str = "", tags: List[str] = None, capabilities: Dict[str, Any] = None,
                    dependencies: Dict[str, str] = None, examples: List[str] = None,
                    priority: int = 0, configuration_schema: Dict[str, Any] = None):
    """
    Décorateur avancé pour enregistrer une classe d'explainer avec métadonnées enrichies.
    
    Peut être utilisé comme:
        @register_explainer
        class MyExplainer(ExplainerBase):
            ...
    
    Ou avec des métadonnées détaillées:
        @register_explainer(
            version="1.2.0",
            description="Un explainer spécialisé pour les modèles de traduction",
            author="Equipe XPLIA",
            tags=["nlp", "translation", "transformer"],
            capabilities={"multilingual": True, "batch_processing": True},
            dependencies={"transformers": "4.0.0"},
            examples=["examples/translation_explanation.py"],
            priority=10
        )
        class TranslationExplainer(ExplainerBase):
            ...
    
    Args:
        cls: Classe à enregistrer (optionnel)
        version: Version sémantique du composant
        description: Description détaillée du composant
        author: Auteur ou équipe responsable
        tags: Mots-clés pour la catégorisation et la recherche
        capabilities: Capacités spécifiques offertes par ce composant
        dependencies: Dépendances requises avec versions minimales
        examples: Liste des exemples d'utilisation
        priority: Priorité pour la résolution de conflits (plus élevé = plus prioritaire)
        configuration_schema: Schéma JSON des options de configuration
        
    Returns:
        Union[Type, Callable]: Classe enregistrée ou décorateur
    """
    # Initialisation des valeurs par défaut
    tags = tags or []
    capabilities = capabilities or {}
    dependencies = dependencies or {}
    examples = examples or []
    configuration_schema = configuration_schema or {}
    
    def _register(cls_inner):
        # Conservation de la compatibilité avec l'ancien système
        if cls_inner not in _EXPLAINER_REGISTRY:
            _EXPLAINER_REGISTRY.add(cls_inner)
        
        # Extraction du nom du composant
        component_name = getattr(cls_inner, "__name__", str(cls_inner))
        
        # Préparation des métadonnées
        metadata = ComponentMetadata(
            name=component_name,
            component_type=ComponentType.EXPLAINER,
            version=Version(version),
            description=description,
            author=author,
            tags=tags,
            capabilities=capabilities,
            dependencies={k: Version(v) for k, v in dependencies.items()},
            examples=examples,
            priority=priority,
            configuration_schema=configuration_schema
        )
        
        # Enregistrement dans le nouveau système de registre
        _REGISTRY[ComponentType.EXPLAINER].register(cls_inner, metadata)
        
        return cls_inner
    
    # Utilisé sans paramètres
    if cls is not None:
        return _register(cls)
    
    # Utilisé avec paramètres
    return _register


def register_visualizer(cls: Optional[Type] = None, *, version: str = "1.0.0", description: str = "",
                      author: str = "", tags: List[str] = None, capabilities: Dict[str, Any] = None,
                      dependencies: Dict[str, str] = None, examples: List[str] = None,
                      priority: int = 0, configuration_schema: Dict[str, Any] = None,
                      supported_formats: List[str] = None):
    """
    Décorateur avancé pour enregistrer une classe de visualiseur avec métadonnées enrichies.
    
    Peut être utilisé comme:
        @register_visualizer
        class MyVisualizer(VisualizerBase):
            ...
    
    Ou avec des métadonnées détaillées:
        @register_visualizer(
            version="1.2.0",
            description="Visualiseur interactif de graphes",
            author="Equipe XPLIA",
            tags=["graph", "interactive", "network"],
            capabilities={"interactive": True, "export_svg": True},
            supported_formats=["html", "svg", "png"],
            priority=10
        )
        class GraphVisualizer(VisualizerBase):
            ...
    
    Args:
        cls: Classe à enregistrer (optionnel)
        version: Version sémantique du composant
        description: Description détaillée du composant
        author: Auteur ou équipe responsable
        tags: Mots-clés pour la catégorisation et la recherche
        capabilities: Capacités spécifiques offertes par ce visualiseur
        dependencies: Dépendances requises avec versions minimales
        examples: Liste des exemples d'utilisation
        priority: Priorité pour la résolution de conflits
        configuration_schema: Schéma JSON des options de configuration
        supported_formats: Formats d'export supportés par ce visualiseur
        
    Returns:
        Union[Type, Callable]: Classe enregistrée ou décorateur
    """
    # Initialisation des valeurs par défaut
    tags = tags or []
    capabilities = capabilities or {}
    dependencies = dependencies or {}
    examples = examples or []
    configuration_schema = configuration_schema or {}
    supported_formats = supported_formats or ["html"]
    
    # Ajout des formats supportés aux capacités
    if supported_formats:
        capabilities["supported_formats"] = supported_formats
    
    def _register(cls_inner):
        # Conservation de la compatibilité avec l'ancien système
        if cls_inner not in _VISUALIZER_REGISTRY:
            _VISUALIZER_REGISTRY.add(cls_inner)
        
        # Extraction du nom du composant
        component_name = getattr(cls_inner, "__name__", str(cls_inner))
        
        # Préparation des métadonnées
        metadata = ComponentMetadata(
            name=component_name,
            component_type=ComponentType.VISUALIZER,
            version=Version(version),
            description=description,
            author=author,
            tags=tags,
            capabilities=capabilities,
            dependencies={k: Version(v) for k, v in dependencies.items()},
            examples=examples,
            priority=priority,
            configuration_schema=configuration_schema
        )
        
        # Enregistrement dans le nouveau système de registre
        _REGISTRY[ComponentType.VISUALIZER].register(cls_inner, metadata)
        
        return cls_inner
    
    # Utilisé sans paramètres
    if cls is not None:
        return _register(cls)
    
    # Utilisé avec paramètres
    return _register


def register_model_adapter(cls: Optional[Type] = None, *, version: str = "1.0.0", description: str = "",
                         author: str = "", tags: List[str] = None, capabilities: Dict[str, Any] = None,
                         dependencies: Dict[str, str] = None, examples: List[str] = None,
                         priority: int = 0, configuration_schema: Dict[str, Any] = None,
                         supported_frameworks: List[str] = None, model_types: List[str] = None):
    """
    Décorateur avancé pour enregistrer un adaptateur de modèle avec métadonnées enrichies.
    
    Peut être utilisé comme:
        @register_model_adapter
        class MyModelAdapter(ModelAdapterBase):
            ...
    
    Ou avec des métadonnées détaillées:
        @register_model_adapter(
            version="1.2.0",
            description="Adaptateur pour modèles PyTorch",
            author="Equipe XPLIA",
            tags=["pytorch", "deep-learning"],
            supported_frameworks=["pytorch"],
            model_types=["cnn", "transformer", "rnn"],
            capabilities={"gradient_extraction": True, "feature_visualization": True},
            priority=10
        )
        class PyTorchAdapter(ModelAdapterBase):
            ...
    
    Args:
        cls: Classe à enregistrer (optionnel)
        version: Version sémantique du composant
        description: Description détaillée du composant
        author: Auteur ou équipe responsable
        tags: Mots-clés pour la catégorisation et la recherche
        capabilities: Capacités spécifiques offertes par cet adaptateur
        dependencies: Dépendances requises avec versions minimales
        examples: Liste des exemples d'utilisation
        priority: Priorité pour la résolution de conflits
        configuration_schema: Schéma JSON des options de configuration
        supported_frameworks: Frameworks ML/DL supportés ("pytorch", "tensorflow", etc.)
        model_types: Types de modèles supportés ("cnn", "rnn", etc.)
        
    Returns:
        Union[Type, Callable]: Classe enregistrée ou décorateur
    """
    # Initialisation des valeurs par défaut
    tags = tags or []
    capabilities = capabilities or {}
    dependencies = dependencies or {}
    examples = examples or []
    configuration_schema = configuration_schema or {}
    supported_frameworks = supported_frameworks or []
    model_types = model_types or []
    
    # Enrichissement des capacités
    if supported_frameworks:
        capabilities["supported_frameworks"] = supported_frameworks
    if model_types:
        capabilities["model_types"] = model_types
    
    def _register(cls_inner):
        # Conservation de la compatibilité avec l'ancien système
        if cls_inner not in _MODEL_ADAPTER_REGISTRY:
            _MODEL_ADAPTER_REGISTRY.add(cls_inner)
        
        # Extraction du nom du composant
        component_name = getattr(cls_inner, "__name__", str(cls_inner))
        
        # Préparation des métadonnées
        metadata = ComponentMetadata(
            name=component_name,
            component_type=ComponentType.MODEL_ADAPTER,
            version=Version(version),
            description=description,
            author=author,
            tags=tags,
            capabilities=capabilities,
            dependencies={k: Version(v) for k, v in dependencies.items()},
            examples=examples,
            priority=priority,
            configuration_schema=configuration_schema
        )
        
        # Enregistrement dans le nouveau système de registre
        _REGISTRY[ComponentType.MODEL_ADAPTER].register(cls_inner, metadata)
        
        return cls_inner
    
    if cls is not None:
        return _register(cls)
    
    return _register


def register_compliance_checker(cls: Optional[Type] = None, *, version: str = "1.0.0", description: str = "",
                              author: str = "", tags: List[str] = None, capabilities: Dict[str, Any] = None,
                              dependencies: Dict[str, str] = None, examples: List[str] = None,
                              priority: int = 0, configuration_schema: Dict[str, Any] = None,
                              supported_regulations: List[str] = None, certification_level: str = None):
    """
    Décorateur avancé pour enregistrer un vérificateur de conformité avec métadonnées enrichies.
    
    Peut être utilisé comme:
        @register_compliance_checker
        class MyComplianceChecker(ComplianceCheckerBase):
            ...
    
    Ou avec des métadonnées détaillées:
        @register_compliance_checker(
            version="2.0.0",
            description="Vérificateur de conformité pour l'AI Act européen",
            author="Equipe XPLIA",
            tags=["ai_act", "europe", "certification"],
            supported_regulations=["ai_act", "gdpr"],
            certification_level="official",
            capabilities={"audit_trail": True, "compliance_reporting": True},
            priority=90
        )
        class AIActComplianceChecker(ComplianceCheckerBase):
            ...
    
    Args:
        cls: Classe à enregistrer (optionnel)
        version: Version sémantique du composant
        description: Description détaillée du composant
        author: Auteur ou équipe responsable
        tags: Mots-clés pour la catégorisation et la recherche
        capabilities: Capacités spécifiques offertes par ce vérificateur
        dependencies: Dépendances requises avec versions minimales
        examples: Liste des exemples d'utilisation
        priority: Priorité pour la résolution de conflits (important pour les certifications)
        configuration_schema: Schéma JSON des options de configuration
        supported_regulations: Réglementations prises en charge ("gdpr", "ai_act", "hipaa", etc.)
        certification_level: Niveau de certification ("self_assessed", "third_party", "official")
        
    Returns:
        Union[Type, Callable]: Classe enregistrée ou décorateur
    """
    # Initialisation des valeurs par défaut
    tags = tags or []
    capabilities = capabilities or {}
    dependencies = dependencies or {}
    examples = examples or []
    configuration_schema = configuration_schema or {}
    supported_regulations = supported_regulations or []
    
    # Enrichissement des capacités
    if supported_regulations:
        capabilities["supported_regulations"] = supported_regulations
    if certification_level:
        capabilities["certification_level"] = certification_level
    
    def _register(cls_inner):
        # Conservation de la compatibilité avec l'ancien système
        if cls_inner not in _COMPLIANCE_CHECKER_REGISTRY:
            _COMPLIANCE_CHECKER_REGISTRY.add(cls_inner)
        
        # Extraction du nom du composant
        component_name = getattr(cls_inner, "__name__", str(cls_inner))
        
        # Préparation des métadonnées
        metadata = ComponentMetadata(
            name=component_name,
            component_type=ComponentType.COMPLIANCE_CHECKER,
            version=Version(version),
            description=description,
            author=author,
            tags=tags,
            capabilities=capabilities,
            dependencies={k: Version(v) for k, v in dependencies.items()},
            examples=examples,
            priority=priority,
            configuration_schema=configuration_schema
        )
        
        # Enregistrement dans le nouveau système de registre
        _REGISTRY[ComponentType.COMPLIANCE_CHECKER].register(cls_inner, metadata)
        
        return cls_inner
    
    if cls is not None:
        return _register(cls)
    
    return _register


def get_registered_explainers(tags: List[str] = None, capability: str = None, 
                          include_metadata: bool = False) -> List[Union[Type, Tuple[Type, ComponentMetadata]]]:
    """
    Récupère tous les explainers enregistrés avec filtrage avancé.
    
    Args:
        tags: Liste de tags pour filtrer les explainers (optionnel)
        capability: Capacité spécifique requise (optionnel)
        include_metadata: Si True, retourne aussi les métadonnées associées
    
    Returns:
        Si include_metadata=False:
            List[Type]: Liste des classes d'explainers enregistrées
        Sinon:
            List[Tuple[Type, ComponentMetadata]]: Liste des classes avec leurs métadonnées
    """
    # Conservation de la compatibilité avec l'ancien code
    if not tags and not capability and not include_metadata:
        return list(_EXPLAINER_REGISTRY)
    
    # Utilisation du nouveau système de registre avec filtrage
    registry = _REGISTRY[ComponentType.EXPLAINER]
    
    # Préparation des critères de filtrage
    criteria = {}
    if tags:
        criteria["tags"] = tags
    if capability:
        criteria["capabilities"] = {capability: True}
    
    # Récupération des résultats
    if criteria:
        results = registry.find(**criteria)
    else:
        results = registry.get_all()
    
    # Format de retour selon include_metadata
    if include_metadata:
        return [result for result in results if result[0] not in _EXPLAINER_REGISTRY.difference(result[0] for result, _ in results)]
    else:
        return [cls for cls, _ in results if cls not in _EXPLAINER_REGISTRY.difference(cls for cls, _ in results)]


def get_registered_visualizers(tags: List[str] = None, capability: str = None, 
                           supported_format: str = None, include_metadata: bool = False) -> List[Union[Type, Tuple[Type, ComponentMetadata]]]:
    """
    Récupère tous les visualiseurs enregistrés avec filtrage avancé.
    
    Args:
        tags: Liste de tags pour filtrer les visualiseurs (optionnel)
        capability: Capacité spécifique requise (optionnel)
        supported_format: Format d'export spécifique requis ("html", "svg", etc.)
        include_metadata: Si True, retourne aussi les métadonnées associées
    
    Returns:
        Si include_metadata=False:
            List[Type]: Liste des classes de visualiseurs enregistrées
        Sinon:
            List[Tuple[Type, ComponentMetadata]]: Liste des classes avec leurs métadonnées
    """
    # Conservation de la compatibilité avec l'ancien code
    if not any([tags, capability, supported_format, include_metadata]):
        return list(_VISUALIZER_REGISTRY)
    
    # Utilisation du nouveau système de registre avec filtrage
    registry = _REGISTRY[ComponentType.VISUALIZER]
    
    # Préparation des critères de filtrage
    criteria = {}
    if tags:
        criteria["tags"] = tags
    if capability:
        criteria["capabilities"] = {capability: True}
    
    # Récupération des résultats
    results = registry.find(**criteria) if criteria else registry.get_all()
    
    # Filtrage supplémentaire par format si demandé
    if supported_format and results:
        filtered_results = []
        for cls, metadata in results:
            if ("capabilities" in metadata.__dict__ and 
                "supported_formats" in metadata.capabilities and
                supported_format in metadata.capabilities["supported_formats"]):
                filtered_results.append((cls, metadata))
        results = filtered_results
    
    # Format de retour selon include_metadata
    if include_metadata:
        return results
    else:
        return [cls for cls, _ in results]


def get_registered_model_adapters(tags: List[str] = None, capability: str = None,
                              framework: str = None, model_type: str = None,
                              include_metadata: bool = False) -> List[Union[Type, Tuple[Type, ComponentMetadata]]]:
    """
    Récupère tous les adaptateurs de modèles enregistrés avec filtrage avancé.
    
    Args:
        tags: Liste de tags pour filtrer les adaptateurs (optionnel)
        capability: Capacité spécifique requise (optionnel)
        framework: Framework spécifique requis ("pytorch", "tensorflow", etc.)
        model_type: Type de modèle spécifique requis ("cnn", "rnn", etc.)
        include_metadata: Si True, retourne aussi les métadonnées associées
    
    Returns:
        Si include_metadata=False:
            List[Type]: Liste des classes d'adaptateurs enregistrées
        Sinon:
            List[Tuple[Type, ComponentMetadata]]: Liste des classes avec leurs métadonnées
    """
    # Conservation de la compatibilité avec l'ancien code
    if not any([tags, capability, framework, model_type, include_metadata]):
        return list(_MODEL_ADAPTER_REGISTRY)
    
    # Utilisation du nouveau système de registre
    registry = _REGISTRY[ComponentType.MODEL_ADAPTER]
    
    # Préparation des critères de base
    criteria = {}
    if tags:
        criteria["tags"] = tags
    if capability:
        criteria["capabilities"] = {capability: True}
    
    # Récupération des résultats initiaux
    results = registry.find(**criteria) if criteria else registry.get_all()
    
    # Filtrage supplémentaire si nécessaire
    if framework or model_type:
        filtered_results = []
        for cls, metadata in results:
            include = True
            if framework and ("capabilities" not in metadata.__dict__ or 
                           "supported_frameworks" not in metadata.capabilities or
                           framework not in metadata.capabilities["supported_frameworks"]):
                include = False
            if model_type and ("capabilities" not in metadata.__dict__ or
                            "model_types" not in metadata.capabilities or
                            model_type not in metadata.capabilities["model_types"]):
                include = False
            if include:
                filtered_results.append((cls, metadata))
        results = filtered_results
    
    # Format de retour
    if include_metadata:
        return results
    else:
        return [cls for cls, _ in results]


def get_registered_compliance_checkers(tags: List[str] = None, regulation: str = None,
                                   certification_level: str = None, include_metadata: bool = False) -> List[Union[Type, Tuple[Type, ComponentMetadata]]]:
    """
    Récupère tous les vérificateurs de conformité enregistrés avec filtrage avancé.
    
    Args:
        tags: Liste de tags pour filtrer les vérificateurs (optionnel)
        regulation: Réglementation spécifique requise ("gdpr", "ai_act", etc.)
        certification_level: Niveau de certification requis ("self_assessed", "official", etc.)
        include_metadata: Si True, retourne aussi les métadonnées associées
    
    Returns:
        Si include_metadata=False:
            List[Type]: Liste des classes de vérificateurs enregistrées
        Sinon:
            List[Tuple[Type, ComponentMetadata]]: Liste des classes avec leurs métadonnées
    """
    # Conservation de la compatibilité avec l'ancien code
    if not any([tags, regulation, certification_level, include_metadata]):
        return list(_COMPLIANCE_CHECKER_REGISTRY)
    
    # Utilisation du nouveau système de registre
    registry = _REGISTRY[ComponentType.COMPLIANCE_CHECKER]
    
    # Préparation des critères
    criteria = {}
    if tags:
        criteria["tags"] = tags
    
    # Récupération des résultats initiaux
    results = registry.find(**criteria) if criteria else registry.get_all()
    
    # Filtrage supplémentaire si nécessaire
    if regulation or certification_level:
        filtered_results = []
        for cls, metadata in results:
            include = True
            if regulation and ("capabilities" not in metadata.__dict__ or 
                           "supported_regulations" not in metadata.capabilities or
                           regulation not in metadata.capabilities["supported_regulations"]):
                include = False
            if certification_level and ("capabilities" not in metadata.__dict__ or
                                     "certification_level" not in metadata.capabilities or
                                     certification_level != metadata.capabilities["certification_level"]):
                include = False
            if include:
                filtered_results.append((cls, metadata))
        results = filtered_results
    
    # Format de retour
    if include_metadata:
        return results
    else:
        return [cls for cls, _ in results]


def build_dependency_graph() -> Dict[str, List[str]]:
    """
    Construit un graphe orienté des dépendances entre composants enregistrés.
    
    Analyse les métadonnées de dépendances de tous les composants enregistrés et
    construit une représentation sous forme de graphe pour détecter les cycles
    ou d'autres problèmes de dépendance.
    
    Returns:
        Dict[str, List[str]]: Graphe de dépendances où les clés sont les noms des composants
                             et les valeurs sont les listes des noms des composants dépendants
    """
    dependency_graph = {}
    
    # Analyse de tous les types de composants
    for component_type in ComponentType:
        registry = _REGISTRY[component_type]
        components = registry.get_all()
        
        for cls, metadata in components:
            component_name = metadata.name
            dependency_graph[component_name] = []
            
            # Vérifie les dépendances déclarées dans les métadonnées
            for dep_name, _ in metadata.dependencies.items():
                dependency_graph[component_name].append(dep_name)
    
    return dependency_graph


def detect_dependency_cycles() -> List[List[str]]:
    """
    Détecte les cycles de dépendance dans le graphe de dépendances.
    
    Utilise l'algorithme de recherche en profondeur pour détecter tous les cycles élémentaires
    dans le graphe orienté des dépendances.
    
    Returns:
        List[List[str]]: Liste des cycles détectés, chaque cycle étant une liste de noms de composants
    """
    graph = build_dependency_graph()
    cycles = []
    
    def find_cycles_from(node, path=None, visited=None):
        if path is None:
            path = []
        if visited is None:
            visited = set()
            
        if node in path:
            # Cycle détecté
            cycle_start = path.index(node)
            cycle = path[cycle_start:] + [node]
            cycles.append(cycle)
            return
            
        if node in visited:
            return
            
        visited.add(node)
        path.append(node)
        
        for neighbor in graph.get(node, []):
            find_cycles_from(neighbor, path[:], visited)
    
    # Lance la détection depuis chaque nœud
    for node in graph:
        find_cycles_from(node)
    
    return cycles


def validate_component_dependencies() -> Dict[str, List[Dict[str, Any]]]:
    """
    Valide que toutes les dépendances déclarées dans les métadonnées sont satisfaites.
    
    Vérifie pour chaque composant que:
    1. Les composants dépendants existent
    2. La version minimale requise est satisfaite
    3. Aucun cycle de dépendance n'existe
    
    Returns:
        Dict[str, List[Dict[str, Any]]]: Dictionnaire des erreurs de dépendance par composant
    """
    errors = {}
    
    # Récupération des composants de tous les registres
    all_components = {}
    for component_type in ComponentType:
        registry = _REGISTRY[component_type]
        for cls, metadata in registry.get_all():
            all_components[metadata.name] = (cls, metadata)
    
    # Vérification des dépendances pour chaque composant
    for component_name, (_, metadata) in all_components.items():
        component_errors = []
        
        # Vérifie chaque dépendance déclarée
        for dep_name, min_version in metadata.dependencies.items():
            # Vérifie l'existence de la dépendance
            if dep_name not in all_components:
                component_errors.append({
                    "type": "missing_dependency",
                    "dependency": dep_name,
                    "message": f"Dépendance '{dep_name}' non trouvée"
                })
                continue
                
            # Vérifie la version de la dépendance
            _, dep_metadata = all_components[dep_name]
            if dep_metadata.version < min_version:
                component_errors.append({
                    "type": "version_mismatch",
                    "dependency": dep_name,
                    "required": str(min_version),
                    "available": str(dep_metadata.version),
                    "message": f"Version requise '{min_version}' supérieure à '{dep_metadata.version}' disponible"
                })
        
        # Enregistre les erreurs si présentes
        if component_errors:
            errors[component_name] = component_errors
    
    # Détection des cycles
    cycles = detect_dependency_cycles()
    if cycles:
        for cycle in cycles:
            cycle_str = " -> ".join(cycle)
            for component in cycle:
                if component not in errors:
                    errors[component] = []
                errors[component].append({
                    "type": "dependency_cycle",
                    "cycle": cycle,
                    "message": f"Cycle de dépendance détecté: {cycle_str}"
                })
    
    return errors


def scan_modules_for_registrations(package_name="xplia", auto_register=False):
    """
    Analyse récursivement les modules de l'application pour enregistrer les classes décorées.
    
    Cette fonction parcourt tous les modules du package spécifié et recherche les classes
    qui devraient être enregistrées en se basant sur l'héritage ou les annotations.
    
    Args:
        package_name: Nom du package racine à scanner (par défaut: "xplia")
        auto_register: Si True, enregistre automatiquement les classes détectées
                       non encore enregistrées
                       
    Returns:
        Dict[ComponentType, List[Type]]: Dictionnaire des classes détectées par type de composant
    """
    import importlib
    import pkgutil
    import inspect
    import sys
    
    # Classes de base à rechercher par type de composant
    base_classes = {
        # Ces noms devront être adaptés aux noms réels des classes de base
        ComponentType.EXPLAINER: ["BaseExplainer", "ExplainerBase"],
        ComponentType.VISUALIZER: ["BaseVisualizer", "VisualizerBase"],
        ComponentType.MODEL_ADAPTER: ["BaseModelAdapter", "ModelAdapterBase"],
        ComponentType.COMPLIANCE_CHECKER: ["BaseComplianceChecker", "ComplianceCheckerBase"]
    }
    
    discovered_components = {ctype: [] for ctype in ComponentType}
    
    def import_submodules(package):
        """Importe récursivement tous les sous-modules d'un package."""
        if isinstance(package, str):
            package = importlib.import_module(package)
        results = {}
        for _, name, is_pkg in pkgutil.iter_modules(package.__path__, package.__name__ + '.'):
            try:
                results[name] = importlib.import_module(name)
                if is_pkg:
                    results.update(import_submodules(name))
            except Exception as e:
                print(f"Erreur lors de l'importation de {name}: {e}")
        return results
    
    try:
        # Importe récursivement tous les modules
        modules = import_submodules(package_name)
        
        # Scanner chaque module pour les classes potentielles
        for module_name, module in modules.items():
            for item_name, item in inspect.getmembers(module):
                # Vérifie si c'est une classe définie dans ce module
                if inspect.isclass(item) and item.__module__ == module.__name__:
                    # Vérifie l'héritage pour chaque type de composant
                    for ctype, base_class_names in base_classes.items():
                        for base_name in base_class_names:
                            # Vérifie si le nom de la classe parente correspond
                            for base_class in item.__mro__[1:]:  # Skip self
                                if base_class.__name__ == base_name:
                                    discovered_components[ctype].append(item)
                                    break
        
        # Enregistrement automatique si demandé
        if auto_register:
            for ctype, components in discovered_components.items():
                for component in components:
                    # Vérifie si déjà enregistré
                    is_registered = False
                    
                    # Vérifie dans l'ancien système de registre
                    if ctype == ComponentType.EXPLAINER and component in _EXPLAINER_REGISTRY:
                        is_registered = True
                    elif ctype == ComponentType.VISUALIZER and component in _VISUALIZER_REGISTRY:
                        is_registered = True
                    elif ctype == ComponentType.MODEL_ADAPTER and component in _MODEL_ADAPTER_REGISTRY:
                        is_registered = True
                    elif ctype == ComponentType.COMPLIANCE_CHECKER and component in _COMPLIANCE_CHECKER_REGISTRY:
                        is_registered = True
                    
                    # Vérifie dans le nouveau système de registre
                    for cls, _ in _REGISTRY[ctype].get_all():
                        if cls == component:
                            is_registered = True
                            break
                    
                    # Si non enregistré, enregistre avec des métadonnées par défaut
                    if not is_registered:
                        metadata = ComponentMetadata(
                            name=component.__name__,
                            component_type=ctype,
                            version=Version("1.0.0"),
                            description=component.__doc__ or f"Auto-discovered {ctype.name.lower()}",
                            author="Auto-discovery",
                            tags=["auto-discovered", ctype.name.lower()],
                            capabilities={},
                            dependencies={},
                            examples=[],
                            priority=0,
                            configuration_schema={}
                        )
                        _REGISTRY[ctype].register(component, metadata)
                        
                        # Mise à jour de l'ancien registre pour compatibilité
                        if ctype == ComponentType.EXPLAINER:
                            _EXPLAINER_REGISTRY.add(component)
                        elif ctype == ComponentType.VISUALIZER:
                            _VISUALIZER_REGISTRY.add(component)
                        elif ctype == ComponentType.MODEL_ADAPTER:
                            _MODEL_ADAPTER_REGISTRY.add(component)
                        elif ctype == ComponentType.COMPLIANCE_CHECKER:
                            _COMPLIANCE_CHECKER_REGISTRY.add(component)
    
    except Exception as e:
        print(f"Erreur lors du scan des modules: {e}")
    
    return discovered_components


# Déclencher l'enregistrement automatique lors de l'importation
# (commenté pour l'instant car les classes de base ne sont pas encore toutes définies)
# scan_modules_for_registrations()


class Registry:
    """
    Classe principale de registre pour XPLIA.
    
    Fournit une interface unifiée pour accéder à tous les types de composants
    enregistrés dans le système.
    """
    
    def __init__(self):
        """Initialise le registre global."""
        self._registries = _REGISTRY
    
    def get_explainers(self, **filters) -> List[Type]:
        """
        Récupère les explainers enregistrés.
        
        Args:
            **filters: Critères de filtrage (tags, capability, etc.)
            
        Returns:
            List[Type]: Liste des classes d'explainers
        """
        return get_registered_explainers(**filters)
    
    def get_visualizers(self, **filters) -> List[Type]:
        """
        Récupère les visualiseurs enregistrés.
        
        Args:
            **filters: Critères de filtrage
            
        Returns:
            List[Type]: Liste des classes de visualiseurs
        """
        return get_registered_visualizers(**filters)
    
    def get_model_adapters(self, **filters) -> List[Type]:
        """
        Récupère les adaptateurs de modèles enregistrés.
        
        Args:
            **filters: Critères de filtrage
            
        Returns:
            List[Type]: Liste des classes d'adaptateurs
        """
        return get_registered_model_adapters(**filters)
    
    def get_compliance_checkers(self, **filters) -> List[Type]:
        """
        Récupère les vérificateurs de conformité enregistrés.
        
        Args:
            **filters: Critères de filtrage
            
        Returns:
            List[Type]: Liste des classes de vérificateurs
        """
        return get_registered_compliance_checkers(**filters)
    
    def register(self, component_type: ComponentType, component_class: Type, 
                 metadata: Optional[ComponentMetadata] = None) -> None:
        """
        Enregistre un composant dans le registre approprié.
        
        Args:
            component_type: Type de composant
            component_class: Classe du composant
            metadata: Métadonnées optionnelles
        """
        if metadata is None:
            metadata = ComponentMetadata(
                name=component_class.__name__,
                component_type=component_type,
                version=Version("1.0.0"),
                description=component_class.__doc__ or ""
            )
        
        self._registries[component_type].register(component_class, metadata)
    
    def unregister(self, component_type: ComponentType, name: str) -> bool:
        """
        Désenregistre un composant.
        
        Args:
            component_type: Type de composant
            name: Nom du composant
            
        Returns:
            bool: True si le composant a été désenregistré
        """
        return self._registries[component_type].unregister(name)
    
    def get_component(self, component_type: ComponentType, name: str) -> Optional[Tuple[Type, ComponentMetadata]]:
        """
        Récupère un composant spécifique.
        
        Args:
            component_type: Type de composant
            name: Nom du composant
            
        Returns:
            Optional[Tuple[Type, ComponentMetadata]]: Composant et ses métadonnées
        """
        return self._registries[component_type].get(name)
    
    def list_all_components(self) -> Dict[ComponentType, List[str]]:
        """
        Liste tous les composants enregistrés par type.
        
        Returns:
            Dict[ComponentType, List[str]]: Dictionnaire des noms de composants par type
        """
        result = {}
        for ctype, registry in self._registries.items():
            result[ctype] = [metadata.name for _, metadata in registry.get_all()]
        return result
    
    def validate_dependencies(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Valide toutes les dépendances du registre.
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: Erreurs de dépendances par composant
        """
        return validate_component_dependencies()
    
    def detect_cycles(self) -> List[List[str]]:
        """
        Détecte les cycles de dépendances.
        
        Returns:
            List[List[str]]: Liste des cycles détectés
        """
        return detect_dependency_cycles()
    
    def export_metadata(self, component_type: Optional[ComponentType] = None) -> Dict[str, Any]:
        """
        Exporte les métadonnées de tous les composants.
        
        Args:
            component_type: Type de composant spécifique (optionnel)
            
        Returns:
            Dict[str, Any]: Métadonnées exportées
        """
        result = {}
        
        types_to_export = [component_type] if component_type else list(ComponentType)
        
        for ctype in types_to_export:
            registry = self._registries[ctype]
            result[ctype.value] = []
            for _, metadata in registry.get_all():
                result[ctype.value].append(metadata.to_dict())
        
        return result
    
    def clear(self, component_type: Optional[ComponentType] = None) -> None:
        """
        Vide le registre.
        
        Args:
            component_type: Type de composant à vider (optionnel, vide tout si None)
        """
        if component_type:
            self._registries[component_type].components.clear()
        else:
            for registry in self._registries.values():
                registry.components.clear()


# Instance globale du registre
_global_registry = Registry()
