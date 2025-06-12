"""
Classes de base et abstraites pour l'architecture de XPLIA
=============================================================

Ce module fournit les interfaces et classes abstraites qui servent
de fondation à l'ensemble du framework d'explicabilité.
"""

import abc
import datetime
import json
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd

from .config import ConfigManager


class ExplainabilityMethod(str, Enum):
    """Méthodes d'explicabilité supportées."""
    SHAP = "shap"
    LIME = "lime"
    ANCHORS = "anchors"
    COUNTERFACTUAL = "counterfactual"
    PDP = "partial_dependence"
    ICE = "ice"
    FEATURE_IMPORTANCE = "feature_importance"
    ATTENTION = "attention"
    GRADIENT = "gradient"
    INTEGRATED_GRADIENTS = "integrated_gradients"
    UNIFIED = "unified"


class AudienceLevel(str, Enum):
    """Niveaux d'audience pour l'explicabilité."""
    TECHNICAL = "technical"    # Pour data scientists et ingénieurs
    BUSINESS = "business"      # Pour décideurs et managers
    PUBLIC = "public"          # Pour le grand public et non-experts


class ModelType(str, Enum):
    """Types de modèles pris en charge."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    RECOMMENDER = "recommender"
    TIME_SERIES = "time_series"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    GENERATIVE = "generative"


class ConfigurableMixin:
    """Mixin qui ajoute des fonctionnalités de configuration."""
    
    def __init__(self, **kwargs):
        """
        Initialise l'objet configurable avec des paramètres.
        
        Args:
            **kwargs: Paramètres de configuration arbitraires
        """
        self._config = {}
        self._config.update(ConfigManager().get_default_config())
        self._config.update(kwargs)
    
    def configure(self, **kwargs):
        """
        Met à jour la configuration.
        
        Args:
            **kwargs: Paramètres de configuration à mettre à jour
            
        Returns:
            self: Pour permettre le chaînage
        """
        self._config.update(kwargs)
        return self
    
    def get_config(self):
        """
        Récupère la configuration actuelle.
        
        Returns:
            dict: Configuration actuelle
        """
        return self._config.copy()


class AuditableMixin:
    """Mixin qui ajoute des fonctionnalités d'audit et de traçabilité."""
    
    def __init__(self):
        """Initialise l'objet avec des capacités d'audit."""
        self._audit_trail = []
        self._creation_timestamp = datetime.datetime.now().isoformat()
        self._uuid = str(uuid.uuid4())
        
    def add_audit_record(self, action: str, details: Dict[str, Any]) -> None:
        """
        Ajoute un enregistrement d'audit.
        
        Args:
            action: Type d'action réalisée
            details: Détails sur l'action
        """
        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "action": action,
            "details": details
        }
        self._audit_trail.append(record)
    
    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """
        Récupère tout l'historique d'audit.
        
        Returns:
            list: Liste des enregistrements d'audit
        """
        return self._audit_trail.copy()
    
    def export_audit_trail(self, format: str = "json") -> Union[str, Dict]:
        """
        Exporte l'historique d'audit dans le format spécifié.
        
        Args:
            format: Format d'export ("json" ou "dict")
            
        Returns:
            Union[str, dict]: Historique d'audit au format demandé
        """
        if format.lower() == "json":
            return json.dumps({
                "uuid": self._uuid,
                "creation_timestamp": self._creation_timestamp,
                "audit_trail": self._audit_trail
            }, indent=2)
        return {
            "uuid": self._uuid,
            "creation_timestamp": self._creation_timestamp,
            "audit_trail": self._audit_trail
        }


@dataclass
class FeatureImportance:
    """Classe pour stocker les importances des caractéristiques."""
    feature_name: str
    importance: float
    confidence_interval: Optional[Tuple[float, float]] = None
    std_dev: Optional[float] = None
    p_value: Optional[float] = None
    
    def to_dict(self):
        """Convertit en dictionnaire."""
        return asdict(self)


@dataclass
class ModelMetadata:
    """Métadonnées associées à un modèle."""
    model_type: str
    framework: str
    input_shape: Tuple
    output_shape: Tuple
    feature_names: List[str]
    target_names: Optional[List[str]] = None
    model_params: Dict[str, Any] = field(default_factory=dict)
    training_date: Optional[str] = None
    training_dataset_hash: Optional[str] = None
    model_version: str = "1.0.0"
    author: Optional[str] = None
    
    def to_dict(self):
        """Convertit en dictionnaire."""
        return asdict(self)


@dataclass
class ExplanationResult:
    """
    Classe qui encapsule les résultats d'une explication.
    
    Cette classe standardisée permet de manipuler uniformément
    les résultats de différentes méthodes d'explicabilité.
    """
    method: ExplainabilityMethod
    model_metadata: ModelMetadata
    feature_importances: List[FeatureImportance]
    explanation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    creation_timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    raw_explanation: Any = None
    audience_level: AudienceLevel = AudienceLevel.TECHNICAL
    explanation_quality: float = 1.0
    explanation_fidelity: Optional[float] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        """Convertit en dictionnaire."""
        result = asdict(self)
        # Convertir les sous-objets complexes
        result["feature_importances"] = [fi.to_dict() for fi in self.feature_importances]
        result["model_metadata"] = self.model_metadata.to_dict()
        return result
    
    def to_json(self):
        """Convertit en JSON."""
        return json.dumps(self.to_dict())
    
    def save(self, path: Union[str, Path]):
        """
        Sauvegarde l'explication dans un fichier.
        
        Args:
            path: Chemin où sauvegarder l'explication
        """
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "ExplanationResult":
        """
        Charge une explication depuis un fichier.
        
        Args:
            path: Chemin du fichier d'explication
            
        Returns:
            ExplanationResult: Instance chargée
        """
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Reconstruction des objets complexes
        feature_importances = [FeatureImportance(**fi) for fi in data.pop("feature_importances")]
        model_metadata = ModelMetadata(**data.pop("model_metadata"))
        
        return cls(
            feature_importances=feature_importances,
            model_metadata=model_metadata,
            **data
        )


class ExplainerBase(abc.ABC, ConfigurableMixin, AuditableMixin):
    """
    Classe de base abstraite pour tous les explainers.
    
    Cette classe définit l'interface commune que tous les explainers
    doivent implémenter, assurant ainsi une API uniforme.
    """
    
    def __init__(self, model, **kwargs):
        """
        Initialise un explainer.
        
        Args:
            model: Modèle à expliquer
            **kwargs: Paramètres additionnels spécifiques à l'explainer
        """
        ConfigurableMixin.__init__(self, **kwargs)
        AuditableMixin.__init__(self)
        self._model = model
        self._method = None
        self._supported_model_types = []
        self._metadata = None
        
        self.add_audit_record("initialization", {
            "explainer_type": self.__class__.__name__,
            "model_type": str(type(model)),
            "config": self._config
        })
    
    @property
    def model(self):
        """Récupère le modèle associé."""
        return self._model
    
    @property
    def method(self) -> ExplainabilityMethod:
        """Récupère la méthode d'explicabilité."""
        return self._method
    
    @abc.abstractmethod
    def explain(self, X, y=None, **kwargs) -> ExplanationResult:
        """
        Génère des explications pour les entrées données.
        
        Args:
            X: Données d'entrée à expliquer
            y: Sorties/labels réels (optionnel)
            **kwargs: Paramètres additionnels
            
        Returns:
            ExplanationResult: Résultat standardisé de l'explication
        """
        pass
    
    @abc.abstractmethod
    def explain_instance(self, instance, **kwargs) -> ExplanationResult:
        """
        Explique une instance spécifique.
        
        Args:
            instance: Instance à expliquer
            **kwargs: Paramètres additionnels
            
        Returns:
            ExplanationResult: Résultat standardisé de l'explication
        """
        pass
    
    def supports_model(self, model) -> bool:
        """
        Vérifie si l'explainer supporte le type de modèle donné.
        
        Args:
            model: Modèle à vérifier
            
        Returns:
            bool: True si le modèle est supporté, False sinon
        """
        return type(model).__name__ in self._supported_model_types
    
    def get_metadata(self) -> ModelMetadata:
        """
        Récupère les métadonnées du modèle.
        
        Returns:
            ModelMetadata: Métadonnées du modèle
        """
        if self._metadata is None:
            self._extract_metadata()
        return self._metadata
    
    def _extract_metadata(self) -> None:
        """
        Extrait les métadonnées du modèle.
        Cette méthode doit être implémentée par les classes dérivées.
        """
        pass
