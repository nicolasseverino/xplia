"""
XPLIA: La librairie d'explicabilité d'IA la plus avancée et complète
===========================================================================

XPLIA fournit un framework unifié pour l'explicabilité des modèles d'IA à travers
différentes méthodes, visualisations interactives et support réglementaire.

Modules principaux
-----------------
core        -- Fonctionnalités fondamentales et API unifiée
explainers  -- Implémentations des algorithmes d'explicabilité
visualizers -- Outils de visualisation interactifs et statiques
models      -- Support pour différentes architectures de modèles
compliance  -- Outils pour la conformité réglementaire
data_processing -- Traitement et analyse des données
utils       -- Fonctions utilitaires et helpers
api         -- Interface de programmation externe

Voir https://xplia.readthedocs.io pour une documentation complète.
"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("xplia")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0.dev"

# Exposition des API principales
from .core import ExplainerBase, load_model, create_explainer, ExplanationResult
from .core.registry import register_explainer, register_visualizer
from .core.config import set_config, get_config, ConfigManager

# Exposition des sous-modules pour un accès facile
from . import explainers
try:
    from . import visualizers
except ImportError:
    visualizers = None
try:
    from . import models
except ImportError:
    models = None
from . import compliance
try:
    from . import data_processing
except ImportError:
    data_processing = None
from . import utils
try:
    from . import api
except ImportError:
    api = None

__all__ = [
    "ExplainerBase",
    "load_model",
    "create_explainer",
    "ExplanationResult",
    "register_explainer",
    "register_visualizer",
    "set_config",
    "get_config",
    "ConfigManager",
    "explainers",
    "visualizers",
    "models",
    "compliance",
    "data_processing",
    "utils",
    "api",
]
