"""
Adaptateurs de modèles pour XPLIA
===============================

Ce module contient les adaptateurs pour différents frameworks de machine learning,
permettant une intégration transparente avec XPLIA.
"""

from .base import ModelAdapterBase

# Import des adaptateurs spécifiques
try:
    from .sklearn_adapter import SklearnModelAdapter
except ImportError:
    SklearnModelAdapter = None

try:
    from .tensorflow_adapter import TensorFlowModelAdapter
except ImportError:
    TensorFlowModelAdapter = None

try:
    from .pytorch_adapter import PyTorchModelAdapter
except ImportError:
    PyTorchModelAdapter = None

try:
    from .xgboost_adapter import XGBoostModelAdapter
except ImportError:
    XGBoostModelAdapter = None

# Dictionnaire des adaptateurs disponibles
AVAILABLE_ADAPTERS = {
    'sklearn': SklearnModelAdapter,
    'tensorflow': TensorFlowModelAdapter,
    'pytorch': PyTorchModelAdapter,
    'xgboost': XGBoostModelAdapter
}

__all__ = [
    'ModelAdapterBase',
    'SklearnModelAdapter',
    'TensorFlowModelAdapter',
    'PyTorchModelAdapter',
    'XGBoostModelAdapter',
    'AVAILABLE_ADAPTERS'
]
