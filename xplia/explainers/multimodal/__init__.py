"""
XPLIA Multimodal Explainers
============================

Ce module contient les explainers spécialisés pour l'analyse multimodale,
capable d'expliquer simultanément les décisions basées sur différents types de données:
- Texte
- Images
- Audio
- Séries temporelles
- Données tabulaires

Ces explainers avancés permettent de comprendre les modèles combinant plusieurs
modalités d'entrée, comme les modèles de fondation (GPT-5, Claude, Gemini, etc.)
ou les architectures hybrides spécialisées.
"""

from .base import MultimodalExplainerBase
from .registry import register_multimodal_explainer
from .text_image_explainer import TextImageExplainer
from .foundation_model_explainer import FoundationModelExplainer

__all__ = [
    'MultimodalExplainerBase',
    'register_multimodal_explainer',
    'TextImageExplainer',
    'FoundationModelExplainer',
]
