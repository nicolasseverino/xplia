"""
Module de visualisation pour LUMIA
=================================

Ce module fournit des visualisations interactives et personnalisables
pour les résultats d'explication générés par les explainers.
"""

from ..core.registry import Registry

# Import des visualiseurs spécifiques
from .base_visualizer import BaseVisualizer
from .interactive_dashboard import DashboardVisualizer
from .feature_importance import FeatureImportanceVisualizer
from .shap_visualizer import ShapVisualizer
from .counterfactual_visualizer import CounterfactualVisualizer

# Factory de visualiseurs
class VisualizerFactory:
    """Factory pour créer les visualiseurs appropriés selon le type d'explication."""
    
    def __init__(self):
        self.registry = Registry()
    
    def create_visualizer(self, explanation_result, viz_type='auto', **kwargs):
        """
        Crée un visualiseur adapté pour le résultat d'explication donné.
        
        Args:
            explanation_result: Résultat d'explication à visualiser
            viz_type (str): Type de visualisation demandée ou 'auto' pour sélection automatique
            **kwargs: Paramètres additionnels pour le visualiseur
            
        Returns:
            BaseVisualizer: Instance de visualiseur configurée
        """
        if viz_type == 'auto':
            # Sélectionne le meilleur visualiseur en fonction du type d'explication
            viz_type = self._select_best_visualizer(explanation_result)
            
        # Récupérer la classe du visualiseur à partir du registre
        visualizer_class = self.registry.get_visualizer(viz_type)
        if not visualizer_class:
            raise ValueError(f"Aucun visualiseur de type '{viz_type}' n'est enregistré")
            
        # Instancier et configurer le visualiseur
        return visualizer_class(explanation_result, **kwargs)
    
    def _select_best_visualizer(self, explanation_result):
        """
        Sélectionne automatiquement le meilleur visualiseur pour le résultat donné.
        
        Args:
            explanation_result: Résultat d'explication à visualiser
            
        Returns:
            str: Type de visualiseur recommandé
        """
        method = explanation_result.method if hasattr(explanation_result, 'method') else None
        
        if method:
            method_name = method.name if hasattr(method, 'name') else str(method)
            
            # Associer les méthodes d'explication aux visualiseurs appropriés
            method_viz_map = {
                'SHAP': 'shap',
                'LIME': 'feature_importance',
                'COUNTERFACTUAL': 'counterfactual',
                'UNIFIED': 'dashboard'
            }
            
            for method_key, viz_type in method_viz_map.items():
                if method_key in method_name.upper():
                    return viz_type
        
        # Par défaut, utiliser le tableau de bord
        return 'dashboard'

# Exposer les classes et fonctions principales
__all__ = [
    'BaseVisualizer',
    'DashboardVisualizer',
    'FeatureImportanceVisualizer',
    'ShapVisualizer',
    'CounterfactualVisualizer',
    'VisualizerFactory',
]
