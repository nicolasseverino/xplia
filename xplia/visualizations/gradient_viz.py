"""
GradientVisualizer pour XPLIA
============================

Ce module fournit un visualiseur interactif et modulaire pour les explications
par gradient (vanilla, integrated, smoothgrad). Il supporte plusieurs types de
données (tabulaire, image, texte) et s'adapte au niveau d'audience.
"""

from typing import Any
from .base import VisualizerBase
from .registry import register_visualizer
from ..core.enums import ExplainabilityMethod, AudienceLevel

@register_visualizer(ExplainabilityMethod.GRADIENT)
class GradientVisualizer(VisualizerBase):
    """
    Visualiseur interactif pour les explications par gradient dans XPLIA.
    
    Ce visualiseur supporte les données tabulaires, images et texte, et adapte
    la visualisation au niveau d'audience (technique, business, public).
    """
    def visualize(self, explanation_result, audience_level: AudienceLevel = AudienceLevel.TECHNICAL, **kwargs) -> Any:
        """
        Génère une visualisation à partir d'un résultat d'explication par gradient.
        
        Args:
            explanation_result: Résultat d'explication (ExplanationResult)
            audience_level: Niveau d'audience ciblé
            **kwargs: Paramètres additionnels (ex: show_values, top_k, etc.)
        Returns:
            Objet de visualisation (Plotly Figure, Matplotlib Figure, HTML, ...)
        """
        input_type = explanation_result.metadata.input_type if hasattr(explanation_result, 'metadata') and hasattr(explanation_result.metadata, 'input_type') else kwargs.get('input_type', 'tabular')
        feature_importances = explanation_result.feature_importances
        
        if input_type == 'tabular':
            return self._visualize_tabular(feature_importances, audience_level, **kwargs)
        elif input_type == 'image':
            return self._visualize_image(explanation_result, audience_level, **kwargs)
        elif input_type == 'text':
            return self._visualize_text(feature_importances, audience_level, **kwargs)
        else:
            raise ValueError(f"Type d'entrée non supporté pour la visualisation gradient: {input_type}")
    
    def _visualize_tabular(self, feature_importances, audience_level, **kwargs):
        import plotly.graph_objects as go
        top_k = kwargs.get('top_k', 15)
        show_values = kwargs.get('show_values', True)
        features = [fi.feature_name for fi in feature_importances[:top_k]]
        scores = [fi.importance_score for fi in feature_importances[:top_k]]
        
        fig = go.Figure(go.Bar(
            x=scores,
            y=features,
            orientation='h',
            marker_color=self._color_palette[0],
            text=[f"{s:.2f}" for s in scores] if show_values else None,
            textposition='auto'
        ))
        fig.update_layout(
            title="Importances par gradient",
            xaxis_title="Score d'importance",
            yaxis_title="Caractéristique",
            margin=dict(l=80, r=40, t=60, b=40)
        )
        fig = self._set_theme(fig)
        fig = self._adapt_to_audience(fig, audience_level)
        return fig
    
    def _visualize_image(self, explanation_result, audience_level, **kwargs):
        import plotly.graph_objects as go
        import numpy as np
        # On suppose que explanation_result contient 'original_input' et 'attributions' (heatmap)
        original = getattr(explanation_result, 'original_input', None)
        attributions = getattr(explanation_result, 'attributions', None)
        if attributions is None and hasattr(explanation_result, 'feature_importances'):
            # fallback: reconstituer la heatmap à partir des importances
            attributions = np.array([fi.importance_score for fi in explanation_result.feature_importances])
            if original is not None and attributions.size == np.prod(original.shape):
                attributions = attributions.reshape(original.shape)
        if attributions is None:
            raise ValueError("Impossible de visualiser l'attribution gradient pour image: aucune heatmap trouvée.")
        if original is not None:
            fig = go.Figure()
            fig.add_trace(go.Image(z=original))
            fig.add_trace(go.Heatmap(z=attributions, opacity=0.6, colorscale='RdBu', showscale=True, colorbar=dict(title='Attribution')))
            fig.update_layout(title="Attributions par gradient (superposées)", margin=dict(l=40, r=40, t=60, b=40))
        else:
            fig = go.Figure(go.Heatmap(z=attributions, colorscale='RdBu', showscale=True, colorbar=dict(title='Attribution')))
            fig.update_layout(title="Attributions par gradient", margin=dict(l=40, r=40, t=60, b=40))
        fig = self._set_theme(fig)
        fig = self._adapt_to_audience(fig, audience_level)
        return fig
    
    def _visualize_text(self, feature_importances, audience_level, **kwargs):
        import plotly.graph_objects as go
        top_k = kwargs.get('top_k', 20)
        tokens = [fi.feature_name for fi in feature_importances[:top_k]]
        scores = [fi.importance_score for fi in feature_importances[:top_k]]
        colors = [self._color_palette[0] if s >= 0 else self._color_palette[1] for s in scores]
        fig = go.Figure(go.Bar(
            x=tokens,
            y=scores,
            marker_color=colors,
            text=[f"{s:.2f}" for s in scores],
            textposition='auto'
        ))
        fig.update_layout(
            title="Attribution par gradient sur les tokens",
            xaxis_title="Token",
            yaxis_title="Score d'importance",
            margin=dict(l=60, r=40, t=60, b=40)
        )
        fig = self._set_theme(fig)
        fig = self._adapt_to_audience(fig, audience_level)
        return fig
