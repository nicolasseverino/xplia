"""
Visualiseur de base pour LUMIA
============================

Ce module définit la classe de base abstraite pour tous les
visualiseurs spécifiques. Il fournit l'interface commune et les
fonctionnalités partagées par tous les types de visualisations.
"""

import abc
from pathlib import Path
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from ..core.base import AudienceLevel, ConfigurableMixin, ExplanationResult
from ..core.registry import register_visualizer


class BaseVisualizer(ConfigurableMixin, abc.ABC):
    """
    Classe abstraite de base pour tous les visualiseurs.
    
    Cette classe définit l'interface commune et les fonctionnalités
    partagées par tous les visualiseurs dans LUMIA.
    """
    
    def __init__(self, explanation_result: ExplanationResult, 
                 audience_level: AudienceLevel = None,
                 theme: str = None,
                 width: int = None,
                 height: int = None,
                 **kwargs):
        """
        Initialise un visualiseur de base.
        
        Args:
            explanation_result: Résultat d'explication à visualiser
            audience_level: Niveau d'audience ciblé (si différent de celui dans explanation_result)
            theme: Thème de visualisation ('light', 'dark', 'custom')
            width: Largeur de la visualisation en pixels
            height: Hauteur de la visualisation en pixels
            **kwargs: Paramètres additionnels
        """
        super().__init__(**kwargs)
        
        self.explanation_result = explanation_result
        self.audience_level = audience_level or (
            explanation_result.audience_level if hasattr(explanation_result, 'audience_level')
            else AudienceLevel.TECHNICAL
        )
        
        # Configuration du style et de la mise en page
        self.theme = theme or self.get_config('visualization', 'theme', 'light')
        self.width = width or self.get_config('visualization', 'default_width', 900)
        self.height = height or self.get_config('visualization', 'default_height', 600)
        
        # Paramètres de personnalisation supplémentaires
        self.customization = kwargs.get('customization', {})
        
        # Initialisation du logger
        self.logger = logging.getLogger(__name__)
        
        # Initialiser les styles en fonction du thème
        self._init_styles()
    
    def _init_styles(self):
        """Initialise les styles visuels en fonction du thème choisi."""
        # Définitions des thèmes de couleurs
        themes = {
            'light': {
                'background_color': '#FFFFFF',
                'text_color': '#333333',
                'grid_color': '#EEEEEE',
                'colorscale': 'Viridis',
                'positive_color': '#1E88E5',
                'negative_color': '#E53935',
                'neutral_color': '#757575',
                'font_family': 'Arial, sans-serif'
            },
            'dark': {
                'background_color': '#222222',
                'text_color': '#EEEEEE',
                'grid_color': '#444444',
                'colorscale': 'Plasma',
                'positive_color': '#29B6F6',
                'negative_color': '#FF5252',
                'neutral_color': '#AAAAAA',
                'font_family': 'Arial, sans-serif'
            }
        }
        
        # Vérifier si c'est un thème intégré ou un thème personnalisé
        if self.theme in themes:
            self.style = themes[self.theme]
        elif self.theme == 'custom' and 'style' in self.customization:
            # Thème personnalisé défini dans les paramètres de customisation
            self.style = {**themes['light'], **self.customization.get('style', {})}
        else:
            # Fallback sur le thème clair
            self.logger.warning(f"Thème '{self.theme}' non reconnu. Utilisation du thème 'light'.")
            self.style = themes['light']
        
        # Appliquer le thème global de plotly (pour les figures plotly)
        template = 'plotly' if self.theme == 'light' else 'plotly_dark'
        pio.templates.default = template
    
    @abc.abstractmethod
    def render(self, **kwargs):
        """
        Rend la visualisation.
        
        Cette méthode doit être implémentée par toutes les sous-classes.
        
        Args:
            **kwargs: Paramètres additionnels pour la visualisation
            
        Returns:
            Any: La visualisation rendue (Figure, HTML, etc.)
        """
        pass
    
    def to_html(self, filename=None, **kwargs):
        """
        Convertit la visualisation en HTML.
        
        Args:
            filename (str, optional): Chemin du fichier où sauvegarder le HTML
            **kwargs: Paramètres additionnels
            
        Returns:
            str: Le code HTML de la visualisation
        """
        fig = self.render(**kwargs)
        
        if hasattr(fig, 'to_html'):
            html = fig.to_html(include_plotlyjs=True, full_html=True)
        else:
            # Conversion générique si render() ne retourne pas une figure plotly
            html = f"""
            <html>
            <head>
                <title>LUMIA Visualization</title>
                <style>
                    body {{ font-family: {self.style['font_family']}; 
                           background-color: {self.style['background_color']};
                           color: {self.style['text_color']}; }}
                </style>
            </head>
            <body>
                {fig if isinstance(fig, str) else str(fig)}
            </body>
            </html>
            """
        
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html)
        
        return html
    
    def save(self, filename, format=None, **kwargs):
        """
        Sauvegarde la visualisation dans un fichier.
        
        Args:
            filename (str): Chemin du fichier
            format (str, optional): Format de fichier ('png', 'jpg', 'svg', 'html', 'pdf')
            **kwargs: Paramètres additionnels pour la sauvegarde
            
        Returns:
            str: Chemin du fichier sauvegardé
        """
        # Déterminer le format à partir de l'extension si non spécifié
        if format is None:
            format = Path(filename).suffix.lstrip('.').lower()
            if not format:
                format = 'html'
                filename = f"{filename}.html"
        
        # Obtenir la visualisation
        fig = self.render(**kwargs)
        
        # Sauvegarder selon le format
        if format == 'html':
            self.to_html(filename=filename)
        elif hasattr(fig, 'write_image'):
            # Pour les figures plotly
            fig.write_image(filename, **kwargs)
        else:
            # Fallback pour d'autres types de visualisations
            self.to_html(filename=f"{Path(filename).stem}.html")
            self.logger.warning(
                f"Le format '{format}' n'est pas directement supporté pour ce type de visualisation. "
                f"Sauvegardé en HTML: {Path(filename).stem}.html"
            )
        
        return filename
    
    def _adapt_for_audience(self, content):
        """
        Adapte le contenu en fonction du niveau d'audience ciblé.
        
        Args:
            content: Contenu à adapter (figure, texte, etc.)
            
        Returns:
            object: Contenu adapté
        """
        # À implémenter par les sous-classes si nécessaire
        return content
    
    def _create_figure_base(self):
        """
        Crée une figure plotly de base avec les styles appropriés.
        
        Returns:
            go.Figure: Figure plotly de base
        """
        fig = go.Figure()
        
        # Appliquer les styles de base
        fig.update_layout(
            template=pio.templates.default,
            font_family=self.style['font_family'],
            font_color=self.style['text_color'],
            plot_bgcolor=self.style['background_color'],
            paper_bgcolor=self.style['background_color'],
            width=self.width,
            height=self.height,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        
        return fig
    
    def add_explanatory_text(self, fig, text=None):
        """
        Ajoute du texte explicatif à la visualisation selon le niveau d'audience.
        
        Args:
            fig: Figure à compléter
            text (str, optional): Texte explicatif personnalisé
            
        Returns:
            object: Figure avec texte ajouté
        """
        if text is None:
            # Adapter le texte selon le niveau d'audience
            if self.audience_level == AudienceLevel.PUBLIC:
                text = ("Cette visualisation montre l'importance des différents facteurs "
                       "dans la décision du modèle.")
            elif self.audience_level == AudienceLevel.BUSINESS:
                text = ("Cette visualisation montre comment les variables influencent "
                       "le résultat du modèle, avec leur impact relatif.")
            elif self.audience_level == AudienceLevel.TECHNICAL:
                text = ("Visualisation des contributions des features aux prédictions "
                       "selon la méthode d'explication utilisée.")
        
        # Ajouter le texte si la figure est de type plotly
        if hasattr(fig, 'add_annotation'):
            fig.add_annotation(
                text=text,
                xref="paper", yref="paper",
                x=0, y=1.05,
                showarrow=False,
                font=dict(
                    family=self.style['font_family'],
                    size=12,
                    color=self.style['text_color']
                ),
                align="left"
            )
        
        return fig
    
    def _highlight_important_features(self, fig, feature_importances, top_n=5):
        """
        Met en évidence les caractéristiques les plus importantes dans la visualisation.
        
        Args:
            fig: Figure à modifier
            feature_importances (dict): Dictionnaire des importances des features
            top_n (int): Nombre de caractéristiques à mettre en évidence
            
        Returns:
            object: Figure avec mises en évidence
        """
        # À implémenter par les sous-classes si pertinent
        return fig
