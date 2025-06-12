"""
Classes de base pour les visualisations XPLIA
============================================

Ce module définit les classes de base pour les visualiseurs de XPLIA,
assurant une interface cohérente pour toutes les visualisations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import logging

from ..core.enums import AudienceLevel
from ..core.models import ExplanationResult


class VisualizerBase(ABC):
    """
    Classe de base abstraite pour tous les visualiseurs XPLIA.
    
    Cette classe définit l'interface commune que tous les visualiseurs doivent implémenter,
    garantissant une expérience cohérente pour l'utilisateur final.
    
    Attributs:
        _logger: Logger pour la traçabilité
        _supported_libraries: Bibliothèques de visualisation supportées
        _default_library: Bibliothèque de visualisation par défaut
        _theme: Thème de visualisation
        _color_palette: Palette de couleurs
        _interactive: Si les visualisations doivent être interactives
    """
    
    def __init__(self, library: str = "plotly", theme: str = "light", 
                 color_palette: Optional[List[str]] = None, interactive: bool = True):
        """
        Initialise le visualiseur.
        
        Args:
            library: Bibliothèque de visualisation à utiliser ('plotly', 'matplotlib', 'd3')
            theme: Thème de visualisation ('light', 'dark', 'corporate', etc.)
            color_palette: Palette de couleurs personnalisée
            interactive: Si les visualisations doivent être interactives
        """
        self._logger = logging.getLogger(__name__)
        self._supported_libraries = ["plotly", "matplotlib", "d3"]
        self._default_library = "plotly"
        
        # Vérifier si la bibliothèque demandée est supportée
        if library not in self._supported_libraries:
            self._logger.warning(f"Bibliothèque {library} non supportée. Utilisation de {self._default_library}.")
            self._library = self._default_library
        else:
            self._library = library
        
        self._theme = theme
        self._color_palette = color_palette or [
            "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", 
            "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"
        ]
        self._interactive = interactive
    
    @abstractmethod
    def visualize(self, explanation_result: ExplanationResult, audience_level: AudienceLevel = AudienceLevel.TECHNICAL, **kwargs) -> Any:
        """
        Génère une visualisation à partir d'un résultat d'explication.
        
        Args:
            explanation_result: Résultat d'explication à visualiser
            audience_level: Niveau d'audience ciblé
            **kwargs: Paramètres additionnels spécifiques à la visualisation
            
        Returns:
            Any: Objet de visualisation (figure, HTML, etc.)
        """
        pass
    
    def _adapt_to_audience(self, figure: Any, audience_level: AudienceLevel) -> Any:
        """
        Adapte une visualisation au niveau d'audience spécifié.
        
        Args:
            figure: Visualisation à adapter
            audience_level: Niveau d'audience ciblé
            
        Returns:
            Any: Visualisation adaptée
        """
        # Implémentation par défaut (à surcharger dans les sous-classes)
        if self._library == "plotly":
            import plotly.graph_objects as go
            
            if isinstance(figure, go.Figure):
                # Ajuster la complexité selon le niveau d'audience
                if audience_level == AudienceLevel.PUBLIC:
                    # Simplifier au maximum
                    figure.update_layout(
                        showlegend=True,
                        legend=dict(orientation="h"),
                        margin=dict(l=40, r=40, t=50, b=40),
                    )
                    # Supprimer les annotations techniques
                    figure.update_layout(annotations=[])
                    
                elif audience_level == AudienceLevel.BUSINESS:
                    # Niveau intermédiaire
                    figure.update_layout(
                        showlegend=True,
                        legend=dict(orientation="h"),
                        margin=dict(l=40, r=40, t=50, b=40),
                    )
                    
                # Pour AudienceLevel.TECHNICAL, on garde tous les détails
        
        return figure
    
    def _set_theme(self, figure: Any) -> Any:
        """
        Applique le thème configuré à une visualisation.
        
        Args:
            figure: Visualisation à thématiser
            
        Returns:
            Any: Visualisation thématisée
        """
        if self._library == "plotly":
            import plotly.graph_objects as go
            
            if isinstance(figure, go.Figure):
                if self._theme == "dark":
                    figure.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#222",
                        plot_bgcolor="#222",
                        font=dict(color="#fff")
                    )
                elif self._theme == "light":
                    figure.update_layout(
                        template="plotly_white",
                        paper_bgcolor="#fff",
                        plot_bgcolor="#f8f9fa",
                        font=dict(color="#333")
                    )
                elif self._theme == "corporate":
                    figure.update_layout(
                        template="plotly_white",
                        paper_bgcolor="#f8f9fa",
                        plot_bgcolor="#f8f9fa",
                        font=dict(family="Arial, sans-serif", color="#333")
                    )
        
        elif self._library == "matplotlib":
            import matplotlib.pyplot as plt
            
            if self._theme == "dark":
                plt.style.use("dark_background")
            elif self._theme == "light":
                plt.style.use("default")
            elif self._theme == "corporate":
                plt.style.use("seaborn-v0_8-whitegrid")
        
        return figure
    
    def _apply_color_palette(self, figure: Any) -> Any:
        """
        Applique la palette de couleurs configurée à une visualisation.
        
        Args:
            figure: Visualisation à colorer
            
        Returns:
            Any: Visualisation colorée
        """
        if self._library == "plotly":
            import plotly.graph_objects as go
            
            if isinstance(figure, go.Figure):
                # Appliquer la palette aux traces
                for i, trace in enumerate(figure.data):
                    color_idx = i % len(self._color_palette)
                    if hasattr(trace, "marker") and trace.marker:
                        trace.marker.color = self._color_palette[color_idx]
                    elif hasattr(trace, "line") and trace.line:
                        trace.line.color = self._color_palette[color_idx]
                    elif hasattr(trace, "fillcolor"):
                        trace.fillcolor = self._color_palette[color_idx]
        
        elif self._library == "matplotlib":
            import matplotlib.pyplot as plt
            
            # Définir la palette de couleurs pour matplotlib
            plt.rcParams["axes.prop_cycle"] = plt.cycler(color=self._color_palette)
        
        return figure
    
    def save(self, figure: Any, filename: str, **kwargs) -> None:
        """
        Sauvegarde une visualisation dans un fichier.
        
        Args:
            figure: Visualisation à sauvegarder
            filename: Nom du fichier de sortie
            **kwargs: Paramètres additionnels pour la sauvegarde
        """
        if self._library == "plotly":
            import plotly.graph_objects as go
            
            if isinstance(figure, go.Figure):
                figure.write_html(filename, include_plotlyjs="cdn" if kwargs.get("cdn", True) else True)
                self._logger.info(f"Visualisation sauvegardée dans {filename}")
        
        elif self._library == "matplotlib":
            import matplotlib.pyplot as plt
            
            plt.savefig(filename, dpi=kwargs.get("dpi", 300), bbox_inches="tight")
            self._logger.info(f"Visualisation sauvegardée dans {filename}")
        
        elif self._library == "d3":
            # Pour D3.js, on sauvegarde le HTML généré
            with open(filename, "w", encoding="utf-8") as f:
                f.write(figure)
            self._logger.info(f"Visualisation sauvegardée dans {filename}")
    
    def _check_dependencies(self) -> bool:
        """
        Vérifie si les dépendances nécessaires sont installées.
        
        Returns:
            bool: True si toutes les dépendances sont disponibles
        """
        if self._library == "plotly":
            try:
                import plotly
                return True
            except ImportError:
                self._logger.error("La bibliothèque plotly n'est pas installée. Utilisez 'pip install plotly'.")
                return False
        
        elif self._library == "matplotlib":
            try:
                import matplotlib
                return True
            except ImportError:
                self._logger.error("La bibliothèque matplotlib n'est pas installée. Utilisez 'pip install matplotlib'.")
                return False
        
        elif self._library == "d3":
            # Pour D3.js, on vérifie si les templates sont disponibles
            try:
                from ..templates import d3_templates
                return True
            except ImportError:
                self._logger.error("Les templates D3.js ne sont pas disponibles.")
                return False
        
        return False
