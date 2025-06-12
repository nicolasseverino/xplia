"""
Module de visualisations pour les rapports de conformité XPLIA
=============================================================

Ce module fournit des fonctionnalités avancées pour générer des visualisations
interactives et statiques utilisables dans tous les formats de rapports XPLIA.
Il supporte plusieurs bibliothèques de visualisation (Plotly, Matplotlib, D3.js),
et peut s'adapter au contexte d'utilisation (web, documents statiques, etc.).
"""

import logging
import os
import json
import base64
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from pathlib import Path
from io import BytesIO
import datetime

# Import des modules de visualisation
from .visualizations.charts_impl import create_bar_chart
from .visualizations.line_chart import create_line_chart
from .visualizations.pie_chart import create_pie_chart
from .visualizations.scatter_chart import create_scatter_chart
from .visualizations.heatmap_chart import create_heatmap_chart
from .visualizations.radar_chart import create_radar_chart
from .visualizations.boxplot_chart import create_boxplot_chart
from .visualizations.histogram_chart import create_histogram_chart
from .visualizations.treemap_chart import create_treemap_chart
from .visualizations.sankey_chart import create_sankey_chart
from .visualizations.gauge_chart import create_gauge_chart
from .visualizations.table_chart import create_table_chart

# Import du système de registre
from .core.registry import register_visualizer, ComponentType

# Logger dédié au module de visualisations
logger = logging.getLogger(__name__)

# Constantes pour les couleurs et styles par défaut
DEFAULT_COLOR_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

# Types de visualisations disponibles
class ChartType(Enum):
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    RADAR = "radar"
    SANKEY = "sankey"
    TREEMAP = "treemap"
    TABLE = "table"
    GAUGE = "gauge"
    BOXPLOT = "boxplot"
    HISTOGRAM = "histogram"

# Bibliothèques de visualisation supportées
class ChartLibrary(Enum):
    PLOTLY = "plotly"
    MATPLOTLIB = "matplotlib"
    D3 = "d3"
    BOKEH = "bokeh"

# Contextes d'utilisation des visualisations
class OutputContext(Enum):
    WEB = "web"           # Pour affichage dans navigateurs (HTML, JavaScript)
    DOCUMENT = "document" # Pour inclusion dans documents (PDF, Word)
    API = "api"           # Pour retour par API (JSON)
    DASHBOARD = "dashboard" # Pour dashboards interactifs

@register_visualizer(
    version="1.1.0",
    description="Générateur de graphiques avancé et modulaire pour XPLIA",
    author="Équipe XPLIA",
    tags=["visualization", "charts", "interactive", "static"],
    capabilities={
        "interactive": True,
        "static": True,
        "export_formats": ["html", "png", "jpg", "svg", "pdf", "base64"]
    },
    dependencies={
        "plotly": "5.0.0",
        "matplotlib": "3.5.0",
        "bokeh": "3.0.0"
    },
    examples=[
        "examples/visualizations/basic_charts.py",
        "examples/visualizations/advanced_charts.py"
    ],
    supported_formats=["html", "png", "jpg", "svg", "pdf"],
    priority=10,
    configuration_schema={
        "type": "object",
        "properties": {
            "chart_type": {"type": "string", "enum": ["bar", "line", "pie", "scatter", "heatmap", 
                                           "radar", "boxplot", "histogram", "treemap", "sankey", 
                                           "gauge", "table"]},
            "library": {"type": "string", "enum": ["plotly", "matplotlib", "bokeh"]},
            "theme": {"type": "string", "enum": ["light", "dark", "corporate"]},
            "responsive": {"type": "boolean"},
            "interactive": {"type": "boolean"}
        }
    }
)
class ChartGenerator:
    """
    Générateur de visualisations pour les rapports de conformité XPLIA.
    
    Cette classe fournit une interface unifiée pour générer des visualisations
    avec différentes bibliothèques, adaptées à différents contextes d'utilisation.
    Elle permet la génération de graphiques statiques et interactifs, avec une
    personnalisation avancée des styles et des formats.
    """
    
    def __init__(self, 
                 library: Union[str, ChartLibrary] = ChartLibrary.PLOTLY,
                 theme: str = "light",
                 color_palette: Optional[List[str]] = None,
                 output_context: Union[str, OutputContext] = OutputContext.WEB,
                 interactive: bool = True,
                 responsive: bool = True,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialise le générateur de visualisations.
        
        Args:
            library: Bibliothèque de visualisation à utiliser
            theme: Thème de visualisation ('light', 'dark', 'corporate', etc.)
            color_palette: Palette de couleurs personnalisée
            output_context: Contexte d'utilisation de la visualisation
            interactive: Si les visualisations doivent être interactives
            responsive: Si les visualisations doivent s'adapter à la taille du conteneur
            config: Configuration avancée pour la bibliothèque de visualisation
        """
        # Normalisation des paramètres
        self._library = library if isinstance(library, ChartLibrary) else ChartLibrary(library)
        self._output_context = output_context if isinstance(output_context, OutputContext) else OutputContext(output_context)
        
        # Configuration de base
        self._theme = theme
        self._color_palette = color_palette or DEFAULT_COLOR_PALETTE
        self._interactive = interactive
        self._responsive = responsive
        self._config = config or {}
        
        # Dimensions par défaut
        self._default_width = 800
        self._default_height = 500
        
        # Registre des générateurs de graphiques spécifiques
        self._chart_generators = {
            ChartType.BAR: self._create_bar_chart,
            ChartType.LINE: self._create_line_chart,
            ChartType.PIE: self._create_pie_chart,
            ChartType.SCATTER: self._create_scatter_chart,
            ChartType.HEATMAP: self._create_heatmap,
            ChartType.RADAR: self._create_radar_chart,
            ChartType.GAUGE: self._create_gauge_chart,
            ChartType.TABLE: self._create_table_chart,
            ChartType.BOXPLOT: self._create_boxplot_chart,
            ChartType.HISTOGRAM: self._create_histogram_chart,
            ChartType.TREEMAP: self._create_treemap_chart,
            ChartType.SANKEY: self._create_sankey_chart,
        }
        
        # Vérification des dépendances
        self._check_dependencies()
    
    def create_chart(self, 
                     chart_type: Union[str, ChartType],
                     data: Any,
                     title: str = "",
                     x_label: str = "",
                     y_label: str = "",
                     **kwargs) -> Any:
        """
        Crée une visualisation du type spécifié.
        
        Args:
            chart_type: Type de visualisation à créer
            data: Données pour la visualisation
            title: Titre de la visualisation
            x_label: Label de l'axe X
            y_label: Label de l'axe Y
            **kwargs: Paramètres spécifiques au type de visualisation
            
        Returns:
            Objet de visualisation (Figure, HTML, etc. selon la bibliothèque)
        """
        # Normalisation du type de graphique
        if isinstance(chart_type, str):
            chart_type = ChartType(chart_type)
        
        # Vérification de la disponibilité du générateur
        if chart_type not in self._chart_generators:
            raise ValueError(f"Type de graphique non supporté: {chart_type}")
        
        # Configuration commune
        chart_config = {
            'title': title,
            'x_label': x_label,
            'y_label': y_label,
            'width': kwargs.get('width', self._default_width),
            'height': kwargs.get('height', self._default_height),
            'theme': kwargs.get('theme', self._theme),
            'color_palette': kwargs.get('color_palette', self._color_palette),
            'interactive': kwargs.get('interactive', self._interactive),
            'responsive': kwargs.get('responsive', self._responsive),
        }
        
        # Création du graphique
        chart = self._chart_generators[chart_type](data, chart_config, **kwargs)
        
        # Application du thème et des styles
        chart = self._apply_theme(chart, chart_config['theme'])
        
        return chart
    
    def to_html(self, chart: Any, include_plotlyjs: str = "cdn", full_html: bool = False) -> str:
        """
        Convertit un graphique en HTML.
        
        Args:
            chart: Graphique à convertir
            include_plotlyjs: Comment inclure Plotly.js ('cdn', True, False)
            full_html: Si True, génère une page HTML complète
            
        Returns:
            Code HTML du graphique
        """
        if self._library == ChartLibrary.PLOTLY:
            import plotly.io as pio
            return pio.to_html(
                chart,
                include_plotlyjs=include_plotlyjs,
                full_html=full_html
            )
        elif self._library == ChartLibrary.MATPLOTLIB:
            import matplotlib.pyplot as plt
            from mpld3 import fig_to_html
            return fig_to_html(chart, figid=f"chart_{hash(chart)}")
        elif self._library == ChartLibrary.BOKEH:
            from bokeh.embed import components
            script, div = components(chart)
            if full_html:
                return f"""<!DOCTYPE html>
                <html>
                <head>
                    <meta charset="utf-8">
                    <title>Visualisation XPLIA</title>
                    <script src="https://cdn.bokeh.org/bokeh/release/bokeh-2.4.2.min.js"></script>
                    <script src="https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.4.2.min.js"></script>
                    {script}
                </head>
                <body>
                    {div}
                </body>
                </html>"""
            else:
                return f"{script}\n{div}"
        else:
            # Si c'est déjà du HTML (D3.js par exemple)
            if isinstance(chart, str) and ("<svg" in chart or "<div" in chart):
                return chart
            
            logger.warning(f"Conversion en HTML non supportée pour {self._library}")
            return ""
    
    def to_image(self, chart: Any, format: str = "png", scale: float = 1.0) -> bytes:
        """
        Convertit un graphique en image.
        
        Args:
            chart: Graphique à convertir
            format: Format d'image ('png', 'jpeg', 'svg', 'pdf')
            scale: Échelle de l'image
            
        Returns:
            Données binaires de l'image
        """
        if self._library == ChartLibrary.PLOTLY:
            import plotly.io as pio
            return pio.to_image(chart, format=format, scale=scale)
        elif self._library == ChartLibrary.MATPLOTLIB:
            import matplotlib.pyplot as plt
            buffer = BytesIO()
            chart.savefig(buffer, format=format, dpi=100*scale, bbox_inches='tight')
            buffer.seek(0)
            return buffer.read()
        elif self._library == ChartLibrary.BOKEH:
            from bokeh.io import export_png, export_svg
            buffer = BytesIO()
            if format == "svg":
                export_svg(chart, filename=buffer)
            else:
                export_png(chart, filename=buffer)
            buffer.seek(0)
            return buffer.read()
        else:
            logger.warning(f"Conversion en image non supportée pour {self._library}")
            return bytes()
    
    def to_base64(self, chart: Any, format: str = "png", scale: float = 1.0) -> str:
        """
        Convertit un graphique en chaîne base64.
        
        Args:
            chart: Graphique à convertir
            format: Format d'image ('png', 'jpeg', 'svg', 'pdf')
            scale: Échelle de l'image
            
        Returns:
            Chaîne base64 de l'image
        """
        img_data = self.to_image(chart, format, scale)
        return base64.b64encode(img_data).decode('utf-8')
    
    def save(self, chart: Any, filename: str, **kwargs) -> None:
        """
        Sauvegarde un graphique dans un fichier.
        
        Args:
            chart: Graphique à sauvegarder
            filename: Nom du fichier de sortie
            **kwargs: Paramètres spécifiques au format de sortie
        """
        # Détection du format à partir de l'extension
        ext = Path(filename).suffix.lower()[1:]
        
        if ext in ['html', 'htm']:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.to_html(chart, full_html=kwargs.get('full_html', True)))
        elif ext in ['png', 'jpg', 'jpeg', 'svg', 'pdf']:
            with open(filename, 'wb') as f:
                f.write(self.to_image(chart, format=ext, scale=kwargs.get('scale', 1.0)))
        elif ext == 'json' and self._library == ChartLibrary.PLOTLY:
            import plotly.io as pio
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(pio.to_json(chart))
        else:
            logger.warning(f"Format de sauvegarde non supporté: {ext}")
            
        logger.info(f"Graphique sauvegardé dans {filename}")
    
    def _check_dependencies(self) -> None:
        """
        Vérifie si les dépendances nécessaires sont installées.
        """
        try:
            if self._library == ChartLibrary.PLOTLY:
                import plotly
                logger.info(f"Utilisation de Plotly v{plotly.__version__}")
            elif self._library == ChartLibrary.MATPLOTLIB:
                import matplotlib
                logger.info(f"Utilisation de Matplotlib v{matplotlib.__version__}")
            elif self._library == ChartLibrary.BOKEH:
                import bokeh
                logger.info(f"Utilisation de Bokeh v{bokeh.__version__}")
            elif self._library == ChartLibrary.D3:
                # D3.js est utilisé via son API JavaScript, pas de dépendance Python
                pass
        except ImportError as e:
            library_name = self._library.value
            logger.warning(f"La bibliothèque {library_name} n'est pas installée: {e}")
            logger.info(f"Installez-la avec: pip install {library_name}")
    
    def _apply_theme(self, chart: Any, theme: str) -> Any:
        """
        Applique un thème au graphique.
        
        Args:
            chart: Graphique à thématiser
            theme: Nom du thème
            
        Returns:
            Graphique thématisé
        """
        if self._library == ChartLibrary.PLOTLY:
            import plotly.graph_objects as go
            if not isinstance(chart, go.Figure):
                return chart
                
            if theme == "dark":
                chart.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#222222",
                    plot_bgcolor="#222222",
                    font=dict(color="#ffffff")
                )
            elif theme == "light":
                chart.update_layout(
                    template="plotly_white",
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#f8f9fa"
                )
            elif theme == "corporate":
                chart.update_layout(
                    template="plotly_white",
                    paper_bgcolor="#f8f9fa",
                    plot_bgcolor="#f8f9fa",
                    font=dict(family="Arial, sans-serif")
                )
            
            # Ajout des marges standard et responsive
            chart.update_layout(
                margin=dict(l=40, r=40, t=60, b=40, pad=0),
                autosize=self._responsive
            )
        
        elif self._library == ChartLibrary.MATPLOTLIB:
            import matplotlib.pyplot as plt
            if theme == "dark":
                plt.style.use("dark_background")
            elif theme == "light":
                plt.style.use("default")
            elif theme == "corporate":
                plt.style.use("seaborn-v0_8-whitegrid")
        
        return chart

    # Implémentation des méthodes de création de graphiques
    # Chaque méthode délègue le travail au module spécifique
    
    def _create_bar_chart(self, data: Any, config: Dict[str, Any], **kwargs) -> Any:
        """Crée un graphique à barres en utilisant le module charts_impl"""
        return create_bar_chart(
            data, 
            config, 
            library=self._library.value, 
            **kwargs
        )
    
    def _create_line_chart(self, data: Any, config: Dict[str, Any], **kwargs) -> Any:
        """Crée un graphique en ligne en utilisant le module line_chart"""
        return create_line_chart(
            data, 
            config, 
            library=self._library.value, 
            **kwargs
        )
    
    def _create_pie_chart(self, data: Any, config: Dict[str, Any], **kwargs) -> Any:
        """Crée un graphique circulaire en utilisant le module pie_chart"""
        return create_pie_chart(
            data, 
            config, 
            library=self._library.value, 
            **kwargs
        )
    
    def _create_scatter_chart(self, data: Any, config: Dict[str, Any], **kwargs) -> Any:
        """Crée un graphique de dispersion en utilisant le module scatter_chart"""
        return create_scatter_chart(
            data, 
            config, 
            library=self._library.value, 
            **kwargs
        )
    
    def _create_heatmap(self, data: Any, config: Dict[str, Any], **kwargs) -> Any:
        """Crée une carte de chaleur en utilisant le module heatmap_chart"""
        return create_heatmap_chart(
            data, 
            config, 
            library=self._library.value, 
            **kwargs
        )
    
    def _create_radar_chart(self, data: Any, config: Dict[str, Any], **kwargs) -> Any:
        """Crée un graphique radar en utilisant le module radar_chart"""
        return create_radar_chart(
            data, 
            config, 
            library=self._library.value, 
            **kwargs
        )
    
    def _create_boxplot_chart(self, data: Any, config: Dict[str, Any], **kwargs) -> Any:
        """Crée une boîte à moustaches en utilisant le module boxplot_chart"""
        return create_boxplot_chart(
            data, 
            config, 
            library=self._library.value, 
            **kwargs
        )
    
    def _create_histogram_chart(self, data: Any, config: Dict[str, Any], **kwargs) -> Any:
        """Crée un histogramme en utilisant le module histogram_chart"""
        return create_histogram_chart(
            data, 
            config, 
            library=self._library.value, 
            **kwargs
        )
    
    def _create_treemap_chart(self, data: Any, config: Dict[str, Any], **kwargs) -> Any:
        """Crée un treemap en utilisant le module treemap_chart"""
        return create_treemap_chart(
            data, 
            config, 
            library=self._library.value, 
            **kwargs
        )
    
    def _create_sankey_chart(self, data: Any, config: Dict[str, Any], **kwargs) -> Any:
        """Crée un diagramme de flux Sankey en utilisant le module sankey_chart"""
        return create_sankey_chart(
            data, 
            config, 
            library=self._library.value, 
            **kwargs
        )
    
    def _create_gauge_chart(self, data: Any, config: Dict[str, Any], **kwargs) -> Any:
        """Crée un graphique de jauge en utilisant le module gauge_chart"""
        return create_gauge_chart(
            data, 
            config, 
            library=self._library.value, 
            **kwargs
        )
    
    def _create_table_chart(self, data: Any, config: Dict[str, Any], **kwargs) -> Any:
        """Crée un tableau de données en utilisant le module table_chart"""
        return create_table_chart(
            data, 
            config, 
            library=self._library.value, 
            **kwargs
        )
