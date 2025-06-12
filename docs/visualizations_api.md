# Documentation API du Module de Visualisations XPLIA

Ce document présente l'utilisation du module de visualisations pour générer des graphiques et des tableaux interactifs pour les rapports de conformité XPLIA.

## Table des matières
1. [Vue d'ensemble](#vue-densemble)
2. [Installation des dépendances](#installation-des-dépendances)
3. [Utilisation de base](#utilisation-de-base)
4. [Types de graphiques disponibles](#types-de-graphiques-disponibles)
5. [Configuration des graphiques](#configuration-des-graphiques)
6. [Intégration avec les générateurs de rapports](#intégration-avec-les-générateurs-de-rapports)
7. [Exemples](#exemples)
8. [FAQ](#faq)

## Vue d'ensemble

Le module de visualisations fournit une interface unifiée pour générer des représentations visuelles des données, en utilisant diverses bibliothèques comme Plotly, Matplotlib et Bokeh. Il est conçu pour être:

- **Modulaire**: chaque type de graphique est implémenté de manière isolée
- **Configurable**: nombreuses options pour adapter l'apparence et le comportement
- **Multi-bibliothèque**: support pour plusieurs bibliothèques de visualisation
- **Multi-format**: export en HTML, PNG, JPG, SVG, PDF et base64
- **Découvrable**: intégré au système de registre XPLIA pour une découverte dynamique

## Installation des dépendances

Le module supporte plusieurs bibliothèques de visualisation qui doivent être installées selon vos besoins:

```bash
# Pour Plotly
pip install plotly>=5.0.0 kaleido

# Pour Matplotlib
pip install matplotlib>=3.5.0

# Pour Bokeh
pip install bokeh>=3.0.0

# Pour les types de graphiques spécifiques
pip install squarify  # Pour les treemaps avec Matplotlib
```

## Utilisation de base

Voici comment utiliser le générateur de graphiques dans son implémentation la plus simple:

```python
from xplia.visualizations import ChartGenerator, ChartType, ChartLibrary

# Création d'un générateur de graphiques
generator = ChartGenerator(library=ChartLibrary.PLOTLY, theme="light")

# Données et configuration
data = {"x": ["A", "B", "C"], "y": [10, 20, 30]}
config = {"title": "Mon graphique", "x_title": "Catégories", "y_title": "Valeurs"}

# Génération d'un graphique à barres
chart = generator.create_chart(ChartType.BAR, data, config)

# Export du graphique
generator.save(chart, "mon_graphique.html")
generator.save(chart, "mon_graphique.png")
html_content = generator.to_html(chart)
base64_image = generator.to_base64(chart)
```

## Types de graphiques disponibles

Le module supporte les types de graphiques suivants:

| Type | Description | Bibliothèques supportées |
|------|-------------|--------------------------|
| `ChartType.BAR` | Graphique à barres | Plotly, Matplotlib, Bokeh |
| `ChartType.LINE` | Graphique en ligne | Plotly, Matplotlib, Bokeh |
| `ChartType.PIE` | Graphique circulaire | Plotly, Matplotlib, Bokeh |
| `ChartType.SCATTER` | Nuage de points | Plotly, Matplotlib, Bokeh |
| `ChartType.HEATMAP` | Carte de chaleur | Plotly, Matplotlib, Bokeh |
| `ChartType.RADAR` | Graphique radar | Plotly, Matplotlib |
| `ChartType.BOXPLOT` | Boîte à moustaches | Plotly, Matplotlib, Bokeh |
| `ChartType.HISTOGRAM` | Histogramme | Plotly, Matplotlib, Bokeh |
| `ChartType.TREEMAP` | Carte proportionnelle hiérarchique | Plotly, Matplotlib |
| `ChartType.SANKEY` | Diagramme de flux | Plotly |
| `ChartType.GAUGE` | Jauge | Plotly |
| `ChartType.TABLE` | Tableau de données | Plotly, Matplotlib |

## Configuration des graphiques

Chaque type de graphique accepte une configuration spécifique. Voici les options communes:

```python
config = {
    # Options générales
    "title": "Titre du graphique",
    "subtitle": "Sous-titre optionnel",
    "width": 800,
    "height": 500,
    "margin": {"l": 40, "r": 40, "t": 60, "b": 40},
    "legend": True,
    "legend_position": "top-right",
    
    # Axes
    "x_title": "Titre de l'axe X",
    "y_title": "Titre de l'axe Y",
    "x_grid": True,
    "y_grid": True,
    "x_type": "category",  # 'linear', 'log', 'date', etc.
    "y_type": "linear",
    
    # Style
    "colors": ["#3366CC", "#DC3912", "#FF9900", ...],
    "template": "custom",  # 'plotly_white', 'plotly_dark', etc. pour Plotly
    "font_family": "Arial, sans-serif",
    "font_size": 14,
    
    # Interactivité
    "tooltip": True,
    "zoom": True,
    "pan": True,
    "animation": {"duration": 500, "easing": "cubic-in-out"},
    
    # Export
    "image_format": "png",  # 'png', 'jpeg', 'svg', 'pdf', etc.
    "image_scale": 2,       # Échelle pour l'export (résolution)
    "image_width": 1200,    # Largeur spécifique pour l'export
    "image_height": 800     # Hauteur spécifique pour l'export
}
```

## Intégration avec les générateurs de rapports

Le module de visualisations s'intègre facilement avec les générateurs de rapports XPLIA, notamment les formats HTML et PDF:

### Intégration HTML

```python
from xplia.visualizations import ChartGenerator, ChartType
from xplia.compliance.formatters.html_formatter import HTMLReportGenerator
from xplia.compliance.report_base import ReportContent, ReportConfig

# Créer un graphique
chart_generator = ChartGenerator()
data = {"x": ["A", "B", "C"], "y": [10, 20, 30]}
chart = chart_generator.create_chart(ChartType.BAR, data)

# Convertir le graphique en HTML
chart_html = chart_generator.to_html(chart)

# Intégrer dans un rapport HTML
report_config = ReportConfig()
report_generator = HTMLReportGenerator(report_config)
report_content = ReportContent()

# Ajouter le graphique au contenu du rapport
report_content.visualizations = [{"title": "Mon graphique", "content": chart_html}]

# Générer le rapport
html_content = report_generator.generate(report_content)
```

### Intégration PDF

```python
from xplia.visualizations import ChartGenerator, ChartType
from xplia.compliance.formatters.pdf_formatter import PDFReportGenerator
from xplia.compliance.report_base import ReportContent, ReportConfig

# Créer un graphique
chart_generator = ChartGenerator()
data = {"x": ["A", "B", "C"], "y": [10, 20, 30]}
chart = chart_generator.create_chart(ChartType.BAR, data)

# Convertir le graphique en image base64
chart_img = chart_generator.to_base64(chart, format="png")

# Intégrer dans un rapport PDF
report_config = ReportConfig()
report_generator = PDFReportGenerator(report_config)
report_content = ReportContent()

# Ajouter le graphique au contenu du rapport
report_content.visualizations = [{"title": "Mon graphique", "content": chart_img, "type": "image/base64"}]

# Générer le rapport
pdf_content = report_generator.generate(report_content, "rapport.pdf")
```

## Exemples

### Graphique à barres

```python
from xplia.visualizations import ChartGenerator, ChartType

generator = ChartGenerator()

data = {
    "x": ["Produit A", "Produit B", "Produit C", "Produit D"],
    "y": [15, 25, 12, 8],
    "color": ["#3366CC", "#DC3912", "#FF9900", "#109618"]
}

config = {
    "title": "Ventes par produit",
    "x_title": "Produits",
    "y_title": "Ventes (milliers €)",
    "template": "plotly_white",
    "orientation": "v"  # 'v' pour vertical, 'h' pour horizontal
}

chart = generator.create_chart(ChartType.BAR, data, config)
generator.save(chart, "ventes_par_produit.html")
```

### Graphique en ligne avec séries multiples

```python
from xplia.visualizations import ChartGenerator, ChartType

generator = ChartGenerator()

data = {
    "x": ["Jan", "Feb", "Mar", "Avr", "Mai", "Jui"],
    "series": [
        {"name": "2023", "values": [10, 15, 12, 18, 22, 25]},
        {"name": "2024", "values": [12, 18, 20, 24, 25, 30]}
    ]
}

config = {
    "title": "Évolution des ventes",
    "x_title": "Mois",
    "y_title": "Ventes (milliers €)",
    "legend": True,
    "markers": True,
    "line_dash": {"2023": "solid", "2024": "dash"}
}

chart = generator.create_chart(ChartType.LINE, data, config)
generator.save(chart, "evolution_ventes.html")
```

### Carte de chaleur des risques

```python
from xplia.visualizations import ChartGenerator, ChartType

generator = ChartGenerator()

# Matrice de risques 5x5
data = {
    "x": ["Très rare", "Rare", "Possible", "Probable", "Très probable"],
    "y": ["Négligeable", "Mineur", "Modéré", "Majeur", "Critique"],
    "z": [
        [1, 2, 3, 4, 5],
        [2, 4, 6, 8, 10],
        [3, 6, 9, 12, 15],
        [4, 8, 12, 16, 20],
        [5, 10, 15, 20, 25]
    ]
}

config = {
    "title": "Matrice des risques",
    "x_title": "Probabilité",
    "y_title": "Impact",
    "colorscale": [
        [0, "green"],
        [0.3, "yellow"],
        [0.7, "orange"],
        [1, "red"]
    ],
    "showscale": True,
    "annotations": True  # Afficher les valeurs dans les cellules
}

chart = generator.create_chart(ChartType.HEATMAP, data, config)
generator.save(chart, "matrice_risques.html")
```

## FAQ

**Q: Puis-je utiliser plusieurs bibliothèques dans la même application?**  
R: Oui, vous pouvez créer plusieurs instances de `ChartGenerator` avec différentes bibliothèques selon vos besoins.

**Q: Comment choisir la bibliothèque la plus adaptée?**  
R: 
- **Plotly**: Pour des visualisations interactives riches pour le web
- **Matplotlib**: Pour des graphiques statiques, publications scientifiques
- **Bokeh**: Pour des applications web avec interactivité

**Q: Les graphiques sont-ils accessibles?**  
R: Oui, les graphiques interactifs incluent des fonctionnalités d'accessibilité comme des descriptions alternatives et la navigation au clavier.

**Q: Comment personnaliser les couleurs pour la conformité à ma charte graphique?**  
R: Utilisez l'option `colors` dans la configuration ou définissez une palette personnalisée lors de la création du générateur.

**Q: Les graphiques fonctionnent-ils hors ligne?**  
R: Oui, toutes les dépendances sont intégrées dans l'export HTML, permettant une utilisation hors ligne.

**Q: Comment intégrer les graphiques dans une application web externe?**  
R: Utilisez la méthode `to_html()` pour obtenir un fragment HTML à intégrer, ou `to_base64()` pour une image à inclure dans n'importe quel contexte.
