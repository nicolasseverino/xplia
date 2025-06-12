# Guide d'utilisation des visualisations XPLIA

## Introduction

Le module de visualisations XPLIA offre un système modulaire et extensible pour créer des graphiques et des visualisations dans vos rapports. Ce guide vous explique comment utiliser le système de visualisation dans différents contextes.

## Types de graphiques disponibles

XPLIA prend en charge les types de graphiques suivants :

- **Graphiques à barres** (`bar_chart`) : Parfaits pour comparer des catégories distinctes.
- **Graphiques linéaires** (`line_chart`) : Idéaux pour montrer l'évolution temporelle ou des tendances.
- **Graphiques camembert** (`pie_chart`) : Utiles pour montrer des proportions d'un tout.
- **Nuages de points** (`scatter_chart`) : Pour visualiser la relation entre deux variables.
- **Cartes de chaleur** (`heatmap_chart`) : Pour visualiser des données matricielles avec codage couleur.
- **Graphiques radar** (`radar_chart`) : Pour comparer des entités sur plusieurs dimensions.
- **Boîtes à moustaches** (`boxplot_chart`) : Pour visualiser la distribution statistique des données.
- **Histogrammes** (`histogram_chart`) : Pour visualiser la distribution de données continues.
- **Treemaps** (`treemap_chart`) : Pour visualiser des données hiérarchiques avec des rectangles imbriqués.
- **Diagrammes de Sankey** (`sankey_chart`) : Pour visualiser des flux entre des éléments.
- **Tableaux** (`table_chart`) : Pour présenter des données sous forme tabulaire.

## Utilisation dans les rapports

### Intégration dans les rapports HTML

Pour intégrer des visualisations dans un rapport HTML :

```python
from xplia.compliance.formatters.html_formatter import HTMLReportGenerator
from xplia.visualizations.chart_generator import ChartType

# Créer le générateur de rapport
generator = HTMLReportGenerator(language='fr')

# Préparer les visualisations
visualizations = [
    {
        'type': ChartType.BAR,  # Type de graphique
        'data': {
            'labels': ['Catégorie A', 'Catégorie B', 'Catégorie C'],
            'datasets': [{
                'label': 'Données 1',
                'data': [10, 20, 30]
            }]
        },
        'config': {  # Configuration optionnelle
            'title': 'Mon graphique à barres',
            'colors': ['#FF5733', '#33FF57', '#3357FF']
        },
        'title': 'Analyse des catégories',  # Titre de la visualisation dans le rapport
        'description': 'Ce graphique montre la répartition par catégorie.'  # Description optionnelle
    }
]

# Créer le contenu du rapport
content = {
    'title': 'Rapport avec visualisations',
    'visualizations': visualizations,
    # Autres données du rapport...
}

# Générer le rapport
html_content = generator.generate(content)
```

### Intégration dans les rapports PDF

Pour intégrer des visualisations dans un rapport PDF :

```python
from xplia.compliance.formatters.pdf_formatter import PDFReportGenerator
from xplia.visualizations.chart_generator import ChartType

# Créer le générateur de rapport
generator = PDFReportGenerator(language='fr')

# Préparer les visualisations (même structure que pour HTML)
visualizations = [
    {
        'type': ChartType.PIE,
        'data': {
            'labels': ['Segment A', 'Segment B', 'Segment C'],
            'datasets': [{
                'data': [30, 50, 20]
            }]
        },
        'title': 'Répartition par segment',
        'description': 'Ce graphique montre la répartition par segment.'
    }
]

# Créer le contenu du rapport
content = {
    'title': 'Rapport avec visualisations',
    'visualizations': visualizations,
    # Autres données du rapport...
}

# Générer le rapport
pdf_bytes = generator.generate(content)
```

## Utilisation directe du générateur de graphiques

Vous pouvez également utiliser le générateur de graphiques directement :

```python
from xplia.visualizations.chart_generator import ChartGenerator, ChartType

# Créer le générateur
chart_gen = ChartGenerator()

# Générer un graphique à barres
bar_chart = chart_gen.bar_chart(
    labels=['A', 'B', 'C'],
    datasets=[{
        'label': 'Série 1',
        'data': [10, 20, 30]
    }],
    title='Mon graphique',
    colors=['#FF5733', '#33FF57', '#3357FF']
)

# Pour les graphiques Plotly, vous pouvez obtenir une figure Plotly
import plotly.io as pio
fig = bar_chart  # Si le backend est Plotly
pio.show(fig)  # Affiche le graphique dans le navigateur

# Pour les graphiques matplotlib
import matplotlib.pyplot as plt
fig = bar_chart  # Si le backend est matplotlib
plt.show()  # Affiche le graphique
```

## Configuration avancée

### Personnalisation des graphiques

Tous les types de graphiques acceptent une configuration pour personnaliser leur apparence :

```python
# Configuration d'un graphique à barres
bar_chart = chart_gen.bar_chart(
    labels=['A', 'B', 'C'],
    datasets=[{'label': 'Série 1', 'data': [10, 20, 30]}],
    # Options spécifiques aux graphiques à barres
    horizontal=True,  # Barres horizontales
    stacked=False,    # Barres empilées
    # Options générales
    title='Mon graphique',
    colors=['#FF5733', '#33FF57', '#3357FF'],
    height=400,
    width=600,
    show_legend=True,
    theme='light'  # ou 'dark'
)
```

### Utilisation de différents backends

XPLIA peut utiliser différentes bibliothèques de visualisation comme backend :

```python
from xplia.visualizations.chart_generator import ChartGenerator, ChartBackend

# Utiliser Plotly comme backend
chart_gen = ChartGenerator(backend=ChartBackend.PLOTLY)

# Utiliser matplotlib comme backend
chart_gen = ChartGenerator(backend=ChartBackend.MATPLOTLIB)

# Utiliser Chart.js comme backend (pour HTML)
chart_gen = ChartGenerator(backend=ChartBackend.CHARTJS)
```

## Exemples de code par type de graphique

### Graphique à barres

```python
bar_chart = chart_gen.bar_chart(
    labels=['Catégorie A', 'Catégorie B', 'Catégorie C'],
    datasets=[
        {
            'label': 'Groupe 1',
            'data': [10, 20, 30]
        },
        {
            'label': 'Groupe 2',
            'data': [15, 25, 35]
        }
    ],
    title='Comparaison par catégorie',
    horizontal=False,
    stacked=False
)
```

### Graphique linéaire

```python
line_chart = chart_gen.line_chart(
    labels=['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin'],
    datasets=[
        {
            'label': 'Série 2023',
            'data': [10, 20, 15, 25, 30, 35]
        },
        {
            'label': 'Série 2024',
            'data': [15, 25, 20, 30, 35, 40]
        }
    ],
    title='Évolution temporelle',
    fill=False,  # Remplir sous la ligne
    tension=0.3  # Lissage de la courbe
)
```

### Graphique camembert

```python
pie_chart = chart_gen.pie_chart(
    labels=['Segment A', 'Segment B', 'Segment C'],
    datasets=[{
        'data': [30, 50, 20]
    }],
    title='Répartition par segment',
    donut=False  # True pour un graphique en anneau
)
```

### Carte de chaleur

```python
import numpy as np

# Création de données matricielles
data = np.random.rand(10, 12)
x_labels = [f'Col {i}' for i in range(data.shape[1])]
y_labels = [f'Ligne {i}' for i in range(data.shape[0])]

heatmap = chart_gen.heatmap_chart(
    data=data,
    x_labels=x_labels,
    y_labels=y_labels,
    title='Carte de chaleur',
    colorscale='Viridis'  # Échelle de couleurs
)
```

## Bonnes pratiques

1. **Choisissez le bon type de graphique** pour vos données :
   - Utilisez les graphiques à barres pour comparer des catégories
   - Utilisez les graphiques linéaires pour montrer l'évolution temporelle
   - Utilisez les camemberts pour montrer des proportions (limités à 6-7 segments)
   
2. **Limitez la quantité de données** affichées dans un seul graphique pour éviter la surcharge cognitive.

3. **Utilisez une palette de couleurs cohérente** pour assurer la lisibilité et l'accessibilité.

4. **Ajoutez toujours des titres et légendes** pour faciliter la compréhension.

5. **Testez vos visualisations** dans différents formats de sortie (HTML, PDF) pour vérifier leur rendu.

## Dépannage

### Problèmes courants

- **Graphiques non rendus dans le rapport PDF** : Vérifiez que les dépendances nécessaires (matplotlib, plotly) sont installées.

- **Erreurs de données** : Assurez-vous que le format des données correspond à celui attendu par le type de graphique.

- **Problèmes d'affichage** : Essayez un autre backend si vous rencontrez des problèmes avec le backend par défaut.

### Support des navigateurs

- Les visualisations HTML utilisent Chart.js qui est compatible avec tous les navigateurs modernes.
- Pour les visualisations interactives avec Plotly, assurez-vous que JavaScript est activé dans le navigateur.

## Ressources additionnelles

- Documentation de l'API XPLIA : voir `docs/visualizations_api.md`
- Exemples : voir le répertoire `examples/` pour des démonstrations complètes
