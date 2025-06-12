"""
Démonstrateur autonome de visualisations
=======================================

Ce script démontre l'intégration des visualisations dans un rapport HTML
sans dépendre de l'infrastructure complète XPLIA.
"""

import os
import sys
import json
import importlib.util
from datetime import datetime
from pathlib import Path

# Vérification des dépendances requises
def check_dependency(package_name):
    """Vérifie si un package est installé et retourne True/False."""
    return importlib.util.find_spec(package_name) is not None

# Liste des dépendances requises
required_dependencies = {
    'numpy': "Pour les manipulations de données numériques",
    'pandas': "Pour la manipulation de données tabulaires",
    'plotly': "Pour la création de visualisations interactives (ESSENTIEL)"
}

# Vérification des dépendances
missing_dependencies = []
for dep, desc in required_dependencies.items():
    if not check_dependency(dep):
        missing_dependencies.append((dep, desc))

# Afficher un avertissement pour les dépendances manquantes
if missing_dependencies:
    print("⚠️ ATTENTION ⚠️")
    print("Les dépendances suivantes sont manquantes et sont requises pour ce script:")
    for dep, desc in missing_dependencies:
        print(f" - {dep}: {desc}")
    
    # Si plotly est manquant, c'est critique pour ce démonstrateur
    if any(dep == 'plotly' for dep, _ in missing_dependencies):
        print("\n⛔ ERREUR CRITIQUE: 'plotly' est requis pour ce démonstrateur et ne peut pas être contourné.")
        print("Veuillez installer les dépendances requises via: pip install -r requirements.txt")
        print("Ou directement: pip install plotly pandas numpy")
        sys.exit(1)
    
    print("\nInstallation recommandée: pip install -r requirements.txt")
    print("Tentative de continuer avec des fonctionnalités limitées...\n")

# Importer les bibliothèques nécessaires si disponibles
if check_dependency('pandas'):
    import pandas as pd
else:
    print("pandas indisponible, utilisation d'alternatives pour les données de démonstration.")
    # Classe minimale pour simuler pd.date_range
    class DateRangeReplacement:
        def __init__(self, dates):
            self.dates = dates
        
        def strftime(self, format):
            from datetime import datetime, timedelta
            start_date = datetime.strptime(self.dates[0], "%Y-%m-%d")
            return [datetime.strftime(start_date + timedelta(days=i), format) for i in range(len(self.dates))]
    
    # Module de remplacement simple pour pd
    class PandasReplacement:
        def date_range(self, start, periods, freq):
            from datetime import datetime, timedelta
            start_date = datetime.strptime(start, "%Y-%m-%d")
            return DateRangeReplacement([datetime.strftime(start_date + timedelta(days=i), "%Y-%m-%d") for i in range(periods)])
    
    pd = PandasReplacement()

if check_dependency('numpy'):
    import numpy as np
else:
    print("numpy indisponible, utilisation d'alternatives pour les données de démonstration.")
    # Module de remplacement simple pour np.random
    class RandomModule:
        def seed(self, seed_value):
            import random
            random.seed(seed_value)
            
        def randint(self, start, end, size=None):
            import random
            if isinstance(size, tuple):
                # Pour les tableaux multidimensionnels
                size = size[0] if len(size) == 1 else size
            if isinstance(size, int):
                return [random.randint(start, end-1) for _ in range(size)]
            return random.randint(start, end-1)
            
        def normal(self, mean, std, size):
            import random
            return [random.normalvariate(mean, std) for _ in range(size)]
    
    class NumpyReplacement:
        def __init__(self):
            self.random = RandomModule()
            
        def cumsum(self, values):
            result = []
            total = 0
            for v in values:
                total += v
                result.append(total)
            return result
    
    np = NumpyReplacement()

# Ajouter le répertoire racine au PYTHONPATH pour permettre l'importation selectivement
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import UNIQUEMENT les modules de visualisation
# Cet import sélectif évite les dépendances problématiques comme joblib
sys.path.insert(0, os.path.join(project_root, 'xplia'))

# Classes minimales pour simuler les fonctionnalités du rapport
class SimpleReportConfig:
    """Version simplifiée de ReportConfig."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class SimpleReportContent:
    """Version simplifiée de ReportContent."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Fonctions d'aide pour générer un rapport HTML simple
def generate_html_report(title, visualizations_html):
    """Génère un rapport HTML simple avec des visualisations."""
    html_template = """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            .report-header {{
                text-align: center;
                margin-bottom: 40px;
                border-bottom: 1px solid #ddd;
                padding-bottom: 20px;
            }}
            .visualization-container {{
                margin-bottom: 40px;
                padding: 20px;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                background-color: #fff;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }}
            .visualization-title {{
                font-size: 1.4em;
                margin-bottom: 10px;
                color: #2c3e50;
            }}
            .visualization-description {{
                color: #7f8c8d;
                margin-bottom: 20px;
            }}
            footer {{
                margin-top: 50px;
                text-align: center;
                font-size: 0.9em;
                color: #7f8c8d;
                border-top: 1px solid #ddd;
                padding-top: 20px;
            }}
        </style>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <div class="report-header">
            <h1>{title}</h1>
            <p>Généré le {date}</p>
        </div>
        
        <div class="report-content">
            {visualizations}
        </div>
        
        <footer>
            <p>Rapport généré par le démonstrateur autonome XPLIA</p>
        </footer>
    </body>
    </html>
    """
    
    visualizations_html_content = ""
    for viz in visualizations_html:
        viz_container = f"""
        <div class="visualization-container">
            <h2 class="visualization-title">{viz['title']}</h2>
            <p class="visualization-description">{viz['description']}</p>
            <div class="visualization">{viz['html']}</div>
        </div>
        """
        visualizations_html_content += viz_container
    
    return html_template.format(
        title=title,
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        visualizations=visualizations_html_content
    )

def create_bar_chart(data, config=None):
    """Crée un graphique à barres directement avec Plotly."""
    import plotly.graph_objects as go
    
    config = config or {}
    fig = go.Figure(data=[
        go.Bar(
            x=data['x'],
            y=data['y'],
            marker_color=config.get('color', 'royalblue')
        )
    ])
    
    if 'title' in config:
        fig.update_layout(title_text=config['title'])
    if 'axis_titles' in config:
        fig.update_layout(
            xaxis_title=config['axis_titles'].get('x', ''),
            yaxis_title=config['axis_titles'].get('y', '')
        )
    
    return fig

def create_line_chart(data, config=None):
    """Crée un graphique linéaire directement avec Plotly."""
    import plotly.graph_objects as go
    
    config = config or {}
    fig = go.Figure(data=[
        go.Scatter(
            x=data['x'],
            y=data['y'],
            mode='lines+markers' if config.get('markers', True) else 'lines',
            marker=dict(size=8),
            line=dict(width=2, color=config.get('color', 'green'))
        )
    ])
    
    if 'title' in config:
        fig.update_layout(title_text=config['title'])
    if 'axis_titles' in config:
        fig.update_layout(
            xaxis_title=config['axis_titles'].get('x', ''),
            yaxis_title=config['axis_titles'].get('y', '')
        )
    
    return fig

def create_pie_chart(data, config=None):
    """Crée un graphique en camembert directement avec Plotly."""
    import plotly.graph_objects as go
    
    config = config or {}
    fig = go.Figure(data=[
        go.Pie(
            labels=data['labels'],
            values=data['values'],
            hole=0.4 if config.get('donut', False) else 0
        )
    ])
    
    if 'title' in config:
        fig.update_layout(title_text=config['title'])
    
    # Configuration de la légende
    if 'legend' in config:
        fig.update_layout(
            legend=dict(
                orientation=config['legend'].get('orientation', 'h')
            )
        )
    
    return fig

def figure_to_html(fig):
    """Convertit une figure Plotly en HTML."""
    return fig.to_html(include_plotlyjs=False, full_html=False)

def main():
    """Fonction principale de démonstration."""
    print("Génération d'un exemple de rapport avec visualisations autonomes...")
    
    # Création de données d'exemple
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    values = np.random.normal(100, 15, 30).cumsum()
    
    # Données pour bar chart
    categories = ['A', 'B', 'C', 'D', 'E']
    values_bar = np.random.randint(10, 100, size=len(categories))
    bar_data = {
        'x': categories,
        'y': values_bar.tolist()
    }
    
    # Données pour line chart
    line_data = {
        'x': dates.strftime('%Y-%m-%d').tolist(),
        'y': values.tolist()
    }
    
    # Données pour pie chart
    pie_data = {
        'labels': categories,
        'values': np.random.randint(10, 100, size=len(categories)).tolist()
    }
    
    # Création des visualisations avec Plotly directement
    visualizations = []
    
    # Bar chart
    bar_config = {
        'title': 'Répartition par catégorie',
        'color': 'royalblue',
        'axis_titles': {'x': 'Catégories', 'y': 'Valeurs'}
    }
    bar_fig = create_bar_chart(bar_data, bar_config)
    visualizations.append({
        'title': 'Répartition par catégorie',
        'description': 'Ce graphique montre la distribution des valeurs par catégorie.',
        'html': figure_to_html(bar_fig)
    })
    
    # Line chart
    line_config = {
        'title': 'Évolution temporelle',
        'color': 'seagreen',
        'markers': True,
        'axis_titles': {'x': 'Date', 'y': 'Valeur'}
    }
    line_fig = create_line_chart(line_data, line_config)
    visualizations.append({
        'title': 'Évolution temporelle',
        'description': 'Ce graphique montre l\'évolution des valeurs au cours du temps.',
        'html': figure_to_html(line_fig)
    })
    
    # Pie chart
    pie_config = {
        'title': 'Distribution en camembert',
        'donut': True,
        'legend': {'orientation': 'v'}
    }
    pie_fig = create_pie_chart(pie_data, pie_config)
    visualizations.append({
        'title': 'Distribution en camembert',
        'description': 'Ce graphique montre la répartition proportionnelle des catégories.',
        'html': figure_to_html(pie_fig)
    })
    
    # Génération du rapport HTML
    report_title = "Rapport de démonstration de visualisations autonomes"
    html_content = generate_html_report(report_title, visualizations)
    
    # Création du dossier de sortie s'il n'existe pas
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarde du rapport
    output_path = os.path.join(output_dir, "standalone_visualization_demo.html")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Rapport généré avec succès: {output_path}")
    print("Ce rapport contient 3 visualisations: un graphique à barres, un graphique linéaire et un camembert.")

if __name__ == "__main__":
    main()
