"""
Exemple d'intégration des visualisations dans un rapport HTML
===========================================================

Ce script démontre comment intégrer des visualisations dans un rapport HTML
en utilisant le générateur de rapports HTML de XPLIA.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Ajouter le répertoire racine au PYTHONPATH pour permettre l'importation de xplia
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import des modules XPLIA
from xplia.compliance.report_base import ReportContent, ReportConfig
from xplia.compliance.formatters.html_formatter import HTMLReportGenerator
from xplia.visualizations import ChartType, ChartLibrary

def main():
    """Fonction principale de démonstration."""
    print("Génération d'un exemple de rapport avec visualisations...")
    
    # Création de données d'exemple
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    values = np.random.normal(100, 15, 30).cumsum()
    anomalies = np.random.choice([0, 1], size=30, p=[0.9, 0.1])
    
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
    
    # Configuration du rapport
    config = ReportConfig(
        template_name="standard",
        chart_theme="light",
        include_verification_qr=True,
        verification_url="https://xplia.org/verify/123456",
        include_signatures=True,
        language="fr"
    )
    
    # Contenu du rapport
    content = ReportContent(
        title="Rapport de performance avec visualisations",
        subtitle="Démo d'intégration des visualisations dans un rapport HTML",
        author="Équipe XPLIA",
        date=datetime.now().isoformat(),
        summary="Ce rapport démontre l'intégration de différents types de visualisations dans un rapport HTML généré par XPLIA.",
        sections=[
            {
                "title": "Introduction",
                "content": "Cette section présente les différentes visualisations intégrées au rapport."
            },
            {
                "title": "Répartition par catégorie",
                "content": "Le graphique ci-dessous montre la répartition des valeurs par catégorie."
            },
            {
                "title": "Évolution temporelle",
                "content": "Le graphique ci-dessous montre l'évolution des valeurs au cours du temps."
            },
            {
                "title": "Distribution en camembert",
                "content": "Ce graphique montre la répartition proportionnelle des catégories."
            },
            {
                "title": "Conclusion",
                "content": "Les visualisations interactives permettent une meilleure compréhension des données."
            }
        ],
        # Liste de visualisations à intégrer
        visualizations=[
            {
                "type": "bar",
                "title": "Répartition par catégorie",
                "description": "Ce graphique montre la distribution des valeurs par catégorie.",
                "data": bar_data,
                "config": {
                    "colorScale": "Blues",
                    "axis_titles": {"x": "Catégories", "y": "Valeurs"}
                }
            },
            {
                "type": "line",
                "title": "Évolution temporelle",
                "description": "Ce graphique montre l'évolution des valeurs au cours du temps.",
                "data": line_data,
                "config": {
                    "colorScale": "Greens",
                    "axis_titles": {"x": "Date", "y": "Valeur"},
                    "markers": True
                }
            },
            {
                "type": "pie",
                "title": "Distribution en camembert",
                "description": "Ce graphique montre la répartition proportionnelle des catégories.",
                "data": pie_data,
                "config": {
                    "colorScale": "Set2",
                    "donut": True,
                    "legend": {"orientation": "v"}
                }
            }
        ],
        metadata={
            "project_id": "demo-123",
            "classification": "Public",
            "tags": ["demo", "visualisations", "rapport"]
        }
    )
    
    # Création du générateur et du rapport
    generator = HTMLReportGenerator(config)
    output_path = os.path.join(os.getcwd(), "output", "demo_visualizations_report.html")
    
    # S'assurer que le répertoire existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Génération du rapport
    generator.generate(content, output_path)
    
    print(f"Rapport généré avec succès: {output_path}")
    print("Ce rapport contient 3 visualisations: un graphique à barres, un graphique linéaire et un camembert.")

if __name__ == "__main__":
    main()
