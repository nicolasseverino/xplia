#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exemple d'utilisation directe du module de visualisations de XPLIA
=================================================================

Ce script démontre comment utiliser directement les classes de visualisation
sans dépendre des autres composants du système XPLIA.
"""

import os
import sys
import json
import random
from pathlib import Path

# Ajout du répertoire parent au path pour pouvoir importer les modules XPLIA
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.append(str(project_root))

# Import des modules spécifiques de visualisation sans dépendre des autres modules XPLIA
# Importation directe des modules de visualisation pour éviter les dépendances complexes
sys.path.append(str(project_root / "xplia" / "visualizations"))

# Importation directe des modules de charts
from xplia.visualizations.charts.bar_chart import create_bar_chart
from xplia.visualizations.charts.line_chart import create_line_chart
from xplia.visualizations.charts.pie_chart import create_pie_chart
from xplia.visualizations.charts.heatmap_chart import create_heatmap_chart
from xplia.visualizations.charts.radar_chart import create_radar_chart


# Définition des constantes d'énumération pour simuler les enums du code complet
class ChartLibrary:
    PLOTLY = "plotly"
    MATPLOTLIB = "matplotlib"
    BOKEH = "bokeh"


class OutputContext:
    WEB = "web"
    PDF = "pdf"
    IMAGE = "image"


def generate_example_html():
    """
    Génère une page HTML simple avec plusieurs visualisations.
    """
    print("Génération de visualisations...")
    
    # Préparation du répertoire de sortie
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Liste pour stocker le HTML de tous les graphiques
    all_charts_html = []
    
    # 1. Graphique à barres
    try:
        print("Création du graphique à barres...")
        bar_data = {
            "x": ["Produit A", "Produit B", "Produit C", "Produit D", "Produit E"],
            "y": [15, 25, 12, 8, 22],
            "color": ["#3366CC", "#DC3912", "#FF9900", "#109618", "#990099"]
        }
        
        bar_config = {
            "title": "Ventes par produit",
            "x_title": "Produits",
            "y_title": "Ventes (milliers €)",
            "template": "plotly_white"
        }
        
        bar_fig = create_bar_chart(bar_data, bar_config, library=ChartLibrary.PLOTLY)
        bar_html = bar_fig.to_html(include_plotlyjs='cdn', full_html=False)
        
        all_charts_html.append({
            "title": "Graphique à barres",
            "description": "Distribution des ventes par produit",
            "html": bar_html
        })
        
        # Sauvegarde individuelle en HTML et PNG
        with open(str(output_dir / "bar_chart.html"), 'w', encoding='utf-8') as f:
            f.write(bar_fig.to_html(include_plotlyjs='cdn'))
            
    except Exception as e:
        print(f"Erreur lors de la création du graphique à barres: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. Graphique en ligne avec séries multiples
    try:
        print("Création du graphique en ligne...")
        line_data = {
            "x": ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin"],
            "series": [
                {"name": "2023", "values": [10, 15, 12, 18, 22, 25]},
                {"name": "2024", "values": [12, 18, 20, 24, 25, 30]}
            ]
        }
        
        line_config = {
            "title": "Évolution des ventes",
            "x_title": "Mois",
            "y_title": "Ventes (milliers €)",
            "legend": True,
            "markers": True
        }
        
        line_fig = create_line_chart(line_data, line_config, library=ChartLibrary.PLOTLY)
        line_html = line_fig.to_html(include_plotlyjs='cdn', full_html=False)
        
        all_charts_html.append({
            "title": "Graphique en ligne",
            "description": "Évolution des ventes sur 6 mois",
            "html": line_html
        })
        
        # Sauvegarde individuelle
        with open(str(output_dir / "line_chart.html"), 'w', encoding='utf-8') as f:
            f.write(line_fig.to_html(include_plotlyjs='cdn'))
            
    except Exception as e:
        print(f"Erreur lors de la création du graphique en ligne: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. Graphique circulaire
    try:
        print("Création du graphique circulaire...")
        pie_data = {
            "labels": ["Sécurité", "Performance", "Accessibilité", "Autre"],
            "values": [45, 30, 15, 10]
        }
        
        pie_config = {
            "title": "Répartition du budget",
            "hole": 0.4,  # Crée un donut chart
            "show_percent": True
        }
        
        pie_fig = create_pie_chart(pie_data, pie_config, library=ChartLibrary.PLOTLY)
        pie_html = pie_fig.to_html(include_plotlyjs='cdn', full_html=False)
        
        all_charts_html.append({
            "title": "Graphique circulaire",
            "description": "Répartition du budget par catégorie",
            "html": pie_html
        })
        
        # Sauvegarde individuelle
        with open(str(output_dir / "pie_chart.html"), 'w', encoding='utf-8') as f:
            f.write(pie_fig.to_html(include_plotlyjs='cdn'))
            
    except Exception as e:
        print(f"Erreur lors de la création du graphique circulaire: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. Heatmap
    try:
        print("Création de la heatmap...")
        heatmap_data = {
            "x": ["Très rare", "Rare", "Possible", "Probable", "Certain"],
            "y": ["Négligeable", "Mineur", "Modéré", "Majeur", "Critique"],
            "z": [
                [1, 2, 3, 4, 5],
                [2, 4, 6, 8, 10],
                [3, 6, 9, 12, 15],
                [4, 8, 12, 16, 20],
                [5, 10, 15, 20, 25]
            ]
        }
        
        heatmap_config = {
            "title": "Matrice des risques",
            "x_title": "Probabilité",
            "y_title": "Impact",
            "colorscale": [
                [0, "green"],
                [0.4, "yellow"],
                [0.7, "orange"],
                [1, "red"]
            ],
            "annotations": True
        }
        
        heatmap_fig = create_heatmap_chart(heatmap_data, heatmap_config, library=ChartLibrary.PLOTLY)
        heatmap_html = heatmap_fig.to_html(include_plotlyjs='cdn', full_html=False)
        
        all_charts_html.append({
            "title": "Carte de chaleur",
            "description": "Matrice d'évaluation des risques",
            "html": heatmap_html
        })
        
        # Sauvegarde individuelle
        with open(str(output_dir / "heatmap_chart.html"), 'w', encoding='utf-8') as f:
            f.write(heatmap_fig.to_html(include_plotlyjs='cdn'))
            
    except Exception as e:
        print(f"Erreur lors de la création de la heatmap: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. Radar chart
    try:
        print("Création du radar chart...")
        radar_data = {
            "categories": ["Sécurité", "Performance", "Accessibilité", 
                        "Durabilité", "Confidentialité", "Évolutivité"],
            "series": [
                {"name": "Score actuel", "values": [85, 92, 78, 95, 88, 82]},
                {"name": "Objectif", "values": [90, 95, 90, 95, 90, 85]}
            ]
        }
        
        radar_config = {
            "title": "Profil de conformité global",
            "fill": True,
            "show_legend": True
        }
        
        radar_fig = create_radar_chart(radar_data, radar_config, library=ChartLibrary.PLOTLY)
        radar_html = radar_fig.to_html(include_plotlyjs='cdn', full_html=False)
        
        all_charts_html.append({
            "title": "Radar chart",
            "description": "Évaluation multidimensionnelle de la conformité",
            "html": radar_html
        })
        
        # Sauvegarde individuelle
        with open(str(output_dir / "radar_chart.html"), 'w', encoding='utf-8') as f:
            f.write(radar_fig.to_html(include_plotlyjs='cdn'))
            
    except Exception as e:
        print(f"Erreur lors de la création du radar chart: {e}")
        import traceback
        traceback.print_exc()
    
    # Génération d'une page HTML simple avec tous les graphiques
    if all_charts_html:
        print("Génération de la page HTML complète...")
        html_content = generate_html_page(all_charts_html)
        
        # Sauvegarde de la page HTML complète
        output_path = output_dir / "visualisations_demo_direct.html"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Page HTML générée avec succès: {output_path}")
        return str(output_path)
    else:
        print("Aucun graphique n'a pu être généré.")
        return None


def generate_html_page(charts):
    """
    Génère une page HTML complète avec les graphiques fournis.
    
    Args:
        charts: Liste de dictionnaires contenant title, description et html pour chaque graphique
        
    Returns:
        Contenu HTML complet de la page
    """
    html = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Démonstration des visualisations XPLIA (Direct)</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }
        
        header {
            background-color: #0056b3;
            color: white;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        h1 {
            margin: 0;
        }
        
        .container {
            display: flex;
            flex-direction: column;
            gap: 30px;
        }
        
        .chart-container {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            padding: 20px;
            margin-bottom: 30px;
        }
        
        .chart-title {
            font-size: 24px;
            margin-top: 0;
            color: #0056b3;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
            margin-bottom: 10px;
        }
        
        .chart-description {
            color: #666;
            margin-bottom: 20px;
            font-style: italic;
        }
        
        .chart-content {
            width: 100%;
            height: 100%;
            min-height: 400px;
        }
        
        footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            text-align: center;
            color: #666;
        }
        
        @media (max-width: 768px) {
            body {
                padding: 10px;
            }
            
            .chart-container {
                padding: 15px;
            }
        }
    </style>
</head>
<body>
    <header>
        <h1>Démonstration des visualisations XPLIA (Direct)</h1>
        <p>Exemples d'utilisation directe des modules de visualisations</p>
    </header>
    
    <div class="container">
"""
    
    # Ajout de chaque graphique
    for chart in charts:
        html += f"""
        <div class="chart-container">
            <h2 class="chart-title">{chart['title']}</h2>
            <p class="chart-description">{chart['description']}</p>
            <div class="chart-content">
                {chart['html']}
            </div>
        </div>
"""
    
    # Fermeture du HTML
    html += """
    </div>
    
    <footer>
        <p>&copy; 2024 XPLIA - Module de visualisations</p>
    </footer>

    <!-- Script pour ajuster la taille des graphiques -->
    <script>
        window.addEventListener('load', function() {
            // Ajustement de la taille pour les graphiques responsive
            window.dispatchEvent(new Event('resize'));
        });
    </script>
</body>
</html>
"""
    
    return html


if __name__ == "__main__":
    try:
        output_path = generate_example_html()
        
        if output_path:
            # Ouvrir le fichier HTML dans le navigateur par défaut
            import webbrowser
            print(f"Ouverture du fichier dans le navigateur: {output_path}")
            webbrowser.open(f"file://{output_path}")
        
    except Exception as e:
        print(f"Erreur lors de l'exécution du script: {e}")
        import traceback
        traceback.print_exc()
