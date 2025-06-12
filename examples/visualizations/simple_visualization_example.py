#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exemple simple d'utilisation du module de visualisations XPLIA
=============================================================

Ce script démontre comment utiliser le module de visualisations 
pour générer des graphiques et les exporter en HTML et images.
"""

import os
import sys
from pathlib import Path

# Ajout du répertoire parent au path pour pouvoir importer xplia
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

try:
    from xplia.visualizations import (
        ChartGenerator, ChartType, ChartLibrary, OutputContext
    )
except ImportError as e:
    print(f"Erreur d'importation: {e}")
    print("Ce script suppose que le module xplia.visualizations est disponible.")
    sys.exit(1)


def generate_simple_html():
    """
    Génère une page HTML simple avec plusieurs visualisations.
    """
    # Création du générateur de graphiques
    generator = ChartGenerator(
        library=ChartLibrary.PLOTLY,
        theme="light",
        output_context=OutputContext.WEB,
        interactive=True,
        responsive=True
    )
    
    # Préparation du répertoire de sortie
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Liste pour stocker le HTML de tous les graphiques
    all_charts_html = []
    
    # 1. Graphique à barres
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
    
    bar_chart = generator.create_chart(ChartType.BAR, bar_data, bar_config)
    bar_html = generator.to_html(bar_chart)
    all_charts_html.append({
        "title": "Graphique à barres",
        "description": "Distribution des ventes par produit",
        "html": bar_html
    })
    
    # Sauvegarde individuelle
    generator.save(bar_chart, str(output_dir / "bar_chart.html"))
    generator.save(bar_chart, str(output_dir / "bar_chart.png"))
    
    # 2. Graphique en ligne avec séries multiples
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
    
    line_chart = generator.create_chart(ChartType.LINE, line_data, line_config)
    line_html = generator.to_html(line_chart)
    all_charts_html.append({
        "title": "Graphique en ligne",
        "description": "Évolution des ventes sur 6 mois",
        "html": line_html
    })
    
    # 3. Graphique circulaire
    pie_data = {
        "labels": ["Sécurité", "Performance", "Accessibilité", "Autre"],
        "values": [45, 30, 15, 10]
    }
    
    pie_config = {
        "title": "Répartition du budget",
        "hole": 0.4,  # Crée un donut chart
        "show_percent": True
    }
    
    pie_chart = generator.create_chart(ChartType.PIE, pie_data, pie_config)
    pie_html = generator.to_html(pie_chart)
    all_charts_html.append({
        "title": "Graphique circulaire",
        "description": "Répartition du budget par catégorie",
        "html": pie_html
    })
    
    # 4. Heatmap
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
    
    heatmap_chart = generator.create_chart(ChartType.HEATMAP, heatmap_data, heatmap_config)
    heatmap_html = generator.to_html(heatmap_chart)
    all_charts_html.append({
        "title": "Carte de chaleur",
        "description": "Matrice d'évaluation des risques",
        "html": heatmap_html
    })
    
    # Génération d'une page HTML simple avec tous les graphiques
    html_content = generate_html_page(all_charts_html)
    
    # Sauvegarde de la page HTML complète
    output_path = output_dir / "visualisations_demo.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Page HTML générée avec succès: {output_path}")
    return str(output_path)


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
    <title>Démonstration des visualisations XPLIA</title>
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
        <h1>Démonstration des visualisations XPLIA</h1>
        <p>Exemples d'utilisation du module de visualisations pour générer différents types de graphiques</p>
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
</body>
</html>
"""
    
    return html


if __name__ == "__main__":
    try:
        output_path = generate_simple_html()
        
        # Ouvrir le fichier HTML dans le navigateur par défaut
        import webbrowser
        webbrowser.open(f"file://{output_path}")
        
    except Exception as e:
        print(f"Erreur lors de la génération de la page HTML: {e}")
        import traceback
        traceback.print_exc()
