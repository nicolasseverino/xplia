#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exemple d'utilisation des visualisations dans un rapport HTML XPLIA
==================================================================

Ce script démontre comment intégrer des visualisations générées
par le module visualizations.py dans un rapport HTML.
"""

import os
import sys
import datetime
from pathlib import Path

# Ajout du répertoire parent au path pour pouvoir importer xplia
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from xplia.compliance.formatters.html_formatter import HTMLReportGenerator
from xplia.compliance.report_base import ReportContent, ReportConfig
from xplia.visualizations import ChartType


def generate_sample_report():
    """
    Génère un rapport d'exemple avec des visualisations intégrées.
    """
    # Configuration du générateur de rapports
    config = ReportConfig(
        template_name="standard",
        language="fr",
        chart_theme="light",
        include_signatures=True,
        include_verification_qr=True,
        verification_url="https://xplia.com/verify?id=123456"
    )
    
    # Création du générateur HTML
    generator = HTMLReportGenerator(config)
    
    # Contenu du rapport
    content = ReportContent(
        title="Rapport de conformité avec visualisations",
        subtitle="Démonstration d'intégration des graphiques",
        author="XPLIA",
        summary="Ce rapport démontre l'intégration des visualisations dans les rapports de conformité XPLIA. "
                "Il présente plusieurs types de graphiques générés dynamiquement et intégrés dans le document HTML.",
        recommendations=[
            {
                "id": "REC-001",
                "title": "Utiliser des visualisations appropriées",
                "description": "Choisir le type de visualisation le plus adapté au message que l'on souhaite communiquer."
            },
            {
                "id": "REC-002",
                "title": "Maintenir les couleurs cohérentes",
                "description": "Utiliser une palette de couleurs cohérente à travers toutes les visualisations."
            }
        ],
        metadata={
            "created_at": datetime.datetime.now().isoformat(),
            "version": "1.0.0",
            "category": "Exemple"
        }
    )
    
    # Ajout des visualisations
    content.visualizations = [
        # Exemple 1: Graphique à barres
        {
            "title": "Distribution des scores de conformité",
            "description": "Répartition des scores de conformité par catégorie",
            "type": "bar",
            "data": {
                "x": ["Sécurité", "Performance", "Accessibilité", "Durabilité", "Confidentialité"],
                "y": [85, 92, 78, 95, 88]
            },
            "config": {
                "x_title": "Catégories",
                "y_title": "Score (%)",
                "colors": ["#3366CC", "#DC3912", "#FF9900", "#109618", "#990099"]
            }
        },
        
        # Exemple 2: Graphique en ligne avec séries multiples
        {
            "title": "Évolution des scores sur 6 mois",
            "description": "Progression des scores de conformité sur la période",
            "type": "line",
            "data": {
                "x": ["Jan", "Fév", "Mars", "Avr", "Mai", "Juin"],
                "series": [
                    {"name": "Sécurité", "values": [70, 72, 78, 82, 85, 85]},
                    {"name": "Performance", "values": [80, 82, 85, 88, 90, 92]},
                    {"name": "Confidentialité", "values": [75, 78, 80, 84, 86, 88]}
                ]
            },
            "config": {
                "x_title": "Mois",
                "y_title": "Score (%)",
                "markers": True
            }
        },
        
        # Exemple 3: Radar chart pour les évaluations multidimensionnelles
        {
            "title": "Profil de conformité global",
            "description": "Évaluation multidimensionnelle de la conformité",
            "type": "radar",
            "data": {
                "categories": ["Sécurité", "Performance", "Accessibilité", 
                             "Durabilité", "Confidentialité", "Évolutivité"],
                "series": [
                    {"name": "Score actuel", "values": [85, 92, 78, 95, 88, 82]},
                    {"name": "Objectif", "values": [90, 95, 90, 95, 90, 85]}
                ]
            },
            "config": {
                "fill": True,
                "show_legend": True
            }
        },
        
        # Exemple 4: Pie chart pour la distribution
        {
            "title": "Répartition des non-conformités",
            "description": "Distribution des problèmes par catégorie",
            "type": "pie",
            "data": {
                "labels": ["Sécurité", "Performance", "Accessibilité", "Autre"],
                "values": [12, 5, 8, 3]
            },
            "config": {
                "hole": 0.4,  # Pour créer un donut chart
                "show_percent": True
            }
        },
        
        # Exemple 5: Heatmap pour les matrices de risque
        {
            "title": "Matrice des risques de conformité",
            "description": "Impact vs Probabilité des risques identifiés",
            "type": "heatmap",
            "data": {
                "x": ["Très rare", "Rare", "Possible", "Probable", "Certain"],
                "y": ["Négligeable", "Mineur", "Modéré", "Majeur", "Critique"],
                "z": [
                    [1, 2, 3, 4, 5],
                    [2, 4, 6, 8, 10],
                    [3, 6, 9, 12, 15],
                    [4, 8, 12, 16, 20],
                    [5, 10, 15, 20, 25]
                ]
            },
            "config": {
                "colorscale": [
                    [0, "green"],
                    [0.4, "yellow"],
                    [0.7, "orange"],
                    [1, "red"]
                ],
                "annotations": True
            }
        }
    ]
    
    # Ajout de données complémentaires
    content.compliance_score = {
        "global": 88,
        "details": {
            "Sécurité": 85,
            "Performance": 92,
            "Accessibilité": 78,
            "Durabilité": 95,
            "Confidentialité": 88
        }
    }
    
    # Génération du rapport
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "rapport_avec_visualisations.html"
    generator.generate(content, str(output_path))
    
    print(f"Rapport généré avec succès: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    report_path = generate_sample_report()
    
    # Ouvrir le rapport dans le navigateur par défaut
    import webbrowser
    webbrowser.open(f"file://{report_path}")
