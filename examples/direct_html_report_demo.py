"""
Démonstration d'intégration directe avec le formateur HTML XPLIA
=============================================================

Ce script génère un rapport HTML en utilisant directement le formateur HTML
de XPLIA et y intègre des visualisations.
"""

import os
import sys
import importlib.util
from datetime import datetime

# Vérification des dépendances requises
def check_dependency(package_name):
    """Vérifie si un package est installé et retourne True/False."""
    return importlib.util.find_spec(package_name) is not None

# Liste des dépendances requises
required_dependencies = {
    'numpy': "Pour les manipulations de données numériques",
    'joblib': "Pour la sérialisation et le chargement de modèles",
    'matplotlib': "Pour les visualisations statiques",
    'pandas': "Pour la manipulation de données tabulaires"
}

# Vérification des dépendances
missing_dependencies = []
for dep, desc in required_dependencies.items():
    if not check_dependency(dep):
        missing_dependencies.append((dep, desc))

# Afficher un avertissement pour les dépendances manquantes
if missing_dependencies:
    print("⚠️ ATTENTION ⚠️")
    print("Les dépendances suivantes sont manquantes et pourraient être nécessaires:")
    for dep, desc in missing_dependencies:
        print(f" - {dep}: {desc}")
    print("\nInstallation recommandée: pip install -r requirements.txt")
    print("Tentative de continuer avec des fonctionnalités limitées...\n")

# Importer numpy si disponible, sinon utiliser une alternative simplifiée
if check_dependency('numpy'):
    import numpy as np
else:
    print("numpy indisponible, utilisation d'alternatives pour les données de démonstration.")
    # Module de remplacement simple pour np.random
    class RandomModule:
        def randint(self, start, end, size):
            import random
            return [random.randint(start, end-1) for _ in range(size)]
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

# Ajout du répertoire parent au PYTHONPATH pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Import des modules XPLIA nécessaires
    from xplia.compliance.formatters.html_formatter import HTMLReportGenerator
    from xplia.visualizations import ChartType, ChartGenerator
except ImportError as e:
    print(f"Erreur d'importation: {e}")
    print("Ce script ne peut pas être exécuté si les imports XPLIA ne sont pas disponibles.")
    print("Utiliser plutôt le script pure_html_visualization_demo.py qui est autonome.")
    sys.exit(1)

def main():
    """Fonction principale de démonstration."""
    print("Génération d'un rapport HTML avec visualisations en utilisant le formateur XPLIA...")
    
    # Création du dossier de sortie s'il n'existe pas
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialisation du générateur de rapports HTML
    html_generator = HTMLReportGenerator()
    
    # Préparation des données de rapport
    report_data = {
        "title": "Rapport de démonstration d'intégration XPLIA",
        "summary": "Ce rapport démontre l'intégration des visualisations dans le formateur HTML de XPLIA.",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "organization": "XPLIA Demo Corp",
        "responsible": "Administrateur XPLIA",
        "formatted_date": datetime.now().strftime("%Y-%m-%d"),
        "organization_label": "Organisation",
        "responsible_label": "Responsable",
        "date_label": "Date",
        "summary_label": "Résumé",
        "compliance_score_label": "Score de conformité",
        "score": "85%",
        "compliance_status": "Conforme avec recommandations",
    }
    
    # Génération des données pour les visualisations
    np.random.seed(42)
    
    # Données pour graphique à barres
    bar_data = {
        "labels": ["Catégorie A", "Catégorie B", "Catégorie C", "Catégorie D", "Catégorie E"],
        "datasets": [{
            "label": "Valeurs",
            "data": np.random.randint(10, 100, 5).tolist()
        }]
    }
    bar_config = {
        "title": "Répartition par catégorie",
        "xAxisLabel": "Catégories",
        "yAxisLabel": "Valeurs"
    }
    
    # Données pour graphique linéaire
    line_data = {
        "labels": [f"2024-{month:02d}" for month in range(1, 13)],
        "datasets": [{
            "label": "Tendance",
            "data": np.cumsum(np.random.normal(5, 2, 12)).tolist()
        }]
    }
    line_config = {
        "title": "Évolution temporelle",
        "xAxisLabel": "Mois",
        "yAxisLabel": "Valeur cumulée"
    }
    
    # Données pour graphique en camembert
    pie_labels = ["Segment 1", "Segment 2", "Segment 3", "Segment 4", "Segment 5"]
    pie_data = {
        "labels": pie_labels,
        "datasets": [{
            "data": np.random.randint(10, 100, 5).tolist()
        }]
    }
    pie_config = {
        "title": "Distribution par segment",
        "donut": True
    }
    
    # Préparation des visualisations
    visualizations = [
        {
            "type": "BAR",
            "data": bar_data,
            "config": bar_config,
            "title": "Analyse par catégorie",
            "description": "Ce graphique présente la répartition des valeurs par catégorie."
        },
        {
            "type": "LINE",
            "data": line_data,
            "config": line_config,
            "title": "Tendance mensuelle",
            "description": "Ce graphique montre l'évolution des valeurs au cours du temps."
        },
        {
            "type": "PIE",
            "data": pie_data,
            "config": pie_config,
            "title": "Répartition par segments",
            "description": "Ce graphique montre la distribution proportionnelle des segments."
        }
    ]
    
    # Ajout des visualisations au rapport
    report_data["has_visualizations"] = True
    try:
        # Utilisation du générateur de graphiques de XPLIA si possible
        report_data["visualizations"] = html_generator._process_visualizations(visualizations)
    except Exception as e:
        print(f"Erreur lors du traitement des visualisations: {e}")
        print("Génération d'un HTML alternatif pour les visualisations...")
        
        # Alternative en cas d'erreur: génération directe de HTML pour les visualisations
        report_data["visualizations"] = []
        for viz in visualizations:
            viz_html = f"""
            <div style="text-align:center">
                <p>[Visualisation simulée: {viz['type']} - {viz['title']}]</p>
                <div style="border:1px solid #ccc; padding:20px; margin:10px 0; border-radius:5px;">
                    <p>{viz['description']}</p>
                </div>
            </div>
            """
            report_data["visualizations"].append({
                "title": viz.get("title", "Sans titre"),
                "description": viz.get("description", ""),
                "html": viz_html
            })
    
    # Génération du rapport HTML
    try:
        html_content = html_generator.generate(report_data)
        
        # Sauvegarde du rapport
        output_path = os.path.join(output_dir, "direct_html_report_demo.html")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Rapport HTML généré avec succès: {output_path}")
        print("Ce rapport contient 3 visualisations intégrées au formateur HTML XPLIA.")
        print("Ouvrez le fichier dans un navigateur pour visualiser le rapport.")
        
    except Exception as e:
        print(f"Erreur lors de la génération du rapport HTML: {e}")
        print("Veuillez vérifier que toutes les dépendances sont correctement installées.")

if __name__ == "__main__":
    main()
