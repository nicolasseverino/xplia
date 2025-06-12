"""
Démonstration de l'intégration des visualisations dans un rapport PDF XPLIA.

Ce script montre comment intégrer différents types de graphiques dans un rapport PDF
en utilisant le formateur PDF et le générateur de graphiques.
"""

import os
import sys
import datetime
import random
import importlib.util

# Vérification des dépendances requises
def check_dependency(package_name):
    """Vérifie si un package est installé et retourne True/False."""
    return importlib.util.find_spec(package_name) is not None

# Liste des dépendances requises
required_dependencies = {
    'reportlab': "Pour la génération de documents PDF",
    'matplotlib': "Pour la création de visualisations statiques",
    'pillow': "Pour le traitement d'images",
    'joblib': "Pour la sérialisation des objets"  # Souvent utilisé dans XPLIA
}

# Vérification des dépendances
missing_dependencies = []
for dep, desc in required_dependencies.items():
    if not check_dependency(dep):
        missing_dependencies.append((dep, desc))

# Afficher un avertissement pour les dépendances manquantes
if missing_dependencies:
    print("⚠️ ATTENTION ⚠️")
    print("Les dépendances suivantes sont manquantes et pourraient être requises:")
    for dep, desc in missing_dependencies:
        print(f" - {dep}: {desc}")
    
    # Si reportlab est manquant, c'est critique pour ce démonstrateur
    if any(dep == 'reportlab' for dep, _ in missing_dependencies):
        print("\n⛔ ERREUR CRITIQUE: 'reportlab' est requis pour ce démonstrateur PDF et ne peut pas être contourné.")
        print("Veuillez installer les dépendances requises via: pip install -r requirements.txt")
        print("Ou directement: pip install reportlab pillow matplotlib")        
        print("\nTentative de continuer malgré tout, mais des erreurs sont probables...")        
    else:
        print("\nInstallation recommandée: pip install -r requirements.txt")
        print("Tentative de continuer avec des fonctionnalités limitées...\n")    

# Ajout du répertoire parent au chemin de recherche Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from xplia.compliance.formatters.pdf_formatter import PDFReportGenerator
    from xplia.visualizations.chart_generator import ChartGenerator, ChartType
except ImportError as e:
    print(f"\nErreur lors de l'importation des modules XPLIA: {e}")
    print("Vérifiez que vous êtes dans le bon répertoire ou que XPLIA est correctement installé.")
    print("Si l'erreur concerne des dépendances manquantes, installez-les via pip install -r requirements.txt")
    print("\nSi vous rencontrez l'erreur 'No module named xplia', essayez:")
    print("- Vérifiez que le répertoire du projet est bien dans PYTHONPATH")
    print("- Exécutez le script depuis le répertoire racine du projet")
    sys.exit(1)

def generate_demo_report():
    """
    Génère un rapport PDF de démonstration avec différentes visualisations.
    """
    # Création d'un générateur de rapport PDF
    generator = PDFReportGenerator(language='fr')
    
    # Création des données pour les visualisations
    visualizations = []
    
    # 1. Graphique à barres - Scores de conformité par règlement
    regulations = ['GDPR', 'PCI DSS', 'HIPAA', 'SOC 2', 'ISO 27001']
    scores = [random.randint(65, 95) for _ in range(len(regulations))]
    
    visualizations.append({
        'type': ChartType.BAR,
        'data': {
            'labels': regulations,
            'datasets': [{
                'label': 'Score de conformité (%)',
                'data': scores
            }]
        },
        'config': {
            'colors': ['#4285F4', '#34A853', '#FBBC05', '#EA4335', '#8F00FF'],
        },
        'title': 'Scores de conformité par règlement',
        'description': 'Ce graphique présente les scores de conformité pour chaque règlement évalué.'
    })
    
    # 2. Graphique camembert - Répartition des risques
    risk_categories = ['Élevé', 'Moyen', 'Faible']
    risk_counts = [random.randint(3, 8), random.randint(10, 20), random.randint(25, 40)]
    
    visualizations.append({
        'type': ChartType.PIE,
        'data': {
            'labels': risk_categories,
            'datasets': [{
                'data': risk_counts
            }]
        },
        'config': {
            'colors': ['#FF5733', '#FFC300', '#36A2EB'],
        },
        'title': 'Répartition des risques identifiés',
        'description': 'Ce graphique montre la répartition des risques par niveau de sévérité.'
    })
    
    # 3. Graphique linéaire - Évolution des scores sur 6 mois
    months = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin']
    
    # Génération de données pour plusieurs règlements
    datasets = []
    for i, reg in enumerate(regulations[:3]):  # Limité à 3 pour la lisibilité
        # Simuler une amélioration progressive avec quelques fluctuations
        start_score = random.randint(50, 70)
        scores = [start_score]
        for _ in range(5):
            change = random.randint(-5, 10)  # Tendance à l'amélioration
            next_score = min(100, max(0, scores[-1] + change))  # Entre 0 et 100
            scores.append(next_score)
            
        datasets.append({
            'label': reg,
            'data': scores
        })
    
    visualizations.append({
        'type': ChartType.LINE,
        'data': {
            'labels': months,
            'datasets': datasets
        },
        'config': {
            'tension': 0.2,  # Lissage des courbes
        },
        'title': 'Évolution des scores sur 6 mois',
        'description': 'Ce graphique montre l\'évolution des scores de conformité pour les principaux règlements sur les 6 derniers mois.'
    })
    
    # 4. Carte de chaleur - Matrice de risques
    impact_levels = ['Critique', 'Élevé', 'Moyen', 'Faible', 'Négligeable']
    probability_levels = ['Très probable', 'Probable', 'Possible', 'Peu probable', 'Rare']
    
    # Valeurs de la matrice (simulées)
    # Plus la valeur est élevée, plus le risque est important
    heat_data = []
    for _ in range(len(impact_levels)):
        row = []
        for _ in range(len(probability_levels)):
            # Les valeurs plus élevées tendent à se trouver en haut à gauche de la matrice
            row.append(random.randint(1, 10))
        heat_data.append(row)
    
    visualizations.append({
        'type': ChartType.HEATMAP,
        'data': {
            'data': heat_data,
            'x_labels': probability_levels,
            'y_labels': impact_levels
        },
        'config': {
            'colorscale': 'YlOrRd',  # Échelle de couleur jaune-orange-rouge
        },
        'title': 'Matrice de risques',
        'description': 'Cette matrice présente l\'évaluation des risques en fonction de leur impact et de leur probabilité.'
    })
    
    # 5. Graphique radar - Couverture des contrôles de sécurité
    security_domains = [
        'Gestion des accès', 
        'Sécurité réseau', 
        'Sécurité physique',
        'Réponse aux incidents', 
        'Conformité réglementaire',
        'Formation et sensibilisation'
    ]
    
    # Génération des scores pour chaque domaine (entre 0 et 100%)
    current_scores = [random.randint(50, 95) for _ in range(len(security_domains))]
    target_scores = [min(95, score + random.randint(5, 20)) for score in current_scores]
    
    visualizations.append({
        'type': ChartType.RADAR,
        'data': {
            'labels': security_domains,
            'datasets': [
                {
                    'label': 'État actuel',
                    'data': current_scores
                },
                {
                    'label': 'Objectifs',
                    'data': target_scores
                }
            ]
        },
        'title': 'Couverture des contrôles de sécurité',
        'description': 'Ce graphique radar montre le niveau actuel de maturité dans chaque domaine de sécurité par rapport aux objectifs.'
    })
    
    # Création du contenu du rapport
    today = datetime.datetime.now()
    
    content = {
        'title': 'Rapport de conformité et analyse des risques',
        'organization': 'Demo Corporation',
        'date_created': today.isoformat(),
        'authors': ['Équipe de conformité XPLIA'],
        'compliance_score': {
            'score': 78,
            'status': 'Conforme avec observations',
            'details': {reg: score for reg, score in zip(regulations, scores)}
        },
        'audit_trail': [
            {
                'action': 'Évaluation initiale',
                'timestamp': (today - datetime.timedelta(days=30)).isoformat(),
                'user': 'Auditeur A'
            },
            {
                'action': 'Révision des contrôles',
                'timestamp': (today - datetime.timedelta(days=15)).isoformat(),
                'user': 'Auditeur B'
            },
            {
                'action': 'Génération du rapport final',
                'timestamp': today.isoformat(),
                'user': 'Responsable conformité'
            }
        ],
        'score_details': {reg: score for reg, score in zip(regulations, scores)},
        'issues': [
            {
                'title': 'Gestion des accès à réviser',
                'severity': 'Moyenne',
                'description': 'Les processus de révision périodique des accès ne sont pas systématiquement appliqués dans tous les départements.',
                'remediation': 'Mettre en place un processus automatisé de révision trimestrielle des accès avec validation par les responsables de département.'
            },
            {
                'title': 'Documentation insuffisante',
                'severity': 'Faible',
                'description': 'Certaines procédures opérationnelles ne sont pas suffisamment documentées.',
                'remediation': 'Compléter la documentation manquante et mettre en place un processus de révision documentaire annuel.'
            }
        ],
        'recommendations': [
            'Renforcer le programme de sensibilisation à la sécurité pour tous les employés',
            'Mettre en place un système de surveillance continue de la conformité',
            'Améliorer la documentation des procédures d\'intervention en cas d\'incident'
        ],
        # Ajout des visualisations au rapport
        'visualizations': visualizations
    }
    
    # Génération du rapport PDF
    pdf_bytes = generator.generate(content)
    
    # Sauvegarde du rapport
    output_path = os.path.join(os.path.dirname(__file__), 'demo_pdf_report.pdf')
    with open(output_path, 'wb') as f:
        f.write(pdf_bytes)
    
    print(f"Rapport PDF généré avec succès : {output_path}")
    return output_path

if __name__ == "__main__":
    try:
        report_path = generate_demo_report()
        print(f"Le rapport a été généré avec succès à l'emplacement : {report_path}")
        
        # Essayer d'ouvrir le PDF automatiquement (fonctionne sur la plupart des systèmes)
        if sys.platform.startswith('darwin'):  # macOS
            os.system(f'open "{report_path}"')
        elif sys.platform.startswith('win'):  # Windows
            os.system(f'start "" "{report_path}"')
        elif sys.platform.startswith('linux'):  # Linux
            os.system(f'xdg-open "{report_path}"')
            
    except Exception as e:
        print(f"Erreur lors de la génération du rapport: {e}")
        import traceback
        traceback.print_exc()
