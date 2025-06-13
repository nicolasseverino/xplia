"""
Démonstration du Pipeline d'Évaluation de Confiance
================================================

Ce script démontre l'utilisation complète du pipeline d'évaluation de confiance
pour les explications générées par XPLIA, incluant:
1. Quantification d'incertitude
2. Détection de fairwashing
3. Génération de rapports de confiance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

from xplia.explainers.shap import ShapExplainer
from xplia.explainers.lime import LimeExplainer
from xplia.explainers.trust.uncertainty import UncertaintyQuantifier
from xplia.explainers.trust.fairwashing import FairwashingDetector
from xplia.explainers.trust.confidence_report import ConfidenceReport
from xplia.core.model_adapters import SklearnAdapter
from xplia.formatters.html import HTMLReportGenerator


def main():
    """
    Fonction principale démontrant le pipeline d'évaluation de confiance.
    """
    print("Chargement des données...")
    # Charger les données
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    # Définir quelques features comme "sensibles" pour la démonstration
    sensitive_features = ['mean radius', 'mean texture']
    
    # Diviser les données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("Entraînement du modèle...")
    # Entraîner un modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Créer un adaptateur de modèle
    model_adapter = SklearnAdapter(model, feature_names=X.columns.tolist())
    
    print("Génération des explications...")
    # Créer des explainers
    shap_explainer = ShapExplainer()
    lime_explainer = LimeExplainer()
    
    # Sélectionner un exemple à expliquer
    instance_idx = 0
    instance = X_test.iloc[instance_idx:instance_idx+1]
    
    # Générer des explications
    shap_explanation = shap_explainer.explain(
        model_adapter, instance, background_data=X_train.sample(100)
    )
    lime_explanation = lime_explainer.explain(
        model_adapter, instance
    )
    
    print("Évaluation de la confiance des explications...")
    # Créer les évaluateurs de confiance
    uncertainty_quantifier = UncertaintyQuantifier(n_bootstrap_samples=50)
    fairwashing_detector = FairwashingDetector(sensitive_features=sensitive_features)
    confidence_reporter = ConfidenceReport()
    
    # Évaluer l'incertitude des explications
    shap_uncertainty = uncertainty_quantifier.quantify_uncertainty(
        shap_explanation, shap_explainer, X_test.sample(20)
    )
    lime_uncertainty = uncertainty_quantifier.quantify_uncertainty(
        lime_explanation, lime_explainer, X_test.sample(20)
    )
    
    # Détecter le fairwashing potentiel
    shap_fairwashing = fairwashing_detector.detect_fairwashing(
        shap_explanation, X=X_test.sample(20)
    )
    lime_fairwashing = fairwashing_detector.detect_fairwashing(
        lime_explanation, X=X_test.sample(20)
    )
    
    # Générer des rapports de confiance
    shap_confidence = confidence_reporter.generate_report(
        shap_explanation, shap_uncertainty, shap_fairwashing
    )
    lime_confidence = confidence_reporter.generate_report(
        lime_explanation, lime_uncertainty, lime_fairwashing
    )
    
    # Appliquer les rapports aux explications
    shap_explanation = confidence_reporter.apply_to_explanation(
        shap_explanation, shap_confidence
    )
    lime_explanation = confidence_reporter.apply_to_explanation(
        lime_explanation, lime_confidence
    )
    
    print("Génération des rapports HTML...")
    # Générer des rapports HTML
    html_generator = HTMLReportGenerator()
    
    shap_report = html_generator.generate_report(
        explanation=shap_explanation,
        model_adapter=model_adapter,
        title="Rapport SHAP avec Évaluation de Confiance",
        include_confidence=True  # Inclure les métriques de confiance
    )
    
    lime_report = html_generator.generate_report(
        explanation=lime_explanation,
        model_adapter=model_adapter,
        title="Rapport LIME avec Évaluation de Confiance",
        include_confidence=True  # Inclure les métriques de confiance
    )
    
    # Sauvegarder les rapports
    with open("shap_confidence_report.html", "w", encoding="utf-8") as f:
        f.write(shap_report)
    
    with open("lime_confidence_report.html", "w", encoding="utf-8") as f:
        f.write(lime_report)
    
    print("Affichage des résultats de confiance...")
    # Afficher les résultats de confiance
    print("\nRésultats pour l'explication SHAP:")
    print(f"Score de confiance global: {shap_confidence['trust_score']['global_trust']:.2f}")
    print(f"Niveau de confiance: {shap_confidence['trust_score']['trust_level']}")
    print(f"Incertitude: {shap_confidence['detailed_metrics']['uncertainty']['global_uncertainty']:.2f}")
    print(f"Fairwashing: {shap_confidence['detailed_metrics']['fairwashing']['fairwashing_score']:.2f}")
    print(f"Résumé: {shap_confidence['summary']}")
    print("\nRecommandations:")
    for rec in shap_confidence['recommendations']:
        print(f"- {rec}")
    
    print("\nRésultats pour l'explication LIME:")
    print(f"Score de confiance global: {lime_confidence['trust_score']['global_trust']:.2f}")
    print(f"Niveau de confiance: {lime_confidence['trust_score']['trust_level']}")
    print(f"Incertitude: {lime_confidence['detailed_metrics']['uncertainty']['global_uncertainty']:.2f}")
    print(f"Fairwashing: {lime_confidence['detailed_metrics']['fairwashing']['fairwashing_score']:.2f}")
    print(f"Résumé: {lime_confidence['summary']}")
    print("\nRecommandations:")
    for rec in lime_confidence['recommendations']:
        print(f"- {rec}")
    
    print("\nRapports HTML générés:")
    print("- shap_confidence_report.html")
    print("- lime_confidence_report.html")
    
    # Visualiser la comparaison des scores de confiance
    plot_confidence_comparison(shap_confidence, lime_confidence)


def plot_confidence_comparison(shap_confidence, lime_confidence):
    """
    Visualise une comparaison des scores de confiance entre deux explications.
    
    Args:
        shap_confidence: Rapport de confiance pour SHAP
        lime_confidence: Rapport de confiance pour LIME
    """
    # Extraire les scores
    metrics = ['global_trust', 'uncertainty_trust', 'fairwashing_trust', 
               'consistency_trust', 'robustness_trust']
    
    labels = ['Score global', 'Certitude', 'Anti-fairwashing', 
              'Cohérence', 'Robustesse']
    
    shap_scores = [shap_confidence['trust_score'][m] for m in metrics]
    lime_scores = [lime_confidence['trust_score'][m] for m in metrics]
    
    # Créer le graphique
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, shap_scores, width, label='SHAP')
    rects2 = ax.bar(x + width/2, lime_scores, width, label='LIME')
    
    # Ajouter les labels
    ax.set_ylabel('Score de confiance')
    ax.set_title('Comparaison des scores de confiance entre explainers')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Ajouter les valeurs sur les barres
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig('confidence_comparison.png')
    plt.close()
    
    print("Graphique de comparaison sauvegardé: confidence_comparison.png")


if __name__ == "__main__":
    main()
