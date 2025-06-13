#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Démonstration de l'évaluation experte
=====================================

Ce script démontre l'utilisation de l'intégrateur d'évaluation experte
pour évaluer la qualité des explications et la fiabilité des modèles.
"""

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional

# Configuration du chemin d'accès pour l'importation de xplia
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from xplia.compliance.expert_review.integration import ExpertEvaluationIntegrator, quick_evaluate
    from xplia.explainers import lime_explainer
    from xplia.core.model_adapters.sklearn_adapter import SklearnAdapter
    from xplia.formatters.html_formatter import HTMLReportGenerator
    from xplia.formatters.pdf_formatter import PDFReportGenerator
    from xplia.visualizations import ChartGenerator
    
    # Vérification des dépendances optionnelles
    try:
        import sklearn
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.datasets import load_breast_cancer, load_iris
        SKLEARN_AVAILABLE = True
    except ImportError:
        SKLEARN_AVAILABLE = False
    
    try:
        import plotly
        PLOTLY_AVAILABLE = True
    except ImportError:
        PLOTLY_AVAILABLE = False
    
    try:
        import joblib
        JOBLIB_AVAILABLE = True
    except ImportError:
        JOBLIB_AVAILABLE = False
    
    XPLIA_AVAILABLE = True
except ImportError as e:
    print(f"Erreur d'importation: {e}")
    XPLIA_AVAILABLE = False


# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies() -> List[str]:
    """
    Vérifie les dépendances requises pour la démonstration.
    
    Returns:
        Liste des dépendances manquantes
    """
    missing_deps = []
    
    if not XPLIA_AVAILABLE:
        missing_deps.append("xplia")
    
    if not SKLEARN_AVAILABLE:
        missing_deps.append("scikit-learn")
    
    if not PLOTLY_AVAILABLE:
        missing_deps.append("plotly")
    
    if not JOBLIB_AVAILABLE:
        missing_deps.append("joblib")
    
    return missing_deps


def load_dataset() -> Tuple:
    """
    Charge un jeu de données pour la démonstration.
    
    Returns:
        Tuple contenant les données d'entraînement, de test et les noms des features et des classes
    """
    # Chargement du jeu de données
    dataset = load_breast_cancer()
    X, y = dataset.data, dataset.target
    
    # Séparation en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Extraction des noms des features et des classes
    feature_names = dataset.feature_names
    target_names = dataset.target_names
    
    logger.info(f"Jeu de données chargé: {dataset.DESCR.splitlines()[0]}")
    logger.info(f"Nombre d'instances: {X.shape[0]}")
    logger.info(f"Nombre de features: {X.shape[1]}")
    logger.info(f"Nombre de classes: {len(target_names)}")
    
    return X_train, X_test, y_train, y_test, feature_names, target_names


def train_model(X_train, y_train):
    """
    Entraîne un modèle pour la démonstration.
    
    Args:
        X_train: Données d'entraînement
        y_train: Étiquettes d'entraînement
        
    Returns:
        Tuple contenant le modèle entraîné et l'adaptateur de modèle
    """
    # Création et entraînement du modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Création de l'adaptateur de modèle
    model_adapter = SklearnAdapter(model)
    
    # Évaluation du modèle
    train_score = model.score(X_train, y_train)
    logger.info(f"Score d'entraînement: {train_score:.4f}")
    
    return model, model_adapter


def run_expert_evaluation_demo():
    """
    Exécute la démonstration d'évaluation experte.
    """
    try:
        logger.info("Démarrage de la démonstration d'évaluation experte")
        
        # Chargement des données
        logger.info("Chargement des données...")
        X_train, X_test, y_train, y_test, feature_names, target_names = load_dataset()
        
        # Entraînement du modèle
        logger.info("Entraînement du modèle...")
        model, model_adapter = train_model(X_train, y_train)
        
        # Sélection d'une instance à expliquer
        instance_idx = 0
        instance = X_test[instance_idx:instance_idx+1]
        true_label = y_test[instance_idx]
        predicted_label = model.predict(instance)[0]
        
        logger.info(f"Instance sélectionnée: {instance_idx}")
        logger.info(f"Vraie classe: {target_names[true_label]}")
        logger.info(f"Classe prédite: {target_names[predicted_label]}")
        
        # Création de l'explainer
        logger.info("Création de l'explainer...")
        explainer = lime_explainer.LimeExplainer()
        
        # Méthode 1: Utilisation de l'intégrateur d'évaluation experte
        logger.info("Méthode 1: Utilisation de l'intégrateur d'évaluation experte")
        integrator = ExpertEvaluationIntegrator()
        
        # Évaluation complète du pipeline d'explication
        logger.info("Évaluation complète du pipeline d'explication...")
        results = integrator.evaluate_explanation_pipeline(
            explainer=explainer,
            model_adapter=model_adapter,
            instance=instance,
            background_data=X_train,
            sensitive_features=None
        )
        
        # Génération d'un rapport complet
        logger.info("Génération d'un rapport complet...")
        report = integrator.generate_comprehensive_report(
            evaluation_results=results,
            include_visualizations=True
        )
        
        # Affichage des scores
        print("\nRésultats de l'évaluation experte (Méthode 1):")
        print(f"Score global: {report['summary']['overall_score']:.2f}/10")
        print(f"Score de qualité de l'explication: {report['summary']['explanation_quality_score']:.2f}/10")
        print(f"Score de confiance: {report['summary']['trust_score']:.2f}/10")
        
        # Affichage des recommandations
        print("\nRecommandations:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"  {i}. {rec}")
        
        # Méthode 2: Utilisation de la fonction d'évaluation rapide
        logger.info("Méthode 2: Utilisation de la fonction d'évaluation rapide")
        quick_report = quick_evaluate(
            explainer=explainer,
            model_adapter=model_adapter,
            instance=instance,
            background_data=X_train
        )
        
        # Affichage des scores
        print("\nRésultats de l'évaluation experte (Méthode 2):")
        print(f"Score global: {quick_report['summary']['overall_score']:.2f}/10")
        
        # Génération d'un rapport HTML
        logger.info("Génération d'un rapport HTML...")
        html_generator = HTMLReportGenerator()
        html_report = html_generator.generate(
            explanation=results["explanation"],
            model_adapter=model_adapter,
            instance=instance,
            additional_sections={
                "Évaluation experte": {
                    "content": results["trust_evaluation"],
                    "type": "expert_evaluation"
                }
            }
        )
        
        # Sauvegarde du rapport HTML
        html_report_path = "expert_evaluation_demo_report.html"
        with open(html_report_path, "w", encoding="utf-8") as f:
            f.write(html_report)
        
        logger.info(f"Rapport HTML généré et sauvegardé dans {html_report_path}")
        
        # Ouverture du rapport dans le navigateur
        import webbrowser
        webbrowser.open(html_report_path)
        
        logger.info("Démonstration d'évaluation experte terminée avec succès")
        
        return {
            "results": results,
            "report": report,
            "quick_report": quick_report,
            "html_report_path": html_report_path
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de la démonstration d'évaluation experte: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    """
    Point d'entrée principal pour la démonstration d'évaluation experte.
    """
    try:
        # Vérification des dépendances
        missing_deps = check_dependencies()
        if missing_deps:
            print("\nAvertissement: Certaines dépendances sont manquantes:")
            for dep in missing_deps:
                print(f"  - {dep}")
            print("\nCertaines fonctionnalités peuvent être limitées.")
            print("Pour installer les dépendances manquantes, exécutez:")
            print("pip install -r requirements.txt\n")
        
        # Exécution de la démonstration
        print("\nDémarrage de la démonstration d'évaluation experte...\n")
        results = run_expert_evaluation_demo()
        
        if "error" in results:
            print(f"\nErreur lors de la démonstration: {results['error']}")
        else:
            print("\nDémonstration terminée avec succès!")
            print(f"Rapport HTML généré: {results['html_report_path']}")
    
    except KeyboardInterrupt:
        print("\nDémonstration interrompue par l'utilisateur.")
    
    except Exception as e:
        print(f"\nErreur inattendue: {e}")
        import traceback
        traceback.print_exc()
