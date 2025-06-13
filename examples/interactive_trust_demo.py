#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Démonstration interactive des métriques de confiance XPLIA
=========================================================

Ce script fournit une interface interactive pour explorer et visualiser
les métriques de confiance générées par XPLIA pour différents modèles
et jeux de données.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ajout du répertoire parent au path pour l'importation de xplia
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import xplia
    from xplia.core import registry
    from xplia.explainers import lime_explainer, shap_explainer
    from xplia.trust import uncertainty, fairwashing, confidence
    from xplia.compliance.expert_review.trust_expert_evaluator import TrustExpertEvaluator
    from xplia.visualizations import ChartGenerator
except ImportError as e:
    logger.error(f"Erreur d'importation: {e}")
    logger.error("Assurez-vous que le package xplia est correctement installé.")
    sys.exit(1)

# Vérification des dépendances optionnelles
try:
    import sklearn
    from sklearn.datasets import load_breast_cancer, load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn n'est pas disponible. Certaines fonctionnalités seront limitées.")
    SKLEARN_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    logger.warning("plotly n'est pas disponible. Les visualisations interactives seront limitées.")
    PLOTLY_AVAILABLE = False

try:
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output
    DASH_AVAILABLE = True
except ImportError:
    logger.warning("dash n'est pas disponible. L'interface interactive sera limitée.")
    DASH_AVAILABLE = False


def load_dataset(dataset_name: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Charge un jeu de données pour la démonstration.
    
    Args:
        dataset_name: Nom du jeu de données à charger ('cancer', 'diabetes', etc.)
        
    Returns:
        Tuple contenant les features (X), la cible (y) et les noms des features
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn est requis pour charger les jeux de données")
    
    if dataset_name == "cancer":
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")
        feature_names = data.feature_names
        logger.info(f"Jeu de données cancer chargé: {X.shape[0]} échantillons, {X.shape[1]} features")
    
    elif dataset_name == "diabetes":
        data = load_diabetes()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")
        feature_names = data.feature_names
        logger.info(f"Jeu de données diabetes chargé: {X.shape[0]} échantillons, {X.shape[1]} features")
    
    else:
        raise ValueError(f"Jeu de données '{dataset_name}' non reconnu")
    
    return X, y, feature_names


def train_model(X: pd.DataFrame, y: pd.Series, model_type: str) -> Tuple[Any, pd.DataFrame, pd.Series]:
    """
    Entraîne un modèle sur les données fournies.
    
    Args:
        X: Features d'entraînement
        y: Cible d'entraînement
        model_type: Type de modèle à entraîner ('rf_classifier', 'rf_regressor', etc.)
        
    Returns:
        Tuple contenant le modèle entraîné, les features de test et la cible de test
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn est requis pour entraîner les modèles")
    
    # Division en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Entraînement du modèle
    if model_type == "rf_classifier":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        logger.info(f"Modèle RandomForestClassifier entraîné avec une précision de {accuracy:.4f}")
    
    elif model_type == "rf_regressor":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        r2 = model.score(X_test, y_test)
        logger.info(f"Modèle RandomForestRegressor entraîné avec un R² de {r2:.4f}")
    
    else:
        raise ValueError(f"Type de modèle '{model_type}' non reconnu")
    
    return model, X_test, y_test


def get_model_adapter(model: Any, dataset_name: str) -> Any:
    """
    Obtient l'adaptateur de modèle XPLIA approprié pour le modèle fourni.
    
    Args:
        model: Modèle à adapter
        dataset_name: Nom du jeu de données utilisé
        
    Returns:
        Adaptateur de modèle XPLIA
    """
    try:
        # Récupération de l'adaptateur de modèle approprié
        if isinstance(model, RandomForestClassifier):
            from xplia.core.model_adapters.sklearn_adapter import SklearnClassifierAdapter
            adapter = SklearnClassifierAdapter(model)
            logger.info("Adaptateur SklearnClassifierAdapter créé")
            return adapter
        
        elif isinstance(model, RandomForestRegressor):
            from xplia.core.model_adapters.sklearn_adapter import SklearnRegressorAdapter
            adapter = SklearnRegressorAdapter(model)
            logger.info("Adaptateur SklearnRegressorAdapter créé")
            return adapter
        
        else:
            raise ValueError(f"Type de modèle non pris en charge: {type(model)}")
    
    except Exception as e:
        logger.error(f"Erreur lors de la création de l'adaptateur de modèle: {e}")
        return None


def generate_explanation(model_adapter: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                        instance_idx: int, explainer_type: str) -> Dict[str, Any]:
    """
    Génère une explication pour une instance spécifique.
    
    Args:
        model_adapter: Adaptateur de modèle XPLIA
        X_test: Données de test
        y_test: Cibles de test
        instance_idx: Index de l'instance à expliquer
        explainer_type: Type d'explainer à utiliser ('lime', 'shap', etc.)
        
    Returns:
        Dictionnaire contenant l'explication générée
    """
    try:
        # Sélection de l'instance à expliquer
        instance = X_test.iloc[instance_idx]
        true_label = y_test.iloc[instance_idx]
        
        # Création de l'explainer approprié
        if explainer_type == "lime":
            explainer = lime_explainer.LimeExplainer()
            logger.info("Explainer LIME créé")
        elif explainer_type == "shap":
            explainer = shap_explainer.ShapExplainer()
            logger.info("Explainer SHAP créé")
        else:
            raise ValueError(f"Type d'explainer non reconnu: {explainer_type}")
        
        # Génération de l'explication
        explanation = explainer.explain(
            model_adapter=model_adapter,
            instance=instance,
            background_data=X_test.sample(min(100, len(X_test))),
            num_features=10
        )
        
        # Ajout d'informations supplémentaires
        explanation["true_label"] = true_label
        explanation["instance_idx"] = instance_idx
        explanation["explainer_type"] = explainer_type
        
        logger.info(f"Explication générée avec succès pour l'instance {instance_idx}")
        return explanation
    
    except Exception as e:
        logger.error(f"Erreur lors de la génération de l'explication: {e}")
        return {"error": str(e)}


def generate_uncertainty_metrics(model_adapter: Any, explanation: Dict[str, Any], 
                               X_test: pd.DataFrame) -> Dict[str, Any]:
    """
    Génère des métriques d'incertitude pour une explication.
    
    Args:
        model_adapter: Adaptateur de modèle XPLIA
        explanation: Explication générée
        X_test: Données de test
        
    Returns:
        Dictionnaire contenant les métriques d'incertitude
    """
    try:
        # Création de l'estimateur d'incertitude
        uncertainty_estimator = uncertainty.UncertaintyEstimator()
        
        # Calcul des métriques d'incertitude
        instance_idx = explanation.get("instance_idx", 0)
        instance = X_test.iloc[instance_idx]
        
        uncertainty_metrics = uncertainty_estimator.estimate(
            model_adapter=model_adapter,
            instance=instance,
            explanation=explanation,
            background_data=X_test.sample(min(100, len(X_test)))
        )
        
        logger.info(f"Métriques d'incertitude générées pour l'instance {instance_idx}")
        return uncertainty_metrics
    
    except Exception as e:
        logger.error(f"Erreur lors de la génération des métriques d'incertitude: {e}")
        return {"error": str(e)}


def generate_fairwashing_audit(model_adapter: Any, explanation: Dict[str, Any], 
                             X_test: pd.DataFrame, sensitive_features: List[str] = None) -> Dict[str, Any]:
    """
    Génère un audit de fairwashing pour une explication.
    
    Args:
        model_adapter: Adaptateur de modèle XPLIA
        explanation: Explication générée
        X_test: Données de test
        sensitive_features: Liste des features sensibles à surveiller
        
    Returns:
        Dictionnaire contenant l'audit de fairwashing
    """
    try:
        # Création de l'auditeur de fairwashing
        fairwashing_auditor = fairwashing.FairwashingAuditor()
        
        # Si aucune feature sensible n'est spécifiée, on en sélectionne quelques-unes arbitrairement
        if not sensitive_features and len(X_test.columns) > 0:
            sensitive_features = list(X_test.columns[:2])  # Premières colonnes comme exemple
        
        # Calcul de l'audit de fairwashing
        instance_idx = explanation.get("instance_idx", 0)
        instance = X_test.iloc[instance_idx]
        
        fairwashing_audit = fairwashing_auditor.audit(
            model_adapter=model_adapter,
            instance=instance,
            explanation=explanation,
            background_data=X_test.sample(min(100, len(X_test))),
            sensitive_features=sensitive_features
        )
        
        logger.info(f"Audit de fairwashing généré pour l'instance {instance_idx}")
        return fairwashing_audit
    
    except Exception as e:
        logger.error(f"Erreur lors de la génération de l'audit de fairwashing: {e}")
        return {"error": str(e)}


def generate_confidence_report(explanation: Dict[str, Any], uncertainty_metrics: Dict[str, Any], 
                             fairwashing_audit: Dict[str, Any]) -> Dict[str, Any]:
    """
    Génère un rapport de confiance complet.
    
    Args:
        explanation: Explication générée
        uncertainty_metrics: Métriques d'incertitude
        fairwashing_audit: Audit de fairwashing
        
    Returns:
        Dictionnaire contenant le rapport de confiance
    """
    try:
        # Création du générateur de rapport de confiance
        confidence_reporter = confidence.ConfidenceReporter()
        
        # Génération du rapport de confiance
        confidence_report = confidence_reporter.generate_report(
            explanation=explanation,
            uncertainty_metrics=uncertainty_metrics,
            fairwashing_audit=fairwashing_audit
        )
        
        logger.info("Rapport de confiance généré avec succès")
        return confidence_report
    
    except Exception as e:
        logger.error(f"Erreur lors de la génération du rapport de confiance: {e}")
        return {"error": str(e)}


def evaluate_trust_metrics(explanation: Dict[str, Any], uncertainty_metrics: Dict[str, Any], 
                         fairwashing_audit: Dict[str, Any], confidence_report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Évalue les métriques de confiance avec l'évaluateur expert.
    
    Args:
        explanation: Explication générée
        uncertainty_metrics: Métriques d'incertitude
        fairwashing_audit: Audit de fairwashing
        confidence_report: Rapport de confiance
        
    Returns:
        Dictionnaire contenant l'évaluation experte
    """
    try:
        # Création de l'évaluateur expert
        evaluator = TrustExpertEvaluator()
        
        # Évaluation des métriques de confiance
        expert_review = evaluator.evaluate_trust_metrics(
            uncertainty_metrics=uncertainty_metrics,
            fairwashing_audit=fairwashing_audit,
            confidence_report=confidence_report,
            explanation=explanation
        )
        
        logger.info(f"Évaluation experte complétée avec un score global de {expert_review.global_score:.2f}/10")
        return expert_review.to_dict()
    
    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation des métriques de confiance: {e}")
        return {"error": str(e)}


def visualize_explanation(explanation: Dict[str, Any], chart_generator: ChartGenerator = None) -> Dict[str, Any]:
    """
    Génère des visualisations pour une explication.
    
    Args:
        explanation: Explication à visualiser
        chart_generator: Générateur de graphiques à utiliser
        
    Returns:
        Dictionnaire contenant les visualisations générées
    """
    try:
        # Création du générateur de graphiques si nécessaire
        if chart_generator is None:
            chart_generator = ChartGenerator()
        
        # Vérification des attributions de features
        if "feature_attributions" not in explanation:
            return {"error": "Pas d'attributions de features dans l'explication"}
        
        feature_attributions = explanation["feature_attributions"]
        
        # Conversion en format approprié pour la visualisation
        if isinstance(feature_attributions, dict):
            features = list(feature_attributions.keys())
            values = list(feature_attributions.values())
        else:
            return {"error": "Format d'attributions de features non pris en charge"}
        
        # Génération des visualisations
        visualizations = {}
        
        # Graphique à barres des attributions de features
        bar_chart_data = {
            "x": features,
            "y": values,
            "title": "Importance des features",
            "x_label": "Features",
            "y_label": "Importance"
        }
        visualizations["feature_importance_bar"] = chart_generator.bar_chart(**bar_chart_data)
        
        # Graphique radar des attributions de features (si plotly est disponible)
        if PLOTLY_AVAILABLE:
            radar_chart_data = {
                "categories": features,
                "values": [abs(v) for v in values],  # Valeurs absolues pour le radar
                "title": "Importance des features (radar)",
            }
            visualizations["feature_importance_radar"] = chart_generator.radar_chart(**radar_chart_data)
        
        logger.info("Visualisations générées avec succès pour l'explication")
        return visualizations
    
    except Exception as e:
        logger.error(f"Erreur lors de la génération des visualisations: {e}")
        return {"error": str(e)}


def visualize_uncertainty(uncertainty_metrics: Dict[str, Any], chart_generator: ChartGenerator = None) -> Dict[str, Any]:
    """
    Génère des visualisations pour les métriques d'incertitude.
    
    Args:
        uncertainty_metrics: Métriques d'incertitude à visualiser
        chart_generator: Générateur de graphiques à utiliser
        
    Returns:
        Dictionnaire contenant les visualisations générées
    """
    try:
        # Création du générateur de graphiques si nécessaire
        if chart_generator is None:
            chart_generator = ChartGenerator()
        
        # Vérification des métriques d'incertitude
        if "error" in uncertainty_metrics:
            return {"error": uncertainty_metrics["error"]}
        
        # Génération des visualisations
        visualizations = {}
        
        # Graphique de jauge pour l'incertitude globale
        if hasattr(uncertainty_metrics, "global_uncertainty"):
            global_uncertainty = uncertainty_metrics.global_uncertainty
            gauge_data = {
                "value": global_uncertainty,
                "min_value": 0.0,
                "max_value": 1.0,
                "title": "Incertitude globale",
                "thresholds": [0.3, 0.6, 0.9]
            }
            visualizations["global_uncertainty_gauge"] = chart_generator.gauge_chart(**gauge_data)
        
        # Graphique à barres pour les différents types d'incertitude
        if hasattr(uncertainty_metrics, "uncertainty_types") and uncertainty_metrics.uncertainty_types:
            types = list(uncertainty_metrics.uncertainty_types.keys())
            values = list(uncertainty_metrics.uncertainty_types.values())
            bar_data = {
                "x": types,
                "y": values,
                "title": "Types d'incertitude",
                "x_label": "Type",
                "y_label": "Valeur"
            }
            visualizations["uncertainty_types_bar"] = chart_generator.bar_chart(**bar_data)
        
        logger.info("Visualisations générées avec succès pour les métriques d'incertitude")
        return visualizations
    
    except Exception as e:
        logger.error(f"Erreur lors de la génération des visualisations d'incertitude: {e}")
        return {"error": str(e)}


def visualize_fairwashing(fairwashing_audit: Dict[str, Any], chart_generator: ChartGenerator = None) -> Dict[str, Any]:
    """
    Génère des visualisations pour l'audit de fairwashing.
    
    Args:
        fairwashing_audit: Audit de fairwashing à visualiser
        chart_generator: Générateur de graphiques à utiliser
        
    Returns:
        Dictionnaire contenant les visualisations générées
    """
    try:
        # Création du générateur de graphiques si nécessaire
        if chart_generator is None:
            chart_generator = ChartGenerator()
        
        # Vérification de l'audit de fairwashing
        if "error" in fairwashing_audit:
            return {"error": fairwashing_audit["error"]}
        
        # Génération des visualisations
        visualizations = {}
        
        # Graphique de jauge pour le score de fairwashing
        if hasattr(fairwashing_audit, "fairwashing_score"):
            fairwashing_score = fairwashing_audit.fairwashing_score
            gauge_data = {
                "value": fairwashing_score,
                "min_value": 0.0,
                "max_value": 1.0,
                "title": "Score de fairwashing",
                "thresholds": [0.3, 0.6, 0.9]
            }
            visualizations["fairwashing_score_gauge"] = chart_generator.gauge_chart(**gauge_data)
        
        # Graphique à barres pour les différents types de fairwashing détectés
        if hasattr(fairwashing_audit, "detected_types") and fairwashing_audit.detected_types:
            types = list(fairwashing_audit.detected_types.keys())
            values = list(fairwashing_audit.detected_types.values())
            bar_data = {
                "x": types,
                "y": values,
                "title": "Types de fairwashing détectés",
                "x_label": "Type",
                "y_label": "Score"
            }
            visualizations["fairwashing_types_bar"] = chart_generator.bar_chart(**bar_data)
        
        logger.info("Visualisations générées avec succès pour l'audit de fairwashing")
        return visualizations
    
    except Exception as e:
        logger.error(f"Erreur lors de la génération des visualisations de fairwashing: {e}")
        return {"error": str(e)}


def visualize_expert_evaluation(expert_evaluation: Dict[str, Any], chart_generator: ChartGenerator = None) -> Dict[str, Any]:
    """
    Génère des visualisations pour l'évaluation experte.
    
    Args:
        expert_evaluation: Évaluation experte à visualiser
        chart_generator: Générateur de graphiques à utiliser
        
    Returns:
        Dictionnaire contenant les visualisations générées
    """
    try:
        # Création du générateur de graphiques si nécessaire
        if chart_generator is None:
            chart_generator = ChartGenerator()
        
        # Vérification de l'évaluation experte
        if "error" in expert_evaluation:
            return {"error": expert_evaluation["error"]}
        
        # Génération des visualisations
        visualizations = {}
        
        # Graphique de jauge pour le score global
        if "global_score" in expert_evaluation:
            global_score = expert_evaluation["global_score"]
            gauge_data = {
                "value": global_score,
                "min_value": 0.0,
                "max_value": 10.0,
                "title": "Score global de confiance",
                "thresholds": [3.0, 5.0, 7.0, 9.0]
            }
            visualizations["global_score_gauge"] = chart_generator.gauge_chart(**gauge_data)
        
        # Graphique radar pour les scores par catégorie
        if "category_scores" in expert_evaluation:
            category_scores = expert_evaluation["category_scores"]
            categories = list(category_scores.keys())
            values = list(category_scores.values())
            radar_data = {
                "categories": categories,
                "values": values,
                "title": "Scores par catégorie",
                "min_value": 0.0,
                "max_value": 10.0
            }
            visualizations["category_scores_radar"] = chart_generator.radar_chart(**radar_data)
        
        # Graphique à barres pour les scores individuels
        if "scores" in expert_evaluation:
            scores = expert_evaluation["scores"]
            criteria = list(scores.keys())
            values = list(scores.values())
            bar_data = {
                "x": criteria,
                "y": values,
                "title": "Scores par critère",
                "x_label": "Critère",
                "y_label": "Score"
            }
            visualizations["criteria_scores_bar"] = chart_generator.bar_chart(**bar_data)
        
        logger.info("Visualisations générées avec succès pour l'évaluation experte")
        return visualizations
    
    except Exception as e:
        logger.error(f"Erreur lors de la génération des visualisations d'évaluation: {e}")
        return {"error": str(e)}


def generate_html_report(
    explanation: Dict[str, Any],
    uncertainty_metrics: Dict[str, Any],
    fairwashing_audit: Dict[str, Any],
    confidence_report: Dict[str, Any],
    expert_evaluation: Dict[str, Any],
    visualizations: Dict[str, Dict[str, Any]],
    model_adapter: Any = None,
    instance: Any = None,
    feature_names: List[str] = None,
    target_names: List[str] = None,
    true_label: Any = None,
    predicted_label: Any = None
) -> str:
    """
    Génère un rapport HTML pour la démonstration interactive.
    
    Args:
        explanation: Explication générée
        uncertainty_metrics: Métriques d'incertitude
        fairwashing_audit: Audit de fairwashing
        confidence_report: Rapport de confiance
        expert_evaluation: Évaluation experte
        visualizations: Visualisations générées
        model_adapter: Adaptateur de modèle
        instance: Instance expliquée
        feature_names: Noms des features
        target_names: Noms des classes cibles
        true_label: Vraie classe
        predicted_label: Classe prédite
        
    Returns:
        Rapport HTML généré
    """
    try:
        # Lecture du template HTML
        template_path = os.path.join(os.path.dirname(__file__), "interactive_trust_demo_template.html")
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()
        
        # Extraction des données pour le template
        data = {
            "overall_score": f"{expert_evaluation.get('global_score', 0.0):.2f}",
            "explanation_quality_score": f"{confidence_report.get('explanation_quality_score', 0.0):.2f}",
            "trust_score": f"{expert_evaluation.get('global_score', 0.0):.2f}",
            "instance_id": "0",
            "true_label": target_names[true_label] if target_names and true_label is not None else "N/A",
            "predicted_label": target_names[predicted_label] if target_names and predicted_label is not None else "N/A",
            "global_uncertainty": f"{uncertainty_metrics.get('global_uncertainty', 0.0):.2f}",
            "fairwashing_score": f"{fairwashing_audit.get('fairwashing_score', 0.0):.2f}",
        }
        
        # Extraction des points forts, points faibles et recommandations
        data["explanation_strengths"] = confidence_report.get("strengths", [])
        data["explanation_weaknesses"] = confidence_report.get("weaknesses", [])
        data["explanation_recommendations"] = confidence_report.get("recommendations", [])
        
        data["trust_strengths"] = expert_evaluation.get("strengths", [])
        data["trust_weaknesses"] = expert_evaluation.get("weaknesses", [])
        data["trust_recommendations"] = expert_evaluation.get("recommendations", [])
        
        # Génération des scripts de visualisation
        visualization_scripts = []
        
        # Ajout des scripts pour les visualisations
        for viz_type, viz_data in visualizations.items():
            for viz_name, viz_content in viz_data.items():
                if "error" not in viz_content:
                    visualization_scripts.append(viz_content)
        
        # Remplacement des variables dans le template
        html_report = template
        
        # Remplacement des variables simples
        for key, value in data.items():
            if isinstance(value, str):
                html_report = html_report.replace(f"{{{{{key}}}}}", value)
        
        # Remplacement des listes
        for key, values in data.items():
            if isinstance(values, list):
                list_items = ""
                for value in values:
                    list_items += f"<li>{value}</li>\n"
                html_report = html_report.replace(f"{{{{#each {key}}}}}\n            <li>{{{{this}}}}</li>\n            {{{{/each}}}}", list_items)
        
        # Remplacement des scripts de visualisation
        html_report = html_report.replace("{{visualization_scripts}}", "\n".join(visualization_scripts))
        
        logger.info("Rapport HTML généré avec succès")
        return html_report
    
    except Exception as e:
        logger.error(f"Erreur lors de la génération du rapport HTML: {e}")
        return f"<html><body><h1>Erreur</h1><p>{str(e)}</p></body></html>"


def run_interactive_demo():
    """
    Exécute la démonstration interactive des métriques de confiance.
    """
    try:
        # Configuration du logger pour la démonstration
        setup_logging(level=logging.INFO)
        logger.info("Démarrage de la démonstration interactive des métriques de confiance")
        
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
        
        # Génération de l'explication
        logger.info("Génération de l'explication...")
        explanation = generate_explanation(model_adapter, instance, X_train)
        
        # Calcul des métriques d'incertitude
        logger.info("Calcul des métriques d'incertitude...")
        uncertainty_metrics = estimate_uncertainty(model_adapter, instance, explanation, X_train)
        
        # Audit de fairwashing
        logger.info("Audit de fairwashing...")
        fairwashing_audit = audit_fairwashing(model_adapter, instance, explanation, X_train)
        
        # Génération du rapport de confiance
        logger.info("Génération du rapport de confiance...")
        confidence_report = generate_confidence_report(explanation, uncertainty_metrics, fairwashing_audit)
        
        # Évaluation des métriques de confiance
        logger.info("Évaluation des métriques de confiance...")
        expert_evaluation = evaluate_trust_metrics(explanation, uncertainty_metrics, fairwashing_audit, confidence_report)
        
        # Création du générateur de graphiques
        chart_generator = ChartGenerator()
        
        # Génération des visualisations
        logger.info("Génération des visualisations...")
        visualizations = {
            "explanation": visualize_explanation(explanation, chart_generator),
            "uncertainty": visualize_uncertainty(uncertainty_metrics, chart_generator),
            "fairwashing": visualize_fairwashing(fairwashing_audit, chart_generator),
            "expert_evaluation": visualize_expert_evaluation(expert_evaluation, chart_generator)
        }
        
        # Génération du rapport HTML
        logger.info("Génération du rapport HTML...")
        html_report = generate_html_report(
            explanation=explanation,
            uncertainty_metrics=uncertainty_metrics,
            fairwashing_audit=fairwashing_audit,
            confidence_report=confidence_report,
            expert_evaluation=expert_evaluation,
            visualizations=visualizations,
            model_adapter=model_adapter,
            instance=instance,
            feature_names=feature_names,
            target_names=target_names,
            true_label=true_label,
            predicted_label=predicted_label
        )
        
        # Sauvegarde du rapport HTML
        report_path = "interactive_trust_demo_report.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_report)
        
        logger.info(f"Rapport HTML généré et sauvegardé dans {report_path}")
        logger.info("Démonstration interactive terminée avec succès")
        
        # Ouverture du rapport dans le navigateur
        import webbrowser
        webbrowser.open(report_path)
        
        return {
            "explanation": explanation,
            "uncertainty_metrics": uncertainty_metrics,
            "fairwashing_audit": fairwashing_audit,
            "confidence_report": confidence_report,
            "expert_evaluation": expert_evaluation,
            "visualizations": visualizations,
            "report_path": report_path
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de la démonstration interactive: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    """
    Point d'entrée principal pour la démonstration interactive des métriques de confiance.
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
        
        # Exécution de la démonstration interactive
        print("\nDémarrage de la démonstration interactive des métriques de confiance...\n")
        results = run_interactive_demo()
        
        if "error" in results:
            print(f"\nErreur lors de la démonstration: {results['error']}")
        else:
            print("\nDémonstration terminée avec succès!")
            print(f"Rapport HTML généré: {results['report_path']}")
            
            # Affichage des scores
            if "expert_evaluation" in results and "global_score" in results["expert_evaluation"]:
                print(f"\nScore global de confiance: {results['expert_evaluation']['global_score']:.2f}/10")
            
            # Affichage des recommandations
            if "expert_evaluation" in results and "recommendations" in results["expert_evaluation"]:
                print("\nRecommandations:")
                for i, rec in enumerate(results["expert_evaluation"]["recommendations"], 1):
                    print(f"  {i}. {rec}")
    
    except KeyboardInterrupt:
        print("\nDémonstration interrompue par l'utilisateur.")
    
    except Exception as e:
        print(f"\nErreur inattendue: {e}")
        import traceback
        traceback.print_exc()
