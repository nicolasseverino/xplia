"""
Démonstration simple des modules de confiance XPLIA
==================================================

Ce script démontre l'utilisation des modules de confiance XPLIA
(incertitude, fairwashing, rapport de confiance) de manière simple et directe.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ajout du répertoire parent au path pour l'import de xplia
sys.path.insert(0, str(Path(__file__).parent.parent))

# Gestion des importations avec traitement des erreurs
try:
    # Imports XPLIA
    from xplia.explainers import (
        ShapExplainer,
        UncertaintyQuantifier, UncertaintyMetrics, UncertaintyType,
        FairwashingDetector, FairwashingAudit, FairwashingType,
        ConfidenceReport, TrustScore, TrustLevel
    )
    from xplia.core.model_adapters import SklearnAdapter
except ImportError as e:
    logger.error(f"Erreur d'importation XPLIA: {e}")
    logger.error("Assurez-vous que le package xplia est correctement installé.")
    sys.exit(1)

# Vérification des dépendances optionnelles
try:
    import sklearn
    import shap
    import matplotlib.pyplot as plt
except ImportError as e:
    logger.warning(f"Dépendance optionnelle manquante: {e}")
    logger.warning("Certaines fonctionnalités peuvent ne pas être disponibles.")

def create_sample_data():
    """Crée des données d'exemple pour la démonstration."""
    # Création d'un jeu de données simple
    np.random.seed(42)
    X = pd.DataFrame({
        'age': np.random.normal(40, 10, 100),
        'income': np.random.normal(50000, 15000, 100),
        'education': np.random.randint(8, 20, 100),
        'gender': np.random.choice([0, 1], 100),  # Feature sensible
    })
    
    # Création d'une cible avec un léger biais sur le genre
    y = (0.4 * (X['age'] > 40) + 
         0.3 * (X['income'] > 50000) + 
         0.2 * (X['education'] > 15) +
         0.1 * (X['gender'] == 1) +  # Biais léger sur le genre
         np.random.normal(0, 0.1, 100)) > 0.5
    y = y.astype(int)
    
    # Séparation en train/test
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    return X_train, X_test, y_train, y_test

def train_sample_model(X_train, y_train):
    """Entraîne un modèle d'exemple."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        
        # Entraînement d'un modèle simple
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Évaluation rapide
        train_score = model.score(X_train, y_train)
        logger.info(f"Score d'entraînement du modèle: {train_score:.4f}")
        
        return model
    except ImportError:
        logger.error("sklearn n'est pas installé. Impossible d'entraîner le modèle.")
        sys.exit(1)

def demonstrate_uncertainty_quantification(explainer, explanation, X_test):
    """Démontre la quantification d'incertitude."""
    logger.info("\n=== DÉMONSTRATION DE LA QUANTIFICATION D'INCERTITUDE ===")
    
    # Création du quantificateur d'incertitude
    uncertainty_quantifier = UncertaintyQuantifier(
        n_bootstrap_samples=20,  # Nombre d'échantillons bootstrap
        confidence_level=0.95,   # Niveau de confiance
        methods=["bootstrap", "ensemble", "sensitivity"]  # Méthodes d'estimation
    )
    
    # Quantification de l'incertitude
    uncertainty_metrics = uncertainty_quantifier.quantify_uncertainty(
        explanation=explanation,
        explainer=explainer,
        X=X_test.sample(10)  # Échantillon pour l'estimation
    )
    
    # Affichage des métriques d'incertitude
    print("\nMétriques d'incertitude:")
    print(f"Incertitude globale: {uncertainty_metrics.global_uncertainty:.4f}")
    print(f"Incertitude aléatoire: {uncertainty_metrics.aleatoric_uncertainty:.4f}")
    print(f"Incertitude épistémique: {uncertainty_metrics.epistemic_uncertainty:.4f}")
    print(f"Incertitude structurelle: {uncertainty_metrics.structural_uncertainty:.4f}")
    print(f"Incertitude d'approximation: {uncertainty_metrics.approximation_uncertainty:.4f}")
    
    # Affichage des incertitudes par feature
    print("\nIncertitude par feature:")
    for feature, uncertainty in uncertainty_metrics.feature_uncertainties.items():
        print(f"  {feature}: {uncertainty:.4f}")
    
    # Affichage des intervalles de confiance
    print("\nIntervalles de confiance:")
    for feature, interval in uncertainty_metrics.confidence_intervals.items():
        print(f"  {feature}: [{interval[0]:.4f}, {interval[1]:.4f}]")
    
    return uncertainty_metrics

def demonstrate_fairwashing_detection(explanation, X_test):
    """Démontre la détection de fairwashing."""
    logger.info("\n=== DÉMONSTRATION DE LA DÉTECTION DE FAIRWASHING ===")
    
    # Création du détecteur de fairwashing
    fairwashing_detector = FairwashingDetector(
        sensitive_features=['gender'],  # Features sensibles
        detection_threshold=0.7,        # Seuil de détection
        methods=["consistency", "sensitivity", "counterfactual"]  # Méthodes de détection
    )
    
    # Détection de fairwashing
    fairwashing_audit = fairwashing_detector.detect_fairwashing(
        explanation=explanation,
        X=X_test  # Données pour la détection
    )
    
    # Affichage des résultats de l'audit
    print("\nRésultats de l'audit de fairwashing:")
    print(f"Score global de fairwashing: {fairwashing_audit.fairwashing_score:.4f}")
    
    # Affichage des types détectés
    print("\nTypes de fairwashing détectés:")
    if fairwashing_audit.detected_types:
        for fairwashing_type in fairwashing_audit.detected_types:
            print(f"  - {fairwashing_type}")
    else:
        print("  Aucun type de fairwashing détecté")
    
    # Affichage des scores par type
    print("\nScores par type de fairwashing:")
    for type_name, score in fairwashing_audit.type_scores.items():
        print(f"  {type_name}: {score:.4f}")
    
    # Affichage des scores de manipulation par feature
    print("\nScores de manipulation par feature:")
    for feature, score in fairwashing_audit.feature_manipulation_scores.items():
        print(f"  {feature}: {score:.4f}")
    
    return fairwashing_audit

def demonstrate_confidence_report(explanation, uncertainty_metrics, fairwashing_audit):
    """Démontre la génération d'un rapport de confiance."""
    logger.info("\n=== DÉMONSTRATION DU RAPPORT DE CONFIANCE ===")
    
    # Création du générateur de rapports de confiance
    confidence_reporter = ConfidenceReport(
        uncertainty_weight=0.4,    # Poids de l'incertitude
        fairwashing_weight=0.3,    # Poids du fairwashing
        consistency_weight=0.2,    # Poids de la cohérence
        robustness_weight=0.1      # Poids de la robustesse
    )
    
    # Génération du rapport de confiance
    confidence_report = confidence_reporter.generate_report(
        explanation=explanation,
        uncertainty_metrics=uncertainty_metrics,
        fairwashing_audit=fairwashing_audit
    )
    
    # Affichage du score de confiance
    trust_score = confidence_report["trust_score"]
    print("\nScore de confiance:")
    print(f"Score global: {trust_score['global_trust']:.4f}")
    print(f"Niveau de confiance: {trust_score['trust_level']}")
    
    # Affichage des scores par dimension
    print("\nScores par dimension:")
    print(f"Confiance liée à l'incertitude: {trust_score['uncertainty_trust']:.4f}")
    print(f"Confiance liée au fairwashing: {trust_score['fairwashing_trust']:.4f}")
    print(f"Confiance liée à la cohérence: {trust_score['consistency_trust']:.4f}")
    print(f"Confiance liée à la robustesse: {trust_score['robustness_trust']:.4f}")
    
    # Affichage des facteurs influençant le score
    print("\nFacteurs influençant le score:")
    for factor in trust_score['trust_factors']:
        print(f"  - {factor}")
    
    # Affichage du résumé
    print(f"\nRésumé: {confidence_report['summary']}")
    
    # Affichage des recommandations
    print("\nRecommandations:")
    for recommendation in confidence_report['recommendations']:
        print(f"  - {recommendation}")
    
    return confidence_report

def visualize_trust_metrics(uncertainty_metrics, fairwashing_audit, confidence_report):
    """Visualise les métriques de confiance."""
    try:
        logger.info("\n=== VISUALISATION DES MÉTRIQUES DE CONFIANCE ===")
        
        # Création d'une figure avec 3 sous-graphiques
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Graphique d'incertitude
        uncertainty_types = [
            ('Aléatoire', uncertainty_metrics.aleatoric_uncertainty),
            ('Épistémique', uncertainty_metrics.epistemic_uncertainty),
            ('Structurelle', uncertainty_metrics.structural_uncertainty),
            ('Approximation', uncertainty_metrics.approximation_uncertainty)
        ]
        
        labels, values = zip(*uncertainty_types)
        ax1.bar(labels, values, color='skyblue')
        ax1.set_title('Types d\'Incertitude')
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Score d\'incertitude')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Graphique de fairwashing
        fairwashing_scores = fairwashing_audit.type_scores
        if fairwashing_scores:
            fw_labels, fw_values = zip(*fairwashing_scores.items())
            ax2.bar(fw_labels, fw_values, color='salmon')
        else:
            ax2.text(0.5, 0.5, "Aucun score de fairwashing", 
                    horizontalalignment='center', verticalalignment='center')
        
        ax2.set_title('Scores de Fairwashing par Type')
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('Score de fairwashing')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Graphique radar des scores de confiance
        trust_score = confidence_report["trust_score"]
        categories = ['Incertitude', 'Fairwashing', 'Cohérence', 'Robustesse']
        values = [
            trust_score['uncertainty_trust'],
            trust_score['fairwashing_trust'],
            trust_score['consistency_trust'],
            trust_score['robustness_trust']
        ]
        
        # Conversion en coordonnées polaires
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Fermer le graphique
        
        values += values[:1]  # Fermer le graphique
        
        ax3.set_theta_offset(np.pi / 2)
        ax3.set_theta_direction(-1)
        ax3.set_rlabel_position(0)
        
        plt.xticks(angles[:-1], categories)
        ax3.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax3.set_yticklabels(['0.2', '0.4', '0.6', '0.8'])
        ax3.set_ylim(0, 1)
        
        ax3.plot(angles, values, linewidth=2, linestyle='solid')
        ax3.fill(angles, values, alpha=0.25)
        ax3.set_title('Scores de Confiance')
        
        # Ajustement de la mise en page
        plt.tight_layout()
        
        # Sauvegarde de la figure
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "trust_metrics_visualization.png"
        plt.savefig(output_path)
        
        logger.info(f"Visualisation sauvegardée: {output_path}")
        
        # Affichage de la figure
        plt.show()
    except Exception as e:
        logger.error(f"Erreur lors de la visualisation: {e}")

def main():
    """Fonction principale de démonstration."""
    try:
        logger.info("Démarrage de la démonstration des modules de confiance XPLIA...")
        
        # Création des données d'exemple
        logger.info("Création des données d'exemple...")
        X_train, X_test, y_train, y_test = create_sample_data()
        
        # Entraînement du modèle
        logger.info("Entraînement du modèle...")
        model = train_sample_model(X_train, y_train)
        
        # Création de l'adaptateur de modèle
        model_adapter = SklearnAdapter(model)
        
        # Création de l'explainer
        explainer = ShapExplainer()
        
        # Génération de l'explication pour une instance
        instance = X_test.iloc[0]
        logger.info(f"Génération de l'explication pour l'instance: \n{instance}")
        
        explanation = explainer.explain(
            model_adapter=model_adapter,
            instance=instance,
            background_data=X_train
        )
        
        # Affichage des attributions de features
        print("\nAttributions de features:")
        for feature, value in explanation["feature_attributions"].items():
            print(f"  {feature}: {value:.4f}")
        
        # Démonstration de la quantification d'incertitude
        uncertainty_metrics = demonstrate_uncertainty_quantification(explainer, explanation, X_test)
        
        # Démonstration de la détection de fairwashing
        fairwashing_audit = demonstrate_fairwashing_detection(explanation, X_test)
        
        # Démonstration du rapport de confiance
        confidence_report = demonstrate_confidence_report(
            explanation, uncertainty_metrics, fairwashing_audit
        )
        
        # Visualisation des métriques de confiance
        visualize_trust_metrics(uncertainty_metrics, fairwashing_audit, confidence_report)
        
        logger.info("Démonstration terminée.")
    except Exception as e:
        logger.error(f"Erreur lors de la démonstration: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
