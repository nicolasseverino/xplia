"""
Démonstration des formatters avec métriques de confiance
========================================================

Ce script démontre l'utilisation des formatters HTML et PDF avec
intégration des métriques de confiance pour les explications XPLIA.
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

try:
    from xplia.explainers import ShapExplainer, UncertaintyQuantifier, FairwashingDetector, ConfidenceReport
    from xplia.core.model_adapters import SklearnAdapter
    from xplia.compliance.formatters.html_trust_formatter import TrustHTMLReportGenerator
    from xplia.compliance.formatters.pdf_trust_formatter import TrustPDFReportGenerator
    from xplia.compliance.report_base import ReportConfig, ReportContent
except ImportError as e:
    logger.error(f"Erreur d'importation: {e}")
    logger.error("Assurez-vous que le package xplia est correctement installé.")
    sys.exit(1)

# Vérification des dépendances optionnelles
try:
    import sklearn
    import shap
    import matplotlib
    import fpdf
except ImportError as e:
    logger.warning(f"Dépendance optionnelle manquante: {e}")
    logger.warning("Certaines fonctionnalités peuvent ne pas être disponibles.")

def create_sample_data():
    """Crée des données d'exemple pour la démonstration."""
    try:
        # Création d'un jeu de données simple
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.normal(0, 1, 100),
            'feature4': np.random.normal(0, 1, 100),
        })
        y = (X['feature1'] > 0).astype(int) * 0.5 + (X['feature2'] > 0).astype(int) * 0.3 + \
            (X['feature3'] > 0).astype(int) * 0.2 + np.random.normal(0, 0.1, 100)
        y = (y > 0.5).astype(int)
        
        # Séparation en train/test
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Erreur lors de la création des données d'exemple: {e}")
        sys.exit(1)

def train_sample_model(X_train, y_train):
    """Entraîne un modèle d'exemple."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        
        # Entraînement d'un modèle simple
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        return model
    except ImportError:
        logger.error("sklearn n'est pas installé. Impossible d'entraîner le modèle.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement du modèle: {e}")
        sys.exit(1)

def generate_explanation_with_trust_metrics(model, X_train, X_test):
    """Génère une explication avec métriques de confiance."""
    try:
        # Création de l'adaptateur de modèle
        model_adapter = SklearnAdapter(model)
        
        # Création de l'explainer
        explainer = ShapExplainer()
        
        # Génération de l'explication pour une instance
        instance = X_test.iloc[0]
        explanation = explainer.explain(model_adapter, instance, background_data=X_train)
        
        # Évaluation de l'incertitude
        uncertainty_quantifier = UncertaintyQuantifier(n_bootstrap_samples=10)
        uncertainty_metrics = uncertainty_quantifier.quantify_uncertainty(
            explanation, explainer, X_test.sample(5)
        )
        
        # Détection de fairwashing
        fairwashing_detector = FairwashingDetector(sensitive_features=['feature1'])
        fairwashing_audit = fairwashing_detector.detect_fairwashing(
            explanation, X=X_test.sample(5)
        )
        
        # Génération du rapport de confiance
        confidence_reporter = ConfidenceReport()
        confidence_report = confidence_reporter.generate_report(
            explanation, uncertainty_metrics, fairwashing_audit
        )
        
        # Enrichissement de l'explication avec le rapport de confiance
        explanation = confidence_reporter.apply_to_explanation(explanation, confidence_report)
        
        return explanation
    except Exception as e:
        logger.error(f"Erreur lors de la génération de l'explication: {e}")
        sys.exit(1)

def generate_html_report(explanation):
    """Génère un rapport HTML avec métriques de confiance."""
    try:
        # Configuration du rapport
        config = ReportConfig(
            template_name="standard",
            language="fr",
            include_verification_qr=True,
            verification_url="https://xplia.ai/verify",
            include_signatures=True
        )
        
        # Création du générateur de rapports HTML avec métriques de confiance
        html_generator = TrustHTMLReportGenerator(config)
        
        # Création du contenu du rapport
        content = ReportContent(
            title="Rapport d'Explication avec Métriques de Confiance",
            description="Ce rapport présente une explication avec évaluation de confiance.",
            explanations=[explanation],
            metadata={
                "model_type": "RandomForestClassifier",
                "date_created": "2025-06-13T08:30:00+02:00",
                "authors": ["XPLIA System"],
                "version": "1.0.0"
            }
        )
        
        # Génération du rapport
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "trust_report.html"
        
        html_generator.generate(content, str(output_path))
        logger.info(f"Rapport HTML généré: {output_path}")
        
        return output_path
    except Exception as e:
        logger.error(f"Erreur lors de la génération du rapport HTML: {e}")
        return None

def generate_pdf_report(explanation):
    """Génère un rapport PDF avec métriques de confiance."""
    try:
        # Configuration du rapport
        config = ReportConfig(
            template_name="standard",
            language="fr",
            include_verification_qr=True,
            verification_url="https://xplia.ai/verify",
            include_signatures=True
        )
        
        # Création du générateur de rapports PDF avec métriques de confiance
        pdf_generator = TrustPDFReportGenerator(config)
        
        # Création du contenu du rapport
        content = ReportContent(
            title="Rapport d'Explication avec Métriques de Confiance",
            description="Ce rapport présente une explication avec évaluation de confiance.",
            explanations=[explanation],
            metadata={
                "model_type": "RandomForestClassifier",
                "date_created": "2025-06-13T08:30:00+02:00",
                "authors": ["XPLIA System"],
                "version": "1.0.0"
            }
        )
        
        # Génération du rapport
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "trust_report.pdf"
        
        pdf_generator.generate(content, str(output_path))
        logger.info(f"Rapport PDF généré: {output_path}")
        
        return output_path
    except Exception as e:
        logger.error(f"Erreur lors de la génération du rapport PDF: {e}")
        return None

def main():
    """Fonction principale de démonstration."""
    try:
        logger.info("Démarrage de la démonstration des formatters avec métriques de confiance...")
        
        # Création des données d'exemple
        logger.info("Création des données d'exemple...")
        X_train, X_test, y_train, y_test = create_sample_data()
        
        # Entraînement du modèle
        logger.info("Entraînement du modèle...")
        model = train_sample_model(X_train, y_train)
        
        # Génération de l'explication avec métriques de confiance
        logger.info("Génération de l'explication avec métriques de confiance...")
        explanation = generate_explanation_with_trust_metrics(model, X_train, X_test)
        
        # Génération du rapport HTML
        logger.info("Génération du rapport HTML...")
        html_path = generate_html_report(explanation)
        
        # Génération du rapport PDF
        logger.info("Génération du rapport PDF...")
        pdf_path = generate_pdf_report(explanation)
        
        # Affichage des résultats
        if html_path:
            logger.info(f"Rapport HTML généré avec succès: {html_path}")
            logger.info(f"Ouvrez {html_path} dans un navigateur pour visualiser le rapport HTML.")
        
        if pdf_path:
            logger.info(f"Rapport PDF généré avec succès: {pdf_path}")
            logger.info(f"Ouvrez {pdf_path} avec un lecteur PDF pour visualiser le rapport PDF.")
        
        logger.info("Démonstration terminée.")
    except Exception as e:
        logger.error(f"Erreur lors de la démonstration: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
