#!/usr/bin/env python
"""
Exemple d'exécution des tests de validation sur un modèle personnalisé.

Ce script montre comment utiliser les frameworks de test XPLIA pour valider
un modèle personnalisé en termes de conformité réglementaire et d'explicabilité.
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import sys
import os

# Ajouter le répertoire parent au PYTHONPATH pour pouvoir importer xplia
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xplia.compliance.explanation_rights import (
    GDPRComplianceManager,
    DataCategory,
    ProcessingPurpose,
    LegalBasis
)

from xplia.compliance.ai_act import (
    AIActComplianceManager,
    RiskLevel,
    AISystemCategory
)

from xplia.explainers.lime_explainer import LIMEExplainer
from xplia.explainers.shap_explainer import SHAPExplainer


def load_example_data():
    """
    Charge des données d'exemple ou génère des données synthétiques.
    Dans un cas réel, vous chargeriez vos propres données.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    # Option 1: Générer des données synthétiques
    n_samples = 100
    n_features = 5
    
    # Créer des caractéristiques avec des noms significatifs
    np.random.seed(42)
    feature_names = ['feature_A', 'feature_B', 'feature_C', 'feature_D', 'feature_E']
    
    # Générer des données
    X = np.random.randn(n_samples, n_features)
    # Fonction cible: combinaison non-linéaire des features
    y = 0.3 * X[:, 0]**2 + 0.7 * X[:, 1] - 1.2 * X[:, 2] + 0.5 * X[:, 3] * X[:, 4]
    
    # Diviser en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Option 2: Charger des données réelles (commenté)
    # import pandas as pd 
    # df = pd.read_csv('your_data.csv')
    # feature_names = list(df.columns[:-1])
    # X = df.iloc[:, :-1].values
    # y = df.iloc[:, -1].values
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test, feature_names


def create_and_train_model(X_train, y_train, model_type="random_forest"):
    """
    Crée et entraîne un modèle en fonction du type spécifié.
    
    Args:
        X_train: Données d'entraînement
        y_train: Cibles d'entraînement
        model_type: Type de modèle à créer
        
    Returns:
        Le modèle entraîné
    """
    if model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        # Vous pouvez ajouter d'autres types de modèles ici
        raise ValueError(f"Type de modèle non pris en charge: {model_type}")
    
    # Entraîner le modèle
    model.fit(X_train, y_train)
    return model


def setup_gdpr_manager(domain_name="custom", data_categories=None):
    """
    Configure un manager GDPR pour le modèle.
    
    Args:
        domain_name: Nom du domaine
        data_categories: Liste des catégories de données
        
    Returns:
        Le manager GDPR configuré
    """
    gdpr_manager = GDPRComplianceManager()
    
    # Utiliser les catégories par défaut si aucune n'est fournie
    if data_categories is None:
        data_categories = [DataCategory.PERSONAL, DataCategory.OTHER]
    
    # Enregistrer l'activité de traitement
    gdpr_manager.data_processing_registry.register_processing(
        name=f"{domain_name.title()} Model Processing",
        description=f"Automated processing for {domain_name} domain",
        categories=data_categories,
        purpose=ProcessingPurpose.LEGITIMATE_INTEREST,
        legal_basis=LegalBasis.CONSENT,
        retention_period=24
    )
    
    # Configurer les droits des personnes concernées
    gdpr_manager.setup_data_subject_rights(
        access_right_enabled=True,
        rectification_right_enabled=True,
        erasure_right_enabled=True,
        restriction_right_enabled=True,
        portability_right_enabled=True,
        objection_right_enabled=True,
        automated_decision_right_enabled=True
    )
    
    # Ajouter des explications pour chaque catégorie
    for category in data_categories:
        gdpr_manager.register_data_category_explanation(
            category,
            f"Données {category.value} pour modèle {domain_name}",
            f"Ces données sont nécessaires pour l'analyse dans le domaine {domain_name}"
        )
    
    return gdpr_manager


def setup_ai_act_manager(domain_name="custom", risk_level=RiskLevel.MEDIUM):
    """
    Configure un manager AI Act pour le modèle.
    
    Args:
        domain_name: Nom du domaine
        risk_level: Niveau de risque du système
        
    Returns:
        Le manager AI Act configuré
    """
    ai_act_manager = AIActComplianceManager()
    
    # Configurer la catégorie et le niveau de risque
    if risk_level == RiskLevel.HIGH:
        ai_act_manager.set_system_category(AISystemCategory.HIGH_RISK)
    else:
        ai_act_manager.set_system_category(AISystemCategory.GENERAL_PURPOSE)
    
    ai_act_manager.set_risk_level(risk_level)
    
    # Mettre à jour les informations système
    ai_act_manager.update_system_information({
        'name': f'{domain_name.title()} AI System',
        'version': '1.0.0',
        'purpose': f'Analysis for {domain_name} domain',
        'provider': 'XPLIA User',
        'domain': domain_name
    })
    
    # Documentation technique
    ai_act_manager.update_technical_documentation(
        'system_description',
        f'Système d\'intelligence artificielle pour le domaine {domain_name}'
    )
    
    # Ajouter des risques génériques
    ai_act_manager.add_risk({
        'risk_id': 'accuracy_risk',
        'description': 'Risque lié à la précision des prédictions',
        'risk_level': RiskLevel.MEDIUM.value,
        'mitigation': 'Validation croisée et surveillance continue des performances'
    })
    
    ai_act_manager.add_risk({
        'risk_id': 'explainability_risk',
        'description': 'Risque lié à l\'explicabilité des décisions',
        'risk_level': RiskLevel.MEDIUM.value,
        'mitigation': 'Utilisation d\'explainers (LIME, SHAP)'
    })
    
    return ai_act_manager


def run_validation(model, X_test, feature_names, domain_name="custom", risk_level=RiskLevel.MEDIUM):
    """
    Exécute une validation complète du modèle.
    
    Args:
        model: Le modèle à valider
        X_test: Données de test
        feature_names: Noms des caractéristiques
        domain_name: Nom du domaine
        risk_level: Niveau de risque du système
        
    Returns:
        dict: Résultats de la validation
    """
    print(f"\n==== Validation pour modèle {domain_name} (niveau de risque: {risk_level.name}) ====\n")
    
    # Convertir en DataFrame pour faciliter la manipulation
    test_data = pd.DataFrame(X_test, columns=feature_names)
    
    # Sélectionner une instance à expliquer
    instance = test_data.iloc[0:1]
    
    print("1. Génération d'explications pour le modèle...")
    
    # Tester LIME
    try:
        lime_explainer = LIMEExplainer(model)
        lime_explanation = lime_explainer.explain(instance)
        
        print("\n=== Explication LIME ===")
        for item in sorted(lime_explanation, key=lambda x: abs(x['contribution']), reverse=True):
            feature = item['feature']
            contribution = item['contribution']
            direction = "+" if contribution > 0 else "-"
            print(f"  {feature}: {direction} {abs(contribution):.4f}")
    except Exception as e:
        print(f"Échec de l'explication LIME: {str(e)}")
        lime_explanation = None
    
    # Tester SHAP si disponible
    try:
        shap_explainer = SHAPExplainer(model)
        shap_explanation = shap_explainer.explain(instance)
        
        print("\n=== Explication SHAP ===")
        for item in sorted(shap_explanation, key=lambda x: abs(x['contribution']), reverse=True):
            feature = item['feature']
            contribution = item['contribution']
            direction = "+" if contribution > 0 else "-"
            print(f"  {feature}: {direction} {abs(contribution):.4f}")
    except ImportError:
        print("SHAP n'est pas disponible (dépendance manquante)")
        shap_explanation = None
    except Exception as e:
        print(f"Échec de l'explication SHAP: {str(e)}")
        shap_explanation = None
    
    # Configuration des managers
    gdpr_manager = setup_gdpr_manager(domain_name)
    ai_act_manager = setup_ai_act_manager(domain_name, risk_level)
    
    print("\n2. Analyse de conformité GDPR...")
    
    # Format de données pour l'analyse
    test_data_dict = {
        'features': instance.values.tolist(),
        'feature_names': feature_names,
        'metadata': {
            'domain': domain_name,
            'purpose': 'validation_test',
            'has_personal_data': True,
            'explanation': lime_explanation or shap_explanation
        }
    }
    
    # Exécuter l'analyse GDPR
    gdpr_result = gdpr_manager.analyze(model, test_data_dict)
    
    print(f"Statut de conformité GDPR: {gdpr_result['compliance_status']}")
    if 'automated_decision' in gdpr_result:
        automated = gdpr_result['automated_decision']['is_automated']
        print(f"Décision automatisée: {'Oui' if automated else 'Non'}")
        
        if automated and 'rights' in gdpr_result['automated_decision']:
            print("Droits applicables:")
            for right in gdpr_result['automated_decision']['rights']:
                print(f"  - {right}")
    
    print("\n3. Analyse de conformité AI Act...")
    
    # Exécuter l'analyse AI Act
    ai_act_result = ai_act_manager.analyze(model, test_data_dict)
    
    print(f"Catégorie du système: {ai_act_result['system_category']}")
    
    if 'risk_assessment' in ai_act_result:
        risk_count = ai_act_result['risk_assessment'].get('risk_count', 0)
        print(f"Nombre de risques identifiés: {risk_count}")
        
        if risk_count > 0 and 'risks' in ai_act_result['risk_assessment']:
            print("Risques principaux:")
            for risk in ai_act_result['risk_assessment']['risks']:
                print(f"  - {risk['description']} (Niveau: {risk['risk_level']})")
    
    if 'requirements_assessment' in ai_act_result:
        compliant = ai_act_result['requirements_assessment'].get('compliant_count', 0)
        total = ai_act_result['requirements_assessment'].get('total_count', 0)
        print(f"Exigences satisfaites: {compliant}/{total}")
    
    print("\n==== Validation terminée ====\n")
    
    return {
        'gdpr_result': gdpr_result,
        'ai_act_result': ai_act_result,
        'lime_explanation': lime_explanation,
        'shap_explanation': shap_explanation
    }


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description='Valide un modèle personnalisé avec XPLIA')
    parser.add_argument('--domain', type=str, default='custom',
                        help='Domaine du modèle (par défaut: custom)')
    parser.add_argument('--risk', type=str, choices=['low', 'medium', 'high'], default='medium',
                        help='Niveau de risque AI Act (par défaut: medium)')
    
    args = parser.parse_args()
    
    # Convertir l'argument de risque en enum
    risk_level_map = {
        'low': RiskLevel.LOW,
        'medium': RiskLevel.MEDIUM,
        'high': RiskLevel.HIGH
    }
    risk_level = risk_level_map[args.risk]
    
    # Charger les données
    X_train, X_test, y_train, y_test, feature_names = load_example_data()
    
    # Créer et entraîner le modèle
    model = create_and_train_model(X_train, y_train)
    
    # Évaluer les performances du modèle
    predictions = model.predict(X_test)
    mse = np.mean((predictions - y_test) ** 2)
    print(f"Performance du modèle - MSE: {mse:.4f}")
    
    # Exécuter la validation
    validation_results = run_validation(
        model, 
        X_test, 
        feature_names, 
        domain_name=args.domain, 
        risk_level=risk_level
    )
    
    print("Validation complète. Consultez les résultats ci-dessus pour les détails.")


if __name__ == "__main__":
    main()
