"""
Test de validation sur un cas d'utilisation réel dans le secteur bancaire.

Ce module teste les modules de conformité GDPR et AI Act sur un scénario
d'approbation de crédit utilisant un modèle de scoring.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from xplia.compliance.explanation_rights import (
    GDPRComplianceManager,
    DataCategory,
    ProcessingPurpose,
    LegalBasis
)

from xplia.compliance.ai_act import (
    AIActComplianceManager,
    RiskLevel,
    AISystemCategory,
    RequirementStatus
)

from xplia.explainers.lime_explainer import LIMEExplainer
from xplia.explainers.shap_explainer import SHAPExplainer


class TestBankingUseCase(unittest.TestCase):
    """Test de validation sur un cas d'utilisation bancaire."""

    def setUp(self):
        """Préparation des données et du modèle pour un scénario d'approbation de crédit."""
        # Création d'un jeu de données synthétique simulant des demandes de crédit
        np.random.seed(42)
        n_samples = 500
        
        # Features: âge, revenu, ancienneté professionnelle, historique de crédit, montant demandé
        age = np.random.normal(40, 10, n_samples).clip(min=21, max=70)
        income = np.random.normal(50000, 20000, n_samples).clip(min=15000)
        job_years = np.random.normal(8, 5, n_samples).clip(min=0)
        credit_history = np.random.normal(650, 100, n_samples).clip(min=300, max=850)
        loan_amount = np.random.normal(150000, 100000, n_samples).clip(min=5000)
        
        # Calcul d'une probabilité de remboursement basée sur ces features
        repay_prob = (
            0.4 * (credit_history - 300) / 550 +
            0.3 * np.clip((income - 20000) / 80000, 0, 1) +
            0.2 * np.clip(job_years / 15, 0, 1) +
            0.1 * (1 - np.clip((loan_amount - 5000) / 400000, 0, 1))
        )
        
        # Création d'une variable de décision binaire (approbation ou refus)
        y = (repay_prob >= np.random.uniform(0.4, 0.6, n_samples)).astype(int)
        
        # Création du DataFrame
        self.data = pd.DataFrame({
            'age': age,
            'income': income,
            'job_years': job_years,
            'credit_score': credit_history,
            'loan_amount': loan_amount,
        })
        
        # Cible: 1 = approbation, 0 = refus
        self.target = y
        
        # Division en ensembles d'entraînement et de test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, self.target, test_size=0.3, random_state=42
        )
        
        # Entraînement d'un modèle de classification
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        
        # Préparation des données pour l'analyse
        self.test_data = {
            'features': self.X_test.values.tolist(),
            'feature_names': self.X_test.columns.tolist(),
            'metadata': {
                'domain': 'banking',
                'purpose': 'credit_approval',
                'has_personal_data': True
            }
        }
        
        # Initialisation des managers de conformité
        self.gdpr_manager = GDPRComplianceManager()
        self.ai_act_manager = AIActComplianceManager()
        
        # Configuration GDPR pour un cas bancaire
        self._setup_gdpr_banking()
        
        # Configuration AI Act pour un cas bancaire (considéré comme à haut risque)
        self._setup_ai_act_banking()
    
    def _setup_gdpr_banking(self):
        """Configure le manager GDPR pour un cas d'utilisation bancaire."""
        # Enregistrement des activités de traitement
        self.gdpr_manager.data_processing_registry.register_processing(
            name="Credit Scoring",
            description="Automated credit approval based on customer data",
            categories=[
                DataCategory.PERSONAL,
                DataCategory.FINANCIAL,
                DataCategory.PROFESSIONAL
            ],
            purpose=ProcessingPurpose.LEGITIMATE_INTEREST,
            legal_basis=LegalBasis.CONTRACT,
            retention_period=36  # 36 mois
        )
        
        # Ajout des informations sur les données traitées
        self.gdpr_manager.register_data_category_explanation(
            DataCategory.PERSONAL,
            "Données personnelles client (âge)",
            "Ces données sont nécessaires pour évaluer l'éligibilité au crédit"
        )
        
        self.gdpr_manager.register_data_category_explanation(
            DataCategory.FINANCIAL,
            "Données financières client (revenus, historique crédit, montant demandé)",
            "Ces données sont nécessaires pour évaluer la solvabilité et le risque"
        )
        
        self.gdpr_manager.register_data_category_explanation(
            DataCategory.PROFESSIONAL,
            "Données professionnelles client (ancienneté)",
            "Ces données sont nécessaires pour évaluer la stabilité financière"
        )
        
        # Configuration du système de droits des personnes concernées
        self.gdpr_manager.setup_data_subject_rights(
            access_right_enabled=True,
            rectification_right_enabled=True, 
            erasure_right_enabled=True,
            restriction_right_enabled=True,
            portability_right_enabled=True,
            objection_right_enabled=True,
            automated_decision_right_enabled=True
        )
    
    def _setup_ai_act_banking(self):
        """Configure le manager AI Act pour un cas d'utilisation bancaire."""
        # Configuration du système en tant que système à haut risque
        # (décisions de crédit affectant l'accès aux services essentiels)
        self.ai_act_manager.set_system_category(AISystemCategory.HIGH_RISK)
        self.ai_act_manager.set_risk_level(RiskLevel.HIGH)
        
        # Ajout d'informations métier
        self.ai_act_manager.update_system_information({
            'name': 'Credit Approval System',
            'version': '1.0.0',
            'purpose': 'Automated credit scoring and approval',
            'provider': 'Example Bank',
            'domain': 'Financial Services'
        })
        
        # Documentation technique
        self.ai_act_manager.update_technical_documentation(
            'system_description',
            'Système automatisé d\'approbation de crédit utilisant un modèle de Random Forest'
        )
        self.ai_act_manager.update_technical_documentation(
            'architecture',
            'Modèle d\'ensemble basé sur 100 arbres de décision analysant 5 paramètres clients'
        )
        self.ai_act_manager.update_technical_documentation(
            'development_details',
            'Entraîné sur des données historiques, optimisé pour minimiser les faux négatifs'
        )
        self.ai_act_manager.update_technical_documentation(
            'data_governance',
            'Données personnelles et financières avec un cycle de conservation de 36 mois'
        )
        
        # Ajout des risques identifiés
        self.ai_act_manager.add_risk({
            'risk_id': 'bias_risk',
            'description': 'Risque de biais dans l\'évaluation du crédit',
            'risk_level': RiskLevel.HIGH.value,
            'mitigation': 'Tests réguliers de biais et ajustement des seuils'
        })
        
        self.ai_act_manager.add_risk({
            'risk_id': 'transparency_risk',
            'description': 'Manque de transparence dans les décisions automatisées',
            'risk_level': RiskLevel.MEDIUM.value,
            'mitigation': 'Utilisation d\'explainers (LIME, SHAP) pour clarifier les décisions'
        })
        
        # Mise à jour de certaines exigences
        requirements = self.ai_act_manager.get_applicable_requirements()
        for req in requirements[:5]:  # Limiter pour ne pas surcharger le test
            self.ai_act_manager.update_requirement_status(
                req['id'],
                RequirementStatus.COMPLIANT,
                f"Mis en conformité pour le système bancaire: {req['description']}"
            )
    
    def test_gdpr_banking_analysis(self):
        """
        Test d'analyse GDPR sur le cas d'utilisation bancaire.
        Vérifie que l'analyse GDPR fonctionne correctement avec des données et
        un modèle représentatifs d'un cas d'approbation de crédit.
        """
        # Exécuter l'analyse GDPR
        result = self.gdpr_manager.analyze(self.model, self.test_data)
        
        # Vérifier les éléments essentiels de l'analyse
        self.assertIsNotNone(result)
        self.assertIn('compliance_status', result)
        self.assertIn('explanation', result)
        self.assertIn('rights_summary', result)
        
        # Les données bancaires devraient être identifiées comme personnelles et financières
        self.assertIn('data_categories', result)
        categories = result['data_categories']
        self.assertTrue(any(cat['category'] == DataCategory.PERSONAL.value for cat in categories))
        self.assertTrue(any(cat['category'] == DataCategory.FINANCIAL.value for cat in categories))
        
        # La décision automatisée devrait être identifiée
        self.assertIn('automated_decision', result)
        self.assertTrue(result['automated_decision']['is_automated'])
        
        # Vérifier que l'explication de la décision est disponible
        self.assertIn('decision_explanation', result['automated_decision'])
        self.assertIsNotNone(result['automated_decision']['decision_explanation'])
    
    def test_ai_act_banking_analysis(self):
        """
        Test d'analyse AI Act sur le cas d'utilisation bancaire.
        Vérifie que l'analyse AI Act fonctionne correctement avec des données et
        un modèle représentatifs d'un cas d'approbation de crédit.
        """
        # Exécuter l'analyse AI Act
        result = self.ai_act_manager.analyze(self.model, self.test_data)
        
        # Vérifier les éléments essentiels de l'analyse
        self.assertIsNotNone(result)
        self.assertIn('system_category', result)
        self.assertEqual(result['system_category'], AISystemCategory.HIGH_RISK.value)
        
        # Vérifier que l'évaluation des risques est présente
        self.assertIn('risk_assessment', result)
        self.assertIn('risks', result['risk_assessment'])
        risks = result['risk_assessment']['risks']
        self.assertEqual(len(risks), 2)  # Deux risques ont été ajoutés
        
        # Vérifier qu'au moins un risque élevé est identifié (biais)
        high_risks = [risk for risk in risks if risk['risk_level'] == RiskLevel.HIGH.value]
        self.assertGreater(len(high_risks), 0)
        
        # Vérifier que les exigences sont évaluées
        self.assertIn('requirements_assessment', result)
        self.assertIn('compliant_count', result['requirements_assessment'])
        self.assertGreater(result['requirements_assessment']['compliant_count'], 0)
    
    def test_explainer_integration(self):
        """
        Test d'intégration des explainers dans le contexte bancaire.
        Vérifie que les explainers peuvent être utilisés pour expliquer les
        décisions du modèle d'approbation de crédit.
        """
        # Sélectionner une instance à expliquer
        instance_idx = 0
        instance = self.X_test.iloc[instance_idx:instance_idx+1]
        
        # Vérifier que LIME fonctionne
        lime_explainer = LIMEExplainer(self.model)
        lime_explanation = lime_explainer.explain(instance)
        
        self.assertIsNotNone(lime_explanation)
        self.assertGreater(len(lime_explanation), 0)
        
        # Vérifier que la contribution d'au moins une feature est non nulle
        has_nonzero_contribution = any(
            abs(item['contribution']) > 1e-10 for item in lime_explanation
        )
        self.assertTrue(has_nonzero_contribution)
        
        # Vérifier que SHAP fonctionne également
        try:
            shap_explainer = SHAPExplainer(self.model)
            shap_explanation = shap_explainer.explain(instance)
            
            self.assertIsNotNone(shap_explanation)
            self.assertGreater(len(shap_explanation), 0)
            
            # Vérifier que la contribution d'au moins une feature est non nulle
            has_nonzero_contribution = any(
                abs(item['contribution']) > 1e-10 for item in shap_explanation
            )
            self.assertTrue(has_nonzero_contribution)
        except ImportError:
            # SHAP peut nécessiter des dépendances spécifiques
            self.skipTest("SHAP n'est pas disponible dans l'environnement de test.")
    
    def test_end_to_end_compliance(self):
        """
        Test d'intégration de bout en bout pour la conformité réglementaire.
        Simule un flux complet d'approbation de crédit avec génération d'explications
        et vérification de conformité GDPR et AI Act.
        """
        # Sélectionner une demande de crédit à évaluer
        credit_application_idx = 5
        application = self.X_test.iloc[credit_application_idx:credit_application_idx+1]
        
        # 1. Prédiction du modèle
        prediction = self.model.predict_proba(application)[0]
        decision = "Approuvé" if prediction[1] > 0.5 else "Refusé"
        
        print(f"\nDécision pour la demande de crédit: {decision} (probabilité: {prediction[1]:.2f})")
        
        # 2. Générer une explication
        lime_explainer = LIMEExplainer(self.model)
        explanation = lime_explainer.explain(application)
        
        # Afficher les principales features
        print("Facteurs principaux influençant la décision:")
        for item in sorted(explanation, key=lambda x: abs(x['contribution']), reverse=True)[:3]:
            feature = item['feature']
            contribution = item['contribution']
            direction = "favorable" if contribution > 0 else "défavorable"
            print(f"- {feature}: impact {direction} ({contribution:.4f})")
        
        # 3. Vérifier la conformité GDPR
        gdpr_result = self.gdpr_manager.analyze(self.model, {
            'features': application.values.tolist(),
            'feature_names': application.columns.tolist(),
            'metadata': {'domain': 'banking', 'purpose': 'credit_approval'}
        })
        
        gdpr_compliant = gdpr_result['compliance_status'] == 'compliant'
        print(f"\nConformité GDPR: {'Oui' if gdpr_compliant else 'Non'}")
        
        if 'automated_decision' in gdpr_result:
            rights = gdpr_result['automated_decision'].get('rights', [])
            if rights:
                print("Droits applicables concernant la décision automatisée:")
                for right in rights:
                    print(f"- {right}")
        
        # 4. Vérifier la conformité AI Act
        ai_act_result = self.ai_act_manager.analyze(self.model, {
            'features': application.values.tolist(),
            'feature_names': application.columns.tolist(),
            'metadata': {'domain': 'banking', 'purpose': 'credit_approval'}
        })
        
        high_risk = ai_act_result['system_category'] == AISystemCategory.HIGH_RISK.value
        print(f"\nSystème à haut risque selon l'AI Act: {'Oui' if high_risk else 'Non'}")
        
        if 'risk_assessment' in ai_act_result:
            risk_count = ai_act_result['risk_assessment'].get('risk_count', 0)
            print(f"Risques identifiés: {risk_count}")
            
            if risk_count > 0 and 'risks' in ai_act_result['risk_assessment']:
                for risk in ai_act_result['risk_assessment']['risks']:
                    print(f"- {risk['description']} (Niveau: {risk['risk_level']})")
        
        # 5. Assertions pour valider le test
        self.assertIsNotNone(explanation)
        self.assertIsNotNone(gdpr_result)
        self.assertIsNotNone(ai_act_result)
        
        # Une application de crédit devrait être considérée comme une décision automatisée
        if 'automated_decision' in gdpr_result:
            self.assertTrue(gdpr_result['automated_decision']['is_automated'])
        
        # Un système d'approbation de crédit devrait être identifié comme à haut risque
        self.assertEqual(ai_act_result['system_category'], AISystemCategory.HIGH_RISK.value)


if __name__ == '__main__':
    unittest.main()
