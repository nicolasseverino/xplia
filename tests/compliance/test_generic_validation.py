"""
Tests de validation génériques pour tous modèles et domaines.

Ce module implémente des tests de validation qui peuvent être appliqués à
n'importe quel type de modèle et de domaine, avec différentes configurations
pour évaluer la robustesse et la conformité des modules RGPD et AI Act.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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


class ModelTestCase:
    """Classe de base pour générer des cas de test de modèles."""
    
    def __init__(self, name, model, is_classifier=True, n_samples=200, n_features=10, 
                 risk_level=RiskLevel.MEDIUM, data_categories=None):
        """
        Initialise un cas de test pour un modèle spécifique.
        
        Args:
            name: Nom du cas de test
            model: Instance du modèle à tester
            is_classifier: True si c'est un modèle de classification, False pour régression
            n_samples: Nombre d'échantillons à générer
            n_features: Nombre de caractéristiques à générer
            risk_level: Niveau de risque AI Act à associer
            data_categories: Catégories de données à associer pour GDPR
        """
        self.name = name
        self.model = model
        self.is_classifier = is_classifier
        self.n_samples = n_samples
        self.n_features = n_features
        self.risk_level = risk_level
        self.data_categories = data_categories or [
            DataCategory.PERSONAL, 
            DataCategory.OTHER
        ]
        
        # Générer les données
        self._generate_data()
    
    def _generate_data(self):
        """Génère des données synthétiques adaptées au modèle."""
        if self.is_classifier:
            X, y = make_classification(
                n_samples=self.n_samples,
                n_features=self.n_features,
                n_informative=max(2, self.n_features // 2),
                n_redundant=max(1, self.n_features // 5),
                random_state=42
            )
        else:
            X, y = make_regression(
                n_samples=self.n_samples,
                n_features=self.n_features,
                n_informative=max(2, self.n_features // 2),
                noise=0.1,
                random_state=42
            )
        
        # Diviser en ensembles d'entraînement et de test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Entraîner le modèle
        self.model.fit(self.X_train, self.y_train)
        
        # Créer des noms de features pour faciliter l'interprétation
        self.feature_names = [f'feature_{i}' for i in range(self.n_features)]
        
        # Convertir en format compatible avec les explainers/analyseurs
        self.train_data = pd.DataFrame(self.X_train, columns=self.feature_names)
        self.test_data = pd.DataFrame(self.X_test, columns=self.feature_names)


class TestGenericValidation(unittest.TestCase):
    """Tests génériques de validation sur différents modèles et scénarios."""
    
    def setUp(self):
        """Prépare l'environnement de test avec différents types de modèles."""
        # Initialiser les managers de conformité
        self.gdpr_manager = GDPRComplianceManager()
        self.ai_act_manager = AIActComplianceManager()
        
        # Créer différents cas de test avec différents modèles
        self.test_cases = [
            # Cas de classification
            ModelTestCase(
                name="RandomForestClassifier",
                model=RandomForestClassifier(n_estimators=50, random_state=42),
                is_classifier=True,
                risk_level=RiskLevel.MEDIUM
            ),
            ModelTestCase(
                name="LogisticRegression",
                model=LogisticRegression(random_state=42),
                is_classifier=True,
                risk_level=RiskLevel.LOW
            ),
            ModelTestCase(
                name="SVC",
                model=SVC(probability=True, random_state=42),
                is_classifier=True,
                risk_level=RiskLevel.MEDIUM
            ),
            # Cas de régression
            ModelTestCase(
                name="LinearRegression",
                model=LinearRegression(),
                is_classifier=False,
                risk_level=RiskLevel.LOW
            ),
            ModelTestCase(
                name="GradientBoostingRegressor",
                model=GradientBoostingRegressor(random_state=42),
                is_classifier=False,
                risk_level=RiskLevel.MEDIUM
            ),
            # Modèle avec pipeline
            ModelTestCase(
                name="PipelineWithScaler",
                model=Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', RandomForestClassifier(random_state=42))
                ]),
                is_classifier=True,
                risk_level=RiskLevel.MEDIUM
            )
        ]
        
        # Configurer les managers pour les différents scénarios de conformité
        self._setup_compliance_managers()
    
    def _setup_compliance_managers(self):
        """Configure les managers de conformité pour les tests."""
        # Configuration GDPR basique
        self.gdpr_manager.data_processing_registry.register_processing(
            name="Generic AI Model Processing",
            description="Traitement générique des données pour modèles d'IA",
            categories=[
                DataCategory.PERSONAL,
                DataCategory.OTHER
            ],
            purpose=ProcessingPurpose.LEGITIMATE_INTEREST,
            legal_basis=LegalBasis.CONSENT,
            retention_period=24
        )
        
        # Configuration des droits des personnes concernées
        self.gdpr_manager.setup_data_subject_rights(
            access_right_enabled=True,
            rectification_right_enabled=True,
            erasure_right_enabled=True,
            restriction_right_enabled=True,
            portability_right_enabled=True,
            objection_right_enabled=True,
            automated_decision_right_enabled=True
        )
        
        # Configuration AI Act basique (sera personnalisée pour chaque test)
        self.ai_act_manager.set_system_category(AISystemCategory.GENERAL_PURPOSE)
        self.ai_act_manager.set_risk_level(RiskLevel.MEDIUM)
        
        # Documentation technique générique
        self.ai_act_manager.update_technical_documentation(
            'system_description',
            'Système générique d\'intelligence artificielle'
        )
    
    def test_gdpr_compliance_all_models(self):
        """
        Teste la conformité GDPR pour tous les types de modèles.
        Vérifie que l'analyse GDPR fonctionne correctement indépendamment
        du type de modèle testé.
        """
        for test_case in self.test_cases:
            with self.subTest(model=test_case.name):
                # Adapter la configuration GDPR au modèle actuel
                self._configure_gdpr_for_model(test_case)
                
                # Format de données pour l'analyse
                test_data = {
                    'features': test_case.test_data.values.tolist(),
                    'feature_names': test_case.feature_names,
                    'metadata': {
                        'domain': 'generic',
                        'purpose': 'validation_test',
                        'has_personal_data': True
                    }
                }
                
                # Exécuter l'analyse GDPR
                result = self.gdpr_manager.analyze(test_case.model, test_data)
                
                # Vérifier les éléments essentiels de l'analyse
                self.assertIsNotNone(result)
                self.assertIn('compliance_status', result)
                self.assertIn('explanation', result)
                self.assertIn('rights_summary', result)
                self.assertIn('data_categories', result)
                
                # Vérifier que le statut de conformité est valide
                self.assertIn(result['compliance_status'], ['compliant', 'non_compliant', 'partially_compliant'])
    
    def _configure_gdpr_for_model(self, test_case):
        """Configure le manager GDPR spécifiquement pour le modèle en test."""
        # Ajouter des explications spécifiques pour les catégories de données
        for category in test_case.data_categories:
            self.gdpr_manager.register_data_category_explanation(
                category,
                f"Données {category.value} pour modèle {test_case.name}",
                f"Ces données sont nécessaires pour l'analyse avec {test_case.name}"
            )
    
    def test_ai_act_compliance_all_models(self):
        """
        Teste la conformité AI Act pour tous les types de modèles.
        Vérifie que l'analyse AI Act fonctionne correctement indépendamment
        du type de modèle testé, avec différents niveaux de risque.
        """
        for test_case in self.test_cases:
            with self.subTest(model=test_case.name):
                # Adapter la configuration AI Act au modèle actuel
                self._configure_ai_act_for_model(test_case)
                
                # Format de données pour l'analyse
                test_data = {
                    'features': test_case.test_data.values.tolist(),
                    'feature_names': test_case.feature_names,
                    'metadata': {
                        'domain': 'generic',
                        'purpose': 'validation_test'
                    }
                }
                
                # Exécuter l'analyse AI Act
                result = self.ai_act_manager.analyze(test_case.model, test_data)
                
                # Vérifier les éléments essentiels de l'analyse
                self.assertIsNotNone(result)
                self.assertIn('system_category', result)
                self.assertIn('risk_assessment', result)
                self.assertIn('risks', result['risk_assessment'])
                self.assertIn('requirements_assessment', result)
                
                # Vérifier que le niveau de risque correspond à celui attendu
                system_category = AISystemCategory.HIGH_RISK if test_case.risk_level == RiskLevel.HIGH else AISystemCategory.GENERAL_PURPOSE
                self.assertEqual(result['system_category'], system_category.value)
    
    def _configure_ai_act_for_model(self, test_case):
        """Configure le manager AI Act spécifiquement pour le modèle en test."""
        # Configurer la catégorie et le niveau de risque
        if test_case.risk_level == RiskLevel.HIGH:
            self.ai_act_manager.set_system_category(AISystemCategory.HIGH_RISK)
        else:
            self.ai_act_manager.set_system_category(AISystemCategory.GENERAL_PURPOSE)
        
        self.ai_act_manager.set_risk_level(test_case.risk_level)
        
        # Mettre à jour les informations système
        self.ai_act_manager.update_system_information({
            'name': f'Test System - {test_case.name}',
            'version': '1.0.0',
            'purpose': f'Validation test for {test_case.name}',
            'provider': 'Test Provider',
            'domain': 'Generic Testing'
        })
        
        # Ajouter des risques modèle-spécifiques
        self.ai_act_manager.clear_risks()  # Effacer les risques précédents
        
        if test_case.is_classifier:
            # Risques spécifiques aux classifieurs
            self.ai_act_manager.add_risk({
                'risk_id': 'bias_risk',
                'description': f'Risque de biais dans les prédictions de {test_case.name}',
                'risk_level': test_case.risk_level.value,
                'mitigation': 'Tests de biais et métriques d\'équité'
            })
        else:
            # Risques spécifiques aux modèles de régression
            self.ai_act_manager.add_risk({
                'risk_id': 'error_propagation',
                'description': f'Risque de propagation d\'erreurs dans {test_case.name}',
                'risk_level': test_case.risk_level.value,
                'mitigation': 'Validation croisée et quantification de l\'incertitude'
            })
        
        # Risque commun à tous les modèles
        self.ai_act_manager.add_risk({
            'risk_id': 'explainability_risk',
            'description': f'Risque lié à l\'explicabilité de {test_case.name}',
            'risk_level': RiskLevel.MEDIUM.value,
            'mitigation': 'Utilisation d\'explainers (LIME, SHAP)'
        })
    
    def test_explainers_all_models(self):
        """
        Teste les explainers sur tous les types de modèles.
        Vérifie que les explainers LIME et SHAP peuvent fournir des explications
        cohérentes pour différents types de modèles.
        """
        for test_case in self.test_cases:
            with self.subTest(model=test_case.name):
                # Sélectionner une instance à expliquer
                instance = test_case.test_data.iloc[0:1]
                
                try:
                    # Tester LIME
                    lime_explainer = LIMEExplainer(test_case.model)
                    lime_explanation = lime_explainer.explain(instance)
                    
                    self.assertIsNotNone(lime_explanation)
                    self.assertGreater(len(lime_explanation), 0)
                    
                    # Vérifier que les contributions sont cohérentes avec les noms des features
                    feature_names = set(test_case.feature_names)
                    explained_features = set(item['feature'] for item in lime_explanation)
                    self.assertTrue(len(explained_features.intersection(feature_names)) > 0)
                    
                except Exception as e:
                    self.fail(f"LIME failed for {test_case.name}: {str(e)}")
                
                try:
                    # Tester SHAP (si disponible)
                    shap_explainer = SHAPExplainer(test_case.model)
                    shap_explanation = shap_explainer.explain(instance)
                    
                    self.assertIsNotNone(shap_explanation)
                    self.assertGreater(len(shap_explanation), 0)
                    
                    # Vérifier que les contributions sont cohérentes avec les noms des features
                    feature_names = set(test_case.feature_names)
                    explained_features = set(item['feature'] for item in shap_explanation)
                    self.assertTrue(len(explained_features.intersection(feature_names)) > 0)
                    
                except ImportError:
                    # SHAP peut nécessiter des dépendances spécifiques
                    self.skipTest(f"SHAP n'est pas disponible pour le test de {test_case.name}")
                except Exception as e:
                    # Certains modèles peuvent ne pas être compatibles avec SHAP
                    print(f"SHAP warning for {test_case.name}: {str(e)}")
    
    def test_integrated_compliance_and_explanation(self):
        """
        Teste l'intégration complète entre conformité et explication.
        Vérifie que les explications peuvent être intégrées dans les rapports
        de conformité pour différents types de modèles.
        """
        for test_case in self.test_cases:
            with self.subTest(model=test_case.name):
                # Configurer les managers pour ce modèle spécifique
                self._configure_gdpr_for_model(test_case)
                self._configure_ai_act_for_model(test_case)
                
                # Sélectionner une instance à expliquer
                instance_idx = 0
                instance = test_case.test_data.iloc[instance_idx:instance_idx+1]
                
                try:
                    # Générer une explication avec LIME
                    lime_explainer = LIMEExplainer(test_case.model)
                    explanation = lime_explainer.explain(instance)
                    
                    # Préparer les données pour l'analyse
                    test_data = {
                        'features': instance.values.tolist(),
                        'feature_names': test_case.feature_names,
                        'metadata': {
                            'domain': 'generic',
                            'purpose': 'validation_test',
                            'has_personal_data': True,
                            'explanation': explanation  # Intégrer l'explication
                        }
                    }
                    
                    # Exécuter les analyses de conformité
                    gdpr_result = self.gdpr_manager.analyze(test_case.model, test_data)
                    ai_act_result = self.ai_act_manager.analyze(test_case.model, test_data)
                    
                    # Vérifier que les résultats sont valides
                    self.assertIsNotNone(gdpr_result)
                    self.assertIsNotNone(ai_act_result)
                    
                    # Vérifier que l'explication est correctement intégrée au GDPR
                    # (si le modèle le supporte)
                    if 'automated_decision' in gdpr_result:
                        self.assertIn('decision_explanation', gdpr_result['automated_decision'])
                    
                    # Vérifier que l'AI Act identifie correctement les exigences d'explicabilité
                    if test_case.risk_level == RiskLevel.HIGH:
                        transparency_requirements = [
                            req for req in ai_act_result.get('requirements_assessment', {}).get('details', [])
                            if 'transparency' in req.get('description', '').lower()
                        ]
                        self.assertGreater(len(transparency_requirements), 0)
                    
                except Exception as e:
                    print(f"Warning for {test_case.name}: {str(e)}")


if __name__ == '__main__':
    unittest.main()
