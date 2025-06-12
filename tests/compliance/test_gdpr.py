"""
Tests unitaires pour le module de conformité GDPR (RGPD).

Ce module contient des tests complets pour toutes les fonctionnalités du
module de conformité GDPR, y compris les droits d'accès, la gestion des
demandes d'explication, et l'audit.
"""

import unittest
from unittest.mock import MagicMock, patch
import datetime
import json
import tempfile
import os

from xplia.compliance.explanation_rights import (
    GDPRComplianceManager,
    DataSubjectRightsManager,
    DataCategory,
    LegalBasis,
    ProcessingActivity,
    DPIAManager,
    DataBreachManager
)

class TestGDPRBasicFunctionality(unittest.TestCase):
    """Teste les fonctionnalités de base du module GDPR."""
    
    def setUp(self):
        """Configure l'environnement de test."""
        self.gdpr_manager = GDPRComplianceManager()
        
        # Simuler un modèle
        self.mock_model = MagicMock()
        self.mock_model.__class__.__name__ = "RandomForestClassifier"
        
        # Simuler des données
        self.mock_data = {
            'personal_info': [{'name': 'John Doe', 'age': 30}],
            'features': [[0.5, 0.7, 0.2, 0.1]]
        }
    
    def test_initialization(self):
        """Vérifie que le gestionnaire est correctement initialisé."""
        self.assertIsNotNone(self.gdpr_manager)
        self.assertIsNotNone(self.gdpr_manager.request_log)
        self.assertIsNotNone(self.gdpr_manager.data_processing_registry)
    
    def test_set_dpo_contact(self):
        """Teste la définition des informations de contact DPO."""
        dpo_info = {
            'name': 'Jane Smith',
            'email': 'jane.smith@company.com',
            'phone': '+33123456789'
        }
        
        self.gdpr_manager.set_dpo_contact(dpo_info)
        self.assertEqual(self.gdpr_manager.dpo_contact, dpo_info)
    
    def test_analyze(self):
        """Teste l'analyse de conformité GDPR."""
        with patch.object(self.gdpr_manager, '_check_data_processing') as mock_check:
            mock_check.return_value = {'status': 'compliant', 'details': []}
            
            result = self.gdpr_manager.analyze(self.mock_model, self.mock_data)
            
            self.assertIn('status', result)
            self.assertIn('timestamp', result)
            self.assertIn('details', result)
            mock_check.assert_called_once()


class TestDataProcessingRegistry(unittest.TestCase):
    """Teste les fonctionnalités du registre de traitement des données."""
    
    def setUp(self):
        """Configure l'environnement de test."""
        self.gdpr_manager = GDPRComplianceManager()
        self.registry = self.gdpr_manager.data_processing_registry
    
    def test_register_processing(self):
        """Teste l'enregistrement d'une activité de traitement."""
        activity = self.registry.register_processing(
            name="Système de scoring",
            description="Calcul du score de crédit",
            categories=[DataCategory.FINANCIAL, DataCategory.PERSONAL],
            legal_basis=LegalBasis.LEGITIMATE_INTEREST,
            retention_period=36
        )
        
        self.assertIsInstance(activity, ProcessingActivity)
        self.assertEqual(activity.name, "Système de scoring")
        self.assertEqual(activity.legal_basis, LegalBasis.LEGITIMATE_INTEREST)
        self.assertEqual(len(activity.categories), 2)
    
    def test_get_processing_activities(self):
        """Teste la récupération des activités de traitement."""
        # Enregistrer deux activités
        self.registry.register_processing(
            name="Activité 1", 
            description="Description 1",
            categories=[DataCategory.FINANCIAL]
        )
        
        self.registry.register_processing(
            name="Activité 2", 
            description="Description 2",
            categories=[DataCategory.HEALTH]
        )
        
        activities = self.registry.get_processing_activities()
        
        self.assertEqual(len(activities), 2)
        self.assertEqual(activities[0].name, "Activité 1")
        self.assertEqual(activities[1].name, "Activité 2")
    
    def test_activity_to_dict(self):
        """Teste la conversion d'une activité en dictionnaire."""
        activity = self.registry.register_processing(
            name="Test Activity",
            description="Test Description",
            categories=[DataCategory.PERSONAL],
            legal_basis=LegalBasis.CONSENT,
            retention_period=12
        )
        
        activity_dict = activity.to_dict()
        
        self.assertEqual(activity_dict["name"], "Test Activity")
        self.assertEqual(activity_dict["description"], "Test Description")
        self.assertEqual(activity_dict["legal_basis"], LegalBasis.CONSENT.value)
        self.assertEqual(activity_dict["retention_period"], 12)


class TestDataSubjectRights(unittest.TestCase):
    """Teste les fonctionnalités liées aux droits des personnes concernées."""
    
    def setUp(self):
        """Configure l'environnement de test."""
        self.rights_manager = DataSubjectRightsManager()
    
    def test_handle_access_request(self):
        """Teste le traitement d'une demande d'accès aux données."""
        # Simuler des données pour un utilisateur
        self.rights_manager.user_data["user123"] = {
            "personal_info": {"name": "John Doe", "email": "john@example.com"},
            "processing_history": [
                {"timestamp": "2023-01-01", "action": "model_prediction", "result": "approved"}
            ]
        }
        
        response = self.rights_manager.handle_access_request("user123")
        
        self.assertIsNotNone(response)
        self.assertIn("personal_info", response)
        self.assertEqual(response["personal_info"]["name"], "John Doe")
        self.assertIn("processing_history", response)
    
    def test_handle_erasure_request(self):
        """Teste le traitement d'une demande d'effacement."""
        # Simuler des données pour un utilisateur
        self.rights_manager.user_data["user456"] = {
            "personal_info": {"name": "Jane Smith", "email": "jane@example.com"}
        }
        
        response = self.rights_manager.handle_erasure_request("user456")
        
        self.assertTrue(response["success"])
        self.assertNotIn("user456", self.rights_manager.user_data)
    
    def test_handle_explanation_request(self):
        """Teste le traitement d'une demande d'explication."""
        with patch.object(self.rights_manager, '_generate_explanation') as mock_explain:
            mock_explain.return_value = {
                "explanation": "Cette décision est basée sur votre historique de crédit et votre revenu",
                "factors": [{"name": "revenu", "importance": 0.7}, {"name": "historique", "importance": 0.3}]
            }
            
            response = self.rights_manager.handle_explanation_request(
                "user789",
                decision_id="decision123"
            )
            
            self.assertIsNotNone(response)
            self.assertIn("explanation", response)
            self.assertIn("factors", response)
            mock_explain.assert_called_once_with("decision123")


class TestDPIA(unittest.TestCase):
    """Teste les fonctionnalités liées à l'analyse d'impact (DPIA)."""
    
    def setUp(self):
        """Configure l'environnement de test."""
        self.gdpr_manager = GDPRComplianceManager()
        
    def test_create_dpia(self):
        """Teste la création d'une DPIA."""
        dpia = self.gdpr_manager.create_dpia(
            system_name="Système de recommandation",
            description="Système recommandant des produits aux utilisateurs",
            processing_purpose="Marketing personnalisé",
            necessity_assessment="Nécessaire pour améliorer l'expérience utilisateur"
        )
        
        self.assertIsInstance(dpia, DPIAManager)
        self.assertEqual(dpia.system_name, "Système de recommandation")
        self.assertEqual(dpia.processing_purpose, "Marketing personnalisé")
        
    def test_dpia_risk_assessment(self):
        """Teste l'évaluation des risques dans une DPIA."""
        dpia = self.gdpr_manager.create_dpia(
            system_name="Test System",
            description="Description for testing"
        )
        
        # Ajouter des risques
        dpia.add_risk({
            "description": "Risque de profilage excessif",
            "likelihood": "high",
            "impact": "medium",
            "mitigation": "Limiter les catégories de données collectées"
        })
        
        dpia.add_risk({
            "description": "Réutilisation des données",
            "likelihood": "low",
            "impact": "high",
            "mitigation": "Suppression automatique après la période de rétention"
        })
        
        risks = dpia.get_risks()
        
        self.assertEqual(len(risks), 2)
        self.assertEqual(risks[0]["likelihood"], "high")
        self.assertEqual(risks[1]["impact"], "high")
        
    def test_dpia_export(self):
        """Teste l'exportation d'une DPIA."""
        dpia = self.gdpr_manager.create_dpia(
            system_name="Export Test",
            description="Testing export functionality"
        )
        
        dpia.add_risk({"description": "Test Risk", "likelihood": "medium", "impact": "medium"})
        
        export_data = dpia.export_dpia()
        
        self.assertIsNotNone(export_data)
        self.assertEqual(export_data["system_name"], "Export Test")
        self.assertIn("risks", export_data)
        self.assertEqual(len(export_data["risks"]), 1)


class TestDataBreachManagement(unittest.TestCase):
    """Teste les fonctionnalités de gestion des violations de données."""
    
    def setUp(self):
        """Configure l'environnement de test."""
        self.breach_manager = DataBreachManager()
        
    def test_report_breach(self):
        """Teste le signalement d'une violation."""
        breach = self.breach_manager.report_breach(
            description="Fuite de données client",
            affected_data_categories=[DataCategory.PERSONAL, DataCategory.FINANCIAL],
            estimated_affected_count=1000,
            date_discovered=datetime.datetime.now(),
            nature_of_breach="unauthorized_access"
        )
        
        self.assertIsNotNone(breach)
        self.assertEqual(breach["description"], "Fuite de données client")
        self.assertEqual(breach["estimated_affected_count"], 1000)
        self.assertEqual(breach["nature_of_breach"], "unauthorized_access")
        self.assertIn(breach["id"], self.breach_manager.breaches)
        
    def test_update_breach(self):
        """Teste la mise à jour d'une violation."""
        breach = self.breach_manager.report_breach(
            description="Initial description",
            affected_data_categories=[DataCategory.PERSONAL],
            estimated_affected_count=100
        )
        
        # Mettre à jour la violation
        updated_breach = self.breach_manager.update_breach(
            breach_id=breach["id"],
            description="Updated description",
            estimated_affected_count=200
        )
        
        self.assertEqual(updated_breach["description"], "Updated description")
        self.assertEqual(updated_breach["estimated_affected_count"], 200)
        
    def test_generate_notification(self):
        """Teste la génération d'une notification de violation."""
        breach = self.breach_manager.report_breach(
            description="Test breach",
            affected_data_categories=[DataCategory.PERSONAL],
            estimated_affected_count=5000,
            high_risk=True
        )
        
        with patch.object(self.breach_manager, '_should_notify_authority') as mock_should_notify:
            mock_should_notify.return_value = True
            
            notification = self.breach_manager.generate_notification(breach["id"])
            
            self.assertIsNotNone(notification)
            self.assertIn("authority_notice", notification)
            self.assertIn("data_subject_notice", notification)
            mock_should_notify.assert_called_once()


class TestRobustness(unittest.TestCase):
    """Tests de robustesse pour le module GDPR."""
    
    def setUp(self):
        """Configure l'environnement de test."""
        self.gdpr_manager = GDPRComplianceManager()
    
    def test_empty_data(self):
        """Teste le comportement avec des données vides."""
        result = self.gdpr_manager.analyze(MagicMock(), {})
        self.assertIsNotNone(result)
    
    def test_invalid_parameters(self):
        """Teste le comportement avec des paramètres invalides."""
        with self.assertRaises(ValueError):
            self.gdpr_manager.set_dpo_contact("invalid_contact")
            
        with self.assertRaises(ValueError):
            self.gdpr_manager.data_processing_registry.register_processing(
                name=123,  # Devrait être une chaîne
                description="Description"
            )
            
    def test_file_operations(self):
        """Teste les opérations de fichier (export/import)."""
        # Enregistrer une activité de traitement
        self.gdpr_manager.data_processing_registry.register_processing(
            name="Test Activity",
            description="Test Description",
            categories=[DataCategory.PERSONAL]
        )
        
        # Exporter les données
        export_data = self.gdpr_manager.export_data()
        
        # Créer un fichier temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp:
            temp_path = temp.name
            json.dump(export_data, temp)
        
        try:
            # Créer un nouveau gestionnaire et importer les données
            new_manager = GDPRComplianceManager()
            
            with open(temp_path, 'r') as f:
                import_data = json.load(f)
            
            new_manager.import_data(import_data)
            
            # Vérifier que les données ont été importées correctement
            activities = new_manager.data_processing_registry.get_processing_activities()
            self.assertEqual(len(activities), 1)
            self.assertEqual(activities[0].name, "Test Activity")
            
        finally:
            # Nettoyer
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestIntegrationWithExplainers(unittest.TestCase):
    """Tests d'intégration avec les explainers."""
    
    @patch('xplia.core.factory.ExplainerFactory')
    def test_explanation_integration(self, mock_factory):
        """Teste l'intégration avec les explainers."""
        # Configurer les mocks
        mock_explainer = MagicMock()
        mock_explanation = MagicMock()
        mock_factory.return_value.create_explainer.return_value = mock_explainer
        mock_explainer.explain_model.return_value = mock_explanation
        
        # Gestionnaire GDPR
        gdpr_manager = GDPRComplianceManager()
        
        # Simuler une explication
        from xplia.api import explain_model
        
        model = MagicMock()
        data = {"features": [[1, 2, 3]]}
        
        # Ajouter un hook pour enregistrer l'explication
        with patch.object(gdpr_manager, 'request_log') as mock_log:
            explain_model(model, data, user_id="test_user")
            mock_log.add.assert_called_once()


if __name__ == '__main__':
    unittest.main()
