"""
Tests de robustesse pour le module de conformité GDPR.

Ce module teste la résilience et la stabilité du module GDPR dans diverses
conditions extrêmes, erronées ou limites.
"""

import unittest
from unittest.mock import MagicMock, patch
import tempfile
import os
import json

from xplia.compliance.explanation_rights import (
    GDPRComplianceManager,
    DataCategory,
    ProcessingRecord,
    ProcessingPurpose,
    LegalBasis
)

class TestGDPRInputRobustness(unittest.TestCase):
    """Tests de robustesse pour les entrées du module GDPR."""
    
    def setUp(self):
        """Configure l'environnement de test."""
        self.gdpr_manager = GDPRComplianceManager()
        
        # Modèle simulé
        self.mock_model = MagicMock()
        self.mock_model.__class__.__name__ = "RandomForestClassifier"
    
    def test_empty_data(self):
        """Teste le comportement avec des données vides."""
        # Données vides
        empty_data = {}
        
        # Vérifier que l'analyse ne plante pas
        result = self.gdpr_manager.analyze(self.mock_model, empty_data)
        
        # L'analyse devrait renvoyer un résultat même avec des données vides
        self.assertIsNotNone(result)
        self.assertIn('status', result)
    
    def test_none_data(self):
        """Teste le comportement avec des données None."""
        # Vérifier que l'exception est levée correctement
        with self.assertRaises(ValueError):
            self.gdpr_manager.analyze(self.mock_model, None)
    
    def test_none_model(self):
        """Teste le comportement avec un modèle None."""
        # Données valides
        valid_data = {'features': [[1, 2, 3]]}
        
        # Vérifier que l'exception est levée correctement
        with self.assertRaises(ValueError):
            self.gdpr_manager.analyze(None, valid_data)
    
    def test_malformed_data(self):
        """Teste le comportement avec des données malformées."""
        # Données avec des valeurs inattendues
        malformed_data = {
            'features': 'not_a_list',
            'metadata': 42
        }
        
        # Vérifier que l'analyse gère correctement les données malformées
        result = self.gdpr_manager.analyze(self.mock_model, malformed_data)
        
        self.assertIsNotNone(result)
        self.assertIn('status', result)
        # Statut d'erreur ou avertissement attendu
        self.assertIn(result['status'], ['error', 'warning', 'non_compliant'])
    
    def test_invalid_model_type(self):
        """Teste le comportement avec un type de modèle invalide."""
        # Objet qui n'est pas un modèle ML
        not_a_model = "This is a string, not a model"
        valid_data = {'features': [[1, 2, 3]]}
        
        # Vérifier que l'analyse gère correctement un type de modèle invalide
        with self.assertRaises((ValueError, TypeError)):
            self.gdpr_manager.analyze(not_a_model, valid_data)


class TestGDPRRegistryRobustness(unittest.TestCase):
    """Tests de robustesse pour le registre de traitement GDPR."""
    
    def setUp(self):
        """Configure l'environnement de test."""
        self.gdpr_manager = GDPRComplianceManager()
    
    def test_duplicate_processing_records(self):
        """Teste le comportement avec des enregistrements de traitement dupliqués."""
        # Ajouter un enregistrement de traitement
        self.gdpr_manager.data_processing_registry.register_processing(
            name="Test Processing",
            description="Description for test",
            categories=[DataCategory.PERSONAL]
        )
        
        # Ajouter un autre enregistrement avec le même nom (devrait mettre à jour, pas planter)
        self.gdpr_manager.data_processing_registry.register_processing(
            name="Test Processing",
            description="Updated description",
            categories=[DataCategory.FINANCIAL]
        )
        
        # Vérifier que l'enregistrement a été mis à jour
        activities = self.gdpr_manager.data_processing_registry.get_processing_activities()
        matching = [a for a in activities if a['name'] == "Test Processing"]
        
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]['description'], "Updated description")
        self.assertEqual(matching[0]['categories'], [DataCategory.FINANCIAL.value])
    
    def test_invalid_data_category(self):
        """Teste le comportement avec une catégorie de données invalide."""
        # Tenter d'ajouter un enregistrement avec une catégorie invalide
        with self.assertRaises(ValueError):
            self.gdpr_manager.data_processing_registry.register_processing(
                name="Invalid Category Test",
                description="Test",
                categories=["invalid_category"]
            )
    
    def test_invalid_purpose(self):
        """Teste le comportement avec un objectif de traitement invalide."""
        # Tenter d'ajouter un enregistrement avec un objectif invalide
        with self.assertRaises(ValueError):
            self.gdpr_manager.data_processing_registry.register_processing(
                name="Invalid Purpose Test",
                description="Test",
                categories=[DataCategory.PERSONAL],
                purposes=["invalid_purpose"]
            )
    
    def test_invalid_legal_basis(self):
        """Teste le comportement avec une base légale invalide."""
        # Tenter d'ajouter un enregistrement avec une base légale invalide
        with self.assertRaises(ValueError):
            self.gdpr_manager.data_processing_registry.register_processing(
                name="Invalid Basis Test",
                description="Test",
                categories=[DataCategory.PERSONAL],
                purposes=[ProcessingPurpose.LEGITIMATE_INTEREST],
                legal_basis="invalid_basis"
            )
    
    def test_bulk_update_with_invalid_records(self):
        """Teste la mise à jour en masse avec des enregistrements invalides mélangés."""
        # Préparer une liste avec des enregistrements valides et invalides
        records = [
            {
                'name': 'Valid Record 1',
                'description': 'Valid record',
                'categories': [DataCategory.PERSONAL.value]
            },
            {
                'name': 'Invalid Record',
                'description': 'Invalid record',
                'categories': ["invalid_category"]
            },
            {
                'name': 'Valid Record 2',
                'description': 'Another valid record',
                'categories': [DataCategory.FINANCIAL.value]
            }
        ]
        
        # Effectuer la mise à jour en masse
        result = self.gdpr_manager.data_processing_registry.bulk_update(records)
        
        # Vérifier que les enregistrements valides ont été ajoutés
        # et que les invalides ont été ignorés
        self.assertEqual(result['success_count'], 2)
        self.assertEqual(result['failure_count'], 1)
        
        activities = self.gdpr_manager.data_processing_registry.get_processing_activities()
        self.assertEqual(len(activities), 2)  # Seuls les deux valides


class TestGDPRExportImportRobustness(unittest.TestCase):
    """Tests de robustesse pour l'exportation et l'importation des données GDPR."""
    
    def setUp(self):
        """Configure l'environnement de test."""
        self.gdpr_manager = GDPRComplianceManager()
        
        # Ajouter quelques données de test
        self.gdpr_manager.data_processing_registry.register_processing(
            name="Export Test",
            description="Test for export",
            categories=[DataCategory.PERSONAL],
            purposes=[ProcessingPurpose.CONSENT],
            legal_basis=LegalBasis.CONSENT
        )
        
        self.gdpr_manager.set_dpo_contact({
            'name': 'Test DPO',
            'email': 'dpo@test.com'
        })
        
        # Simuler un utilisateur
        mock_explanation = MagicMock()
        mock_explanation.to_dict.return_value = {'importance': [0.5, 0.5]}
        self.gdpr_manager.request_log.add(
            user_id="test_user",
            data_requested={'features': [1, 2]},
            explanation_provided=mock_explanation
        )
    
    def test_export_import(self):
        """Teste l'exportation puis l'importation des données."""
        # Exporter les données
        export_data = self.gdpr_manager.export_data()
        self.assertIsNotNone(export_data)
        
        # Créer un nouveau manager et importer les données
        new_manager = GDPRComplianceManager()
        result = new_manager.import_data(export_data)
        
        # Vérifier que l'import a réussi
        self.assertTrue(result)
        
        # Vérifier que les données ont été correctement importées
        activities = new_manager.data_processing_registry.get_processing_activities()
        self.assertEqual(len(activities), 1)
        self.assertEqual(activities[0]['name'], "Export Test")
        
        dpo = new_manager.get_dpo_contact()
        self.assertEqual(dpo['name'], "Test DPO")
        
        requests = new_manager.request_log.get_requests()
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]['user_id'], "test_user")
    
    def test_import_malformed_data(self):
        """Teste l'importation avec des données malformées."""
        # Données d'import malformées
        malformed_data = {
            'registry': 'not_a_dict',
            'dpo_contact': 42,
            'request_log': 'invalid'
        }
        
        # Tenter d'importer les données
        result = self.gdpr_manager.import_data(malformed_data)
        
        # L'importation devrait échouer mais ne pas planter
        self.assertFalse(result)
        
        # Les données d'origine devraient rester intactes
        activities = self.gdpr_manager.data_processing_registry.get_processing_activities()
        self.assertEqual(len(activities), 1)
        self.assertEqual(activities[0]['name'], "Export Test")
    
    def test_import_partial_data(self):
        """Teste l'importation avec des données partielles."""
        # Données d'import partielles (seulement la partie registry)
        partial_data = {
            'registry': {
                'activities': [
                    {
                        'name': 'Partial Import',
                        'description': 'Partial data test',
                        'categories': [DataCategory.FINANCIAL.value]
                    }
                ]
            }
        }
        
        # Importer les données partielles
        result = self.gdpr_manager.import_data(partial_data)
        
        # L'importation devrait réussir malgré les données partielles
        self.assertTrue(result)
        
        # Vérifier que les données partielles ont été importées
        activities = self.gdpr_manager.data_processing_registry.get_processing_activities()
        self.assertEqual(len(activities), 1)
        self.assertEqual(activities[0]['name'], "Partial Import")
        
        # Les autres composants devraient rester inchangés ou initialisés
        dpo = self.gdpr_manager.get_dpo_contact()
        self.assertIsNotNone(dpo)


class TestGDPRErrorHandling(unittest.TestCase):
    """Tests pour la gestion des erreurs dans le module GDPR."""
    
    def setUp(self):
        """Configure l'environnement de test."""
        self.gdpr_manager = GDPRComplianceManager()
    
    @patch('xplia.compliance.explanation_rights.DataProcessingRegistry.register_processing')
    def test_exception_handling(self, mock_register):
        """Teste la gestion des exceptions lors de l'enregistrement d'un traitement."""
        # Simuler une exception dans le registre
        mock_register.side_effect = Exception("Simulated error")
        
        # Tenter d'enregistrer un traitement
        with self.assertRaises(Exception):
            self.gdpr_manager.data_processing_registry.register_processing(
                name="Error Test",
                description="Test description",
                categories=[DataCategory.PERSONAL]
            )
    
    @patch('xplia.compliance.explanation_rights.GDPRComplianceManager._assess_processing')
    def test_analyze_error_handling(self, mock_assess):
        """Teste la gestion des erreurs lors de l'analyse."""
        # Simuler une exception dans la méthode d'évaluation
        mock_assess.side_effect = Exception("Simulated assessment error")
        
        # Effectuer l'analyse
        mock_model = MagicMock()
        mock_data = {'features': [[1, 2, 3]]}
        
        # L'analyse devrait capturer l'exception et renvoyer un statut d'erreur
        result = self.gdpr_manager.analyze(mock_model, mock_data)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['status'], 'error')
        self.assertIn('error_message', result)
    
    @patch('xplia.compliance.explanation_rights.RequestLog.add')
    def test_log_request_error_handling(self, mock_add):
        """Teste la gestion des erreurs lors de l'ajout d'une requête au journal."""
        # Simuler une exception dans l'ajout au journal
        mock_add.side_effect = Exception("Simulated log error")
        
        # Tenter d'ajouter une requête
        try:
            self.gdpr_manager.request_log.add(
                user_id="test_user",
                data_requested={'test': 'data'},
                explanation_provided=None
            )
            self.fail("Exception not raised")
        except Exception as e:
            self.assertIn("Simulated log error", str(e))


if __name__ == '__main__':
    unittest.main()
