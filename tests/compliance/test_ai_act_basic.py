"""
Tests unitaires pour les fonctionnalités de base du module de conformité AI Act.

Ce module teste les fonctionnalités fondamentales du gestionnaire AI Act,
y compris l'initialisation, la configuration et l'analyse de base.
"""

import unittest
from unittest.mock import MagicMock, patch
import datetime
import json
import tempfile
import os

from xplia.compliance.ai_act import (
    AIActComplianceManager,
    RiskLevel,
    AISystemCategory,
    RequirementStatus
)

class TestAIActBasicFunctionality(unittest.TestCase):
    """Tests des fonctionnalités de base du module AI Act."""
    
    def setUp(self):
        """Configure l'environnement de test."""
        self.ai_act_manager = AIActComplianceManager()
        
        # Simuler un modèle
        self.mock_model = MagicMock()
        self.mock_model.__class__.__name__ = "NeuralNetwork"
        
        # Simuler des données
        self.mock_data = {
            'features': [[0.5, 0.7, 0.2, 0.1]],
            'metadata': {'domain': 'healthcare'}
        }
    
    def test_initialization(self):
        """Vérifie que le gestionnaire est correctement initialisé."""
        self.assertIsNotNone(self.ai_act_manager)
        self.assertIsNotNone(self.ai_act_manager.decision_log)
        self.assertEqual(self.ai_act_manager.risk_level, RiskLevel.MEDIUM)
    
    def test_set_system_category(self):
        """Teste la définition de la catégorie du système IA."""
        self.ai_act_manager.set_system_category(AISystemCategory.HIGH_RISK)
        self.assertEqual(self.ai_act_manager.system_category, AISystemCategory.HIGH_RISK)
        
        self.ai_act_manager.set_system_category(AISystemCategory.LIMITED_RISK)
        self.assertEqual(self.ai_act_manager.system_category, AISystemCategory.LIMITED_RISK)
    
    def test_set_risk_level(self):
        """Teste la définition du niveau de risque."""
        self.ai_act_manager.set_risk_level(RiskLevel.HIGH)
        self.assertEqual(self.ai_act_manager.risk_level, RiskLevel.HIGH)
        
        self.ai_act_manager.set_risk_level('low')
        self.assertEqual(self.ai_act_manager.risk_level, RiskLevel.LOW)
        
        # Test avec une valeur invalide
        with self.assertRaises(ValueError):
            self.ai_act_manager.set_risk_level('invalid_level')
    
    def test_set_use_case(self):
        """Teste la définition du cas d'utilisation."""
        test_use_case = "healthcare_diagnosis"
        self.ai_act_manager.set_use_case(test_use_case)
        self.assertEqual(self.ai_act_manager.use_case, test_use_case)
        
        # Vérifier que le cas d'utilisation affecte la catégorie et le niveau de risque
        self.ai_act_manager.set_use_case("biometric_identification")
        self.assertEqual(self.ai_act_manager.system_category, AISystemCategory.HIGH_RISK)
    
    def test_analyze(self):
        """Teste l'analyse de conformité AI Act."""
        with patch.object(self.ai_act_manager, '_assess_model') as mock_assess:
            mock_assess.return_value = {
                'risk_level': RiskLevel.MEDIUM.value,
                'applicable_requirements': [],
                'identified_risks': []
            }
            
            result = self.ai_act_manager.analyze(self.mock_model, self.mock_data)
            
            self.assertIn('status', result)
            self.assertIn('timestamp', result)
            self.assertIn('details', result)
            mock_assess.assert_called_once()
    
    def test_log_decision(self):
        """Teste l'enregistrement d'une décision."""
        mock_input = {'feature1': 0.5, 'feature2': 0.3}
        mock_output = {'prediction': 'class_a', 'confidence': 0.8}
        mock_explanation = MagicMock()
        mock_explanation.to_dict.return_value = {'importance': [0.7, 0.3]}
        
        self.ai_act_manager.log_decision(
            input_data=mock_input,
            output=mock_output,
            explanation=mock_explanation,
            user_id="test_user"
        )
        
        # Vérifier que la décision a été enregistrée
        decisions = self.ai_act_manager.decision_log.get_decisions()
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0]['user_id'], "test_user")
        self.assertEqual(decisions[0]['output'], mock_output)


class TestRequirementsManagement(unittest.TestCase):
    """Tests pour la gestion des exigences de l'AI Act."""
    
    def setUp(self):
        """Configure l'environnement de test."""
        self.ai_act_manager = AIActComplianceManager()
    
    def test_get_applicable_requirements(self):
        """Teste la récupération des exigences applicables."""
        # Définir une catégorie à haut risque pour avoir des exigences
        self.ai_act_manager.set_system_category(AISystemCategory.HIGH_RISK)
        
        requirements = self.ai_act_manager.get_applicable_requirements()
        
        self.assertIsNotNone(requirements)
        self.assertIsInstance(requirements, list)
        self.assertGreater(len(requirements), 0)
        
        # Vérifier la structure d'une exigence
        if requirements:
            req = requirements[0]
            self.assertIn('id', req)
            self.assertIn('description', req)
            self.assertIn('category', req)
    
    def test_update_requirement_status(self):
        """Teste la mise à jour du statut d'une exigence."""
        # Définir une catégorie à haut risque
        self.ai_act_manager.set_system_category(AISystemCategory.HIGH_RISK)
        
        # Récupérer les exigences
        requirements = self.ai_act_manager.get_applicable_requirements()
        
        if not requirements:
            self.skipTest("Aucune exigence disponible pour le test")
        
        # Mettre à jour le statut d'une exigence
        req_id = requirements[0]['id']
        result = self.ai_act_manager.update_requirement_status(
            req_id,
            RequirementStatus.COMPLIANT,
            "Mise à jour pour le test"
        )
        
        self.assertTrue(result)
        
        # Vérifier que le statut a été mis à jour
        updated_requirements = self.ai_act_manager.get_applicable_requirements()
        updated_req = next((r for r in updated_requirements if r['id'] == req_id), None)
        
        self.assertIsNotNone(updated_req)
        self.assertEqual(updated_req['status'], RequirementStatus.COMPLIANT.value)
        self.assertEqual(updated_req['details'], "Mise à jour pour le test")
    
    def test_bulk_update_requirements(self):
        """Teste la mise à jour en masse des statuts d'exigences."""
        self.ai_act_manager.set_system_category(AISystemCategory.HIGH_RISK)
        requirements = self.ai_act_manager.get_applicable_requirements()
        
        if len(requirements) < 2:
            self.skipTest("Pas assez d'exigences disponibles pour le test")
        
        # Préparer les mises à jour
        updates = [
            {
                'id': requirements[0]['id'],
                'status': RequirementStatus.COMPLIANT,
                'details': "Premier test"
            },
            {
                'id': requirements[1]['id'],
                'status': RequirementStatus.PARTIAL,
                'details': "Second test"
            }
        ]
        
        result = self.ai_act_manager.bulk_update_requirements(updates)
        
        self.assertEqual(result['success_count'], 2)
        self.assertEqual(result['failure_count'], 0)
        
        # Vérifier les mises à jour
        updated_reqs = self.ai_act_manager.get_applicable_requirements()
        first_req = next((r for r in updated_reqs if r['id'] == requirements[0]['id']), None)
        second_req = next((r for r in updated_reqs if r['id'] == requirements[1]['id']), None)
        
        self.assertEqual(first_req['status'], RequirementStatus.COMPLIANT.value)
        self.assertEqual(second_req['status'], RequirementStatus.PARTIAL.value)


if __name__ == '__main__':
    unittest.main()
