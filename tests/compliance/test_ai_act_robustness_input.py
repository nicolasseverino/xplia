"""
Tests de robustesse pour les entrées du module de conformité AI Act.

Ce module teste la résilience du module AI Act face à diverses entrées,
y compris les entrées vides, nulles, malformées ou incorrectes.
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

class TestAIActInputRobustness(unittest.TestCase):
    """Tests de robustesse pour les entrées du module AI Act."""
    
    def setUp(self):
        """Configure l'environnement de test."""
        self.ai_act_manager = AIActComplianceManager()
        
        # Modèle simulé
        self.mock_model = MagicMock()
        self.mock_model.__class__.__name__ = "RandomForestClassifier"
    
    def test_empty_data(self):
        """Teste le comportement avec des données vides."""
        # Données vides
        empty_data = {}
        
        # Vérifier que l'analyse ne plante pas
        result = self.ai_act_manager.analyze(self.mock_model, empty_data)
        
        # L'analyse devrait renvoyer un résultat même avec des données vides
        self.assertIsNotNone(result)
        self.assertIn('status', result)
    
    def test_none_data(self):
        """Teste le comportement avec des données None."""
        # Vérifier que l'exception est levée correctement
        with self.assertRaises(ValueError):
            self.ai_act_manager.analyze(self.mock_model, None)
    
    def test_none_model(self):
        """Teste le comportement avec un modèle None."""
        # Données valides
        valid_data = {'features': [[1, 2, 3]]}
        
        # Vérifier que l'exception est levée correctement
        with self.assertRaises(ValueError):
            self.ai_act_manager.analyze(None, valid_data)
    
    def test_malformed_data(self):
        """Teste le comportement avec des données malformées."""
        # Données avec des valeurs inattendues
        malformed_data = {
            'features': 'not_a_list',
            'metadata': 42
        }
        
        # Vérifier que l'analyse gère correctement les données malformées
        result = self.ai_act_manager.analyze(self.mock_model, malformed_data)
        
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
            self.ai_act_manager.analyze(not_a_model, valid_data)


class TestAIActEnumRobustness(unittest.TestCase):
    """Tests de robustesse pour les énumérations du module AI Act."""
    
    def setUp(self):
        """Configure l'environnement de test."""
        self.ai_act_manager = AIActComplianceManager()
    
    def test_invalid_risk_level(self):
        """Teste le comportement avec un niveau de risque invalide."""
        # Tenter de définir un niveau de risque invalide
        with self.assertRaises(ValueError):
            self.ai_act_manager.set_risk_level("not_a_risk_level")
    
    def test_string_risk_level(self):
        """Teste le comportement avec une chaîne pour le niveau de risque."""
        # Définir un niveau de risque via une chaîne valide
        self.ai_act_manager.set_risk_level("high")
        
        # Vérifier que le niveau de risque a été correctement converti
        self.assertEqual(self.ai_act_manager.risk_level, RiskLevel.HIGH)
        
        # Tester avec une autre chaîne valide
        self.ai_act_manager.set_risk_level("low")
        self.assertEqual(self.ai_act_manager.risk_level, RiskLevel.LOW)
    
    def test_invalid_system_category(self):
        """Teste le comportement avec une catégorie de système invalide."""
        # Tenter de définir une catégorie de système invalide
        with self.assertRaises(ValueError):
            self.ai_act_manager.set_system_category("not_a_category")
    
    def test_string_system_category(self):
        """Teste le comportement avec une chaîne pour la catégorie de système."""
        # Définir une catégorie via une chaîne valide
        self.ai_act_manager.set_system_category("high_risk")
        
        # Vérifier que la catégorie a été correctement convertie
        self.assertEqual(self.ai_act_manager.system_category, AISystemCategory.HIGH_RISK)
        
        # Tester avec une autre chaîne valide
        self.ai_act_manager.set_system_category("minimal_risk")
        self.assertEqual(self.ai_act_manager.system_category, AISystemCategory.MINIMAL_RISK)
    
    def test_invalid_requirement_status(self):
        """Teste le comportement avec un statut d'exigence invalide."""
        # Configurer un système à haut risque pour avoir des exigences
        self.ai_act_manager.set_system_category(AISystemCategory.HIGH_RISK)
        
        # Récupérer une exigence
        requirements = self.ai_act_manager.get_applicable_requirements()
        
        if not requirements:
            self.skipTest("Aucune exigence disponible pour le test")
        
        req_id = requirements[0]['id']
        
        # Tenter de définir un statut invalide
        with self.assertRaises(ValueError):
            self.ai_act_manager.update_requirement_status(
                req_id, "not_a_status", "Détails de test"
            )


class TestAIActDecisionLogRobustness(unittest.TestCase):
    """Tests de robustesse pour le journal des décisions du module AI Act."""
    
    def setUp(self):
        """Configure l'environnement de test."""
        self.ai_act_manager = AIActComplianceManager()
    
    def test_log_decision_missing_fields(self):
        """Teste l'enregistrement d'une décision avec des champs manquants."""
        # Tester avec uniquement les données d'entrée
        self.ai_act_manager.log_decision(
            input_data={'feature': 0.5},
            output=None,
            explanation=None,
            user_id=None
        )
        
        # La décision devrait être enregistrée malgré les champs manquants
        decisions = self.ai_act_manager.decision_log.get_decisions()
        self.assertEqual(len(decisions), 1)
        self.assertIn('input_data', decisions[0])
        self.assertEqual(decisions[0]['input_data'], {'feature': 0.5})
    
    def test_log_decision_invalid_explanation(self):
        """Teste l'enregistrement d'une décision avec une explication invalide."""
        # Utiliser une explication qui n'est pas un objet avec to_dict()
        invalid_explanation = "This is not a valid explanation object"
        
        # Enregistrer la décision avec une explication invalide
        self.ai_act_manager.log_decision(
            input_data={'feature': 0.5},
            output={'prediction': 0.8},
            explanation=invalid_explanation,
            user_id="test_user"
        )
        
        # La décision devrait être enregistrée malgré l'explication invalide
        decisions = self.ai_act_manager.decision_log.get_decisions()
        self.assertEqual(len(decisions), 1)
        self.assertIn('input_data', decisions[0])
        self.assertIn('output', decisions[0])
        # L'explication devrait être stockée comme chaîne ou ignorée
        if 'explanation' in decisions[0]:
            self.assertIsNotNone(decisions[0]['explanation'])
    
    def test_get_decisions_with_invalid_user(self):
        """Teste la récupération des décisions avec un ID utilisateur invalide."""
        # Ajouter quelques décisions
        self.ai_act_manager.log_decision(
            input_data={'feature': 0.5},
            output={'prediction': 0.8},
            explanation=None,
            user_id="valid_user"
        )
        
        # Récupérer les décisions pour un utilisateur inexistant
        decisions = self.ai_act_manager.decision_log.get_decisions_by_user("non_existent_user")
        
        # La liste devrait être vide et non None
        self.assertIsNotNone(decisions)
        self.assertEqual(len(decisions), 0)


if __name__ == '__main__':
    unittest.main()
