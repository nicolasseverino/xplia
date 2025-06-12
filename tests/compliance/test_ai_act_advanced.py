"""
Tests unitaires pour les fonctionnalités avancées du module de conformité AI Act.

Ce module teste les fonctionnalités avancées comme la gestion des risques,
les mesures d'atténuation et la documentation technique.
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

class TestRiskManagement(unittest.TestCase):
    """Tests des fonctionnalités de gestion des risques."""
    
    def setUp(self):
        """Configure l'environnement de test."""
        self.ai_act_manager = AIActComplianceManager()
    
    def test_add_risk(self):
        """Teste l'ajout d'un risque identifié."""
        risk = {
            'risk_id': 'risk001',
            'description': 'Risque de biais algorithmique',
            'risk_level': 'high',
            'impact_areas': ['fairness', 'rights'],
            'likelihood': 0.7,
            'potential_harm': 'Discrimination contre certains groupes'
        }
        
        result = self.ai_act_manager.add_risk(risk)
        
        self.assertTrue(result)
        
        # Vérifier que le risque a été ajouté
        risks = self.ai_act_manager.get_identified_risks()
        self.assertEqual(len(risks), 1)
        self.assertEqual(risks[0]['risk_id'], 'risk001')
        self.assertEqual(risks[0]['risk_level'], 'high')
    
    def test_update_risk(self):
        """Teste la mise à jour d'un risque existant."""
        # Ajouter un risque
        risk = {
            'risk_id': 'risk002',
            'description': 'Description initiale',
            'risk_level': 'medium'
        }
        
        self.ai_act_manager.add_risk(risk)
        
        # Mettre à jour le risque
        updated_risk = {
            'risk_id': 'risk002',
            'description': 'Description mise à jour',
            'risk_level': 'high',
            'likelihood': 0.8
        }
        
        result = self.ai_act_manager.update_risk('risk002', updated_risk)
        
        self.assertTrue(result)
        
        # Vérifier la mise à jour
        risks = self.ai_act_manager.get_identified_risks()
        updated = next((r for r in risks if r['risk_id'] == 'risk002'), None)
        
        self.assertIsNotNone(updated)
        self.assertEqual(updated['description'], 'Description mise à jour')
        self.assertEqual(updated['risk_level'], 'high')
        self.assertEqual(updated['likelihood'], 0.8)
    
    def test_remove_risk(self):
        """Teste la suppression d'un risque."""
        # Ajouter un risque
        risk = {
            'risk_id': 'risk003',
            'description': 'Risque à supprimer',
            'risk_level': 'low'
        }
        
        self.ai_act_manager.add_risk(risk)
        
        # Vérifier qu'il a été ajouté
        risks_before = self.ai_act_manager.get_identified_risks()
        self.assertIn('risk003', [r['risk_id'] for r in risks_before])
        
        # Supprimer le risque
        result = self.ai_act_manager.remove_risk('risk003')
        
        self.assertTrue(result)
        
        # Vérifier qu'il a été supprimé
        risks_after = self.ai_act_manager.get_identified_risks()
        self.assertNotIn('risk003', [r['risk_id'] for r in risks_after])
    
    def test_get_risks_by_level(self):
        """Teste la récupération des risques filtrés par niveau."""
        # Ajouter plusieurs risques
        risks = [
            {'risk_id': 'high1', 'description': 'Risque élevé 1', 'risk_level': 'high'},
            {'risk_id': 'high2', 'description': 'Risque élevé 2', 'risk_level': 'high'},
            {'risk_id': 'med1', 'description': 'Risque moyen', 'risk_level': 'medium'},
            {'risk_id': 'low1', 'description': 'Risque faible', 'risk_level': 'low'}
        ]
        
        for risk in risks:
            self.ai_act_manager.add_risk(risk)
        
        # Récupérer les risques élevés
        high_risks = self.ai_act_manager.get_risks_by_level(RiskLevel.HIGH)
        
        self.assertEqual(len(high_risks), 2)
        self.assertTrue(all(r['risk_level'] == 'high' for r in high_risks))
        
        # Récupérer les risques moyens
        medium_risks = self.ai_act_manager.get_risks_by_level(RiskLevel.MEDIUM)
        
        self.assertEqual(len(medium_risks), 1)
        self.assertEqual(medium_risks[0]['risk_id'], 'med1')


class TestRiskMitigation(unittest.TestCase):
    """Tests des fonctionnalités de mesures d'atténuation des risques."""
    
    def setUp(self):
        """Configure l'environnement de test."""
        self.ai_act_manager = AIActComplianceManager()
        
        # Ajouter un risque pour les mesures d'atténuation
        self.ai_act_manager.add_risk({
            'risk_id': 'test_risk',
            'description': 'Risque pour tests',
            'risk_level': 'high'
        })
    
    def test_add_risk_mitigation(self):
        """Teste l'ajout d'une mesure d'atténuation."""
        mitigation = {
            'mitigation_id': 'mit001',
            'target_risk_id': 'test_risk',
            'description': 'Mesure d\'atténuation de test',
            'implementation_status': 'implemented',
            'effectiveness': 0.8
        }
        
        result = self.ai_act_manager.add_risk_mitigation(mitigation)
        
        self.assertTrue(result)
        
        # Vérifier que la mesure a été ajoutée
        mitigations = self.ai_act_manager.get_risk_mitigation_measures()
        self.assertEqual(len(mitigations), 1)
        self.assertEqual(mitigations[0]['mitigation_id'], 'mit001')
        self.assertEqual(mitigations[0]['target_risk_id'], 'test_risk')
    
    def test_update_mitigation_status(self):
        """Teste la mise à jour du statut d'une mesure d'atténuation."""
        # Ajouter une mesure
        mitigation = {
            'mitigation_id': 'mit002',
            'target_risk_id': 'test_risk',
            'description': 'Mesure à mettre à jour',
            'implementation_status': 'planned',
            'effectiveness': 0.5
        }
        
        self.ai_act_manager.add_risk_mitigation(mitigation)
        
        # Mettre à jour le statut
        result = self.ai_act_manager.update_mitigation_status(
            'mit002',
            'implemented',
            effectiveness=0.9
        )
        
        self.assertTrue(result)
        
        # Vérifier la mise à jour
        mitigations = self.ai_act_manager.get_risk_mitigation_measures()
        updated = next((m for m in mitigations if m['mitigation_id'] == 'mit002'), None)
        
        self.assertIsNotNone(updated)
        self.assertEqual(updated['implementation_status'], 'implemented')
        self.assertEqual(updated['effectiveness'], 0.9)
    
    def test_get_mitigations_for_risk(self):
        """Teste la récupération des mesures d'atténuation pour un risque spécifique."""
        # Ajouter plusieurs mesures pour différents risques
        self.ai_act_manager.add_risk({'risk_id': 'risk_a', 'description': 'Risque A'})
        self.ai_act_manager.add_risk({'risk_id': 'risk_b', 'description': 'Risque B'})
        
        mitigations = [
            {'mitigation_id': 'm1', 'target_risk_id': 'risk_a', 'description': 'Mesure 1 pour A'},
            {'mitigation_id': 'm2', 'target_risk_id': 'risk_a', 'description': 'Mesure 2 pour A'},
            {'mitigation_id': 'm3', 'target_risk_id': 'risk_b', 'description': 'Mesure pour B'}
        ]
        
        for m in mitigations:
            self.ai_act_manager.add_risk_mitigation(m)
        
        # Récupérer les mesures pour risk_a
        risk_a_mitigations = self.ai_act_manager.get_mitigations_for_risk('risk_a')
        
        self.assertEqual(len(risk_a_mitigations), 2)
        self.assertTrue(all(m['target_risk_id'] == 'risk_a' for m in risk_a_mitigations))
        
        # Récupérer les mesures pour risk_b
        risk_b_mitigations = self.ai_act_manager.get_mitigations_for_risk('risk_b')
        
        self.assertEqual(len(risk_b_mitigations), 1)
        self.assertEqual(risk_b_mitigations[0]['mitigation_id'], 'm3')


class TestTechnicalDocumentation(unittest.TestCase):
    """Tests des fonctionnalités de documentation technique."""
    
    def setUp(self):
        """Configure l'environnement de test."""
        self.ai_act_manager = AIActComplianceManager()
    
    def test_generate_technical_documentation(self):
        """Teste la génération de documentation technique."""
        # Définir une catégorie à haut risque pour la documentation complète
        self.ai_act_manager.set_system_category(AISystemCategory.HIGH_RISK)
        
        # Générer la documentation
        docs = self.ai_act_manager.generate_technical_documentation()
        
        self.assertIsNotNone(docs)
        self.assertIsInstance(docs, dict)
        
        # Vérifier la présence des sections standard
        standard_sections = [
            'system_description',
            'architecture',
            'development_details',
            'data_governance'
        ]
        
        for section in standard_sections:
            self.assertIn(section, docs)
    
    def test_update_technical_documentation(self):
        """Teste la mise à jour d'une section de documentation."""
        test_content = "Contenu de test pour la documentation technique"
        
        self.ai_act_manager.update_technical_documentation(
            'system_description',
            test_content
        )
        
        # Récupérer la documentation mise à jour
        docs = self.ai_act_manager.generate_technical_documentation()
        
        self.assertEqual(docs['system_description'], test_content)
    
    def test_get_specific_documentation_section(self):
        """Teste la récupération d'une section spécifique de la documentation."""
        # Mettre à jour deux sections
        section1_content = "Contenu de la section 1"
        section2_content = "Contenu de la section 2"
        
        self.ai_act_manager.update_technical_documentation('architecture', section1_content)
        self.ai_act_manager.update_technical_documentation('quality_control', section2_content)
        
        # Récupérer une section spécifique
        architecture_content = self.ai_act_manager.get_documentation_section('architecture')
        
        self.assertEqual(architecture_content, section1_content)


class TestAudit(unittest.TestCase):
    """Tests des fonctionnalités d'audit."""
    
    def setUp(self):
        """Configure l'environnement de test."""
        self.ai_act_manager = AIActComplianceManager()
        
        # Simuler un modèle
        self.mock_model = MagicMock()
        self.mock_model.__class__.__name__ = "RandomForestClassifier"
        
        # Simuler des données
        self.mock_data = {'features': [[1, 2, 3]]}
    
    def test_audit(self):
        """Teste la fonction d'audit."""
        with patch.object(self.ai_act_manager, '_assess_model') as mock_assess:
            mock_assess.return_value = {
                'risk_level': RiskLevel.HIGH.value,
                'applicable_requirements': [
                    {'id': 'req1', 'description': 'Exigence 1', 'status': 'partial'}
                ],
                'identified_risks': [
                    {'risk_id': 'risk1', 'description': 'Risque 1', 'risk_level': 'high'}
                ]
            }
            
            audit_result = self.ai_act_manager.audit(self.mock_model, self.mock_data)
            
            self.assertIsNotNone(audit_result)
            self.assertIn('timestamp', audit_result)
            self.assertIn('risk_level', audit_result)
            self.assertIn('compliance_status', audit_result)
            self.assertIn('requirements_summary', audit_result)
            mock_assess.assert_called_once()
    
    def test_evaluate_conformity(self):
        """Teste l'évaluation de conformité."""
        # Simuler des exigences avec différents statuts
        requirements = [
            {'id': 'req1', 'status': 'compliant'},
            {'id': 'req2', 'status': 'compliant'},
            {'id': 'req3', 'status': 'partial'},
            {'id': 'req4', 'status': 'non_compliant'}
        ]
        
        with patch.object(self.ai_act_manager, 'get_applicable_requirements') as mock_get:
            mock_get.return_value = requirements
            
            conformity = self.ai_act_manager._evaluate_conformity()
            
            self.assertIsNotNone(conformity)
            self.assertEqual(conformity['total_requirements'], 4)
            self.assertEqual(conformity['compliant_count'], 2)
            self.assertEqual(conformity['partial_count'], 1)
            self.assertEqual(conformity['non_compliant_count'], 1)
            self.assertEqual(conformity['compliance_percentage'], 50.0)  # (2/4) * 100


class TestDecisionLog(unittest.TestCase):
    """Tests de la fonctionnalité de journal des décisions."""
    
    def setUp(self):
        """Configure l'environnement de test."""
        from xplia.compliance.ai_act import DecisionLog
        self.decision_log = DecisionLog()
    
    def test_add_decision(self):
        """Teste l'ajout d'une décision au journal."""
        decision = {
            'input_data': {'feature1': 1.0, 'feature2': 2.0},
            'output': {'prediction': 'class_a'},
            'model_id': 'model123',
            'timestamp': datetime.datetime.now().isoformat(),
            'user_id': 'user456'
        }
        
        self.decision_log.add(decision)
        
        decisions = self.decision_log.get_decisions()
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0]['user_id'], 'user456')
        self.assertEqual(decisions[0]['output'], {'prediction': 'class_a'})
    
    def test_get_decisions_by_user(self):
        """Teste la récupération des décisions par utilisateur."""
        # Ajouter plusieurs décisions pour différents utilisateurs
        decisions = [
            {'user_id': 'user1', 'output': {'prediction': 'class_a'}},
            {'user_id': 'user2', 'output': {'prediction': 'class_b'}},
            {'user_id': 'user1', 'output': {'prediction': 'class_c'}}
        ]
        
        for d in decisions:
            self.decision_log.add(d)
        
        # Récupérer les décisions pour user1
        user1_decisions = self.decision_log.get_decisions_by_user('user1')
        
        self.assertEqual(len(user1_decisions), 2)
        self.assertTrue(all(d['user_id'] == 'user1' for d in user1_decisions))
    
    def test_clear_log(self):
        """Teste la suppression du journal des décisions."""
        # Ajouter quelques décisions
        self.decision_log.add({'user_id': 'user1'})
        self.decision_log.add({'user_id': 'user2'})
        
        # Vérifier qu'elles sont là
        self.assertEqual(len(self.decision_log.get_decisions()), 2)
        
        # Effacer le journal
        self.decision_log.clear()
        
        # Vérifier qu'il est vide
        self.assertEqual(len(self.decision_log.get_decisions()), 0)
    
    def test_export_log(self):
        """Teste l'exportation du journal des décisions."""
        # Ajouter une décision
        self.decision_log.add({
            'input_data': {'feature': 0.5},
            'output': {'prediction': 'positive'},
            'timestamp': '2023-01-01T12:00:00',
            'user_id': 'test_user'
        })
        
        # Exporter le journal
        export_data = self.decision_log.export()
        
        self.assertIsNotNone(export_data)
        self.assertEqual(len(export_data), 1)
        self.assertEqual(export_data[0]['user_id'], 'test_user')


if __name__ == '__main__':
    unittest.main()
