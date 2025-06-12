"""
Tests de robustesse pour la gestion des risques du module de conformité AI Act.

Ce module teste la résilience des fonctionnalités de gestion des risques et
des mesures d'atténuation face à des situations limites ou des erreurs.
"""

import unittest
from unittest.mock import MagicMock, patch
import json
import tempfile
import os

from xplia.compliance.ai_act import (
    AIActComplianceManager,
    RiskLevel,
    AISystemCategory
)

class TestRiskManagementRobustness(unittest.TestCase):
    """Tests de robustesse pour la gestion des risques."""
    
    def setUp(self):
        """Configure l'environnement de test."""
        self.ai_act_manager = AIActComplianceManager()
    
    def test_add_risk_empty(self):
        """Teste l'ajout d'un risque avec des données minimales."""
        # Risque avec seulement l'ID
        minimal_risk = {
            'risk_id': 'risk001'
        }
        
        # L'ajout devrait fonctionner avec des données minimales
        result = self.ai_act_manager.add_risk(minimal_risk)
        
        self.assertTrue(result)
        
        # Vérifier que le risque a été ajouté
        risks = self.ai_act_manager.get_identified_risks()
        self.assertEqual(len(risks), 1)
        self.assertEqual(risks[0]['risk_id'], 'risk001')
    
    def test_add_risk_none(self):
        """Teste l'ajout d'un risque None."""
        # Tenter d'ajouter un risque None
        with self.assertRaises(ValueError):
            self.ai_act_manager.add_risk(None)
    
    def test_add_risk_missing_id(self):
        """Teste l'ajout d'un risque sans ID."""
        # Risque sans identifiant
        invalid_risk = {
            'description': 'Risque sans ID',
            'risk_level': 'high'
        }
        
        # L'ajout devrait échouer sans ID
        with self.assertRaises(ValueError):
            self.ai_act_manager.add_risk(invalid_risk)
    
    def test_add_duplicate_risk(self):
        """Teste l'ajout d'un risque avec un ID dupliqué."""
        # Ajouter un premier risque
        self.ai_act_manager.add_risk({
            'risk_id': 'duplicate_risk',
            'description': 'Premier risque'
        })
        
        # Ajouter un deuxième risque avec le même ID
        duplicate_result = self.ai_act_manager.add_risk({
            'risk_id': 'duplicate_risk',
            'description': 'Risque dupliqué'
        })
        
        # Le deuxième ajout devrait retourner False ou lever une exception
        self.assertFalse(duplicate_result)
        
        # Vérifier qu'il n'y a qu'un seul risque avec cet ID
        risks = self.ai_act_manager.get_identified_risks()
        matching_risks = [r for r in risks if r['risk_id'] == 'duplicate_risk']
        self.assertEqual(len(matching_risks), 1)
        
        # Le risque existant ne devrait pas être modifié
        self.assertEqual(matching_risks[0]['description'], 'Premier risque')
    
    def test_update_nonexistent_risk(self):
        """Teste la mise à jour d'un risque inexistant."""
        # Tenter de mettre à jour un risque qui n'existe pas
        result = self.ai_act_manager.update_risk(
            'nonexistent_risk',
            {'description': 'Nouvelle description'}
        )
        
        # La mise à jour devrait échouer
        self.assertFalse(result)
    
    def test_update_risk_invalid_data(self):
        """Teste la mise à jour d'un risque avec des données invalides."""
        # Ajouter un risque
        self.ai_act_manager.add_risk({
            'risk_id': 'valid_risk',
            'description': 'Description initiale'
        })
        
        # Tenter de mettre à jour avec des données invalides
        with self.assertRaises((ValueError, TypeError)):
            self.ai_act_manager.update_risk('valid_risk', "not_a_dict")
    
    def test_get_risks_by_invalid_level(self):
        """Teste la récupération des risques avec un niveau invalide."""
        # Ajouter quelques risques
        self.ai_act_manager.add_risk({
            'risk_id': 'risk1',
            'risk_level': 'high'
        })
        
        # Tenter de récupérer les risques avec un niveau invalide
        with self.assertRaises(ValueError):
            self.ai_act_manager.get_risks_by_level("invalid_level")


class TestRiskMitigationRobustness(unittest.TestCase):
    """Tests de robustesse pour les mesures d'atténuation des risques."""
    
    def setUp(self):
        """Configure l'environnement de test."""
        self.ai_act_manager = AIActComplianceManager()
        
        # Ajouter un risque de référence
        self.ai_act_manager.add_risk({
            'risk_id': 'test_risk',
            'description': 'Risque pour les tests',
            'risk_level': 'medium'
        })
    
    def test_add_mitigation_empty(self):
        """Teste l'ajout d'une mesure d'atténuation minimale."""
        # Mesure d'atténuation minimale
        minimal_mitigation = {
            'mitigation_id': 'mit001',
            'target_risk_id': 'test_risk'
        }
        
        # L'ajout devrait fonctionner avec des données minimales
        result = self.ai_act_manager.add_risk_mitigation(minimal_mitigation)
        
        self.assertTrue(result)
        
        # Vérifier que la mesure a été ajoutée
        mitigations = self.ai_act_manager.get_risk_mitigation_measures()
        self.assertEqual(len(mitigations), 1)
        self.assertEqual(mitigations[0]['mitigation_id'], 'mit001')
    
    def test_add_mitigation_none(self):
        """Teste l'ajout d'une mesure d'atténuation None."""
        # Tenter d'ajouter une mesure None
        with self.assertRaises(ValueError):
            self.ai_act_manager.add_risk_mitigation(None)
    
    def test_add_mitigation_missing_fields(self):
        """Teste l'ajout d'une mesure sans champs requis."""
        # Mesure sans ID
        missing_id = {
            'target_risk_id': 'test_risk',
            'description': 'Mesure sans ID'
        }
        
        # L'ajout devrait échouer sans ID
        with self.assertRaises(ValueError):
            self.ai_act_manager.add_risk_mitigation(missing_id)
        
        # Mesure sans risque cible
        missing_target = {
            'mitigation_id': 'mit002',
            'description': 'Mesure sans risque cible'
        }
        
        # L'ajout devrait échouer sans risque cible
        with self.assertRaises(ValueError):
            self.ai_act_manager.add_risk_mitigation(missing_target)
    
    def test_add_mitigation_nonexistent_risk(self):
        """Teste l'ajout d'une mesure pour un risque inexistant."""
        # Mesure pour un risque qui n'existe pas
        invalid_mitigation = {
            'mitigation_id': 'mit003',
            'target_risk_id': 'nonexistent_risk',
            'description': 'Mesure pour risque inexistant'
        }
        
        # Le comportement dépend de l'implémentation:
        # - Soit l'ajout échoue avec une erreur/exception
        # - Soit l'ajout réussit mais la mesure est "orpheline"
        
        try:
            result = self.ai_act_manager.add_risk_mitigation(invalid_mitigation)
            
            # Si l'appel réussit, vérifier si la mesure existe
            if result:
                mitigations = self.ai_act_manager.get_risk_mitigation_measures()
                found = any(m['mitigation_id'] == 'mit003' for m in mitigations)
                self.assertTrue(found, "La mesure devrait être présente si l'ajout a réussi")
        except ValueError:
            # Si une exception est levée, c'est aussi un comportement valide
            pass
    
    def test_add_duplicate_mitigation(self):
        """Teste l'ajout d'une mesure d'atténuation avec un ID dupliqué."""
        # Ajouter une première mesure
        self.ai_act_manager.add_risk_mitigation({
            'mitigation_id': 'duplicate_mit',
            'target_risk_id': 'test_risk',
            'description': 'Première mesure'
        })
        
        # Ajouter une deuxième mesure avec le même ID
        duplicate_result = self.ai_act_manager.add_risk_mitigation({
            'mitigation_id': 'duplicate_mit',
            'target_risk_id': 'test_risk',
            'description': 'Mesure dupliquée'
        })
        
        # Le deuxième ajout devrait retourner False ou lever une exception
        self.assertFalse(duplicate_result)
        
        # Vérifier qu'il n'y a qu'une seule mesure avec cet ID
        mitigations = self.ai_act_manager.get_risk_mitigation_measures()
        matching_mits = [m for m in mitigations if m['mitigation_id'] == 'duplicate_mit']
        self.assertEqual(len(matching_mits), 1)
        
        # La mesure existante ne devrait pas être modifiée
        self.assertEqual(matching_mits[0]['description'], 'Première mesure')
    
    def test_update_nonexistent_mitigation(self):
        """Teste la mise à jour d'une mesure inexistante."""
        # Tenter de mettre à jour le statut d'une mesure qui n'existe pas
        result = self.ai_act_manager.update_mitigation_status(
            'nonexistent_mit',
            'implemented'
        )
        
        # La mise à jour devrait échouer
        self.assertFalse(result)
    
    def test_update_mitigation_invalid_status(self):
        """Teste la mise à jour d'une mesure avec un statut invalide."""
        # Ajouter une mesure
        self.ai_act_manager.add_risk_mitigation({
            'mitigation_id': 'valid_mit',
            'target_risk_id': 'test_risk',
            'implementation_status': 'planned'
        })
        
        # Tenter de mettre à jour avec un statut invalide
        with self.assertRaises(ValueError):
            self.ai_act_manager.update_mitigation_status(
                'valid_mit', 
                'invalid_status'
            )
    
    def test_get_mitigations_for_nonexistent_risk(self):
        """Teste la récupération des mesures pour un risque inexistant."""
        # Récupérer les mesures pour un risque qui n'existe pas
        mitigations = self.ai_act_manager.get_mitigations_for_risk('nonexistent_risk')
        
        # Le résultat devrait être une liste vide, pas None
        self.assertIsNotNone(mitigations)
        self.assertEqual(len(mitigations), 0)


class TestRiskExportImportRobustness(unittest.TestCase):
    """Tests de robustesse pour l'exportation et l'importation des données de risque."""
    
    def setUp(self):
        """Configure l'environnement de test."""
        self.ai_act_manager = AIActComplianceManager()
        
        # Ajouter un risque et une mesure d'atténuation
        self.ai_act_manager.add_risk({
            'risk_id': 'export_risk',
            'description': 'Risque pour export',
            'risk_level': 'high'
        })
        
        self.ai_act_manager.add_risk_mitigation({
            'mitigation_id': 'export_mit',
            'target_risk_id': 'export_risk',
            'description': 'Mesure pour export',
            'implementation_status': 'implemented'
        })
    
    def test_export_import(self):
        """Teste l'exportation puis l'importation des données de risque."""
        # Exporter les données
        export_data = self.ai_act_manager.export_data()
        self.assertIsNotNone(export_data)
        
        # Créer un nouveau manager et importer les données
        new_manager = AIActComplianceManager()
        result = new_manager.import_data(export_data)
        
        # Vérifier que l'import a réussi
        self.assertTrue(result)
        
        # Vérifier que les risques ont été correctement importés
        risks = new_manager.get_identified_risks()
        self.assertEqual(len(risks), 1)
        self.assertEqual(risks[0]['risk_id'], 'export_risk')
        
        # Vérifier que les mesures ont été correctement importées
        mitigations = new_manager.get_risk_mitigation_measures()
        self.assertEqual(len(mitigations), 1)
        self.assertEqual(mitigations[0]['mitigation_id'], 'export_mit')
    
    def test_import_malformed_data(self):
        """Teste l'importation avec des données malformées."""
        # Données d'import malformées
        malformed_data = {
            'risks': 'not_a_list',
            'mitigations': 42
        }
        
        # Tenter d'importer les données
        result = self.ai_act_manager.import_data(malformed_data)
        
        # L'importation devrait échouer mais ne pas planter
        self.assertFalse(result)
        
        # Les données d'origine devraient rester intactes
        risks = self.ai_act_manager.get_identified_risks()
        self.assertEqual(len(risks), 1)
        self.assertEqual(risks[0]['risk_id'], 'export_risk')
    
    def test_import_partial_data(self):
        """Teste l'importation avec des données partielles."""
        # Données d'import partielles (seulement la partie risques)
        partial_data = {
            'risks': [
                {
                    'risk_id': 'new_risk',
                    'description': 'Nouveau risque',
                    'risk_level': 'medium'
                }
            ]
        }
        
        # Importer les données partielles
        result = self.ai_act_manager.import_data(partial_data)
        
        # L'importation devrait réussir malgré les données partielles
        self.assertTrue(result)
        
        # Vérifier que les risques ont été importés
        risks = self.ai_act_manager.get_identified_risks()
        self.assertEqual(len(risks), 1)
        self.assertEqual(risks[0]['risk_id'], 'new_risk')
        
        # Les mesures devraient être vides ou inchangées
        mitigations = self.ai_act_manager.get_risk_mitigation_measures()
        self.assertEqual(len(mitigations), 0)


if __name__ == '__main__':
    unittest.main()
