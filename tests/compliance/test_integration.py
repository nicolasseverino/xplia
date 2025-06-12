"""
Tests d'intégration pour les modules de conformité réglementaire.

Ce module teste l'intégration entre les différents composants du système
de conformité réglementaire, y compris GDPR et AI Act, ainsi que leur
intégration avec le générateur de rapport.
"""

import unittest
from unittest.mock import MagicMock, patch
import tempfile
import os
import json

from xplia.compliance.explanation_rights import GDPRComplianceManager, DataCategory
from xplia.compliance.ai_act import AIActComplianceManager, AISystemCategory
from xplia.compliance.report_generator import ComplianceReportGenerator

class TestGDPRAIActIntegration(unittest.TestCase):
    """Tests d'intégration entre les modules GDPR et AI Act."""
    
    def setUp(self):
        """Configure l'environnement de test."""
        self.gdpr_manager = GDPRComplianceManager()
        self.ai_act_manager = AIActComplianceManager()
        
        # Modèle et données de test
        self.mock_model = MagicMock()
        self.mock_model.__class__.__name__ = "RandomForestRegressor"
        
        self.mock_data = {
            'features': [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            'metadata': {'domain': 'finance'}
        }
    
    def test_analyze_both_systems(self):
        """Teste l'analyse GDPR et AI Act sur le même modèle et données."""
        # Configurer le registre GDPR
        self.gdpr_manager.data_processing_registry.register_processing(
            name="Scoring financier",
            description="Système d'évaluation financière",
            categories=[DataCategory.FINANCIAL]
        )
        
        # Configurer AI Act
        self.ai_act_manager.set_system_category(AISystemCategory.HIGH_RISK)
        self.ai_act_manager.set_use_case('credit_scoring')
        
        # Analyser avec les deux systèmes
        gdpr_result = self.gdpr_manager.analyze(self.mock_model, self.mock_data)
        ai_act_result = self.ai_act_manager.analyze(self.mock_model, self.mock_data)
        
        # Vérifier que les deux analyses ont fonctionné
        self.assertIsNotNone(gdpr_result)
        self.assertIsNotNone(ai_act_result)
        self.assertIn('status', gdpr_result)
        self.assertIn('status', ai_act_result)
    
    def test_cross_data_references(self):
        """Teste les références croisées entre GDPR et AI Act."""
        # Créer un ID utilisateur commun pour les deux systèmes
        test_user_id = "integration_test_user"
        
        # Simuler une explication
        mock_explanation = MagicMock()
        mock_explanation.to_dict.return_value = {'importance': [0.6, 0.3, 0.1]}
        
        # Enregistrer une action dans GDPR
        self.gdpr_manager.request_log.add(
            user_id=test_user_id,
            data_requested=self.mock_data,
            explanation_provided=mock_explanation
        )
        
        # Enregistrer une décision dans AI Act
        self.ai_act_manager.log_decision(
            input_data=self.mock_data,
            output={'prediction': 0.75},
            explanation=mock_explanation,
            user_id=test_user_id
        )
        
        # Vérifier les références croisées
        gdpr_requests = self.gdpr_manager.request_log.get_requests_by_user(test_user_id)
        ai_act_decisions = self.ai_act_manager.decision_log.get_decisions_by_user(test_user_id)
        
        self.assertEqual(len(gdpr_requests), 1)
        self.assertEqual(len(ai_act_decisions), 1)
        
        # Vérifier que l'explication est cohérente
        gdpr_explanation = gdpr_requests[0].get('explanation_provided')
        ai_act_explanation = ai_act_decisions[0].get('explanation')
        
        self.assertEqual(
            gdpr_explanation.to_dict() if gdpr_explanation else None,
            ai_act_explanation.to_dict() if ai_act_explanation else None
        )


class TestReportGeneratorIntegration(unittest.TestCase):
    """Tests d'intégration avec le générateur de rapports de conformité."""
    
    def setUp(self):
        """Configure l'environnement de test."""
        self.gdpr_manager = GDPRComplianceManager()
        self.ai_act_manager = AIActComplianceManager()
        
        # Configurer le GDPR
        self.gdpr_manager.data_processing_registry.register_processing(
            name="Test Processing",
            description="Processing for integration test",
            categories=[DataCategory.PERSONAL]
        )
        
        self.gdpr_manager.set_dpo_contact({
            'name': 'Test DPO',
            'email': 'dpo@test.com'
        })
        
        # Configurer l'AI Act
        self.ai_act_manager.set_system_category(AISystemCategory.HIGH_RISK)
        
        self.ai_act_manager.add_risk({
            'risk_id': 'risk001',
            'description': 'Test Risk',
            'risk_level': 'medium'
        })
        
        # Générer les données pour le rapport
        self.gdpr_data = self.gdpr_manager.export_data()
        self.ai_act_data = self.ai_act_manager.export_data()
        
        # Créer le générateur de rapport
        self.report_generator = ComplianceReportGenerator()
    
    def test_init_with_both_managers(self):
        """Teste l'initialisation du générateur avec les deux managers."""
        self.report_generator.init_gdpr_manager(self.gdpr_manager)
        self.report_generator.init_ai_act_manager(self.ai_act_manager)
        
        # Vérifier que les managers ont été correctement initialisés
        self.assertIsNotNone(self.report_generator.gdpr_manager)
        self.assertIsNotNone(self.report_generator.ai_act_manager)
        
        # Vérifier que les données sont accessibles
        gdpr_activities = self.report_generator.gdpr_manager.data_processing_registry.get_processing_activities()
        ai_act_risks = self.report_generator.ai_act_manager.get_identified_risks()
        
        self.assertEqual(len(gdpr_activities), 1)
        self.assertEqual(len(ai_act_risks), 1)
    
    def test_init_with_data(self):
        """Teste l'initialisation du générateur avec les données exportées."""
        self.report_generator.init_gdpr_data(self.gdpr_data)
        self.report_generator.init_ai_act_data(self.ai_act_data)
        
        # Vérifier que les données sont accessibles
        self.assertIsNotNone(self.report_generator.gdpr_data)
        self.assertIsNotNone(self.report_generator.ai_act_data)
    
    def test_generate_json_report(self):
        """Teste la génération d'un rapport au format JSON."""
        # Configurer le générateur
        self.report_generator.init_gdpr_data(self.gdpr_data)
        self.report_generator.init_ai_act_data(self.ai_act_data)
        
        # Créer un fichier temporaire pour le rapport
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            report_path = tmp.name
        
        try:
            # Générer le rapport
            output_path = self.report_generator.generate('json', output_path=report_path)
            
            # Vérifier que le fichier a été créé
            self.assertTrue(os.path.exists(output_path))
            
            # Vérifier le contenu du rapport
            with open(output_path, 'r', encoding='utf-8') as f:
                report_content = json.load(f)
            
            self.assertIn('title', report_content)
            self.assertIn('sections', report_content)
            
            # Vérifier que les deux sections GDPR et AI Act sont présentes
            section_titles = [s.get('title') for s in report_content['sections']]
            self.assertTrue(any('GDPR' in title for title in section_titles))
            self.assertTrue(any('AI Act' in title for title in section_titles))
            
        finally:
            # Nettoyer
            if os.path.exists(report_path):
                os.unlink(report_path)

    @patch('xplia.compliance.report_generator.HTMLReportFormatter')
    def test_generate_html_report(self, mock_formatter):
        """Teste la génération d'un rapport au format HTML."""
        # Configurer le mock formatter
        mock_instance = MagicMock()
        mock_formatter.return_value = mock_instance
        mock_instance.format_report.return_value = "<html>Rapport simulé</html>"
        
        # Configurer le générateur
        self.report_generator.init_gdpr_data(self.gdpr_data)
        self.report_generator.init_ai_act_data(self.ai_act_data)
        
        # Générer le rapport sans sauvegarde
        report_content = self.report_generator.generate('html')
        
        # Vérifier que le formateur a été appelé correctement
        mock_formatter.assert_called_once()
        mock_instance.format_report.assert_called_once()
        
        # Vérifier que le contenu est celui attendu
        self.assertEqual(report_content, "<html>Rapport simulé</html>")


class TestAPIIntegration(unittest.TestCase):
    """Tests d'intégration avec l'API publique."""
    
    @patch('xplia.compliance.explanation_rights.GDPRComplianceManager')
    @patch('xplia.compliance.ai_act.AIActComplianceManager')
    def test_analyze_gdpr_compliance(self, mock_ai_act_cls, mock_gdpr_cls):
        """Teste la fonction API analyze_gdpr_compliance."""
        # Configurer le mock
        mock_gdpr_instance = MagicMock()
        mock_gdpr_cls.return_value = mock_gdpr_instance
        mock_gdpr_instance.analyze.return_value = {"status": "compliant"}
        
        # Importer la fonction API
        from xplia.api import analyze_gdpr_compliance
        
        # Utiliser la fonction API
        mock_model = MagicMock()
        mock_data = {"features": [[1, 2, 3]]}
        
        result = analyze_gdpr_compliance(
            model=mock_model,
            data=mock_data,
            data_processing_records={"test": "record"},
            dpo_contact={"name": "Test DPO"}
        )
        
        # Vérifier que le manager a été appelé correctement
        mock_gdpr_cls.assert_called_once()
        mock_gdpr_instance.analyze.assert_called_once_with(mock_model, mock_data)
        mock_gdpr_instance.data_processing_registry.update.assert_called_once()
        mock_gdpr_instance.set_dpo_contact.assert_called_once()
        
        # Vérifier le résultat
        self.assertEqual(result, {"status": "compliant"})
    
    @patch('xplia.compliance.ai_act.AIActComplianceManager')
    def test_analyze_ai_act_compliance(self, mock_ai_act_cls):
        """Teste la fonction API analyze_ai_act_compliance."""
        # Configurer le mock
        mock_ai_act_instance = MagicMock()
        mock_ai_act_cls.return_value = mock_ai_act_instance
        mock_ai_act_instance.analyze.return_value = {"status": "compliant"}
        
        # Importer la fonction API
        from xplia.api import analyze_ai_act_compliance
        
        # Utiliser la fonction API
        mock_model = MagicMock()
        mock_data = {"features": [[1, 2, 3]]}
        
        result = analyze_ai_act_compliance(
            model=mock_model,
            data=mock_data,
            risk_level='high',
            use_case='healthcare'
        )
        
        # Vérifier que le manager a été appelé correctement
        mock_ai_act_cls.assert_called_once()
        mock_ai_act_instance.analyze.assert_called_once()
        mock_ai_act_instance.set_risk_level.assert_called_once_with('high')
        mock_ai_act_instance.set_use_case.assert_called_once_with('healthcare')
        
        # Vérifier le résultat
        self.assertEqual(result, {"status": "compliant"})
    
    @patch('xplia.compliance.report_generator.ComplianceReportGenerator')
    def test_generate_compliance_report(self, mock_generator_cls):
        """Teste la fonction API generate_compliance_report."""
        # Configurer le mock
        mock_generator = MagicMock()
        mock_generator_cls.return_value = mock_generator
        mock_generator.generate.return_value = "/path/to/report.html"
        
        # Importer la fonction API
        from xplia.api import generate_compliance_report
        
        # Créer des données de test
        gdpr_data = {"activities": [{"name": "Test Activity"}]}
        ai_act_data = {"risks": [{"risk_id": "risk001"}]}
        
        # Utiliser la fonction API
        result = generate_compliance_report(
            output_formats=['html'],
            gdpr_data=gdpr_data,
            ai_act_data=ai_act_data,
            output_path='/tmp/test_report'
        )
        
        # Vérifier que le générateur a été appelé correctement
        mock_generator_cls.assert_called_once()
        mock_generator.init_gdpr_data.assert_called_once_with(gdpr_data)
        mock_generator.init_ai_act_data.assert_called_once_with(ai_act_data)
        mock_generator.generate.assert_called_once()
        
        # Vérifier le résultat
        self.assertEqual(result, {'html': "/path/to/report.html"})


if __name__ == '__main__':
    unittest.main()
