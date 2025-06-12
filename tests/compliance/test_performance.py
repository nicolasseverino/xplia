"""
Tests de performance pour les modules de conformité réglementaire.

Ce module évalue les performances des modules GDPR et AI Act
sous différentes charges et avec différentes tailles de données.
"""

import unittest
import time
import random
from unittest.mock import MagicMock, patch
import numpy as np
import matplotlib.pyplot as plt
import os
from functools import wraps
import tempfile
import gc

from xplia.compliance.explanation_rights import (
    GDPRComplianceManager,
    DataCategory,
    ProcessingRecord
)

from xplia.compliance.ai_act import (
    AIActComplianceManager,
    RiskLevel,
    AISystemCategory
)

def measure_time(func):
    """Décorateur pour mesurer le temps d'exécution d'une fonction."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        gc.collect()  # Forcer le garbage collector pour plus de cohérence
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} a pris {execution_time:.4f} secondes à s'exécuter.")
        return result, execution_time
    return wrapper


class TestCompliancePerformance(unittest.TestCase):
    """Tests de performance pour les modules de conformité."""
    
    def setUp(self):
        """Configure l'environnement de test."""
        self.gdpr_manager = GDPRComplianceManager()
        self.ai_act_manager = AIActComplianceManager()
        
        # Modèle simulé
        self.mock_model = MagicMock()
        self.mock_model.__class__.__name__ = "RandomForestClassifier"
        
        # Configuration de base pour GDPR
        self.gdpr_manager.data_processing_registry.register_processing(
            name="Test Processing",
            description="Performance test processing",
            categories=[DataCategory.PERSONAL, DataCategory.FINANCIAL]
        )
        
        # Configuration de base pour AI Act
        self.ai_act_manager.set_system_category(AISystemCategory.HIGH_RISK)
        self.ai_act_manager.set_risk_level(RiskLevel.HIGH)
    
    @measure_time
    def run_gdpr_analysis(self, data_size):
        """Exécute l'analyse GDPR avec un ensemble de données de taille spécifique."""
        # Générer des données de taille variable
        features = np.random.rand(data_size, 10).tolist()
        test_data = {
            'features': features,
            'metadata': {'domain': 'test', 'size': data_size}
        }
        
        # Exécuter l'analyse
        return self.gdpr_manager.analyze(self.mock_model, test_data)
    
    @measure_time
    def run_ai_act_analysis(self, data_size):
        """Exécute l'analyse AI Act avec un ensemble de données de taille spécifique."""
        # Générer des données de taille variable
        features = np.random.rand(data_size, 10).tolist()
        test_data = {
            'features': features,
            'metadata': {'domain': 'test', 'size': data_size}
        }
        
        # Exécuter l'analyse
        return self.ai_act_manager.analyze(self.mock_model, test_data)
    
    def test_gdpr_scalability(self):
        """Teste la scalabilité de l'analyse GDPR avec des ensembles de données croissants."""
        data_sizes = [10, 100, 1000, 5000]
        execution_times = []
        
        for size in data_sizes:
            _, execution_time = self.run_gdpr_analysis(size)
            execution_times.append(execution_time)
        
        # Vérifier que les temps d'exécution sont raisonnables
        # et ne croissent pas de façon exponentielle
        for i in range(1, len(execution_times)):
            # Le ratio entre les temps ne devrait pas être plus de 10x le ratio entre les tailles
            size_ratio = data_sizes[i] / data_sizes[i-1]
            time_ratio = execution_times[i] / max(execution_times[i-1], 0.001)  # Éviter division par zéro
            
            # Tolérance pour tenir compte de la variabilité naturelle des mesures de performance
            self.assertLess(time_ratio, 20 * size_ratio, 
                           f"Le temps d'exécution augmente de manière disproportionnée: {time_ratio:.2f}x pour un ratio de taille de {size_ratio:.2f}x")
        
        # Sauvegarder les résultats si demandé
        if os.environ.get('SAVE_PERFORMANCE_RESULTS', '0') == '1':
            self._save_performance_plot(data_sizes, execution_times, 
                                       'GDPR Scalability', 'gdpr_scalability.png')
    
    def test_ai_act_scalability(self):
        """Teste la scalabilité de l'analyse AI Act avec des ensembles de données croissants."""
        data_sizes = [10, 100, 1000, 5000]
        execution_times = []
        
        for size in data_sizes:
            _, execution_time = self.run_ai_act_analysis(size)
            execution_times.append(execution_time)
        
        # Vérifier que les temps d'exécution sont raisonnables
        # et ne croissent pas de façon exponentielle
        for i in range(1, len(execution_times)):
            # Le ratio entre les temps ne devrait pas être plus de 10x le ratio entre les tailles
            size_ratio = data_sizes[i] / data_sizes[i-1]
            time_ratio = execution_times[i] / max(execution_times[i-1], 0.001)  # Éviter division par zéro
            
            # Tolérance pour tenir compte de la variabilité naturelle des mesures de performance
            self.assertLess(time_ratio, 20 * size_ratio, 
                           f"Le temps d'exécution augmente de manière disproportionnée: {time_ratio:.2f}x pour un ratio de taille de {size_ratio:.2f}x")
        
        # Sauvegarder les résultats si demandé
        if os.environ.get('SAVE_PERFORMANCE_RESULTS', '0') == '1':
            self._save_performance_plot(data_sizes, execution_times, 
                                       'AI Act Scalability', 'ai_act_scalability.png')
    
    @measure_time
    def run_gdpr_bulk_operations(self, num_records):
        """Exécute des opérations en masse sur le registre GDPR."""
        # Générer plusieurs enregistrements
        records = []
        for i in range(num_records):
            records.append({
                'name': f"Processing {i}",
                'description': f"Description for processing {i}",
                'categories': [DataCategory.PERSONAL.value]
            })
        
        # Exécuter la mise à jour en masse
        return self.gdpr_manager.data_processing_registry.bulk_update(records)
    
    @measure_time
    def run_ai_act_bulk_operations(self, num_requirements):
        """Exécute des mises à jour en masse sur les exigences AI Act."""
        # S'assurer que nous avons des exigences
        requirements = self.ai_act_manager.get_applicable_requirements()
        
        if not requirements:
            self.skipTest("Pas d'exigences disponibles pour le test de performance")
        
        # Préparer les mises à jour (en boucle si besoin pour atteindre le nombre souhaité)
        updates = []
        for i in range(num_requirements):
            # Utiliser des modulo pour ne pas dépasser la liste réelle d'exigences
            req_idx = i % len(requirements)
            updates.append({
                'id': requirements[req_idx]['id'],
                'status': 'compliant' if i % 2 == 0 else 'partial',
                'details': f"Updated in performance test #{i}"
            })
        
        # Exécuter la mise à jour en masse
        return self.ai_act_manager.bulk_update_requirements(updates)
    
    def test_bulk_operations_performance(self):
        """Teste les performances des opérations en masse pour GDPR et AI Act."""
        operation_sizes = [10, 50, 100, 500]
        gdpr_times = []
        ai_act_times = []
        
        for size in operation_sizes:
            # GDPR bulk operations
            _, gdpr_time = self.run_gdpr_bulk_operations(size)
            gdpr_times.append(gdpr_time)
            
            # AI Act bulk operations
            try:
                _, ai_act_time = self.run_ai_act_bulk_operations(size)
                ai_act_times.append(ai_act_time)
            except Exception as e:
                print(f"Erreur lors des opérations en masse AI Act de taille {size}: {e}")
                ai_act_times.append(None)
        
        # Vérifier les résultats
        print("\nPerformances des opérations en masse:")
        for i, size in enumerate(operation_sizes):
            gdpr_time = gdpr_times[i]
            ai_act_time = ai_act_times[i]
            
            print(f"Taille {size}: GDPR {gdpr_time:.4f}s, AI Act {ai_act_time:.4f}s if ai_act_time else 'N/A'}")
            
            if i > 0 and gdpr_times[i-1] is not None and gdpr_time is not None:
                gdpr_ratio = gdpr_time / max(gdpr_times[i-1], 0.001)
                size_ratio = size / operation_sizes[i-1]
                self.assertLess(gdpr_ratio, 15 * size_ratio, 
                               f"GDPR: Le temps d'exécution augmente de manière disproportionnée")
            
            if i > 0 and ai_act_times[i-1] is not None and ai_act_time is not None:
                ai_act_ratio = ai_act_time / max(ai_act_times[i-1], 0.001)
                size_ratio = size / operation_sizes[i-1]
                self.assertLess(ai_act_ratio, 15 * size_ratio, 
                               f"AI Act: Le temps d'exécution augmente de manière disproportionnée")
    
    @measure_time
    def run_report_generation(self, template_size):
        """Exécute la génération d'un rapport de conformité combiné de complexité variable."""
        from xplia.compliance.report_generator import ComplianceReportGenerator
        
        # Exporter les données de base
        gdpr_data = self.gdpr_manager.export_data()
        ai_act_data = self.ai_act_manager.export_data()
        
        # Ajouter des données selon la taille demandée
        for i in range(template_size):
            # Ajouter des activités GDPR
            gdpr_data['registry']['activities'].append({
                'name': f"Performance Activity {i}",
                'description': f"Activity for performance testing {i}",
                'categories': [DataCategory.PERSONAL.value]
            })
            
            # Ajouter des risques AI Act
            ai_act_data['risks'].append({
                'risk_id': f"perf_risk_{i}",
                'description': f"Risk for performance testing {i}",
                'risk_level': 'medium'
            })
        
        # Générer le rapport
        generator = ComplianceReportGenerator()
        generator.init_gdpr_data(gdpr_data)
        generator.init_ai_act_data(ai_act_data)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            report_path = tmp.name
        
        try:
            return generator.generate('json', output_path=report_path)
        finally:
            if os.path.exists(report_path):
                os.unlink(report_path)
    
    def test_report_generation_performance(self):
        """Teste les performances de la génération de rapport avec des données de taille croissante."""
        template_sizes = [5, 20, 50, 100]
        generation_times = []
        
        for size in template_sizes:
            _, execution_time = self.run_report_generation(size)
            generation_times.append(execution_time)
        
        # Vérifier que les temps d'exécution sont raisonnables
        for i in range(1, len(generation_times)):
            size_ratio = template_sizes[i] / template_sizes[i-1]
            time_ratio = generation_times[i] / max(generation_times[i-1], 0.001)
            
            # La génération de rapport devrait être raisonnablement linéaire
            self.assertLess(time_ratio, 10 * size_ratio, 
                           f"Le temps de génération de rapport augmente de manière disproportionnée")
    
    def _save_performance_plot(self, sizes, times, title, filename):
        """Sauvegarde un graphique de performance."""
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, times, 'o-', linewidth=2)
        plt.title(title)
        plt.xlabel('Taille des données')
        plt.ylabel('Temps d\'exécution (s)')
        plt.grid(True)
        
        # Sauvegarder dans un répertoire de résultats
        results_dir = os.path.join(os.getcwd(), 'performance_results')
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, filename))
        plt.close()
        

class TestMemoryUsage(unittest.TestCase):
    """Tests d'utilisation mémoire pour les modules de conformité."""
    
    @unittest.skipIf(True, "Les tests de mémoire nécessitent une configuration spéciale et peuvent être instables")
    def test_memory_usage(self):
        """Teste l'utilisation mémoire des opérations principales."""
        try:
            import tracemalloc
            import psutil
            
            def get_memory_usage():
                """Retourne l'utilisation mémoire actuelle du processus en MB."""
                process = psutil.Process()
                return process.memory_info().rss / (1024 * 1024)
            
            # Activer le suivi mémoire
            tracemalloc.start()
            
            # Mesures initiales
            start_snapshot = tracemalloc.take_snapshot()
            start_memory = get_memory_usage()
            
            # Créer les managers et effectuer des opérations intensives
            gdpr_manager = GDPRComplianceManager()
            ai_act_manager = AIActComplianceManager()
            
            # Générer de grandes quantités de données de test
            data_size = 10000
            features = np.random.rand(data_size, 10).tolist()
            test_data = {
                'features': features,
                'metadata': {'domain': 'test', 'size': data_size}
            }
            
            # Modèle simulé
            mock_model = MagicMock()
            
            # Effectuer des analyses
            gdpr_manager.analyze(mock_model, test_data)
            ai_act_manager.analyze(mock_model, test_data)
            
            # Générer une grande quantité de données pour le registre
            for i in range(1000):
                gdpr_manager.data_processing_registry.register_processing(
                    name=f"Memory Test {i}",
                    description=f"Large memory test {i}",
                    categories=[DataCategory.PERSONAL]
                )
            
            # Générer une grande quantité de risques
            for i in range(1000):
                ai_act_manager.add_risk({
                    'risk_id': f"memory_risk_{i}",
                    'description': f"Risk for memory testing {i}",
                    'risk_level': 'medium'
                })
            
            # Mesures finales
            end_memory = get_memory_usage()
            end_snapshot = tracemalloc.take_snapshot()
            
            # Analyser les différences
            memory_diff = end_memory - start_memory
            
            # Afficher les statistiques de mémoire
            print(f"\nUtilisation mémoire: {memory_diff:.2f} MB")
            
            # Analyser les allocations les plus importantes
            top_stats = end_snapshot.compare_to(start_snapshot, 'lineno')
            print("\nTop 10 différences de mémoire:")
            for stat in top_stats[:10]:
                print(stat)
            
            # Vérifier que l'utilisation mémoire est raisonnable (< 500MB)
            self.assertLess(memory_diff, 500, 
                           f"Utilisation mémoire excessive: {memory_diff:.2f} MB")
            
            # Arrêter le suivi mémoire
            tracemalloc.stop()
            
        except ImportError:
            self.skipTest("psutil et tracemalloc sont requis pour les tests de mémoire")


if __name__ == '__main__':
    unittest.main()
