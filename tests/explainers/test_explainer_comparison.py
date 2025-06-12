"""
Tests de comparaison directe entre les différents explainers.

Ce module compare les performances des différents explainers implémentés
dans xplia selon plusieurs dimensions : fidélité, stabilité, temps d'exécution
et compréhensibilité.
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

from tests.explainers.test_explainer_comparison_base import ExplainerComparisonTest
from tests.explainers.test_explainer_lime import LIMEEvaluator
from tests.explainers.test_explainer_shap import SHAPEvaluator
from xplia.explainers.lime_explainer import LIMEExplainer
from xplia.explainers.shap_explainer import SHAPExplainer


class ComparativeExplainerTests(ExplainerComparisonTest):
    """Tests comparatifs entre les différents explainers."""

    def setUp(self):
        """Configure l'environnement de test avec des données communes."""
        # Créer des données et un modèle communs pour une comparaison équitable
        X, y = make_classification(
            n_samples=200, 
            n_features=10,
            n_informative=6,
            n_redundant=2,
            random_state=42
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Créer les évaluateurs pour chaque explainer
        self.lime_evaluator = LIMEEvaluator(LIMEExplainer, model, X_test, y_test)
        self.shap_evaluator = SHAPEvaluator(SHAPExplainer, model, X_test, y_test)
        
        # Initialiser les explainers
        self.lime_evaluator.initialize_explainer()
        self.shap_evaluator.initialize_explainer()
        
        # Liste de tous les évaluateurs pour faciliter les tests
        self.all_evaluators = [self.lime_evaluator, self.shap_evaluator]
        
    def test_all_metrics_comparison(self):
        """
        Compare tous les explainers sur toutes les métriques disponibles.
        Ce test génère un rapport complet de comparaison.
        """
        # Exécuter toutes les évaluations pour chaque explainer
        for evaluator in self.all_evaluators:
            evaluator.run_all_evaluations()
        
        # Générer le rapport de comparaison
        report = self.generate_comparison_report(self.all_evaluators)
        
        # Afficher le rapport
        print("\nRapport de comparaison des explainers:")
        self.print_comparison_report(report)
        
        # Vérifier que toutes les métriques sont présentes pour tous les explainers
        metrics = ['fidelity', 'stability', 'execution_time', 'comprehensibility']
        for metric in metrics:
            self.assertEqual(len(report['metrics'][metric]), len(self.all_evaluators))
            
            # Vérifier que toutes les valeurs sont valides
            for value in report['metrics'][metric]:
                if metric != 'execution_time':
                    self.assertGreaterEqual(value, 0)
                    self.assertLessEqual(value, 1)
                else:
                    self.assertGreater(value, 0)
        
        # Générer un graphique radar pour la comparaison visuelle
        # (si matplotlib est disponible)
        try:
            self._generate_comparison_radar(report)
        except Exception as e:
            print(f"Impossible de générer le graphique radar: {e}")
    
    def test_fidelity_comparison(self):
        """
        Compare spécifiquement la fidélité des différents explainers.
        C'est une métrique clé pour évaluer la qualité des explications.
        """
        # Collecter les scores de fidélité
        fidelity_scores = {}
        
        for evaluator in self.all_evaluators:
            explainer_name = evaluator.explainer_class.__name__
            fidelity = evaluator.evaluate_fidelity()
            fidelity_scores[explainer_name] = fidelity
        
        # Afficher les résultats
        print("\nComparaison des scores de fidélité:")
        for explainer, score in fidelity_scores.items():
            print(f"{explainer}: {score:.4f}")
        
        # Vérifier que tous les scores sont valides
        for score in fidelity_scores.values():
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)
    
    def test_execution_time_comparison(self):
        """
        Compare le temps d'exécution des différents explainers.
        C'est important pour évaluer la performance en situation réelle.
        """
        # Collecter les temps d'exécution
        execution_times = {}
        
        for evaluator in self.all_evaluators:
            explainer_name = evaluator.explainer_class.__name__
            execution_time = evaluator.evaluate_performance()
            execution_times[explainer_name] = execution_time
        
        # Afficher les résultats
        print("\nComparaison des temps d'exécution:")
        for explainer, time in execution_times.items():
            print(f"{explainer}: {time:.4f} secondes")
        
        # Vérifier que tous les temps sont positifs
        for time in execution_times.values():
            self.assertGreater(time, 0)
    
    def _generate_comparison_radar(self, report):
        """
        Génère un graphique radar pour visualiser la comparaison des explainers.
        
        Args:
            report: Le rapport de comparaison généré par generate_comparison_report
        """
        # Métriques à inclure (exclure execution_time car l'échelle est différente)
        metrics = ['fidelity', 'stability', 'comprehensibility']
        
        # Préparer les données pour le graphique radar
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Fermer le polygone
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        # Pour chaque explainer, tracer un polygone sur le radar
        for i, explainer in enumerate(report['explainers']):
            # Collecter les valeurs pour cet explainer
            values = [report['metrics'][metric][i] for metric in metrics]
            values += values[:1]  # Fermer le polygone
            
            # Tracer le polygone
            ax.plot(angles, values, linewidth=2, label=explainer)
            ax.fill(angles, values, alpha=0.25)
        
        # Configurer le graphique
        ax.set_theta_offset(np.pi / 2)  # Commencer à 12h
        ax.set_theta_direction(-1)  # Sens horaire
        
        # Étiquettes pour les axes
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        
        # Limite des axes et grille
        ax.set_ylim(0, 1)
        ax.grid(True)
        
        # Légende et titre
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        ax.set_title("Comparaison des Explainers", va='bottom')
        
        # Enregistrer le graphique (si nécessaire)
        # plt.savefig('explainer_comparison_radar.png')
        
        # Afficher le graphique (commenté pour éviter des problèmes dans l'environnement de test)
        # plt.show()


if __name__ == '__main__':
    unittest.main()
