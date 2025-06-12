"""
Cadre de base pour l'évaluation comparative des explainers.

Ce module fournit une infrastructure commune pour évaluer et comparer
différents explainers sur des métriques clés comme la fidélité, 
la compréhensibilité, la stabilité et le temps d'exécution.
"""

import unittest
import time
import numpy as np
from abc import ABC, abstractmethod
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class ExplainerEvaluator(ABC):
    """Classe de base pour l'évaluation des explainers."""
    
    def __init__(self, explainer_class, model=None, X=None, y=None, random_state=42):
        """
        Initialise l'évaluateur avec un explainer et des données.
        
        Args:
            explainer_class: La classe d'explainer à évaluer
            model: Un modèle entraîné (optionnel)
            X: Données d'entrée pour l'évaluation (optionnel)
            y: Données de sortie pour l'entraînement (optionnel)
            random_state: Graine aléatoire pour la reproductibilité
        """
        self.explainer_class = explainer_class
        self.model = model
        self.X = X
        self.y = y
        self.random_state = random_state
        self.explainer = None
        
        # Métriques d'évaluation
        self.metrics = {}
    
    def prepare_data(self, n_samples=1000, n_features=10, classification=True):
        """
        Prépare des données synthétiques pour l'évaluation.
        
        Args:
            n_samples: Nombre d'échantillons
            n_features: Nombre de caractéristiques
            classification: True pour un problème de classification, False pour régression
        """
        if classification:
            X, y = make_classification(
                n_samples=n_samples, 
                n_features=n_features,
                n_informative=n_features // 2,
                n_redundant=n_features // 10,
                random_state=self.random_state
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=self.random_state
            )
            
            if self.model is None:
                self.model = RandomForestClassifier(random_state=self.random_state)
                self.model.fit(X_train, y_train)
        else:
            X, y = make_regression(
                n_samples=n_samples, 
                n_features=n_features,
                n_informative=n_features // 2,
                random_state=self.random_state
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=self.random_state
            )
            
            if self.model is None:
                self.model = RandomForestRegressor(random_state=self.random_state)
                self.model.fit(X_train, y_train)
        
        self.X = X_test
        self.y = y_test
        self.X_train = X_train
        self.y_train = y_train
        
        return X_train, X_test, y_train, y_test
    
    def initialize_explainer(self, **kwargs):
        """
        Initialise l'explainer avec le modèle et les données.
        
        Args:
            **kwargs: Arguments additionnels pour l'initialisation de l'explainer
        """
        if self.model is None or self.X is None:
            raise ValueError("Le modèle et les données doivent être définis avant d'initialiser l'explainer.")
        
        self.explainer = self.explainer_class(self.model, **kwargs)
        return self.explainer
    
    @abstractmethod
    def evaluate_fidelity(self):
        """
        Évalue la fidélité de l'explainer.
        La fidélité mesure à quel point les explications reflètent
        fidèlement le comportement du modèle.
        
        Returns:
            score: Un score de fidélité (plus élevé = meilleur)
        """
        pass
    
    @abstractmethod
    def evaluate_stability(self):
        """
        Évalue la stabilité de l'explainer.
        La stabilité mesure à quel point les explications sont cohérentes
        pour des entrées similaires.
        
        Returns:
            score: Un score de stabilité (plus élevé = meilleur)
        """
        pass
    
    def evaluate_performance(self, n_repeats=10):
        """
        Évalue la performance en temps d'exécution de l'explainer.
        
        Args:
            n_repeats: Nombre de répétitions pour la mesure
            
        Returns:
            avg_time: Temps d'exécution moyen en secondes
        """
        if self.explainer is None:
            raise ValueError("L'explainer doit être initialisé avant l'évaluation.")
        
        # Sélectionner un échantillon pour l'évaluation
        if self.X.shape[0] > 10:
            sample_indices = np.random.choice(self.X.shape[0], 10, replace=False)
            X_sample = self.X[sample_indices]
        else:
            X_sample = self.X
        
        # Mesurer le temps d'exécution
        times = []
        for _ in range(n_repeats):
            start_time = time.time()
            _ = self.explainer.explain(X_sample)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        self.metrics['execution_time'] = avg_time
        return avg_time
    
    @abstractmethod
    def evaluate_comprehensibility(self):
        """
        Évalue la compréhensibilité des explications.
        La compréhensibilité mesure à quel point les explications sont
        faciles à comprendre pour un humain.
        
        Returns:
            score: Un score de compréhensibilité (plus élevé = meilleur)
        """
        pass
    
    def run_all_evaluations(self):
        """
        Exécute toutes les évaluations disponibles et retourne les résultats.
        
        Returns:
            metrics: Dictionnaire contenant toutes les métriques d'évaluation
        """
        if self.explainer is None:
            self.initialize_explainer()
        
        # Exécuter les évaluations
        self.metrics['fidelity'] = self.evaluate_fidelity()
        self.metrics['stability'] = self.evaluate_stability()
        self.metrics['execution_time'] = self.evaluate_performance()
        self.metrics['comprehensibility'] = self.evaluate_comprehensibility()
        
        return self.metrics


class ExplainerComparisonTest(unittest.TestCase):
    """Test case de base pour la comparaison des explainers."""
    
    def setUp(self):
        """Configure l'environnement de test."""
        # Sera implémenté dans les sous-classes concrètes
        pass
    
    def generate_comparison_report(self, evaluators, metrics=None):
        """
        Génère un rapport de comparaison entre différents explainers.
        
        Args:
            evaluators: Liste d'évaluateurs ExplainerEvaluator
            metrics: Liste des métriques à inclure (par défaut, toutes)
            
        Returns:
            report: Dictionnaire contenant les résultats comparatifs
        """
        if metrics is None:
            metrics = ['fidelity', 'stability', 'execution_time', 'comprehensibility']
        
        report = {
            'explainers': [],
            'metrics': {}
        }
        
        # Initialiser les métriques
        for metric in metrics:
            report['metrics'][metric] = []
        
        # Collecter les résultats
        for evaluator in evaluators:
            explainer_name = evaluator.explainer_class.__name__
            report['explainers'].append(explainer_name)
            
            # S'assurer que toutes les métriques sont calculées
            evaluator_metrics = evaluator.metrics
            if len(evaluator_metrics) == 0:
                evaluator_metrics = evaluator.run_all_evaluations()
            
            # Ajouter les métriques au rapport
            for metric in metrics:
                if metric in evaluator_metrics:
                    report['metrics'][metric].append(evaluator_metrics[metric])
                else:
                    report['metrics'][metric].append(None)
        
        return report
    
    def print_comparison_report(self, report):
        """
        Affiche un rapport de comparaison formaté.
        
        Args:
            report: Dictionnaire contenant les résultats comparatifs
        """
        print("\n===== RAPPORT DE COMPARAISON DES EXPLAINERS =====")
        
        # Déterminer la largeur des colonnes
        max_name_len = max(len(name) for name in report['explainers'])
        col_width = max(max_name_len, 12)
        
        # En-tête
        header = "Explainer".ljust(col_width)
        for metric in report['metrics']:
            header += f" | {metric}".ljust(15)
        print(header)
        print("-" * len(header))
        
        # Données
        for i, explainer in enumerate(report['explainers']):
            row = explainer.ljust(col_width)
            for metric, values in report['metrics'].items():
                value = values[i]
                if value is not None:
                    if metric == 'execution_time':
                        row += f" | {value:.4f}s".ljust(15)
                    else:
                        row += f" | {value:.4f}".ljust(15)
                else:
                    row += f" | {'N/A'}".ljust(15)
            print(row)
        
        print("=" * len(header))


# Exemple minimal d'utilisation
if __name__ == "__main__":
    class DummyEvaluator(ExplainerEvaluator):
        """Implémentation minimale pour démonstration."""
        def evaluate_fidelity(self):
            return 0.85
        
        def evaluate_stability(self):
            return 0.9
        
        def evaluate_comprehensibility(self):
            return 0.7
    
    # Créer un évaluateur factice
    dummy_evaluator = DummyEvaluator(None)
    dummy_evaluator.metrics = {
        'fidelity': 0.85,
        'stability': 0.9,
        'execution_time': 0.123,
        'comprehensibility': 0.7
    }
    
    # Générer un rapport
    test_case = ExplainerComparisonTest()
    report = test_case.generate_comparison_report([dummy_evaluator])
    test_case.print_comparison_report(report)
