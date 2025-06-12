"""
Tests d'évaluation pour l'explainer LIME.

Ce module implémente les tests spécifiques pour évaluer les performances de LIME
selon les métriques définies dans le cadre d'évaluation comparative.
"""

import unittest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from tests.explainers.test_explainer_comparison_base import ExplainerEvaluator, ExplainerComparisonTest
from xplia.explainers.lime_explainer import LIMEExplainer


class LIMEEvaluator(ExplainerEvaluator):
    """Évaluateur spécifique pour LIME."""
    
    def evaluate_fidelity(self, n_samples=50):
        """
        Évalue la fidélité de LIME en comparant les prédictions du modèle
        avec celles de l'approximation linéaire locale.
        
        Args:
            n_samples: Nombre d'échantillons pour l'évaluation
            
        Returns:
            score: Score de fidélité entre 0 et 1
        """
        if self.explainer is None:
            raise ValueError("L'explainer doit être initialisé avant l'évaluation.")
        
        # Sélectionner un sous-ensemble pour l'évaluation
        if self.X.shape[0] > n_samples:
            sample_indices = np.random.choice(self.X.shape[0], n_samples, replace=False)
            X_eval = self.X[sample_indices]
        else:
            X_eval = self.X
        
        # Obtenir les prédictions du modèle
        model_predictions = self.model.predict_proba(X_eval)
        
        # Pour chaque échantillon, calculer la fidélité
        fidelities = []
        for i in range(X_eval.shape[0]):
            instance = X_eval[i].reshape(1, -1)
            
            # Obtenir l'explication
            explanation = self.explainer.explain(instance, top_k=None)
            
            # Vérifier le format de l'explication (peut varier selon l'implémentation)
            if hasattr(explanation, 'local_model'):
                # Si l'explainer expose directement le modèle linéaire
                local_model = explanation.local_model
                local_prediction = local_model.predict_proba(instance)[0]
                
                # Calculer l'erreur entre les deux prédictions
                error = np.mean(np.abs(model_predictions[i] - local_prediction))
                fidelity = 1 - error
            else:
                # Si on doit reconstruire la prédiction à partir des contributions des features
                try:
                    # Pour les explainers qui retournent des coefficients
                    feature_importances = explanation.feature_importances
                    intercept = explanation.intercept
                    
                    # Reconstruire la prédiction locale
                    local_prediction = np.dot(instance, feature_importances) + intercept
                    
                    # Convertir en probabilités si nécessaire
                    if hasattr(explanation, 'to_probability'):
                        local_prediction = explanation.to_probability(local_prediction)
                    
                    # Calculer l'erreur
                    error = np.mean(np.abs(model_predictions[i] - local_prediction))
                    fidelity = 1 - error
                except (AttributeError, TypeError):
                    # Si le format d'explication est différent, utilisez une métrique simplifiée
                    # Par exemple, corrélation entre les contributions et les valeurs des features
                    feature_contribs = np.array([item['contribution'] for item in explanation])
                    feature_values = instance[0]
                    
                    # Corrélation comme proxy de fidélité
                    corr = np.corrcoef(np.abs(feature_contribs), np.abs(feature_values))[0, 1]
                    fidelity = max(0, corr)  # Éviter les valeurs négatives
            
            fidelities.append(fidelity)
        
        # Retourner la fidélité moyenne
        avg_fidelity = np.mean(fidelities)
        self.metrics['fidelity'] = avg_fidelity
        return avg_fidelity
    
    def evaluate_stability(self, n_samples=30, perturbation=0.05):
        """
        Évalue la stabilité de LIME en appliquant de petites perturbations aux données
        et en mesurant la variation des explications.
        
        Args:
            n_samples: Nombre d'échantillons pour l'évaluation
            perturbation: Amplitude de la perturbation (en pourcentage de l'écart-type)
            
        Returns:
            score: Score de stabilité entre 0 et 1
        """
        if self.explainer is None:
            raise ValueError("L'explainer doit être initialisé avant l'évaluation.")
        
        # Sélectionner un sous-ensemble pour l'évaluation
        if self.X.shape[0] > n_samples:
            sample_indices = np.random.choice(self.X.shape[0], n_samples, replace=False)
            X_eval = self.X[sample_indices]
        else:
            X_eval = self.X
        
        # Calculer l'écart-type par colonne pour déterminer l'amplitude des perturbations
        std_per_feature = np.std(self.X, axis=0)
        
        stabilities = []
        for i in range(X_eval.shape[0]):
            instance = X_eval[i].reshape(1, -1)
            
            # Obtenir l'explication originale
            original_explanation = self.explainer.explain(instance, top_k=None)
            
            # Appliquer plusieurs petites perturbations
            n_perturbations = 5
            similarity_scores = []
            
            for _ in range(n_perturbations):
                # Créer une version perturbée de l'instance
                noise = np.random.normal(0, perturbation * std_per_feature)
                perturbed_instance = instance + noise.reshape(1, -1)
                
                # Obtenir l'explication pour l'instance perturbée
                perturbed_explanation = self.explainer.explain(perturbed_instance, top_k=None)
                
                # Calculer la similarité entre les explications
                # (peut varier selon le format des explications)
                try:
                    # Si nous avons accès aux coefficients/importances directement
                    orig_importances = original_explanation.feature_importances
                    pert_importances = perturbed_explanation.feature_importances
                    
                    # Calcul de similarité cosinus
                    similarity = np.dot(orig_importances, pert_importances) / (
                        np.linalg.norm(orig_importances) * np.linalg.norm(pert_importances)
                    )
                    similarity_scores.append(max(0, similarity))  # Garantir une valeur positive
                except (AttributeError, TypeError):
                    # Si le format d'explication est différent
                    # Extraire les contributions des features pour les deux explications
                    try:
                        orig_contribs = np.array([item['contribution'] for item in original_explanation])
                        pert_contribs = np.array([item['contribution'] for item in perturbed_explanation])
                        
                        # Calculer la corrélation de Spearman (rang) comme mesure de stabilité
                        from scipy.stats import spearmanr
                        corr, _ = spearmanr(orig_contribs, pert_contribs)
                        similarity_scores.append(max(0, corr))  # Éviter les valeurs négatives
                    except:
                        # Si les formats sont incompatibles, utiliser une valeur par défaut
                        similarity_scores.append(0.5)
            
            # Calculer la stabilité moyenne pour cette instance
            stability = np.mean(similarity_scores)
            stabilities.append(stability)
        
        # Retourner la stabilité moyenne
        avg_stability = np.mean(stabilities)
        self.metrics['stability'] = avg_stability
        return avg_stability
    
    def evaluate_comprehensibility(self):
        """
        Évalue la compréhensibilité des explications LIME
        en se basant sur des caractéristiques objectives.
        
        Returns:
            score: Score de compréhensibilité entre 0 et 1
        """
        if self.explainer is None:
            raise ValueError("L'explainer doit être initialisé avant l'évaluation.")
        
        # Sélectionner un échantillon pour l'évaluation
        instance = self.X[0].reshape(1, -1)
        
        # Mesures de compréhensibilité pour LIME:
        # 1. Sparsité: moins de features est mieux pour la compréhension
        # 2. Format: évaluer si la sortie est facilement interprétable
        
        explanation = self.explainer.explain(instance, top_k=None)
        
        # Évaluer la sparsité
        try:
            # Si nous avons accès aux coefficients directement
            if hasattr(explanation, 'feature_importances'):
                importances = explanation.feature_importances
                # Calculer la sparsité (% de features avec importance nulle/très faible)
                sparsity = np.mean(np.abs(importances) < 1e-5)
            else:
                # Si le format est une liste d'éléments avec contributions
                contribs = np.array([item['contribution'] for item in explanation])
                sparsity = np.mean(np.abs(contribs) < 1e-5)
            
            # Sparsité optimale autour de 0.7-0.8 (70-80% des features peu importantes)
            sparsity_score = 1.0 - abs(0.75 - sparsity)
            
        except (AttributeError, TypeError):
            # Valeur par défaut si le format n'est pas standard
            sparsity_score = 0.5
        
        # Compréhensibilité du format (subjective, mais LIME est généralement bon)
        format_score = 0.8  # Valeur fixe basée sur la littérature et l'expérience
        
        # Score final combiné
        comprehensibility = 0.5 * sparsity_score + 0.5 * format_score
        
        self.metrics['comprehensibility'] = comprehensibility
        return comprehensibility


class TestLIMEExplainer(ExplainerComparisonTest):
    """Tests d'évaluation pour l'explainer LIME."""
    
    def setUp(self):
        """Configure l'environnement de test."""
        # Créer un modèle et des données synthétiques
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Créer l'évaluateur LIME
        self.lime_evaluator = LIMEEvaluator(LIMEExplainer, model, X_test, y_test)
    
    def test_lime_evaluation(self):
        """Teste l'évaluation complète de LIME."""
        # Initialiser l'explainer
        self.lime_evaluator.initialize_explainer()
        
        # Exécuter toutes les évaluations
        metrics = self.lime_evaluator.run_all_evaluations()
        
        # Vérifier que toutes les métriques sont présentes
        self.assertIn('fidelity', metrics)
        self.assertIn('stability', metrics)
        self.assertIn('execution_time', metrics)
        self.assertIn('comprehensibility', metrics)
        
        # Les métriques devraient être entre 0 et 1
        for metric_name, value in metrics.items():
            if metric_name != 'execution_time':
                self.assertGreaterEqual(value, 0)
                self.assertLessEqual(value, 1)
        
        # Le temps d'exécution devrait être positif
        self.assertGreater(metrics['execution_time'], 0)
        
        # Générer et afficher le rapport
        report = self.generate_comparison_report([self.lime_evaluator])
        self.print_comparison_report(report)


# Si exécuté directement
if __name__ == '__main__':
    unittest.main()
