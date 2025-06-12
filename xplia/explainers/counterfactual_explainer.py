"""
Counterfactual Explainer pour XPLIA
==================================

Ce module implémente l'explainer basé sur les exemples contrefactuels
dans le framework XPLIA. Cette approche génère des exemples alternatifs
qui auraient conduit à une prédiction différente.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..core.base import (AudienceLevel, ExplainerBase, ExplainabilityMethod,
                        ExplanationResult, FeatureImportance, ModelMetadata)
from ..core.registry import register_explainer


@register_explainer
class CounterfactualExplainer(ExplainerBase):
    """
    Explainer basé sur les exemples contrefactuels.
    
    Cette classe génère des exemples contrefactuels qui montrent comment
    modifier les entrées pour obtenir une prédiction différente du modèle.
    
    Caractéristiques principales:
    - Génération d'exemples contrefactuels pour des instances spécifiques
    - Support de différents algorithmes de génération de contrefactuels
    - Contraintes personnalisables sur les caractéristiques modifiables
    - Visualisations adaptées au niveau d'audience
    """
    
    def __init__(self, model, **kwargs):
        """
        Initialise l'explainer contrefactuel.
        
        Args:
            model: Modèle à expliquer
            **kwargs: Paramètres additionnels
                feature_names: Noms des caractéristiques
                categorical_features: Indices ou noms des caractéristiques catégorielles
                continuous_features: Indices ou noms des caractéristiques continues
                method: Méthode de génération ('dice', 'genetic', 'optimization')
                target_class: Classe cible pour les contrefactuels (None = opposée à la prédiction)
                feature_ranges: Dictionnaire des plages valides pour chaque caractéristique
                proximity_weight: Poids pour la proximité aux données d'origine
                sparsity_weight: Poids pour la parcimonie (moins de changements)
                diversity_weight: Poids pour la diversité des contrefactuels
        """
        super().__init__(model, **kwargs)
        self._method = ExplainabilityMethod.COUNTERFACTUAL
        
        # Paramètres
        self._feature_names = kwargs.get('feature_names', None)
        self._categorical_features = kwargs.get('categorical_features', [])
        self._continuous_features = kwargs.get('continuous_features', [])
        self._counterfactual_method = kwargs.get('method', 'dice')  # 'dice', 'genetic', 'optimization'
        self._target_class = kwargs.get('target_class', None)
        self._feature_ranges = kwargs.get('feature_ranges', {})
        self._proximity_weight = kwargs.get('proximity_weight', 0.5)
        self._sparsity_weight = kwargs.get('sparsity_weight', 0.2)
        self._diversity_weight = kwargs.get('diversity_weight', 0.3)
        
        # Explainer spécifique
        self._cf_explainer = None
        
        # Métadonnées du modèle
        self._metadata = None
        
        # Logger
        self._logger = logging.getLogger(__name__)
        
    def explain_instance(self, instance, X_background=None, **kwargs) -> ExplanationResult:
        """
        Génère des explications contrefactuelles pour une instance spécifique.
        
        Args:
            instance: Instance à expliquer (array, liste, dict ou pandas.Series)
            X_background: Données d'arrière-plan pour contraindre les contrefactuels (optionnel)
            **kwargs: Paramètres additionnels
                desired_class: Classe désirée pour le contrefactuel (par défaut: opposée à la prédiction)
                num_counterfactuals: Nombre de contrefactuels à générer
                feature_weights: Poids pour chaque caractéristique (importance du changement)
                immutable_features: Caractéristiques qui ne peuvent pas être modifiées
                audience_level: Niveau d'audience (TECHNICAL, BUSINESS, PUBLIC)
                
        Returns:
            ExplanationResult: Résultat standardisé de l'explication
        """
        # Paramètres
        audience_level = kwargs.get('audience_level', AudienceLevel.TECHNICAL)
        desired_class = kwargs.get('desired_class', self._target_class)
        num_counterfactuals = kwargs.get('num_counterfactuals', 3)
        feature_weights = kwargs.get('feature_weights', None)
        immutable_features = kwargs.get('immutable_features', [])
        
        # Convertir l'instance en format approprié
        if isinstance(instance, dict):
            # Convertir dict en array
            feature_names = list(instance.keys())
            instance_array = np.array([instance[f] for f in feature_names]).reshape(1, -1)
        elif isinstance(instance, pd.Series):
            feature_names = instance.index.tolist()
            instance_array = instance.values.reshape(1, -1)
        elif isinstance(instance, (list, np.ndarray)):
            instance_array = np.array(instance).reshape(1, -1)
            feature_names = kwargs.get('feature_names', self._feature_names) or \
                           [f"feature_{i}" for i in range(instance_array.shape[1])]
        else:
            raise ValueError("Format d'instance non supporté. Utilisez un dict, pandas.Series, liste ou numpy.ndarray.")
        
        # Convertir les données d'arrière-plan en format approprié
        if X_background is not None:
            if isinstance(X_background, pd.DataFrame):
                X_values = X_background.values
            else:
                X_values = np.array(X_background)
        else:
            X_values = None
        
        # Tracer l'action
        self.add_audit_record("explain_instance", {
            "n_features": len(feature_names),
            "audience_level": audience_level.value if isinstance(audience_level, AudienceLevel) else audience_level,
            "counterfactual_method": self._counterfactual_method,
            "num_counterfactuals": num_counterfactuals
        })
        
        try:
            # Initialiser l'explainer contrefactuel si nécessaire
            if self._cf_explainer is None:
                self._initialize_cf_explainer(feature_names, X_values)
            
            # Générer les contrefactuels
            counterfactuals, metadata = self._generate_counterfactuals(
                instance_array, 
                feature_names, 
                desired_class, 
                num_counterfactuals,
                feature_weights,
                immutable_features
            )
            
            # Calculer les changements de caractéristiques
            feature_changes = self._calculate_feature_changes(instance_array[0], counterfactuals, feature_names)
            
            # Extraire les métadonnées du modèle
            if not self._metadata:
                self._extract_metadata()
            
            # Créer le résultat d'explication
            result = ExplanationResult(
                method=ExplainabilityMethod.COUNTERFACTUAL,
                model_metadata=self._metadata,
                feature_importances=self._convert_changes_to_importances(feature_changes),
                raw_explanation={
                    "instance": instance_array[0].tolist(),
                    "counterfactuals": counterfactuals.tolist() if isinstance(counterfactuals, np.ndarray) else counterfactuals,
                    "feature_names": feature_names,
                    "feature_changes": feature_changes,
                    "metadata": metadata
                },
                audience_level=audience_level
            )
            
            return result
            
        except Exception as e:
            self._logger.error(f"Erreur lors de la génération des contrefactuels: {str(e)}")
            raise RuntimeError(f"Échec de l'explication contrefactuelle: {str(e)}")
    
    def explain(self, X, y=None, **kwargs) -> ExplanationResult:
        """
        Génère des explications contrefactuelles pour un ensemble de données.
        Pour les explications contrefactuelles, cette méthode sélectionne un échantillon
        représentatif et génère des contrefactuels pour chaque instance.
        
        Args:
            X: Données d'entrée à expliquer
            y: Valeurs cibles réelles (optionnel)
            **kwargs: Paramètres additionnels
                max_instances: Nombre maximum d'instances à expliquer
                sampling_strategy: Stratégie d'échantillonnage ('random', 'stratified', 'kmeans')
                
        Returns:
            ExplanationResult: Résultat standardisé de l'explication
        """
        # Paramètres
        audience_level = kwargs.get('audience_level', AudienceLevel.TECHNICAL)
        max_instances = kwargs.get('max_instances', 5)
        sampling_strategy = kwargs.get('sampling_strategy', 'random')
        
        # Conversion des données en format approprié
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X_values = X.values
        else:
            X_values = np.array(X)
            feature_names = kwargs.get('feature_names', self._feature_names) or \
                           [f"feature_{i}" for i in range(X_values.shape[1])]
        
        # Échantillonner des instances représentatives
        sampled_indices = self._sample_instances(X_values, y, max_instances, sampling_strategy)
        sampled_instances = X_values[sampled_indices]
        
        # Tracer l'action
        self.add_audit_record("explain", {
            "n_samples": X_values.shape[0],
            "n_features": X_values.shape[1],
            "audience_level": audience_level.value if isinstance(audience_level, AudienceLevel) else audience_level,
            "max_instances": max_instances,
            "sampling_strategy": sampling_strategy
        })
        
        try:
            # Générer des contrefactuels pour chaque instance échantillonnée
            all_counterfactuals = []
            all_feature_changes = []
            
            for instance in sampled_instances:
                # Utiliser explain_instance pour chaque instance
                instance_result = self.explain_instance(
                    instance, 
                    X_background=X_values, 
                    feature_names=feature_names,
                    audience_level=audience_level,
                    **kwargs
                )
                
                # Collecter les résultats
                all_counterfactuals.append(instance_result.raw_explanation["counterfactuals"])
                all_feature_changes.append(instance_result.raw_explanation["feature_changes"])
            
            # Agréger les changements de caractéristiques
            aggregated_changes = self._aggregate_feature_changes(all_feature_changes, feature_names)
            
            # Extraire les métadonnées du modèle
            if not self._metadata:
                self._extract_metadata()
            
            # Créer le résultat d'explication
            result = ExplanationResult(
                method=ExplainabilityMethod.COUNTERFACTUAL,
                model_metadata=self._metadata,
                feature_importances=self._convert_changes_to_importances(aggregated_changes),
                raw_explanation={
                    "sampled_instances": sampled_instances.tolist(),
                    "sampled_indices": sampled_indices.tolist(),
                    "all_counterfactuals": all_counterfactuals,
                    "all_feature_changes": all_feature_changes,
                    "aggregated_changes": aggregated_changes,
                    "feature_names": feature_names
                },
                audience_level=audience_level
            )
            
            return result
            
        except Exception as e:
            self._logger.error(f"Erreur lors de la génération des contrefactuels: {str(e)}")
            raise RuntimeError(f"Échec de l'explication contrefactuelle: {str(e)}")
            
    def _initialize_cf_explainer(self, feature_names, X_background=None):
        """
        Initialise l'explainer contrefactuel selon la méthode choisie.
        
        Args:
            feature_names: Noms des caractéristiques
            X_background: Données d'arrière-plan pour contraindre les contrefactuels
        """
        self._feature_names = feature_names
        
        if self._counterfactual_method == 'dice':
            try:
                import dice_ml
            except ImportError:
                raise ImportError("Le package DiCE est requis pour cette méthode. "
                                "Installez-le avec 'pip install dice-ml'.")
                
            # L'initialisation complète sera faite lors de la première génération de contrefactuels
            # car elle nécessite des données d'arrière-plan et des informations sur le modèle
            self._cf_explainer = 'dice'
            
        elif self._counterfactual_method == 'genetic':
            # Implémentation d'un algorithme génétique personnalisé
            self._cf_explainer = 'genetic'
            
        elif self._counterfactual_method == 'optimization':
            # Implémentation basée sur l'optimisation
            self._cf_explainer = 'optimization'
            
        else:
            raise ValueError(f"Méthode contrefactuelle non supportée: {self._counterfactual_method}")
    
    def _sample_instances(self, X, y=None, max_instances=5, strategy='random'):
        """
        Échantillonne des instances représentatives à partir des données.
        
        Args:
            X: Données d'entrée
            y: Valeurs cibles réelles (optionnel)
            max_instances: Nombre maximum d'instances à sélectionner
            strategy: Stratégie d'échantillonnage ('random', 'stratified', 'kmeans')
            
        Returns:
            numpy.ndarray: Indices des instances sélectionnées
        """
        n_samples = X.shape[0]
        max_instances = min(max_instances, n_samples)
        
        if strategy == 'random':
            # Échantillonnage aléatoire
            indices = np.random.choice(n_samples, max_instances, replace=False)
            
        elif strategy == 'stratified' and y is not None:
            # Échantillonnage stratifié par classe
            try:
                from sklearn.model_selection import StratifiedShuffleSplit
            except ImportError:
                self._logger.warning("scikit-learn est requis pour l'échantillonnage stratifié. "
                                  "Utilisation de l'échantillonnage aléatoire.")
                return np.random.choice(n_samples, max_instances, replace=False)
                
            sss = StratifiedShuffleSplit(n_splits=1, test_size=max_instances/n_samples, random_state=42)
            for _, test_idx in sss.split(X, y):
                indices = test_idx
                if len(indices) > max_instances:
                    indices = indices[:max_instances]
                break
                
        elif strategy == 'kmeans':
            # Échantillonnage basé sur le clustering k-means
            try:
                from sklearn.cluster import KMeans
                from sklearn.preprocessing import StandardScaler
            except ImportError:
                self._logger.warning("scikit-learn est requis pour l'échantillonnage k-means. "
                                  "Utilisation de l'échantillonnage aléatoire.")
                return np.random.choice(n_samples, max_instances, replace=False)
                
            # Normaliser les données
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Appliquer k-means
            kmeans = KMeans(n_clusters=max_instances, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Sélectionner l'instance la plus proche de chaque centroide
            indices = []
            for i in range(max_instances):
                cluster_points = np.where(clusters == i)[0]
                if len(cluster_points) > 0:
                    # Trouver le point le plus proche du centroide
                    centroid = kmeans.cluster_centers_[i]
                    distances = np.linalg.norm(X_scaled[cluster_points] - centroid, axis=1)
                    closest_idx = cluster_points[np.argmin(distances)]
                    indices.append(closest_idx)
            
            # Compléter si nécessaire
            if len(indices) < max_instances:
                remaining = np.setdiff1d(np.arange(n_samples), indices)
                additional = np.random.choice(remaining, max_instances - len(indices), replace=False)
                indices = np.concatenate([indices, additional])
        else:
            # Par défaut, échantillonnage aléatoire
            indices = np.random.choice(n_samples, max_instances, replace=False)
            
        return indices
        
    def _generate_counterfactuals(self, instance, feature_names, desired_class=None, 
                                num_counterfactuals=3, feature_weights=None, immutable_features=None):
        """
        Génère des exemples contrefactuels pour une instance donnée.
        
        Args:
            instance: Instance à expliquer
            feature_names: Noms des caractéristiques
            desired_class: Classe désirée pour le contrefactuel
            num_counterfactuals: Nombre de contrefactuels à générer
            feature_weights: Poids pour chaque caractéristique
            immutable_features: Caractéristiques qui ne peuvent pas être modifiées
            
        Returns:
            tuple: (contrefactuels, métadonnées)
        """
        if self._cf_explainer == 'dice':
            return self._generate_dice_counterfactuals(
                instance, feature_names, desired_class, 
                num_counterfactuals, feature_weights, immutable_features
            )
        elif self._cf_explainer == 'genetic':
            return self._generate_genetic_counterfactuals(
                instance, feature_names, desired_class, 
                num_counterfactuals, feature_weights, immutable_features
            )
        elif self._cf_explainer == 'optimization':
            return self._generate_optimization_counterfactuals(
                instance, feature_names, desired_class, 
                num_counterfactuals, feature_weights, immutable_features
            )
        else:
            raise ValueError(f"Méthode contrefactuelle non initialisée")
    
    def _generate_dice_counterfactuals(self, instance, feature_names, desired_class=None, 
                                    num_counterfactuals=3, feature_weights=None, immutable_features=None):
        """
        Génère des exemples contrefactuels en utilisant DiCE.
        
        Args:
            instance: Instance à expliquer
            feature_names: Noms des caractéristiques
            desired_class: Classe désirée pour le contrefactuel
            num_counterfactuals: Nombre de contrefactuels à générer
            feature_weights: Poids pour chaque caractéristique
            immutable_features: Caractéristiques qui ne peuvent pas être modifiées
            
        Returns:
            tuple: (contrefactuels, métadonnées)
        """
        try:
            import dice_ml
            from dice_ml.utils import helpers
        except ImportError:
            raise ImportError("Le package DiCE est requis pour cette méthode. "
                            "Installez-le avec 'pip install dice-ml'.")
        
        # Créer un ensemble de données synthétique pour initialiser DiCE
        # (nécessaire car DiCE a besoin d'un dataset)
        synthetic_data = np.random.normal(0, 1, size=(100, instance.shape[1]))
        
        # Déterminer les types de caractéristiques
        continuous_features = []
        categorical_features = []
        
        if self._continuous_features:
            continuous_features = self._continuous_features
        if self._categorical_features:
            categorical_features = self._categorical_features
            
        # Si aucun n'est spécifié, supposer que toutes sont continues
        if not continuous_features and not categorical_features:
            continuous_features = list(range(instance.shape[1]))
            
        # Créer le dataset DiCE
        d = dice_ml.Data(
            dataframe=pd.DataFrame(synthetic_data, columns=feature_names),
            continuous_features=[feature_names[i] for i in continuous_features if i < len(feature_names)],
            outcome_name='target'  # Nom factice
        )
        
        # Créer le modèle backend DiCE
        # Utiliser le backend approprié selon le type de modèle
        model_type = self._get_model_type()
        if model_type in ['sklearn', 'xgboost', 'lightgbm', 'catboost']:
            m = dice_ml.Model(model=self._model, backend='sklearn')
        elif model_type == 'tensorflow':
            m = dice_ml.Model(model=self._model, backend='TF2')
        elif model_type == 'pytorch':
            m = dice_ml.Model(model=self._model, backend='PYT')
        else:
            # Utiliser le backend générique
            m = dice_ml.Model(model=self._model, backend='sklearn')
        
        # Créer l'explainer DiCE
        exp = dice_ml.Dice(d, m)
        
        # Préparer les paramètres pour la génération de contrefactuels
        query_instance = pd.DataFrame(instance, columns=feature_names)
        
        # Déterminer la classe désirée
        if desired_class is None:
            # Prédire la classe actuelle
            if hasattr(self._model, 'predict_proba'):
                pred = self._model.predict_proba(instance)[0]
                current_class = np.argmax(pred)
                # Classe opposée pour binaire, classe avec 2e plus grande proba pour multi-classe
                if len(pred) == 2:
                    desired_class = 1 - current_class
                else:
                    sorted_indices = np.argsort(pred)[::-1]
                    desired_class = sorted_indices[1]  # 2e classe la plus probable
            else:
                pred = self._model.predict(instance)[0]
                if isinstance(pred, (int, float, np.integer, np.floating)):
                    # Pour la régression, augmenter la valeur de 20%
                    desired_class = pred * 1.2
                else:
                    # Pour la classification sans probabilités
                    desired_class = 1 - pred if pred in [0, 1] else None
        
        # Paramètres pour DiCE
        dice_params = {
            'total_CFs': num_counterfactuals,
            'desired_class': desired_class,
            'proximity_weight': self._proximity_weight,
            'diversity_weight': self._diversity_weight,
            'sparsity_weight': self._sparsity_weight
        }
        
        # Ajouter les caractéristiques immuables si spécifiées
        if immutable_features:
            dice_params['features_to_vary'] = [f for f in feature_names if f not in immutable_features]
        
        # Générer les contrefactuels
        cf_obj = exp.generate_counterfactuals(query_instance, **dice_params)
        
        # Extraire les contrefactuels
        if hasattr(cf_obj, 'cf_examples_list') and cf_obj.cf_examples_list:
            counterfactuals = cf_obj.cf_examples_list[0].final_cfs_df.values
            metadata = {
                'proximity': cf_obj.cf_examples_list[0].proximity,
                'sparsity': cf_obj.cf_examples_list[0].sparsity,
                'diversity': cf_obj.cf_examples_list[0].diversity,
                'desired_class': desired_class
            }
        else:
            # Fallback si DiCE échoue
            self._logger.warning("DiCE n'a pas réussi à générer des contrefactuels. Utilisation de contrefactuels aléatoires.")
            counterfactuals = self._generate_random_counterfactuals(instance, feature_names, num_counterfactuals)
            metadata = {
                'method': 'random_fallback',
                'desired_class': desired_class
            }
        
        return counterfactuals, metadata
    
    def _generate_genetic_counterfactuals(self, instance, feature_names, desired_class=None, 
                                        num_counterfactuals=3, feature_weights=None, immutable_features=None):
        """
        Génère des exemples contrefactuels en utilisant un algorithme génétique.
        
        Note: Cette méthode est un placeholder pour une implémentation future.
        Pour l'instant, elle génère des contrefactuels aléatoires.
        """
        self._logger.warning("L'algorithme génétique pour les contrefactuels n'est pas encore implémenté. "
                          "Utilisation de contrefactuels aléatoires.")
        
        counterfactuals = self._generate_random_counterfactuals(instance, feature_names, num_counterfactuals)
        metadata = {
            'method': 'random_placeholder',
            'desired_class': desired_class
        }
        
        return counterfactuals, metadata
    
    def _generate_optimization_counterfactuals(self, instance, feature_names, desired_class=None, 
                                            num_counterfactuals=3, feature_weights=None, immutable_features=None):
        """
        Génère des exemples contrefactuels en utilisant des méthodes d'optimisation.
        
        Note: Cette méthode est un placeholder pour une implémentation future.
        Pour l'instant, elle génère des contrefactuels aléatoires.
        """
        self._logger.warning("L'algorithme d'optimisation pour les contrefactuels n'est pas encore implémenté. "
                          "Utilisation de contrefactuels aléatoires.")
        
        counterfactuals = self._generate_random_counterfactuals(instance, feature_names, num_counterfactuals)
        metadata = {
            'method': 'random_placeholder',
            'desired_class': desired_class
        }
        
        return counterfactuals, metadata
    
    def _generate_random_counterfactuals(self, instance, feature_names, num_counterfactuals=3):
        """
        Génère des exemples contrefactuels aléatoires (pour fallback).
        """
        # Créer des variations aléatoires de l'instance
        counterfactuals = []
        for _ in range(num_counterfactuals):
            # Copier l'instance
            cf = instance.copy()
            
            # Modifier aléatoirement 1 à 3 caractéristiques
            n_features_to_change = np.random.randint(1, min(4, instance.shape[1] + 1))
            features_to_change = np.random.choice(instance.shape[1], n_features_to_change, replace=False)
            
            for idx in features_to_change:
                # Pour les caractéristiques continues, ajouter une perturbation
                if idx in self._continuous_features or not self._categorical_features:
                    cf[0, idx] = cf[0, idx] + np.random.normal(0, abs(cf[0, idx]) * 0.2 + 0.1)
                # Pour les caractéristiques catégorielles, changer la valeur
                elif idx in self._categorical_features:
                    current_val = cf[0, idx]
                    # Supposer que les valeurs catégorielles sont des entiers
                    possible_values = list(range(10))  # Valeurs arbitraires
                    new_val = np.random.choice([v for v in possible_values if v != current_val])
                    cf[0, idx] = new_val
            
            counterfactuals.append(cf[0])
        
        return np.array(counterfactuals)
    
    def _calculate_feature_changes(self, instance, counterfactuals, feature_names):
        """
        Calcule les changements de caractéristiques entre l'instance et les contrefactuels.
        
        Args:
            instance: Instance d'origine
            counterfactuals: Contrefactuels générés
            feature_names: Noms des caractéristiques
            
        Returns:
            dict: Changements de caractéristiques
        """
        changes = {}
        
        # Convertir en array numpy si nécessaire
        if not isinstance(counterfactuals, np.ndarray):
            counterfactuals = np.array(counterfactuals)
        
        # Pour chaque caractéristique
        for i, feature in enumerate(feature_names):
            if i >= instance.shape[0]:
                continue
                
            # Valeur d'origine
            original_value = instance[i]
            
            # Valeurs dans les contrefactuels
            if counterfactuals.ndim > 1 and i < counterfactuals.shape[1]:
                cf_values = counterfactuals[:, i]
            else:
                continue
            
            # Calculer les changements
            absolute_diffs = np.abs(cf_values - original_value)
            mean_diff = np.mean(absolute_diffs)
            max_diff = np.max(absolute_diffs)
            min_diff = np.min(absolute_diffs)
            std_diff = np.std(absolute_diffs)
            
            # Calculer le pourcentage de contrefactuels qui ont changé cette caractéristique
            changed_ratio = np.sum(absolute_diffs > 0) / len(cf_values)
            
            # Stocker les résultats
            changes[feature] = {
                'original_value': float(original_value),
                'mean_diff': float(mean_diff),
                'max_diff': float(max_diff),
                'min_diff': float(min_diff),
                'std_diff': float(std_diff),
                'changed_ratio': float(changed_ratio)
            }
        
        return changes
        
    def _aggregate_feature_changes(self, all_feature_changes, feature_names):
        """
        Agrège les changements de caractéristiques à partir de plusieurs instances.
        
        Args:
            all_feature_changes: Liste de dictionnaires de changements de caractéristiques
            feature_names: Noms des caractéristiques
            
        Returns:
            dict: Changements de caractéristiques agrégés
        """
        aggregated = {}
        
        # Pour chaque caractéristique
        for feature in feature_names:
            # Collecter les statistiques de tous les changements
            mean_diffs = []
            max_diffs = []
            changed_ratios = []
            
            for changes in all_feature_changes:
                if feature in changes:
                    mean_diffs.append(changes[feature]['mean_diff'])
                    max_diffs.append(changes[feature]['max_diff'])
                    changed_ratios.append(changes[feature]['changed_ratio'])
            
            # Calculer les statistiques agrégées
            if mean_diffs:
                aggregated[feature] = {
                    'mean_diff': float(np.mean(mean_diffs)),
                    'max_diff': float(np.max(max_diffs)),
                    'changed_ratio': float(np.mean(changed_ratios)),
                    'importance_score': float(np.mean(changed_ratios) * np.mean(mean_diffs))
                }
            else:
                aggregated[feature] = {
                    'mean_diff': 0.0,
                    'max_diff': 0.0,
                    'changed_ratio': 0.0,
                    'importance_score': 0.0
                }
        
        return aggregated
    
    def _convert_changes_to_importances(self, feature_changes):
        """
        Convertit les changements de caractéristiques en importances de caractéristiques.
        
        Args:
            feature_changes: Dictionnaire de changements de caractéristiques
            
        Returns:
            list: Liste d'objets FeatureImportance
        """
        importances = []
        
        # Calculer un score d'importance pour chaque caractéristique
        for feature, changes in feature_changes.items():
            # Pour les changements agrégés
            if 'importance_score' in changes:
                importance = changes['importance_score']
            # Pour les changements d'une seule instance
            else:
                # Le score d'importance est une combinaison de la fréquence de changement et de l'ampleur du changement
                importance = changes['changed_ratio'] * changes['mean_diff']
            
            importances.append(FeatureImportance(
                feature_name=feature,
                importance=float(importance),
                std=float(changes.get('std_diff', 0.0)),
                additional_info={
                    'changed_ratio': float(changes['changed_ratio']),
                    'mean_diff': float(changes['mean_diff']),
                    'max_diff': float(changes.get('max_diff', 0.0))
                }
            ))
        
        # Normaliser les importances
        total_importance = sum(imp.importance for imp in importances)
        if total_importance > 0:
            for imp in importances:
                imp.importance = imp.importance / total_importance
        
        # Trier par importance décroissante
        importances.sort(key=lambda x: x.importance, reverse=True)
        
        return importances
    
    def _extract_metadata(self):
        """
        Extrait les métadonnées du modèle.
        """
        # Déterminer le type de modèle
        model_type = self._get_model_type()
        
        # Déterminer le type de tâche (classification ou régression)
        task_type = self._get_task_type()
        
        # Créer les métadonnées du modèle
        self._metadata = ModelMetadata(
            model_type=model_type,
            framework=model_type,  # Utiliser le type comme framework pour simplifier
            task_type=task_type,
            feature_names=self._feature_names,
            target_names=None,  # Pas disponible pour les contrefactuels
            model_parameters={
                'counterfactual_method': self._counterfactual_method,
                'proximity_weight': self._proximity_weight,
                'diversity_weight': self._diversity_weight,
                'sparsity_weight': self._sparsity_weight
            }
        )
    
    def _get_model_type(self):
        """
        Détermine le type de modèle (framework).
        
        Returns:
            str: Type de modèle (sklearn, tensorflow, pytorch, etc.)
        """
        model_module = self._model.__module__.split('.')[0].lower()
        
        if model_module in ['sklearn', 'scikit']:
            return 'sklearn'
        elif model_module in ['tensorflow', 'tf', 'keras']:
            return 'tensorflow'
        elif model_module in ['torch', 'pytorch']:
            return 'pytorch'
        elif model_module in ['xgboost']:
            return 'xgboost'
        elif model_module in ['lightgbm']:
            return 'lightgbm'
        elif model_module in ['catboost']:
            return 'catboost'
        else:
            return 'unknown'
    
    def _get_task_type(self):
        """
        Détermine le type de tâche (classification ou régression).
        
        Returns:
            str: Type de tâche ('classification' ou 'regression')
        """
        # Vérifier les méthodes disponibles sur le modèle
        if hasattr(self._model, 'predict_proba'):
            return 'classification'
        elif hasattr(self._model, '_estimator_type'):
            return self._model._estimator_type
        else:
            # Par défaut, supposer que c'est une classification
            return 'classification'
