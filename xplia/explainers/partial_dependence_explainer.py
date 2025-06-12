"""
Partial Dependence Explainer pour XPLIA
======================================

Ce module implémente l'explainer basé sur les graphiques de dépendance partielle (PDP)
dans le framework XPLIA. Cette approche permet de visualiser l'effet marginal d'une ou
plusieurs caractéristiques sur la prédiction d'un modèle.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..core.base import (AudienceLevel, ExplainerBase, ExplainabilityMethod,
                        ExplanationResult, FeatureImportance, ModelMetadata)
from ..core.registry import register_explainer


@register_explainer
class PartialDependenceExplainer(ExplainerBase):
    """
    Explainer basé sur les graphiques de dépendance partielle (PDP).
    
    Cette classe calcule et visualise comment la prédiction moyenne du modèle
    change en fonction des valeurs d'une caractéristique spécifique, en
    marginalisant sur toutes les autres caractéristiques.
    
    Caractéristiques principales:
    - Calcul de la dépendance partielle pour une ou plusieurs caractéristiques
    - Support des interactions entre caractéristiques (PDP 2D)
    - Visualisations adaptées au niveau d'audience
    - Optimisations pour les grands ensembles de données
    """
    
    def __init__(self, model, **kwargs):
        """
        Initialise l'explainer de dépendance partielle.
        
        Args:
            model: Modèle à expliquer
            **kwargs: Paramètres additionnels
                feature_names: Noms des caractéristiques
                grid_resolution: Résolution de la grille pour calculer les PDP (10 par défaut)
                percentiles: Utiliser des percentiles pour définir la grille (True par défaut)
                grid_type: Type de grille ('percentile' ou 'uniform')
        """
        super().__init__(model, **kwargs)
        self._method = ExplainabilityMethod.PARTIAL_DEPENDENCE
        
        # Paramètres
        self._feature_names = kwargs.get('feature_names', None)
        self._grid_resolution = kwargs.get('grid_resolution', 10)
        self._percentiles = kwargs.get('percentiles', True)
        self._grid_type = kwargs.get('grid_type', 'percentile')
        
        # Métadonnées du modèle
        self._metadata = None
        
        # Logger
        self._logger = logging.getLogger(__name__)
        
    def explain(self, X, y=None, **kwargs) -> ExplanationResult:
        """
        Génère des explications basées sur la dépendance partielle.
        
        Args:
            X: Données d'entrée pour calculer les dépendances partielles
            y: Valeurs cibles réelles (non utilisées pour PDP)
            **kwargs: Paramètres additionnels
                features: Liste des caractéristiques à expliquer
                interactions: Liste de paires de caractéristiques pour les interactions
                target_idx: Indice de la classe cible pour les classifieurs multi-classes
                audience_level: Niveau d'audience (TECHNICAL, BUSINESS, PUBLIC)
                
        Returns:
            ExplanationResult: Résultat standardisé de l'explication
        """
        # Paramètres
        audience_level = kwargs.get('audience_level', AudienceLevel.TECHNICAL)
        features = kwargs.get('features', None)
        interactions = kwargs.get('interactions', None)
        target_idx = kwargs.get('target_idx', 1)  # Par défaut, classe positive pour binaire
        
        # Conversion des données en format approprié
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X_values = X.values
        else:
            X_values = np.array(X)
            feature_names = kwargs.get('feature_names', self._feature_names) or \
                           [f"feature_{i}" for i in range(X_values.shape[1])]
        
        # Si aucune caractéristique n'est spécifiée, sélectionner les plus importantes
        if features is None:
            try:
                # Essayer d'utiliser un explainer d'importance de caractéristiques
                from .feature_importance_explainer import FeatureImportanceExplainer
                fi_explainer = FeatureImportanceExplainer(self._model, feature_names=feature_names)
                fi_result = fi_explainer.explain(X, y)
                top_features = [fi.feature_name for fi in fi_result.feature_importances[:5]]
                features = top_features
            except Exception:
                # Sinon, prendre les 5 premières caractéristiques
                features = feature_names[:5]
        
        # Tracer l'action
        self.add_audit_record("explain", {
            "n_samples": X_values.shape[0],
            "n_features": X_values.shape[1],
            "audience_level": audience_level.value if isinstance(audience_level, AudienceLevel) else audience_level,
            "features": features,
            "interactions": interactions,
            "grid_resolution": self._grid_resolution
        })
        
        try:
            # Calculer les dépendances partielles
            pdp_results = self._compute_partial_dependence(X_values, features, feature_names, target_idx)
            
            # Calculer les interactions si demandées
            interaction_results = None
            if interactions:
                interaction_results = self._compute_interactions(X_values, interactions, feature_names, target_idx)
            
            # Extraire les métadonnées du modèle
            if not self._metadata:
                self._extract_metadata()
            
            # Créer le résultat d'explication
            result = ExplanationResult(
                method=ExplainabilityMethod.PARTIAL_DEPENDENCE,
                model_metadata=self._metadata,
                feature_importances=None,  # PDP ne fournit pas d'importances directes
                raw_explanation={
                    "pdp_results": pdp_results,
                    "interaction_results": interaction_results,
                    "features": features,
                    "interactions": interactions,
                    "feature_names": feature_names,
                    "grid_resolution": self._grid_resolution,
                    "grid_type": self._grid_type
                },
                audience_level=audience_level
            )
            
            return result
            
        except Exception as e:
            self._logger.error(f"Erreur lors du calcul des dépendances partielles: {str(e)}")
            raise RuntimeError(f"Échec de l'explication par dépendance partielle: {str(e)}")
    
    def explain_instance(self, instance, X_background, **kwargs) -> ExplanationResult:
        """
        Pour l'explainer de dépendance partielle, cette méthode calcule l'Individual
        Conditional Expectation (ICE) pour une instance spécifique.
        
        Args:
            instance: Instance à expliquer
            X_background: Données d'arrière-plan pour le calcul
            **kwargs: Paramètres additionnels
                features: Liste des caractéristiques à expliquer
                target_idx: Indice de la classe cible pour les classifieurs multi-classes
                audience_level: Niveau d'audience (TECHNICAL, BUSINESS, PUBLIC)
                
        Returns:
            ExplanationResult: Résultat standardisé de l'explication
        """
        # Paramètres
        audience_level = kwargs.get('audience_level', AudienceLevel.TECHNICAL)
        features = kwargs.get('features', None)
        target_idx = kwargs.get('target_idx', 1)  # Par défaut, classe positive pour binaire
        
        # Conversion des données en format approprié
        if isinstance(X_background, pd.DataFrame):
            feature_names = X_background.columns.tolist()
            X_values = X_background.values
        else:
            X_values = np.array(X_background)
            feature_names = kwargs.get('feature_names', self._feature_names) or \
                           [f"feature_{i}" for i in range(X_values.shape[1])]
        
        # Convertir l'instance en format approprié
        if isinstance(instance, dict):
            # Convertir dict en array
            instance_array = np.array([instance[f] for f in feature_names]).reshape(1, -1)
        elif isinstance(instance, pd.Series):
            instance_array = instance.values.reshape(1, -1)
        elif isinstance(instance, (list, np.ndarray)):
            instance_array = np.array(instance).reshape(1, -1)
        else:
            raise ValueError("Format d'instance non supporté. Utilisez un dict, pandas.Series, liste ou numpy.ndarray.")
        
        # Si aucune caractéristique n'est spécifiée, sélectionner les plus importantes
        if features is None:
            try:
                # Essayer d'utiliser un explainer d'importance de caractéristiques
                from .feature_importance_explainer import FeatureImportanceExplainer
                fi_explainer = FeatureImportanceExplainer(self._model, feature_names=feature_names)
                fi_result = fi_explainer.explain(X_background, None)
                top_features = [fi.feature_name for fi in fi_result.feature_importances[:5]]
                features = top_features
            except Exception:
                # Sinon, prendre les 5 premières caractéristiques
                features = feature_names[:5]
        
        # Tracer l'action
        self.add_audit_record("explain_instance", {
            "n_features": len(feature_names),
            "audience_level": audience_level.value if isinstance(audience_level, AudienceLevel) else audience_level,
            "features": features,
            "grid_resolution": self._grid_resolution
        })
        
        try:
            # Calculer les ICE pour l'instance
            ice_results = self._compute_ice_curves(X_values, instance_array, features, feature_names, target_idx)
            
            # Extraire les métadonnées du modèle
            if not self._metadata:
                self._extract_metadata()
            
            # Créer le résultat d'explication
            result = ExplanationResult(
                method=ExplainabilityMethod.PARTIAL_DEPENDENCE,
                model_metadata=self._metadata,
                feature_importances=None,  # ICE ne fournit pas d'importances directes
                raw_explanation={
                    "ice_results": ice_results,
                    "pdp_results": self._compute_partial_dependence(X_values, features, feature_names, target_idx),
                    "features": features,
                    "feature_names": feature_names,
                    "instance": instance_array,
                    "grid_resolution": self._grid_resolution,
                    "grid_type": self._grid_type
                },
                audience_level=audience_level
            )
            
            return result
            
        except Exception as e:
            self._logger.error(f"Erreur lors du calcul des courbes ICE: {str(e)}")
            raise RuntimeError(f"Échec de l'explication ICE pour l'instance: {str(e)}")
    
    def _compute_partial_dependence(self, X, features, feature_names, target_idx=1):
        """
        Calcule les dépendances partielles pour les caractéristiques spécifiées.
        
        Args:
            X: Données d'entrée
            features: Liste des caractéristiques à expliquer (noms ou indices)
            feature_names: Noms des caractéristiques
            target_idx: Indice de la classe cible pour les classifieurs multi-classes
            
        Returns:
            dict: Résultats des dépendances partielles
        """
        try:
            from sklearn.inspection import partial_dependence
        except ImportError:
            raise ImportError("scikit-learn est requis pour le calcul des dépendances partielles. "
                            "Installez-le avec 'pip install scikit-learn'.")
        
        # Convertir les noms de caractéristiques en indices si nécessaire
        feature_indices = []
        for feature in features:
            if isinstance(feature, str):
                if feature in feature_names:
                    feature_indices.append(feature_names.index(feature))
                else:
                    raise ValueError(f"Caractéristique '{feature}' non trouvée dans les données.")
            else:
                feature_indices.append(feature)
        
        # Paramètres pour le calcul des dépendances partielles
        grid_params = {}
        if self._grid_type == 'percentile':
            grid_params = {
                'percentiles': (0, 1),
                'grid_resolution': self._grid_resolution
            }
        else:
            grid_params = {
                'grid_resolution': self._grid_resolution
            }
        
        # Calculer les dépendances partielles
        pdp_results = {}
        for idx in feature_indices:
            feature_name = feature_names[idx]
            
            # Calculer la dépendance partielle
            pdp_result = partial_dependence(
                self._model, X, [idx],
                kind='average',
                **grid_params
            )
            
            # Extraire les résultats
            grid_values = pdp_result['values'][0]
            pdp_values = pdp_result['average'][0]
            
            # Stocker les résultats
            pdp_results[feature_name] = {
                'grid_values': grid_values.tolist(),
                'pdp_values': pdp_values.tolist()
            }
            
        return pdp_results
    
    def _compute_interactions(self, X, interactions, feature_names, target_idx=1):
        """
        Calcule les interactions de dépendance partielle pour les paires de caractéristiques spécifiées.
        
        Args:
            X: Données d'entrée
            interactions: Liste des paires de caractéristiques à expliquer
            feature_names: Noms des caractéristiques
            target_idx: Indice de la classe cible pour les classifieurs multi-classes
            
        Returns:
            dict: Résultats des interactions
        """
        try:
            from sklearn.inspection import partial_dependence
        except ImportError:
            raise ImportError("scikit-learn est requis pour le calcul des dépendances partielles. "
                            "Installez-le avec 'pip install scikit-learn'.")
        
        # Convertir les noms de caractéristiques en indices si nécessaire
        interaction_indices = []
        for pair in interactions:
            if isinstance(pair[0], str) and isinstance(pair[1], str):
                if pair[0] in feature_names and pair[1] in feature_names:
                    idx1 = feature_names.index(pair[0])
                    idx2 = feature_names.index(pair[1])
                    interaction_indices.append((idx1, idx2))
                else:
                    raise ValueError(f"Caractéristique '{pair[0]}' ou '{pair[1]}' non trouvée dans les données.")
            else:
                interaction_indices.append(pair)
        
        # Paramètres pour le calcul des dépendances partielles
        grid_params = {}
        if self._grid_type == 'percentile':
            grid_params = {
                'percentiles': (0, 1),
                'grid_resolution': self._grid_resolution
            }
        else:
            grid_params = {
                'grid_resolution': self._grid_resolution
            }
        
        # Calculer les interactions
        interaction_results = {}
        for idx_pair in interaction_indices:
            feature1 = feature_names[idx_pair[0]]
            feature2 = feature_names[idx_pair[1]]
            pair_key = f"{feature1}_{feature2}"
            
            # Calculer l'interaction
            pdp_result = partial_dependence(
                self._model, X, [idx_pair],
                kind='average',
                **grid_params
            )
            
            # Extraire les résultats
            grid_values1 = pdp_result['values'][0][0]
            grid_values2 = pdp_result['values'][0][1]
            pdp_values = pdp_result['average'][0]
            
            # Stocker les résultats
            interaction_results[pair_key] = {
                'grid_values1': grid_values1.tolist(),
                'grid_values2': grid_values2.tolist(),
                'pdp_values': pdp_values.tolist(),
                'feature1': feature1,
                'feature2': feature2
            }
            
        return interaction_results
    
    def _compute_ice_curves(self, X, instance, features, feature_names, target_idx=1):
        """
        Calcule les courbes ICE (Individual Conditional Expectation) pour une instance spécifique.
        
        Args:
            X: Données d'arrière-plan
            instance: Instance à expliquer
            features: Liste des caractéristiques à expliquer
            feature_names: Noms des caractéristiques
            target_idx: Indice de la classe cible pour les classifieurs multi-classes
            
        Returns:
            dict: Résultats des courbes ICE
        """
        try:
            from sklearn.inspection import partial_dependence
        except ImportError:
            raise ImportError("scikit-learn est requis pour le calcul des courbes ICE. "
                            "Installez-le avec 'pip install scikit-learn'.")
        
        # Convertir les noms de caractéristiques en indices si nécessaire
        feature_indices = []
        for feature in features:
            if isinstance(feature, str):
                if feature in feature_names:
                    feature_indices.append(feature_names.index(feature))
                else:
                    raise ValueError(f"Caractéristique '{feature}' non trouvée dans les données.")
            else:
                feature_indices.append(feature)
        
        # Paramètres pour le calcul des dépendances partielles
        grid_params = {}
        if self._grid_type == 'percentile':
            grid_params = {
                'percentiles': (0, 1),
                'grid_resolution': self._grid_resolution
            }
        else:
            grid_params = {
                'grid_resolution': self._grid_resolution
            }
        
        # Calculer les courbes ICE
        ice_results = {}
        for idx in feature_indices:
            feature_name = feature_names[idx]
            
            # Calculer la courbe ICE
            pdp_result = partial_dependence(
                self._model, X, [idx],
                kind='individual',
                **grid_params
            )
            
            # Extraire les résultats
            grid_values = pdp_result['values'][0]
            
            # Créer un ensemble de données synthétique basé sur l'instance
            X_synth = np.tile(instance, (len(grid_values), 1))
            for i, val in enumerate(grid_values):
                X_synth[i, idx] = val
            
            # Prédire avec le modèle
            if hasattr(self._model, 'predict_proba'):
                preds = self._model.predict_proba(X_synth)
                if preds.shape[1] > 1:
                    ice_values = preds[:, target_idx]
                else:
                    ice_values = preds.ravel()
            else:
                ice_values = self._model.predict(X_synth)
            
            # Stocker les résultats
            ice_results[feature_name] = {
                'grid_values': grid_values.tolist(),
                'ice_values': ice_values.tolist(),
                'instance_value': instance[0, idx]
            }
            
        return ice_results
    
    def _extract_metadata(self) -> None:
        """
        Extrait les métadonnées du modèle pour l'explication.
        """
        model_module = self._model.__class__.__module__
        framework = None
        
        # Déterminer le framework
        if 'sklearn' in model_module:
            framework = 'sklearn'
        elif 'xgboost' in model_module:
            framework = 'xgboost'
        elif 'lightgbm' in model_module:
            framework = 'lightgbm'
        elif 'catboost' in model_module:
            framework = 'catboost'
        elif 'tensorflow' in model_module or 'keras' in model_module:
            framework = 'tensorflow'
        elif 'torch' in model_module:
            framework = 'pytorch'
        else:
            framework = 'unknown'
        
        # Déterminer le type de modèle (classification ou régression)
        is_classifier = False
        if hasattr(self._model, 'predict_proba'):
            is_classifier = True
        elif hasattr(self._model, 'classes_'):
            is_classifier = True
        
        # Créer les métadonnées
        self._metadata = ModelMetadata(
            model_type="classification" if is_classifier else "regression",
            framework=framework,
            input_shape=None,  # À compléter si nécessaire
            output_shape=None,  # À compléter si nécessaire
            feature_names=self._feature_names,
            target_names=None,  # À compléter si disponible
            model_params={},
            model_version="1.0.0"
        )
