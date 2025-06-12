"""
Feature Importance Explainer pour XPLIA
======================================

Ce module implémente l'explainer basé sur les importances de caractéristiques natives des modèles
dans le framework XPLIA. Cette approche utilise les méthodes intégrées des modèles pour
déterminer l'importance des caractéristiques.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..core.base import (AudienceLevel, ExplainerBase, ExplainabilityMethod,
                        ExplanationResult, FeatureImportance, ModelMetadata)
from ..core.registry import register_explainer


@register_explainer
class FeatureImportanceExplainer(ExplainerBase):
    """
    Explainer basé sur les importances de caractéristiques natives des modèles.
    
    Cette classe extrait les importances de caractéristiques directement à partir
    des attributs et méthodes intégrés des modèles, comme feature_importances_
    pour les modèles basés sur des arbres ou les coefficients pour les modèles linéaires.
    
    Caractéristiques principales:
    - Support de différents types de modèles (arbres, linéaires, réseaux de neurones)
    - Extraction des importances natives des modèles
    - Normalisation et standardisation des importances
    - Visualisations adaptées au niveau d'audience (technique, business, public)
    """
    
    def __init__(self, model, **kwargs):
        """
        Initialise l'explainer d'importance de caractéristiques.
        
        Args:
            model: Modèle à expliquer
            **kwargs: Paramètres additionnels
                feature_names: Noms des caractéristiques
                normalize: Normaliser les importances (True par défaut)
                absolute: Utiliser les valeurs absolues pour les coefficients (True par défaut)
                permutation: Utiliser l'importance par permutation (False par défaut)
                n_repeats: Nombre de répétitions pour l'importance par permutation
        """
        super().__init__(model, **kwargs)
        self._method = ExplainabilityMethod.FEATURE_IMPORTANCE
        self._supported_model_types = [
            'RandomForestClassifier', 'RandomForestRegressor',
            'GradientBoostingClassifier', 'GradientBoostingRegressor',
            'XGBClassifier', 'XGBRegressor',
            'LGBMClassifier', 'LGBMRegressor',
            'CatBoostClassifier', 'CatBoostRegressor',
            'LogisticRegression', 'LinearRegression',
            'Ridge', 'Lasso', 'ElasticNet',
            'SVC', 'SVR',
            'Sequential', 'Model',  # Keras
            'Module'  # PyTorch
        ]
        
        # Paramètres
        self._feature_names = kwargs.get('feature_names', None)
        self._normalize = kwargs.get('normalize', True)
        self._absolute = kwargs.get('absolute', True)
        self._permutation = kwargs.get('permutation', False)
        self._n_repeats = kwargs.get('n_repeats', 5)
        
        # Métadonnées du modèle
        self._metadata = None
        
        # Logger
        self._logger = logging.getLogger(__name__)
        
    def explain(self, X, y=None, **kwargs) -> ExplanationResult:
        """
        Génère des explications basées sur l'importance des caractéristiques.
        
        Args:
            X: Données d'entrée (utilisées pour l'importance par permutation si activée)
            y: Valeurs cibles réelles (utilisées pour l'importance par permutation si activée)
            **kwargs: Paramètres additionnels
                audience_level: Niveau d'audience (TECHNICAL, BUSINESS, PUBLIC)
                
        Returns:
            ExplanationResult: Résultat standardisé de l'explication
        """
        # Paramètres
        audience_level = kwargs.get('audience_level', AudienceLevel.TECHNICAL)
        
        # Conversion des données en format approprié pour l'importance par permutation
        if self._permutation and X is not None:
            if isinstance(X, pd.DataFrame):
                feature_names = X.columns.tolist()
                X_values = X.values
            else:
                X_values = np.array(X)
                feature_names = kwargs.get('feature_names', self._feature_names) or \
                               [f"feature_{i}" for i in range(X_values.shape[1])]
        else:
            feature_names = kwargs.get('feature_names', self._feature_names)
            X_values = X
        
        # Tracer l'action
        self.add_audit_record("explain", {
            "audience_level": audience_level.value if isinstance(audience_level, AudienceLevel) else audience_level,
            "permutation": self._permutation,
            "normalize": self._normalize,
            "absolute": self._absolute
        })
        
        try:
            # Extraire les importances de caractéristiques
            if self._permutation and X is not None and y is not None:
                feature_importances = self._get_permutation_importance(X_values, y, feature_names)
            else:
                feature_importances = self._get_native_importance(feature_names)
            
            # Extraire les métadonnées du modèle
            if not self._metadata:
                self._extract_metadata()
            
            # Créer le résultat d'explication
            result = ExplanationResult(
                method=ExplainabilityMethod.FEATURE_IMPORTANCE,
                model_metadata=self._metadata,
                feature_importances=feature_importances,
                raw_explanation={
                    "importance_type": "permutation" if self._permutation else "native",
                    "normalized": self._normalize,
                    "absolute_values": self._absolute
                },
                audience_level=audience_level
            )
            
            return result
            
        except Exception as e:
            self._logger.error(f"Erreur lors du calcul des importances de caractéristiques: {str(e)}")
            raise RuntimeError(f"Échec de l'explication par importance de caractéristiques: {str(e)}")
    
    def explain_instance(self, instance, **kwargs) -> ExplanationResult:
        """
        Pour l'explainer d'importance de caractéristiques, cette méthode renvoie
        les mêmes importances globales que explain(), car cette méthode ne fournit
        pas d'explications spécifiques aux instances.
        
        Args:
            instance: Instance à expliquer (non utilisée)
            **kwargs: Paramètres additionnels
                
        Returns:
            ExplanationResult: Résultat standardisé de l'explication
        """
        self._logger.warning("L'explainer d'importance de caractéristiques ne fournit pas "
                           "d'explications spécifiques aux instances. Utilisation des importances globales.")
        
        # Utiliser la méthode explain standard
        return self.explain(None, None, **kwargs)
    
    def _get_native_importance(self, feature_names=None):
        """
        Extrait les importances de caractéristiques natives du modèle.
        
        Args:
            feature_names: Noms des caractéristiques
            
        Returns:
            List[FeatureImportance]: Liste des importances de caractéristiques
        """
        model_type = self._get_model_type()
        importances = None
        
        # Extraire les importances selon le type de modèle
        if hasattr(self._model, 'feature_importances_'):
            # Pour les modèles basés sur des arbres (RandomForest, XGBoost, etc.)
            importances = self._model.feature_importances_
        elif hasattr(self._model, 'coef_'):
            # Pour les modèles linéaires (LogisticRegression, LinearRegression, etc.)
            coef = self._model.coef_
            if coef.ndim > 1:
                # Pour les modèles multi-classes, prendre la moyenne des coefficients
                importances = np.mean(np.abs(coef) if self._absolute else coef, axis=0)
            else:
                importances = np.abs(coef) if self._absolute else coef
        elif hasattr(self._model, 'feature_importance'):
            # Pour certains modèles comme LightGBM, CatBoost
            try:
                importances = self._model.feature_importance()
            except:
                pass
        elif 'tensorflow' in self._model.__class__.__module__ or 'keras' in self._model.__class__.__module__:
            # Pour les modèles Keras/TensorFlow, utiliser les gradients
            importances = self._get_gradient_importance()
        elif 'torch' in self._model.__class__.__module__:
            # Pour les modèles PyTorch, utiliser les gradients
            importances = self._get_gradient_importance()
            
        if importances is None:
            raise ValueError("Impossible d'extraire les importances de caractéristiques du modèle. "
                           "Utilisez l'option permutation=True pour calculer l'importance par permutation.")
            
        # Normaliser si demandé
        if self._normalize and importances is not None:
            if np.sum(np.abs(importances)) > 0:
                importances = importances / np.sum(np.abs(importances))
            
        # Créer les objets FeatureImportance
        feature_importances = []
        if feature_names is None:
            # Essayer d'extraire les noms des caractéristiques du modèle
            if hasattr(self._model, 'feature_names_in_'):
                feature_names = self._model.feature_names_in_
            else:
                feature_names = [f"feature_{i}" for i in range(len(importances))]
                
        for name, importance in zip(feature_names, importances):
            feature_importances.append(FeatureImportance(
                feature_name=name,
                importance=float(importance)
            ))
            
        # Trier par importance absolue décroissante
        feature_importances.sort(key=lambda x: abs(x.importance), reverse=True)
        
        return feature_importances
    
    def _get_permutation_importance(self, X, y, feature_names=None):
        """
        Calcule l'importance des caractéristiques par permutation.
        
        Args:
            X: Données d'entrée
            y: Valeurs cibles
            feature_names: Noms des caractéristiques
            
        Returns:
            List[FeatureImportance]: Liste des importances de caractéristiques
        """
        try:
            from sklearn.inspection import permutation_importance
        except ImportError:
            raise ImportError("scikit-learn est requis pour l'importance par permutation. "
                            "Installez-le avec 'pip install scikit-learn'.")
            
        # Calculer l'importance par permutation
        result = permutation_importance(
            self._model, X, y,
            n_repeats=self._n_repeats,
            random_state=42
        )
        
        importances = result.importances_mean
        std_devs = result.importances_std
        
        # Normaliser si demandé
        if self._normalize and importances is not None:
            if np.sum(np.abs(importances)) > 0:
                importances = importances / np.sum(np.abs(importances))
            
        # Créer les objets FeatureImportance
        feature_importances = []
        if feature_names is None:
            # Essayer d'extraire les noms des caractéristiques du modèle
            if hasattr(self._model, 'feature_names_in_'):
                feature_names = self._model.feature_names_in_
            else:
                feature_names = [f"feature_{i}" for i in range(len(importances))]
                
        for name, importance, std_dev in zip(feature_names, importances, std_devs):
            feature_importances.append(FeatureImportance(
                feature_name=name,
                importance=float(importance),
                std_dev=float(std_dev)
            ))
            
        # Trier par importance absolue décroissante
        feature_importances.sort(key=lambda x: abs(x.importance), reverse=True)
        
        return feature_importances
    
    def _get_gradient_importance(self):
        """
        Calcule l'importance des caractéristiques basée sur les gradients pour les modèles de deep learning.
        
        Returns:
            numpy.ndarray: Importances des caractéristiques
        """
        # Cette méthode est un placeholder pour l'implémentation des gradients
        # Elle nécessiterait des données d'entrée et une implémentation spécifique
        # selon le framework (TensorFlow/Keras ou PyTorch)
        self._logger.warning("L'importance par gradients n'est pas encore implémentée. "
                           "Utilisez l'option permutation=True pour les modèles de deep learning.")
        return None
    
    def _get_model_type(self):
        """
        Détermine le type de modèle.
        
        Returns:
            str: Type de modèle
        """
        model_class = self._model.__class__.__name__
        model_module = self._model.__class__.__module__
        
        if 'ensemble' in model_module:
            return 'ensemble'
        elif 'linear_model' in model_module:
            return 'linear'
        elif 'svm' in model_module:
            return 'svm'
        elif 'xgboost' in model_module:
            return 'xgboost'
        elif 'lightgbm' in model_module:
            return 'lightgbm'
        elif 'catboost' in model_module:
            return 'catboost'
        elif 'tensorflow' in model_module or 'keras' in model_module:
            return 'tensorflow'
        elif 'torch' in model_module:
            return 'pytorch'
        else:
            return 'unknown'
    
    def _extract_metadata(self) -> None:
        """
        Extrait les métadonnées du modèle pour l'explication.
        """
        model_type = self._get_model_type()
        
        # Déterminer le type de modèle (classification ou régression)
        is_classifier = False
        if hasattr(self._model, 'predict_proba'):
            is_classifier = True
        elif hasattr(self._model, 'classes_'):
            is_classifier = True
        elif model_type in ['tensorflow', 'pytorch']:
            # Pour les modèles deep learning, c'est plus difficile à déterminer
            # sans données d'entrée
            pass
        
        # Créer les métadonnées
        self._metadata = ModelMetadata(
            model_type="classification" if is_classifier else "regression",
            framework=model_type,
            input_shape=self._infer_input_shape(),
            output_shape=None,  # À compléter si nécessaire
            feature_names=self._feature_names,
            target_names=None,  # À compléter si disponible
            model_params={},
            model_version="1.0.0"
        )
        
    def _infer_input_shape(self):
        """
        Tente de déduire la forme des entrées du modèle.
        
        Returns:
            tuple ou None: Forme déduite ou None si impossible
        """
        # Pour les modèles scikit-learn
        if hasattr(self._model, 'n_features_in_'):
            return (self._model.n_features_in_,)
        
        # Pour les modèles Keras/TensorFlow
        if 'tensorflow' in self._model.__class__.__module__ or 'keras' in self._model.__class__.__module__:
            try:
                return self._model.input_shape[1:]
            except (AttributeError, IndexError):
                pass
        
        # Impossible de déduire
        return None
