"""
Adaptateur pour les modèles scikit-learn
======================================

Ce module fournit un adaptateur pour les modèles de scikit-learn.
"""

import inspect
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
try:
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Définitions minimales pour la compatibilité des types
    class BaseEstimator: pass
    class Pipeline: pass

from ...core import ModelMetadata, ModelType, ModelAdapterBase  # Import depuis le package principal
from ...core.registry import register_model_adapter

@register_model_adapter(version="1.0.0", description="Adaptateur pour les modèles scikit-learn")
class SklearnModelAdapter(ModelAdapterBase):
    """
    Adaptateur pour les modèles scikit-learn.
    
    Cet adaptateur prend en charge:
    - Tous les classifieurs et régresseurs scikit-learn
    - Les Pipelines scikit-learn
    - Les modèles avec et sans probabilités
    """
    
    def __init__(self, model: BaseEstimator, **kwargs):
        """
        Initialise l'adaptateur pour un modèle scikit-learn.
        
        Args:
            model: Modèle scikit-learn à adapter
            **kwargs: Arguments additionnels
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn n'est pas installé. Installez-le avec 'pip install scikit-learn'.")
            
        if not isinstance(model, BaseEstimator):
            raise ValueError("Le modèle doit être une instance de sklearn.base.BaseEstimator")
            
        super().__init__(model, **kwargs)
        self._framework = "sklearn"
        self._extract_feature_names()
        
    def _extract_metadata(self, **kwargs) -> ModelMetadata:
        """Extrait les métadonnées du modèle scikit-learn."""
        # Déterminer le type de modèle
        if isinstance(self.model, Pipeline):
            final_estimator = self.model.steps[-1][1]
            is_classifier = hasattr(final_estimator, 'predict_proba') or hasattr(final_estimator, 'classes_')
        else:
            is_classifier = hasattr(self.model, 'predict_proba') or hasattr(self.model, 'classes_')
            
        self._model_type = ModelType.CLASSIFICATION if is_classifier else ModelType.REGRESSION
            
        # Extraire les classes pour les classifieurs
        classes = None
        if hasattr(self.model, 'classes_'):
            classes = self.model.classes_.tolist()
        elif isinstance(self.model, Pipeline) and hasattr(self.model.steps[-1][1], 'classes_'):
            classes = self.model.steps[-1][1].classes_.tolist()
            
        # Extraire les noms de caractéristiques si disponibles
        feature_names = self.get_feature_names()
        
        # Extraire les paramètres du modèle
        try:
            model_params = self.model.get_params()
        except:
            model_params = {}
            
        # Extraire les informations sur l'entraînement si disponibles
        training_info = {}
        if hasattr(self.model, 'n_features_in_'):
            training_info['n_features'] = self.model.n_features_in_
        if hasattr(self.model, 'n_classes_'):
            training_info['n_classes'] = self.model.n_classes_
        if hasattr(self.model, 'feature_importances_'):
            training_info['has_feature_importances'] = True
            
        return ModelMetadata(
            framework='sklearn',
            model_type=self._model_type,
            model_class=self.model.__class__.__name__,
            classes=classes,
            feature_names=feature_names,
            model_params=model_params,
            training_info=training_info
        )
        
    def _extract_feature_names(self):
        """Tente d'extraire les noms des caractéristiques du modèle."""
        # Si les noms sont déjà définis, ne rien faire
        if self._feature_names is not None:
            return
            
        # Pour les pipelines, essayer d'extraire depuis le dernier step
        if isinstance(self.model, Pipeline):
            last_step = self.model.steps[-1][1]
            if hasattr(last_step, 'feature_names_in_'):
                self.set_feature_names(last_step.feature_names_in_.tolist())
            return
                
        # Pour les modèles standards avec feature_names_in_
        if hasattr(self.model, 'feature_names_in_'):
            self.set_feature_names(self.model.feature_names_in_.tolist())
            
    def predict(self, X: Union[np.ndarray, pd.DataFrame], **kwargs) -> np.ndarray:
        """
        Effectue une prédiction avec le modèle.
        
        Args:
            X: Données d'entrée
            **kwargs: Arguments additionnels pour la prédiction
            
        Returns:
            Prédictions du modèle
        """
        return self.model.predict(X, **kwargs)
        
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame], **kwargs) -> np.ndarray:
        """
        Retourne les probabilités de prédiction.
        
        Args:
            X: Données d'entrée
            **kwargs: Arguments additionnels pour la prédiction
            
        Returns:
            Probabilités de prédiction (shape: n_samples, n_classes)
            
        Raises:
            NotImplementedError: Si le modèle ne supporte pas les probabilités
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X, **kwargs)
        elif hasattr(self.model, 'predict_probs'):  # Pour certains modèles personnalisés
            return self.model.predict_probs(X, **kwargs)
        else:
            # Pour les régresseurs, on retourne une "probabilité" artificielle
            if self._model_type == ModelType.REGRESSION:
                y_pred = self.predict(X, **kwargs)
                # Normaliser les prédictions pour qu'elles ressemblent à des probabilités
                y_pred = y_pred.reshape(-1, 1)
                return np.hstack([1-y_pred, y_pred])  # Format binaire artificiel
            else:
                raise NotImplementedError("Ce modèle ne supporte pas les probabilités de prédiction")
            
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Retourne l'importance des caractéristiques si disponible.
        
        Returns:
            Dict[str, float] ou None: Dictionnaire des importances ou None si non disponible
        """
        feature_names = self.get_feature_names()
        
        # Vérifier si le modèle a des importances de caractéristiques
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            if feature_names and len(feature_names) == len(importances):
                return {name: float(imp) for name, imp in zip(feature_names, importances)}
            else:
                return {f"feature_{i}": float(imp) for i, imp in enumerate(importances)}
                
        # Pour les modèles linéaires
        elif hasattr(self.model, 'coef_'):
            coefs = self.model.coef_
            # Gérer les différentes formes possibles
            if coefs.ndim > 1 and coefs.shape[0] > 1:  # Multi-classe
                # Moyenne des coefficients absolus sur toutes les classes
                coefs = np.mean(np.abs(coefs), axis=0)
            else:
                coefs = np.abs(coefs).flatten()
                
            if feature_names and len(feature_names) == len(coefs):
                return {name: float(coef) for name, coef in zip(feature_names, coefs)}
            else:
                return {f"feature_{i}": float(coef) for i, coef in enumerate(coefs)}
        
        return None
        
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Retourne un résumé détaillé du modèle.
        
        Returns:
            dict: Dictionnaire contenant les informations du modèle
        """
        summary = super().get_model_summary()
        
        # Ajouter des informations spécifiques à scikit-learn
        summary.update({
            'is_pipeline': isinstance(self.model, Pipeline),
            'n_features': getattr(self.model, 'n_features_in_', None),
            'feature_importance': self.get_feature_importance() is not None,
            'has_coef': hasattr(self.model, 'coef_'),
            'has_intercept': hasattr(self.model, 'intercept_')
        })
        
        # Ajouter des informations sur les étapes si c'est un pipeline
        if isinstance(self.model, Pipeline):
            summary['pipeline_steps'] = [step[0] for step in self.model.steps]
            
        return summary
