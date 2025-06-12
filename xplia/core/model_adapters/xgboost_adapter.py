"""
Adaptateur pour les modèles XGBoost
=================================

Ce module fournit un adaptateur pour les modèles XGBoost.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    # Définitions minimales pour la compatibilité des types
    class xgb:
        class Booster: pass
        class DMatrix: pass

from ...core import ModelMetadata, ModelType, ModelAdapterBase  # Import depuis le package principal
from ...core.registry import register_model_adapter

@register_model_adapter(version="1.0.0", description="Adaptateur pour les modèles XGBoost")
class XGBoostModelAdapter(ModelAdapterBase):
    """
    Adaptateur pour les modèles XGBoost.
    
    Cet adaptateur prend en charge:
    - Les modèles XGBoost (Booster)
    - Les modèles de classification et de régression
    - L'extraction des importances de caractéristiques
    """
    
    def __init__(self, model: Any, **kwargs):
        """
        Initialise l'adaptateur pour un modèle XGBoost.
        
        Args:
            model: Modèle XGBoost à adapter (peut être un Booster ou un wrapper scikit-learn)
            **kwargs: Arguments additionnels
                - objective: Objectif du modèle ('binary:logistic', 'multi:softprob', 'reg:squarederror', etc.)
                - num_class: Nombre de classes pour les modèles multi-classes
                - feature_names: Noms des caractéristiques
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost n'est pas installé. Installez-le avec 'pip install xgboost'.")
            
        # Extraire le booster si c'est un wrapper scikit-learn
        if hasattr(model, 'get_booster'):
            self._sklearn_wrapper = model
            model = model.get_booster()
        else:
            self._sklearn_wrapper = None
            
        # Vérifier que le modèle est un modèle XGBoost valide
        if not isinstance(model, xgb.Booster) and not hasattr(model, 'predict'):
            raise ValueError("Le modèle doit être une instance de xgboost.Booster ou avoir une méthode predict")
            
        self._objective = kwargs.get('objective', getattr(model, 'objective', None))
        self._num_class = kwargs.get('num_class', None)
        
        super().__init__(model, **kwargs)
        self._framework = "xgboost"
        
        # Extraire les noms de caractéristiques si disponibles
        if kwargs.get('feature_names'):
            self.set_feature_names(kwargs.get('feature_names'))
        elif hasattr(model, 'feature_names'):
            self.set_feature_names(model.feature_names)
        elif self._sklearn_wrapper and hasattr(self._sklearn_wrapper, 'feature_names_in_'):
            self.set_feature_names(self._sklearn_wrapper.feature_names_in_.tolist())
            
        # Déterminer le type de modèle
        self._infer_model_type()
        
    def _infer_model_type(self):
        """Détermine le type de modèle (classification ou régression)."""
        # Si l'objectif est spécifié, l'utiliser pour déterminer le type
        if self._objective:
            if any(obj in self._objective for obj in ['binary', 'multi', 'softmax', 'softprob']):
                self._model_type = ModelType.CLASSIFICATION
            else:
                self._model_type = ModelType.REGRESSION
        # Sinon, essayer de déterminer à partir du wrapper scikit-learn
        elif self._sklearn_wrapper:
            if hasattr(self._sklearn_wrapper, 'classes_'):
                self._model_type = ModelType.CLASSIFICATION
                # Extraire le nombre de classes
                self._num_class = len(self._sklearn_wrapper.classes_)
            else:
                self._model_type = ModelType.REGRESSION
        # Par défaut, supposer que c'est une régression
        else:
            self._model_type = ModelType.REGRESSION
            
    def _extract_metadata(self, **kwargs) -> ModelMetadata:
        """Extrait les métadonnées du modèle XGBoost."""
        # Extraire les paramètres du modèle
        try:
            if hasattr(self.model, 'get_params'):
                model_params = self.model.get_params()
            elif hasattr(self.model, 'save_config'):
                import json
                model_params = json.loads(self.model.save_config())
            else:
                model_params = {}
        except:
            model_params = {}
            
        # Extraire les classes pour les modèles de classification
        classes = None
        if self._model_type == ModelType.CLASSIFICATION:
            if self._sklearn_wrapper and hasattr(self._sklearn_wrapper, 'classes_'):
                classes = self._sklearn_wrapper.classes_.tolist()
                
        # Extraire les informations sur l'entraînement
        training_info = {}
        if hasattr(self.model, 'attributes'):
            attrs = self.model.attributes()
            if 'best_iteration' in attrs:
                training_info['best_iteration'] = int(attrs['best_iteration'])
            if 'best_score' in attrs:
                training_info['best_score'] = float(attrs['best_score'])
                
        # Ajouter l'objectif et le nombre de classes
        if self._objective:
            training_info['objective'] = self._objective
        if self._num_class:
            training_info['num_class'] = self._num_class
            
        return ModelMetadata(
            framework='xgboost',
            model_type=self._model_type,
            model_class='XGBoostModel',
            classes=classes,
            feature_names=self.get_feature_names(),
            model_params=model_params,
            training_info=training_info
        )
        
    def _create_dmatrix(self, X: Union[np.ndarray, pd.DataFrame]) -> xgb.DMatrix:
        """
        Crée une DMatrix XGBoost à partir des données d'entrée.
        
        Args:
            X: Données d'entrée (numpy array ou pandas DataFrame)
            
        Returns:
            xgb.DMatrix: Données converties en DMatrix XGBoost
        """
        feature_names = self.get_feature_names()
        
        # Si c'est déjà une DMatrix, la retourner telle quelle
        if isinstance(X, xgb.DMatrix):
            return X
            
        # Créer une DMatrix avec les noms de caractéristiques si disponibles
        if feature_names:
            return xgb.DMatrix(data=X, feature_names=feature_names)
        else:
            return xgb.DMatrix(data=X)
            
    def predict(self, X: Union[np.ndarray, pd.DataFrame, xgb.DMatrix], **kwargs) -> np.ndarray:
        """
        Effectue une prédiction avec le modèle.
        
        Args:
            X: Données d'entrée (numpy array, pandas DataFrame ou XGBoost DMatrix)
            **kwargs: Arguments additionnels pour la prédiction
                - output_margin: Si True, retourne les scores bruts avant transformation
                
        Returns:
            Prédictions du modèle sous forme de numpy array
        """
        # Si c'est un wrapper scikit-learn, utiliser sa méthode predict
        if self._sklearn_wrapper:
            return self._sklearn_wrapper.predict(X)
            
        # Sinon, utiliser directement le booster
        output_margin = kwargs.get('output_margin', False)
        
        # Convertir en DMatrix si nécessaire
        if not isinstance(X, xgb.DMatrix):
            X_dmatrix = self._create_dmatrix(X)
        else:
            X_dmatrix = X
            
        # Pour les modèles de classification multi-classe
        if self._model_type == ModelType.CLASSIFICATION and self._num_class and self._num_class > 2:
            # Prédire les probabilités et prendre la classe avec la probabilité maximale
            probs = self.model.predict(X_dmatrix, output_margin=output_margin)
            if probs.ndim > 1:
                return np.argmax(probs, axis=1)
            else:
                return (probs > 0.5).astype(int)  # Cas binaire
        else:
            # Prédiction standard
            preds = self.model.predict(X_dmatrix, output_margin=output_margin)
            
            # Pour la classification binaire, convertir les probabilités en classes
            if self._model_type == ModelType.CLASSIFICATION and not output_margin:
                return (preds > 0.5).astype(int)
            else:
                return preds
                
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame, xgb.DMatrix], **kwargs) -> np.ndarray:
        """
        Retourne les probabilités de prédiction.
        
        Args:
            X: Données d'entrée
            **kwargs: Arguments additionnels pour la prédiction
            
        Returns:
            Probabilités de prédiction (shape: n_samples, n_classes)
            
        Raises:
            NotImplementedError: Si le modèle n'est pas un modèle de classification
        """
        if self._model_type != ModelType.CLASSIFICATION:
            raise NotImplementedError("Ce modèle n'est pas un modèle de classification")
            
        # Si c'est un wrapper scikit-learn, utiliser sa méthode predict_proba
        if self._sklearn_wrapper and hasattr(self._sklearn_wrapper, 'predict_proba'):
            return self._sklearn_wrapper.predict_proba(X)
            
        # Convertir en DMatrix si nécessaire
        if not isinstance(X, xgb.DMatrix):
            X_dmatrix = self._create_dmatrix(X)
        else:
            X_dmatrix = X
            
        # Prédire les probabilités
        probs = self.model.predict(X_dmatrix, output_margin=False)
        
        # Pour les modèles binaires, convertir en format [1-p, p]
        if probs.ndim == 1 or (probs.ndim == 2 and probs.shape[1] == 1):
            probs = probs.flatten()
            return np.vstack([1 - probs, probs]).T
            
        return probs
        
    def get_feature_importance(self, importance_type: str = 'weight') -> Dict[str, float]:
        """
        Retourne l'importance des caractéristiques.
        
        Args:
            importance_type: Type d'importance ('weight', 'gain', 'cover', 'total_gain', 'total_cover')
            
        Returns:
            Dict[str, float]: Dictionnaire des importances
        """
        # Obtenir les scores d'importance
        scores = self.model.get_score(importance_type=importance_type)
        
        # Si les noms de caractéristiques sont disponibles, les utiliser
        feature_names = self.get_feature_names()
        if feature_names:
            # Créer un dictionnaire avec toutes les caractéristiques (même celles avec score 0)
            result = {name: 0.0 for name in feature_names}
            # Mettre à jour avec les scores disponibles
            result.update(scores)
            return result
        else:
            return scores
            
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Retourne un résumé détaillé du modèle.
        
        Returns:
            dict: Dictionnaire contenant les informations du modèle
        """
        summary = super().get_model_summary()
        
        # Ajouter des informations spécifiques à XGBoost
        feature_importance = self.get_feature_importance()
        
        summary.update({
            'objective': self._objective,
            'num_class': self._num_class,
            'feature_importance': feature_importance,
            'is_sklearn_wrapper': self._sklearn_wrapper is not None
        })
        
        # Ajouter des informations sur l'arbre si disponibles
        try:
            num_trees = len(self.model.get_dump())
            summary['num_trees'] = num_trees
        except:
            pass
            
        return summary
