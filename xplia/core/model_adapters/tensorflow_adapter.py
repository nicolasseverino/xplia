"""
Adaptateur pour les modèles TensorFlow/Keras
=========================================

Ce module fournit un adaptateur pour les modèles TensorFlow et Keras.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model as KerasModel
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    # Définitions minimales pour la compatibilité des types
    class KerasModel: pass

from ...core import ModelMetadata, ModelType, ModelAdapterBase  # Import depuis le package principal
from ...core.registry import register_model_adapter
from ...utils.tensor_utils import convert_to_numpy

@register_model_adapter(version="1.0.0", description="Adaptateur pour les modèles TensorFlow")
class TensorFlowModelAdapter(ModelAdapterBase):
    """
    Adaptateur pour les modèles TensorFlow/Keras.
    
    Cet adaptateur prend en charge:
    - Les modèles Keras séquentiels et fonctionnels
    - Les modèles TensorFlow 2.x avec API d'exportation
    - Les modèles de classification et de régression
    """
    
    def __init__(self, model: Any, **kwargs):
        """
        Initialise l'adaptateur pour un modèle TensorFlow/Keras.
        
        Args:
            model: Modèle TensorFlow/Keras à adapter
            **kwargs: Arguments additionnels
                - input_shape: Forme des données d'entrée (obligatoire si non déterminable)
                - output_shape: Forme des données de sortie (optionnel)
                - classification: Booléen indiquant si c'est un modèle de classification (optionnel)
                - classes: Liste des classes pour les modèles de classification (optionnel)
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow n'est pas installé. Installez-le avec 'pip install tensorflow'.")
        
        # Vérifier que le modèle est un modèle TensorFlow/Keras valide
        if not isinstance(model, (tf.Module, KerasModel)) and not callable(getattr(model, '__call__', None)):
            raise ValueError("Le modèle doit être une instance de tf.Module, tf.keras.Model ou un callable")
            
        self._input_shape = kwargs.get('input_shape', None)
        self._output_shape = kwargs.get('output_shape', None)
        self._is_classification = kwargs.get('classification', None)
        self._classes = kwargs.get('classes', None)
        
        super().__init__(model, **kwargs)
        self._framework = "tensorflow"
        
        # Déterminer le type de modèle si non spécifié
        if self._is_classification is None:
            self._infer_model_type()
        else:
            self._model_type = ModelType.CLASSIFICATION if self._is_classification else ModelType.REGRESSION
            
    def _infer_model_type(self):
        """Détermine le type de modèle (classification ou régression)."""
        # Pour les modèles Keras, on peut souvent déterminer à partir de la dernière couche
        if isinstance(self.model, KerasModel) and self.model.layers:
            last_layer = self.model.layers[-1]
            last_activation = getattr(last_layer, 'activation', None)
            
            # Si la dernière activation est softmax ou sigmoid, c'est probablement une classification
            if last_activation:
                if last_activation.__name__ in ['softmax', 'sigmoid']:
                    self._model_type = ModelType.CLASSIFICATION
                    return
                    
            # Vérifier le nom de la dernière couche
            if 'classifier' in last_layer.name.lower():
                self._model_type = ModelType.CLASSIFICATION
                return
                
        # Par défaut, on suppose que c'est une régression
        self._model_type = ModelType.REGRESSION
        
    def _extract_metadata(self, **kwargs) -> ModelMetadata:
        """Extrait les métadonnées du modèle TensorFlow/Keras."""
        model_class = self.model.__class__.__name__
        
        # Extraire la structure du modèle pour les modèles Keras
        model_structure = {}
        if isinstance(self.model, KerasModel):
            try:
                # Récupérer les informations sur les couches
                layers_info = []
                for layer in self.model.layers:
                    layer_info = {
                        'name': layer.name,
                        'class': layer.__class__.__name__,
                        'output_shape': str(layer.output_shape),
                        'params': layer.count_params()
                    }
                    if hasattr(layer, 'activation'):
                        layer_info['activation'] = layer.activation.__name__ if callable(layer.activation) else str(layer.activation)
                    layers_info.append(layer_info)
                
                model_structure = {
                    'layers': layers_info,
                    'total_params': self.model.count_params(),
                    'trainable_params': sum(tf.keras.backend.count_params(w) for w in self.model.trainable_weights),
                    'non_trainable_params': sum(tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights)
                }
            except:
                # En cas d'erreur, on capture silencieusement
                pass
                
        # Déterminer les formes d'entrée et de sortie
        input_shape = self._input_shape
        output_shape = self._output_shape
        
        if input_shape is None and isinstance(self.model, KerasModel) and hasattr(self.model, 'input_shape'):
            input_shape = self.model.input_shape
            
        if output_shape is None and isinstance(self.model, KerasModel) and hasattr(self.model, 'output_shape'):
            output_shape = self.model.output_shape
            
        # Extraire les classes pour les modèles de classification
        classes = self._classes
        
        # Construire les métadonnées
        return ModelMetadata(
            framework='tensorflow',
            model_type=self._model_type,
            model_class=model_class,
            classes=classes,
            feature_names=self.get_feature_names(),
            model_params={},  # Les paramètres Keras sont trop complexes pour être sérialisés simplement
            training_info={
                'input_shape': input_shape,
                'output_shape': output_shape,
                'model_structure': model_structure
            }
        )
        
    def predict(self, X: Union[np.ndarray, pd.DataFrame, tf.Tensor], **kwargs) -> np.ndarray:
        """
        Effectue une prédiction avec le modèle.
        
        Args:
            X: Données d'entrée (numpy array, pandas DataFrame ou TensorFlow tensor)
            **kwargs: Arguments additionnels pour la prédiction
            
        Returns:
            Prédictions du modèle sous forme de numpy array
        """
        # Convertir les données en format approprié
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
            
        # Effectuer la prédiction
        predictions = self.model(X_array, **kwargs)
        
        # Convertir le résultat en numpy array
        return convert_to_numpy(predictions)
        
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame, tf.Tensor], **kwargs) -> np.ndarray:
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
            
        # Pour les modèles Keras, predict retourne déjà les probabilités
        predictions = self.predict(X, **kwargs)
        
        # Si la sortie est un scalaire ou un vecteur 1D, le transformer en format binaire
        if predictions.ndim == 1 or (predictions.ndim == 2 and predictions.shape[1] == 1):
            probs = predictions.flatten()
            return np.vstack([1 - probs, probs]).T
            
        return predictions
        
    def get_gradients(self, X: Union[np.ndarray, pd.DataFrame], target_idx: Optional[int] = None) -> np.ndarray:
        """
        Calcule les gradients de la sortie par rapport à l'entrée.
        
        Args:
            X: Données d'entrée
            target_idx: Indice de la classe cible (pour les modèles de classification)
            
        Returns:
            Gradients (même forme que X)
            
        Raises:
            NotImplementedError: Si TensorFlow n'est pas disponible
        """
        if not TENSORFLOW_AVAILABLE:
            raise NotImplementedError("TensorFlow n'est pas disponible")
            
        # Convertir les données en format approprié
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
            
        # Convertir en tensor TensorFlow
        if not isinstance(X_array, tf.Tensor):
            X_tensor = tf.convert_to_tensor(X_array, dtype=tf.float32)
        else:
            X_tensor = X_array
            
        # Calculer les gradients
        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            predictions = self.model(X_tensor)
            
            # Pour les modèles de classification multi-classe, sélectionner la classe cible
            if predictions.shape[-1] > 1 and target_idx is not None:
                predictions = predictions[:, target_idx]
                
        gradients = tape.gradient(predictions, X_tensor)
        
        return convert_to_numpy(gradients)
        
    def get_layer_outputs(self, X: Union[np.ndarray, pd.DataFrame], layer_name: str) -> np.ndarray:
        """
        Retourne les sorties d'une couche spécifique pour une entrée donnée.
        
        Args:
            X: Données d'entrée
            layer_name: Nom de la couche
            
        Returns:
            Sorties de la couche
            
        Raises:
            ValueError: Si le modèle n'est pas un modèle Keras ou si la couche n'existe pas
        """
        if not isinstance(self.model, KerasModel):
            raise ValueError("Cette méthode n'est disponible que pour les modèles Keras")
            
        # Trouver la couche par son nom
        layer = None
        for l in self.model.layers:
            if l.name == layer_name:
                layer = l
                break
                
        if layer is None:
            raise ValueError(f"Couche '{layer_name}' non trouvée dans le modèle")
            
        # Créer un modèle intermédiaire qui retourne la sortie de la couche
        intermediate_model = tf.keras.Model(inputs=self.model.input, outputs=layer.output)
        
        # Convertir les données en format approprié
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
            
        # Obtenir les sorties de la couche
        layer_outputs = intermediate_model(X_array)
        
        return convert_to_numpy(layer_outputs)
