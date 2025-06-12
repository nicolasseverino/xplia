"""
Adaptateur pour les modèles PyTorch
=================================

Ce module fournit un adaptateur pour les modèles PyTorch.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    # Définitions minimales pour la compatibilité des types
    class nn:
        class Module: pass

from ...core import ModelMetadata, ModelType, ModelAdapterBase  # Import depuis le package principal
from ...core.registry import register_model_adapter
from ...utils.tensor_utils import convert_to_numpy

@register_model_adapter(version="1.0.0", description="Adaptateur pour les modèles PyTorch")
class PyTorchModelAdapter(ModelAdapterBase):
    """
    Adaptateur pour les modèles PyTorch.
    
    Cet adaptateur prend en charge:
    - Les modèles PyTorch (nn.Module)
    - Les modèles de classification et de régression
    - L'extraction automatique de gradients
    """
    
    def __init__(self, model: nn.Module, **kwargs):
        """
        Initialise l'adaptateur pour un modèle PyTorch.
        
        Args:
            model: Modèle PyTorch à adapter (doit être une instance de nn.Module)
            **kwargs: Arguments additionnels
                - input_shape: Forme des données d'entrée (obligatoire si non déterminable)
                - output_shape: Forme des données de sortie (optionnel)
                - classification: Booléen indiquant si c'est un modèle de classification (optionnel)
                - classes: Liste des classes pour les modèles de classification (optionnel)
                - device: Dispositif sur lequel exécuter le modèle ('cpu', 'cuda', etc.)
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch n'est pas installé. Installez-le avec 'pip install torch'.")
        
        # Vérifier que le modèle est un modèle PyTorch valide
        if not isinstance(model, nn.Module):
            raise ValueError("Le modèle doit être une instance de torch.nn.Module")
            
        self._input_shape = kwargs.get('input_shape', None)
        self._output_shape = kwargs.get('output_shape', None)
        self._is_classification = kwargs.get('classification', None)
        self._classes = kwargs.get('classes', None)
        self._device = kwargs.get('device', 'cpu')
        
        # Mettre le modèle en mode évaluation et sur le dispositif spécifié
        model.eval()
        self._original_device = next(model.parameters()).device
        if self._device != str(self._original_device):
            model = model.to(self._device)
            
        super().__init__(model, **kwargs)
        self._framework = "pytorch"
        
        # Déterminer le type de modèle si non spécifié
        if self._is_classification is None:
            self._infer_model_type()
        else:
            self._model_type = ModelType.CLASSIFICATION if self._is_classification else ModelType.REGRESSION
            
    def _infer_model_type(self):
        """Détermine le type de modèle (classification ou régression)."""
        # Vérifier si le modèle a une couche de sortie typique de classification
        has_softmax = False
        has_sigmoid = False
        
        # Parcourir les modules pour détecter des indices
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Softmax):
                has_softmax = True
                break
            elif isinstance(module, nn.Sigmoid):
                has_sigmoid = True
                break
                
            # Vérifier également les noms des modules
            if 'classifier' in name.lower():
                self._model_type = ModelType.CLASSIFICATION
                return
                
        # Si on a trouvé un softmax ou sigmoid, c'est probablement une classification
        if has_softmax or has_sigmoid:
            self._model_type = ModelType.CLASSIFICATION
            return
            
        # Par défaut, on suppose que c'est une régression
        self._model_type = ModelType.REGRESSION
        
    def _extract_metadata(self, **kwargs) -> ModelMetadata:
        """Extrait les métadonnées du modèle PyTorch."""
        model_class = self.model.__class__.__name__
        
        # Extraire la structure du modèle
        model_structure = {}
        try:
            # Récupérer les informations sur les modules
            modules_info = []
            for name, module in self.model.named_modules():
                if name == '':  # Ignorer le module racine
                    continue
                    
                module_info = {
                    'name': name,
                    'class': module.__class__.__name__
                }
                
                # Essayer de compter les paramètres
                try:
                    params = sum(p.numel() for p in module.parameters())
                    module_info['params'] = params
                except:
                    pass
                    
                modules_info.append(module_info)
                
            # Compter les paramètres totaux
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            model_structure = {
                'modules': modules_info,
                'total_params': total_params,
                'trainable_params': trainable_params,
                'non_trainable_params': total_params - trainable_params
            }
        except:
            # En cas d'erreur, on capture silencieusement
            pass
            
        # Construire les métadonnées
        return ModelMetadata(
            framework='pytorch',
            model_type=self._model_type,
            model_class=model_class,
            classes=self._classes,
            feature_names=self.get_feature_names(),
            model_params={},  # Les paramètres PyTorch sont trop complexes pour être sérialisés simplement
            training_info={
                'input_shape': self._input_shape,
                'output_shape': self._output_shape,
                'device': self._device,
                'model_structure': model_structure
            }
        )
        
    def _prepare_input(self, X: Union[np.ndarray, pd.DataFrame]) -> torch.Tensor:
        """
        Prépare les données d'entrée pour le modèle PyTorch.
        
        Args:
            X: Données d'entrée (numpy array ou pandas DataFrame)
            
        Returns:
            torch.Tensor: Données converties en tensor PyTorch
        """
        # Convertir en numpy array si nécessaire
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
            
        # Convertir en tensor PyTorch
        if not isinstance(X_array, torch.Tensor):
            X_tensor = torch.tensor(X_array, dtype=torch.float32, device=self._device)
        else:
            # S'assurer que le tensor est sur le bon dispositif
            X_tensor = X_array.to(self._device)
            
        return X_tensor
        
    def predict(self, X: Union[np.ndarray, pd.DataFrame, torch.Tensor], **kwargs) -> np.ndarray:
        """
        Effectue une prédiction avec le modèle.
        
        Args:
            X: Données d'entrée (numpy array, pandas DataFrame ou PyTorch tensor)
            **kwargs: Arguments additionnels pour la prédiction
            
        Returns:
            Prédictions du modèle sous forme de numpy array
        """
        # Préparer les données d'entrée
        X_tensor = self._prepare_input(X)
        
        # Désactiver le calcul de gradients pour l'inférence
        with torch.no_grad():
            # Effectuer la prédiction
            predictions = self.model(X_tensor, **kwargs)
            
            # Pour les modèles de classification, appliquer softmax si nécessaire
            if self._model_type == ModelType.CLASSIFICATION and predictions.dim() > 1 and predictions.shape[1] > 1:
                # Vérifier si le modèle n'a pas déjà appliqué softmax
                if torch.allclose(torch.sum(predictions, dim=1), torch.tensor(1.0, device=predictions.device), atol=1e-3):
                    # Softmax déjà appliqué
                    pass
                else:
                    # Appliquer softmax
                    predictions = torch.softmax(predictions, dim=1)
        
        # Convertir le résultat en numpy array
        return convert_to_numpy(predictions)
        
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame, torch.Tensor], **kwargs) -> np.ndarray:
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
            
        # Préparer les données d'entrée
        X_tensor = self._prepare_input(X)
        
        # Désactiver le calcul de gradients pour l'inférence
        with torch.no_grad():
            # Effectuer la prédiction
            predictions = self.model(X_tensor, **kwargs)
            
            # Appliquer softmax/sigmoid si nécessaire
            if predictions.dim() == 1 or (predictions.dim() == 2 and predictions.shape[1] == 1):
                # Cas binaire, appliquer sigmoid
                predictions = torch.sigmoid(predictions)
                # Convertir en format binaire [1-p, p]
                if predictions.dim() == 1:
                    probs = predictions.unsqueeze(1)
                else:
                    probs = predictions
                probs = torch.cat([1 - probs, probs], dim=1)
            else:
                # Cas multi-classe, appliquer softmax si pas déjà normalisé
                if not torch.allclose(torch.sum(predictions, dim=1), torch.tensor(1.0, device=predictions.device), atol=1e-3):
                    probs = torch.softmax(predictions, dim=1)
                else:
                    probs = predictions
        
        # Convertir le résultat en numpy array
        return convert_to_numpy(probs)
        
    def get_gradients(self, X: Union[np.ndarray, pd.DataFrame], target_idx: Optional[int] = None) -> np.ndarray:
        """
        Calcule les gradients de la sortie par rapport à l'entrée.
        
        Args:
            X: Données d'entrée
            target_idx: Indice de la classe cible (pour les modèles de classification)
            
        Returns:
            Gradients (même forme que X)
        """
        # Préparer les données d'entrée
        X_tensor = self._prepare_input(X)
        X_tensor.requires_grad_(True)
        
        # Forward pass
        predictions = self.model(X_tensor)
        
        # Pour les modèles de classification multi-classe, sélectionner la classe cible
        if predictions.dim() > 1 and predictions.shape[1] > 1 and target_idx is not None:
            predictions = predictions[:, target_idx]
            
        # Calculer les gradients
        gradients = torch.autograd.grad(
            outputs=predictions.sum(),
            inputs=X_tensor,
            create_graph=False,
            retain_graph=False
        )[0]
        
        # Détacher et convertir en numpy
        return convert_to_numpy(gradients.detach())
        
    def get_intermediate_outputs(self, X: Union[np.ndarray, pd.DataFrame], layer_names: List[str]) -> Dict[str, np.ndarray]:
        """
        Retourne les sorties de couches intermédiaires pour une entrée donnée.
        
        Args:
            X: Données d'entrée
            layer_names: Liste des noms des couches dont on veut les sorties
            
        Returns:
            Dictionnaire {nom_couche: sortie} des sorties des couches spécifiées
            
        Note:
            Cette méthode nécessite de modifier temporairement le modèle pour capturer les sorties intermédiaires.
        """
        # Préparer les données d'entrée
        X_tensor = self._prepare_input(X)
        
        # Dictionnaire pour stocker les activations
        activations = {}
        hooks = []
        
        # Fonction pour capturer les sorties
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output
            return hook
            
        # Enregistrer les hooks pour les couches spécifiées
        for name, module in self.model.named_modules():
            if name in layer_names:
                hooks.append(module.register_forward_hook(hook_fn(name)))
                
        # Forward pass
        with torch.no_grad():
            self.model(X_tensor)
            
        # Supprimer les hooks
        for hook in hooks:
            hook.remove()
            
        # Convertir les résultats en numpy
        return {name: convert_to_numpy(output) for name, output in activations.items()}
