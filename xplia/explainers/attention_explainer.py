"""
AttentionExplainer pour XPLIA
=============================

Ce module implémente l'AttentionExplainer qui permet de visualiser et d'interpréter
les mécanismes d'attention dans les modèles de deep learning comme les Transformers,
les LSTM et autres architectures avec des mécanismes d'attention.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Union, Optional, Tuple, Any, Callable

from ..core.base import ExplainerBase
from ..core.registry import register_explainer
from ..core.models import ExplanationResult, FeatureImportance
from ..core.enums import ExplainabilityMethod, AudienceLevel
from ..core.metadata import ModelMetadata

@register_explainer
class AttentionExplainer(ExplainerBase):
    """
    Explainer qui extrait et interprète les mécanismes d'attention dans les modèles de deep learning.
    
    Cet explainer est conçu pour fonctionner avec des modèles qui utilisent des mécanismes d'attention,
    comme les Transformers (BERT, GPT, etc.), les modèles LSTM avec attention, et d'autres architectures
    similaires. Il peut extraire les poids d'attention et les convertir en scores d'importance pour
    les tokens ou les caractéristiques d'entrée.
    
    Attributs:
        _model: Modèle à expliquer
        _framework: Framework du modèle ('tensorflow', 'pytorch', 'huggingface')
        _attention_layer_names: Noms des couches d'attention à extraire
        _tokenizer: Tokenizer pour les modèles de NLP (optionnel)
        _aggregation_method: Méthode d'agrégation des poids d'attention ('mean', 'max', 'sum', 'last')
        _head_importance: Importance relative des différentes têtes d'attention (optionnel)
        _layer_importance: Importance relative des différentes couches (optionnel)
        _feature_names: Noms des caractéristiques ou tokens
        _metadata: Métadonnées du modèle
    """
    
    def __init__(self, model, tokenizer=None, framework=None, attention_layer_names=None, 
                 aggregation_method='mean', head_importance=None, layer_importance=None,
                 feature_names=None):
        """
        Initialise l'AttentionExplainer.
        
        Args:
            model: Modèle à expliquer avec mécanismes d'attention
            tokenizer: Tokenizer pour les modèles de NLP (optionnel)
            framework: Framework du modèle ('tensorflow', 'pytorch', 'huggingface')
            attention_layer_names: Noms des couches d'attention à extraire (si None, tente de les détecter)
            aggregation_method: Méthode d'agrégation des poids d'attention ('mean', 'max', 'sum', 'last')
            head_importance: Importance relative des différentes têtes d'attention (optionnel)
            layer_importance: Importance relative des différentes couches (optionnel)
            feature_names: Noms des caractéristiques ou tokens (optionnel)
        """
        super().__init__()
        
        # Modèle et framework
        self._model = model
        self._framework = framework or self._detect_framework()
        
        # Tokenizer pour les modèles NLP
        self._tokenizer = tokenizer
        
        # Couches d'attention
        self._attention_layer_names = attention_layer_names
        
        # Méthode d'agrégation
        self._aggregation_method = aggregation_method
        
        # Importance des têtes et des couches
        self._head_importance = head_importance
        self._layer_importance = layer_importance
        
        # Noms des caractéristiques
        self._feature_names = feature_names
        
        # Métadonnées du modèle
        self._metadata = None
        
        # Logger
        self._logger = logging.getLogger(__name__)
    
    def explain_instance(self, instance, **kwargs) -> ExplanationResult:
        """
        Génère une explication basée sur les mécanismes d'attention pour une instance spécifique.
        
        Args:
            instance: Instance à expliquer (texte, séquence de tokens, ou tenseur d'entrée)
            **kwargs: Paramètres additionnels
                input_type: Type d'entrée ('text', 'tokens', 'tensor')
                attention_threshold: Seuil pour filtrer les poids d'attention faibles (défaut: 0.05)
                max_length: Longueur maximale pour les séquences (défaut: 512)
                batch_size: Taille du batch pour l'inférence (défaut: 1)
                include_special_tokens: Inclure les tokens spéciaux dans l'explication (défaut: False)
                audience_level: Niveau d'audience (TECHNICAL, BUSINESS, PUBLIC)
                
        Returns:
            ExplanationResult: Résultat standardisé de l'explication
        """
        # Paramètres
        audience_level = kwargs.get('audience_level', AudienceLevel.TECHNICAL)
        input_type = kwargs.get('input_type', 'text' if isinstance(instance, str) else 'tensor')
        attention_threshold = kwargs.get('attention_threshold', 0.05)
        max_length = kwargs.get('max_length', 512)
        batch_size = kwargs.get('batch_size', 1)
        include_special_tokens = kwargs.get('include_special_tokens', False)
        
        # Tracer l'action
        self.add_audit_record("explain_instance", {
            "input_type": input_type,
            "audience_level": audience_level.value if isinstance(audience_level, AudienceLevel) else audience_level,
            "attention_threshold": attention_threshold,
            "max_length": max_length
        })
        
        try:
            # Préparer l'entrée selon le type
            prepared_input, tokens = self._prepare_input(instance, input_type, max_length)
            
            # Extraire les poids d'attention
            attention_weights = self._extract_attention_weights(prepared_input, batch_size)
            
            # Agréger les poids d'attention
            aggregated_attention = self._aggregate_attention_weights(
                attention_weights, 
                method=self._aggregation_method,
                head_importance=self._head_importance,
                layer_importance=self._layer_importance
            )
            
            # Convertir les poids d'attention en importances de caractéristiques
            feature_importances = self._convert_attention_to_importances(
                aggregated_attention, 
                tokens, 
                threshold=attention_threshold,
                include_special_tokens=include_special_tokens
            )
            
            # Extraire les métadonnées du modèle
            if not self._metadata:
                self._extract_metadata()
            
            # Créer le résultat d'explication
            result = ExplanationResult(
                method=ExplainabilityMethod.ATTENTION,
                model_metadata=self._metadata,
                feature_importances=feature_importances,
                raw_explanation={
                    "attention_weights": self._convert_attention_to_serializable(attention_weights),
                    "aggregated_attention": aggregated_attention.tolist() if isinstance(aggregated_attention, np.ndarray) else aggregated_attention,
                    "tokens": tokens,
                    "input_type": input_type
                },
                audience_level=audience_level
            )
            
            return result
            
        except Exception as e:
            self._logger.error(f"Erreur lors de l'extraction des mécanismes d'attention: {str(e)}")
            raise RuntimeError(f"Échec de l'explication par attention: {str(e)}")
    
    def explain(self, X, y=None, **kwargs) -> ExplanationResult:
        """
        Génère des explications basées sur les mécanismes d'attention pour un ensemble de données.
        Pour les explications par attention, cette méthode sélectionne un échantillon
        représentatif et génère des explications pour chaque instance.
        
        Args:
            X: Données d'entrée à expliquer (textes, séquences de tokens, ou tenseurs)
            y: Valeurs cibles réelles (optionnel)
            **kwargs: Paramètres additionnels
                max_instances: Nombre maximum d'instances à expliquer
                sampling_strategy: Stratégie d'échantillonnage ('random', 'stratified', 'diverse')
                input_type: Type d'entrée ('text', 'tokens', 'tensor')
                
        Returns:
            ExplanationResult: Résultat standardisé de l'explication
        """
        # Paramètres
        audience_level = kwargs.get('audience_level', AudienceLevel.TECHNICAL)
        max_instances = kwargs.get('max_instances', 5)
        sampling_strategy = kwargs.get('sampling_strategy', 'random')
        input_type = kwargs.get('input_type', 'text' if isinstance(X[0], str) else 'tensor')
        
        # Tracer l'action
        self.add_audit_record("explain", {
            "n_samples": len(X),
            "audience_level": audience_level.value if isinstance(audience_level, AudienceLevel) else audience_level,
            "max_instances": max_instances,
            "sampling_strategy": sampling_strategy,
            "input_type": input_type
        })
        
        try:
            # Échantillonner des instances représentatives
            sampled_indices = self._sample_instances(X, y, max_instances, sampling_strategy)
            sampled_instances = [X[i] for i in sampled_indices]
            
            # Générer des explications pour chaque instance échantillonnée
            all_feature_importances = []
            all_attention_weights = []
            all_tokens = []
            
            for instance in sampled_instances:
                # Utiliser explain_instance pour chaque instance
                instance_result = self.explain_instance(
                    instance, 
                    input_type=input_type,
                    audience_level=audience_level,
                    **kwargs
                )
                
                # Collecter les résultats
                all_feature_importances.append(instance_result.feature_importances)
                all_attention_weights.append(instance_result.raw_explanation["attention_weights"])
                all_tokens.append(instance_result.raw_explanation["tokens"])
            
            # Agréger les importances de caractéristiques
            if input_type == 'text':
                # Pour le texte, on ne peut pas agréger directement car les tokens sont différents
                aggregated_importances = self._aggregate_text_importances(all_feature_importances)
            else:
                # Pour les tenseurs ou tokens pré-définis, on peut agréger directement
                feature_names = self._feature_names or [f"feature_{i}" for i in range(len(all_feature_importances[0]))]
                aggregated_importances = self._aggregate_feature_importances(all_feature_importances, feature_names)
            
            # Extraire les métadonnées du modèle
            if not self._metadata:
                self._extract_metadata()
            
            # Créer le résultat d'explication
            result = ExplanationResult(
                method=ExplainabilityMethod.ATTENTION,
                model_metadata=self._metadata,
                feature_importances=aggregated_importances,
                raw_explanation={
                    "sampled_instances": sampled_indices,
                    "all_attention_weights": all_attention_weights,
                    "all_tokens": all_tokens,
                    "input_type": input_type
                },
                audience_level=audience_level
            )
            
            return result
            
        except Exception as e:
            self._logger.error(f"Erreur lors de l'extraction des mécanismes d'attention: {str(e)}")
            raise RuntimeError(f"Échec de l'explication par attention: {str(e)}")
    
    def _detect_framework(self):
        """
        Détecte automatiquement le framework du modèle.
        
        Returns:
            str: Framework détecté ('tensorflow', 'pytorch', 'huggingface')
        """
        model_module = self._model.__module__.split('.')[0].lower()
        
        if model_module in ['tensorflow', 'tf', 'keras']:
            return 'tensorflow'
        elif model_module in ['torch', 'pytorch']:
            return 'pytorch'
        elif model_module in ['transformers']:
            return 'huggingface'
        else:
            # Essayer de détecter par les attributs
            if hasattr(self._model, 'layers'):
                return 'tensorflow'
            elif hasattr(self._model, 'modules'):
                return 'pytorch'
            else:
                self._logger.warning("Impossible de détecter automatiquement le framework. "
                                  "Utilisation du framework par défaut: 'tensorflow'.")
                return 'tensorflow'
    
    def _prepare_input(self, instance, input_type, max_length):
        """
        Prépare l'entrée pour l'extraction des poids d'attention.
        
        Args:
            instance: Instance à expliquer
            input_type: Type d'entrée ('text', 'tokens', 'tensor')
            max_length: Longueur maximale pour les séquences
            
        Returns:
            tuple: (entrée préparée, liste de tokens)
        """
        if input_type == 'text':
            # Vérifier si un tokenizer est disponible
            if self._tokenizer is None:
                raise ValueError("Un tokenizer est requis pour traiter les entrées textuelles.")
            
            # Tokenizer le texte selon le framework
            if self._framework == 'huggingface':
                # Pour les modèles Hugging Face
                encoding = self._tokenizer(instance, return_tensors="pt" if self._framework == 'pytorch' else "tf",
                                         max_length=max_length, truncation=True, padding='max_length')
                tokens = self._tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
                return encoding, tokens
            
            elif self._framework == 'tensorflow':
                # Pour les modèles TensorFlow/Keras
                import tensorflow as tf
                tokens = self._tokenizer.texts_to_sequences([instance])[0]
                if max_length:
                    from tensorflow.keras.preprocessing.sequence import pad_sequences
                    tokens = pad_sequences([tokens], maxlen=max_length, padding='post')[0]
                token_strings = [self._tokenizer.index_word.get(t, '[UNK]') for t in tokens]
                return tf.constant([tokens]), token_strings
            
            elif self._framework == 'pytorch':
                # Pour les modèles PyTorch
                import torch
                if hasattr(self._tokenizer, 'encode'):
                    tokens = self._tokenizer.encode(instance, max_length=max_length, truncation=True)
                    token_strings = [self._tokenizer.decode([t]) for t in tokens]
                    return torch.tensor([tokens]), token_strings
                else:
                    # Tokenizer personnalisé
                    tokens = self._tokenizer(instance)
                    return torch.tensor([tokens]), tokens
        
        elif input_type == 'tokens':
            # Déjà tokenisé
            if self._framework == 'tensorflow':
                import tensorflow as tf
                return tf.constant([instance]), instance
            elif self._framework == 'pytorch':
                import torch
                return torch.tensor([instance]), instance
            else:
                return instance, instance
        
        elif input_type == 'tensor':
            # Déjà sous forme de tenseur
            # Générer des noms de tokens génériques
            if hasattr(instance, 'shape'):
                seq_length = instance.shape[-1]
            else:
                seq_length = len(instance)
            tokens = [f"token_{i}" for i in range(seq_length)]
            return instance, tokens
        
        else:
            raise ValueError(f"Type d'entrée non supporté: {input_type}")
    
    def _extract_attention_weights(self, prepared_input, batch_size=1):
        """
        Extrait les poids d'attention du modèle.
        
        Args:
            prepared_input: Entrée préparée pour le modèle
            batch_size: Taille du batch pour l'inférence
            
        Returns:
            dict: Poids d'attention extraits
        """
        # Selon le framework, extraire les poids d'attention de manière différente
        if self._framework == 'huggingface':
            return self._extract_huggingface_attention(prepared_input)
        elif self._framework == 'tensorflow':
            return self._extract_tensorflow_attention(prepared_input, batch_size)
        elif self._framework == 'pytorch':
            return self._extract_pytorch_attention(prepared_input, batch_size)
        else:
            raise ValueError(f"Framework non supporté pour l'extraction d'attention: {self._framework}")
    
    def _extract_huggingface_attention(self, prepared_input):
        """
        Extrait les poids d'attention d'un modèle Hugging Face.
        
        Args:
            prepared_input: Entrée préparée pour le modèle
            
        Returns:
            dict: Poids d'attention extraits
        """
        # Utiliser l'option output_attentions=True pour les modèles Hugging Face
        outputs = self._model(**prepared_input, output_attentions=True)
        
        # Les poids d'attention sont généralement dans le champ 'attentions'
        attention_weights = outputs.attentions
        
        # Convertir en format standard
        result = {
            'num_layers': len(attention_weights),
            'num_heads': attention_weights[0].shape[1],
            'attention_weights': []
        }
        
        # Traiter chaque couche
        for layer_idx, layer_attention in enumerate(attention_weights):
            # Convertir en numpy pour la sérialisation
            if hasattr(layer_attention, 'detach'):
                layer_attention = layer_attention.detach().cpu().numpy()
            elif hasattr(layer_attention, 'numpy'):
                layer_attention = layer_attention.numpy()
                
            # Ajouter à la liste des poids
            result['attention_weights'].append({
                'layer': layer_idx,
                'weights': layer_attention[0]  # Premier élément du batch
            })
        
        return result
    
    def _extract_tensorflow_attention(self, prepared_input, batch_size):
        """
        Extrait les poids d'attention d'un modèle TensorFlow/Keras.
        
        Args:
            prepared_input: Entrée préparée pour le modèle
            batch_size: Taille du batch pour l'inférence
            
        Returns:
            dict: Poids d'attention extraits
        """
        import tensorflow as tf
        
        # Créer un modèle pour extraire les sorties des couches d'attention
        if self._attention_layer_names:
            # Utiliser les noms de couches spécifiés
            attention_layers = [layer for layer in self._model.layers 
                               if any(name in layer.name for name in self._attention_layer_names)]
        else:
            # Essayer de détecter automatiquement les couches d'attention
            attention_layers = [layer for layer in self._model.layers 
                               if 'attention' in layer.name.lower()]
            
            # Si aucune couche d'attention n'est trouvée, essayer de trouver des couches avec des têtes d'attention
            if not attention_layers:
                for layer in self._model.layers:
                    if hasattr(layer, 'attention_heads') or hasattr(layer, 'num_heads'):
                        attention_layers.append(layer)
        
        if not attention_layers:
            raise ValueError("Aucune couche d'attention détectée dans le modèle.")
        
        # Créer un modèle pour extraire les sorties des couches d'attention
        attention_outputs = [layer.output for layer in attention_layers]
        attention_model = tf.keras.Model(inputs=self._model.input, outputs=attention_outputs)
        
        # Faire une prédiction pour obtenir les poids d'attention
        attention_values = attention_model.predict(prepared_input, batch_size=batch_size)
        
        # Convertir en liste si un seul élément
        if not isinstance(attention_values, list):
            attention_values = [attention_values]
        
        # Créer le résultat
        result = {
            'num_layers': len(attention_values),
            'num_heads': self._get_num_heads(attention_layers[0]) if attention_layers else 1,
            'attention_weights': []
        }
        
        # Traiter chaque couche
        for layer_idx, layer_attention in enumerate(attention_values):
            # Pour les modèles avec plusieurs têtes d'attention
            if len(layer_attention.shape) >= 4:  # [batch, seq_len, seq_len, num_heads]
                weights = layer_attention[0]  # Premier élément du batch
            else:  # [batch, seq_len, seq_len]
                weights = layer_attention[0, :, :, np.newaxis]  # Ajouter une dimension pour les têtes
                
            result['attention_weights'].append({
                'layer': layer_idx,
                'weights': weights
            })
        
        return result
    
    def _extract_pytorch_attention(self, prepared_input, batch_size):
        """
        Extrait les poids d'attention d'un modèle PyTorch.
        
        Args:
            prepared_input: Entrée préparée pour le modèle
            batch_size: Taille du batch pour l'inférence
            
        Returns:
            dict: Poids d'attention extraits
        """
        import torch
        
        # Mettre le modèle en mode évaluation
        self._model.eval()
        
        # Enregistrer les hooks pour capturer les poids d'attention
        attention_maps = []
        hooks = []
        
        def get_attention_hook(name):
            def hook(module, input, output):
                # Capturer les poids d'attention
                if isinstance(output, tuple):
                    # Certains modules retournent (output, attention_weights)
                    for item in output:
                        if isinstance(item, torch.Tensor) and 'attention' in name.lower():
                            attention_maps.append(item.detach())
                else:
                    attention_maps.append(output.detach())
            return hook
        
        # Attacher les hooks aux modules d'attention
        for name, module in self._model.named_modules():
            if self._attention_layer_names:
                # Utiliser les noms de modules spécifiés
                if any(pattern in name for pattern in self._attention_layer_names):
                    hooks.append(module.register_forward_hook(get_attention_hook(name)))
            else:
                # Détection automatique des modules d'attention
                if 'attention' in name.lower() or hasattr(module, 'num_attention_heads'):
                    hooks.append(module.register_forward_hook(get_attention_hook(name)))
        
        if not attention_maps:
            raise ValueError("Aucun poids d'attention capturé. Vérifiez les noms des couches d'attention.")
        
        # Faire une prédiction pour déclencher les hooks
        with torch.no_grad():
            _ = self._model(prepared_input)
        
        # Supprimer les hooks
        for hook in hooks:
            hook.remove()
        
        # Vérifier si des poids d'attention ont été capturés
        if not attention_maps:
            raise ValueError("Aucun poids d'attention capturé. Vérifiez les noms des couches d'attention.")
        
        # Créer le résultat
        result = {
            'num_layers': len(attention_maps),
            'num_heads': attention_maps[0].shape[1] if len(attention_maps[0].shape) > 3 else 1,
            'attention_weights': []
        }
        
        # Traiter chaque couche
        for layer_idx, layer_attention in enumerate(attention_maps):
            # Convertir en numpy
            layer_attention = layer_attention.cpu().numpy()
            
            # Pour les modèles avec plusieurs têtes d'attention
            if len(layer_attention.shape) >= 4:  # [batch, num_heads, seq_len, seq_len]
                weights = layer_attention[0]  # Premier élément du batch
            else:  # [batch, seq_len, seq_len]
                weights = layer_attention[0, :, :, np.newaxis]  # Ajouter une dimension pour les têtes
                
            result['attention_weights'].append({
                'layer': layer_idx,
                'weights': weights
            })
        
        return result
    
    def _aggregate_attention_weights(self, attention_weights, method='mean', head_importance=None, layer_importance=None):
        """
        Agrège les poids d'attention de plusieurs couches et têtes.
        
        Args:
            attention_weights: Poids d'attention extraits
            method: Méthode d'agrégation ('mean', 'max', 'sum', 'last')
            head_importance: Importance relative des différentes têtes d'attention
            layer_importance: Importance relative des différentes couches
            
        Returns:
            numpy.ndarray: Poids d'attention agrégés
        """
        num_layers = attention_weights['num_layers']
        
        # Initialiser les poids agrégés
        if num_layers == 0:
            return np.array([])
            
        # Récupérer les dimensions de la première couche
        first_layer = attention_weights['attention_weights'][0]['weights']
        
        # Pour les modèles avec plusieurs têtes d'attention
        if len(first_layer.shape) == 3:  # [num_heads, seq_len, seq_len]
            num_heads = first_layer.shape[0]
            seq_len = first_layer.shape[1]
        else:  # [seq_len, seq_len]
            num_heads = 1
            seq_len = first_layer.shape[0]
            
        # Normaliser les importances des têtes et des couches si fournies
        if head_importance is not None:
            head_importance = np.array(head_importance)
            head_importance = head_importance / head_importance.sum()
        else:
            head_importance = np.ones(num_heads) / num_heads
            
        if layer_importance is not None:
            layer_importance = np.array(layer_importance)
            layer_importance = layer_importance / layer_importance.sum()
        else:
            layer_importance = np.ones(num_layers) / num_layers
        
        # Agréger selon la méthode spécifiée
        if method == 'last':
            # Utiliser uniquement la dernière couche
            last_layer = attention_weights['attention_weights'][-1]['weights']
            if len(last_layer.shape) == 3:  # [num_heads, seq_len, seq_len]
                # Pondérer les têtes d'attention
                weighted_heads = np.zeros((seq_len, seq_len))
                for h in range(num_heads):
                    weighted_heads += head_importance[h] * last_layer[h]
                return weighted_heads
            else:
                return last_layer
        else:
            # Initialiser le tenseur pour stocker les poids agrégés
            aggregated = np.zeros((seq_len, seq_len))
            
            # Agréger toutes les couches et têtes
            for l, layer_data in enumerate(attention_weights['attention_weights']):
                layer_weights = layer_data['weights']
                
                # Pondérer les têtes d'attention
                if len(layer_weights.shape) == 3:  # [num_heads, seq_len, seq_len]
                    layer_aggregated = np.zeros((seq_len, seq_len))
                    for h in range(num_heads):
                        layer_aggregated += head_importance[h] * layer_weights[h]
                else:
                    layer_aggregated = layer_weights
                
                # Appliquer la pondération de la couche
                aggregated += layer_importance[l] * layer_aggregated
            
            return aggregated
    
    def _convert_attention_to_importances(self, aggregated_attention, tokens, threshold=0.05, include_special_tokens=False):
        """
        Convertit les poids d'attention agrégés en importances de caractéristiques.
        
        Args:
            aggregated_attention: Poids d'attention agrégés
            tokens: Liste des tokens
            threshold: Seuil pour filtrer les poids d'attention faibles
            include_special_tokens: Inclure les tokens spéciaux dans l'explication
            
        Returns:
            list: Liste d'objets FeatureImportance
        """
        # Vérifier si les dimensions correspondent
        if aggregated_attention.shape[0] != len(tokens):
            self._logger.warning(f"Dimensions incompatibles: {aggregated_attention.shape[0]} vs {len(tokens)}. "
                              f"Troncature appliquée.")
            min_len = min(aggregated_attention.shape[0], len(tokens))
            aggregated_attention = aggregated_attention[:min_len, :min_len]
            tokens = tokens[:min_len]
        
        # Calculer l'importance de chaque token (somme des poids d'attention entrants)
        token_importances = aggregated_attention.sum(axis=0)
        
        # Normaliser les importances
        if token_importances.sum() > 0:
            token_importances = token_importances / token_importances.sum()
        
        # Créer les objets FeatureImportance
        importances = []
        for i, token in enumerate(tokens):
            # Filtrer les tokens spéciaux si demandé
            if not include_special_tokens and (token.startswith('[') and token.endswith(']')):
                continue
                
            # Filtrer les tokens avec une importance inférieure au seuil
            if token_importances[i] >= threshold:
                importances.append(FeatureImportance(
                    feature_name=token,
                    importance=float(token_importances[i]),
                    std=0.0,  # Pas d'écart-type pour une seule instance
                    additional_info={
                        'position': i,
                        'attention_in': float(aggregated_attention[:, i].sum()),
                        'attention_out': float(aggregated_attention[i, :].sum())
                    }
                ))
        
        # Trier par importance décroissante
        importances.sort(key=lambda x: x.importance, reverse=True)
        
        return importances
    
    def _convert_attention_to_serializable(self, attention_weights):
        """
        Convertit les poids d'attention en format sérialisable.
        
        Args:
            attention_weights: Poids d'attention extraits
            
        Returns:
            dict: Poids d'attention en format sérialisable
        """
        result = {
            'num_layers': attention_weights['num_layers'],
            'num_heads': attention_weights['num_heads'],
            'attention_weights': []
        }
        
        for layer_data in attention_weights['attention_weights']:
            result['attention_weights'].append({
                'layer': layer_data['layer'],
                'weights': layer_data['weights'].tolist() if isinstance(layer_data['weights'], np.ndarray) else layer_data['weights']
            })
            
        return result
    
    def _extract_metadata(self):
        """
        Extrait les métadonnées du modèle pour la traçabilité et la conformité réglementaire.
        """
        # Initialiser les métadonnées
        self._metadata = {
            'model_type': self._detect_model_type(),
            'task_type': self._detect_task_type(),
            'framework': self._framework,
            'attention_layers': self._attention_layer_names if self._attention_layer_names else 'auto-detected',
            'aggregation_method': self._aggregation_method,
            'model_name': self._get_model_name(),
            'timestamp': datetime.now().isoformat(),
            'explainer_version': '1.0.0',
            'explainer_config': {
                'include_special_tokens': self._include_special_tokens,
                'attention_threshold': self._attention_threshold
            }
        }
    
    def _detect_model_type(self):
        """
        Détecte le type de modèle (transformer, RNN, etc.).
        
        Returns:
            str: Type de modèle détecté
        """
        # Vérifier le nom du modèle ou de ses modules
        model_name = str(self._model.__class__.__name__).lower()
        model_module = self._model.__module__.lower()
        
        if any(name in model_name for name in ['bert', 'gpt', 'transformer', 't5', 'roberta', 'xlm', 'xlnet', 'distil']):
            return 'transformer'
        elif any(name in model_name for name in ['lstm', 'gru', 'rnn']):
            return 'rnn'
        elif 'attention' in model_name:
            return 'attention-based'
        elif 'transformers' in model_module:
            return 'transformer'
        else:
            # Essayer de détecter par les attributs
            if hasattr(self._model, 'transformer') or hasattr(self._model, 'encoder') and hasattr(self._model.encoder, 'layer'):
                return 'transformer'
            elif hasattr(self._model, 'lstm') or hasattr(self._model, 'gru'):
                return 'rnn'
            else:
                return 'unknown'
    
    def _detect_task_type(self):
        """
        Détecte le type de tâche (classification, génération de texte, etc.).
        
        Returns:
            str: Type de tâche détecté
        """
        # Vérifier le nom du modèle ou de ses modules
        model_name = str(self._model.__class__.__name__).lower()
        
        if any(name in model_name for name in ['classifier', 'classification']):
            return 'classification'
        elif any(name in model_name for name in ['regressor', 'regression']):
            return 'regression'
        elif any(name in model_name for name in ['generator', 'gpt', 'llm', 'decoder']):
            return 'text_generation'
        elif any(name in model_name for name in ['encoder', 'bert', 'embedding']):
            return 'embedding'
        elif any(name in model_name for name in ['seq2seq', 't5', 'translation']):
            return 'sequence_to_sequence'
        else:
            # Vérifier la structure de sortie du modèle
            if hasattr(self._model, 'config'):
                if hasattr(self._model.config, 'num_labels'):
                    return 'classification' if self._model.config.num_labels > 1 else 'regression'
                elif hasattr(self._model.config, 'is_decoder') and self._model.config.is_decoder:
                    return 'text_generation'
                elif hasattr(self._model.config, 'is_encoder_decoder') and self._model.config.is_encoder_decoder:
                    return 'sequence_to_sequence'
            
            # Par défaut, supposer qu'il s'agit d'un modèle de langage
            return 'language_model'
    
    def _get_model_name(self):
        """
        Récupère le nom du modèle.
        
        Returns:
            str: Nom du modèle
        """
        # Essayer de récupérer le nom du modèle depuis la configuration
        if hasattr(self._model, 'config') and hasattr(self._model.config, 'name_or_path'):
            return self._model.config.name_or_path
        elif hasattr(self._model, 'name'):
            return self._model.name
        else:
            # Utiliser le nom de la classe comme fallback
            return self._model.__class__.__name__
    
    def _get_num_heads(self, layer):
        """
        Récupère le nombre de têtes d'attention d'une couche.
        
        Args:
            layer: Couche d'attention
            
        Returns:
            int: Nombre de têtes d'attention
        """
        # Essayer de récupérer le nombre de têtes d'attention
        if hasattr(layer, 'num_heads'):
            return layer.num_heads
        elif hasattr(layer, 'num_attention_heads'):
            return layer.num_attention_heads
        elif hasattr(layer, 'attention_heads'):
            return layer.attention_heads
        else:
            # Valeur par défaut
            return 1
    
    def _sample_instances(self, X, y=None, max_instances=5, strategy='random'):
        """
        Échantillonne des instances représentatives pour l'explication globale.
        
        Args:
            X: Données d'entrée
            y: Valeurs cibles (optionnel)
            max_instances: Nombre maximum d'instances à échantillonner
            strategy: Stratégie d'échantillonnage ('random', 'stratified', 'diverse')
            
        Returns:
            list: Indices des instances échantillonnées
        """
        n_samples = len(X)
        max_instances = min(max_instances, n_samples)
        
        if strategy == 'random':
            # Échantillonnage aléatoire
            return np.random.choice(n_samples, size=max_instances, replace=False).tolist()
        
        elif strategy == 'stratified' and y is not None:
            # Échantillonnage stratifié par classe
            try:
                from sklearn.model_selection import StratifiedShuffleSplit
                
                # Vérifier si y est catégoriel
                unique_y = np.unique(y)
                if len(unique_y) < n_samples / 10:  # Heuristique pour détecter les variables catégorielles
                    sss = StratifiedShuffleSplit(n_splits=1, test_size=max_instances, random_state=42)
                    for _, test_idx in sss.split(X, y):
                        return test_idx.tolist()
            except (ImportError, ValueError):
                self._logger.warning("L'échantillonnage stratifié a échoué, utilisation de l'échantillonnage aléatoire.")
        
        elif strategy == 'diverse':
            # Échantillonnage divers basé sur k-means
            try:
                # Pour le texte, on ne peut pas utiliser k-means directement
                if isinstance(X[0], str):
                    return np.random.choice(n_samples, size=max_instances, replace=False).tolist()
                
                # Pour les tenseurs ou tokens pré-définis
                from sklearn.cluster import KMeans
                
                # Convertir en tableau numpy si nécessaire
                X_array = np.array(X)
                if len(X_array.shape) > 2:
                    # Aplatir les tenseurs pour k-means
                    X_array = X_array.reshape(X_array.shape[0], -1)
                
                # Appliquer k-means
                kmeans = KMeans(n_clusters=max_instances, random_state=42, n_init=10)
                kmeans.fit(X_array)
                
                # Sélectionner l'instance la plus proche de chaque centre
                closest_indices = []
                for center in kmeans.cluster_centers_:
                    distances = np.linalg.norm(X_array - center, axis=1)
                    closest_idx = np.argmin(distances)
                    closest_indices.append(closest_idx)
                
                return closest_indices
            except (ImportError, ValueError):
                self._logger.warning("L'échantillonnage divers a échoué, utilisation de l'échantillonnage aléatoire.")
        
        # Fallback sur l'échantillonnage aléatoire
        return np.random.choice(n_samples, size=max_instances, replace=False).tolist()
    
    def _aggregate_text_importances(self, all_feature_importances):
        """
        Agrège les importances de caractéristiques pour des textes différents.
        
        Args:
            all_feature_importances: Liste des importances de caractéristiques pour chaque instance
            
        Returns:
            list: Importances de caractéristiques agrégées
        """
        # Collecter tous les tokens uniques avec leurs importances
        token_importances = {}
        
        for instance_importances in all_feature_importances:
            for feature in instance_importances:
                token = feature.feature_name
                importance = feature.importance
                
                if token in token_importances:
                    token_importances[token]['sum'] += importance
                    token_importances[token]['count'] += 1
                    token_importances[token]['values'].append(importance)
                else:
                    token_importances[token] = {
                        'sum': importance,
                        'count': 1,
                        'values': [importance]
                    }
        
        # Calculer les moyennes et écarts-types
        aggregated_importances = []
        for token, data in token_importances.items():
            mean_importance = data['sum'] / data['count']
            std_importance = np.std(data['values']) if len(data['values']) > 1 else 0.0
            
            aggregated_importances.append(FeatureImportance(
                feature_name=token,
                importance=float(mean_importance),
                std=float(std_importance),
                additional_info={
                    'occurrence_count': data['count'],
                    'min_importance': float(min(data['values'])),
                    'max_importance': float(max(data['values']))
                }
            ))
        
        # Trier par importance décroissante
        aggregated_importances.sort(key=lambda x: x.importance, reverse=True)
        
        return aggregated_importances
    
    def _aggregate_feature_importances(self, all_feature_importances, feature_names):
        """
        Agrège les importances de caractéristiques pour des instances avec les mêmes caractéristiques.
        
        Args:
            all_feature_importances: Liste des importances de caractéristiques pour chaque instance
            feature_names: Noms des caractéristiques
            
        Returns:
            list: Importances de caractéristiques agrégées
        """
        # Initialiser les tableaux pour stocker les importances
        n_features = len(feature_names)
        importance_values = np.zeros((len(all_feature_importances), n_features))
        
        # Remplir le tableau d'importances
        for i, instance_importances in enumerate(all_feature_importances):
            for j, feature_name in enumerate(feature_names):
                # Trouver l'importance correspondante
                for feature in instance_importances:
                    if feature.feature_name == feature_name:
                        importance_values[i, j] = feature.importance
                        break
        
        # Calculer les moyennes et écarts-types
        mean_importances = np.mean(importance_values, axis=0)
        std_importances = np.std(importance_values, axis=0)
        
        # Créer les objets FeatureImportance
        aggregated_importances = []
        for i, feature_name in enumerate(feature_names):
            aggregated_importances.append(FeatureImportance(
                feature_name=feature_name,
                importance=float(mean_importances[i]),
                std=float(std_importances[i]),
                additional_info={
                    'min_importance': float(np.min(importance_values[:, i])),
                    'max_importance': float(np.max(importance_values[:, i]))
                }
            ))
        
        # Trier par importance décroissante
        aggregated_importances.sort(key=lambda x: x.importance, reverse=True)
        
        return aggregated_importances
