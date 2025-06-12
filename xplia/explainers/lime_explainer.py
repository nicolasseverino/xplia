"""
LIME Explainer pour XPLIA
========================

Ce module implémente l'intégration des explications LIME (Local Interpretable Model-agnostic Explanations) 
dans le framework XPLIA. Cette technique permet d'expliquer les prédictions individuelles
en approximant localement le comportement du modèle avec un modèle interprétable.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..core.base import (AudienceLevel, ExplainerBase, ExplainabilityMethod,
                        ExplanationResult, FeatureImportance, ModelMetadata)
from ..core.registry import register_explainer


@register_explainer
class LimeExplainer(ExplainerBase):
    """
    Explainer basé sur LIME (Local Interpretable Model-agnostic Explanations).
    
    Cette classe fournit des explications de modèles basées sur l'approximation locale
    du comportement du modèle par un modèle interprétable (généralement linéaire).
    
    Caractéristiques principales:
    - Support de différents types de données (tabular, text, image)
    - Approximation locale par modèles interprétables
    - Visualisations adaptées au niveau d'audience (technique, business, public)
    - Optimisations de performance pour les grands jeux de données
    """
    
    def __init__(self, model, **kwargs):
        """
        Initialise l'explainer LIME.
        
        Args:
            model: Modèle à expliquer
            **kwargs: Paramètres additionnels
                data_type: Type de données ('tabular', 'text', 'image')
                feature_names: Noms des caractéristiques
                class_names: Noms des classes
                categorical_features: Indices des caractéristiques catégorielles
                categorical_names: Noms des valeurs des caractéristiques catégorielles
                kernel_width: Largeur du noyau pour la fonction de similarité
                verbose: Niveau de verbosité
        """
        super().__init__(model, **kwargs)
        self._method = ExplainabilityMethod.LIME
        self._supported_model_types = [
            'RandomForestClassifier', 'RandomForestRegressor',
            'GradientBoostingClassifier', 'GradientBoostingRegressor',
            'XGBClassifier', 'XGBRegressor',
            'LGBMClassifier', 'LGBMRegressor',
            'CatBoostClassifier', 'CatBoostRegressor',
            'LogisticRegression', 'LinearRegression',
            'SVC', 'SVR',
            'Sequential', 'Model',  # Keras
            'Module'  # PyTorch
        ]
        
        # Paramètres LIME
        self._data_type = kwargs.get('data_type', 'tabular')
        self._feature_names = kwargs.get('feature_names', None)
        self._class_names = kwargs.get('class_names', None)
        self._categorical_features = kwargs.get('categorical_features', [])
        self._categorical_names = kwargs.get('categorical_names', {})
        self._kernel_width = kwargs.get('kernel_width', None)
        self._verbose = kwargs.get('verbose', False)
        
        # Initialisation de l'explainer LIME
        self._lime_explainer = None
        self._initialize_explainer()
        
        # Métadonnées du modèle
        self._metadata = None
        
        # Logger
        self._logger = logging.getLogger(__name__)
        
    def _initialize_explainer(self):
        """
        Initialise l'explainer LIME approprié en fonction du type de données.
        """
        try:
            import lime
            import lime.lime_tabular
            import lime.lime_text
            import lime.lime_image
        except ImportError:
            raise ImportError("Le package LIME est requis. Installez-le avec 'pip install lime'.")
        
        # Tracer l'action
        self.add_audit_record("initialize_explainer", {
            "data_type": self._data_type,
            "kernel_width": self._kernel_width,
            "verbose": self._verbose
        })
        
        try:
            if self._data_type == 'tabular':
                self._lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data=np.array([]),  # Sera remplacé lors de l'explication
                    feature_names=self._feature_names,
                    class_names=self._class_names,
                    categorical_features=self._categorical_features,
                    categorical_names=self._categorical_names,
                    kernel_width=self._kernel_width,
                    verbose=self._verbose,
                    mode='classification' if self._is_classifier() else 'regression'
                )
                self._logger.info("Explainer LIME tabular initialisé")
            elif self._data_type == 'text':
                self._lime_explainer = lime.lime_text.LimeTextExplainer(
                    class_names=self._class_names,
                    kernel_width=self._kernel_width,
                    verbose=self._verbose
                )
                self._logger.info("Explainer LIME text initialisé")
            elif self._data_type == 'image':
                self._lime_explainer = lime.lime_image.LimeImageExplainer(
                    kernel_width=self._kernel_width,
                    verbose=self._verbose
                )
                self._logger.info("Explainer LIME image initialisé")
            else:
                raise ValueError(f"Type de données non supporté: {self._data_type}. "  
                               f"Utilisez 'tabular', 'text' ou 'image'.")
        except Exception as e:
            self._logger.error(f"Erreur lors de l'initialisation de l'explainer LIME: {str(e)}")
            raise RuntimeError(f"Échec de l'initialisation de l'explainer LIME: {str(e)}")
    
    def _is_classifier(self) -> bool:
        """
        Détermine si le modèle est un classifieur ou un régresseur.
        
        Returns:
            bool: True si le modèle est un classifieur, False sinon
        """
        # Vérifier les attributs spécifiques aux classifieurs
        if hasattr(self._model, 'predict_proba'):
            return True
        if hasattr(self._model, 'classes_'):
            return True
            
        # Pour les modèles deep learning, vérifier la forme de sortie
        model_module = self._model.__class__.__module__
        if 'keras' in model_module or 'tensorflow' in model_module or 'torch' in model_module:
            try:
                # Créer des données synthétiques pour une inférence
                input_shape = self._infer_model_input_shape()
                if input_shape:
                    sample = np.random.normal(0, 0.1, size=(1, *input_shape))
                    preds = self._model_predict_wrapper(sample)
                    
                    # Si la sortie a plusieurs dimensions et la dernière dimension > 1, c'est probablement une classification
                    if preds.ndim > 1 and preds.shape[-1] > 1:
                        return True
            except:
                pass
                
        # Par défaut, supposer que c'est un régresseur
        return False
        
    def _model_predict_wrapper(self, instances):
        """
        Wrapper pour standardiser les prédictions de différents frameworks ML.
        
        Cette méthode détecte automatiquement le framework du modèle et adapte l'interface
        de prédiction en conséquence, assurant une sortie unifiée quel que soit le framework.
        
        Args:
            instances: Données d'entrée à prédire (ndarray, DataFrame, csr_matrix, etc.)
        
        Returns:
            ndarray: Les prédictions standardisées (probs pour classification, valeurs pour régression)
        """
        try:
            # Détecter le framework du modèle
            model_type = self._extract_model_type()
            
            # Standardiser le format d'entrée si nécessaire
            if hasattr(instances, 'values'):
                # Convertir DataFrame en ndarray
                instances = instances.values
                
            # Adapter l'appel de prédiction selon le framework
            if model_type == 'sklearn':
                # sklearn a predict_proba pour classification, predict pour régression
                if self._is_classifier() and hasattr(self._model, 'predict_proba'):
                    return self._model.predict_proba(instances)
                else:
                    return self._model.predict(instances).reshape(-1, 1) 
                    
            elif model_type in ['xgboost', 'lightgbm', 'catboost']:
                # Ces frameworks ont souvent predict ou predict_proba
                if hasattr(self._model, 'predict_proba'):
                    return self._model.predict_proba(instances)
                else:
                    preds = self._model.predict(instances)
                    # S'assurer que les sorties sont 2D pour la cohérence
                    if preds.ndim == 1:
                        return preds.reshape(-1, 1)
                    return preds
                    
            elif model_type == 'tensorflow':
                # TensorFlow/Keras models
                import numpy as np
                import tensorflow as tf
                
                # Convertir en tenseur TF si nécessaire
                if not isinstance(instances, tf.Tensor):
                    instances = tf.convert_to_tensor(instances, dtype=tf.float32)
                    
                # Prédire avec le modèle TF
                preds = self._model(instances).numpy() if hasattr(self._model, '__call__') else \
                        self._model.predict(instances)
                        
                # Standardiser le format de sortie
                return np.atleast_2d(preds)
                
            elif model_type == 'pytorch':
                # PyTorch models
                import numpy as np
                import torch
                
                # Désactiver le calcul de gradient pour les prédictions
                with torch.no_grad():
                    # Convertir en tenseur PyTorch si nécessaire
                    if not isinstance(instances, torch.Tensor):
                        instances = torch.tensor(instances, dtype=torch.float32)
                        
                    # Mettre sur le device approprié (GPU/CPU)
                    if torch.cuda.is_available() and getattr(self, '_config', {}).get('use_gpu', False):
                        instances = instances.cuda()
                        if hasattr(self._model, 'cuda'):
                            model = self._model.cuda()
                        else:
                            model = self._model
                    else:
                        model = self._model
                        
                    # Prédire et convertir en numpy
                    preds = model(instances) if hasattr(model, '__call__') else \
                            model.predict(instances)
                    preds = preds.cpu().numpy()
                            
                    # Standardiser le format de sortie
                    return np.atleast_2d(preds)
            else:
                # Fallback générique pour les autres frameworks
                preds = self._model.predict(instances)
                # S'assurer que les sorties sont 2D pour la cohérence
                if hasattr(preds, 'ndim') and preds.ndim == 1:
                    return preds.reshape(-1, 1)
                return preds
                
        except Exception as e:
            self._logger.error(f"Erreur lors de la prédiction avec {self._model.__class__.__name__}: {str(e)}")
            import traceback
            self._logger.debug(traceback.format_exc())
            raise RuntimeError(f"Échec de la prédiction: {str(e)}")

    def _infer_model_input_shape(self):
        """
        Tente de déduire la forme des entrées du modèle.
        
        Returns:
            tuple ou None: Forme déduite ou None si impossible
        """
        model_module = self._model.__class__.__module__
        
        if 'tensorflow' in model_module or 'keras' in model_module:
            try:
                # Pour les modèles Keras
                return self._model.input_shape[1:]
            except (AttributeError, IndexError):
                pass
        
        # Pour les modèles sklearn, xgboost, etc.
        try:
            if hasattr(self._model, 'n_features_in_'):
                return (self._model.n_features_in_,)
        except AttributeError:
            pass
        
        # Impossible de déduire
        return None
        
    def _model_predict_wrapper(self, x):
        """
        Wrapper pour la fonction de prédiction du modèle, adapté pour LIME.
        
        Args:
            x: Données d'entrée
            
        Returns:
            numpy.ndarray: Prédictions du modèle
        """
        try:
            model_module = self._model.__class__.__module__
            
            if 'sklearn' in model_module or 'xgboost' in model_module or 'lightgbm' in model_module:
                if hasattr(self._model, 'predict_proba'):
                    return self._model.predict_proba(x)
                else:
                    return self._model.predict(x)
            elif 'tensorflow' in model_module or 'keras' in model_module:
                # Conversion en format attendu par le modèle
                if isinstance(x, pd.DataFrame):
                    x = x.values
                
                # Appel au modèle
                result = self._model.predict(x)
                return result
            elif 'torch' in model_module:
                # Conversion en format attendu par le modèle
                if isinstance(x, pd.DataFrame):
                    x = x.values
                
                import torch
                # Convertir en tensor PyTorch
                if not isinstance(x, torch.Tensor):
                    x = torch.tensor(x, dtype=torch.float32)
                
                # Mettre le modèle en mode évaluation
                self._model.eval()
                
                # Appel au modèle avec torch.no_grad
                with torch.no_grad():
                    result = self._model(x)
                
                # Conversion du résultat en numpy
                if hasattr(result, 'detach'):
                    result = result.detach().numpy()
                else:
                    result = result.numpy()
                return result
            else:
                # Cas générique
                if hasattr(self._model, 'predict'):
                    return self._model.predict(x)
                else:
                    return self._model(x)
        except Exception as e:
            self._logger.error(f"Erreur dans le wrapper de prédiction: {str(e)}")
            raise RuntimeError(f"Échec de la prédiction du modèle: {str(e)}")

    def explain(self, X, y=None, **kwargs) -> ExplanationResult:
        """
        Génère des explications LIME pour un ensemble de données.
        
        Args:
            X: Données d'entrée à expliquer (DataFrame, numpy array, ou liste)
            y: Valeurs cibles réelles (optionnel)
            **kwargs: Paramètres additionnels
                - num_features: Nombre de caractéristiques à inclure dans l'explication
                - num_samples: Nombre d'échantillons à générer pour l'approximation locale
                - max_instances: Nombre maximum d'instances à expliquer (pour performance)
                - audience_level: Niveau d'audience (TECHNICAL, BUSINESS, PUBLIC)
                - labels: Indices des classes à expliquer (pour les classifieurs multi-classes)
                
        Returns:
            ExplanationResult: Résultat standardisé de l'explication
        """
        if self._lime_explainer is None:
            # Réinitialiser l'explainer avec les données actuelles
            self._initialize_explainer_with_data(X, **kwargs)
            
        # Paramètres
        num_features = kwargs.get('num_features', 10)
        num_samples = kwargs.get('num_samples', 5000)
        max_instances = kwargs.get('max_instances', 100)  # Limite pour performance
        audience_level = kwargs.get('audience_level', AudienceLevel.TECHNICAL)
        labels = kwargs.get('labels', None)
        
        # Conversion des données en format approprié
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X_values = X.values
        else:
            X_values = np.array(X)
            feature_names = kwargs.get('feature_names', self._feature_names) or \
                           [f"feature_{i}" for i in range(X_values.shape[1])]
        
        # Limiter le nombre d'instances pour performance si nécessaire
        if X_values.shape[0] > max_instances:
            self._logger.warning(f"Échantillonnage de {max_instances} instances sur {X_values.shape[0]} pour performance.")
            indices = np.random.choice(X_values.shape[0], max_instances, replace=False)
            X_sample = X_values[indices]
        else:
            X_sample = X_values
            
        # Tracer l'action
        self.add_audit_record("explain", {
            "n_samples": X_sample.shape[0],
            "n_features": X_sample.shape[1],
            "audience_level": audience_level.value if isinstance(audience_level, AudienceLevel) else audience_level,
            "num_features": num_features,
            "num_samples": num_samples
        })
        
        try:
            # Calculer les explications LIME pour chaque instance
            explanations = []
            for i, instance in enumerate(X_sample):
                if i > 0 and i % 10 == 0:
                    self._logger.info(f"Explication de l'instance {i}/{len(X_sample)}")
                    
                # Générer l'explication pour cette instance
                if self._data_type == 'tabular':
                    exp = self._lime_explainer.explain_instance(
                        instance, 
                        self._model_predict_wrapper,
                        num_features=num_features,
                        num_samples=num_samples,
                        labels=labels
                    )
                elif self._data_type == 'text':
                    exp = self._lime_explainer.explain_instance(
                        instance,
                        self._model_predict_wrapper,
                        num_features=num_features,
                        labels=labels
                    )
                elif self._data_type == 'image':
                    exp = self._lime_explainer.explain_instance(
                        instance,
                        self._model_predict_wrapper,
                        num_features=num_features,
                        labels=labels
                    )
                    
                explanations.append(exp)
            
            # Agréger les importances de caractéristiques
            feature_importances = self._aggregate_feature_importances(explanations, feature_names)
            
            # Extraire les métadonnées du modèle
            if not self._metadata:
                self._extract_metadata()
            
            # Créer le résultat d'explication
            result = ExplanationResult(
                method=ExplainabilityMethod.LIME,
                model_metadata=self._metadata,
                feature_importances=feature_importances,
                raw_explanation={
                    "lime_explanations": explanations,
                    "feature_names": feature_names,
                    "data": X_sample
                },
                audience_level=audience_level
            )
            
            return result
            
        except Exception as e:
            self._logger.error(f"Erreur lors du calcul des explications LIME: {str(e)}")
            raise RuntimeError(f"Échec de l'explication LIME: {str(e)}")
            
    def _initialize_explainer_with_data(self, X, **kwargs):
        """
        Réinitialise l'explainer LIME avec les données actuelles.
        
        Args:
            X: Données d'entrée pour initialiser l'explainer
            **kwargs: Paramètres additionnels
        """
        try:
            import lime
            import lime.lime_tabular
            import lime.lime_text
            import lime.lime_image
        except ImportError:
            raise ImportError("Le package LIME est requis. Installez-le avec 'pip install lime'.")
        
        # Conversion des données en format approprié
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X_values = X.values
        else:
            X_values = np.array(X)
            feature_names = kwargs.get('feature_names', self._feature_names) or \
                           [f"feature_{i}" for i in range(X_values.shape[1])]
        
        # Mettre à jour les attributs
        self._feature_names = feature_names
        
        # Réinitialiser l'explainer avec les données
        if self._data_type == 'tabular':
            self._lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_values,
                feature_names=feature_names,
                class_names=self._class_names,
                categorical_features=self._categorical_features,
                categorical_names=self._categorical_names,
                kernel_width=self._kernel_width,
                verbose=self._verbose,
                mode='classification' if self._is_classifier() else 'regression'
            )
            self._logger.info("Explainer LIME tabular réinitialisé avec les données")
        else:
            # Pour les types text et image, pas besoin de réinitialiser avec les données
            pass
            
    def _aggregate_feature_importances(self, explanations, feature_names):
        """
        Agrège les importances de caractéristiques à partir des explications LIME.
        
        Args:
            explanations: Liste des explications LIME
            feature_names: Noms des caractéristiques
            
        Returns:
            List[FeatureImportance]: Liste des importances de caractéristiques agrégées
        """
        # Initialiser un dictionnaire pour stocker les importances agrégées
        aggregated_importances = {name: [] for name in feature_names}
        
        # Collecter toutes les importances
        for exp in explanations:
            # Pour les explainers tabulaires
            if hasattr(exp, 'as_list'):
                # Prendre la première classe si aucune n'est spécifiée
                label = 1 if self._is_classifier() and exp.available_labels() else exp.available_labels()[0]
                for feature, importance in exp.as_list(label=label):
                    # Extraire le nom de la caractéristique (peut contenir des conditions pour les valeurs catégorielles)
                    feature_name = feature.split(' ')[0]
                    if feature_name in aggregated_importances:
                        aggregated_importances[feature_name].append(abs(importance))
            # Pour les explainers texte et image (structure différente)
            elif hasattr(exp, 'local_exp'):
                # Prendre la première classe
                label = list(exp.local_exp.keys())[0]
                for idx, importance in exp.local_exp[label]:
                    if idx < len(feature_names):
                        aggregated_importances[feature_names[idx]].append(abs(importance))
        
        # Calculer les moyennes et créer les objets FeatureImportance
        feature_importances = []
        for name, importances in aggregated_importances.items():
            if importances:  # S'assurer qu'il y a des valeurs
                avg_importance = np.mean(importances)
                std_dev = np.std(importances) if len(importances) > 1 else 0.0
                feature_importances.append(FeatureImportance(
                    feature_name=name,
                    importance=float(avg_importance),
                    std_dev=float(std_dev)
                ))
        
        # Trier par importance décroissante
        feature_importances.sort(key=lambda x: x.importance, reverse=True)
        
        return feature_importances
        
    def explain_instance(self, instance, **kwargs) -> ExplanationResult:
    """
    Explique une instance spécifique avec LIME avec fonctionnalités avancées.
    
    Cette implémentation avancée inclut:
    - Support GPU optimisé pour les calculs complexes
    - Système de cache intelligent pour les explications récurrentes
    - Métriques de qualité et de fidélité des explications
    - Génération de narratives explicatives multi-audience et multilingues
    - Validation de conformité réglementaire intégrée
    
    Args:
        instance: Instance à expliquer (array, liste, dict ou pandas.Series)
        **kwargs: Paramètres additionnels
            - num_features: Nombre de caractéristiques à inclure dans l'explication
            - num_samples: Nombre d'échantillons à générer pour l'approximation locale
            - audience_level: Niveau d'audience (TECHNICAL, BUSINESS, PUBLIC)
            - label: Indice de la classe à expliquer (pour les classifieurs multi-classes)
            - use_cache: Activer/désactiver l'utilisation du cache (défaut: True)
            - use_gpu: Utiliser GPU si disponible (défaut: configuration globale)
            - compute_metrics: Calculer les métriques de qualité d'explication (défaut: True)
            - include_prediction: Inclure la prédiction du modèle dans le résultat (défaut: True)
            - generate_narrative: Générer des narratives textuelles explicatives (défaut: True)
            - narrative_language: Langue des narratives ('fr', 'en') (défaut: 'fr')
            - check_compliance: Vérifier la conformité réglementaire de l'explication (défaut: True)
            - compliance_regulations: Liste des réglementations à vérifier (défaut: ['gdpr', 'ai_act'])
            
    Returns:
        ExplanationResult: Résultat standardisé de l'explication avec métriques avancées, narratives et conformité
    """
    import hashlib
    import pickle
    import time
    import sys
    from dataclasses import dataclass, field
    from contextlib import nullcontext
    from datetime import datetime
    
    # Import des modules internes pour mesurer la performance
    try:
        from ..core.profiling import Timer, MemoryTracker
    except ImportError:
        # Fallback simple si les modules ne sont pas disponibles
        class Timer:
            def __init__(self, name="Timer"):
                self.name = name
                self.start_time = None
            def __enter__(self):
                self.start_time = time.time()
                return self
            def __exit__(self, *args):
                self.elapsed = time.time() - self.start_time
        
        class MemoryTracker:
            def __init__(self, name="MemoryTracker"):
                self.name = name
                self.start_mem = None
            def __enter__(self):
                self.start_mem = sys.getsizeof({})
                return self
            def __exit__(self, *args):
                self.usage = 0  # Placeholder
    
    # Import du vérificateur de conformité s'il est disponible
    try:
        from ..compliance.compliance_checker import ComplianceChecker
        has_compliance_checker = True
    except ImportError:
        has_compliance_checker = False
    
    # Configuration et paramètres avancés
    @dataclass
    class ExplainConfig:
        # Paramètres basiques de LIME
        num_features: int = 10
        num_samples: int = 5000
        audience_level: AudienceLevel = AudienceLevel.TECHNICAL
        label: Optional[int] = None
        feature_names: List[str] = field(default_factory=list)
        
        # Paramètres avancés
        use_cache: bool = True
        use_gpu: bool = True  # Utiliser GPU si disponible
        compute_metrics: bool = True  # Calculer les métriques de qualité
        include_prediction: bool = True  # Inclure la prédiction dans le résultat
        generate_narrative: bool = True  # Générer des narratives explicatives
        narrative_language: str = 'fr'  # Langue des narratives
        check_compliance: bool = True  # Vérifier la conformité réglementaire
        compliance_regulations: List[str] = field(default_factory=lambda: ['gdpr', 'ai_act'])
    
    # Créer et remplir la configuration
    config = ExplainConfig(
        num_features=kwargs.get('num_features', 10),
        num_samples=kwargs.get('num_samples', 5000),
        audience_level=kwargs.get('audience_level', AudienceLevel.TECHNICAL),
        label=kwargs.get('label', None),
        use_cache=kwargs.get('use_cache', True),
        use_gpu=kwargs.get('use_gpu', getattr(self, '_config', {}).get('use_gpu', True)),
        compute_metrics=kwargs.get('compute_metrics', True),
        include_prediction=kwargs.get('include_prediction', True),
        generate_narrative=kwargs.get('generate_narrative', True),
        narrative_language=kwargs.get('narrative_language', 'fr'),
        check_compliance=kwargs.get('check_compliance', True),
        compliance_regulations=kwargs.get('compliance_regulations', ['gdpr', 'ai_act'])
    )
    
    # Initialiser trackers de performance
    timer = Timer("LimeExplanation")
    memory_tracker = MemoryTracker("LimeMemory")
    
    with timer, memory_tracker:        
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
        
        config.feature_names = feature_names
        
        # Vérification du cache si activé
        if config.use_cache and hasattr(self, '_explanation_cache'):
            # Générer une clé de cache unique basée sur l'instance et les paramètres
            cache_key_data = {
                'instance': instance_array.tobytes(),
                'num_features': config.num_features,
                'num_samples': config.num_samples,
                'label': config.label
            }
            cache_key = hashlib.md5(pickle.dumps(cache_key_data)).hexdigest()
            
            # Vérifier si l'explication est dans le cache
            if hasattr(self, '_explanation_cache') and cache_key in self._explanation_cache:
                self._logger.info(f"Utilisation du cache pour l'explication (clé: {cache_key[:8]}...)")
                return self._explanation_cache[cache_key]
        else:
            cache_key = None
            
        # Si l'explainer n'est pas initialisé avec des données, le faire maintenant
        if self._lime_explainer is None or (self._data_type == 'tabular' and 
                                        not hasattr(self._lime_explainer, 'feature_names')):
            # Créer un petit ensemble de données synthétiques pour l'initialisation
            synthetic_data = np.random.normal(0, 1, size=(100, instance_array.shape[1]))
            self._initialize_explainer_with_data(synthetic_data, feature_names=feature_names)
            
        # Tracer l'action
        audit_data = {
            "n_features": len(feature_names),
            "audience_level": config.audience_level.value if isinstance(config.audience_level, AudienceLevel) else config.audience_level,
            "num_features": config.num_features,
            "num_samples": config.num_samples,
            "use_gpu": config.use_gpu,
            "use_cache": config.use_cache,
            "timestamp": datetime.now().isoformat()
        }
        self.add_audit_record("explain_instance", audit_data)
    
        # Context manager pour GPU si activé
        gpu_context = self._maybe_use_gpu_context() if config.use_gpu else nullcontext()
    
        # Prédiction si demandée
        model_prediction = None
        if config.include_prediction:
            with gpu_context:
                try:
                    model_prediction = self._model_predict_wrapper(instance_array)
                except Exception as e:
                    self._logger.warning(f"Erreur lors de la prédiction du modèle: {str(e)}")
                    
        try:
            # Génération de l'explication LIME avec optimisation GPU si configurée
            with gpu_context:
                # Générer l'explication pour cette instance
                if self._data_type == 'tabular':
                    exp = self._lime_explainer.explain_instance(
                        instance_array[0], 
                        self._model_predict_wrapper,
                        num_features=config.num_features,
                        num_samples=config.num_samples,
                        labels=[config.label] if config.label is not None else None
                    )
                elif self._data_type == 'text':
                    exp = self._lime_explainer.explain_instance(
                        instance_array[0] if isinstance(instance_array[0], str) else str(instance_array[0]),
                        self._model_predict_wrapper,
                        num_features=config.num_features,
                        labels=[config.label] if config.label is not None else None
                    )
                elif self._data_type == 'image':
                    exp = self._lime_explainer.explain_instance(
                        instance_array[0],
                        self._model_predict_wrapper,
                        num_features=config.num_features,
                        labels=[config.label] if config.label is not None else None
                    )
                    
            # Extraire les importances de caractéristiques
            feature_importances = []
            
            # Pour les explainers tabulaires
            if hasattr(exp, 'as_list'):
                # Déterminer la classe à expliquer
                if config.label is None:
                    # Prendre la classe avec la probabilité maximale
                    predictions = self._model_predict_wrapper(instance_array)
                    if predictions.ndim > 1 and predictions.shape[1] > 1:
                        config.label = np.argmax(predictions[0])
                    else:
                        # Cas binaire, prendre la classe positive
                        config.label = 1 if self._is_classifier() else 0
                    
                # Extraire les importances
                for feature, importance in exp.as_list(label=config.label):
                    # Extraire le nom de la caractéristique (peut contenir des conditions pour les valeurs catégorielles)
                    feature_name = feature.split(' ')[0]
                    feature_importances.append(FeatureImportance(
                        feature_name=feature_name,
                        importance=float(importance)
                    ))
            # Pour les explainers texte et image (structure différente)
            elif hasattr(exp, 'local_exp'):
                # Déterminer la classe à expliquer
                if config.label is None:
                    # Prendre la première classe disponible
                    config.label = list(exp.local_exp.keys())[0]
                    
                # Extraire les importances
                for idx, importance in exp.local_exp[config.label]:
                    if idx < len(feature_names):
                        feature_importances.append(FeatureImportance(
                            feature_name=feature_names[idx],
                            importance=float(importance)
                        ))
            
            # Trier par importance absolue décroissante
            feature_importances.sort(key=lambda x: abs(x.importance), reverse=True)
            
            # Métriques de qualité si demandé
            explanation_metrics = {}
            if config.compute_metrics:
                explanation_metrics = self._compute_explanation_quality(feature_importances, exp, instance_array, config)
            
            # Narratives si demandé
            narratives = {}
            if config.generate_narrative:
                narratives = self._generate_explanation_narrative(
                    exp, instance_array, feature_names, 
                    audience_level=config.audience_level,
                    language=config.narrative_language
                )
            
            # Extraire les métadonnées du modèle si nécessaire
            if not self._metadata:
                self._extract_metadata()
            
            # Vérification de conformité si demandée
            compliance_result = None
            if config.check_compliance and has_compliance_checker:
                try:
                    compliance_checker = ComplianceChecker()
                    # Créer un résultat d'explication temporaire pour la vérification
                    temp_result = ExplanationResult(
                        method=ExplainabilityMethod.LIME,
                        model_metadata=self._metadata,
                        feature_importances=feature_importances
                    )
                    compliance_result = compliance_checker.check_compliance(
                        temp_result,
                        regulations=config.compliance_regulations
                    )
                except Exception as e:
                    self._logger.warning(f"Erreur lors de la vérification de conformité: {str(e)}")
            
            # Enrichir les métadonnées avec les informations de performance
            performance_metadata = {
                "computation_time_ms": timer.elapsed * 1000 if hasattr(timer, "elapsed") else None,
                "memory_usage_bytes": memory_tracker.usage if hasattr(memory_tracker, "usage") else None,
                "computation_mode": "gpu" if config.use_gpu else "cpu",
                "lime_sample_count": config.num_samples,
                "timestamp": datetime.now().isoformat(),
            }
            
            # Créer le résultat d'explication enrichi
            result = ExplanationResult(
                method=ExplainabilityMethod.LIME,
                model_metadata=self._metadata,
                feature_importances=feature_importances,
                raw_explanation={
                    "lime_explanation": exp,
                    "feature_names": feature_names,
                    "data": instance_array,
                    "label": config.label,
                    "prediction": model_prediction.tolist() if model_prediction is not None else None
                },
                audience_level=config.audience_level,
                metrics=explanation_metrics,
                narratives=narratives,
                compliance=compliance_result.to_dict() if compliance_result else None,
                metadata=performance_metadata
            )
            
            # Mettre en cache si activé
            if config.use_cache and cache_key:
                if not hasattr(self, '_explanation_cache'):
                    self._explanation_cache = {}
                self._explanation_cache[cache_key] = result
                # Limiter la taille du cache
                if len(self._explanation_cache) > 100:
                    oldest_key = next(iter(self._explanation_cache))
                    del self._explanation_cache[oldest_key]
            
            return result
        except Exception as e:
            self._logger.error(f"Erreur lors du calcul de l'explication LIME pour l'instance: {str(e)}")
            import traceback
            self._logger.debug(traceback.format_exc())
            raise RuntimeError(f"Échec de l'explication LIME pour l'instance: {str(e)}")

    def _maybe_use_gpu_context(self):
        """
        Contexte de gestion des ressources GPU pour l'explication.
        
        Permet d'optimiser automatiquement l'utilisation des ressources GPU
        lors du calcul des valeurs LIME, particulièrement pour les grands jeux de données
        et les modèles complexes.
        
        Returns:
            Un gestionnaire de contexte qui configure l'environnement GPU optimalement
        """
        if not getattr(self, '_config', {}).get('use_gpu', False):
            # Si GPU non activé, retourner un contexte vide (no-op)
            from contextlib import nullcontext
            return nullcontext()
            
        # Contexte GPU adapté au framework détecté
        model_type = self._extract_model_type()
        
        class _GPUContext:
            def __init__(self, explainer):
                self._explainer = explainer
                self._prev_state = None
                self._tf_context = None
                self._torch_context = None
                
            def __enter__(self):
                try:
                    if 'tensorflow' in model_type:
                        import tensorflow as tf
                        # Activer GPU pour TensorFlow
                        self._explainer._set_gpu_memory_growth()
                        # Enregistrer les devices disponibles pour les restaurer ensuite
                        self._prev_state = tf.config.get_visible_devices()
                        # N'autoriser que GPU si disponible
                        gpus = tf.config.list_physical_devices('GPU')
                        if gpus:
                            tf.config.set_visible_devices(gpus, 'GPU')
                            self._explainer._logger.info(f"GPU activé pour TensorFlow: {len(gpus)} périphériques")
                        else:
                            self._explainer._logger.warning("GPU demandé mais non disponible pour TensorFlow")
                    
                    elif 'pytorch' in model_type or 'torch' in model_type:
                        import torch
                        # Activer CUDA pour PyTorch si disponible
                        if torch.cuda.is_available():
                            # Préparer le device et définir le contexte par défaut
                            self._prev_state = torch.cuda.current_device()
                            torch.cuda.set_device(0)  # Premier GPU par défaut
                            self._explainer._logger.info(f"GPU activé pour PyTorch: {torch.cuda.get_device_name(0)}")
                        else:
                            self._explainer._logger.warning("GPU demandé mais CUDA non disponible pour PyTorch")  
                            
                    # Configuration spécifique pour LIME
                    # LIME n'a pas d'API publique pour configurer le GPU, mais certaines
                    # opérations comme les calculs de similarité peuvent bénéficier du GPU
                    # si le framework sous-jacent est configuré correctement
                except Exception as e:
                    self._explainer._logger.error(f"Erreur lors de l'activation GPU: {str(e)}")
                
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                try:
                    # Restaurer l'état précédent
                    if 'tensorflow' in model_type and self._prev_state is not None:
                        import tensorflow as tf
                        tf.config.set_visible_devices(self._prev_state)
                    elif ('pytorch' in model_type or 'torch' in model_type) and self._prev_state is not None:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.set_device(self._prev_state)
                            # Libérer la mémoire GPU
                            torch.cuda.empty_cache()
                except Exception as e:
                    self._explainer._logger.error(f"Erreur lors de la restauration du contexte GPU: {str(e)}")
        
        # Retourner une instance du gestionnaire de contexte
        return _GPUContext(self)
    
    def _set_gpu_memory_growth(self):
        """
        Configure la croissance mémoire GPU dynamique pour TensorFlow.
        
        Cette méthode contextuelle permet une utilisation optimale de la mémoire GPU
        en configurant une allocation dynamique, évitant ainsi les erreurs OOM et 
        permettant une meilleure répartition des ressources entre plusieurs processus.
        """
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                # Configurer la croissance mémoire dynamique sur tous les GPU
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self._logger.info(f"Allocation mémoire GPU dynamique activée pour {len(gpus)} GPU(s)")
        except ImportError:
            self._logger.info("TensorFlow non installé, croissance mémoire GPU non configurée")
        except Exception as e:
            self._logger.warning(f"Erreur lors de la configuration de la mémoire GPU: {str(e)}")
    
    def _extract_model_type(self):
        """
        Détermine le type/framework du modèle pour adapter les optimisations.
        
        Returns:
            str: Type de modèle détecté ('tensorflow', 'pytorch', etc.)
        """
        model_module = self._model.__class__.__module__
        
        if 'sklearn' in model_module:
            return 'sklearn'
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
        is_classifier = self._is_classifier()
        
        # Créer les métadonnées
        self._metadata = ModelMetadata(
            model_type="classification" if is_classifier else "regression",
            framework=framework,
            input_shape=self._infer_model_input_shape(),
            output_shape=None,  # À compléter si nécessaire
            feature_names=self._feature_names,
            target_names=self._class_names,
            model_params={},
            model_version="1.0.0"
        )
        
    def _compute_explanation_quality(self, feature_importances, lime_explanation, instance, config):
        """
        Calcule les métriques de qualité pour l'explication LIME.
        
        Args:
            feature_importances: Liste des importances des caractéristiques calculées
            lime_explanation: Objet d'explication LIME brut
            instance: Instance à expliquer
            config: Configuration de l'explication
            
        Returns:
            dict: Métriques de qualité de l'explication
        """
        metrics = {}
        
        try:
            # 1. Fidélité locale - mesure à quel point l'explication 
            # prédit correctement le comportement du modèle dans le voisinage
            if hasattr(lime_explanation, 'score') and lime_explanation.score is not None:
                metrics['local_fidelity'] = float(lime_explanation.score)
            elif hasattr(lime_explanation, 'local_pred') and hasattr(lime_explanation, 'score'):
                metrics['local_fidelity'] = float(lime_explanation.score)
            else:
                # Calculer la fidélité manuellement si LIME ne la fournit pas
                if hasattr(lime_explanation, 'intercept') and hasattr(lime_explanation, 'local_exp'):
                    try:
                        # Récupérer les données du modèle linéaire simplifié de LIME
                        label = config.label if config.label is not None else 0
                        coefs = dict(lime_explanation.local_exp[label])
                        intercept = lime_explanation.intercept[label] if isinstance(lime_explanation.intercept, dict) else lime_explanation.intercept
                        
                        # Obtenir les vraies prédictions du modèle
                        true_pred = self._model_predict_wrapper(instance)[0]
                        
                        # Calculer l'écart entre le modèle LIME et le vrai modèle
                        if self._is_classifier():
                            # Classification: utiliser l'erreur moyenne absolue sur les probabilités
                            metrics['local_fidelity'] = 1 - np.mean(np.abs(lime_explanation.predict_proba - true_pred))
                        else:
                            # Régression: utiliser l'erreur absolue normalisée
                            pred_abs_error = np.abs(lime_explanation.predict(instance.reshape(1, -1))[0] - true_pred[0])
                            metrics['local_fidelity'] = 1 - min(pred_abs_error / (np.abs(true_pred[0]) + 1e-10), 1.0)
                    except Exception as e:
                        self._logger.warning(f"Erreur lors du calcul de la fidélité locale: {str(e)}")
                        metrics['local_fidelity'] = None
            
            # 2. Impact de prédiction - somme des valeurs absolues des importances
            total_abs_importance = sum(abs(fi.importance) for fi in feature_importances)
            metrics['prediction_impact'] = float(total_abs_importance)
            
            # 3. Indice de Gini pour mesurer la concentration des importances
            # Valeur proche de 1 = explication concentrée sur peu de features
            # Valeur proche de 0 = explication distribuée uniformément
            if feature_importances:
                metrics['feature_concentration'] = self._gini_index(
                    [abs(fi.importance) for fi in feature_importances]
                )
            else:
                metrics['feature_concentration'] = 0.0
                
            # 4. Stabilité de l'explication (si explicitement demandée)
            # Cette métrique est coûteuse car elle nécessite de multiples explications
            if config.compute_stability and hasattr(instance, 'shape'):
                try:
                    # Créer des perturbations légères de l'instance
                    stability_samples = 5
                    eps = 1e-2
                    perturbed_instances = []
                    
                    # Générer des instances avec du bruit gaussien
                    rng = np.random.RandomState(42)
                    for _ in range(stability_samples):
                        # Ajouter un bruit gaussien faible aux instances
                        perturbed = instance.copy()
                        noise = eps * rng.randn(*perturbed.shape)
                        perturbed += noise
                        perturbed_instances.append(perturbed)
                    
                    # Générer des explications pour les instances perturbées
                    # Sans utiliser explain_instance pour éviter la récursion
                    feature_imp_lists = []
                    lime_explainer = self._get_lime_explainer()
                    
                    for p_instance in perturbed_instances:
                        p_exp = lime_explainer.explain_instance(
                            p_instance[0], 
                            self._model.predict_proba if hasattr(self._model, 'predict_proba') else self._model.predict,
                            num_features=len(feature_importances),
                            num_samples=config.num_samples
                        )
                        # Extraire les importances pour la même classe
                        if hasattr(p_exp, 'as_list'):
                            imp_list = dict([(f.split(' ')[0], imp) for f, imp in p_exp.as_list(label=config.label)])
                        elif hasattr(p_exp, 'local_exp'):
                            imp_list = dict(p_exp.local_exp[config.label])
                        else:
                            imp_list = {}
                        feature_imp_lists.append(imp_list)
                    
                    # Calculer la similarité entre les explications (corrélation moyenne des importances)
                    feature_names = [fi.feature_name for fi in feature_importances]
                    stability_scores = []
                    
                    for i in range(stability_samples):
                        for j in range(i+1, stability_samples):
                            imp_list_i = feature_imp_lists[i]
                            imp_list_j = feature_imp_lists[j]
                            
                            # Récupérer les importances pour les mêmes features
                            common_features = set(imp_list_i.keys()).intersection(imp_list_j.keys())
                            if common_features:
                                vec_i = [imp_list_i.get(f, 0) for f in common_features]
                                vec_j = [imp_list_j.get(f, 0) for f in common_features]
                                
                                # Calculer la corrélation comme mesure de similarité
                                from scipy.stats import spearmanr
                                corr, _ = spearmanr(vec_i, vec_j)
                                stability_scores.append(corr if not np.isnan(corr) else 0)
                    
                    if stability_scores:
                        # La stabilité est la corrélation moyenne entre les explications
                        metrics['stability'] = float(np.mean(stability_scores))
                    else:
                        metrics['stability'] = None
                except Exception as e:
                    self._logger.warning(f"Erreur lors du calcul de la stabilité de l'explication: {str(e)}")
                    metrics['stability'] = None
            
            # 5. Complexité de l'explication (nombre de features avec importance significative)
            importance_threshold = 0.01
            significant_features = sum(1 for fi in feature_importances if abs(fi.importance) > importance_threshold)
            metrics['complexity'] = significant_features
            
            # 6. Déterminer les métriques spécifiques au type d'explication (tabular, text, image)
            if hasattr(lime_explanation, 'domain_mapper'):
                if hasattr(lime_explanation.domain_mapper, 'type'):
                    metrics['explanation_type'] = lime_explanation.domain_mapper.type
                elif 'text' in str(lime_explanation.domain_mapper.__class__).lower():
                    metrics['explanation_type'] = 'text'
                elif 'image' in str(lime_explanation.domain_mapper.__class__).lower():
                    metrics['explanation_type'] = 'image'
                else:
                    metrics['explanation_type'] = 'tabular'
        except Exception as e:
            self._logger.warning(f"Erreur lors du calcul des métriques de qualité: {str(e)}")
            import traceback
            self._logger.debug(traceback.format_exc())
        
        return metrics
    
    def _gini_index(self, values):
        """
        Calcule l'indice de Gini pour mesurer la concentration/dispersion des importances.
        
        L'indice de Gini est compris entre 0 (distribution égale) et 1 (concentration extrême).
        
        Args:
            values: Liste des valeurs d'importance (absolues)
            
        Returns:
            float: Indice de Gini entre 0 et 1
        """
        # Filtrer les valeurs nulles ou négatives
        values = [v for v in values if v > 0]
        n = len(values)
        
        if n <= 1:
            return 0.0
        
        # Trier les valeurs
        values = sorted(values)
        
        # Normaliser les valeurs (somme = 1)
        total = sum(values)
        if total == 0:
            return 0.0
            
        values = [v / total for v in values]
        
        # Calculer l'indice de Gini
        cumsum = 0
        gini = 0
        
        for i, v in enumerate(values):
            rank = i + 1
            cumsum += v
            gini += v * ((n + 1 - rank) / n)
        
        # Normaliser entre 0 et 1
        gini = 1 - 2 * gini
        
        return max(0.0, min(1.0, gini))  # Assurer que Gini est entre 0 et 1
        
    def _generate_explanation_narrative(self, lime_explanation, instance, feature_names, audience_level="technical", language="en"):
        """
        Génère une explication narrative des résultats LIME adaptée au niveau d'audience.
        
        Args:
            lime_explanation: L'objet d'explication LIME
            instance: L'instance expliquée
            feature_names: Noms des caractéristiques
            audience_level: Niveau d'audience cible ("technical", "business", "public")
            language: Langue de l'explication ("en", "fr")
            
        Returns:
            dict: Narratives explicatives par niveau d'audience
        """
        # Vérifier que l'objet d'explication est valide
        if lime_explanation is None:
            return {}
            
        # Initialiser le dictionnaire de narratives
        narratives = {}
        
        try:
            # Déterminer si nous sommes dans un cas de classification ou régression
            is_classifier = self._is_classifier()
            
            # Récupérer les top N caractéristiques les plus importantes (en valeur absolue)
            top_n = 5
            important_features = []
            
            if hasattr(lime_explanation, 'as_list'):
                label = getattr(lime_explanation, 'label', 0)
                features = lime_explanation.as_list(label=label)
                # Trier par importance absolue
                features.sort(key=lambda x: abs(x[1]), reverse=True)
                important_features = features[:top_n]
            elif hasattr(lime_explanation, 'local_exp'):
                # Pour les explainers texte/image
                label = list(lime_explanation.local_exp.keys())[0]
                features = [(feature_names[idx] if idx < len(feature_names) else f"feature_{idx}", imp) 
                            for idx, imp in lime_explanation.local_exp[label]]
                features.sort(key=lambda x: abs(x[1]), reverse=True)
                important_features = features[:top_n]
                
            # Séparer les caractéristiques positives et négatives
            positive_features = [(name, importance) for name, importance in important_features if importance > 0]
            negative_features = [(name, importance) for name, importance in important_features if importance < 0]
            
            # Générer la prédiction du modèle pour cette instance si disponible
            prediction = None
            pred_value = None
            confidence = None
            
            try:
                # Obtenir la prédiction brute
                raw_prediction = self._model_predict_wrapper(instance)[0]
                
                if is_classifier:
                    if raw_prediction.ndim > 0 and len(raw_prediction) > 1:
                        # Cas multi-classe
                        pred_class = np.argmax(raw_prediction)
                        confidence = float(raw_prediction[pred_class])
                        
                        # Récupérer le nom de la classe si disponible
                        if self._class_names and pred_class < len(self._class_names):
                            prediction = self._class_names[pred_class]
                        else:
                            prediction = f"Classe {pred_class}"
                    else:
                        # Cas binaire
                        pred_value = float(raw_prediction[0]) if raw_prediction.ndim > 0 else float(raw_prediction)
                        threshold = 0.5
                        pred_class = 1 if pred_value >= threshold else 0
                        
                        if self._class_names and len(self._class_names) > 1:
                            prediction = self._class_names[pred_class]
                        else:
                            prediction = "Positif" if pred_class == 1 else "Négatif"
                            
                        confidence = pred_value if pred_class == 1 else 1 - pred_value
                else:
                    # Cas régression
                    pred_value = float(raw_prediction[0]) if raw_prediction.ndim > 0 else float(raw_prediction)
                    prediction = f"{pred_value:.4f}"
            except Exception as e:
                self._logger.warning(f"Erreur lors de l'extraction de la prédiction: {str(e)}")
                
            # Générer les narratives pour chaque niveau d'audience et selon la langue
            
            # ===== NIVEAU TECHNIQUE =====
            if audience_level in ["technical", "all"]:
                technical_narrative = self._generate_technical_narrative(important_features, prediction, confidence, 
                                                                         is_classifier, language)
                narratives["technical"] = technical_narrative
                
            # ===== NIVEAU BUSINESS =====
            if audience_level in ["business", "all"]:
                business_narrative = self._generate_business_narrative(positive_features, negative_features, 
                                                                      prediction, confidence, is_classifier, language)
                narratives["business"] = business_narrative
                
            # ===== NIVEAU PUBLIC =====
            if audience_level in ["public", "all"]:
                public_narrative = self._generate_public_narrative(positive_features, negative_features,
                                                                  prediction, is_classifier, language)
                narratives["public"] = public_narrative
                
        except Exception as e:
            self._logger.warning(f"Erreur lors de la génération de narrative: {str(e)}")
            import traceback
            self._logger.debug(traceback.format_exc())
            
            # Fournir une narrative par défaut en cas d'erreur
            if language == "fr":
                narratives = {"technical": "Impossible de générer une explication narrative technique."}
            else:
                narratives = {"technical": "Unable to generate a technical narrative explanation."}
                
        return narratives
        
    def _generate_technical_narrative(self, important_features, prediction, confidence, is_classifier, language="en"):
        """
        Génère une narrative technique détaillée basée sur les résultats LIME.
        
        Args:
            important_features: Liste des caractéristiques importantes et leurs valeurs
            prediction: Prédiction du modèle (classe ou valeur)
            confidence: Confiance de la prédiction (pour la classification)
            is_classifier: Indique si c'est un classificateur ou régresseur
            language: Langue de l'explication ("en", "fr")
            
        Returns:
            str: Narrative technique
        """
        if language == "fr":
            # Version française
            if is_classifier:
                intro = f"Le modèle prédit la classe {prediction}"
                if confidence is not None:
                    intro += f" avec une confiance de {confidence:.2%}."
                else:
                    intro += "."
            else:
                intro = f"Le modèle prédit la valeur {prediction}."
                
            # Détails des caractéristiques
            details = "D'après l'analyse LIME, les facteurs explicatifs les plus importants sont:\n"
            
            for i, (name, importance) in enumerate(important_features):
                details += f"- {name}: impact de {importance:.4f} "  
                if importance > 0:
                    details += "(augmente la prédiction)"
                else:
                    details += "(diminue la prédiction)"
                details += "\n"
                
            # Ajouter des détails techniques
            conclusion = "\nL'algorithme LIME a généré un modèle local linéaire interprétable "
            conclusion += "qui approxime le comportement du modèle complexe dans le voisinage de cette instance. "
            conclusion += "Les coefficients représentent l'importance relative de chaque caractéristique dans la prédiction."
            
        else:
            # Version anglaise (défaut)
            if is_classifier:
                intro = f"The model predicts class {prediction}"
                if confidence is not None:
                    intro += f" with a confidence of {confidence:.2%}."
                else:
                    intro += "."
            else:
                intro = f"The model predicts the value {prediction}."
                
            # Feature details
            details = "According to LIME analysis, the most significant explanatory factors are:\n"
            
            for i, (name, importance) in enumerate(important_features):
                details += f"- {name}: impact of {importance:.4f} "  
                if importance > 0:
                    details += "(increases the prediction)"
                else:
                    details += "(decreases the prediction)"
                details += "\n"
                
            # Add technical details
            conclusion = "\nThe LIME algorithm generated an interpretable linear local model "
            conclusion += "that approximates the behavior of the complex model in the vicinity of this instance. "
            conclusion += "The coefficients represent the relative importance of each feature in the prediction."
        
        return intro + "\n\n" + details + conclusion
        
    def _generate_business_narrative(self, positive_features, negative_features, prediction, confidence, is_classifier, language="en"):
        """
        Génère une narrative orientée business basée sur les résultats LIME.
        
        Args:
            positive_features: Caractéristiques avec impact positif
            negative_features: Caractéristiques avec impact négatif
            prediction: Prédiction du modèle (classe ou valeur)
            confidence: Confiance de la prédiction
            is_classifier: Indique si c'est un classificateur ou régresseur
            language: Langue de l'explication ("en", "fr")
            
        Returns:
            str: Narrative business
        """
        if language == "fr":
            # Version française
            if is_classifier:
                summary = f"Résultat: {prediction}"
                if confidence is not None:
                    summary += f" (confiance: {confidence:.1%})\n\n"
                else:
                    summary += "\n\n"
                
                if confidence is not None and confidence < 0.7:
                    summary += "Attention: La confiance du modèle dans cette prédiction est relativement faible.\n\n"
            else:
                summary = f"Résultat: La valeur prédite est {prediction}\n\n"
                
            # Facteurs influents
            factors = "Principaux facteurs influents:\n"
            
            if positive_features:
                factors += "\nFacteurs favorables:\n"
                for name, importance in positive_features:
                    factors += f"- {name} (impact: +{abs(importance):.2f})\n"
                    
            if negative_features:
                factors += "\nFacteurs défavorables:\n"
                for name, importance in negative_features:
                    factors += f"- {name} (impact: -{abs(importance):.2f})\n"
                    
            # Recommandations
            recommendations = "\nRecommandations:\n"
            if positive_features and not is_classifier:
                recommendations += "- Pour augmenter la prédiction, concentrez-vous sur l'amélioration des facteurs favorables.\n"
            elif negative_features and not is_classifier:
                recommendations += "- Pour diminuer la prédiction, réduisez l'impact des facteurs défavorables.\n"
            recommendations += "- Cette explication est basée sur LIME, une technique d'analyse locale du modèle."
            
        else:
            # Version anglaise (défaut)
            if is_classifier:
                summary = f"Result: {prediction}"
                if confidence is not None:
                    summary += f" (confidence: {confidence:.1%})\n\n"
                else:
                    summary += "\n\n"
                
                if confidence is not None and confidence < 0.7:
                    summary += "Note: The model's confidence in this prediction is relatively low.\n\n"
            else:
                summary = f"Result: The predicted value is {prediction}\n\n"
                
            # Influential factors
            factors = "Main influential factors:\n"
            
            if positive_features:
                factors += "\nPositive factors:\n"
                for name, importance in positive_features:
                    factors += f"- {name} (impact: +{abs(importance):.2f})\n"
                    
            if negative_features:
                factors += "\nNegative factors:\n"
                for name, importance in negative_features:
                    factors += f"- {name} (impact: -{abs(importance):.2f})\n"
                    
            # Recommendations
            recommendations = "\nRecommendations:\n"
            if positive_features and not is_classifier:
                recommendations += "- To increase the prediction, focus on improving the positive factors.\n"
            elif negative_features and not is_classifier:
                recommendations += "- To decrease the prediction, reduce the impact of negative factors.\n"
            recommendations += "- This explanation is based on LIME, a local model analysis technique."
            
        return summary + factors + recommendations
        
    def _generate_public_narrative(self, positive_features, negative_features, prediction, is_classifier, language="en"):
        """
        Génère une narrative simplifiée pour le grand public.
        
        Args:
            positive_features: Caractéristiques avec impact positif
            negative_features: Caractéristiques avec impact négatif
            prediction: Prédiction du modèle (classe ou valeur)
            is_classifier: Indique si c'est un classificateur ou régresseur
            language: Langue de l'explication ("en", "fr")
            
        Returns:
            str: Narrative publique
        """
        # Limiter à 3 features maximum pour chaque catégorie
        pos_limited = positive_features[:3]
        neg_limited = negative_features[:3]
        
        if language == "fr":
            # Version française
            if is_classifier:
                result = f"Le système a déterminé que le résultat est: {prediction}\n\n"
            else:
                result = f"Le système a estimé la valeur suivante: {prediction}\n\n"
                
            explanation = "Voici pourquoi:\n"
            
            if pos_limited:
                explanation += "\nPrincipalement à cause de:\n"
                for i, (name, _) in enumerate(pos_limited):
                    explanation += f"- {name}\n"
                    
            if neg_limited:
                explanation += "\nMalgré:\n"
                for i, (name, _) in enumerate(neg_limited):
                    explanation += f"- {name}\n"
                    
            if not pos_limited and not neg_limited:
                explanation += "\nLe système n'a pas pu identifier de facteurs clairement déterminants pour cette prédiction."
                
        else:
            # Version anglaise (défaut)
            if is_classifier:
                result = f"The system has determined that the result is: {prediction}\n\n"
            else:
                result = f"The system has estimated the following value: {prediction}\n\n"
                
            explanation = "Here's why:\n"
            
            if pos_limited:
                explanation += "\nMainly because of:\n"
                for i, (name, _) in enumerate(pos_limited):
                    explanation += f"- {name}\n"
                    
            if neg_limited:
                explanation += "\nDespite:\n"
                for i, (name, _) in enumerate(neg_limited):
                    explanation += f"- {name}\n"
                    
            if not pos_limited and not neg_limited:
                explanation += "\nThe system could not identify clearly determining factors for this prediction."
                
        return result + explanation
