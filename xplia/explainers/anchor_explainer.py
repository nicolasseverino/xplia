"""
Explainer basé sur Anchor pour XPLIA
====================================

Ce module implémente l'explainer Anchor qui génère des règles d'ancrage
pour expliquer les prédictions d'un modèle.

Les règles d'ancrage sont des règles SI-ALORS qui suffisent à "ancrer" une prédiction,
c'est-à-dire que si les conditions de la règle sont satisfaites, la prédiction
reste presque certainement la même, indépendamment des autres caractéristiques.
"""

import logging
import numpy as np
import pandas as pd
import os
import hashlib
import json
import time
import traceback
import datetime
from functools import lru_cache
from contextlib import contextmanager
from typing import List, Dict, Any, Union, Optional, Tuple, Callable
from dataclasses import dataclass, field

from ..core.base import ExplainerBase, register_explainer
from ..core.explanation import (
    ExplanationResult,
    FeatureImportance,
    ModelMetadata,
    ExplainabilityMethod,
    AudienceLevel
)

@dataclass
class AnchorExplainerConfig:
    """Configuration pour l'AnchorExplainer permettant un paramétrage complet."""
    # Paramètres généraux
    feature_names: Optional[List[str]] = None
    categorical_features: Optional[List[int]] = None
    class_names: Optional[List[str]] = None
    discretize_continuous: bool = True
    
    # Paramètres d'optimisation
    use_cache: bool = True
    cache_size: int = 128
    use_gpu: bool = True
    compute_quality_metrics: bool = True
    
    # Paramètres d'ancrage
    threshold: float = 0.95
    tau: float = 0.15
    batch_size: int = 100
    coverage_samples: int = 10000
    beam_size: int = 5
    max_anchor_size: Optional[int] = None
    stop_on_first: bool = False
    binary_cache_size: int = 10000
    cache_margin: int = 1000
    
    # Paramètres de narrative
    narrative_audiences: List[str] = field(default_factory=lambda: ["technical"])
    supported_languages: List[str] = field(default_factory=lambda: ["en", "fr"])
    
    # Paramètres de conformité
    verify_compliance: bool = True
    compliance_rules: Dict[str, Any] = field(default_factory=dict)
    
    # Paramètres d'échantillonnage pour les explications de groupes
    sampling_method: str = "random"  # 'random', 'kmeans', 'stratified'
    max_instances: int = 10

@register_explainer
class AnchorExplainer(ExplainerBase):
    """
    Explainer qui génère des règles d'ancrage pour expliquer les prédictions d'un modèle.
    
    Les règles d'ancrage sont des règles SI-ALORS qui suffisent à "ancrer" une prédiction,
    c'est-à-dire que si les conditions de la règle sont satisfaites, la prédiction
    reste presque certainement la même, indépendamment des autres caractéristiques.
    
    Cette implémentation utilise la bibliothèque Anchor originale.
    
    Attributes:
        _model: Modèle à expliquer
        _feature_names: Noms des caractéristiques
        _categorical_features: Indices des caractéristiques catégorielles
        _class_names: Noms des classes (pour les modèles de classification)
        _explainer: Instance de l'explainer Anchor
        _metadata: Métadonnées du modèle
        _discretizer: Discrétiseur pour les caractéristiques continues
        _logger: Logger pour les messages de débogage
    """
    
    def __init__(self, model, config=None, **kwargs):
        """
        Initialise l'explainer Anchor avec des fonctionnalités avancées.
        
        Args:
            model: Modèle à expliquer
            config: Configuration via AnchorExplainerConfig
            **kwargs: Arguments additionnels pour configuration
                feature_names: Liste des noms des caractéristiques
                categorical_features: Liste des indices des caractéristiques catégorielles
                class_names: Liste des noms des classes
                discretize_continuous: Si True, discrétise les caractéristiques continues
                use_cache: Utiliser le cache pour les explications répétées
                cache_size: Taille du cache LRU
                use_gpu: Utiliser le GPU si disponible
                compute_quality_metrics: Calculer les métriques de qualité
                verify_compliance: Vérifier la conformité réglementaire
                narrative_audiences: Audiences pour les narratives
                supported_languages: Langues supportées
        """
        super().__init__(model=model, **kwargs)
        
        # Initialiser la configuration
        if config is None:
            self._config = AnchorExplainerConfig()
        elif isinstance(config, dict):
            self._config = AnchorExplainerConfig(**config)
        else:
            self._config = config
            
        # Surcharger la configuration avec les paramètres spécifiques
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
        
        # Initialiser les attributs de base
        self._feature_names = self._config.feature_names
        self._categorical_features = self._config.categorical_features or []
        self._class_names = self._config.class_names
        self._discretize_continuous = self._config.discretize_continuous
        
        # Explainer Anchor
        self._explainer = None
        
        # Discrétiseur pour les caractéristiques continues
        self._discretizer = None
        
        # Métadonnées du modèle
        self._metadata = None
        
        # Logger
        self._logger = logging.getLogger(__name__)
        
        # Détecter le type de framework (TensorFlow, PyTorch, etc.)
        self._framework = self._extract_model_type()
        
        # Initialiser la gestion du GPU si demandée
        if self._config.use_gpu:
            self._initialize_gpu()
            
        # Initialiser le vérificateur de conformité réglementaire
        if self._config.verify_compliance:
            self._initialize_compliance_checker()
        else:
            self._compliance_checker = None
            
        # Tracer l'initialisation
        self._logger.info(f"AnchorExplainer initialisé avec le framework '{self._framework}'")
        if self._compliance_checker:
            self._logger.info("Vérificateur de conformité initialisé")
        if self._config.use_gpu:
            self._logger.info("Support GPU activé")
            
    def _initialize_gpu(self):
        """
        Initialise la configuration GPU pour le framework détecté.
        Cette méthode configure le GPU pour TensorFlow/PyTorch si disponible.
        """
        try:
            if self._framework == 'tensorflow':
                # Configuration GPU pour TensorFlow
                try:
                    import tensorflow as tf
                    gpus = tf.config.list_physical_devices('GPU')
                    if gpus:
                        # Configuration de la croissance dynamique de la mémoire
                        self._set_gpu_memory_growth(gpus)
                        self._logger.info(f"GPU TensorFlow configuré: {len(gpus)} GPU(s) disponible(s)")
                    else:
                        self._logger.info("Aucun GPU TensorFlow détecté, utilisation du CPU")
                except ImportError:
                    self._logger.warning("TensorFlow non disponible pour la configuration GPU")
                        
            elif self._framework == 'pytorch':
                # Configuration GPU pour PyTorch
                try:
                    import torch
                    if torch.cuda.is_available():
                        # Réserver la mémoire au fur et à mesure
                        torch.cuda.set_per_process_memory_fraction(0.9)  # Réserver 90% de la mémoire GPU
                        torch.cuda.empty_cache()
                        self._logger.info(f"GPU PyTorch configuré: {torch.cuda.device_count()} GPU(s) disponible(s)")
                    else:
                        self._logger.info("Aucun GPU PyTorch détecté, utilisation du CPU")
                except ImportError:
                    self._logger.warning("PyTorch non disponible pour la configuration GPU")
        except Exception as e:
            self._logger.warning(f"Erreur lors de l'initialisation GPU: {str(e)}")
            
    def _set_gpu_memory_growth(self, gpus):
        """
        Configure la croissance dynamique de la mémoire GPU pour TensorFlow
        afin d'éviter les erreurs OOM (Out of Memory).
        
        Args:
            gpus: Liste des GPU physiques disponibles (tf.config.list_physical_devices)
        """
        try:
            import tensorflow as tf
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            self._logger.warning(f"Erreur lors de la configuration de la croissance mémoire GPU: {str(e)}")
            
    @contextmanager
    def _maybe_use_gpu_context(self, use_gpu=None):
        """
        Context manager pour gérer l'utilisation du GPU selon le framework.
        
        Args:
            use_gpu: Si True, utilise le GPU si disponible. Si None, utilise la valeur de configuration.
            
        Yields:
            Contexte GPU approprié pour le framework
        """
        # Déterminer si on utilise le GPU
        if use_gpu is None:
            use_gpu = self._config.use_gpu
            
        if not use_gpu:
            # Si GPU non demandé, simplement passer
            yield
            return
            
        # Sauvegarder le device/contexte actuel
        original_device = None
        device_changed = False
            
        try:
            # Utiliser le GPU selon le framework
            if self._framework == 'tensorflow':
                try:
                    import tensorflow as tf
                    # Vérifier si GPU est disponible
                    gpus = tf.config.list_physical_devices('GPU')
                    if gpus:
                        # Utiliser le premier GPU
                        with tf.device('/GPU:0'):
                            yield
                    else:
                        # Pas de GPU disponible
                        yield
                except ImportError:
                    # TensorFlow non disponible
                    yield
                        
            elif self._framework == 'pytorch':
                try:
                    import torch
                    if torch.cuda.is_available():
                        # Sauvegarder le device actuel
                        original_device = torch.cuda.current_device()
                        # Utiliser le premier GPU
                        torch.cuda.set_device(0)
                        device_changed = True
                        yield
                    else:
                        # Pas de GPU disponible
                        yield
                except ImportError:
                    # PyTorch non disponible
                    yield
            else:
                # Autre framework non supporté pour GPU
                yield
        finally:
            # Restaurer le device original si nécessaire
            if device_changed and original_device is not None:
                try:
                    import torch
                    torch.cuda.set_device(original_device)
                    torch.cuda.empty_cache()  # Libérer la mémoire GPU
                except:
                    pass
                    
    def _initialize_compliance_checker(self):
        """
        Initialise le vérificateur de conformité réglementaire.
        """
        try:
            from ..compliance.compliance_checker import ComplianceChecker
            self._compliance_checker = ComplianceChecker(
                rules=self._config.compliance_rules,
                explainer_type="AnchorExplainer"
            )
            self._logger.info("Vérificateur de conformité initialisé avec succès")
        except ImportError:
            self._logger.warning("Module de conformité non disponible")
            self._compliance_checker = None
        except Exception as e:
            self._logger.warning(f"Impossible d'initialiser le vérificateur de conformité: {str(e)}")
            self._compliance_checker = None
        
    def explain_instance(self, instance, **kwargs) -> ExplanationResult:
        """
        Génère une explication basée sur des règles d'ancrage pour une instance spécifique.
        Cette version améliorée inclut le cache, la gestion GPU, les métriques de qualité,
        les narratives multilingues et la vérification de conformité réglementaire.
        
        Args:
            instance: Instance à expliquer (array, liste, dict ou pandas.Series)
            **kwargs: Paramètres additionnels
                data_type: Type de données ('tabular', 'text', 'image')
                threshold: Seuil de précision pour l'ancrage (défaut: selon config)
                tau: Paramètre de relaxation (défaut: selon config)
                beam_size: Taille du faisceau pour la recherche (défaut: selon config)
                max_anchor_size: Taille maximum de l'ancrage (défaut: selon config)
                stop_on_first: Arrêter à la première explication valide (défaut: selon config)
                binary_cache_size: Taille du cache binaire (défaut: selon config)
                cache_margin: Marge du cache (défaut: selon config)
                feature_names: Noms des caractéristiques (défaut: ceux de l'initialisation)
                audience_level: Niveau d'audience (TECHNICAL, BUSINESS, PUBLIC, ALL)
                language: Code de langue ('en', 'fr', etc.) (défaut: 'en')
                use_gpu: Utiliser le GPU si disponible (défaut: selon config)
                use_cache: Utiliser le cache pour les explications répétées (défaut: selon config)
                compute_quality_metrics: Calculer les métriques de qualité (défaut: selon config)
                verify_compliance: Vérifier la conformité réglementaire (défaut: selon config)
                
        Returns:
            ExplanationResult: Résultat standardisé de l'explication
        """
        # Paramètres de base
        audience_level = kwargs.get('audience_level', AudienceLevel.TECHNICAL)
        data_type = kwargs.get('data_type', 'tabular')
        language = kwargs.get('language', 'en')
        use_cache = kwargs.get('use_cache', self._config.use_cache)
        
        # Initialiser la mesure du temps d'exécution
        start_time = time.time()
        execution_metadata = {
            'start_time': datetime.datetime.now().isoformat(),
            'data_type': data_type,
            'audience_level': str(audience_level),
            'language': language,
            'from_cache': False,
            'framework': getattr(self, '_framework', 'unknown')
        }
        
        try:
            # Audit de l'action
            self.add_audit_record("explain_instance", {
                "data_type": data_type,
                "audience_level": audience_level.value if isinstance(audience_level, AudienceLevel) else audience_level,
                "language": language,
                "use_cache": use_cache,
                "use_gpu": kwargs.get('use_gpu', self._config.use_gpu)
            })
            
            # 1. Vérifier si une explication en cache est disponible
            explanation_data = None
            if use_cache:
                cache_key = self._get_cache_key(instance, **kwargs)
                if cache_key:
                    self._logger.debug(f"Recherche dans le cache avec la clé: {cache_key[:10]}...")
                    # Récupérer du cache ou calculer si nécessaire
                    explanation_data = self._get_cached_explanation(cache_key, instance, **kwargs)
                    if explanation_data and explanation_data.get('metadata', {}).get('from_cache', False):
                        self._logger.info(f"Explication trouvée dans le cache: {cache_key[:10]}...")
                        execution_metadata['from_cache'] = True
            
            # 2. Si pas de cache ou non trouvée, calculer l'explication
            if not use_cache or not explanation_data or 'error' in explanation_data:
                explanation_data = self._compute_explanation_cached(instance, **kwargs)
            
            # 3. Convertir le résultat au format ExplanationResult
            if not explanation_data:
                raise ValueError("Échec du calcul de l'explication, résultat vide")
            
            if 'error' in explanation_data:
                # Gérer les erreurs
                error_msg = explanation_data.get('error', "Erreur inconnue lors de l'explication")
                self._logger.error(f"Erreur lors de l'explication: {error_msg}")
                raise ValueError(error_msg)
            
            # 4. Extraire les informations pour créer l'ExplanationResult
            # Recréer les objets FeatureImportance à partir des dictionnaires si nécessaire
            feature_importances_data = explanation_data.get('feature_importances', [])
            feature_importances = []
            
            for fi_data in feature_importances_data:
                if isinstance(fi_data, dict):
                    feature_importances.append(FeatureImportance(
                        feature_name=fi_data.get('feature_name', ""),
                        importance=fi_data.get('importance', 0.0),
                        local_value=fi_data.get('local_value')
                    ))
                else:
                    feature_importances.append(fi_data)  # Déjà un objet FeatureImportance
            
            # Récupérer les règles d'ancrage et autres informations
            anchor_rules = explanation_data.get('anchor_rules', [])
            prediction = explanation_data.get('prediction')
            narratives = explanation_data.get('narratives', {})
            
            # Extraire la narrative appropriée selon le niveau d'audience et la langue
            explanation_narrative = None
            if language in narratives:
                lang_narratives = narratives[language]
                audience_key = audience_level.lower() if isinstance(audience_level, AudienceLevel) else str(audience_level).lower()
                
                if audience_key in lang_narratives:
                    explanation_narrative = lang_narratives[audience_key]
                elif 'technical' in lang_narratives:  # Fallback sur technical
                    explanation_narrative = lang_narratives['technical']
            
            # Préparer les métadonnées enrichies
            metadata = {
                'data_type': data_type,
                'audience_level': str(audience_level),
                'language': language,
                'execution_time_ms': int((time.time() - start_time) * 1000)
            }
            
            # Ajouter les métriques d'ancrage aux métadonnées
            if 'anchor_metrics' in explanation_data:
                metadata.update(explanation_data['anchor_metrics'])
            
            # Ajouter les métriques de qualité si disponibles
            if 'quality_metrics' in explanation_data:
                metadata['quality'] = explanation_data['quality_metrics']
                
            # Ajouter les résultats de conformité si disponibles
            if 'compliance' in explanation_data:
                metadata['compliance'] = explanation_data['compliance']
                
            # Ajouter les métadonnées d'exécution
            metadata.update(execution_metadata)
            
            # 5. Créer et retourner l'ExplanationResult final
            result = ExplanationResult(
                method=ExplainabilityMethod.ANCHOR,
                feature_importances=feature_importances,
                anchor_rules=anchor_rules,
                metadata=metadata,
                audience_level=audience_level,
                explanation_narrative=explanation_narrative
            )
            
            return result
            
        except Exception as e:
            # Enregistrer l'erreur
            self._logger.error(f"Erreur lors de l'explication Anchor: {str(e)}")
            self._logger.debug(traceback.format_exc())
            
            # Mettre à jour les métadonnées d'exécution avec l'erreur
            execution_metadata['error'] = str(e)
            execution_metadata['traceback'] = traceback.format_exc()
            execution_metadata['execution_time_ms'] = int((time.time() - start_time) * 1000)
            
            # Retourner un résultat d'erreur
            return ExplanationResult(
                method=ExplainabilityMethod.ANCHOR,
                feature_importances=[],
                metadata=execution_metadata,
                audience_level=audience_level
            )
    
    def explain(self, X, y=None, **kwargs) -> ExplanationResult:
        """
        Génère des explications basées sur des règles d'ancrage pour un ensemble de données.
        Pour les explications par ancrage, cette méthode sélectionne un échantillon
        représentatif et génère des règles d'ancrage pour chaque instance.
        
        Args:
            X: Données d'entrée à expliquer
            y: Valeurs cibles réelles (optionnel)
            **kwargs: Paramètres additionnels
                max_instances: Nombre maximum d'instances à expliquer
                sampling_strategy: Stratégie d'échantillonnage ('random', 'stratified', 'kmeans')
                data_type: Type de données ('tabular', 'text', 'image')
                
        Returns:
            ExplanationResult: Résultat standardisé de l'explication
        """
        # Paramètres
        audience_level = kwargs.get('audience_level', AudienceLevel.TECHNICAL)
        max_instances = kwargs.get('max_instances', 5)
        sampling_strategy = kwargs.get('sampling_strategy', 'random')
        data_type = kwargs.get('data_type', 'tabular')
        
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
            "sampling_strategy": sampling_strategy,
            "data_type": data_type
        })
        
        try:
            # Générer des règles d'ancrage pour chaque instance échantillonnée
            all_anchor_rules = []
            all_anchor_metrics = []
            all_feature_importances = []
            
            for instance in sampled_instances:
                # Utiliser explain_instance pour chaque instance
                instance_result = self.explain_instance(
                    instance, 
                    feature_names=feature_names,
                    audience_level=audience_level,
                    data_type=data_type,
                    **kwargs
                )
                
                # Collecter les résultats
                all_anchor_rules.append(instance_result.raw_explanation["anchor_rules"])
                all_anchor_metrics.append(instance_result.raw_explanation["anchor_metrics"])
                all_feature_importances.append(instance_result.feature_importances)
            
            # Agréger les importances de caractéristiques
            aggregated_importances = self._aggregate_feature_importances(all_feature_importances, feature_names)
            
            # Extraire les métadonnées du modèle
            if not self._metadata:
                self._extract_metadata()
            
            # Créer le résultat d'explication
            result = ExplanationResult(
                method=ExplainabilityMethod.ANCHOR,
                model_metadata=self._metadata,
                feature_importances=aggregated_importances,
                raw_explanation={
                    "sampled_instances": sampled_instances.tolist(),
                    "sampled_indices": sampled_indices.tolist(),
                    "all_anchor_rules": all_anchor_rules,
                    "all_anchor_metrics": all_anchor_metrics,
                    "feature_names": feature_names,
                    "data_type": data_type
                },
                audience_level=audience_level
            )
            
            return result
            
        except Exception as e:
            self._logger.error(f"Erreur lors de la génération des règles d'ancrage: {str(e)}")
            raise RuntimeError(f"Échec de l'explication par ancrage: {str(e)}")
    
    def _initialize_explainer(self, X, data_type='tabular'):
        """
        Initialise l'explainer Anchor selon le type de données.
        
        Args:
            X: Données d'entrée
            data_type: Type de données ('tabular', 'text', 'image')
        """
        try:
            import anchor
            from anchor import anchor_tabular, anchor_text, anchor_image
        except ImportError:
            raise ImportError("Le package Anchor est requis pour cette méthode. "
                            "Installez-le avec 'pip install anchor-exp'.")
        
        if data_type == 'tabular':
            # Initialiser le discrétiseur si nécessaire
            if self._discretize_continuous and self._discretizer is None:
                self._discretizer = anchor_tabular.AnchorTabularExplainer.discretizer(
                    X, 
                    self._categorical_features, 
                    self._feature_names
                )
            
            # Créer l'explainer pour données tabulaires
            self._explainer = anchor_tabular.AnchorTabularExplainer(
                class_names=self._class_names,
                feature_names=self._feature_names,
                train_data=X,
                categorical_names={i: [] for i in self._categorical_features},
                discretizer=self._discretizer
            )
            
        elif data_type == 'text':
            # Créer l'explainer pour données textuelles
            self._explainer = anchor_text.AnchorText(
                nlp=None,  # Utiliser spaCy par défaut
                class_names=self._class_names
            )
            
        elif data_type == 'image':
            # Créer l'explainer pour images
            self._explainer = anchor_image.AnchorImage(
                class_names=self._class_names
            )
            
        else:
            raise ValueError(f"Type de données non supporté: {data_type}")
        
        # Définir la fonction de prédiction
        self._explainer.predictor = self._get_prediction_function()
    
    def _get_prediction_function(self):
        """
        Crée une fonction de prédiction pour l'explainer Anchor.
        
        Returns:
            function: Fonction de prédiction
        """
        # Déterminer le type de modèle
        model_type = self._get_model_type()
        
        # Créer une fonction de prédiction appropriée
        def predict_fn(instances):
            # Pour les modèles scikit-learn et similaires
            if model_type in ['sklearn', 'xgboost', 'lightgbm', 'catboost']:
                if hasattr(self._model, 'predict'):
                    return self._model.predict(instances)
                else:
                    raise ValueError("Le modèle ne possède pas de méthode 'predict'.")
            
            # Pour TensorFlow/Keras
            elif model_type == 'tensorflow':
                import tensorflow as tf
                # Convertir en tensor si nécessaire
                if not isinstance(instances, tf.Tensor):
                    instances = tf.convert_to_tensor(instances, dtype=tf.float32)
                # Faire la prédiction
                preds = self._model(instances)
                # Convertir en classes si nécessaire
                if len(preds.shape) > 1 and preds.shape[1] > 1:
                    return tf.argmax(preds, axis=1).numpy()
                else:
                    return tf.round(preds).numpy()
            
            # Pour PyTorch
            elif model_type == 'pytorch':
                import torch
                # Mettre le modèle en mode évaluation
                self._model.eval()
                # Convertir en tensor si nécessaire
                if not isinstance(instances, torch.Tensor):
                    instances = torch.tensor(instances, dtype=torch.float32)
                # Désactiver le calcul de gradient
                with torch.no_grad():
                    # Faire la prédiction
                    preds = self._model(instances)
                    # Convertir en classes
                    if preds.shape[1] > 1:
                        return torch.argmax(preds, dim=1).numpy()
                    else:
                        return torch.round(preds).numpy()
            
            # Pour les autres types de modèles
            else:
                # Essayer d'appeler predict
                if hasattr(self._model, 'predict'):
                    return self._model.predict(instances)
                else:
                    # Essayer d'appeler le modèle directement
                    try:
                        return self._model(instances)
                    except:
                        raise ValueError("Impossible de faire des prédictions avec le modèle fourni.")
        
        return predict_fn
    
    def _explain_tabular_instance(self, instance, **kwargs):
        """
        Génère une explication pour une instance tabulaire.
        
        Args:
            instance: Instance à expliquer
            **kwargs: Paramètres pour l'explainer Anchor
            
        Returns:
            object: Explication générée par Anchor
        """
        return self._explainer.explain_instance(
            instance, 
            threshold=kwargs.get('threshold', 0.95),
            tau=kwargs.get('tau', 0.15),
            batch_size=kwargs.get('batch_size', 100),
            coverage_samples=kwargs.get('coverage_samples', 10000),
            beam_size=kwargs.get('beam_size', 5),
            max_anchor_size=kwargs.get('max_anchor_size'),
            stop_on_first=kwargs.get('stop_on_first', False),
            binary_cache_size=kwargs.get('binary_cache_size', 10000),
            cache_margin=kwargs.get('cache_margin', 1000),
            verbose=kwargs.get('verbose', False),
            verbose_every=kwargs.get('verbose_every', 1)
        )
    
    def _explain_text_instance(self, text, **kwargs):
        """
        Génère une explication pour une instance textuelle.
        
        Args:
            text: Texte à expliquer
            **kwargs: Paramètres pour l'explainer Anchor
            
        Returns:
            object: Explication générée par Anchor
        """
        return self._explainer.explain_instance(
            text, 
            threshold=kwargs.get('threshold', 0.95),
            use_unk=kwargs.get('use_unk', True),
            use_proba=kwargs.get('use_proba', False),
            sample_proba=kwargs.get('sample_proba', 0.5),
            top_n=kwargs.get('top_n', 100),
            temperature=kwargs.get('temperature', 1.0),
            beam_size=kwargs.get('beam_size', 5)
        )
    
    def _explain_image_instance(self, image, **kwargs):
        """
        Génère une explication pour une image.
        
        Args:
            image: Image à expliquer
            **kwargs: Paramètres pour l'explainer Anchor
            
        Returns:
            object: Explication générée par Anchor
        """
        segmentation_fn = kwargs.get('segmentation_fn')
        if segmentation_fn is None:
            # Utiliser une segmentation par défaut si non fournie
            try:
                from skimage.segmentation import felzenszwalb
                segmentation_fn = lambda x: felzenszwalb(x, scale=100, sigma=0.5, min_size=50)
            except ImportError:
                raise ImportError("scikit-image est requis pour la segmentation d'image. "
                                "Installez-le avec 'pip install scikit-image'.")
        
        return self._explainer.explain_instance(
            image, 
            segmentation_fn,
            threshold=kwargs.get('threshold', 0.95),
            p_sample=kwargs.get('p_sample', 0.5),
            **{k: v for k, v in kwargs.items() if k not in ['threshold', 'p_sample', 'segmentation_fn']}
        )
    
    def _extract_anchor_rules(self, explanation, feature_names):
        """
        Extrait les règles d'ancrage à partir de l'explication.
        
        Args:
            explanation: Explication générée par Anchor
            feature_names: Noms des caractéristiques
            
        Returns:
            list: Liste des règles d'ancrage
        """
        rules = []
        
        # Pour les explications tabulaires
        if hasattr(explanation, 'names'):
            for rule in explanation.names():
                rules.append(rule)
        # Pour les explications textuelles
        elif hasattr(explanation, 'exp_map'):
            for pos in explanation.exp_map['names']:
                rules.append(pos)
        # Pour les explications d'images
        elif hasattr(explanation, 'segments'):
            rules = [f"segment_{i}" for i in explanation.segments()]
        # Fallback
        else:
            try:
                # Essayer d'extraire les règles directement
                rules = explanation.names()
            except:
                self._logger.warning("Impossible d'extraire les règles d'ancrage. Format d'explication non reconnu.")
                rules = []
        
        return rules
    
    def _extract_anchor_metrics(self, explanation):
        """
        Extrait les métriques de l'explication.
        
        Args:
            explanation: Explication générée par Anchor
            
        Returns:
            dict: Métriques de l'explication
        """
        metrics = {}
        
        # Métriques communes
        if hasattr(explanation, 'precision'):
            metrics['precision'] = float(explanation.precision)
        if hasattr(explanation, 'coverage'):
            metrics['coverage'] = float(explanation.coverage)
            
        # Métriques spécifiques aux données tabulaires
        if hasattr(explanation, 'anchor'):
            metrics['anchor_size'] = len(explanation.anchor)
            
        # Métriques spécifiques aux textes
        if hasattr(explanation, 'exp_map'):
            if 'precision' in explanation.exp_map:
                metrics['precision'] = float(explanation.exp_map['precision'])
            if 'coverage' in explanation.exp_map:
                metrics['coverage'] = float(explanation.exp_map['coverage'])
                
        # Métriques spécifiques aux images
        if hasattr(explanation, 'segments'):
            metrics['num_segments'] = len(explanation.segments())
        
        return metrics
    
    def _convert_rules_to_importances(self, anchor_rules, feature_names):
        """
        Convertit les règles d'ancrage en importances de caractéristiques.
        
        Args:
            anchor_rules: Règles d'ancrage
            feature_names: Noms des caractéristiques
            
        Returns:
            list: Liste d'objets FeatureImportance
        """
        importances = []
        feature_counts = {}
        
        # Compter les occurrences des caractéristiques dans les règles
        for rule in anchor_rules:
            # Extraire les noms des caractéristiques de la règle
            for feature in feature_names:
                if feature in rule:
                    feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        # Convertir les comptages en importances
        total_counts = sum(feature_counts.values()) if feature_counts else 1
        for feature, count in feature_counts.items():
            importances.append(FeatureImportance(
                feature_name=feature,
                importance=count / total_counts,
                std=0.0,  # Pas d'écart-type pour les règles d'ancrage
                additional_info={
                    'occurrence_count': count,
                    'total_rules': len(anchor_rules)
                }
            ))
        
        # Ajouter les caractéristiques non utilisées avec importance nulle
        for feature in feature_names:
            if feature not in feature_counts:
                importances.append(FeatureImportance(
                    feature_name=feature,
                    importance=0.0,
                    std=0.0,
                    additional_info={
                        'occurrence_count': 0,
                        'total_rules': len(anchor_rules)
                    }
                ))
        
        # Trier par importance décroissante
        importances.sort(key=lambda x: x.importance, reverse=True)
        
        return importances
    
    def _aggregate_feature_importances(self, all_importances, feature_names):
        """
        Agrège les importances de caractéristiques de plusieurs instances.
        
        Args:
            all_importances: Liste de listes d'objets FeatureImportance
            feature_names: Noms des caractéristiques
            
        Returns:
            list: Liste agrégée d'objets FeatureImportance
        """
        # Initialiser un dictionnaire pour stocker les importances agrégées
        aggregated = {feature: {'sum': 0.0, 'count': 0, 'values': []} for feature in feature_names}
        
        # Collecter toutes les importances par caractéristique
        for importances in all_importances:
            for imp in importances:
                feature = imp.feature_name
                if feature in aggregated:
                    aggregated[feature]['sum'] += imp.importance
                    aggregated[feature]['count'] += 1
                    aggregated[feature]['values'].append(imp.importance)
        
        # Calculer les importances moyennes et les écarts-types
        result = []
        for feature, data in aggregated.items():
            if data['count'] > 0:
                mean_importance = data['sum'] / data['count']
                std_importance = np.std(data['values']) if len(data['values']) > 1 else 0.0
                
                result.append(FeatureImportance(
                    feature_name=feature,
                    importance=mean_importance,
                    std=float(std_importance),
                    additional_info={
                        'count': data['count'],
                        'min': float(min(data['values'])) if data['values'] else 0.0,
                        'max': float(max(data['values'])) if data['values'] else 0.0
                    }
                ))
            else:
                # Caractéristique non utilisée
                result.append(FeatureImportance(
                    feature_name=feature,
                    importance=0.0,
                    std=0.0,
                    additional_info={
                        'count': 0,
                        'min': 0.0,
                        'max': 0.0
                    }
                ))
        
        # Trier par importance décroissante
        result.sort(key=lambda x: x.importance, reverse=True)
        
        return result
    
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
            target_names=self._class_names,
            model_parameters={
                'categorical_features': self._categorical_features,
                'discretize_continuous': self._discretize_continuous
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
            
    def _get_cache_key(self, instance, **kwargs):
        """
        Génère une clé de cache unique pour une instance et des paramètres d'explication.
        
        Args:
            instance: Instance à expliquer
            **kwargs: Paramètres d'explication
            
        Returns:
            str: Clé de cache unique ou None si impossible de générer
        """
        try:
            # Convertir l'instance en structure serializable
            if isinstance(instance, pd.DataFrame):
                instance_data = instance.to_dict('records')
            elif isinstance(instance, pd.Series):
                instance_data = instance.to_dict()
            elif isinstance(instance, dict):
                instance_data = instance
            elif isinstance(instance, (list, np.ndarray)):
                instance_data = instance
                # Convertir les ndarray en listes
                if isinstance(instance_data, np.ndarray):
                    instance_data = instance_data.tolist()
            else:
                self._logger.warning(f"Type d'instance non supporté pour le cache: {type(instance)}")
                return None
                
            # Extraire les paramètres pertinents pour la clé de cache
            cache_params = {
                'data_type': kwargs.get('data_type', 'tabular'),
                'threshold': kwargs.get('threshold', self._config.threshold),
                'tau': kwargs.get('tau', self._config.tau),
                'max_anchor_size': kwargs.get('max_anchor_size', self._config.max_anchor_size),
                'beam_size': kwargs.get('beam_size', self._config.beam_size),
            }
            
            # Créer une chaîne unique pour l'instance et les paramètres
            cache_data = {
                'instance': instance_data,
                'params': cache_params
            }
            
            # Générer un hash SHA-256 comme clé de cache
            cache_key = hashlib.sha256(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()
            
            return cache_key
            
        except Exception as e:
            self._logger.warning(f"Erreur lors de la génération de la clé de cache: {str(e)}")
            return None
            
    @lru_cache(maxsize=128)
    def _compute_explanation(self, instance_hash, instance_serialized, **kwargs):
        """
        Calcule l'explication pour une instance avec gestion du cache LRU.
        Cette méthode est décorée avec lru_cache pour mémoriser les résultats.
        
        Args:
            instance_hash: Hash de l'instance (pour le cache)
            instance_serialized: Représentation sérialisée de l'instance
            **kwargs: Paramètres d'explication
            
        Returns:
            dict: Résultat d'explication
        """
        # Désérialiser l'instance
        instance = json.loads(instance_serialized)
        
        # Convertir en format approprié selon le type de données
        data_type = kwargs.get('data_type', 'tabular')
        
        if data_type == 'tabular':
            # Reconvertir en array numpy
            instance_array = np.array(instance)
        elif data_type == 'text':
            # String pour texte
            instance_array = instance
        elif data_type == 'image':
            # Reconvertir en array numpy pour image
            instance_array = np.array(instance)
        else:
            raise ValueError(f"Type de données non supporté: {data_type}")
            
        # Calculer l'explication via la méthode adéquate selon le type de données
        # Déléguer à la méthode complète
        return self._compute_explanation_cached(instance_array, **kwargs)
        
    def _get_cached_explanation(self, cache_key, instance, **kwargs):
        """
        Récupère une explication du cache ou la calcule si nécessaire.
        
        Args:
            cache_key: Clé de cache
            instance: Instance à expliquer
            **kwargs: Paramètres d'explication
            
        Returns:
            dict: Résultat d'explication
        """
        try:
            # Sérialiser l'instance pour le cache
            if isinstance(instance, pd.DataFrame):
                serialized_instance = json.dumps(instance.to_dict('records')[0])
            elif isinstance(instance, pd.Series):
                serialized_instance = json.dumps(instance.to_dict())
            elif isinstance(instance, dict):
                serialized_instance = json.dumps(instance)
            elif isinstance(instance, (list, np.ndarray)):
                if isinstance(instance, np.ndarray):
                    serialized_instance = json.dumps(instance.tolist())
                else:
                    serialized_instance = json.dumps(instance)
            else:
                raise ValueError(f"Type d'instance non supporté pour le cache: {type(instance)}")
                
            # Appeler la méthode cachée
            explanation = self._compute_explanation(cache_key, serialized_instance, **kwargs)
            
            # Marquer comme provenant du cache
            if explanation and 'metadata' in explanation:
                explanation['metadata']['from_cache'] = True
                
            return explanation
            
        except Exception as e:
            self._logger.warning(f"Erreur lors de la récupération du cache: {str(e)}")
            # En cas d'erreur, calculer directement
            return self._compute_explanation_cached(instance, **kwargs)
            
    def _compute_explanation_quality_metrics(self, anchor_rules, explanation=None, instance=None, prediction=None):
        """
        Calcule les métriques de qualité de l'explication par ancrage.
        
        Args:
            anchor_rules: Règles d'ancrage générées
            explanation: Objet d'explication original (optionnel)
            instance: Instance expliquée (optionnel)
            prediction: Prédiction du modèle (optionnel)
            
        Returns:
            dict: Métriques de qualité de l'explication
        """
        metrics = {}
        
        try:
            # 1. Précision de l'ancrage (directement depuis l'objet explanation)
            if explanation and hasattr(explanation, 'precision'):
                metrics['precision'] = float(explanation.precision)
            else:
                metrics['precision'] = None
                
            # 2. Couverture de l'ancrage (directement depuis l'objet explanation)
            if explanation and hasattr(explanation, 'coverage'):
                metrics['coverage'] = float(explanation.coverage)
            else:
                metrics['coverage'] = None
                
            # 3. Nombre de conditions dans la règle d'ancrage
            metrics['rule_length'] = len(anchor_rules) if anchor_rules else 0
            
            # 4. Score de complexité de la règle (bas = simple, haut = complexe)
            # Normalisé par le nombre de caractéristiques disponibles
            n_features = len(self._feature_names) if self._feature_names else 1
            metrics['complexity_score'] = metrics['rule_length'] / n_features if n_features > 0 else 0
            
            # 5. Stabilité de l'explication (indice de Jaccard moyen pour des petites perturbations)
            # Pour Anchor, nous utilisons la métrique de stabilité approximative basée sur la précision
            metrics['stability_score'] = float(metrics['precision']) if metrics['precision'] is not None else None
            
            # 6. Score global de qualité (moyenne pondérée des scores individuels)
            # Combiner précision, couverture et inverse de la complexité
            quality_components = []
            weights = [0.5, 0.3, 0.2]  # Pondération: précision, couverture, simplicité
            
            if metrics['precision'] is not None:
                quality_components.append(metrics['precision'] * weights[0])
            if metrics['coverage'] is not None:
                quality_components.append(metrics['coverage'] * weights[1])
            if metrics['complexity_score'] is not None:
                # Inverser la complexité (plus simple = meilleur score)
                simplicity = 1.0 - min(1.0, metrics['complexity_score'])
                quality_components.append(simplicity * weights[2])
                
            if quality_components:
                # Normaliser par la somme des poids utilisés
                used_weights = weights[:len(quality_components)]
                metrics['quality_score'] = sum(quality_components) / sum(used_weights)
            else:
                metrics['quality_score'] = None
                
        except Exception as e:
            self._logger.warning(f"Erreur lors du calcul des métriques de qualité: {str(e)}")
            self._logger.debug(traceback.format_exc())
            
        return metrics
        
    def _get_cache_key(self, instance, **kwargs):
        """
        Génère une clé de cache unique pour l'instance et les paramètres.
        
        Args:
            instance: Instance à expliquer
            **kwargs: Paramètres d'explication
            
        Returns:
            str: Clé de cache unique (hash MD5)
        """
        try:
            # Extraire les paramètres qui affectent le résultat de l'explication
            cache_key_params = {
                'data_type': kwargs.get('data_type', 'tabular'),
                'threshold': kwargs.get('threshold', self._config.threshold),
                'tau': kwargs.get('tau', self._config.tau),
                'beam_size': kwargs.get('beam_size', self._config.beam_size),
                'max_anchor_size': kwargs.get('max_anchor_size', self._config.max_anchor_size),
                'stop_on_first': kwargs.get('stop_on_first', self._config.stop_on_first),
                'audience_level': str(kwargs.get('audience_level', AudienceLevel.TECHNICAL)),
                'language': kwargs.get('language', 'en'),
            }
            
            # Sérialiser l'instance selon son type
            instance_str = self._serialize_instance(instance)
                
            # Créer un dictionnaire complet pour le hachage
            hash_dict = {
                'instance': instance_str,
                'params': cache_key_params,
                'model_id': id(self._model),  # Ajouter l'ID du modèle pour éviter les collisions entre différents modèles
            }
            
            # Générer la clé de cache (hash MD5)
            hash_str = json.dumps(hash_dict, sort_keys=True)
            return hashlib.md5(hash_str.encode('utf-8')).hexdigest()
            
        except Exception as e:
            self._logger.warning(f"Erreur lors de la génération de la clé de cache: {str(e)}")
            return None
    
    def _serialize_instance(self, instance):
        """
        Sérialise une instance pour le hachage du cache.
        
        Args:
            instance: Instance à sérialiser
            
        Returns:
            str: Représentation sérialisée de l'instance
        """
        try:
            if isinstance(instance, np.ndarray):
                return instance.tolist()
            elif isinstance(instance, pd.DataFrame) or isinstance(instance, pd.Series):
                return instance.to_dict()
            elif isinstance(instance, (list, dict, str, int, float, bool)):
                return instance
            else:
                return str(instance)
        except Exception as e:
            self._logger.warning(f"Erreur lors de la sérialisation de l'instance: {str(e)}")
            return str(instance)
    
    def reset_cache(self):
        """
        Réinitialise le cache des explications.
        Utile lorsque le modèle est mis à jour ou lorsqu'on veut forcer
        le recalcul des explications.
        """
        try:
            self._logger.info("Réinitialisation du cache des explications")
            self._get_cached_explanation.cache_clear()
            return {"status": "success", "message": "Cache réinitialisé avec succès"}
        except Exception as e:
            self._logger.error(f"Erreur lors de la réinitialisation du cache: {str(e)}")
            return {"status": "error", "message": f"Erreur lors de la réinitialisation du cache: {str(e)}"}
            
    @lru_cache(maxsize=128)
    def _get_cached_explanation(self, cache_key, instance, **kwargs):
        """
        Récupère ou calcule une explication avec cache LRU.
        Cette méthode est décorée avec lru_cache pour mémoriser les explications précédemment calculées.
        
        Args:
            cache_key: Clé unique pour l'instance et les paramètres
            instance: Instance à expliquer
            **kwargs: Paramètres d'explication
            
        Returns:
            dict: Résultat d'explication
        """
        try:
            # Décorer cette méthode avec lru_cache permet de mettre automatiquement en cache
            # les résultats selon la clé fournie. Si la même clé est utilisée à nouveau,
            # le résultat en cache est retourné sans recalcul.
            
            # Calculer l'explication
            explanation = self._compute_explanation_cached(instance, **kwargs)
            
            # Marquer comme provenant du cache
            if explanation and 'metadata' in explanation:
                explanation['metadata']['from_cache'] = True
                
            return explanation
            
        except Exception as e:
            self._logger.error(f"Erreur lors de la récupération du cache: {str(e)}")
            return {"error": f"Erreur de cache: {str(e)}"}
    
    def _extract_model_type(self):
        """
        Extrait le type de modèle/framework utilisé (scikit-learn, TensorFlow, PyTorch, etc.)
        
        Returns:
            str: Type de framework détecté
        """
        module_name = self._model.__class__.__module__.split('.')[0].lower()
        
        if module_name in ('sklearn', 'lightgbm', 'xgboost'):
            return module_name
        elif module_name == 'keras' or 'tensorflow' in module_name:
            return 'tensorflow'
        elif 'torch' in module_name:
            return 'pytorch'
        else:
            return 'unknown'
            
    def _generate_explanation_narrative(self, anchor_rules, feature_importances, prediction=None, audience_level=AudienceLevel.TECHNICAL, language='en'):
        """
        Génère une narrative d'explication adaptée à l'audience et à la langue.
        
        Args:
            anchor_rules: Règles d'ancrage générées
            feature_importances: Importances des caractéristiques
            prediction: Prédiction du modèle (optionnel)
            audience_level: Niveau d'audience (technique, business, public)
            language: Code de langue ('en', 'fr', etc.)
            
        Returns:
            dict: Narratives d'explication par niveau d'audience et langue
        """
        narratives = {}
        
        try:
            # Définir les templates par niveau d'audience et langue
            templates = {
                'en': {
                    'technical': {
                        'intro': "The model's prediction is explained by the following anchor rule conditions:",
                        'rule': "The model predicts '{prediction}' when {conditions}.",
                        'precision': "This rule has {precision:.0%} precision, meaning the prediction is consistent {precision:.0%} of the time when these conditions are met.",
                        'coverage': "The rule covers {coverage:.0%} of instances similar to this one.",
                        'outro': "The model relies primarily on {top_features_str} for this prediction."
                    },
                    'business': {
                        'intro': "The prediction for this case is based on the following key factors:",
                        'rule': "When {conditions_simple}, our model predicts '{prediction}'.",
                        'reliability': "This explanation is reliable in {precision:.0%} of similar cases.",
                        'outro': "The most influential factors were: {top_features_simple}."
                    },
                    'public': {
                        'intro': "Here's why this decision was made:",
                        'rule': "The system predicted '{prediction}' because {conditions_very_simple}.",
                        'outro': "The main factors affecting this decision were: {top_features_very_simple}."
                    }
                },
                'fr': {
                    'technical': {
                        'intro': "La prédiction du modèle est expliquée par les conditions d'ancrage suivantes :",
                        'rule': "Le modèle prédit '{prediction}' lorsque {conditions}.",
                        'precision': "Cette règle a une précision de {precision:.0%}, ce qui signifie que la prédiction est cohérente dans {precision:.0%} des cas où ces conditions sont respectées.",
                        'coverage': "La règle couvre {coverage:.0%} des instances similaires à celle-ci.",
                        'outro': "Le modèle s'appuie principalement sur {top_features_str} pour cette prédiction."
                    },
                    'business': {
                        'intro': "La prédiction pour ce cas est basée sur les facteurs clés suivants :",
                        'rule': "Lorsque {conditions_simple}, notre modèle prédit '{prediction}'.",
                        'reliability': "Cette explication est fiable dans {precision:.0%} des cas similaires.",
                        'outro': "Les facteurs les plus influents étaient : {top_features_simple}."
                    },
                    'public': {
                        'intro': "Voici pourquoi cette décision a été prise :",
                        'rule': "Le système a prédit '{prediction}' parce que {conditions_very_simple}.",
                        'outro': "Les principaux facteurs affectant cette décision étaient : {top_features_very_simple}."
                    }
                }
            }
            
            # S'assurer que la langue est supportée, sinon utiliser l'anglais
            if language not in templates:
                language = 'en'
                
            # Préparer les éléments pour les narratives
            precision = 0.0
            coverage = 0.0
            prediction_str = str(prediction) if prediction is not None else "unknown"
            
            # Extraire métriques de l'ancrage si disponibles
            if hasattr(anchor_rules, 'precision'):
                precision = float(anchor_rules.precision)
            if hasattr(anchor_rules, 'coverage'):
                coverage = float(anchor_rules.coverage)
                
            # Extraire les règles d'ancrage en texte
            conditions_text = "no specific conditions found"
            if isinstance(anchor_rules, list) and anchor_rules:
                conditions_text = " AND ".join(anchor_rules)
            elif hasattr(anchor_rules, 'names') and anchor_rules.names:
                conditions_text = " AND ".join(anchor_rules.names)
                
            # Simplifier les conditions pour les audiences non techniques
            conditions_simple = conditions_text.replace(' AND ', ' et ')  # Version plus simple
            conditions_very_simple = conditions_text.replace(' AND ', ' et ')
            if len(conditions_very_simple.split(' et ')) > 2:
                # Limiter à 2 conditions max pour le public
                parts = conditions_very_simple.split(' et ')
                conditions_very_simple = ' et '.join(parts[:2]) + f" et {len(parts)-2} autres facteurs"
                
            # Extraire les principales caractéristiques
            top_features = []
            if feature_importances:
                # Trier par importance décroissante et prendre les 3 premières
                sorted_features = sorted(feature_importances, key=lambda x: abs(x.importance), reverse=True)
                top_features = sorted_features[:3]
                
            # Formater les noms des caractéristiques principales
            top_features_str = ", ".join([f"{feature.feature_name} ({feature.importance:.2f})" for feature in top_features]) if top_features else "aucune caractéristique identifiée"
            top_features_simple = ", ".join([feature.feature_name for feature in top_features]) if top_features else "aucun facteur identifié"
            top_features_very_simple = top_features_simple
                
            # Construire la narrative pour chaque niveau d'audience
            for audience in ['technical', 'business', 'public']:
                if audience == audience_level.lower() or audience_level == AudienceLevel.ALL:
                    template = templates[language][audience]
                    narrative_parts = []
                    
                    # Construction de la narrative selon le niveau d'audience
                    if audience == 'technical':
                        narrative_parts.append(template['intro'])
                        narrative_parts.append(template['rule'].format(
                            prediction=prediction_str,
                            conditions=conditions_text
                        ))
                        narrative_parts.append(template['precision'].format(precision=precision))
                        narrative_parts.append(template['coverage'].format(coverage=coverage))
                        narrative_parts.append(template['outro'].format(top_features_str=top_features_str))
                    elif audience == 'business':
                        narrative_parts.append(template['intro'])
                        narrative_parts.append(template['rule'].format(
                            prediction=prediction_str,
                            conditions_simple=conditions_simple
                        ))
                        narrative_parts.append(template['reliability'].format(precision=precision))
                        narrative_parts.append(template['outro'].format(top_features_simple=top_features_simple))
                    else:  # public
                        narrative_parts.append(template['intro'])
                        narrative_parts.append(template['rule'].format(
                            prediction=prediction_str,
                            conditions_very_simple=conditions_very_simple
                        ))
                        narrative_parts.append(template['outro'].format(top_features_very_simple=top_features_very_simple))
                        
                    narratives[audience] = " ".join(narrative_parts)
                    
        except Exception as e:
            self._logger.warning(f"Erreur lors de la génération de narratives d'explication: {str(e)}")
            narratives['error'] = f"Unable to generate explanation narrative: {str(e)}"
            
        return narratives
        
    def _compute_explanation_cached(self, instance, **kwargs):
        """
        Méthode centrale pour calculer l'explication avec toutes les fonctionnalités avancées.
        Cette méthode intègre la gestion du GPU, le calcul des métriques de qualité,
        la génération de narratives multilingues et la vérification de conformité.
        
        Args:
            instance: Instance à expliquer
            **kwargs: Paramètres d'explication
                data_type: Type de données ('tabular', 'text', 'image')
                threshold: Seuil de précision pour l'ancrage
            
            # Créer un résultat d'erreur
            result = ExplanationResult(
                method=ExplainabilityMethod.ANCHOR,
                feature_importances=[],
                metadata={
                    'error': error_msg,
                    'execution_time_ms': int((time.time() - start_time) * 1000)
                },
                audience_level=audience_level
            )
        else:
            # Extraire les informations importantes du résultat calculé
            feature_importances_data = explanation_data.get('feature_importances', [])
            feature_importances = []
                                                                   use_unk=kwargs.get('use_unk', True),
                                                                   use_uppercase_unk=kwargs.get('use_uppercase_unk', True),
                                                                   sample_proba=kwargs.get('sample_proba', 0.5),
                                                                   top_n=kwargs.get('top_n', 100),
                                                                   temperature=kwargs.get('temperature', 1.0))
                    
                elif data_type == 'image':
                    # Pour les images
                    anchor_explanation = self._explain_image_instance(instance_array,
                                                                    threshold=threshold,
                                                                    segmentation_fn=kwargs.get('segmentation_fn', None),
                                                                    p_sample=kwargs.get('p_sample', 0.5),
                                                                    n_segments=kwargs.get('n_segments', 10))
            
            # 3. Vérifier que l'explication a été générée avec succès
            if anchor_explanation is None:
                raise ValueError("L'explication Anchor n'a pas pu être générée")
                
            # 4. Obtenir une prédiction du modèle pour l'instance
            try:
                if hasattr(self._model, 'predict_proba') and self._get_task_type() == 'classification':
                    prediction = self._model.predict_proba(instance_array)[0]
                    predicted_class = np.argmax(prediction)
                    if self._class_names and predicted_class < len(self._class_names):
                        prediction_label = self._class_names[predicted_class]
                    else:
                        prediction_label = str(predicted_class)
                else:
                    prediction = self._model.predict(instance_array)[0]
                    prediction_label = str(prediction)
            except Exception as pred_err:
                self._logger.warning(f"Erreur lors de la prédiction: {str(pred_err)}")
                prediction = None
                prediction_label = "unknown"
                
            # 5. Extraire les règles d'ancrage
            feature_names = kwargs.get('feature_names', self._feature_names)
            anchor_rules = self._extract_anchor_rules(anchor_explanation, feature_names)
                
            # 6. Convertir les règles en importances de caractéristiques
            feature_importances = self._convert_rules_to_importances(anchor_rules, feature_names)
                
            # 7. Calculer les métriques de qualité de l'explication
            quality_metrics = {}
            if self._config.compute_quality_metrics:
                quality_metrics = self._compute_explanation_quality_metrics(
                    anchor_rules=anchor_rules, 
                    explanation=anchor_explanation, 
                    instance=instance, 
                    prediction=prediction
                )
            
            # 8. Générer les narratives d'explication
            narratives = {}
            narratives[language] = self._generate_explanation_narrative(
                anchor_rules=anchor_explanation,
                feature_importances=feature_importances,
                prediction=prediction_label,
                audience_level=audience_level,
                language=language
            )
            
            # 9. Construire le résultat d'explication
            result = {
                "method": "anchor",
                "feature_importances": [fi.__dict__ for fi in feature_importances],
                "anchor_rules": anchor_rules,
                "anchor_metrics": {
                    "precision": float(anchor_explanation.precision) if hasattr(anchor_explanation, 'precision') else None,
                    "coverage": float(anchor_explanation.coverage) if hasattr(anchor_explanation, 'coverage') else None,
                },
                "prediction": prediction_label,
                "quality_metrics": quality_metrics,
                "narratives": narratives,
                "metadata": execution_metadata
            }
            
            # 10. Vérifier la conformité réglementaire si demandé
            if self._config.verify_compliance:
                compliance_result = self._verify_compliance_requirements(result, instance)
                result["compliance"] = compliance_result
                
            # Marquer comme réussi
            execution_metadata['success'] = True
            
        except Exception as e:
            # En cas d'erreur, enregistrer les détails
            self._logger.error(f"Erreur lors du calcul de l'explication: {str(e)}")
            self._logger.debug(traceback.format_exc())
            
            # Retourner un résultat minimal avec l'erreur
            result = {
                "method": "anchor",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "metadata": execution_metadata
            }
            
        finally:
            # Calculer le temps d'exécution
            execution_time_ms = int((time.time() - start_time) * 1000)
            execution_metadata['execution_time_ms'] = execution_time_ms
            result["metadata"] = execution_metadata
            
            # Tracer les métriques d'exécution
            self._logger.info(f"Explication Anchor calculée en {execution_time_ms}ms (succès: {execution_metadata['success']})")
            
            # Collecter des statistiques d'utilisation de mémoire/GPU si disponible
            try:
                import psutil
                memory_info = psutil.Process().memory_info()
                execution_metadata['memory_rss_bytes'] = memory_info.rss
                execution_metadata['memory_vms_bytes'] = memory_info.vms
                
                # Statistiques GPU si disponible
                if use_gpu:
                    if self._framework == 'tensorflow':
                        try:
                            import tensorflow as tf
                            if tf.config.list_physical_devices('GPU'):
                                gpu_stats = tf.config.experimental.get_memory_info('GPU:0')
                                execution_metadata['gpu_memory_bytes'] = gpu_stats['current']
                        except:
                            pass
                    elif self._framework == 'pytorch':
                        try:
                            import torch
                            if torch.cuda.is_available():
                                execution_metadata['gpu_memory_bytes'] = torch.cuda.memory_allocated()
                                execution_metadata['gpu_max_memory_bytes'] = torch.cuda.max_memory_allocated()
                        except:
                            pass
            except ImportError:
                pass  # psutil non disponible
            except Exception as mem_err:
                self._logger.debug(f"Erreur lors de la collecte des métriques de mémoire: {str(mem_err)}")
        
        return result
        
    def _verify_compliance_requirements(self, explanation_data, instance):
        """
        Vérifie la conformité réglementaire de l'explication générée.
        
        Args:
            explanation_data: Données d'explication générées
            instance: Instance expliquée
            
        Returns:
            dict: Résultat de la vérification de conformité
        """
        if not self._compliance_checker:
            return {"status": "skip", "reason": "Aucun vérificateur de conformité disponible"}
            
        try:
            # Extraire les éléments pertinents pour la vérification
            check_data = {
                "explainer_type": "AnchorExplainer",
                "feature_importances": explanation_data.get("feature_importances", []),
                "anchor_rules": explanation_data.get("anchor_rules", []),
                "anchor_metrics": explanation_data.get("anchor_metrics", {}),
                "instance": instance,
                "metadata": explanation_data.get("metadata", {})
            }
            
            # Vérifier la conformité
            compliance_result = self._compliance_checker.check_explanation(check_data)
            
            # Enregistrer un audit de conformité
            try:
                from ..audit import AuditLogger
                audit_logger = AuditLogger()
                audit_logger.log_compliance_check(
                    explainer_type="AnchorExplainer",
                    status=compliance_result.get("status", "unknown"),
                    requirements=compliance_result.get("requirements", {}),
                    details=compliance_result
                )
            except ImportError:
                self._logger.debug("Module d'audit non disponible")
            except Exception as audit_err:
                self._logger.debug(f"Erreur lors de l'audit de conformité: {str(audit_err)}")
                
            return compliance_result
            
        except Exception as e:
            error_result = {
                "status": "error",
                "reason": f"Erreur lors de la vérification de conformité: {str(e)}",
                "details": traceback.format_exc()
            }
            self._logger.warning(f"Erreur lors de la vérification de conformité: {str(e)}")
            return error_result
