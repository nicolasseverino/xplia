"""
SHAP Explainer pour XPLIA
========================

Module d'explicabilité avancé basé sur SHAP (SHapley Additive exPlanations) pour XPLIA.

Ce module implémente une intégration sophistiquée des explications SHAP dans le framework 
XPLIA, offrant une solution complète et optimisée pour l'explicabilité des modèles d'IA 
via la méthode des valeurs de Shapley.

Caractéristiques principales:
-----------------------------
* Support multi-framework automatique (scikit-learn, XGBoost, LightGBM, TensorFlow, PyTorch)
* Optimisations de performance avancées pour grands jeux de données
* Échantillonnage intelligent et parallélisation
* Métriques de qualité d'explication intégrées (fidélité, cohérence, stabilité)
* Adaptation sophistiquée par niveau d'audience (technique, métier, réglementaire)
* Génération automatique de narratives en langage naturel
* Gestion des valeurs d'interaction et des analyses contrefactuelles
* Robustesse face aux valeurs manquantes et aberrantes
* Visualisations avancées et interactives
* Intégration transparente avec les modules de conformité réglementaire
* Benchmarking de performance et qualité intégré
* Cache intelligent pour réexplications rapides

Références théoriques:
---------------------
* Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions.
* Štrumbelj, E., & Kononenko, I. (2014). Explaining prediction models and individual predictions.
* Sundararajan, M., & Najmi, A. (2020). The many Shapley values for model explanation.
"""

import logging
import time
import warnings
import inspect
import os
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache, partial
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set, Type

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Import des optimisations de performance XPLIA
from ..core.optimizations import (
    optimize, parallel_map, chunked_processing, cached_call,
    optimize_memory, XPLIAOptimizer
)
from ..core.performance import (
    ParallelExecutor, cached_result, memory_efficient, 
    process_in_chunks, optimize_explanations as opt_explanations
)

from ..core.base import (
    AudienceLevel, ExplainerBase, ExplainabilityMethod,
    ExplanationResult, FeatureImportance, ModelMetadata,
    ExplanationQuality, ExplanationFormat
)
from ..core.registry import register_explainer
from ..utils.performance import Timer, MemoryTracker
from ..compliance import ComplianceChecker

# Définition de types personnalisés pour une meilleure lisibilité
DataType = Union[np.ndarray, pd.DataFrame, List[Union[List, Dict]]]  
ModelType = Any
ShapValuesType = Union[np.ndarray, List[np.ndarray]]

@dataclass
class ShapConfig:
    """Configuration avancée pour l'explainer SHAP."""
    # Paramètres d'initialisation
    background_data: Optional[DataType] = None
    n_samples: int = 100
    link: str = 'identity'
    shap_type: str = 'auto'  # 'kernel', 'tree', 'deep', 'gradient', 'auto'
    
    # Paramètres de performance
    use_gpu: bool = False
    n_jobs: int = -1  # -1 = tous les CPU disponibles
    batch_size: int = 100
    cache_size: int = 128  # Taille du cache LRU
    timeout: Optional[float] = None  # Timeout en secondes
    
    # Paramètres d'explicabilité
    include_interactions: bool = False
    feature_perturbation: str = 'interventional'  # ou 'tree_path_dependent'
    masker_type: Optional[str] = None  # 'independent', 'tabular', 'image', 'text'
    approximate: bool = False  # Approximation pour gagner en performance
    clustering: bool = False  # Clustering pour réduire la dimensionnalité
    
    # Métriques de qualité
    compute_metrics: bool = True
    
    # Contraintes réglementaires
    compliance_mode: bool = False
    compliance_regs: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.n_jobs == -1:
            import multiprocessing
            self.n_jobs = multiprocessing.cpu_count()
        
        # Vérification de la cohérence des paramètres
        if self.clustering and not self.approximate:
            warnings.warn("Le clustering est plus efficace avec approximate=True")
            
        if self.use_gpu:
            try:
                import tensorflow as tf
                if not tf.test.is_gpu_available():
                    warnings.warn("GPU demandé mais non disponible. Utilisation du CPU.")
                    self.use_gpu = False
            except ImportError:
                warnings.warn("TensorFlow requis pour l'utilisation du GPU. Utilisation du CPU.")
                self.use_gpu = False


@register_explainer
class ShapExplainer(ExplainerBase):
    """
    Explainer avancé basé sur SHAP (SHapley Additive exPlanations).
    
    Cette classe implémente une solution sophistiquée d'explicabilité basée sur la théorie 
    des jeux coopératifs, attribuant à chaque caractéristique une valeur d'importance 
    optimale pour des décisions plus transparentes et justifiables.
    
    Caractéristiques principales:
    --------------------------
    * Détection automatique et optimisée du type de modèle et du framework
    * Sélection intelligente de l'implémentation SHAP optimale (kernel, tree, deep, etc.)
    * Optimisations avancées de performance:
      - Calcul parallélisé multi-CPU/GPU
      - Mise en cache intelligente des résultats
      - Échantillonnage adaptatif pour grands jeux de données
      - Gestion avancée de la mémoire et timeouts configurables
    * Métriques de qualité d'explication intégrées:
      - Fidélité (conformité aux prédictions du modèle)
      - Cohérence (stabilité face à des variations mineures)
      - Sparsité (concision des explications)
      - Exactitude (précision des attributions)
    * Support de cas d'usage spécialisés:
      - Analysis d'interactions entre variables
      - Évaluation contrefactuelle
      - Robustesse à l'adversarial noise
      - Agrégation multi-modèles 
    * Interface flexible et multi-format:
      - Visualisations adaptatives selon le niveau d'audience
      - Génération de narratives en langage naturel
      - Exportations multi-formats (JSON, HTML, PDF)
    * Intégration complète avec les modules de conformité réglementaire
    * Robustesse face aux données manquantes et aberrantes
    * Documentation contextuelle et exemples intégrés
    
    Cette implémentation inclut les dernières avancées de recherche en matière 
    d'explicabilité et d'interprétabilité des modèles d'IA, tout en offrant 
    une interface simple et des performances optimales pour l'utilisation en 
    production.
    """
    
    # Constantes de la classe
    _SHAP_TYPES = ["kernel", "tree", "deep", "gradient", "partition", "permutation", "auto"]
    _LINK_FUNCTIONS = ["identity", "logit"]    
    _MODEL_TYPES = {
        "tree": [
            "RandomForestClassifier", "RandomForestRegressor",
            "GradientBoostingClassifier", "GradientBoostingRegressor",
            "XGBClassifier", "XGBRegressor", "Booster", 
            "LGBMClassifier", "LGBMRegressor",
            "CatBoostClassifier", "CatBoostRegressor",
            "DecisionTreeClassifier", "DecisionTreeRegressor",
            "GradientBoosting", "RandomForest", "ExtraTreesClassifier",
            "ExtraTreesRegressor", "IsolationForest"
        ],
        "neural": [
            "Sequential", "Model", "Module", "Functional",
            "MLPClassifier", "MLPRegressor", "KerasClassifier", "KerasRegressor"
        ],
        "differentiable": [
            "LogisticRegression", "LinearRegression", 
            "SGDClassifier", "SGDRegressor", "LinearSVC", "LinearSVR",
            "Ridge", "Lasso", "ElasticNet"
        ],
        "generic": [
            "SVC", "SVR", "KNeighborsClassifier", "KNeighborsRegressor",
            "GaussianProcessClassifier", "GaussianProcessRegressor",
            "Pipeline", "BaggingClassifier", "BaggingRegressor",
            "VotingClassifier", "VotingRegressor", "StackingClassifier", "StackingRegressor"
        ]
    }
    _FRAMEWORK_IDENTIFIERS = {
        "sklearn": ["sklearn"],
        "xgboost": ["xgboost"],
        "lightgbm": ["lightgbm"],
        "catboost": ["catboost"],
        "tensorflow": ["tensorflow", "keras", "tf"],
        "pytorch": ["torch"],
        "mxnet": ["mxnet"]
    }
    
    def __init__(self, model: ModelType, **kwargs):
        """
        Initialise l'explainer SHAP avancé avec détection intelligente et optimisations.
        
        Args:
            model: Le modèle à expliquer (support multi-framework)
            **kwargs: Configuration avancée (voir ShapConfig pour tous les paramètres)
                background_data: Données d'arrière-plan pour les explications
                n_samples: Nombre d'échantillons pour l'approximation
                link: Fonction de lien ('identity', 'logit')
                shap_type: Type SHAP ('kernel', 'tree', 'deep', 'gradient', 'auto')
                use_gpu: Utiliser l'accélération GPU si disponible
                n_jobs: Nombre de processus parallèles (-1 = tous les CPU)
                batch_size: Taille des batches pour les grands datasets
                include_interactions: Calculer les interactions entre variables
                compute_metrics: Calculer les métriques de qualité d'explication
                approximate: Utiliser des approximations pour grandes dimensions
                compliance_mode: Activer le mode conformité réglementaire
                compliance_regs: Liste des réglementations à vérifier
        
        Exemples:
            >>> explainer = ShapExplainer(model)
            >>> explainer = ShapExplainer(model, background_data=X_train[:100])
            >>> explainer = ShapExplainer(
            ...     model, 
            ...     shap_type='tree', 
            ...     use_gpu=True,
            ...     compute_metrics=True,
            ...     include_interactions=True
            ... )
        """
        super().__init__(model, **kwargs)
        self._method = ExplainabilityMethod.SHAP
        
        # Configuration avancée
        if isinstance(kwargs.get('config'), ShapConfig):
            self._config = kwargs.get('config')
        else:
            self._config = ShapConfig(**kwargs)
        
        # Journalisation avancée
        self._logger = logging.getLogger(__name__)
        self._logger.info(f"Initialisation de ShapExplainer avec {self._config.shap_type} SHAP")
        
        # Statistiques internes et métriques
        self._stats = {
            "explanations_count": 0,
            "total_time_seconds": 0,
            "avg_time_seconds": 0,
            "max_time_seconds": 0,
            "min_time_seconds": float('inf'),
            "cache_hits": 0,
            "feature_importance_consistency": [],
            "explanation_quality": {}
        }
        
        # Extraction de métadonnées et configuration du mode de fonctionnement
        self._model_type = None
        self._model_framework = None
        self._is_classifier = False
        self._output_names = None
        self._feature_names = None
        self._extract_model_metadata()
        
        # Initialisation des hooks de pré/post-processing
        self._pre_explain_hooks = []
        self._post_explain_hooks = []
        
        # Compatibilité réglementaire si activée
        self._compliance_checker = None
        if self._config.compliance_mode:
            self._setup_compliance()
        
        # Configuration du cache selon les paramètres
        self._setup_caching()
        
        # Initialisation de l'explainer SHAP adapté au modèle
        self._shap_explainer = None
        with Timer() as timer:
            self._initialize_explainer()
        self._logger.info(f"Initialisation de l'explainer SHAP terminée en {timer.duration:.3f}s")
    
    def _extract_model_metadata(self):
        """Extrait des métadonnées complètes du modèle pour optimiser l'explication."""
        # Détection du type de modèle et du framework
        self._model_type = self._detect_model_type()
        self._model_framework = self._detect_model_framework()
        self._is_classifier = self._detect_is_classifier()
        
        # Recherche des noms de variables
        self._feature_names = self._extract_feature_names()
        self._output_names = self._extract_output_names()
        
        self._logger.info(f"Modèle détecté: {self._model_framework} - {self._model_type} - "
                       f"{'classificateur' if self._is_classifier else 'régresseur'}")
    
    def _setup_caching(self):
        """Configure le cache intelligent pour les explications."""
        cache_size = self._config.cache_size
        if cache_size > 0:
            # Application du décorateur LRU cache à certaines méthodes internes
            self._compute_shap_values = lru_cache(maxsize=cache_size)(self._compute_shap_values)
            self._feature_importance_to_explanation = lru_cache(maxsize=cache_size)(self._feature_importance_to_explanation)
            self._logger.info(f"Cache LRU activé avec {cache_size} entrées")
    
    def _setup_compliance(self):
        """Configure le vérificateur de conformité réglementaire pour les explications."""
        try:
            self._compliance_checker = ComplianceChecker()
            self._logger.info(f"Module de conformité initialisé avec les réglementations: "
                           f"{self._config.compliance_regs or ['par défaut']}")
        except Exception as e:
            self._logger.warning(f"Impossible d'initialiser le module de conformité: {str(e)}")
            self._compliance_checker = None
        
    def _maybe_use_gpu_context(self):
        """
        Contexte de gestion des ressources GPU pour l'explication.
        
        Permet d'optimiser automatiquement l'utilisation des ressources GPU
        lors du calcul des valeurs SHAP, particulièrement pour les grands jeux de données
        et les modèles complexes.
        
        Returns:
            Un gestionnaire de contexte qui configure l'environnement GPU optimalement
        """
        if not self._config.use_gpu:
            # Si GPU non activé, retourner un contexte vide (no-op)
            from contextlib import nullcontext
            return nullcontext()
            
        # Contexte GPU adapté au framework détecté
        model_type = self._detect_model_type()
        
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
                            
                    # Configuration spécifique pour SHAP
                    # L'API SHAP n'a pas d'API publique pour configurer le GPU, mais certaines
                    # implémentations de SHAP détectent automatiquement et utilisent le GPU si disponible
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
    
    def _initialize_explainer(self):
        """
        Initialise l'explainer SHAP optimal en fonction du type de modèle détecté.
        
        Cette méthode implémente une sélection intelligente et optimisée de l'explainer
        SHAP approprié selon une analyse approfondie du modèle et du contexte.
        Inclut des optimisations avancées pour maximiser performance et précision.
        """
        import shap
        
        # Mesure des performances
        with Timer() as timer, MemoryTracker() as mem_tracker:
            # Détection sophistiquée du type de modèle
            model_type = self._detect_model_type()
            framework = model_type.split('-')[0] if '-' in model_type else 'unknown'
            
            self._logger.info(f"Initialisation de l'explainer SHAP pour: {model_type}")
            
            # Préparation des paramètres communs
            kwargs = {}
            
            # Appliquer les optimisations avancées selon la configuration
            if hasattr(shap, 'utils') and hasattr(shap.utils, 'set_parallelism'):
                try:
                    shap.utils.set_parallelism(self._config.n_jobs)
                    self._logger.debug(f"Parallélisme SHAP configuré avec {self._config.n_jobs} threads")
                except Exception as e:
                    self._logger.debug(f"Impossible de configurer le parallélisme SHAP: {str(e)}")
            
            # Configuration spécifique aux modèles d'arbres
            if self._config.shap_type == 'tree' or (self._config.shap_type == 'auto' and 
                   (self._is_tree_model() or any(x in model_type.lower() for x in ['tree', 'forest', 'boost', 'xgb', 'lgbm', 'catboost']))):
                
                # Optimisations pour les modèles d'arbres
                kwargs = {
                    'model_output': 'raw',  # ou 'probability', 'margin', 'logit'
                    'feature_perturbation': self._config.feature_perturbation
                }
                
                # Gérer les interactions si nécessaire
                if self._config.include_interactions:
                    kwargs['interactions'] = True
                    self._logger.debug("Calcul des interactions activé pour TreeExplainer")
                    
                # Détecter le framework XGBoost/LightGBM/Catboost pour appliquer les optimisations spécifiques
                if 'xgboost' in framework:
                    self._logger.debug("Optimisations spécifiques pour XGBoost activées")
                    kwargs['data'] = self._get_background_data()
                    if self._config.approximate:
                        kwargs['approximate'] = True
                    if self._config.use_gpu:
                        # Vérifier si XGBoost est compilé avec GPU
                        try:
                            import xgboost
                            if hasattr(xgboost, 'gpu_predictor'):
                                kwargs['gpu_predictor'] = True
                                self._logger.debug("Support GPU XGBoost activé")
                        except (ImportError, AttributeError):
                            pass
                
                # Création de l'explainer avec tous les paramètres optimisés
                try:
                    self._logger.info("Initialisation de TreeExplainer avec paramètres optimisés")
                    self._shap_explainer = shap.TreeExplainer(self._model, **kwargs)
                    return
                except Exception as e:
                    self._logger.warning(f"Erreur lors de l'initialisation de TreeExplainer: {str(e)}. Repli...")
            
            # Configuration pour les réseaux de neurones profonds
            elif self._config.shap_type == 'deep' or (self._config.shap_type == 'auto' and 
                    (self._is_deep_model() or any(x in model_type.lower() for x in ['tensorflow', 'keras', 'torch', 'neural']))):
                
                try:
                    background_data = self._get_background_data()
                    
                    # Optimisations pour réseaux de neurones
                    if framework == 'tensorflow':
                        # Vérifier si on peut utiliser GradientExplainer qui est plus efficace pour TF
                        if hasattr(shap, 'GradientExplainer'):
                            self._logger.info("Initialisation de GradientExplainer pour TensorFlow")
                            self._shap_explainer = shap.GradientExplainer(self._model, background_data)
                            return
                    
                    # Déterminer le bon type d'explainer pour réseaux profonds
                    self._logger.info("Initialisation de DeepExplainer")
                    if self._config.use_gpu:
                        with self._set_gpu_memory_growth():
                            self._shap_explainer = shap.DeepExplainer(self._model, background_data)
                    else:
                        self._shap_explainer = shap.DeepExplainer(self._model, background_data)
                    return
                except Exception as e:
                    self._logger.warning(f"Erreur lors de l'initialisation de DeepExplainer: {str(e)}. Repli...")
            
            # Configuration pour les modèles différentiables
            elif self._config.shap_type == 'gradient' or (self._config.shap_type == 'auto' and 
                      self._is_differentiable_model()):
                
                try:
                    background_data = self._get_background_data()
                    self._logger.info("Initialisation de GradientExplainer")
                    self._shap_explainer = shap.GradientExplainer(self._model, background_data)
                    return
                except Exception as e:
                    self._logger.warning(f"Erreur lors de l'initialisation de GradientExplainer: {str(e)}. Repli...")
            
            # Configuration par défaut: KernelSHAP (universel mais plus lent)
            try:
                background_data = self._get_background_data()
                
                # Optimisations pour KernelExplainer
                kwargs = {
                    'link': self._config.link,
                }
                
                if hasattr(self._model, 'predict_proba'):
                    kwargs['output_names'] = 'probability'
                
                if self._config.approximate:
                    # Mode approximation pour grands jeux de données
                    kwargs['nsamples'] = self._config.n_samples
                    # Plus l'approximation est fine, plus précis mais plus lent
                    if self._config.n_samples <= 50:
                        self._logger.warning("Petit nombre d'échantillons, précision possiblement limitée")
                
                if self._config.clustering and hasattr(shap, 'kmeans'):
                    # Clustering pour réduire les calculs
                    with Timer() as cluster_timer:
                        n_clusters = min(50, len(background_data) // 10) if len(background_data) > 500 else None
                        if n_clusters:
                            try:
                                background_data = shap.kmeans(background_data, n_clusters)
                                self._logger.debug(f"Clustering appliqué: {n_clusters} clusters en {cluster_timer.duration:.2f}s")
                            except Exception as e:
                                self._logger.warning(f"Erreur lors du clustering: {str(e)}")
                
                # Implémentation de maskers si disponible dans la version de SHAP
                masker = None
                if self._config.masker_type and hasattr(shap, 'maskers'):
                    if self._config.masker_type == 'tabular' and hasattr(shap.maskers, 'Tabular'):
                        masker = shap.maskers.Tabular(background_data)
                    elif self._config.masker_type == 'text' and hasattr(shap.maskers, 'Text'):
                        masker = shap.maskers.Text()
                    elif self._config.masker_type == 'image' and hasattr(shap.maskers, 'Image'):
                        masker = shap.maskers.Image()
                
                self._logger.info("Initialisation de KernelExplainer (explainer générique)")
                
                if masker and hasattr(shap, 'Explainer'):
                    # Utiliser le nouvel API SHAP si disponible
                    self._shap_explainer = shap.Explainer(self._model_predict_wrapper, masker, **kwargs)
                else:                
                    # API standard KernelExplainer
                    self._shap_explainer = shap.KernelExplainer(
                        self._model_predict_wrapper,
                        background_data,
                        **kwargs
                    )
                    
            except Exception as e:
                self._logger.error(f"Erreur critique lors de l'initialisation de l'explainer SHAP: {str(e)}")
                raise RuntimeError(f"Initialisation de l'explainer SHAP impossible: {str(e)}")
        
        # Log des performances de l'initialisation
        memory_usage = mem_tracker.peak_usage_mb
        self._logger.info(
            f"Explainer SHAP initialisé en {timer.duration:.2f}s avec {memory_usage:.1f}MB "
            f"(Type: {self._config.shap_type if self._config.shap_type != 'auto' else 'auto-detect'})"
        )
    
    def _maybe_use_gpu_context(self):
        """
        Contexte de gestion des ressources GPU pour l'explication.
        
        Permet d'optimiser automatiquement l'utilisation des ressources GPU
        lors du calcul des valeurs SHAP, particulièrement pour les grands jeux de données
        et les modèles complexes.
        
        Returns:
            Un gestionnaire de contexte qui configure l'environnement GPU optimalement
        """
        import contextlib
        
        @contextlib.contextmanager
        def _gpu_context():
            if not self._config.use_gpu:
                yield
                return
                
            try:
                # TensorFlow GPU configuration
                try:
                    import tensorflow as tf
                    if hasattr(tf, 'config') and hasattr(tf.config, 'experimental'):
                        with self._set_gpu_memory_growth():
                            yield
                            return
                except (ImportError, AttributeError):
                    pass
                    
                # PyTorch GPU configuration
                try:
                    import torch
                    if torch.cuda.is_available():
                        self._logger.debug(f"Utilisation de PyTorch avec GPU: {torch.cuda.get_device_name(0)}")
                        # Optionally set additional PyTorch GPU configurations here
                        pass
                except (ImportError, AttributeError):
                    pass
                    
                # No specific GPU config needed/available
                yield
            except Exception as e:
                self._logger.warning(f"Erreur lors de la configuration GPU: {str(e)}")
                yield
        
        return _gpu_context()
        
    def _set_gpu_memory_growth(self):
        """Configure la croissance mémoire GPU dynamique pour TensorFlow.
        
        Cette méthode contextuelle permet une utilisation optimale de la mémoire GPU
        en configurant une allocation dynamique, évitant ainsi les erreurs OOM et 
        permettant une meilleure répartition des ressources entre plusieurs processus.
        """
        import contextlib
        
        @contextlib.contextmanager
        def _set_tf_memory_growth():
            try:
                import tensorflow as tf
                if hasattr(tf, 'config') and hasattr(tf.config, 'experimental'):
                    gpus = tf.config.experimental.list_physical_devices('GPU')
                    if gpus:
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                        self._logger.debug(f"{len(gpus)} GPU(s) configuré(s) avec memory growth")
            except Exception as e:
                self._logger.debug(f"Impossible de configurer la mémoire GPU: {str(e)}")
            yield
        
        return _set_tf_memory_growth()
    
    def _detect_model_type(self) -> str:
        """
        Détecte le type de modèle pour choisir l'explainer SHAP optimal.
        
        Cette méthode avancée identifie avec précision la nature du modèle à travers
        une analyse multiniveau (framework, architecture, caractéristiques), permettant
        de sélectionner l'implémentation SHAP la plus performante et précise pour ce modèle.
        
        Returns:
            str: Type de modèle détecté avec son framework ('sklearn-RandomForestClassifier', etc.)
        """
        # Analyser le framework du modèle
        model_module = self._get_model_module()
        framework = 'unknown'
        for fw_name, identifiers in self._FRAMEWORK_IDENTIFIERS.items():
            if any(ident in model_module.lower() for ident in identifiers):
                framework = fw_name
                break
                
        # Récupérer le nom spécifique du modèle avec gestion avancée des cas complexes
        model_name = type(self._model).__name__
        
        # Cas 1: Pipeline scikit-learn - récupérer le modèle final
        if self._is_pipeline():
            try:
                # Extraire le dernier estimateur du pipeline
                if hasattr(self._model, 'steps') and self._model.steps:
                    last_step = self._model.steps[-1][1]
                    model_name = type(last_step).__name__
                    # Vérifier le module du dernier estimateur pour plus de précision
                    last_module = last_step.__module__
                    for fw_name, identifiers in self._FRAMEWORK_IDENTIFIERS.items():
                        if any(ident in last_module.lower() for ident in identifiers):
                            framework = fw_name
                            break
            except (IndexError, AttributeError) as e:
                self._logger.debug(f"Impossible d'analyser le pipeline: {str(e)}")
                
        # Cas 2: Méta-estimateurs et ensembles - analyser les estimateurs de base
        elif self._is_ensemble():
            try:
                base_estimator = None
                # Essayer différentes structures d'ensembles connues
                if hasattr(self._model, 'estimators_') and self._model.estimators_:
                    base_estimator = self._model.estimators_[0]
                elif hasattr(self._model, 'estimators') and self._model.estimators:
                    base_estimator = self._model.estimators[0]
                elif hasattr(self._model, 'base_estimator'):
                    base_estimator = self._model.base_estimator
                    
                if base_estimator:
                    model_name = f"Ensemble({type(base_estimator).__name__})"
            except (IndexError, AttributeError) as e:
                self._logger.debug(f"Impossible d'analyser l'ensemble: {str(e)}")
        
        # Cas 3: Modèles à noyaux - identifier le type de noyau
        elif any(kernel_type in model_name.lower() for kernel_type in ["svc", "svr", "gaussianprocess"]):
            kernel_name = "rbf"  # noyau par défaut
            if hasattr(self._model, "kernel") and isinstance(self._model.kernel, str):
                kernel_name = self._model.kernel
            elif hasattr(self._model, "get_params"):
                params = self._model.get_params()
                if "kernel" in params and isinstance(params["kernel"], str):
                    kernel_name = params["kernel"]
            model_name = f"{model_name}({kernel_name})"
        
        # Cas 4: Modèles de deep learning - analyser l'architecture
        elif framework in ["tensorflow", "pytorch", "mxnet"]:
            # Ajouter des informations sur la profondeur ou le type d'architecture
            if framework == "tensorflow" and hasattr(self._model, "layers"):
                try:
                    n_layers = len(self._model.layers)
                    layer_types = [l.__class__.__name__ for l in self._model.layers][:3]  # Premiers types de couches
                    model_name = f"{model_name}(layers={n_layers}, types={','.join(layer_types)}...)"
                except:
                    pass
            elif framework == "pytorch" and hasattr(self._model, "_modules"):
                try:
                    n_modules = len(list(self._model._modules.items()))
                    model_name = f"{model_name}(modules={n_modules})"
                except:
                    pass
        
        # Cas 5: Détection avancée pour modèles spécifiques nécessitant un traitement particulier
        if framework == "xgboost" and "booster" in model_name.lower():
            # XGBoost nécessite une détection spécifique pour les types d'objectif
            if hasattr(self._model, "objective"):
                model_name = f"{model_name}({self._model.objective})"
        
        # Enregistrer et retourner le résultat final avec format standard
        detected_type = f"{framework}-{model_name}"
        self._logger.info(f"Type de modèle détecté: {detected_type}")
        return detected_type
        
    def _get_model_module(self) -> str:
        """Obtient le module du modèle pour aider à la détection du framework."""
        try:
            return str(self._model.__module__)
        except (AttributeError, TypeError):
            # Essayer d'obtenir le module via le type
            return str(type(self._model).__module__)
    
    def _is_pipeline(self) -> bool:
        """Vérifie si le modèle est un pipeline ou une chaîne de traitement."""
        # Vérifier scikit-learn Pipeline et ColumnTransformer
        if hasattr(self._model, "steps") and callable(getattr(self._model, "fit", None)):
            return True
        # Vérifier les pipelines PyTorch (nn.Sequential)
        if hasattr(self._model, "forward") and hasattr(self._model, "_modules"):
            return True
        # Vérifier les pipelines Keras (Sequential)
        if hasattr(self._model, "layers") and hasattr(self._model, "add"):
            return True
        return False
    
    def _is_ensemble(self) -> bool:
        """Vérifie si le modèle est un ensemble ou un méta-estimateur."""
        # Ensembles scikit-learn standards
        if hasattr(self._model, "estimators_") or hasattr(self._model, "estimators"):
            return True
        # Autres méta-estimateurs
        if hasattr(self._model, "base_estimator") or hasattr(self._model, "base_estimator_"):
            return True
        # Méta-estimateurs de vote
        if hasattr(self._model, "estimators") and isinstance(getattr(self._model, "estimators", None), list):
            return True
        return False
    
    def _is_tree_model(self) -> bool:
        """Vérifie si le modèle est un modèle basé sur des arbres."""
        tree_modules = ['sklearn.ensemble', 'xgboost', 'lightgbm', 'catboost']
        model_module = self._model.__class__.__module__
        return any(module in model_module for module in tree_modules)
    
    def _is_deep_model(self) -> bool:
        """Vérifie si le modèle est un réseau de neurones profond."""
        deep_modules = ['keras', 'tensorflow', 'torch']
        model_module = self._model.__class__.__module__
        return any(module in model_module for module in deep_modules)
    
    def _is_differentiable_model(self) -> bool:
        """Vérifie si le modèle est différentiable (pour GradientExplainer)."""
        model_type = self._detect_model_type()
        return any(framework in model_type for framework in ['tensorflow', 'pytorch', 'torch'])
        
    def _model_predict_wrapper(self, X):
        """
        Wrapper unifié pour obtenir des prédictions standardisées de différents types de modèles.
        
        Cette méthode gère intelligemment les spécificités des différents frameworks de ML
        et normalise les formats de sortie pour faciliter les calculs de métriques de qualité.
        
        Args:
            X: Données d'entrée (DataFrame, ndarray, etc.)
            
        Returns:
            np.ndarray: Prédictions normalisées
        """
        model_type = self._detect_model_type()
        
        # Optimisation: utiliser le GPU si configuré
        with self._maybe_use_gpu_context():
            try:
                # Convertir en format approprié si nécessaire
                X_processed = X
                if hasattr(self._model, 'predict') and not ('tensorflow' in model_type or 'pytorch' in model_type):
                    # Modèles scikit-learn, XGBoost, etc.
                    if hasattr(self._model, 'predict_proba'):
                        # Classifier avec probabilités
                        predictions = self._model.predict_proba(X_processed)
                    else:
                        # Régression ou autre
                        predictions = self._model.predict(X_processed)
                        # Conversion 1D -> 2D si nécessaire
                        if predictions.ndim == 1:
                            predictions = predictions.reshape(-1, 1)
                elif 'tensorflow' in model_type:
                    # Modèles TensorFlow
                    import tensorflow as tf
                    # Conversion en tenseur TF si nécessaire
                    if not isinstance(X_processed, tf.Tensor) and not isinstance(X_processed, tf.Variable):
                        X_processed = tf.convert_to_tensor(X_processed, dtype=tf.float32)
                    predictions = self._model(X_processed).numpy()
                elif 'pytorch' in model_type or 'torch' in model_type:
                    # Modèles PyTorch
                    import torch
                    # Conversion en tenseur PyTorch si nécessaire
                    if not isinstance(X_processed, torch.Tensor):
                        X_processed = torch.tensor(X_processed.values if hasattr(X_processed, 'values') else X_processed, 
                                               dtype=torch.float32)
                    # Désactiver le calcul de gradient pour l'inférence
                    with torch.no_grad():
                        predictions = self._model(X_processed).cpu().numpy()
                else:
                    # Méthode générique pour les autres cas
                    if hasattr(self._model, '__call__'):
                        predictions = self._model(X_processed)
                        # Convertir en numpy si nécessaire
                        if not isinstance(predictions, np.ndarray):
                            predictions = np.array(predictions)
                    else:
                        raise ValueError(f"Type de modèle non supporté pour les prédictions: {model_type}")
                
                return predictions
                
            except Exception as e:
                self._logger.error(f"Erreur lors de la prédiction avec le modèle: {str(e)}")
                raise RuntimeError(f"Erreur lors de la prédiction: {str(e)}")
    
    def _get_background_data(self):
        """
        Récupère les données d'arrière-plan pour les explainers qui en ont besoin.
        
        Returns:
            numpy.ndarray: Données d'arrière-plan
        """
        if self._background_data is not None:
            return self._background_data
        
        # Données synthétiques si aucune donnée n'est fournie
        # C'est une solution de repli, l'idéal est de fournir de vraies données
        self._logger.warning("Aucune donnée d'arrière-plan fournie. "
                           "Génération de données synthétiques, ce qui peut affecter la qualité des explications.")
        
        # Tenter de déduire la forme d'entrée du modèle
        input_shape = self._infer_model_input_shape()
        if input_shape:
            # Générer des données aléatoires normalisées
            return np.random.normal(0, 0.1, size=(self._n_samples, *input_shape))
        
        # Si impossible de déduire la forme, erreur
        raise ValueError("Impossible de générer des données d'arrière-plan. "
                       "Veuillez fournir des données via le paramètre 'background_data'.")
    
    def _infer_model_input_shape(self):
        """
        Tente de déduire la forme des entrées du modèle.
        
        Returns:
            tuple ou None: Forme déduite ou None si impossible
        """
        model_type = self._detect_model_type()
        
        if model_type == 'tensorflow':
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
        Wrapper pour la fonction de prédiction du modèle, adapté pour SHAP.
        
        Args:
            x: Données d'entrée
            
        Returns:
            numpy.ndarray: Prédictions du modèle
        """
        try:
            model_type = self._detect_model_type()
            
            if model_type in ['sklearn-ensemble', 'sklearn-linear', 'xgboost', 'lightgbm', 'catboost']:
                if hasattr(self._model, 'predict_proba'):
                    return self._model.predict_proba(x)
                else:
                    return self._model.predict(x)
            elif model_type in ['tensorflow', 'pytorch']:
                # Conversion en format attendu par le modèle
                if isinstance(x, pd.DataFrame):
                    x = x.values
                
                # Appel au modèle
                result = self._model(x)
                
                # Conversion du résultat si nécessaire (pour PyTorch)
                if hasattr(result, 'detach') and hasattr(result.detach(), 'numpy'):
                    return result.detach().numpy()
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
            
    def explain(self, X, y=None, **kwargs):
        """
        Génère des explications SHAP optimisées pour un ensemble de données avec métriques de qualité.
        
        Cette implémentation avancée prend en charge l'explicabilité à grande échelle avec
        des optimisations de performance, des métriques de qualité d'explication, des contrôles
        de conformité réglementaire et une adaptation dynamique au niveau d'audience.
        
        Args:
            X: Données d'entrée à expliquer (DataFrame, numpy array, ou liste)
            y: Valeurs cibles réelles (optionnel), utilisées pour évaluer la fidélité
            **kwargs: Paramètres avancés
                output_index: Indice de sortie pour les modèles multi-sorties
                audience_level: Niveau d'audience (AudienceLevel.TECHNICAL, BUSINESS, PUBLIC)
                summarize: Résumer les résultats pour l'ensemble du dataset
                sample_size: Pour grands datasets, taille d'échantillon représentatif à utiliser
                compute_quality_metrics: Calculer les métriques de qualité d'explication (True par défaut)
                verify_compliance: Vérifier la conformité réglementaire de l'explication
                regulations: Liste des réglementations à vérifier (ex: ['RGPD', 'AI_ACT', 'HIPAA'])
                return_raw_shap: Inclure les valeurs SHAP brutes dans le résultat (False par défaut)
                batch_size: Nombre d'instances à traiter simultanément (pour grandes dimensions)
                include_predictions: Inclure les prédictions du modèle avec l'explication
                generate_narratives: Générer des descriptions textuelles des explications
                language: Langue pour les narratives générées ('fr' ou 'en')
                
        Returns:
            ExplanationResult: Résultat enrichi avec métriques de qualité et informations de conformité
            
        Raises:
            ValueError: Si le format des données est incompatible
            RuntimeError: Si l'explication échoue pour des raisons techniques
            TimeoutError: Si le calcul dépasse le timeout configuré
        """
        # Validation et prétraitement des données
        if self._shap_explainer is None:
            raise ValueError("L'explainer SHAP n'a pas été correctement initialisé.")
            
        # Extraction des paramètres avancés avec valeurs par défaut
        output_index = kwargs.get('output_index', None)
        audience_level = kwargs.get('audience_level', AudienceLevel.TECHNICAL)
        summarize = kwargs.get('summarize', False)
        sample_size = kwargs.get('sample_size', None)
        compute_quality = kwargs.get('compute_quality_metrics', True)
        verify_compliance = kwargs.get('verify_compliance', self._config.verify_compliance)
        regulations = kwargs.get('regulations', self._config.compliance_regulations)
        return_raw_shap = kwargs.get('return_raw_shap', False)
        batch_size = kwargs.get('batch_size', self._config.batch_size)
        include_predictions = kwargs.get('include_predictions', True)
        generate_narratives = kwargs.get('generate_narratives', False)
        language = kwargs.get('language', 'fr')
        
        # Gestion des grands jeux de données avec échantillonnage intelligent
        if sample_size and len(X) > sample_size:
            self._logger.info(f"Échantillonnage de {sample_size} instances parmi {len(X)} pour performance")
            if hasattr(X, 'sample'):
                # Échantillonnage stratifié si y est disponible
                if y is not None and hasattr(pd, 'DataFrame'):
                    try:
                        # Tentative d'échantillonnage stratifié
                        temp_df = pd.DataFrame(X)
                        temp_df['_target'] = y
                        stratified = temp_df.groupby('_target', group_keys=False).apply(
                            lambda x: x.sample(min(len(x), int(sample_size * len(x) / len(temp_df))))
                        )
                        indices = stratified.index
                        X_sample = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
                        y_sample = y[indices] if y is not None else None
                        self._logger.debug("Échantillonnage stratifié appliqué")
                    except Exception as e:
                        self._logger.debug(f"Échec de l'échantillonnage stratifié: {str(e)}. Retour à aléatoire.")
                        X_sample = X.sample(sample_size) if hasattr(X, 'sample') else X[np.random.choice(len(X), sample_size, replace=False)]
                        y_sample = y[X_sample.index] if y is not None and hasattr(y, '__getitem__') else None
                else:
                    # Échantillonnage aléatoire
                    X_sample = X.sample(sample_size)
                    y_sample = y[X_sample.index] if y is not None and hasattr(y, '__getitem__') else None
            else:
                # Échantillonnage numpy
                indices = np.random.choice(len(X), sample_size, replace=False)
                X_sample = X[indices]
                y_sample = y[indices] if y is not None and hasattr(y, '__getitem__') else None
                
            # Utilisation des échantillons à la place des données complètes
            X, y = X_sample, y_sample
            
        # Préparation du calcul des valeurs SHAP
        with Timer() as compute_timer, MemoryTracker() as compute_memory:
            # Utilisation du cache si activé
            cache_key = None
            if self._config.use_cache:
                try:
                    import hashlib
                    import pickle
                    
                    # Génération d'une clé de cache unique pour ces données
                    data_hash = hashlib.md5(pickle.dumps((X, output_index))).hexdigest()
                    model_hash = self._model_hash if hasattr(self, '_model_hash') else hashlib.md5(
                        pickle.dumps(self._extract_model_signature())).hexdigest()
                    cache_key = f"shap_{model_hash}_{data_hash}_{self._config.n_samples}"
                    
                    # Tentative de récupération depuis le cache
                    if hasattr(self, '_explanation_cache') and cache_key in self._explanation_cache:
                        self._logger.info("Explication récupérée depuis le cache")
                        shap_values, expected_value, execution_time = self._explanation_cache[cache_key]
                        cached = True
                except Exception as e:
                    self._logger.debug(f"Erreur lors de l'accès au cache: {str(e)}")
                    cache_key = None
            
            # Calcul des valeurs SHAP si non trouvées dans le cache
            if not cache_key or not hasattr(self, '_explanation_cache') or cache_key not in self._explanation_cache:
                self._logger.info("Calcul des valeurs SHAP en cours...")
                
                # Traitement par lots pour grands jeux de données
                if batch_size and len(X) > batch_size:
                    self._logger.info(f"Traitement par lots de {batch_size} instances")
                    
                    # Division en lots
                    num_batches = (len(X) + batch_size - 1) // batch_size
                    results = []
                    expected_values = []
                    
                    # Paramètres de parallélisation
                    parallel = self._config.n_jobs > 1
                    
                    if parallel:
                        from concurrent.futures import ProcessPoolExecutor
                        
                        # Fonction pour calcul parallèle des valeurs SHAP par lot
                        def process_batch(batch_idx):
                            start_idx = batch_idx * batch_size
                            end_idx = min(start_idx + batch_size, len(X))
                            batch_X = X[start_idx:end_idx]
                            with self._maybe_use_gpu_context():
                                batch_result = self._shap_explainer.shap_values(
                                    batch_X, l1_reg=self._config.l1_reg, output_index=output_index
                                )
                            return batch_result
                            
                        try:
                            self._logger.debug(f"Exécution parallèle avec {self._config.n_jobs} workers")
                            with ProcessPoolExecutor(max_workers=self._config.n_jobs) as executor:
                                batch_results = list(executor.map(process_batch, range(num_batches)))
                                
                            # Fusion des résultats
                            if isinstance(batch_results[0], list):
                                # Multi-output: list of arrays per class
                                shap_values = [np.vstack([batch[i] for batch in batch_results]) 
                                              for i in range(len(batch_results[0]))]
                            else:
                                # Single output: array per instance
                                shap_values = np.vstack(batch_results)
                        except Exception as e:
                            self._logger.warning(f"Erreur lors du calcul parallèle: {str(e)}. Passage en mode séquentiel.")
                            parallel = False
                    
                    # Traitement séquentiel si parallélisation impossible
                    if not parallel:
                        batch_results = []
                        for batch_idx in range(num_batches):
                            self._logger.debug(f"Traitement du lot {batch_idx+1}/{num_batches}")
                            start_idx = batch_idx * batch_size
                            end_idx = min(start_idx + batch_size, len(X))
                            batch_X = X[start_idx:end_idx]
                            
                            with self._maybe_use_gpu_context():
                                batch_result = self._shap_explainer.shap_values(
                                    batch_X, l1_reg=self._config.l1_reg, output_index=output_index
                                )
                            batch_results.append(batch_result)
                        
                        # Fusion des résultats
                        if isinstance(batch_results[0], list):
                            # Multi-output: list of arrays per class
                            shap_values = [np.vstack([batch[i] for batch in batch_results]) 
                                          for i in range(len(batch_results[0]))]
                        else:
                            # Single output: array per instance
                            shap_values = np.vstack(batch_results)
                    
                    # Récupération de la valeur attendue
                    if hasattr(self._shap_explainer, 'expected_value'):
                        expected_value = self._shap_explainer.expected_value
                    else:
                        # Calcul manuel si non disponible
                        try:
                            if isinstance(shap_values, list):
                                expected_value = [np.mean(self._model_predict_wrapper(background_data)) 
                                                 for _ in range(len(shap_values))]
                            else:
                                expected_value = np.mean(self._model_predict_wrapper(background_data))
                        except Exception as e:
                            self._logger.warning(f"Erreur lors du calcul de expected_value: {str(e)}")
                            expected_value = 0
                else:
                    # Traitement standard sans lots
                    with self._maybe_use_gpu_context():
                        shap_values = self._shap_explainer.shap_values(
                            X, l1_reg=self._config.l1_reg, output_index=output_index
                        )
                        expected_value = self._shap_explainer.expected_value if hasattr(self._shap_explainer, 'expected_value') else 0
                
                # Mise en cache des résultats
                if cache_key:
                    if not hasattr(self, '_explanation_cache'):
                        from functools import lru_cache
                        self._explanation_cache = lru_cache(maxsize=self._config.cache_size)(lambda x: x)({})
                    self._explanation_cache[cache_key] = (shap_values, expected_value, compute_timer.duration)
            
            # Métriques de qualité d'explication
            quality_metrics = {}
            if compute_quality:
                try:
                    # Calcul des métriques de qualité
                    quality_metrics = self._compute_explanation_quality(X, y, shap_values, expected_value)
                except Exception as e:
                    self._logger.warning(f"Erreur lors du calcul des métriques de qualité: {str(e)}")
            
            # Vérification de la conformité réglementaire
            compliance_results = None
            if verify_compliance and hasattr(self, '_compliance_checker'):
                try:
                    compliance_results = self._compliance_checker.check_explanation(
                        model=self._model,
                        data=X, 
                        shap_values=shap_values,
                        explainer=self,
                        regulations=regulations
                    )
                except Exception as e:
                    self._logger.warning(f"Erreur lors de la vérification de conformité: {str(e)}")
            
            # Génération des narratives explicatives
            narrative = None
            if generate_narratives:
                try:
                    narrative = self._generate_explanation_narrative(
                        shap_values, X, audience_level=audience_level, language=language
                    )
                except Exception as e:
                    self._logger.warning(f"Erreur lors de la génération de narratives: {str(e)}")
            
            # Extraction des métadonnées complètes
            metadata = self._extract_metadata()
            metadata.update({
                'computation_time_seconds': compute_timer.duration,
                'memory_usage_mb': compute_memory.peak_usage_mb,
                'sample_size': len(X),
                'quality_metrics': quality_metrics,
                'audience_level': audience_level,
                'computation_mode': 'parallel' if self._config.n_jobs > 1 else 'sequential',
                'batch_processing': bool(batch_size and len(X) > batch_size),
                'cache_hit': bool(cache_key and hasattr(self, '_explanation_cache') and cache_key in self._explanation_cache)
            })
            
            # Création du résultat d'explication
            result = self._create_explanation_result(
                shap_values=shap_values,
                expected_value=expected_value,
                X=X,
                y=y if include_predictions else None,
                metadata=metadata,
                compliance=compliance_results,
                narrative=narrative,
                raw_shap=shap_values if return_raw_shap else None
            )
        
        # Conversion des données en format approprié
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X_values = X.values
        else:
            X_values = X
            feature_names = kwargs.get('feature_names', 
                                      [f"feature_{i}" for i in range(X_values.shape[1])])
        
        # Limiter le nombre d'échantillons pour performance si nécessaire
        if X_values.shape[0] > max_samples:
            self._logger.warning(f"Échantillonnage de {max_samples} instances sur {X_values.shape[0]} pour performance.")
            indices = np.random.choice(X_values.shape[0], max_samples, replace=False)
            X_sample = X_values[indices]
        else:
            X_sample = X_values
            
        # Tracer l'action
        self.add_audit_record("explain", {
            "n_samples": X_sample.shape[0],
            "n_features": X_sample.shape[1],
            "audience_level": audience_level.value if isinstance(audience_level, AudienceLevel) else audience_level,
            "include_interaction_values": include_interaction_values,
            "summarize": summarize
        })
        
        try:
            # Calculer les valeurs SHAP
            shap_values = self._shap_explainer.shap_values(X_sample)
            
            # Gérer les différents formats de sortie selon le type d'explainer SHAP
            if isinstance(shap_values, list):
                # Cas multi-classe: une liste de tableaux
                if output_indices is not None:
                    if isinstance(output_indices, int):
                        output_indices = [output_indices]
                    shap_values = [shap_values[i] for i in output_indices]
            
            # Calculer les valeurs d'interaction si demandé
            interaction_values = None
            if include_interaction_values and hasattr(self._shap_explainer, 'shap_interaction_values'):
                try:
                    interaction_values = self._shap_explainer.shap_interaction_values(X_sample)
                except Exception as e:
                    self._logger.warning(f"Impossible de calculer les valeurs d'interaction: {str(e)}")
            
            # Préparer les importances de caractéristiques
            feature_importances = []
            
            # Cas où on résume les résultats pour l'ensemble du dataset
            if summarize:
                if isinstance(shap_values, list):
                    # Moyenne des valeurs absolues pour chaque classe
                    global_importances = np.zeros(X_sample.shape[1])
                    for sv in shap_values:
                        global_importances += np.mean(np.abs(sv), axis=0)
                    global_importances /= len(shap_values)
                else:
                    # Moyenne des valeurs absolues
                    global_importances = np.mean(np.abs(shap_values), axis=0)
                
                # Créer les objets FeatureImportance
                for i, (name, importance) in enumerate(zip(feature_names, global_importances)):
                    feature_importances.append(FeatureImportance(
                        feature_name=name,
                        importance=float(importance),
                        # Ajouter des statistiques si disponibles
                        std_dev=float(np.std(np.abs(shap_values[0][:, i])) if isinstance(shap_values, list) 
                                     else np.std(np.abs(shap_values[:, i])))
                    ))
            
            # Extraire les métadonnées du modèle
            if not self._metadata:
                self._extract_metadata()
            
            # Créer le résultat d'explication
            result = ExplanationResult(
                method=ExplainabilityMethod.SHAP,
                model_metadata=self._metadata,
                feature_importances=feature_importances,
                raw_explanation={
                    "shap_values": shap_values,
                    "interaction_values": interaction_values,
                    "feature_names": feature_names,
                    "data": X_sample
                },
                audience_level=audience_level
            )
            
            return result
            
        except Exception as e:
            self._logger.error(f"Erreur lors du calcul des valeurs SHAP: {str(e)}")
            raise RuntimeError(f"Échec de l'explication SHAP: {str(e)}")
    
    def explain_instance(self, instance, **kwargs) -> ExplanationResult:
        """
        Explique une instance spécifique avec explications SHAP optimisées et enrichies.
        
        Cette méthode implémente une explication avancée d'instance individuelle avec:
        - Optimisations de performance (GPU, cache)
        - Métriques de qualité d'explication
        - Conformité réglementaire
        - Générations de narratives explicatives adaptées à l'audience
        - Métadonnées complètes et traces d'audit
        
        Args:
            instance: Instance à expliquer (array, liste, dict ou pandas.Series)
            **kwargs: Paramètres additionnels
                - output_index: Indice de la sortie à expliquer pour les modèles multi-sorties
                - audience_level: Niveau d'audience (TECHNICAL, BUSINESS, PUBLIC)
                - language: Langue des narratives ('fr', 'en')
                - include_interaction_values: Calculer les valeurs d'interaction entre caractéristiques
                - check_compliance: Vérifier la conformité réglementaire
                - compliance_regs: Liste des réglementations à vérifier
                - compute_metrics: Calculer des métriques de qualité d'explication
                - return_raw_shap: Renvoyer les valeurs SHAP brutes (potentiellement volumineuses)
                - generate_narratives: Générer des descriptions textuelles des explications
                - include_predictions: Inclure les prédictions du modèle dans le résultat
                - feature_names: Noms des caractéristiques (si non disponibles dans les données)
                
        Returns:
            ExplanationResult: Résultat standardisé et enrichi de l'explication
            
        Raises:
            ValueError: Si le format d'instance n'est pas supporté
            RuntimeError: Si l'explication échoue pour des raisons techniques
        """
        # Convertir l'instance en format approprié
        if isinstance(instance, dict):
            # Convertir dict en DataFrame
            instance_df = pd.DataFrame([instance])
            feature_names = list(instance.keys())
        elif isinstance(instance, pd.Series):
            instance_df = pd.DataFrame([instance])
            feature_names = instance.index.tolist()
        elif isinstance(instance, (list, np.ndarray)):
            instance_array = np.array(instance).reshape(1, -1)
            instance_df = pd.DataFrame(instance_array)
            feature_names = kwargs.get('feature_names', 
                                      [f"feature_{i}" for i in range(instance_array.shape[1])])
        else:
            raise ValueError("Format d'instance non supporté. Utilisez un dict, pandas.Series, liste ou numpy.ndarray.")
            
        # Extraction des paramètres avancés
        output_index = kwargs.get('output_index', None)
        audience_level = kwargs.get('audience_level', AudienceLevel.TECHNICAL)
        language = kwargs.get('language', 'fr')
        include_interaction_values = kwargs.get('include_interaction_values', False)
        check_compliance = kwargs.get('check_compliance', self._config.compliance_mode)
        compliance_regs = kwargs.get('compliance_regs', self._config.compliance_regs)
        compute_metrics = kwargs.get('compute_metrics', self._config.compute_metrics)
        return_raw_shap = kwargs.get('return_raw_shap', False)
        generate_narratives = kwargs.get('generate_narratives', True)
        include_predictions = kwargs.get('include_predictions', True)
        
        # Préparation du cache et des métriques de performance
        compute_timer = Timer()
        compute_memory = MemoryTracker()
        cache_key = None
        
        # Création d'une clé de cache unique pour cette instance
        if hasattr(self, '_explanation_cache'):
            try:
                # Génère une clé de hachage unique pour cette instance
                instance_hash = hashlib.md5(pickle.dumps(instance_df)).hexdigest()
                params_hash = hashlib.md5(str({
                    'output_index': output_index,
                    'audience_level': audience_level,
                    'include_interaction_values': include_interaction_values
                }).encode()).hexdigest()
                cache_key = f"instance_{instance_hash}_{params_hash}"
                
                # Vérifie si l'explication est déjà en cache
                if cache_key in self._explanation_cache:
                    self._logger.info("Explication récupérée du cache")
                    cached_result = self._explanation_cache[cache_key]
                    # Mise à jour des métadonnées pour refléter l'utilisation du cache
                    if isinstance(cached_result, ExplanationResult) and hasattr(cached_result, 'metadata'):
                        cached_result.metadata['cache_hit'] = True
                        cached_result.metadata['computation_time_seconds'] = 0.0
                    return cached_result
            except Exception as e:
                self._logger.warning(f"Erreur lors de l'accès au cache: {str(e)}")
                cache_key = None
        
        # Tracer l'action avec paramètres enrichis
        self.add_audit_record("explain_instance", {
            "n_features": len(feature_names),
            "audience_level": audience_level.value if isinstance(audience_level, AudienceLevel) else audience_level,
            "include_interaction_values": include_interaction_values,
            "check_compliance": check_compliance,
            "compute_metrics": compute_metrics,
            "language": language,
            "use_gpu": self._config.use_gpu
        })
        
        compute_timer.start()
        compute_memory.start_tracking()
        
        try:
            # Récupération de la valeur attendue (baseline) de l'explainer
            expected_value = self._shap_explainer.expected_value
            if isinstance(expected_value, np.ndarray):
                expected_value = expected_value.tolist()
            elif isinstance(expected_value, list) and all(isinstance(x, np.ndarray) for x in expected_value):
                expected_value = [x.tolist() for x in expected_value]
                
            # Utilisation du contexte GPU si configuré
            with self._maybe_use_gpu_context():
                # Calculer les valeurs SHAP avec optimisations
                shap_values = self._shap_explainer.shap_values(instance_df)
            
            # Gérer les différents formats de sortie selon le type d'explainer SHAP
            if isinstance(shap_values, list):
                # Cas multi-classe: une liste de tableaux
                if output_index is not None:
                    shap_values = shap_values[output_index]
                else:
                    # Prendre la classe avec la probabilité maximale
                    predictions = self._model_predict_wrapper(instance_df)
                    if predictions.ndim > 1 and predictions.shape[1] > 1:
                        output_index = np.argmax(predictions[0])
                        shap_values = shap_values[output_index]
                    else:
                        # Cas binaire, prendre la classe positive
                        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            # Calcul des valeurs d'interaction si demandé
            interaction_values = None
            if include_interaction_values and hasattr(self._shap_explainer, 'shap_interaction_values'):
                try:
                    with self._maybe_use_gpu_context():
                        interaction_values = self._shap_explainer.shap_interaction_values(instance_df)
                    if isinstance(interaction_values, list) and output_index is not None:
                        interaction_values = interaction_values[output_index]
                except Exception as e:
                    self._logger.warning(f"Impossible de calculer les valeurs d'interaction: {str(e)}")
            
            # Génération des prédictions du modèle si demandé
            y_pred = None
            if include_predictions:
                try:
                    y_pred = self._model_predict_wrapper(instance_df)
                except Exception as e:
                    self._logger.warning(f"Impossible de générer les prédictions: {str(e)}")
            
            # Métriques de qualité d'explication
            quality_metrics = {}
            if compute_metrics:
                try:
                    # Pour une instance unique, nous calculons des métriques spécifiques à l'instance
                    quality_metrics = {
                        'local_fidelity': self._compute_local_fidelity(instance_df, shap_values, expected_value),
                        'feature_sparsity': self._gini_index(np.abs(shap_values)),
                    }
                    
                    # Si des prédictions ont été faites, ajout de métriques supplémentaires
                    if y_pred is not None:
                        quality_metrics['prediction_impact'] = self._compute_prediction_impact(shap_values, expected_value, y_pred)
                except Exception as e:
                    self._logger.warning(f"Erreur lors du calcul des métriques de qualité: {str(e)}")
            
            # Vérification de conformité
            compliance_results = None
            if check_compliance and hasattr(self, '_compliance_checker'):
                try:
                    compliance_results = self._compliance_checker.check_explanation(
                        shap_values=shap_values,
                        data=instance_df,
                        model=self._model,
                        regulations=compliance_regs
                    )
                except Exception as e:
                    self._logger.warning(f"Erreur lors de la vérification de conformité: {str(e)}")
            
            # Génération des narratives explicatives
            narrative = None
            if generate_narratives:
                try:
                    narrative = self._generate_explanation_narrative(
                        shap_values, instance_df, audience_level=audience_level, language=language
                    )
                except Exception as e:
                    self._logger.warning(f"Erreur lors de la génération de narratives: {str(e)}")
            
            # Arrêt du minuteur et du suivi mémoire
            compute_timer.stop()
            compute_memory.stop_tracking()
            
            # Extraction des métadonnées complètes
            metadata = self._extract_metadata()
            metadata.update({
                'computation_time_seconds': compute_timer.duration,
                'memory_usage_mb': compute_memory.peak_usage_mb,
                'quality_metrics': quality_metrics,
                'audience_level': audience_level,
                'computation_mode': 'gpu' if self._config.use_gpu else 'cpu',
                'cache_hit': False,
                'is_single_instance': True
            })
            
            # Créer le résultat d'explication enrichi en utilisant notre méthode avancée
            result = self._create_explanation_result(
                shap_values=shap_values,
                expected_value=expected_value,
                X=instance_df,
                y=y_pred if include_predictions else None,
                metadata=metadata,
                compliance=compliance_results,
                narrative=narrative,
                raw_shap=shap_values if return_raw_shap else None,
                interaction_values=interaction_values if include_interaction_values else None,
                feature_names=feature_names,
                output_index=output_index
            )
            
            # Mise en cache du résultat si possible
            if cache_key and hasattr(self, '_explanation_cache'):
                try:
                    self._explanation_cache[cache_key] = result
                except Exception as e:
                    self._logger.warning(f"Erreur lors de la mise en cache du résultat: {str(e)}")
            
            return result
            
        except Exception as e:
            self._logger.error(f"Erreur lors du calcul des valeurs SHAP pour l'instance: {str(e)}")
            raise RuntimeError(f"Échec de l'explication SHAP pour l'instance: {str(e)}")
    
    def _compute_local_fidelity(self, X, shap_values, expected_value):
        """
        Calcule la fidélité locale de l'explication SHAP pour une instance.
        
        La fidélité locale mesure comment les attributions SHAP expliquent
        effectivement la différence entre la prédiction du modèle et la valeur de référence.
        
        Args:
            X: Instance à expliquer (DataFrame ou ndarray)
            shap_values: Valeurs SHAP calculées
            expected_value: Valeur attendue (baseline)
            
        Returns:
            float: Score de fidélité locale (0-1)
        """
        try:
            # Prédiction du modèle pour l'instance
            prediction = self._model_predict_wrapper(X)
            
            # Format des données selon le type de sortie
            if hasattr(prediction, 'shape') and len(prediction.shape) > 1 and prediction.shape[1] > 1:
                # Classification multi-classes
                if isinstance(shap_values, list):
                    # Vérifier la somme des valeurs SHAP pour chaque classe
                    class_fidelities = []
                    for i, sv in enumerate(shap_values):
                        if sv.ndim > 1:
                            sv_sum = sv[0].sum()
                        else:
                            sv_sum = sv.sum()
                            
                        ev = expected_value[i] if isinstance(expected_value, list) else expected_value
                        pred = prediction[0, i] if prediction.ndim > 1 else prediction[i]
                        
                        # Différence entre la prédiction réelle et la prédiction explicable
                        fidelity = 1.0 - min(1.0, abs(pred - (ev + sv_sum)) / max(0.01, abs(pred)))
                        class_fidelities.append(fidelity)
                    
                    # Moyenne des fidélités par classe
                    return np.mean(class_fidelities)
                else:
                    # Cas binaire avec shap_values pour classe positive
                    if shap_values.ndim > 1:
                        sv_sum = shap_values[0].sum()
                    else:
                        sv_sum = shap_values.sum()
                    
                    ev = expected_value[1] if isinstance(expected_value, list) else expected_value
                    pred = prediction[0, 1] if prediction.ndim > 1 else prediction[1]
                    
                    return 1.0 - min(1.0, abs(pred - (ev + sv_sum)) / max(0.01, abs(pred)))  
            else:
                # Régression ou classification binaire (sortie simple)
                if shap_values.ndim > 1:
                    sv_sum = shap_values[0].sum()
                else:
                    sv_sum = shap_values.sum()
                
                ev = expected_value[0] if isinstance(expected_value, list) else expected_value
                pred = prediction[0] if hasattr(prediction, '__len__') else prediction
                
                return 1.0 - min(1.0, abs(pred - (ev + sv_sum)) / max(0.01, abs(pred)))
        except Exception as e:
            self._logger.warning(f"Erreur lors du calcul de la fidélité locale: {str(e)}")
            return None
            
    def _compute_prediction_impact(self, shap_values, expected_value, prediction):
        """
        Évalue l'impact des attributions sur la prédiction du modèle.
        
        Cette méthode détermine dans quelle mesure les attributions SHAP
        influencent substantiellement la prédiction finale par rapport à la valeur de référence.
        
        Args:
            shap_values: Valeurs SHAP calculées
            expected_value: Valeur attendue (baseline) 
            prediction: Prédiction du modèle
            
        Returns:
            float: Ratio d'impact (0-1) indiquant l'importance relative des attributions 
        """
        try:
            # Gérer les formats de données diverses
            if isinstance(shap_values, list):
                # Classification multi-classes
                impacts = []
                for i, sv in enumerate(shap_values):
                    sv_sum = np.abs(sv).sum() if sv.ndim == 1 else np.abs(sv[0]).sum()
                    pred_val = prediction[0, i] if prediction.ndim > 1 else prediction[i]
                    baseline = expected_value[i] if isinstance(expected_value, list) else expected_value
                    
                    # Impact normalisé: contrib des attributions / écart total à la baseline
                    impacts.append(sv_sum / max(0.001, abs(pred_val - baseline) + sv_sum))
                return np.mean(impacts)
            else:
                # Régression ou classification binaire
                sv_sum = np.abs(shap_values).sum() if shap_values.ndim == 1 else np.abs(shap_values[0]).sum()
                pred_val = prediction[0] if hasattr(prediction, '__len__') else prediction
                baseline = expected_value[0] if isinstance(expected_value, list) else expected_value
                
                return sv_sum / max(0.001, abs(pred_val - baseline) + sv_sum)
        except Exception as e:
            self._logger.warning(f"Erreur lors du calcul de l'impact prédictif: {str(e)}")
            return None
            
    def _compute_explanation_quality(self, X, y, shap_values, expected_value):
        """
        Calcule des métriques avancées de qualité pour évaluer la fiabilité des explications.
        
        Cette méthode implémente un ensemble de métriques pour quantifier divers aspects 
        de la qualité d'explication tels que:
        - Fidélité: Comment les valeurs SHAP prédisent correctement les sorties du modèle
        - Stabilité: Cohérence des explications sur des variations minimales des données
        - Sparsité: Concentration des attributions sur un sous-ensemble de caractéristiques
        
        Args:
            X: Données d'entrée
            y: Valeurs cibles réelles ou prédites
            shap_values: Valeurs SHAP calculées
            expected_value: Valeur attendue du modèle
            
        Returns:
            dict: Dictionnaire de métriques de qualité avec leurs valeurs
        """
        metrics = {}
        
        try:
            # Conversion en numpy pour calculs homogènes
            X_np = X.values if hasattr(X, 'values') else np.array(X)
            
            # 1. Métrique de fidélité (corrélation prédiction-explication)
            if isinstance(shap_values, list):
                # Multi-class case
                predictions = self._model_predict_wrapper(X)
                if predictions.ndim > 1 and predictions.shape[1] > 1:
                    # Prendre la classe maximale pour chaque prédiction
                    pred_class = np.argmax(predictions, axis=1)
                    # Extraire les valeurs SHAP correspondantes à la classe prédite
                    class_shap = np.array([shap_values[pred_class[i]][i].sum() for i in range(len(X_np))])
                    
                    # Calculer la corrélation
                    if hasattr(y, 'values'):
                        y_np = y.values
                    else:
                        y_np = np.array(y) if y is not None else pred_class
                    
                    fidelity_score = np.corrcoef(class_shap, predictions.max(axis=1))[0, 1]
                    metrics['fidelity_correlation'] = float(fidelity_score)
            else:
                # Regression/binary case
                predictions = self._model_predict_wrapper(X)
                if predictions.ndim > 1 and predictions.shape[1] > 1:
                    # Binary with probability output
                    predictions = predictions[:, 1]  # Take positive class probability
                
                # Sum des valeurs SHAP + expected value devrait être proche de la prédiction
                if isinstance(shap_values, np.ndarray):
                    shap_sum = shap_values.sum(axis=1)
                    if isinstance(expected_value, (list, np.ndarray)):
                        expected_scalar = expected_value[0] if len(expected_value) > 0 else 0
                    else:
                        expected_scalar = expected_value
                    
                    shap_predictions = shap_sum + expected_scalar
                    fidelity_score = np.corrcoef(shap_predictions.flatten(), predictions.flatten())[0, 1]
                    metrics['fidelity_correlation'] = float(fidelity_score)
                    
                    # RMSE entre prédictions et reconstruction SHAP
                    fidelity_rmse = np.sqrt(np.mean((shap_predictions.flatten() - predictions.flatten()) ** 2))
                    metrics['fidelity_rmse'] = float(fidelity_rmse)
            
            # 2. Métrique de stabilité (variance des explications)
            if isinstance(shap_values, list):
                avg_variance = np.mean([np.var(sv, axis=0).mean() for sv in shap_values])
            else:
                avg_variance = np.var(shap_values, axis=0).mean()
            metrics['stability_variance'] = float(avg_variance)
            
            # 3. Métrique de sparsité (concentration des attributions)
            if isinstance(shap_values, list):
                avg_gini = np.mean([self._gini_index(np.abs(sv).mean(axis=0)) for sv in shap_values])
            else:
                feature_importance = np.abs(shap_values).mean(axis=0)
                avg_gini = self._gini_index(feature_importance)
            metrics['sparsity_gini'] = float(avg_gini)
            
            # 4. Proportion d'attributions nulles ou négligeables
            threshold = 0.01 * (shap_values[0].max() if isinstance(shap_values, list) else shap_values.max())
            if isinstance(shap_values, list):
                avg_sparsity = np.mean([np.mean(np.abs(sv) < threshold) for sv in shap_values])
            else:
                avg_sparsity = np.mean(np.abs(shap_values) < threshold)
            metrics['feature_sparsity'] = float(avg_sparsity)
        
        except Exception as e:
            self._logger.warning(f"Erreur lors du calcul des métriques de qualité: {str(e)}")
        
        return metrics
            
    def _generate_explanation_narrative(self, shap_values, X, audience_level=AudienceLevel.TECHNICAL, language='fr'):
        """
        Génère des descriptions textuelles des explications SHAP adaptées au niveau d'audience.
        
        Cette méthode traduit les valeurs SHAP complexes en narratives explicatives claires
        qui s'adaptent automatiquement au niveau de technicité requis par l'audience.
        
        Args:
            shap_values: Les valeurs SHAP calculées
            X: Données expliquées
            audience_level: Niveau d'audience ('TECHNICAL', 'BUSINESS', 'PUBLIC')
            language: Langue des narratives ('fr' ou 'en')
            
        Returns:
            dict: Narratives explicatives structurées par niveau et fonction
        """
        narratives = {}
        feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
        
        try:
            # Calcul des importances globales moyennes
            if isinstance(shap_values, list):
                # Multi-class
                global_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
                classes = len(shap_values)
            else:
                # Binary/Regression
                global_importance = np.abs(shap_values).mean(axis=0)
                classes = 1
            
            # Tri des caractéristiques par importance
            sorted_idx = np.argsort(-global_importance)
            top_features = [feature_names[idx] for idx in sorted_idx[:5]]  # Top 5 features
            top_importance = global_importance[sorted_idx[:5]]
            top_importance_norm = top_importance / top_importance.sum() * 100  # Normalisation en %
            
            # Génération des narratives selon le niveau d'audience
            if language == 'fr':
                if audience_level == AudienceLevel.TECHNICAL:
                    # Version technique détaillée
                    narratives['summary'] = f"L'analyse SHAP a identifié {len(feature_names)} variables explicatives pour ce modèle. "
                    narratives['main_drivers'] = f"Les principaux facteurs contributifs sont {', '.join(top_features[:3])}, "
                    narratives['main_drivers'] += f"avec des importances relatives de {', '.join([f'{v:.1f}%' for v in top_importance_norm[:3]])}."
                    
                    # Détails techniques
                    narratives['technical_details'] = f"L'indice de Gini des attributions SHAP est de {self._gini_index(global_importance):.3f}, "
                    narratives['technical_details'] += "indiquant "  
                    if self._gini_index(global_importance) > 0.7:
                        narratives['technical_details'] += "une forte concentration des effets sur un petit nombre de variables."
                    elif self._gini_index(global_importance) > 0.4:
                        narratives['technical_details'] += "une distribution moyennement concentrée des effets sur les variables."
                    else:
                        narratives['technical_details'] += "une distribution relativement équilibrée des effets sur l'ensemble des variables."
                    
                    # Interactions (si disponible)
                    if hasattr(self, '_last_interaction_values') and self._last_interaction_values is not None:
                        narratives['interactions'] = "Des effets d'interaction significatifs ont été détectés entre certaines variables."
                    
                elif audience_level == AudienceLevel.BUSINESS:
                    # Version intermédiaire orientée métier
                    narratives['summary'] = f"Ce modèle utilise {len(feature_names)} variables pour établir ses prédictions."
                    narratives['main_drivers'] = f"Les 3 facteurs les plus influents sont {', '.join(top_features[:3])}."
                    narratives['business_impact'] = "Ces facteurs représentent "
                    total_pct = sum(top_importance_norm[:3])
                    narratives['business_impact'] += f"ensemble {total_pct:.1f}% de l'impact total sur les décisions du modèle."
                    
                    # Recommandation métier
                    if total_pct > 70:
                        narratives['recommendation'] = "Recommandation: Concentrez votre attention sur ces facteurs clés qui dominent la décision."
                    else:
                        narratives['recommendation'] = "Recommandation: Tenez compte de l'ensemble des facteurs qui contribuent de manière significative aux décisions."
                
                else:  # PUBLIC
                    # Version simplifiée grand public
                    narratives['summary'] = "Voici une explication simplifiée des résultats de ce modèle."
                    narratives['main_factors'] = f"Les principaux éléments qui ont influencé ce résultat sont {', '.join(top_features[:3])}."
                    if len(top_features) > 3:
                        narratives['additional_info'] = f"D'autres facteurs comme {', '.join(top_features[3:5])} ont également joué un rôle, mais moins important."
            else:  # English
                if audience_level == AudienceLevel.TECHNICAL:
                    # Technical detailed version
                    narratives['summary'] = f"SHAP analysis identified {len(feature_names)} explanatory variables for this model. "
                    narratives['main_drivers'] = f"The main contributing factors are {', '.join(top_features[:3])}, "
                    narratives['main_drivers'] += f"with relative importances of {', '.join([f'{v:.1f}%' for v in top_importance_norm[:3]])}."
                    
                    # Technical details
                    narratives['technical_details'] = f"The Gini index of SHAP attributions is {self._gini_index(global_importance):.3f}, "
                    narratives['technical_details'] += "indicating "  
                    if self._gini_index(global_importance) > 0.7:
                        narratives['technical_details'] += "a high concentration of effects on a small number of variables."
                    elif self._gini_index(global_importance) > 0.4:
                        narratives['technical_details'] += "a moderately concentrated distribution of effects across variables."
                    else:
                        narratives['technical_details'] += "a relatively balanced distribution of effects across all variables."
                    
                    # Interactions (if available)
                    if hasattr(self, '_last_interaction_values') and self._last_interaction_values is not None:
                        narratives['interactions'] = "Significant interaction effects were detected between certain variables."
                    
                elif audience_level == AudienceLevel.BUSINESS:
                    # Intermediate business-oriented version
                    narratives['summary'] = f"This model uses {len(feature_names)} variables to make its predictions."
                    narratives['main_drivers'] = f"The 3 most influential factors are {', '.join(top_features[:3])}."
                    narratives['business_impact'] = "These factors together represent "
                    total_pct = sum(top_importance_norm[:3])
                    narratives['business_impact'] += f"{total_pct:.1f}% of the total impact on model decisions."
                    
                    # Business recommendation
                    if total_pct > 70:
                        narratives['recommendation'] = "Recommendation: Focus your attention on these key factors that dominate the decision."
                    else:
                        narratives['recommendation'] = "Recommendation: Consider all factors that contribute significantly to decisions."
                
                else:  # PUBLIC
                    # Simplified public version
                    narratives['summary'] = "Here is a simplified explanation of this model's results."
                    narratives['main_factors'] = f"The main elements that influenced this result are {', '.join(top_features[:3])}."
                    if len(top_features) > 3:
                        narratives['additional_info'] = f"Other factors like {', '.join(top_features[3:5])} also played a role, but less important."
            
            # Ajout d'informations spécifiques à l'instance
            # Valeurs des features importantes pour cette prédiction spécifique
            instance_values = {}
            for i, feature in enumerate(top_features[:3]):
                idx = feature_names.index(feature)
                value = X.iloc[0, idx] if hasattr(X, 'iloc') else X[0, idx]
                instance_values[feature] = value
                
            if language == 'fr':
                narratives['instance_specific'] = "Pour cette prédiction spécifique, les valeurs clés sont: "
            else:  # English
                narratives['instance_specific'] = "For this specific prediction, the key values are: "
                
            instance_details = [f"{feature}: {value}" for feature, value in instance_values.items()]
            narratives['instance_specific'] += ", ".join(instance_details)
            
        except Exception as e:
            self._logger.warning(f"Erreur lors de la génération des narratives: {str(e)}")
            if language == 'fr':
                narratives['error'] = "Impossible de générer une explication narrative complète."
            else:  # English
                narratives['error'] = "Unable to generate a complete narrative explanation."
        
        return narratives
        
    def _gini_index(self, array):
        """
        Calcule l'indice de Gini pour mesurer l'inégalité des attributions.
        Un Gini proche de 1 indique une forte concentration (attributions inégales),
        proche de 0 indique des attributions uniformes.
        """
        # Assurer que l'array est 1D et non négatif
        if array.ndim > 1:
            array = np.abs(array).mean(axis=0)
        else:
            array = np.abs(array)
            
        # Trier les valeurs
        array = np.sort(array)
        n = array.size
        index = np.arange(1, n + 1)
        
        # Calculer l'indice de Gini
        return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))
    
    def _generate_explanation_narrative(self, shap_values, X, audience_level='TECHNICAL', language='fr'):
        """
        Génère des descriptions textuelles des explications SHAP adaptées au niveau d'audience.
        
        Cette méthode traduit les valeurs SHAP complexes en narratives explicatives claires
        qui s'adaptent automatiquement au niveau de technicité requis par l'audience.
        
        Args:
            shap_values: Les valeurs SHAP calculées
            X: Données expliquées
            audience_level: Niveau d'audience ('TECHNICAL', 'BUSINESS', 'PUBLIC')
            language: Langue des narratives ('fr' ou 'en')
            
        Returns:
            dict: Narratives explicatives structurées par niveau et fonction
        """
        narratives = {}
        feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
        
        try:
            # Calcul des importances globales moyennes
            if isinstance(shap_values, list):
                # Multi-class
                global_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
                classes = len(shap_values)
            else:
                # Binary/Regression
                global_importance = np.abs(shap_values).mean(axis=0)
                classes = 1
            
            # Tri des caractéristiques par importance
            sorted_idx = np.argsort(-global_importance)
            top_features = [feature_names[idx] for idx in sorted_idx[:5]]  # Top 5 features
            top_importance = global_importance[sorted_idx[:5]]
            top_importance_norm = top_importance / top_importance.sum() * 100  # Normalisation en %
            
            # Génération des narratives selon le niveau d'audience
            if language == 'fr':
                # Français
                if audience_level == 'TECHNICAL':
                    # Narrative technique détaillée
                    gini_value = self._gini_index(global_importance)
                    concentration_type = "forte concentration" if gini_value > 0.6 else "répartition équilibrée"
                    model_type_str = 'multi-classes' if classes > 1 else 'binaire/régression'
                    summary = (
                        f"L'analyse SHAP a identifié {len(feature_names)} caractéristiques dont les "
                        f"principales sont {', '.join(top_features[:3])} représentant "
                        f"{top_importance_norm[:3].sum():.1f}% "
                        f"Le modèle {model_type_str} "
                        f"présente une distribution d'attributions avec un indice de Gini "
                        f"de {gini_value:.3f}, indiquant une "
                        f"{concentration_type} "
                        f"de l'influence prédictive."
                    )
                elif audience_level == 'BUSINESS':
                    # Narrative simplifiée pour business
                    summary = (
                        f"Les facteurs principaux influençant les prédictions sont "
                        f"{', '.join(top_features[:3])} avec un impact respectif de "
                        f"{', '.join([f'{v:.1f}%' for v in top_importance_norm[:3]])}. "
                        f"Le modèle s'appuie {'largement sur un petit nombre de facteurs clés' if self._gini_index(global_importance) > 0.6 else 'sur un ensemble diversifié de facteurs'} "
                        f"pour établir ses prédictions."
                    )
                else:  # 'PUBLIC'
                    # Narrative très simplifiée pour le grand public
                    summary = (
                        f"Les éléments les plus importants dans cette décision sont "
                        f"{', '.join(top_features[:3])}, "
                        f"qui représentent ensemble {top_importance_norm[:3].sum():.0f}% "
                        f"de l'influence sur le résultat."
                    )
            else:
                # Anglais
                if audience_level == 'TECHNICAL':
                    gini_value = self._gini_index(global_importance)
                    model_type_str = "multi-class" if classes > 1 else "binary/regression"
                    concentration_str = "high concentration" if gini_value > 0.6 else "balanced distribution"
                    summary = (
                        f"SHAP analysis identified {len(feature_names)} features with "
                        f"the top drivers being {', '.join(top_features[:3])} representing "
                        f"{top_importance_norm[:3].sum():.1f}% of explanatory impact. "
                        f"The {model_type_str} model "
                        f"displays an attribution distribution with Gini index "
                        f"of {gini_value:.3f}, indicating "
                        f"{concentration_str} "
                        f"of predictive influence."
                    )
                elif audience_level == 'BUSINESS':
                    gini_value = self._gini_index(global_importance)
                    model_strategy = "relies heavily on a small number of key factors" if gini_value > 0.6 else "leverages a diverse set of factors"
                    summary = (
                        f"The main factors influencing predictions are "
                        f"{', '.join(top_features[:3])} with respective impacts of "
                        f"{', '.join([f'{v:.1f}%' for v in top_importance_norm[:3]])}. "
                        f"The model {model_strategy} "
                        f"to establish its predictions."
                    )
                else:  # 'PUBLIC'
                    summary = (
                        f"The most important elements in this decision are "
                        f"{', '.join(top_features[:3])}, "
                        f"which together represent {top_importance_norm[:3].sum():.0f}% "
                        f"of the influence on the outcome."
                    )
                    
            narratives['summary'] = summary
            
            # Ajouter d'autres narratives spécialisées selon le besoin
            # Par exemple: analyses contrefactuelles, alertes sur des biais potentiels, etc.
            
        except Exception as e:
            self._logger.warning(f"Erreur lors de la génération des narratives: {str(e)}")
            # Narrative de secours en cas d'échec
            narratives['summary'] = f"Analyse SHAP effectuée sur {len(feature_names)} caractéristiques."
        
        return narratives
    
    def _create_explanation_result(self, shap_values, expected_value, X, metadata, y=None,
                                 compliance=None, narrative=None, raw_shap=None, **kwargs):
        """
        Crée un objet ExplanationResult enrichi avec métriques de qualité et conformité.
        
        Cette implémentation avancée génère un résultat d'explication complet avec données
        structurées pour tous les aspects de l'explicabilité: attributions, métriques de qualité,
        conformité réglementaire, narratives adaptées à différentes audiences, et métadonnées 
        exhaustives pour la traçabilité et l'audit.
        
        Args:
            shap_values: Valeurs SHAP calculées (par instance et par caractéristique)
            expected_value: Valeur attendue (baseline) du modèle
            X: Données d'entrée expliquées
            metadata: Métadonnées de l'explication et du processus
            y: Valeurs réelles ou prédites (optionnel)
            compliance: Résultats de conformité réglementaire (optionnel)
            narrative: Descriptions textuelles des explications (optionnel)
            raw_shap: Valeurs SHAP brutes pour analyses avancées (optionnel)
            **kwargs: Paramètres additionnels
                feature_names: Noms des caractéristiques
                class_names: Noms des classes pour classification
                interaction_values: Valeurs d'interaction entre caractéristiques
                output_indices: Indices des sorties expliquées
                
        Returns:
            ExplanationResult: Résultat complet et enrichi de l'explication
        """
        # Détermine les noms des caractéristiques
        feature_names = kwargs.get('feature_names')
        if feature_names is None:
            if hasattr(X, 'columns'):
                feature_names = X.columns.tolist()
            else:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Détermine les noms des classes
        class_names = kwargs.get('class_names')
        if class_names is None:
            if isinstance(shap_values, list):
                class_names = [f"class_{i}" for i in range(len(shap_values))]
        
        # Calcul de l'importance globale des caractéristiques
        if isinstance(shap_values, list):
            # Multi-class
            importance_values = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            # Binary/Regression
            importance_values = np.abs(shap_values).mean(axis=0)
        
        # Création des objets FeatureImportance
        feature_importances = []
        for i, (name, value) in enumerate(zip(feature_names, importance_values)):
            feature_importances.append(FeatureImportance(
                feature_name=name,
                importance_value=float(value),
                importance_rank=i + 1,
                direction="positive" if value >= 0 else "negative"
            ))
        
        # Tri par importance décroissante
        feature_importances.sort(key=lambda x: x.importance_value, reverse=True)
        
        # Prépare les métriques de qualité
        quality_metrics = metadata.get('quality_metrics', {})
        explanation_quality = None
        if quality_metrics:
            explanation_quality = ExplanationQuality(
                fidelity=quality_metrics.get('fidelity_correlation', None),
                stability=1.0 - min(1.0, quality_metrics.get('stability_variance', 0)),
                sparsity=quality_metrics.get('feature_sparsity', None),
                consistency=None  # À implémenter si disponible
            )
        
        # Création de l'objet ExplanationResult
        result = ExplanationResult(
            explanation_method=ExplainabilityMethod.SHAP,
            feature_importances=feature_importances,
            instance_explanations=self._create_instance_explanations(shap_values, X, expected_value),
            global_explanation={
                'expected_value': expected_value,
                'feature_importance_mean': dict(zip(feature_names, importance_values.tolist())),
                'feature_importance_std': dict(zip(feature_names, np.std([np.abs(sv) for sv in shap_values], axis=0).tolist())) 
                if isinstance(shap_values, list) else dict(zip(feature_names, np.std(np.abs(shap_values), axis=0).tolist()))
            },
            metadata=metadata,
            quality=explanation_quality,
            target_values=y.tolist() if y is not None and hasattr(y, 'tolist') else y,
            raw_data={
                'raw_shap_values': raw_shap,
                'compliance_results': compliance,
                'narrative': narrative
            } if raw_shap is not None or compliance is not None or narrative is not None else None,
            visualizations=None  # Les visualisations seront générées séparément
        )
        
        return result
    
    def _create_instance_explanations(self, shap_values, X, expected_value):
        """
        Crée des explications détaillées pour chaque instance expliquée.
        
        Args:
            shap_values: Valeurs SHAP calculées
            X: Données d'entrée
            expected_value: Valeur de base (expected value)
            
        Returns:
            list: Liste des explications par instance
        """
        instance_explanations = []
        feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f"feature_{i}" for i in range(X.shape[1])]
        
        # Traitement selon le type de sortie (classification ou régression)
        if isinstance(shap_values, list):
            # Classification multi-classes
            for instance_idx in range(shap_values[0].shape[0]):
                instance_exp = {}
                for class_idx, sv in enumerate(shap_values):
                    class_exp = {
                        'base_value': float(expected_value[class_idx]) if isinstance(expected_value, (list, np.ndarray)) else float(expected_value),
                        'output_value': float(expected_value[class_idx] + sv[instance_idx].sum()) 
                            if isinstance(expected_value, (list, np.ndarray)) else float(expected_value + sv[instance_idx].sum()),
                        'features': {}
                    }
                    
                    # Détails des contributions par caractéristique
                    for j, fname in enumerate(feature_names):
                        class_exp['features'][fname] = {
                            'contribution': float(sv[instance_idx, j]),
                            'value': float(X.iloc[instance_idx, j]) if hasattr(X, 'iloc') else float(X[instance_idx, j])
                        }
                    
                    instance_exp[f"class_{class_idx}"] = class_exp
                instance_explanations.append(instance_exp)
        else:
            # Régression ou classification binaire
            for instance_idx in range(shap_values.shape[0]):
                instance_exp = {
                    'base_value': float(expected_value) if isinstance(expected_value, (int, float)) else float(expected_value[0]),
                    'output_value': float(expected_value + shap_values[instance_idx].sum()) 
                        if isinstance(expected_value, (int, float)) else float(expected_value[0] + shap_values[instance_idx].sum()),
                    'features': {}
                }
                
                # Détails des contributions par caractéristique
                for j, fname in enumerate(feature_names):
                    instance_exp['features'][fname] = {
                        'contribution': float(shap_values[instance_idx, j]),
                        'value': float(X.iloc[instance_idx, j]) if hasattr(X, 'iloc') else float(X[instance_idx, j])
                    }
                    
                instance_explanations.append(instance_exp)
        
        return instance_explanations
                
    def _extract_metadata(self):
        """
        Extrait les métadonnées de l'explainer et du modèle.
        
        Returns:
            dict: Métadonnées structurées
        """
        model_type = self._detect_model_type()
        framework = model_type.split('-')[0] if '-' in model_type else model_type
        
        # Déterminer le type de modèle (classification ou régression)
        is_classifier = False
        if hasattr(self._model, 'predict_proba'):
            is_classifier = True
        elif hasattr(self._model, '_estimator_type') and self._model._estimator_type == 'classifier':
            is_classifier = True
        elif model_type in ['tensorflow', 'pytorch']:
            # Pour les modèles deep learning, vérifier la forme de sortie
            try:
                # Utiliser les données d'arrière-plan pour une inférence
                bg_data = self._get_background_data()
                sample = bg_data[0:1]
                preds = self._model_predict_wrapper(sample)
                
                # Si la sortie a plusieurs dimensions et la dernière dimension > 1, c'est probablement une classification
                if preds.ndim > 1 and preds.shape[-1] > 1:
                    is_classifier = True
            except:
                pass
        
        # Créer les métadonnées
        self._metadata = ModelMetadata(
            model_type="classification" if is_classifier else "regression",
            framework=framework,
            input_shape=self._infer_model_input_shape(),
            output_shape=None,  # À compléter si nécessaire
            feature_names=getattr(self, '_feature_names', None),
            target_names=None,  # À compléter si disponible
            model_params={},
            model_version="1.0.0"
        )
