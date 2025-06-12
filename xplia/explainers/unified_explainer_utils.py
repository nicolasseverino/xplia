"""
Fonctions utilitaires avancées pour l'UnifiedExplainer de XPLIA
===============================================================

Ce module fournit les méthodes auxiliaires nécessaires au bon fonctionnement
de l'UnifiedExplainer, notamment la détection de framework, la sélection optimale
des méthodes d'explicabilité, la gestion de conformité, et les utilitaires d'évaluation.
"""
import inspect
import logging
import numpy as np
from typing import Dict, List, Any, Union, Optional, Tuple, Set

from ..core.base import ExplainabilityMethod, AudienceLevel

logger = logging.getLogger(__name__)

def detect_model_framework(model: Any) -> str:
    """
    Détecte automatiquement le framework du modèle (scikit-learn, TensorFlow, PyTorch, etc.)
    
    Args:
        model: Le modèle à analyser
        
    Returns:
        str: Identifiant du framework ('sklearn', 'tensorflow', 'pytorch', 'xgboost', etc.)
    """
    module_name = model.__class__.__module__.split('.')[0].lower()
    
    framework_mapping = {
        'sklearn': 'sklearn',
        'torch': 'pytorch',
        'tensorflow': 'tensorflow',
        'tf': 'tensorflow',
        'keras': 'tensorflow',
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
        'catboost': 'catboost'
    }
    
    for key, value in framework_mapping.items():
        if key in module_name:
            return value
    
    # Détection avancée par l'analyse des méthodes/attributs
    if hasattr(model, 'predict_proba') and hasattr(model, 'feature_importances_'):
        return 'sklearn-like'
    elif hasattr(model, 'forward') and hasattr(model, 'parameters'):
        return 'pytorch-like'
    elif hasattr(model, 'predict') and hasattr(model, 'layers'):
        return 'tensorflow-like'
    
    return 'unknown'

def detect_is_classifier(model: Any) -> bool:
    """
    Détecte si un modèle est un classificateur ou un régresseur
    
    Args:
        model: Le modèle à analyser
        
    Returns:
        bool: True si classificateur, False si régresseur
    """
    # Analyse du nom de la classe
    class_name = model.__class__.__name__.lower()
    if 'classifier' in class_name:
        return True
    elif 'regressor' in class_name:
        return False
    
    # Analyse des méthodes disponibles
    if hasattr(model, 'predict_proba'):
        return True
    
    # Pour les modèles Tensorflow/Keras
    if hasattr(model, 'output_shape'):
        # Classificateur si sortie multi-classes ou activation softmax/sigmoid
        try:
            last_layer = model.layers[-1]
            if hasattr(last_layer, 'activation'):
                if last_layer.activation.__name__ in ['softmax', 'sigmoid']:
                    return True
        except (AttributeError, IndexError):
            pass
    
    # Par défaut, on suppose régresseur (supposition conservatrice)
    return False

def get_supported_model_types() -> List[str]:
    """
    Renvoie la liste exhaustive des types de modèles supportés par l'UnifiedExplainer
    
    Returns:
        List[str]: Liste des types de modèles supportés
    """
    return [
        # scikit-learn
        'RandomForestClassifier', 'RandomForestRegressor',
        'GradientBoostingClassifier', 'GradientBoostingRegressor',
        'LogisticRegression', 'LinearRegression',
        'SVC', 'SVR',
        'DecisionTreeClassifier', 'DecisionTreeRegressor',
        'KNeighborsClassifier', 'KNeighborsRegressor',
        
        # XGBoost
        'XGBClassifier', 'XGBRegressor', 'Booster',
        
        # LightGBM
        'LGBMClassifier', 'LGBMRegressor',
        
        # CatBoost
        'CatBoostClassifier', 'CatBoostRegressor',
        
        # TensorFlow/Keras
        'Sequential', 'Model', 'Functional',
        
        # PyTorch
        'Module', 'Sequential',
        
        # Ensemble et pipeline
        'VotingClassifier', 'VotingRegressor',
        'StackingClassifier', 'StackingRegressor',
        'BaggingClassifier', 'BaggingRegressor',
        'Pipeline',
        
        # Autres
        'TabularPredictor',  # AutoGluon
        'AutoMLPredictor',   # Auto-Sklearn
        'H2OEstimator'       # H2O
    ]

def setup_compliance(compliance_requirements: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Configure et initialise les exigences de conformité réglementaire
    
    Args:
        compliance_requirements: Liste des réglementations à suivre ('rgpd', 'ai_act', 'hipaa', etc.)
        
    Returns:
        Dict[str, Dict[str, Any]]: Configuration des exigences de conformité
    """
    compliance_config = {}
    
    for req in compliance_requirements:
        if req.lower() == 'rgpd' or req.lower() == 'gdpr':
            compliance_config['rgpd'] = {
                'enabled': True,
                'explanation_right': True,
                'data_minimization': True,
                'audit_trail': True,
                'user_access': True
            }
        elif req.lower() == 'ai_act':
            compliance_config['ai_act'] = {
                'enabled': True,
                'risk_category': 'high',
                'documentation': True,
                'human_oversight': True,
                'technical_robustness': True,
                'record_keeping': True,
                'transparency': True
            }
        elif req.lower() == 'hipaa':
            compliance_config['hipaa'] = {
                'enabled': True,
                'phi_protection': True,
                'access_logs': True,
                'authorizations': True
            }
    
    return compliance_config

def initialize_weights(weights: Dict[ExplainabilityMethod, float], 
                     methods: List[ExplainabilityMethod],
                     model_type: str) -> Dict[ExplainabilityMethod, float]:
    """
    Initialise ou ajuste les poids des différentes méthodes d'explicabilité
    
    Args:
        weights: Dictionnaire de poids fourni par l'utilisateur
        methods: Liste des méthodes d'explicabilité utilisées
        model_type: Type de modèle ('sklearn', 'tensorflow', etc.)
        
    Returns:
        Dict[ExplainabilityMethod, float]: Poids optimisés pour les méthodes
    """
    if weights and all(method in weights for method in methods):
        # Normalisation des poids fournis
        total = sum(weights.values())
        return {method: weight / total for method, weight in weights.items()}
    
    # Poids par défaut selon le type de modèle
    default_weights = {
        # Modèles basés sur les arbres
        'tree_based': {
            ExplainabilityMethod.SHAP: 0.35,
            ExplainabilityMethod.FEATURE_IMPORTANCE: 0.25,
            ExplainabilityMethod.LIME: 0.15,
            ExplainabilityMethod.COUNTERFACTUAL: 0.15,
            ExplainabilityMethod.PARTIAL_DEPENDENCE: 0.1
        },
        # Modèles linéaires
        'linear': {
            ExplainabilityMethod.FEATURE_IMPORTANCE: 0.4,
            ExplainabilityMethod.SHAP: 0.25,
            ExplainabilityMethod.LIME: 0.2,
            ExplainabilityMethod.COUNTERFACTUAL: 0.15
        },
        # Réseaux de neurones
        'neural_network': {
            ExplainabilityMethod.SHAP: 0.3,
            ExplainabilityMethod.LIME: 0.25,
            ExplainabilityMethod.ATTENTION: 0.2,
            ExplainabilityMethod.COUNTERFACTUAL: 0.15,
            ExplainabilityMethod.GRADIENT: 0.1
        }
    }
    
    # Sélection du profil de poids approprié
    if 'tree' in model_type or model_type in ['xgboost', 'lightgbm', 'catboost']:
        weights_profile = default_weights['tree_based']
    elif 'linear' in model_type:
        weights_profile = default_weights['linear']
    elif 'tensorflow' in model_type or 'pytorch' in model_type:
        weights_profile = default_weights['neural_network']
    else:
        # Profil générique
        weights_profile = default_weights['tree_based']
    
    # Filtrer pour ne garder que les méthodes disponibles
    result_weights = {method: weights_profile.get(method, 0.1) for method in methods}
    
    # Normalisation
    total = sum(result_weights.values())
    return {method: weight / total for method, weight in result_weights.items()}

def select_optimal_methods(methods: Optional[List[ExplainabilityMethod]], 
                         model_framework: str,
                         is_classifier: bool) -> List[ExplainabilityMethod]:
    """
    Sélectionne les méthodes d'explicabilité optimales selon le modèle
    
    Args:
        methods: Liste des méthodes spécifiées par l'utilisateur (ou None)
        model_framework: Type de framework du modèle
        is_classifier: Si le modèle est un classificateur
        
    Returns:
        List[ExplainabilityMethod]: Méthodes d'explicabilité optimales
    """
    if methods:
        return methods
    
    # Méthodes recommandées selon le type de modèle
    if 'sklearn' in model_framework or model_framework in ['xgboost', 'lightgbm', 'catboost']:
        methods = [
            ExplainabilityMethod.SHAP,
            ExplainabilityMethod.FEATURE_IMPORTANCE,
            ExplainabilityMethod.LIME,
            ExplainabilityMethod.PARTIAL_DEPENDENCE
        ]
        # Ajouter les counterfactuals pour les classificateurs
        if is_classifier:
            methods.append(ExplainabilityMethod.COUNTERFACTUAL)
            
    elif 'tensorflow' in model_framework or 'keras' in model_framework:
        methods = [
            ExplainabilityMethod.SHAP,
            ExplainabilityMethod.LIME,
            ExplainabilityMethod.GRADIENT
        ]
        # Ajouter l'attention pour les modèles qui la supportent
        methods.append(ExplainabilityMethod.ATTENTION)
        
    elif 'pytorch' in model_framework:
        methods = [
            ExplainabilityMethod.SHAP,
            ExplainabilityMethod.LIME,
            ExplainabilityMethod.GRADIENT
        ]
        # Ajouter l'attention pour les modèles qui la supportent
        methods.append(ExplainabilityMethod.ATTENTION)
        
    else:
        # Méthodes génériques qui fonctionnent avec la plupart des modèles
        methods = [
            ExplainabilityMethod.SHAP,
            ExplainabilityMethod.LIME,
            ExplainabilityMethod.FEATURE_IMPORTANCE
        ]
    
    return methods

def load_plugins():
    """
    Charge les plugins d'explicabilité personnalisés
    
    Returns:
        Dict[str, Any]: Dictionnaire des plugins chargés
    """
    try:
        from ..plugins import PluginRegistry
        plugins = PluginRegistry.auto_discover()
        logger.info(f"Plugins d'explicabilité chargés: {len(plugins)}")
        return plugins
    except (ImportError, AttributeError) as e:
        logger.warning(f"Erreur lors du chargement des plugins: {str(e)}")
        return {}
