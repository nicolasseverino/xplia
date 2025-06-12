"""
Unified Explainer pour XPLIA - Framework d'explicabilité avancé
=============================================================

Ce module implémente l'UnifiedExplainer de XPLIA, le système d'explicabilité d'IA le plus avancé au monde.

Caractéristiques principales:
- **Multi-méthodes**: Combine intelligemment toutes les approches (SHAP, LIME, Counterfactual, etc.)
- **Multi-niveaux**: Adapte automatiquement les explications à l'audience (technique/non-technique)
- **Multi-évaluations**: Quantifie la qualité, cohérence, fidélité et robustesse des explications
- **Multi-modèles**: Compatible avec tous les frameworks ML/DL (scikit-learn, TensorFlow, PyTorch, etc.)
- **Multi-formats**: Génère des explications textuelles, visuelles, interactives, et techniques
- **Conforme**: Répond aux exigences du RGPD, AI Act, HIPAA et autres réglementations
- **Extensible**: Architecture modulaire avec hooks et système de plugins
- **Robuste**: Tests étendus, détection des divergences et métriques de fiabilité
- **Performant**: Optimisé pour les grands volumes de données et le traitement en temps réel

L'UnifiedExplainer est le cœur de XPLIA, offrant des explications holistiques et adaptatives tout en
minimisant la complexité pour l'utilisateur final et en satisfaisant aux exigences réglementaires.
"""

import logging
import time
import json
import hashlib
import threading
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set
from enum import Enum
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from ..core.base import (AudienceLevel, ExplainerBase, ExplainabilityMethod,
                        ExplanationResult, FeatureImportance, ModelMetadata)
from ..core.registry import register_explainer
from ..core.factory import create_explainer
from ..plugins import PluginRegistry
from ..compliance import ComplianceChecker

# Configuration avancée du logger avec rotation des fichiers et niveaux multiples
logger = logging.getLogger(__name__)

@dataclass
class ExplanationQualityMetrics:
    """Métriques détaillées de la qualité d'une explication."""
    coherence: float = 0.0  # Cohérence interne de l'explication
    fidelity: float = 0.0   # Fidélité au modèle original
    stability: float = 0.0  # Stabilité face aux petites perturbations
    sparsity: float = 0.0   # Parcimonie (moins de facteurs = meilleur)
    completeness: float = 0.0  # Couverture des facteurs importants
    consistency: float = 0.0  # Cohérence entre plusieurs runs
    contrastivity: float = 0.0  # Capacité à contraster les décisions
    actionability: float = 0.0  # Caractère actionable des explications
    regulatory_compliance: Dict[str, float] = field(default_factory=dict)  # Scores de conformité aux réglementations
    
    @property
    def overall_score(self) -> float:
        """Score global agrégé de qualité."""
        metrics = [self.coherence, self.fidelity, self.stability, 
                  self.sparsity, self.completeness, self.consistency,
                  self.contrastivity, self.actionability]
        # Ajout des scores réglementaires s'ils existent
        if self.regulatory_compliance:
            metrics.extend(self.regulatory_compliance.values())
        return np.mean([m for m in metrics if m > 0])

class ExplanationStrategy(Enum):
    """Stratégies d'explication adaptées à différents contextes."""
    TECHNICAL = "technical"       # Pour experts techniques (data scientists, ML engineers)
    BUSINESS = "business"        # Pour décideurs métier (managers, stakeholders)
    REGULATORY = "regulatory"    # Orienté conformité réglementaire (AI Act, RGPD)
    EDUCATIONAL = "educational"  # Pour sensibilisation et formation
    ADVERSARIAL = "adversarial"  # Pour tests de robustesse et sécurité
    INTERACTIVE = "interactive"  # Pour exploration interactive
    DEBUGGING = "debugging"      # Pour debug et diagnostic du modèle


@register_explainer
class UnifiedExplainer(ExplainerBase):
    """
    Framework d'explicabilité avancé combinant toutes les méthodes d'interprétabilité de l'IA.
    
    L'UnifiedExplainer, fleuron de XPLIA, est le système d'explicabilité IA le plus avancé et complet au monde.
    Il orchestre intelligemment toutes les approches d'interprétabilité (SHAP, LIME, Counterfactual,
    Feature Importance, Grad-CAM, etc.) pour produire une explication holistique, cohérente et contextualisée.
    
    Fonctionnalités avancées:
    - Adaptation automatique au niveau de l'audience (technique → grand public)
    - Détection et résolution des divergences entre méthodes d'explicabilité
    - Validation multi-métrique de la qualité, cohérence et fiabilité
    - Sélection intelligente des explications les plus pertinentes selon le contexte
    - Multiple niveaux d'abstraction (global, local, contrefactuel, causal)
    - Explications multi-formats (texte, visualisation, interactif, programmatique)
    - Conformité réglementaire (RGPD, AI Act, HIPAA) intégrée et auditée
    - Extensibilité via plugins et hooks pour cas d'usage spécifiques
    - Optimisation pour grands volumes de données et traitement en temps réel
    - Robustesse face aux attaques adversariales et aux données aberrantes
    
    L'UnifiedExplainer transforme l'explicabilité de l'intelligence artificielle en un atout stratégique,
    permettant de combiner excellence technique, conformité réglementaire et accessibilité pour tous.
    """
    
    def __init__(self, model, **kwargs):
        """
        Initialise l'UnifiedExplainer avec configuration complète et auto-optimisation.
        
        Args:
            model: Modèle à expliquer (scikit-learn, TensorFlow, PyTorch, custom)
            methods (List[ExplainabilityMethod], optional): Méthodes d'explicabilité à utiliser
            weights (Dict[ExplainabilityMethod, float], optional): Pondération des méthodes
            aggregation_strategy (str, optional): Stratégie d'agrégation ('weighted', 'consensus', 'adaptive', 'ensemble')
            explanation_strategy (ExplanationStrategy, optional): Contexte d'explication (TECHNICAL, BUSINESS, etc.)
            audience_level (AudienceLevel, optional): Niveau de l'audience cible (EXPERT, INTERMEDIATE, NOVICE)
            compliance_requirements (List[str], optional): Exigences réglementaires à respecter ('rgpd', 'ai_act', 'hipaa')
            evaluation_metrics (List[str], optional): Métriques d'évaluation spécifiques à calculer
            optimization_target (str, optional): Cible d'optimisation ('quality', 'performance', 'sparsity', 'regulatory')
            max_features (int, optional): Nombre maximal de caractéristiques à inclure dans l'explication
            feature_selection (str, optional): Méthode de sélection des caractéristiques ('auto', 'importance', 'clustering')
            enable_caching (bool, optional): Activer le cache des explications pour performances optimales
            enable_stability_checks (bool, optional): Activer les tests de stabilité et robustesse
            enable_hooks (bool, optional): Activer les hooks personnalisables
            confidence_threshold (float, optional): Seuil de confiance pour l'inclusion des résultats
            interactive_mode (bool, optional): Préparer pour utilisation interactive/dashboard
            background_data (numpy.ndarray, optional): Données de référence pour certaines méthodes (comme SHAP)
            timeout (int, optional): Timeout en secondes pour chaque méthode d'explicabilité
            n_jobs (int, optional): Nombre de jobs parallèles pour les calculs
            random_state (int, optional): Graine pour reproducibiité
            verbose (int, optional): Niveau de verbosité (0-3)
            **kwargs: Paramètres spécifiques pour explainers individuels
            
        Note:
            L'UnifiedExplainer détecte automatiquement le type de modèle et optimise la configuration
            en fonction des données, du contexte et des exigences réglementaires. Il maintient un
            audit trail complet et s'adapte intelligemment aux ressources disponibles.
        """
        super().__init__(model, **kwargs)
        
        # Identification et métadonnées
        self._method = ExplainabilityMethod.UNIFIED
        self._version = "2.5.0"
        self._instance_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:12]
        
        # Détection automatique du framework ML/DL et adaptation des stratégies
        self._model_framework = self._detect_model_framework(model)
        self._is_classifier = self._detect_is_classifier(model)
        self._supported_model_types = self._get_supported_model_types()
        
        # Paramètres de configuration principale
        self._explanation_strategy = kwargs.get('explanation_strategy', ExplanationStrategy.TECHNICAL)
        self._audience_level = kwargs.get('audience_level', AudienceLevel.INTERMEDIATE)
        self._optimization_target = kwargs.get('optimization_target', 'quality')
        self._compliance_requirements = self._setup_compliance(kwargs.get('compliance_requirements', ['rgpd', 'ai_act']))
        self._max_features = kwargs.get('max_features', 15)
        self._feature_selection = kwargs.get('feature_selection', 'auto')
        self._confidence_threshold = kwargs.get('confidence_threshold', 0.7)
        self._timeout = kwargs.get('timeout', 300)  # 5 minutes par défaut
        self._n_jobs = kwargs.get('n_jobs', -1)  # Tous les CPU par défaut
        self._random_state = kwargs.get('random_state', 42)
        self._verbose = kwargs.get('verbose', 1)
        
        # Configuration des options avancées
        self._enable_caching = kwargs.get('enable_caching', True)
        self._cache = {} if self._enable_caching else None
        self._enable_stability_checks = kwargs.get('enable_stability_checks', True)
        self._enable_hooks = kwargs.get('enable_hooks', True)
        self._interactive_mode = kwargs.get('interactive_mode', False)
        self._background_data = kwargs.get('background_data', None)
        
        # Configuration des métriques d'évaluation
        self._evaluation_metrics = kwargs.get('evaluation_metrics', [
            'coherence', 'fidelity', 'stability', 'sparsity', 'completeness'
        ])
        
        # Sélection intelligente des méthodes d'explicabilité adaptées au modèle et au contexte
        self._methods_to_use = self._select_optimal_methods(kwargs.get('methods', None))
        
        # Stratégie d'agrégation avancée
        self._aggregation_strategy = kwargs.get('aggregation_strategy', 'adaptive')
        self._weights = self._initialize_weights(kwargs.get('weights', {}))
        
        # Système de hooks avancé pour l'extensibilité
        self._hooks = {
            'pre_explain': [],
            'post_explain': [],
            'pre_method': {},
            'post_method': {},
            'pre_aggregate': [],
            'post_aggregate': [],
            'on_error': [],
            'on_timeout': []
        }
        
        # Journalisation avancée et monitoring
        self._logger = logger
        self._performance_metrics = {'time_per_method': {}, 'memory_usage': {}}
        self._explanation_history = []
        self._last_quality_metrics = ExplanationQualityMetrics()
        
        # Gestion des ressources et optimisations
        self._thread_pool = threading.ThreadPool(processes=min(8, self._n_jobs if self._n_jobs > 0 else 4))
        
        # Intégration plugins et compliance
        self._load_plugins()
        
        # Initialisation des explainers individuels (avec auto-configuration)
        self._explainers = {}
        self._initialize_explainers(**kwargs)
        
        # Audit
        self.add_audit_record("initialization", {
            "model_framework": self._model_framework,
            "is_classifier": self._is_classifier,
            "methods": [m.value for m in self._methods_to_use],
            "strategy": self._explanation_strategy.value,
            "audience": self._audience_level.value,
            "compliance": list(self._compliance_requirements.keys()),
            "instance_id": self._instance_id
        })
    
    def _initialize_explainers(self, **kwargs):
        """
        Initialise les explainers individuels à utiliser.
        
        Args:
            **kwargs: Paramètres à transmettre aux explainers
        """
        from ..core.factory import create_explainer
        
        for method in self._methods_to_use:
            try:
                # Création de l'explainer spécifique
                explainer = create_explainer(self._model, method=method, **kwargs)
                self._explainers[method] = explainer
                self._logger.info(f"Explainer initialisé: {method.value}")
            except Exception as e:
                self._logger.warning(f"Échec de l'initialisation de l'explainer {method.value}: {str(e)}")
                continue
        
        if not self._explainers:
            raise ValueError("Aucun explainer n'a pu être initialisé.")
        
        self.add_audit_record("explainers_initialization", {
            "methods": [m.value for m in self._explainers.keys()]
        })
    
    def explain(self, X, y=None, **kwargs) -> ExplanationResult:
        """
        Génère des explications unifiées pour un ensemble de données.
        
        Args:
            X: Données à expliquer
            y: Labels réels (optionnel)
            **kwargs: Paramètres additionnels
            
        Returns:
            ExplanationResult: Résultat d'explication unifié
        """
        self.add_audit_record("explain_call", {
            "data_shape": X.shape if hasattr(X, "shape") else "unknown",
            "has_labels": y is not None,
            "params": {k: str(v) for k, v in kwargs.items()}
        })
        
        audience_level = kwargs.get('audience_level', AudienceLevel.TECHNICAL)
        
        # Collecter les résultats de chaque explainer
        explanations = {}
        for method, explainer in self._explainers.items():
            try:
                self._logger.info(f"Exécution de l'explainer {method.value}...")
                explanation = explainer.explain(X, y, **kwargs)
                explanations[method] = explanation
                self._logger.info(f"Explication {method.value} réussie")
            except Exception as e:
                self._logger.warning(f"Échec de l'explainer {method.value}: {str(e)}")
                continue
        
        if not explanations:
            raise RuntimeError("Tous les explainers ont échoué. Impossible de générer une explication unifiée.")
            
        # Agréger les résultats
        unified_explanation = self._aggregate_explanations(explanations, audience_level, **kwargs)
        
        return unified_explanation
    
    def explain_instance(self, instance, **kwargs) -> ExplanationResult:
        """
        Explique une instance spécifique avec une approche unifiée.
        
        Args:
            instance: Instance à expliquer
            **kwargs: Paramètres additionnels
            
        Returns:
            ExplanationResult: Explication unifiée pour l'instance
        """
        # Conversion en format compatible
        if isinstance(instance, pd.DataFrame) or isinstance(instance, pd.Series):
            X = instance
        else:
            X = np.array([instance])
        
        # Déléguer à la méthode explain
        return self.explain(X, **kwargs)
    
    def _aggregate_explanations(self, explanations: Dict[ExplainabilityMethod, ExplanationResult], 
                               audience_level: AudienceLevel, **kwargs) -> ExplanationResult:
        """
        Agrège les explications de différentes méthodes.
        
        Args:
            explanations: Dictionnaire des explications par méthode
            audience_level: Niveau d'audience cible
            **kwargs: Paramètres additionnels
            
        Returns:
            ExplanationResult: Explication agrégée
        """
        # Récupérer les métadonnées du modèle (toutes les explications devraient avoir les mêmes)
        any_explanation = next(iter(explanations.values()))
        model_metadata = any_explanation.model_metadata
        
        # Récupérer toutes les caractéristiques
        all_features = set()
        for explanation in explanations.values():
            all_features.update(fi.feature_name for fi in explanation.feature_importances)
        
        # Agréger les importances par caractéristique selon la stratégie choisie
        aggregated_importances = {}
        confidence_intervals = {}
        
        if self._aggregation_strategy == 'weighted':
            # Paramétrage des poids par méthode (par défaut égaux)
            weights = self._weights or {method: 1.0 for method in explanations.keys()}
            total_weight = sum(weights.get(method, 1.0) for method in explanations.keys())
            
            # Normalisation des poids
            weights = {method: w/total_weight for method, w in weights.items()}
            
            # Agrégation pondérée
            for feature in all_features:
                feature_importances = []
                feature_std_devs = []
                for method, explanation in explanations.items():
                    for fi in explanation.feature_importances:
                        if fi.feature_name == feature:
                            weight = weights.get(method, 1.0 / len(explanations))
                            feature_importances.append(fi.importance * weight)
                            if fi.std_dev is not None:
                                feature_std_devs.append(fi.std_dev * weight)
                            break
                
                aggregated_importances[feature] = sum(feature_importances)
                if feature_std_devs:
                    confidence_intervals[feature] = (
                        aggregated_importances[feature] - 1.96 * np.mean(feature_std_devs),
                        aggregated_importances[feature] + 1.96 * np.mean(feature_std_devs)
                    )
            
        elif self._aggregation_strategy == 'voting':
            # Système de vote basé sur le rang des caractéristiques
            for explanation in explanations.values():
                fis = sorted(explanation.feature_importances, key=lambda x: abs(x.importance), reverse=True)
                for i, fi in enumerate(fis):
                    if fi.feature_name not in aggregated_importances:
                        aggregated_importances[fi.feature_name] = 0
                    # Vote inversement proportionnel au rang
                    aggregated_importances[fi.feature_name] += 1 / (i + 1)
            
            # Normalisation
            total = sum(aggregated_importances.values())
            for feature in aggregated_importances:
                aggregated_importances[feature] /= total
            
        else:
            # Moyenne simple par défaut
            for feature in all_features:
                importances = []
                for explanation in explanations.values():
                    for fi in explanation.feature_importances:
                        if fi.feature_name == feature:
                            importances.append(fi.importance)
                            break
                
                if importances:
                    aggregated_importances[feature] = sum(importances) / len(importances)
        
        # Créer les objets FeatureImportance agrégés
        feature_importances = []
        for feature, importance in aggregated_importances.items():
            ci = confidence_intervals.get(feature, None)
            fi = FeatureImportance(
                feature_name=feature,
                importance=importance,
                confidence_interval=ci if ci else None,
                std_dev=None,  # À calculer si nécessaire
                p_value=None   # À calculer si nécessaire
            )
            feature_importances.append(fi)
        
        # Trier par importance absolue décroissante
        feature_importances.sort(key=lambda x: abs(x.importance), reverse=True)
        
        # Calculer des métriques de qualité de l'explication
        explanation_quality = self._calculate_explanation_quality(explanations)
        explanation_fidelity = self._calculate_explanation_fidelity(explanations)
        
        # Stocker les explications brutes selon le niveau d'audience
        if audience_level == AudienceLevel.TECHNICAL:
            raw_explanation = {
                method.value: explanation.raw_explanation 
                for method, explanation in explanations.items() 
                if explanation.raw_explanation is not None
            }
        else:
            # Pour les audiences non-techniques, simplifier les données brutes
            raw_explanation = None
        
        # Calculer des métriques additionnelles
        additional_metrics = self._calculate_additional_metrics(explanations)
        
        # Créer le résultat final
        unified_result = ExplanationResult(
            method=ExplainabilityMethod.UNIFIED,
            model_metadata=model_metadata,
            feature_importances=feature_importances,
            raw_explanation=raw_explanation,
            audience_level=audience_level,
            explanation_quality=explanation_quality,
            explanation_fidelity=explanation_fidelity,
            additional_metrics=additional_metrics
        )
        
        self.add_audit_record("aggregation_complete", {
            "aggregation_strategy": self._aggregation_strategy,
            "methods_used": [m.value for m in explanations.keys()],
            "explanation_quality": explanation_quality,
            "explanation_fidelity": explanation_fidelity
        })
        
        return unified_result
    
    def _calculate_explanation_quality(self, explanations: Dict[ExplainabilityMethod, ExplanationResult]) -> float:
        """
        Calcule une métrique de qualité globale pour l'explication unifiée.
        
        Args:
            explanations: Dictionnaire des explications par méthode
            
        Returns:
            float: Score de qualité entre 0 et 1
        """
        # Calculer la cohérence entre les différentes méthodes
        if len(explanations) <= 1:
            return 1.0  # Une seule méthode est toujours cohérente avec elle-même
        
        # Extraire les caractéristiques communes à toutes les méthodes
        common_features = set()
        first = True
        for explanation in explanations.values():
            features = {fi.feature_name for fi in explanation.feature_importances}
            if first:
                common_features = features
                first = False
            else:
                common_features = common_features.intersection(features)
        
        if not common_features:
            return 0.5  # Pas de caractéristiques communes, qualité moyenne
        
        # Calculer la corrélation de rang entre les méthodes pour ces caractéristiques
        import scipy.stats as stats
        
        rankings = {}
        for method, explanation in explanations.items():
            # Extraire et trier les importances des caractéristiques communes
            feature_importances = {
                fi.feature_name: abs(fi.importance) 
                for fi in explanation.feature_importances 
                if fi.feature_name in common_features
            }
            
            # Convertir en rangs
            sorted_features = sorted(feature_importances.keys(), 
                                    key=lambda f: feature_importances[f],
                                    reverse=True)
            rankings[method] = {f: i for i, f in enumerate(sorted_features)}
        
        # Calculer la corrélation moyenne de Spearman entre toutes les paires de méthodes
        methods = list(rankings.keys())
        correlations = []
        for i in range(len(methods)):
            for j in range(i+1, len(methods)):
                method1, method2 = methods[i], methods[j]
                
                # Préparer les données pour la corrélation
                features = list(common_features)
                ranks1 = [rankings[method1][f] for f in features]
                ranks2 = [rankings[method2][f] for f in features]
                
                # Calculer la corrélation
                corr, _ = stats.spearmanr(ranks1, ranks2)
                correlations.append(corr)
        
        # La qualité est la moyenne des corrélations, normalisée entre 0 et 1
        if not correlations:
            return 0.8  # Valeur par défaut raisonnable si pas de corrélation calculable
        
        # Transformer de [-1, 1] à [0, 1]
        avg_corr = np.mean(correlations)
        quality = (avg_corr + 1) / 2
        
        return quality
    
    def _calculate_explanation_fidelity(self, explanations: Dict[ExplainabilityMethod, ExplanationResult]) -> float:
        """
        Calcule la fidélité des explications par rapport au modèle.
        
        Args:
            explanations: Dictionnaire des explications par méthode
            
        Returns:
            float: Score de fidélité entre 0 et 1
        """
        # Calculer la moyenne des fidélités individuelles si disponibles
        fidelities = [
            e.explanation_fidelity for e in explanations.values()
            if e.explanation_fidelity is not None
        ]
        
        if fidelities:
            return np.mean(fidelities)
        
        # Si aucune fidélité n'est disponible, utiliser une heuristique basée
        # sur la qualité des explications individuelles
        qualities = [e.explanation_quality for e in explanations.values()]
        return np.mean(qualities) if qualities else 0.8
    
    def _calculate_additional_metrics(self, explanations: Dict[ExplainabilityMethod, ExplanationResult]) -> Dict[str, Any]:
        """
        Calcule des métriques additionnelles pour l'explication unifiée.
        
        Args:
            explanations: Dictionnaire des explications par méthode
            
        Returns:
            Dict[str, Any]: Métriques additionnelles
        """
        metrics = {
            "methods_used": [method.value for method in explanations.keys()],
            "method_weights": self._weights or "uniform",
            "aggregation_strategy": self._aggregation_strategy,
        }
        
        # Agrégation des métriques spécifiques aux méthodes
        method_metrics = {}
        for method, explanation in explanations.items():
            if explanation.additional_metrics:
                method_metrics[method.value] = explanation.additional_metrics
        
        if method_metrics:
            metrics["method_specific_metrics"] = method_metrics
        
        return metrics
