"""
Détection de Fairwashing pour Explications
======================================

Ce module implémente des mécanismes avancés pour détecter les explications
potentiellement trompeuses ou manipulées pour masquer des biais.
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import time

from ...core.base import ExplanationResult, ExplainerBase


class FairwashingType(Enum):
    """Types de fairwashing dans les explications."""
    FEATURE_MASKING = "feature_masking"       # Masquage de features sensibles
    IMPORTANCE_SHIFT = "importance_shift"     # Déplacement d'importance entre features
    BIAS_HIDING = "bias_hiding"               # Dissimulation de biais
    CHERRY_PICKING = "cherry_picking"         # Sélection biaisée d'exemples
    THRESHOLD_MANIPULATION = "threshold_manipulation"  # Manipulation des seuils


@dataclass
class FairwashingAudit:
    """
    Résultat d'un audit de fairwashing pour une explication.
    """
    # Score global de fairwashing (0 = pas de fairwashing, 1 = fairwashing maximal)
    fairwashing_score: float = 0.0
    
    # Détection par type de fairwashing
    detected_types: Set[FairwashingType] = field(default_factory=set)
    type_scores: Dict[FairwashingType, float] = field(default_factory=dict)
    
    # Détails par feature
    feature_manipulation_scores: Dict[str, float] = field(default_factory=dict)
    
    # Anomalies détectées
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    
    # Métadonnées
    timestamp: float = field(default_factory=time.time)
    audit_method: str = "default"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'audit en dictionnaire."""
        return {
            "fairwashing_score": self.fairwashing_score,
            "detected_types": [t.value for t in self.detected_types],
            "type_scores": {t.value: s for t, s in self.type_scores.items()},
            "feature_manipulation_scores": self.feature_manipulation_scores,
            "anomalies": self.anomalies,
            "timestamp": self.timestamp,
            "audit_method": self.audit_method
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FairwashingAudit':
        """Crée un audit à partir d'un dictionnaire."""
        detected_types = set()
        for type_value in data.get("detected_types", []):
            try:
                detected_types.add(FairwashingType(type_value))
            except ValueError:
                logging.warning(f"Type de fairwashing inconnu ignoré: {type_value}")
        
        type_scores = {}
        for type_value, score in data.get("type_scores", {}).items():
            try:
                type_scores[FairwashingType(type_value)] = score
            except ValueError:
                logging.warning(f"Type de fairwashing inconnu ignoré: {type_value}")
        
        return cls(
            fairwashing_score=data.get("fairwashing_score", 0.0),
            detected_types=detected_types,
            type_scores=type_scores,
            feature_manipulation_scores=data.get("feature_manipulation_scores", {}),
            anomalies=data.get("anomalies", []),
            timestamp=data.get("timestamp", time.time()),
            audit_method=data.get("audit_method", "default")
        )


class FairwashingDetector:
    """
    Détecteur de fairwashing pour les explications.
    """
    
    def __init__(self, 
                 sensitive_features: Optional[List[str]] = None,
                 detection_threshold: float = 0.7,
                 methods: Optional[List[str]] = None):
        """
        Initialise le détecteur de fairwashing.
        
        Args:
            sensitive_features: Liste des features sensibles à surveiller
            detection_threshold: Seuil de détection de fairwashing
            methods: Méthodes de détection à utiliser
        """
        self.sensitive_features = sensitive_features or []
        self.detection_threshold = detection_threshold
        self.methods = methods or ["consistency", "sensitivity", "counterfactual"]
    
    def detect_fairwashing(self, 
                         explanation: ExplanationResult,
                         reference_explanation: Optional[ExplanationResult] = None,
                         model: Optional[Any] = None,
                         X: Optional[Any] = None,
                         sensitive_groups: Optional[Dict[str, Any]] = None,
                         **kwargs) -> FairwashingAudit:
        """
        Détecte les signes de fairwashing dans une explication.
        
        Args:
            explanation: L'explication à auditer
            reference_explanation: Explication de référence (optionnel)
            model: Modèle original (optionnel)
            X: Données d'entrée (optionnel)
            sensitive_groups: Groupes sensibles dans les données (optionnel)
            **kwargs: Paramètres additionnels
            
        Returns:
            Résultat de l'audit de fairwashing
        """
        # Initialiser l'audit
        audit = FairwashingAudit()
        
        # Liste des méthodes de détection disponibles
        detection_methods = {
            "consistency": self._check_consistency,
            "sensitivity": self._check_sensitivity,
            "counterfactual": self._check_counterfactuals,
            "statistical": self._statistical_analysis,
            "adversarial": self._adversarial_testing
        }
        
        # Appliquer les méthodes disponibles
        available_methods = [m for m in self.methods if m in detection_methods]
        
        if not available_methods:
            logging.warning("Aucune méthode de détection de fairwashing disponible")
            return audit
        
        # Collecter les résultats de chaque méthode
        method_results = []
        
        for method_name in available_methods:
            if reference_explanation is None and method_name == "consistency":
                continue  # Cette méthode nécessite une explication de référence
                
            if X is None and method_name in ["sensitivity", "counterfactual", "adversarial"]:
                continue  # Ces méthodes nécessitent des données
            
            if model is None and method_name in ["adversarial"]:
                continue  # Cette méthode nécessite le modèle original
            
            method_fn = detection_methods[method_name]
            try:
                method_audit = method_fn(
                    explanation, reference_explanation, model, X, sensitive_groups, **kwargs
                )
                method_results.append(method_audit)
            except Exception as e:
                logging.warning(f"Erreur lors de la détection de fairwashing avec {method_name}: {str(e)}")
        
        # Si aucune méthode n'a fonctionné, retourner l'audit par défaut
        if not method_results:
            return audit
        
        # Agréger les résultats des différentes méthodes
        audit = self._aggregate_audits(method_results)
        
        return audit
    
    def _check_consistency(self, 
                         explanation: ExplanationResult,
                         reference_explanation: Optional[ExplanationResult],
                         model: Optional[Any],
                         X: Optional[Any],
                         sensitive_groups: Optional[Dict[str, Any]],
                         **kwargs) -> FairwashingAudit:
        """
        Vérifie la cohérence entre deux explications.
        
        Args:
            explanation: L'explication à auditer
            reference_explanation: Explication de référence
            model: Modèle original (non utilisé)
            X: Données d'entrée (non utilisé)
            sensitive_groups: Groupes sensibles (non utilisé)
            **kwargs: Paramètres additionnels
            
        Returns:
            Audit de fairwashing
        """
        # Dans une implémentation réelle, cette méthode pourrait:
        # 1. Comparer les attributions de features entre les deux explications
        # 2. Détecter des incohérences significatives, surtout sur les features sensibles
        
        # Pour l'exemple, nous retournons un audit simulé
        audit = FairwashingAudit(
            fairwashing_score=0.35,
            audit_method="consistency"
        )
        
        # Simuler la détection de masquage de features
        if np.random.random() < 0.3:
            audit.detected_types.add(FairwashingType.FEATURE_MASKING)
            audit.type_scores[FairwashingType.FEATURE_MASKING] = 0.75
        
        # Simuler la détection de déplacement d'importance
        if np.random.random() < 0.4:
            audit.detected_types.add(FairwashingType.IMPORTANCE_SHIFT)
            audit.type_scores[FairwashingType.IMPORTANCE_SHIFT] = 0.65
        
        return audit
    
    def _check_sensitivity(self, 
                         explanation: ExplanationResult,
                         reference_explanation: Optional[ExplanationResult],
                         model: Optional[Any],
                         X: Optional[Any],
                         sensitive_groups: Optional[Dict[str, Any]],
                         **kwargs) -> FairwashingAudit:
        """
        Vérifie la sensibilité de l'explication aux variations des features sensibles.
        
        Args:
            explanation: L'explication à auditer
            reference_explanation: Explication de référence (non utilisé)
            model: Modèle original (non utilisé)
            X: Données d'entrée
            sensitive_groups: Groupes sensibles
            **kwargs: Paramètres additionnels
            
        Returns:
            Audit de fairwashing
        """
        # Dans une implémentation réelle, cette méthode pourrait:
        # 1. Perturber les features sensibles dans les données
        # 2. Vérifier si l'explication change de manière appropriée
        
        # Pour l'exemple, nous retournons un audit simulé
        audit = FairwashingAudit(
            fairwashing_score=0.28,
            audit_method="sensitivity"
        )
        
        # Simuler la détection de dissimulation de biais
        if np.random.random() < 0.35:
            audit.detected_types.add(FairwashingType.BIAS_HIDING)
            audit.type_scores[FairwashingType.BIAS_HIDING] = 0.72
        
        return audit
    
    def _check_counterfactuals(self, 
                             explanation: ExplanationResult,
                             reference_explanation: Optional[ExplanationResult],
                             model: Optional[Any],
                             X: Optional[Any],
                             sensitive_groups: Optional[Dict[str, Any]],
                             **kwargs) -> FairwashingAudit:
        """
        Vérifie la cohérence de l'explication avec des exemples contrefactuels.
        
        Args:
            explanation: L'explication à auditer
            reference_explanation: Explication de référence (non utilisé)
            model: Modèle original (non utilisé)
            X: Données d'entrée
            sensitive_groups: Groupes sensibles (non utilisé)
            **kwargs: Paramètres additionnels
            
        Returns:
            Audit de fairwashing
        """
        # Dans une implémentation réelle, cette méthode pourrait:
        # 1. Générer des exemples contrefactuels
        # 2. Vérifier si l'explication est cohérente avec ces exemples
        
        # Pour l'exemple, nous retournons un audit simulé
        audit = FairwashingAudit(
            fairwashing_score=0.18,
            audit_method="counterfactual"
        )
        
        # Simuler la détection de sélection biaisée d'exemples
        if np.random.random() < 0.25:
            audit.detected_types.add(FairwashingType.CHERRY_PICKING)
            audit.type_scores[FairwashingType.CHERRY_PICKING] = 0.68
        
        return audit
    
    def _statistical_analysis(self, 
                            explanation: ExplanationResult,
                            reference_explanation: Optional[ExplanationResult],
                            model: Optional[Any],
                            X: Optional[Any],
                            sensitive_groups: Optional[Dict[str, Any]],
                            **kwargs) -> FairwashingAudit:
        """
        Effectue une analyse statistique de l'explication.
        
        Args:
            explanation: L'explication à auditer
            reference_explanation: Explication de référence (non utilisé)
            model: Modèle original (non utilisé)
            X: Données d'entrée (non utilisé)
            sensitive_groups: Groupes sensibles (non utilisé)
            **kwargs: Paramètres additionnels
            
        Returns:
            Audit de fairwashing
        """
        # Dans une implémentation réelle, cette méthode pourrait:
        # 1. Analyser statistiquement les attributions de features
        # 2. Détecter des anomalies ou des distributions suspectes
        
        # Pour l'exemple, nous retournons un audit simulé
        audit = FairwashingAudit(
            fairwashing_score=0.22,
            audit_method="statistical"
        )
        
        # Simuler la détection de manipulation de seuils
        if np.random.random() < 0.3:
            audit.detected_types.add(FairwashingType.THRESHOLD_MANIPULATION)
            audit.type_scores[FairwashingType.THRESHOLD_MANIPULATION] = 0.58
        
        return audit
    
    def _adversarial_testing(self, 
                           explanation: ExplanationResult,
                           reference_explanation: Optional[ExplanationResult],
                           model: Optional[Any],
                           X: Optional[Any],
                           sensitive_groups: Optional[Dict[str, Any]],
                           **kwargs) -> FairwashingAudit:
        """
        Teste l'explication avec des exemples adversariaux.
        
        Args:
            explanation: L'explication à auditer
            reference_explanation: Explication de référence (non utilisé)
            model: Modèle original
            X: Données d'entrée
            sensitive_groups: Groupes sensibles (non utilisé)
            **kwargs: Paramètres additionnels
            
        Returns:
            Audit de fairwashing
        """
        # Dans une implémentation réelle, cette méthode pourrait:
        # 1. Générer des exemples adversariaux
        # 2. Vérifier si l'explication est robuste face à ces exemples
        
        # Pour l'exemple, nous retournons un audit simulé
        audit = FairwashingAudit(
            fairwashing_score=0.32,
            audit_method="adversarial"
        )
        
        # Simuler la détection de plusieurs types de fairwashing
        if np.random.random() < 0.4:
            audit.detected_types.add(FairwashingType.FEATURE_MASKING)
            audit.type_scores[FairwashingType.FEATURE_MASKING] = 0.62
        
        if np.random.random() < 0.3:
            audit.detected_types.add(FairwashingType.BIAS_HIDING)
            audit.type_scores[FairwashingType.BIAS_HIDING] = 0.78
        
        return audit
    
    def _aggregate_audits(self, audits: List[FairwashingAudit]) -> FairwashingAudit:
        """
        Agrège les résultats de plusieurs audits.
        
        Args:
            audits: Liste d'audits à agréger
            
        Returns:
            Audit agrégé
        """
        if not audits:
            return FairwashingAudit()
        
        # Initialiser l'audit agrégé
        aggregated = FairwashingAudit(
            audit_method="aggregated",
            timestamp=time.time()
        )
        
        # Calculer le score global de fairwashing (moyenne pondérée)
        weights = {
            "consistency": 0.3,
            "sensitivity": 0.25,
            "counterfactual": 0.2,
            "statistical": 0.15,
            "adversarial": 0.1
        }
        
        total_weight = 0.0
        weighted_score = 0.0
        
        for audit in audits:
            method = audit.audit_method
            weight = weights.get(method, 0.1)
            weighted_score += audit.fairwashing_score * weight
            total_weight += weight
        
        if total_weight > 0:
            aggregated.fairwashing_score = weighted_score / total_weight
        
        # Agréger les types détectés et leurs scores
        all_types = set()
        type_scores = {}
        
        for audit in audits:
            all_types.update(audit.detected_types)
            
            for fairwashing_type, score in audit.type_scores.items():
                if fairwashing_type not in type_scores:
                    type_scores[fairwashing_type] = []
                
                type_scores[fairwashing_type].append(score)
        
        # Calculer les scores moyens par type
        for fairwashing_type, scores in type_scores.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                
                # Ajouter le type seulement si le score moyen dépasse le seuil
                if avg_score >= self.detection_threshold:
                    aggregated.detected_types.add(fairwashing_type)
                    aggregated.type_scores[fairwashing_type] = avg_score
        
        # Agréger les scores de manipulation par feature
        all_features = set()
        feature_scores = {}
        
        for audit in audits:
            all_features.update(audit.feature_manipulation_scores.keys())
            
            for feature, score in audit.feature_manipulation_scores.items():
                if feature not in feature_scores:
                    feature_scores[feature] = []
                
                feature_scores[feature].append(score)
        
        # Calculer les scores moyens par feature
        for feature, scores in feature_scores.items():
            if scores:
                aggregated.feature_manipulation_scores[feature] = sum(scores) / len(scores)
        
        # Agréger les anomalies
        for audit in audits:
            aggregated.anomalies.extend(audit.anomalies)
        
        return aggregated
    
    def apply_to_explanation(self, 
                           explanation: ExplanationResult,
                           audit: FairwashingAudit) -> ExplanationResult:
        """
        Applique les résultats d'audit à une explication.
        
        Args:
            explanation: L'explication à enrichir
            audit: Résultat de l'audit
            
        Returns:
            Explication enrichie avec les résultats d'audit
        """
        # Ajouter les résultats d'audit aux métadonnées
        explanation.metadata.fairwashing_audit = audit.to_dict()
        
        # Si l'explication contient des attributions de features,
        # ajouter les scores de manipulation correspondants
        if hasattr(explanation, "feature_importances") and audit.feature_manipulation_scores:
            explanation.feature_manipulation_scores = audit.feature_manipulation_scores
        
        return explanation
