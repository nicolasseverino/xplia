"""
Critères d'évaluation pour les explications et métriques de confiance
====================================================================

Ce module définit les critères d'évaluation utilisés par les évaluateurs experts
pour noter la qualité des explications et des métriques de confiance.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union, Tuple


class CriteriaCategory(Enum):
    """Catégories de critères d'évaluation."""
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    ROBUSTNESS = "robustness"
    TRANSPARENCY = "transparency"
    FAIRNESS = "fairness"
    UNCERTAINTY = "uncertainty"
    INTERPRETABILITY = "interpretability"
    USABILITY = "usability"
    PERFORMANCE = "performance"


@dataclass
class EvaluationCriteria:
    """
    Critère d'évaluation pour les explications et métriques de confiance.
    
    Attributes:
        name: Nom du critère
        description: Description détaillée du critère
        category: Catégorie du critère
        weight: Poids du critère dans l'évaluation globale (0-1)
        min_score: Score minimum possible
        max_score: Score maximum possible
        thresholds: Seuils pour différents niveaux de qualité
    """
    name: str
    description: str
    category: CriteriaCategory
    weight: float = 1.0
    min_score: float = 0.0
    max_score: float = 10.0
    thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        """Initialisation après la création de l'instance."""
        if self.thresholds is None:
            self.thresholds = {
                "critical": 2.0,
                "poor": 4.0,
                "moderate": 6.0,
                "good": 8.0,
                "excellent": 10.0
            }
    
    def get_quality_level(self, score: float) -> str:
        """
        Détermine le niveau de qualité correspondant au score.
        
        Args:
            score: Score obtenu pour ce critère
            
        Returns:
            Niveau de qualité (critical, poor, moderate, good, excellent)
        """
        if score < self.thresholds["critical"]:
            return "critical"
        elif score < self.thresholds["poor"]:
            return "poor"
        elif score < self.thresholds["moderate"]:
            return "moderate"
        elif score < self.thresholds["good"]:
            return "good"
        else:
            return "excellent"


# Critères d'évaluation pour les métriques de confiance
TRUST_EVALUATION_CRITERIA = [
    EvaluationCriteria(
        name="Précision de l'incertitude",
        description="Évalue la précision des estimations d'incertitude par rapport à des références connues",
        category=CriteriaCategory.ACCURACY,
        weight=1.0
    ),
    EvaluationCriteria(
        name="Détection de fairwashing",
        description="Évalue la capacité à détecter correctement les cas de fairwashing",
        category=CriteriaCategory.FAIRNESS,
        weight=1.0
    ),
    EvaluationCriteria(
        name="Cohérence des métriques",
        description="Évalue la cohérence des différentes métriques de confiance entre elles",
        category=CriteriaCategory.CONSISTENCY,
        weight=0.8
    ),
    EvaluationCriteria(
        name="Robustesse aux perturbations",
        description="Évalue la stabilité des métriques face à de petites perturbations des données",
        category=CriteriaCategory.ROBUSTNESS,
        weight=0.9
    ),
    EvaluationCriteria(
        name="Complétude des métriques",
        description="Évalue si toutes les dimensions importantes de la confiance sont couvertes",
        category=CriteriaCategory.COMPLETENESS,
        weight=0.7
    ),
    EvaluationCriteria(
        name="Transparence des calculs",
        description="Évalue la transparence des méthodes de calcul des métriques de confiance",
        category=CriteriaCategory.TRANSPARENCY,
        weight=0.6
    ),
    EvaluationCriteria(
        name="Interprétabilité des scores",
        description="Évalue la facilité d'interprétation des scores de confiance",
        category=CriteriaCategory.INTERPRETABILITY,
        weight=0.8
    ),
    EvaluationCriteria(
        name="Utilité des recommandations",
        description="Évalue l'utilité pratique des recommandations générées",
        category=CriteriaCategory.USABILITY,
        weight=0.7
    ),
    EvaluationCriteria(
        name="Performance de calcul",
        description="Évalue l'efficacité computationnelle des métriques de confiance",
        category=CriteriaCategory.PERFORMANCE,
        weight=0.5
    ),
    EvaluationCriteria(
        name="Calibration des intervalles",
        description="Évalue la calibration des intervalles de confiance",
        category=CriteriaCategory.UNCERTAINTY,
        weight=0.9
    )
]

# Critères d'évaluation pour la qualité des explications
EXPLANATION_QUALITY_CRITERIA = [
    EvaluationCriteria(
        name="Fidélité au modèle",
        description="Évalue à quel point l'explication reflète fidèlement le comportement du modèle",
        category=CriteriaCategory.ACCURACY,
        weight=1.0
    ),
    EvaluationCriteria(
        name="Complétude de l'explication",
        description="Évalue si l'explication couvre tous les aspects importants de la prédiction",
        category=CriteriaCategory.COMPLETENESS,
        weight=0.9
    ),
    EvaluationCriteria(
        name="Cohérence entre instances",
        description="Évalue la cohérence des explications entre différentes instances similaires",
        category=CriteriaCategory.CONSISTENCY,
        weight=0.8
    ),
    EvaluationCriteria(
        name="Robustesse de l'explication",
        description="Évalue la stabilité de l'explication face à de petites perturbations",
        category=CriteriaCategory.ROBUSTNESS,
        weight=0.7
    ),
    EvaluationCriteria(
        name="Simplicité et clarté",
        description="Évalue la simplicité et la clarté de l'explication pour les utilisateurs",
        category=CriteriaCategory.INTERPRETABILITY,
        weight=0.8
    ),
    EvaluationCriteria(
        name="Équité de l'explication",
        description="Évalue si l'explication est équitable vis-à-vis des groupes protégés",
        category=CriteriaCategory.FAIRNESS,
        weight=0.9
    ),
    EvaluationCriteria(
        name="Transparence des méthodes",
        description="Évalue la transparence des méthodes utilisées pour générer l'explication",
        category=CriteriaCategory.TRANSPARENCY,
        weight=0.6
    ),
    EvaluationCriteria(
        name="Actionabilité",
        description="Évalue si l'explication fournit des informations actionnables",
        category=CriteriaCategory.USABILITY,
        weight=0.7
    ),
    EvaluationCriteria(
        name="Performance de génération",
        description="Évalue l'efficacité computationnelle de la génération d'explications",
        category=CriteriaCategory.PERFORMANCE,
        weight=0.5
    ),
    EvaluationCriteria(
        name="Quantification de l'incertitude",
        description="Évalue si l'explication quantifie correctement l'incertitude",
        category=CriteriaCategory.UNCERTAINTY,
        weight=0.8
    )
]
