"""
Intégration de l'évaluation experte avec les explainers et modules de conformité
===============================================================================

Ce module fournit des fonctions pour intégrer l'évaluation experte
avec les explainers et les modules de conformité de XPLIA.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple

from xplia.explainers.base_explainer import BaseExplainer
from xplia.trust.uncertainty import UncertaintyEstimator
from xplia.trust.fairwashing import FairwashingAuditor
from xplia.trust.confidence import ConfidenceReporter
from xplia.compliance.expert_review.trust_expert_evaluator import TrustExpertEvaluator
from xplia.explainers.expert_evaluator import ExplanationQualityEvaluator

logger = logging.getLogger(__name__)


class ExpertEvaluationIntegrator:
    """
    Intégrateur pour l'évaluation experte.
    
    Cette classe facilite l'intégration de l'évaluation experte
    avec les explainers et les modules de conformité de XPLIA.
    """
    
    def __init__(
        self,
        uncertainty_estimator: Optional[UncertaintyEstimator] = None,
        fairwashing_auditor: Optional[FairwashingAuditor] = None,
        confidence_reporter: Optional[ConfidenceReporter] = None,
        trust_evaluator: Optional[TrustExpertEvaluator] = None,
        explanation_evaluator: Optional[ExplanationQualityEvaluator] = None
    ):
        """
        Initialise l'intégrateur d'évaluation experte.
        
        Args:
            uncertainty_estimator: Estimateur d'incertitude
            fairwashing_auditor: Auditeur de fairwashing
            confidence_reporter: Générateur de rapport de confiance
            trust_evaluator: Évaluateur expert pour les métriques de confiance
            explanation_evaluator: Évaluateur de qualité des explications
        """
        self.uncertainty_estimator = uncertainty_estimator or UncertaintyEstimator()
        self.fairwashing_auditor = fairwashing_auditor or FairwashingAuditor()
        self.confidence_reporter = confidence_reporter or ConfidenceReporter()
        self.trust_evaluator = trust_evaluator or TrustExpertEvaluator()
        self.explanation_evaluator = explanation_evaluator or ExplanationQualityEvaluator()
        
        logger.info("Intégrateur d'évaluation experte initialisé")
    
    def evaluate_explanation_pipeline(
        self,
        explainer: BaseExplainer,
        model_adapter: Any,
        instance: Any,
        background_data: Any = None,
        sensitive_features: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Évalue l'ensemble du pipeline d'explication.
        
        Args:
            explainer: Explainer à utiliser
            model_adapter: Adaptateur de modèle
            instance: Instance à expliquer
            background_data: Données de fond pour l'explication
            sensitive_features: Features sensibles pour l'audit de fairwashing
            **kwargs: Arguments supplémentaires pour l'explainer
            
        Returns:
            Dictionnaire contenant les résultats de l'évaluation
        """
        try:
            # Génération de l'explication
            logger.info("Génération de l'explication...")
            explanation = explainer.explain(
                model_adapter=model_adapter,
                instance=instance,
                background_data=background_data,
                **kwargs
            )
            
            # Calcul des métriques d'incertitude
            logger.info("Calcul des métriques d'incertitude...")
            uncertainty_metrics = self.uncertainty_estimator.estimate(
                model_adapter=model_adapter,
                instance=instance,
                explanation=explanation,
                background_data=background_data
            )
            
            # Audit de fairwashing
            logger.info("Audit de fairwashing...")
            fairwashing_audit = self.fairwashing_auditor.audit(
                model_adapter=model_adapter,
                instance=instance,
                explanation=explanation,
                background_data=background_data,
                sensitive_features=sensitive_features
            )
            
            # Génération du rapport de confiance
            logger.info("Génération du rapport de confiance...")
            confidence_report = self.confidence_reporter.generate_report(
                explanation=explanation,
                uncertainty_metrics=uncertainty_metrics,
                fairwashing_audit=fairwashing_audit
            )
            
            # Évaluation de la qualité de l'explication
            logger.info("Évaluation de la qualité de l'explication...")
            explanation_quality = self.explanation_evaluator.evaluate_explanation(
                explanation=explanation,
                model_adapter=model_adapter,
                instance=instance,
                background_data=background_data
            )
            
            # Évaluation des métriques de confiance
            logger.info("Évaluation des métriques de confiance...")
            trust_evaluation = self.trust_evaluator.evaluate_trust_metrics(
                uncertainty_metrics=uncertainty_metrics,
                fairwashing_audit=fairwashing_audit,
                confidence_report=confidence_report,
                explanation=explanation
            )
            
            # Compilation des résultats
            results = {
                "explanation": explanation,
                "uncertainty_metrics": uncertainty_metrics,
                "fairwashing_audit": fairwashing_audit,
                "confidence_report": confidence_report,
                "explanation_quality": explanation_quality.to_dict(),
                "trust_evaluation": trust_evaluation.to_dict()
            }
            
            logger.info("Évaluation du pipeline d'explication complétée avec succès")
            return results
        
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation du pipeline d'explication: {e}")
            return {"error": str(e)}
    
    def generate_comprehensive_report(
        self,
        evaluation_results: Dict[str, Any],
        include_visualizations: bool = True
    ) -> Dict[str, Any]:
        """
        Génère un rapport complet à partir des résultats d'évaluation.
        
        Args:
            evaluation_results: Résultats de l'évaluation
            include_visualizations: Indique si les visualisations doivent être incluses
            
        Returns:
            Dictionnaire contenant le rapport complet
        """
        try:
            # Extraction des composants
            explanation = evaluation_results.get("explanation", {})
            uncertainty_metrics = evaluation_results.get("uncertainty_metrics", {})
            fairwashing_audit = evaluation_results.get("fairwashing_audit", {})
            confidence_report = evaluation_results.get("confidence_report", {})
            explanation_quality = evaluation_results.get("explanation_quality", {})
            trust_evaluation = evaluation_results.get("trust_evaluation", {})
            
            # Compilation du rapport
            report = {
                "summary": {
                    "explanation_quality_score": explanation_quality.get("global_score", 0.0),
                    "trust_score": trust_evaluation.get("global_score", 0.0),
                    "overall_score": (
                        explanation_quality.get("global_score", 0.0) * 0.5 +
                        trust_evaluation.get("global_score", 0.0) * 0.5
                    )
                },
                "explanation": {
                    "quality": explanation_quality,
                    "details": explanation
                },
                "trust": {
                    "evaluation": trust_evaluation,
                    "uncertainty": uncertainty_metrics,
                    "fairwashing": fairwashing_audit,
                    "confidence": confidence_report
                },
                "recommendations": []
            }
            
            # Compilation des recommandations
            if "recommendations" in explanation_quality:
                report["recommendations"].extend(explanation_quality["recommendations"])
            
            if "recommendations" in trust_evaluation:
                report["recommendations"].extend(trust_evaluation["recommendations"])
            
            if "recommendations" in confidence_report:
                report["recommendations"].extend(confidence_report["recommendations"])
            
            # Ajout de visualisations si demandé
            if include_visualizations:
                try:
                    from xplia.visualizations import ChartGenerator
                    
                    chart_generator = ChartGenerator()
                    
                    # Visualisations pour le score global
                    report["visualizations"] = {
                        "overall_score": chart_generator.gauge_chart(
                            value=report["summary"]["overall_score"],
                            min_value=0.0,
                            max_value=10.0,
                            title="Score global",
                            thresholds=[3.0, 5.0, 7.0, 9.0]
                        )
                    }
                    
                except ImportError:
                    logger.warning("Module de visualisations non disponible")
            
            logger.info("Rapport complet généré avec succès")
            return report
        
        except Exception as e:
            logger.error(f"Erreur lors de la génération du rapport complet: {e}")
            return {"error": str(e)}


# Fonction utilitaire pour une évaluation rapide
def quick_evaluate(
    explainer: BaseExplainer,
    model_adapter: Any,
    instance: Any,
    background_data: Any = None,
    sensitive_features: List[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Fonction utilitaire pour une évaluation rapide du pipeline d'explication.
    
    Args:
        explainer: Explainer à utiliser
        model_adapter: Adaptateur de modèle
        instance: Instance à expliquer
        background_data: Données de fond pour l'explication
        sensitive_features: Features sensibles pour l'audit de fairwashing
        **kwargs: Arguments supplémentaires pour l'explainer
        
    Returns:
        Dictionnaire contenant les résultats de l'évaluation
    """
    integrator = ExpertEvaluationIntegrator()
    results = integrator.evaluate_explanation_pipeline(
        explainer=explainer,
        model_adapter=model_adapter,
        instance=instance,
        background_data=background_data,
        sensitive_features=sensitive_features,
        **kwargs
    )
    
    return integrator.generate_comprehensive_report(results)
