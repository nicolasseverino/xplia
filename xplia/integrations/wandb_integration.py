"""
Weights & Biases integration for XPLIA.

Log explanations, compliance reports, and trust metrics to W&B.

Author: XPLIA Team
License: MIT
"""

from typing import Any, Dict, Optional, List
import tempfile
from pathlib import Path

try:
    import wandb
except ImportError:
    raise ImportError(
        "Weights & Biases not installed. Install with: pip install xplia[mlops]"
    )

from xplia.core.base import ExplanationResult


class XPLIAWandBLogger:
    """
    Logger for XPLIA explanations and metrics to Weights & Biases.

    Examples
    --------
    >>> logger = XPLIAWandBLogger(project="my-ml-project")
    >>> logger.log_explanation(explanation)
    """

    def __init__(
        self,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize W&B logger.

        Parameters
        ----------
        project : str, optional
            W&B project name.

        entity : str, optional
            W&B entity (username or team).

        config : dict, optional
            Configuration to log.
        """
        self.project = project
        self.entity = entity
        self.config = config or {}

    def init_run(
        self,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None
    ):
        """
        Initialize W&B run.

        Parameters
        ----------
        name : str, optional
            Run name.

        tags : list of str, optional
            Tags for the run.

        notes : str, optional
            Notes for the run.
        """
        default_tags = tags or []
        default_tags.extend(["xplia", "explainability"])

        wandb.init(
            project=self.project,
            entity=self.entity,
            name=name,
            tags=default_tags,
            notes=notes,
            config=self.config
        )

    def log_explanation(
        self,
        explanation: ExplanationResult,
        step: Optional[int] = None
    ):
        """
        Log explanation to W&B.

        Parameters
        ----------
        explanation : ExplanationResult
            Explanation to log.

        step : int, optional
            Step number for logging.
        """
        log_dict = {
            "explanation/method": explanation.method
        }

        # Log quality metrics
        if hasattr(explanation, 'quality_metrics') and explanation.quality_metrics:
            for metric_name, metric_value in explanation.quality_metrics.items():
                if isinstance(metric_value, (int, float)):
                    log_dict[f"explanation/quality/{metric_name}"] = metric_value

        # Log feature importance
        if 'feature_importance' in explanation.explanation_data:
            import numpy as np

            importance = explanation.explanation_data['feature_importance']
            feature_names = explanation.explanation_data.get('feature_names', [])

            if isinstance(importance, np.ndarray):
                importance = importance.tolist()

            # Create bar chart
            if feature_names:
                importance_table = wandb.Table(
                    columns=["feature", "importance"],
                    data=[[name, imp] for name, imp in zip(feature_names, importance)]
                )
                log_dict["explanation/feature_importance"] = wandb.plot.bar(
                    importance_table,
                    "feature",
                    "importance",
                    title="Feature Importance"
                )

                # Also log top features
                top_indices = np.argsort(np.abs(importance))[-5:][::-1]
                for idx in top_indices:
                    log_dict[f"explanation/top_features/{feature_names[idx]}"] = importance[idx]

        wandb.log(log_dict, step=step)

    def log_compliance_report(
        self,
        compliance_result: Any,
        regulation: str,
        step: Optional[int] = None
    ):
        """
        Log compliance report to W&B.

        Parameters
        ----------
        compliance_result : ComplianceResult
            Compliance check result.

        regulation : str
            Regulation name.

        step : int, optional
            Step number.
        """
        log_dict = {
            f"compliance/{regulation}/score": compliance_result.score,
            f"compliance/{regulation}/compliant": compliance_result.compliant
        }

        # Log requirements
        if hasattr(compliance_result, 'requirements'):
            met_requirements = sum(1 for req in compliance_result.requirements if req.get('met', False))
            total_requirements = len(compliance_result.requirements)
            log_dict[f"compliance/{regulation}/requirements_met"] = met_requirements
            log_dict[f"compliance/{regulation}/requirements_total"] = total_requirements

        # Save full report as artifact
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / f"{regulation}_compliance_report.pdf"
            compliance_result.export(str(report_path), format='pdf')

            artifact = wandb.Artifact(
                name=f"{regulation}_compliance_report",
                type="compliance_report",
                description=f"Compliance report for {regulation}"
            )
            artifact.add_file(str(report_path))
            wandb.log_artifact(artifact)

        wandb.log(log_dict, step=step)

    def log_trust_metrics(
        self,
        trust_metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """
        Log trust evaluation metrics to W&B.

        Parameters
        ----------
        trust_metrics : dict
            Trust metrics.

        step : int, optional
            Step number.
        """
        log_dict = {}
        for metric_name, metric_value in trust_metrics.items():
            if isinstance(metric_value, (int, float)):
                log_dict[f"trust/{metric_name}"] = metric_value

        wandb.log(log_dict, step=step)

    def log_fairwashing_detection(
        self,
        fairwashing_result: Any,
        step: Optional[int] = None
    ):
        """
        Log fairwashing detection results.

        Parameters
        ----------
        fairwashing_result : FairwashingResult
            Detection result.

        step : int, optional
            Step number.
        """
        log_dict = {
            "fairwashing/detected": fairwashing_result.detected
        }

        if fairwashing_result.detected:
            log_dict["fairwashing/severity"] = fairwashing_result.severity
            log_dict["fairwashing/types_count"] = len(fairwashing_result.fairwashing_types)

            # Create alert
            wandb.alert(
                title="Fairwashing Detected",
                text=f"Severity: {fairwashing_result.severity}, Types: {', '.join(fairwashing_result.fairwashing_types)}",
                level=wandb.AlertLevel.WARN
            )

        wandb.log(log_dict, step=step)

    def log_uncertainty_metrics(
        self,
        uncertainty_result: Any,
        step: Optional[int] = None
    ):
        """
        Log uncertainty quantification results.

        Parameters
        ----------
        uncertainty_result : UncertaintyResult
            Uncertainty metrics.

        step : int, optional
            Step number.
        """
        log_dict = {
            "uncertainty/global": uncertainty_result.global_uncertainty,
            "uncertainty/aleatoric": uncertainty_result.aleatoric_uncertainty,
            "uncertainty/epistemic": uncertainty_result.epistemic_uncertainty
        }

        if hasattr(uncertainty_result, 'structural_uncertainty'):
            log_dict["uncertainty/structural"] = uncertainty_result.structural_uncertainty

        wandb.log(log_dict, step=step)

    def log_model_with_explainability(
        self,
        model: Any,
        explainer: Any,
        model_name: str,
        sample_input: Optional[Any] = None
    ):
        """
        Log model with explainability artifacts.

        Parameters
        ----------
        model : object
            ML model.

        explainer : Explainer
            XPLIA explainer.

        model_name : str
            Model name.

        sample_input : array-like, optional
            Sample input for demonstration.
        """
        # Save model as artifact
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"

            import joblib
            joblib.dump(model, model_path)

            artifact = wandb.Artifact(
                name=model_name,
                type="model",
                description=f"Model with XPLIA explainability"
            )
            artifact.add_file(str(model_path))

            # Add explainer metadata
            artifact.metadata = {
                "explainer_method": explainer.method if hasattr(explainer, 'method') else 'unknown',
                "xplia_version": "1.0.0"
            }

            wandb.log_artifact(artifact)

        # Log sample explanation if provided
        if sample_input is not None:
            explanation = explainer.explain(sample_input)
            self.log_explanation(explanation)

    def finish(self):
        """Finish W&B run."""
        wandb.finish()


# Context manager
class XPLIAWandBContext:
    """
    Context manager for XPLIA + W&B integration.

    Examples
    --------
    >>> with XPLIAWandBContext(project="my-project", name="experiment-1") as logger:
    ...     # Train model
    ...     model.fit(X, y)
    ...
    ...     # Create and log explanations
    ...     explainer = create_explainer(model, method='shap')
    ...     explanation = explainer.explain(X_test)
    ...     logger.log_explanation(explanation)
    """

    def __init__(
        self,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        config: Optional[Dict] = None
    ):
        self.project = project
        self.entity = entity
        self.name = name
        self.tags = tags
        self.config = config
        self.logger = None

    def __enter__(self):
        self.logger = XPLIAWandBLogger(
            project=self.project,
            entity=self.entity,
            config=self.config
        )
        self.logger.init_run(name=self.name, tags=self.tags)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.logger:
            self.logger.finish()
