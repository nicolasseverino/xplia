"""
MLflow integration for XPLIA.

Log explanations, compliance reports, and trust metrics to MLflow.

Author: XPLIA Team
License: MIT
"""

from typing import Any, Dict, Optional
import tempfile
from pathlib import Path

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ImportError:
    raise ImportError(
        "MLflow not installed. Install with: pip install xplia[mlops]"
    )

from xplia.core.base import ExplanationResult


class XPLIAMLflowLogger:
    """
    Logger for XPLIA explanations and metrics to MLflow.

    Examples
    --------
    >>> logger = XPLIAMLflowLogger()
    >>> logger.log_explanation(explanation, run_id='...')
    """

    def __init__(self, tracking_uri: Optional[str] = None, experiment_name: Optional[str] = None):
        """
        Initialize MLflow logger.

        Parameters
        ----------
        tracking_uri : str, optional
            MLflow tracking server URI.

        experiment_name : str, optional
            MLflow experiment name.
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        if experiment_name:
            mlflow.set_experiment(experiment_name)

        self.client = MlflowClient()

    def log_explanation(
        self,
        explanation: ExplanationResult,
        run_id: Optional[str] = None,
        artifact_path: str = "explanations"
    ):
        """
        Log explanation to MLflow.

        Parameters
        ----------
        explanation : ExplanationResult
            Explanation to log.

        run_id : str, optional
            MLflow run ID. If None, uses active run.

        artifact_path : str
            Path within run's artifact directory.
        """
        if run_id is None:
            run = mlflow.active_run()
            if run is None:
                raise ValueError("No active MLflow run. Call mlflow.start_run() first or provide run_id.")
            run_id = run.info.run_id

        # Log parameters
        mlflow.log_param("explanation_method", explanation.method, run_id=run_id)

        # Log metrics
        if hasattr(explanation, 'quality_metrics') and explanation.quality_metrics:
            for metric_name, metric_value in explanation.quality_metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(f"explanation_{metric_name}", metric_value, run_id=run_id)

        # Log feature importance as artifact
        if 'feature_importance' in explanation.explanation_data:
            import json
            import numpy as np

            importance = explanation.explanation_data['feature_importance']
            feature_names = explanation.explanation_data.get('feature_names', [])

            # Convert to serializable format
            if isinstance(importance, np.ndarray):
                importance = importance.tolist()

            importance_dict = {
                'method': explanation.method,
                'feature_importance': dict(zip(feature_names, importance)) if feature_names else importance
            }

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(importance_dict, f, indent=2)
                temp_path = f.name

            mlflow.log_artifact(temp_path, artifact_path=artifact_path, run_id=run_id)
            Path(temp_path).unlink()

    def log_compliance_report(
        self,
        compliance_result: Any,
        regulation: str,
        run_id: Optional[str] = None,
        artifact_path: str = "compliance"
    ):
        """
        Log compliance report to MLflow.

        Parameters
        ----------
        compliance_result : ComplianceResult
            Compliance check result.

        regulation : str
            Regulation name ('gdpr', 'ai_act', 'hipaa').

        run_id : str, optional
            MLflow run ID.

        artifact_path : str
            Path within run's artifact directory.
        """
        if run_id is None:
            run = mlflow.active_run()
            if run is None:
                raise ValueError("No active MLflow run.")
            run_id = run.info.run_id

        # Log compliance score
        mlflow.log_metric(f"{regulation}_compliance_score", compliance_result.score, run_id=run_id)
        mlflow.log_param(f"{regulation}_compliant", compliance_result.compliant, run_id=run_id)

        # Export and log full report
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / f"{regulation}_compliance_report.pdf"
            compliance_result.export(str(report_path), format='pdf')
            mlflow.log_artifact(str(report_path), artifact_path=artifact_path, run_id=run_id)

    def log_trust_metrics(
        self,
        trust_metrics: Dict[str, float],
        run_id: Optional[str] = None
    ):
        """
        Log trust evaluation metrics to MLflow.

        Parameters
        ----------
        trust_metrics : dict
            Trust metrics (uncertainty, confidence, etc.).

        run_id : str, optional
            MLflow run ID.
        """
        if run_id is None:
            run = mlflow.active_run()
            if run is None:
                raise ValueError("No active MLflow run.")
            run_id = run.info.run_id

        for metric_name, metric_value in trust_metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(f"trust_{metric_name}", metric_value, run_id=run_id)

    def log_fairwashing_detection(
        self,
        fairwashing_result: Any,
        run_id: Optional[str] = None
    ):
        """
        Log fairwashing detection results.

        Parameters
        ----------
        fairwashing_result : FairwashingResult
            Detection result.

        run_id : str, optional
            MLflow run ID.
        """
        if run_id is None:
            run = mlflow.active_run()
            if run is None:
                raise ValueError("No active MLflow run.")
            run_id = run.info.run_id

        mlflow.log_param("fairwashing_detected", fairwashing_result.detected, run_id=run_id)

        if fairwashing_result.detected:
            mlflow.log_param("fairwashing_types", ",".join(fairwashing_result.fairwashing_types), run_id=run_id)
            mlflow.log_metric("fairwashing_severity", fairwashing_result.severity, run_id=run_id)

    def start_run_with_xplia(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Start MLflow run with XPLIA tags.

        Parameters
        ----------
        run_name : str, optional
            Run name.

        tags : dict, optional
            Additional tags.

        Returns
        -------
        ActiveRun
            MLflow active run context.
        """
        default_tags = {
            "xplia.enabled": "true",
            "xplia.version": "1.0.0"
        }

        if tags:
            default_tags.update(tags)

        return mlflow.start_run(run_name=run_name, tags=default_tags)


def log_model_with_explainability(
    model: Any,
    explainer: Any,
    artifact_path: str = "model",
    registered_model_name: Optional[str] = None,
    sample_input: Optional[Any] = None
):
    """
    Log model with explainability artifacts to MLflow.

    Parameters
    ----------
    model : object
        The ML model to log.

    explainer : Explainer
        XPLIA explainer for the model.

    artifact_path : str
        Artifact path in MLflow.

    registered_model_name : str, optional
        Name for model registry.

    sample_input : array-like, optional
        Sample input for signature inference.
    """
    # Log the model
    mlflow.sklearn.log_model(
        model,
        artifact_path=artifact_path,
        registered_model_name=registered_model_name
    )

    # Log explainer configuration
    mlflow.log_param("explainer_method", explainer.method if hasattr(explainer, 'method') else 'unknown')

    # Generate and log sample explanation
    if sample_input is not None:
        explanation = explainer.explain(sample_input)

        logger = XPLIAMLflowLogger()
        logger.log_explanation(explanation)

    print(f"Model and explainability artifacts logged to MLflow")


# Context manager for XPLIA + MLflow
class XPLIAMLflowContext:
    """
    Context manager for XPLIA + MLflow integration.

    Examples
    --------
    >>> with XPLIAMLflowContext(run_name="my_experiment") as logger:
    ...     # Train model
    ...     model.fit(X, y)
    ...
    ...     # Create explainer
    ...     explainer = create_explainer(model, method='shap')
    ...     explanation = explainer.explain(X_test)
    ...
    ...     # Log to MLflow
    ...     logger.log_explanation(explanation)
    """

    def __init__(
        self,
        run_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        tracking_uri: Optional[str] = None
    ):
        self.run_name = run_name
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.logger = None
        self.run = None

    def __enter__(self):
        self.logger = XPLIAMLflowLogger(
            tracking_uri=self.tracking_uri,
            experiment_name=self.experiment_name
        )

        self.run = self.logger.start_run_with_xplia(run_name=self.run_name)
        self.run.__enter__()

        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.run:
            self.run.__exit__(exc_type, exc_val, exc_tb)
