"""
FastAPI integration for XPLIA.

Provides REST API endpoints for model explanations, compliance checks,
and trust evaluation.

Usage:
    uvicorn xplia.api.fastapi_app:app --host 0.0.0.0 --port 8000

Author: XPLIA Team
License: MIT
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import numpy as np
import pandas as pd

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Body
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field, validator
except ImportError:
    raise ImportError(
        "FastAPI not installed. Install with: pip install xplia[api]"
    )

from xplia import create_explainer, __version__
from xplia.core.config import ConfigManager


# Pydantic models for request/response
class ModelMetadata(BaseModel):
    """Model metadata."""
    name: str
    version: str
    framework: str
    task_type: str = Field(..., description="'classification' or 'regression'")


class ExplainRequest(BaseModel):
    """Request for explanation."""
    model_id: str = Field(..., description="Registered model ID")
    instances: List[List[float]] = Field(..., description="Instances to explain")
    method: str = Field(default="unified", description="Explanation method")
    feature_names: Optional[List[str]] = None
    return_visualizations: bool = False

    @validator('method')
    def validate_method(cls, v):
        valid_methods = ['shap', 'lime', 'unified', 'gradient', 'counterfactual', 'feature_importance']
        if v not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        return v


class ExplainResponse(BaseModel):
    """Response from explanation."""
    request_id: str
    model_id: str
    method: str
    explanations: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    timestamp: str


class ComplianceRequest(BaseModel):
    """Request for compliance check."""
    model_id: str
    regulation: str = Field(..., description="'gdpr', 'ai_act', or 'hipaa'")
    model_metadata: Dict[str, Any]

    @validator('regulation')
    def validate_regulation(cls, v):
        if v not in ['gdpr', 'ai_act', 'hipaa']:
            raise ValueError("Regulation must be 'gdpr', 'ai_act', or 'hipaa'")
        return v


class ComplianceResponse(BaseModel):
    """Response from compliance check."""
    request_id: str
    model_id: str
    regulation: str
    compliant: bool
    compliance_score: float
    requirements: List[Dict[str, Any]]
    recommendations: List[str]
    timestamp: str


class TrustEvaluationRequest(BaseModel):
    """Request for trust evaluation."""
    model_id: str
    instances: List[List[float]]
    evaluate_uncertainty: bool = True
    detect_fairwashing: bool = True
    feature_names: Optional[List[str]] = None


class TrustEvaluationResponse(BaseModel):
    """Response from trust evaluation."""
    request_id: str
    model_id: str
    overall_trust_score: float
    uncertainty_metrics: Optional[Dict[str, float]] = None
    fairwashing_detected: bool = False
    fairwashing_details: Optional[Dict[str, Any]] = None
    confidence_intervals: Optional[Dict[str, Any]] = None
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    uptime_seconds: float
    models_loaded: int


# Create FastAPI app
app = FastAPI(
    title="XPLIA API",
    description="RESTful API for AI Explainability, Compliance, and Trust Evaluation",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global state
class AppState:
    """Application state manager."""
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.explainers: Dict[str, Any] = {}
        self.start_time = datetime.now()
        self.request_count = 0

    def register_model(self, model_id: str, model: Any, metadata: ModelMetadata):
        """Register a model."""
        self.models[model_id] = {
            'model': model,
            'metadata': metadata,
            'registered_at': datetime.now().isoformat()
        }

    def get_model(self, model_id: str):
        """Get a registered model."""
        if model_id not in self.models:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        return self.models[model_id]['model']

    def get_explainer(self, model_id: str, method: str):
        """Get or create explainer."""
        key = f"{model_id}_{method}"
        if key not in self.explainers:
            model = self.get_model(model_id)
            self.explainers[key] = create_explainer(model, method=method)
        return self.explainers[key]


state = AppState()


# Utility functions
def generate_request_id() -> str:
    """Generate unique request ID."""
    from uuid import uuid4
    return str(uuid4())


# Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "name": "XPLIA API",
        "version": __version__,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = (datetime.now() - state.start_time).total_seconds()
    return HealthResponse(
        status="healthy",
        version=__version__,
        uptime_seconds=uptime,
        models_loaded=len(state.models)
    )


@app.post("/models/register")
async def register_model(
    model_id: str = Query(..., description="Unique model identifier"),
    metadata: ModelMetadata = Body(...)
):
    """
    Register a new model.

    Note: This is a simplified version. In production, you'd load the actual
    model from a file or model registry.
    """
    if model_id in state.models:
        raise HTTPException(status_code=400, detail=f"Model {model_id} already registered")

    # In production, load actual model here
    # For now, we'll simulate with a placeholder
    state.register_model(model_id, None, metadata)

    return {
        "model_id": model_id,
        "status": "registered",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/models")
async def list_models():
    """List all registered models."""
    return {
        "models": [
            {
                "model_id": model_id,
                "metadata": data['metadata'].dict(),
                "registered_at": data['registered_at']
            }
            for model_id, data in state.models.items()
        ],
        "count": len(state.models)
    }


@app.post("/explain", response_model=ExplainResponse)
async def explain(request: ExplainRequest, background_tasks: BackgroundTasks):
    """
    Generate explanations for model predictions.

    This endpoint accepts instances and returns explanations using the
    specified method.
    """
    request_id = generate_request_id()
    state.request_count += 1

    try:
        # Get model and explainer
        model = state.get_model(request.model_id)
        explainer = state.get_explainer(request.model_id, request.method)

        # Convert instances to numpy array
        X = np.array(request.instances)

        # Generate explanations
        explanation_result = explainer.explain(X)

        # Format response
        explanations = []
        for i in range(len(X)):
            exp = {
                "instance_index": i,
                "feature_importance": explanation_result.explanation_data['feature_importance'].tolist() if isinstance(explanation_result.explanation_data['feature_importance'], np.ndarray) else explanation_result.explanation_data['feature_importance'],
                "feature_names": request.feature_names or explanation_result.explanation_data.get('feature_names', []),
            }

            # Add prediction if available
            if 'predictions' in explanation_result.explanation_data:
                exp['prediction'] = float(explanation_result.explanation_data['predictions'][i])

            explanations.append(exp)

        return ExplainResponse(
            request_id=request_id,
            model_id=request.model_id,
            method=request.method,
            explanations=explanations,
            metadata={
                "quality_metrics": explanation_result.quality_metrics if hasattr(explanation_result, 'quality_metrics') else {},
                "explanation_method": request.method,
                "n_instances": len(X)
            },
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@app.post("/compliance", response_model=ComplianceResponse)
async def check_compliance(request: ComplianceRequest):
    """
    Check regulatory compliance.

    Supports GDPR, EU AI Act, and HIPAA compliance checks.
    """
    request_id = generate_request_id()
    state.request_count += 1

    try:
        model = state.get_model(request.model_id)

        # Import appropriate compliance module
        if request.regulation == 'gdpr':
            from xplia.compliance.gdpr import GDPRCompliance
            compliance_checker = GDPRCompliance(model, model_metadata=request.model_metadata)
        elif request.regulation == 'ai_act':
            from xplia.compliance.ai_act import AIActCompliance
            compliance_checker = AIActCompliance(model, model_metadata=request.model_metadata)
        elif request.regulation == 'hipaa':
            from xplia.compliance.hipaa import HIPAACompliance
            compliance_checker = HIPAACompliance(model, model_metadata=request.model_metadata)
        else:
            raise ValueError(f"Unknown regulation: {request.regulation}")

        # Check compliance
        compliance_result = compliance_checker.check_compliance()

        return ComplianceResponse(
            request_id=request_id,
            model_id=request.model_id,
            regulation=request.regulation,
            compliant=compliance_result.compliant,
            compliance_score=compliance_result.score,
            requirements=compliance_result.requirements,
            recommendations=compliance_result.recommendations,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compliance check failed: {str(e)}")


@app.post("/trust/evaluate", response_model=TrustEvaluationResponse)
async def evaluate_trust(request: TrustEvaluationRequest):
    """
    Evaluate model trustworthiness.

    Includes uncertainty quantification and fairwashing detection.
    """
    request_id = generate_request_id()
    state.request_count += 1

    try:
        model = state.get_model(request.model_id)
        X = np.array(request.instances)

        # Initialize trust evaluators
        uncertainty_metrics = None
        if request.evaluate_uncertainty:
            from xplia.explainers.trust.uncertainty import UncertaintyQuantifier
            explainer = state.get_explainer(request.model_id, 'shap')
            uq = UncertaintyQuantifier(model, explainer)
            uncertainty_result = uq.quantify(X)
            uncertainty_metrics = {
                "global_uncertainty": float(uncertainty_result.global_uncertainty),
                "aleatoric_uncertainty": float(uncertainty_result.aleatoric_uncertainty),
                "epistemic_uncertainty": float(uncertainty_result.epistemic_uncertainty)
            }

        fairwashing_detected = False
        fairwashing_details = None
        if request.detect_fairwashing:
            from xplia.explainers.trust.fairwashing import FairwashingDetector
            explainer = state.get_explainer(request.model_id, 'shap')
            detector = FairwashingDetector(model, explainer)
            fairwashing_result = detector.detect(X)
            fairwashing_detected = fairwashing_result.detected
            if fairwashing_detected:
                fairwashing_details = {
                    "types": fairwashing_result.fairwashing_types,
                    "severity": fairwashing_result.severity,
                    "description": fairwashing_result.description
                }

        # Calculate overall trust score
        trust_score = 1.0
        if uncertainty_metrics:
            trust_score *= (1 - uncertainty_metrics['global_uncertainty'])
        if fairwashing_detected:
            trust_score *= 0.5  # Penalize for fairwashing

        return TrustEvaluationResponse(
            request_id=request_id,
            model_id=request.model_id,
            overall_trust_score=float(trust_score),
            uncertainty_metrics=uncertainty_metrics,
            fairwashing_detected=fairwashing_detected,
            fairwashing_details=fairwashing_details,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trust evaluation failed: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get API statistics."""
    uptime = (datetime.now() - state.start_time).total_seconds()
    return {
        "uptime_seconds": uptime,
        "total_requests": state.request_count,
        "models_registered": len(state.models),
        "explainers_cached": len(state.explainers),
        "requests_per_second": state.request_count / max(uptime, 1)
    }


def create_app(models: Optional[Dict[str, Any]] = None) -> FastAPI:
    """
    Create and configure FastAPI app.

    Parameters
    ----------
    models : dict, optional
        Pre-registered models {model_id: model_object}

    Returns
    -------
    FastAPI
        Configured app instance.
    """
    if models:
        for model_id, model in models.items():
            state.register_model(
                model_id,
                model,
                ModelMetadata(
                    name=model_id,
                    version="1.0.0",
                    framework="sklearn",
                    task_type="classification"
                )
            )

    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
