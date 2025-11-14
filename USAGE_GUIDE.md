# XPLIA 1.0.0 - Complete Usage Guide

**The Ultimate State-of-the-Art AI Explainability Library**

This guide shows you how to use all the advanced features implemented in XPLIA 1.0.0.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Traditional XAI Methods](#traditional-xai-methods)
4. [Advanced Features](#advanced-features)
   - [Causal Inference](#causal-inference)
   - [Certified Explanations](#certified-explanations)
   - [Adversarial XAI](#adversarial-xai)
   - [Privacy-Preserving XAI](#privacy-preserving-xai)
   - [Federated XAI](#federated-xai)
   - [LLM/RAG Explainability](#llmrag-explainability)
   - [Real-Time Streaming XAI](#real-time-streaming-xai)
   - [Advanced Bias Detection](#advanced-bias-detection)
5. [Regulatory Compliance](#regulatory-compliance)
6. [ML Ops Integration](#ml-ops-integration)
7. [Complete Example](#complete-example)

---

## Installation

### Basic Installation

```bash
pip install xplia
```

### Full Installation (Recommended)

```bash
pip install xplia[full]
```

### Selective Installation

```bash
# XAI methods only
pip install xplia[xai]

# Deep learning support
pip install xplia[tensorflow]
pip install xplia[pytorch]

# ML Ops integration
pip install xplia[mlops]

# Development tools
pip install xplia[dev]
```

### Development Installation

```bash
git clone https://github.com/nicolasseverino/xplia.git
cd xplia
pip install -e ".[full]"
```

---

## Quick Start

```python
from xplia import create_explainer
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Train a model
X_train, y_train = ...  # Your training data
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Create explainer
explainer = create_explainer(model, method='shap')

# Explain a prediction
x_test = X_test[0]
explanation = explainer.explain(x_test)

# Access results
print(f"Method: {explanation.method}")
print(f"Feature importance: {explanation.explanation_data['feature_importance']}")
```

---

## Traditional XAI Methods

### SHAP (SHapley Additive exPlanations)

```python
from xplia.explainers.shap import SHAPExplainer

explainer = SHAPExplainer(model, X_train)
explanation = explainer.explain(x_test)

# Visualize
from xplia.visualization import plot_importance
plot_importance(explanation, save_path='shap_explanation.html')
```

### LIME (Local Interpretable Model-agnostic Explanations)

```python
from xplia.explainers.lime import LIMEExplainer

explainer = LIMEExplainer(model, X_train)
explanation = explainer.explain(x_test, num_features=5)
```

### Gradient-Based Methods (Deep Learning)

```python
from xplia.explainers.gradient import IntegratedGradientsExplainer

# For neural networks
explainer = IntegratedGradientsExplainer(
    neural_network,
    baseline='zero',
    n_steps=50
)
explanation = explainer.explain(x_test)
```

---

## Advanced Features

### Causal Inference

**Pearl's Causality Framework with Do-Calculus**

```python
from xplia.explainers.causal import (
    CausalGraph,
    DoCalculus,
    CausalAttributionExplainer
)

# Define causal structure
nodes = ['Feature1', 'Feature2', 'Feature3', 'Outcome']
edges = [
    ('Feature1', 'Outcome'),
    ('Feature2', 'Feature1'),  # Confounder
    ('Feature2', 'Outcome'),
]

causal_graph = CausalGraph(nodes, edges)

# Perform intervention: do(Feature1 = value)
do_calc = DoCalculus(causal_graph)
data_interventional, y_interventional = do_calc.intervention(
    data=X_test,
    intervention={'Feature1': 1.0},
    target='Outcome'
)

# Causal attribution
causal_explainer = CausalAttributionExplainer(model, causal_graph)
causal_explanation = causal_explainer.explain(x_test)

print(f"Causal effects: {causal_explanation.explanation_data['feature_importance']}")
```

**When to use:** When you need to understand **causal** (not just correlational) relationships between features and outcomes. Essential for decision-making and policy interventions.

---

### Certified Explanations

**Formal Guarantees on Explanation Quality**

```python
from xplia.explainers.certified import CertifiedExplainer

# Wrap any explainer with certifier
certified_explainer = CertifiedExplainer(
    model,
    base_explainer,  # e.g., SHAP explainer
    epsilon=0.01  # Perturbation budget
)

certified_exp = certified_explainer.explain_with_certificates(x_test)

# Check certificates
for cert_type, certificate in certified_exp.certificates.items():
    print(f"{cert_type}:")
    print(f"  Certified: {certificate.certified}")
    print(f"  Bound: {certificate.bound}")
    print(f"  Guarantee: {certificate.guarantee}")
```

**Certificates available:**
- **Lipschitz continuity**: Explanation smoothness guarantee
- **L-infinity robustness**: Top-k features stable under perturbations
- **Local stability**: Variance bounds in neighborhoods
- **Monotonicity**: Feature relationships verified

**When to use:** High-stakes applications (healthcare, finance, legal) requiring provable explanation quality.

---

### Adversarial XAI

**Attacks and Defenses on Explanations**

#### Attack Explanations

```python
from xplia.explainers.adversarial import FeatureRankingAttack

# Manipulate feature rankings
attack = FeatureRankingAttack(
    target_feature=5,  # Make this feature appear important
    epsilon=0.1
)

adversarial_instance, attacked_explanation = attack.attack(
    model,
    explainer,
    x_test
)

print(f"Original rankings: {original_rankings}")
print(f"Attacked rankings: {attacked_rankings}")
```

#### Defend Against Attacks

```python
from xplia.explainers.adversarial import (
    EnsembleDefense,
    SmoothDefense,
    AdversarialDetector
)

# Ensemble defense
ensemble = EnsembleDefense([explainer1, explainer2, explainer3])
defended_exp, consensus = ensemble.defend(model, explainer, x_test)

# Smoothing defense
smooth = SmoothDefense(noise_scale=0.1, n_samples=100)
smooth_exp = smooth.defend(model, explainer, x_test)

# Detect adversarial manipulation
detector = AdversarialDetector(method='consistency')
is_adversarial, confidence = detector.detect(x_test, explanation)

if is_adversarial:
    print(f"⚠️  Adversarial manipulation detected (confidence: {confidence})")
```

**When to use:** Security-critical applications, adversarial ML research, robustness testing.

---

### Privacy-Preserving XAI

**Differential Privacy for Explanations**

```python
from xplia.explainers.privacy import (
    PrivacyBudget,
    DPFeatureImportanceExplainer,
    DPAggregatedExplainer
)

# Set privacy budget
budget = PrivacyBudget(epsilon=1.0, delta=1e-5)

# Private explanations
dp_explainer = DPFeatureImportanceExplainer(
    base_explainer=shap_explainer,
    privacy_budget=budget,
    mechanism='gaussian',
    clip_threshold=2.0
)

private_explanation = dp_explainer.explain(
    x_test,
    epsilon_per_query=0.5
)

print(f"Privacy parameters: ε={private_explanation.explanation_data['privacy']['epsilon']}, "
      f"δ={private_explanation.explanation_data['privacy']['delta']}")
print(f"Budget remaining: {budget.remaining()}")

# Aggregate private explanations over dataset
dp_agg = DPAggregatedExplainer(
    base_explainer=shap_explainer,
    privacy_budget=budget,
    n_samples=100
)

aggregated_explanation = dp_agg.aggregate_explanations(X_test[:100])
```

**Privacy mechanisms:**
- **Laplace mechanism**: Pure differential privacy (ε-DP)
- **Gaussian mechanism**: Approximate DP (ε, δ-DP)
- **Exponential mechanism**: For discrete outputs

**When to use:** Healthcare, finance, any domain with sensitive data requiring privacy guarantees.

---

### Federated XAI

**Explanations Without Centralizing Data**

```python
from xplia.explainers.federated import (
    FederatedNode,
    FederatedExplainer,
    FederatedSHAPExplainer
)

# Define federated nodes (e.g., hospitals)
nodes = [
    FederatedNode(
        node_id='Hospital_A',
        data=X_local_A,
        local_model=model_A,
        weight=len(X_local_A)
    ),
    FederatedNode(
        node_id='Hospital_B',
        data=X_local_B,
        local_model=model_B,
        weight=len(X_local_B)
    ),
]

# Create federated explainer
def explainer_factory(model):
    return SHAPExplainer(model)

fed_explainer = FederatedExplainer(
    local_explainer_factory=explainer_factory,
    aggregation_method='weighted_average'  # or 'median', 'consensus'
)

# Get federated explanation
fed_exp = fed_explainer.explain_federated(x_test, nodes)

print(f"Global explanation: {fed_exp.global_explanation.explanation_data['feature_importance']}")
print(f"Local explanations from {len(fed_exp.local_explanations)} nodes")

# Secure aggregation (optional)
from xplia.explainers.federated import FederatedSHAPExplainer

secure_fed_shap = FederatedSHAPExplainer(
    aggregation_method='weighted_average',
    secure=True  # Use cryptographic secure aggregation
)

secure_exp = secure_fed_shap.explain_federated(x_test, nodes)
```

**When to use:** Healthcare consortiums, multi-institutional collaborations, privacy-sensitive deployments.

---

### LLM/RAG Explainability

**Explain Large Language Models and RAG Systems**

#### Attention-Based Explanations

```python
from xplia.explainers.llm import AttentionExplainer

explainer = AttentionExplainer(
    model=llm_model,
    tokenizer=tokenizer,
    layer=-1,  # Last layer
    head=-1    # Averaged across heads
)

token_attribution = explainer.explain(
    "The quick brown fox jumps over the lazy dog"
)

for token, attribution in zip(token_attribution.tokens, token_attribution.attributions):
    print(f"{token}: {attribution:.4f}")
```

#### Integrated Gradients for LLM

```python
from xplia.explainers.llm import IntegratedGradientsLLM

ig_explainer = IntegratedGradientsLLM(
    model=llm_model,
    tokenizer=tokenizer,
    baseline="",  # Empty baseline
    n_steps=50
)

attribution = ig_explainer.explain(
    "Machine learning is transforming healthcare",
    target_token=5  # Explain specific output token
)
```

#### RAG System Explanation

```python
from xplia.explainers.llm import RAGExplainer

rag_explainer = RAGExplainer(
    retriever=retriever_model,
    generator=llm_model,
    tokenizer=tokenizer
)

rag_explanation = rag_explainer.explain(
    query="What is explainable AI?",
    retrieved_docs=[
        "XAI refers to methods that make AI decisions interpretable...",
        "SHAP and LIME are popular XAI techniques...",
        "Explainable AI helps build trust..."
    ],
    response="Explainable AI makes AI decisions interpretable and trustworthy."
)

# Document relevance
for doc, score in zip(rag_explanation.retrieved_docs, rag_explanation.doc_relevance_scores):
    print(f"Score: {score:.4f} - {doc[:50]}...")

# Token attributions in context
print(f"Important context tokens: {rag_explanation.token_attributions.tokens[:5]}")
```

**When to use:** LLM applications, chatbots, RAG systems, prompt engineering.

---

### Real-Time Streaming XAI

**Low-Latency Explanations for Streaming Data**

```python
from xplia.explainers.streaming import RealTimeExplainerPipeline

# Create real-time pipeline
pipeline = RealTimeExplainerPipeline(
    base_explainer=shap_explainer,
    window_size=100,
    enable_drift_detection=True,
    enable_aggregation=True,
    latency_threshold_ms=50.0
)

# Process stream
for x in data_stream:
    result = pipeline.process(x)

    # Check latency
    if result['latency_ms'] > 50:
        print(f"⚠️  High latency: {result['latency_ms']}ms")

    # Check drift
    if result['drift_detected']:
        print(f"🔄 Concept drift detected - consider retraining model")

    # Get explanation
    explanation = result['explanation']

    # Get aggregated statistics
    if result['aggregated_stats']:
        mean_importance = result['aggregated_stats']['mean_importance']

# Get pipeline statistics
stats = pipeline.get_statistics()
print(f"Processed: {stats['total_processed']}")
print(f"Drift rate: {stats['drift_rate']:.2%}")
```

**Components:**
- **IncrementalExplainer**: Efficient incremental updates
- **ApproximateExplainer**: Fast approximate explanations
- **DriftDetector**: Concept drift detection
- **StreamingAggregator**: Rolling statistics

**When to use:** Real-time systems, online learning, IoT, fraud detection, monitoring.

---

### Advanced Bias Detection

**Multi-Level Bias Auditing**

```python
from xplia.explainers.bias import ComprehensiveBiasAuditor

# Define protected attributes
protected_attributes = ['gender', 'race', 'age_group']

# Create auditor
auditor = ComprehensiveBiasAuditor(
    protected_attributes=protected_attributes,
    thresholds={
        'data': 0.1,
        'model': 0.8,
        'explanation': 0.1
    }
)

# Prepare protected attribute data
protected_train = {
    'gender': gender_train,
    'race': race_train,
    'age_group': age_group_train
}

protected_test = {
    'gender': gender_test,
    'race': race_test,
    'age_group': age_group_test
}

# Run comprehensive audit
audit_report = auditor.audit(
    X_train=X_train,
    y_train=y_train,
    protected_train=protected_train,
    model=model,
    X_test=X_test,
    y_test=y_test,
    protected_test=protected_test,
    explanations=test_explanations,  # Optional
    protected_attr_indices={'gender': [5]}  # Feature indices
)

# Review results
print(f"Overall bias detected: {audit_report['summary']['overall_bias_detected']}")

# Data bias
data_report = audit_report['data_bias']
print(f"\nData Bias:")
print(f"  Detected: {data_report.bias_detected}")
print(f"  Level: {data_report.bias_level}")
print(f"  Types: {data_report.bias_types}")
print(f"  Scores: {data_report.bias_scores}")

# Model bias
model_report = audit_report['model_bias']
print(f"\nModel Bias:")
print(f"  Disparate impact: {model_report.bias_scores}")
print(f"  Recommendations:")
for rec in model_report.recommendations:
    print(f"    - {rec}")

# Explanation bias
if 'explanation_bias' in audit_report:
    exp_report = audit_report['explanation_bias']
    print(f"\nExplanation Bias:")
    print(f"  Detected: {exp_report.bias_detected}")
```

**Bias types detected:**
1. **Data bias**: Representation bias, label bias
2. **Model bias**: Disparate impact, equalized odds violations
3. **Explanation bias**: Protected attribute importance, explanation disparity

**When to use:** Fairness-critical applications, regulatory compliance, bias mitigation.

---

## Regulatory Compliance

### GDPR Compliance

```python
from xplia.compliance import GDPRChecker

checker = GDPRChecker()

# Check compliance
compliance_report = checker.check_compliance(
    model=model,
    explainer=explainer,
    X_test=X_test,
    data_info={
        'has_personal_data': True,
        'automated_decision_making': True
    }
)

# Export report
compliance_report.export('gdpr_compliance_report.pdf', format='pdf')

print(f"GDPR Compliant: {compliance_report.compliant}")
print(f"Score: {compliance_report.score}/100")
```

### EU AI Act Compliance

```python
from xplia.compliance import AIActChecker

checker = AIActChecker()

compliance_report = checker.check_compliance(
    model=model,
    explainer=explainer,
    risk_level='high',  # or 'limited', 'minimal'
    use_case='medical_diagnosis'
)

print(f"AI Act Compliant: {compliance_report.compliant}")

# Review requirements
for req in compliance_report.requirements:
    status = "✓" if req['met'] else "✗"
    print(f"{status} {req['name']}: {req['description']}")
```

---

## ML Ops Integration

### MLflow Integration

```python
from xplia.integrations.mlflow_integration import (
    XPLIAMLflowLogger,
    XPLIAMLflowContext
)

# Option 1: Manual logging
import mlflow

mlflow.set_experiment("xplia_explanations")

with mlflow.start_run():
    # Train model
    model.fit(X_train, y_train)

    # Create explainer and explanation
    explainer = create_explainer(model, method='shap')
    explanation = explainer.explain(X_test[0])

    # Log to MLflow
    logger = XPLIAMLflowLogger()
    logger.log_explanation(explanation)
    logger.log_trust_metrics({
        'uncertainty': 0.15,
        'confidence': 0.85
    })

# Option 2: Context manager
with XPLIAMLflowContext(run_name="my_experiment", experiment_name="xplia") as logger:
    # Train model
    model.fit(X_train, y_train)

    # Log everything
    explainer = create_explainer(model, method='shap')
    explanation = explainer.explain(X_test[0])

    logger.log_explanation(explanation)
    logger.log_model_with_explainability(model, explainer, sample_input=X_test[0])
```

### Weights & Biases Integration

```python
from xplia.integrations.wandb_integration import (
    XPLIAWandBLogger,
    XPLIAWandBContext
)

# Context manager approach
with XPLIAWandBContext(
    project="xplia-project",
    name="experiment-1",
    tags=["production", "shap"]
) as logger:
    # Train
    model.fit(X_train, y_train)

    # Explain and log
    explainer = create_explainer(model, method='shap')
    explanation = explainer.explain(X_test[0])

    logger.log_explanation(explanation)
    logger.log_trust_metrics({
        'uncertainty': 0.15,
        'confidence': 0.85
    })

    # Fairwashing detection with alerts
    from xplia.trust import FairwashingDetector

    fw_detector = FairwashingDetector()
    fw_result = fw_detector.detect(model, explainer, X_test)

    logger.log_fairwashing_detection(fw_result)
    # Automatically creates W&B alert if fairwashing detected!
```

---

## Complete Example

Here's a complete end-to-end example using multiple XPLIA features:

```python
"""
Complete XPLIA Workflow Example
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. PREPARE DATA
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Add protected attribute for bias detection
gender = np.random.choice([0, 1], size=len(y), p=[0.4, 0.6])
gender_train, gender_test = gender[:len(y_train)], gender[len(y_train):]

# 2. TRAIN MODEL
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"Test accuracy: {model.score(X_test, y_test):.4f}")

# 3. BASIC EXPLANATION (SHAP)
from xplia import create_explainer

explainer = create_explainer(model, method='shap', data=X_train)
explanation = explainer.explain(X_test[0])

print(f"\nTop 5 features:")
importance = np.array(explanation.explanation_data['feature_importance'])
top_5 = np.argsort(np.abs(importance))[-5:][::-1]
for idx in top_5:
    print(f"  Feature {idx}: {importance[idx]:.4f}")

# 4. CERTIFIED EXPLANATION
from xplia.explainers.certified import CertifiedExplainer

certified_explainer = CertifiedExplainer(model, explainer, epsilon=0.01)
certified_exp = certified_explainer.explain_with_certificates(X_test[0])

print(f"\nCertificates:")
for cert_type, cert in certified_exp.certificates.items():
    print(f"  {cert_type}: Certified={cert.certified}, Bound={cert.bound:.6f}")

# 5. PRIVACY-PRESERVING EXPLANATION
from xplia.explainers.privacy import PrivacyBudget, DPFeatureImportanceExplainer

budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
dp_explainer = DPFeatureImportanceExplainer(explainer, budget)
dp_exp = dp_explainer.explain(X_test[0], epsilon_per_query=0.5)

print(f"\nPrivate explanation: ε={dp_exp.explanation_data['privacy']['epsilon']}")
print(f"Budget remaining: {budget.remaining():.4f}")

# 6. BIAS DETECTION
from xplia.explainers.bias import ComprehensiveBiasAuditor

protected_train = {'gender': gender_train}
protected_test = {'gender': gender_test}

auditor = ComprehensiveBiasAuditor(protected_attributes=['gender'])
audit = auditor.audit(
    X_train, y_train, protected_train,
    model, X_test, y_test, protected_test
)

print(f"\nBias audit:")
print(f"  Overall bias: {audit['summary']['overall_bias_detected']}")
print(f"  Model bias: {audit['model_bias'].bias_detected}")

# 7. ADVERSARIAL ROBUSTNESS
from xplia.explainers.adversarial import EnsembleDefense, AdversarialDetector

ensemble = EnsembleDefense([explainer])
defended_exp, consensus = ensemble.defend(model, explainer, X_test[0])

detector = AdversarialDetector()
is_adv, conf = detector.detect(X_test[0], explanation)

print(f"\nAdversarial robustness:")
print(f"  Ensemble consensus: {consensus:.4f}")
print(f"  Adversarial detected: {is_adv}")

# 8. COMPLIANCE CHECK
from xplia.compliance import GDPRChecker

gdpr_checker = GDPRChecker()
compliance = gdpr_checker.check_compliance(
    model=model,
    explainer=explainer,
    X_test=X_test,
    data_info={'has_personal_data': True}
)

print(f"\nGDPR Compliance:")
print(f"  Compliant: {compliance.compliant}")
print(f"  Score: {compliance.score}/100")

# 9. ML OPS LOGGING (if MLflow available)
try:
    from xplia.integrations.mlflow_integration import XPLIAMLflowLogger
    import mlflow

    mlflow.set_experiment("xplia_complete_example")

    with mlflow.start_run():
        logger = XPLIAMLflowLogger()
        logger.log_explanation(explanation)
        logger.log_compliance_report(compliance, 'gdpr')

        print("\n✓ Logged to MLflow")
except ImportError:
    print("\n⚠️  MLflow not available - skipping logging")

print("\n" + "="*80)
print("COMPLETE EXAMPLE FINISHED!")
print("="*80)
```

---

## Running the Comprehensive Demo

To see ALL features in action:

```bash
cd /home/user/xplia
python examples/comprehensive_xplia_demo.py
```

This will demonstrate:
- ✓ Traditional XAI (SHAP, LIME)
- ✓ Causal inference
- ✓ Certified explanations
- ✓ Adversarial attacks/defenses
- ✓ Privacy-preserving XAI
- ✓ Federated XAI
- ✓ LLM/RAG explainability
- ✓ Real-time streaming
- ✓ Advanced bias detection
- ✓ Regulatory compliance
- ✓ Trust metrics
- ✓ ML Ops integration

---

## Testing Your Installation

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific module tests
pytest tests/explainers/
pytest tests/compliance/
pytest tests/trust/

# Run with coverage
pytest tests/ --cov=xplia --cov-report=html
```

---

## API Documentation

For detailed API documentation, see:

```bash
# Generate API docs
cd docs
make html

# View in browser
open _build/html/index.html
```

Or visit: `docs/ARCHITECTURE.md` for architecture details.

---

## Performance Optimization

### For Large Datasets

```python
# Use approximate explainer for speed
from xplia.explainers.streaming import ApproximateExplainer

approx_explainer = ApproximateExplainer(
    base_explainer,
    approximation_level='high'  # 'low', 'medium', 'high'
)

# Much faster, slightly less accurate
fast_explanation = approx_explainer.explain(x)
```

### For Real-Time Systems

```python
# Pre-compute background data
explainer = create_explainer(
    model,
    method='shap',
    data=X_train[:100]  # Smaller background set
)

# Use streaming pipeline
from xplia.explainers.streaming import RealTimeExplainerPipeline

pipeline = RealTimeExplainerPipeline(
    explainer,
    latency_threshold_ms=50.0
)
```

---

## Troubleshooting

### Common Issues

**Issue: "No module named 'shap'"**
```bash
pip install xplia[xai]
```

**Issue: "MLflow not available"**
```bash
pip install xplia[mlops]
```

**Issue: Slow SHAP explanations**
```python
# Use smaller background dataset
explainer = SHAPExplainer(model, X_train[:100])
```

**Issue: High memory usage**
```python
# Process in batches
for batch in batches(X_test, batch_size=100):
    explanations = [explainer.explain(x) for x in batch]
```

---

## Next Steps

1. **Read the examples**: `examples/` directory
2. **Check the architecture**: `docs/ARCHITECTURE.md`
3. **Contribute**: `CONTRIBUTING.md`
4. **Report issues**: GitHub Issues

---

## Support

- **Documentation**: `docs/`
- **GitHub**: https://github.com/nicolasseverino/xplia
- **Issues**: https://github.com/nicolasseverino/xplia/issues
- **Email**: contact@xplia.com

---

**XPLIA 1.0.0** - The Ultimate AI Explainability Library

*Making AI Transparent, Trustworthy, and Compliant*
