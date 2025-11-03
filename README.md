# XPLIA: The Ultimate State-of-the-Art AI Explainability Library

<div align="center">

![XPLIA Logo](https://via.placeholder.com/800x200/4A90E2/FFFFFF?text=XPLIA+-+Explainable+AI+Library)

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/xplia.svg)](https://pypi.org/project/xplia/)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://xplia.readthedocs.io)
[![CI/CD](https://github.com/nicolasseverino/xplia/workflows/CI/badge.svg)](https://github.com/nicolasseverino/xplia/actions)
[![codecov](https://codecov.io/gh/nicolasseverino/xplia/branch/main/graph/badge.svg)](https://codecov.io/gh/nicolasseverino/xplia)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Downloads](https://pepy.tech/badge/xplia)](https://pepy.tech/project/xplia)

**Production-grade explainability for trustworthy AI systems**

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Examples](#-examples) • [Contributing](#-contributing)

</div>

---

## 🎯 What is XPLIA?

**XPLIA** (eXplainable AI Library) is the most comprehensive, production-ready explainability framework for AI/ML systems. Built for enterprise deployments, XPLIA provides:

- **8+ Explanation Methods**: SHAP, LIME, Counterfactuals, Gradients, Anchors, and more
- **Regulatory Compliance**: GDPR, EU AI Act, HIPAA compliance tools built-in
- **Trust Evaluation**: Uncertainty quantification, fairwashing detection, confidence reports
- **Framework Agnostic**: Works with scikit-learn, TensorFlow, PyTorch, XGBoost, LightGBM, CatBoost
- **Production Ready**: REST API, Docker, Kubernetes, MLflow/W&B integration
- **Enterprise Grade**: Audit trails, compliance reports, multi-audience explanations

**Perfect for:** Financial services, Healthcare, Legal tech, Government, Any regulated industry

---

## ✨ Features

### 🔍 Comprehensive Explainability Methods

<table>
<tr>
<td width="50%">

**Local Explanations**
- **SHAP** - Best for tree models
- **LIME** - Model-agnostic, fast
- **Gradients** - For neural networks
- **Counterfactuals** - "What-if" scenarios
- **Anchors** - Rule-based explanations

</td>
<td width="50%">

**Global Explanations**
- **Feature Importance** - Model-wide patterns
- **Partial Dependence** - Feature effects
- **Unified Explainer** - Combines multiple methods
- **Attention Maps** - For transformers
- **Model Summary** - Complete overview

</td>
</tr>
</table>

### 🏛️ Regulatory Compliance (Industry-First!)

```python
from xplia.compliance import GDPRCompliance, AIActCompliance

# GDPR Right to Explanation
gdpr = GDPRCompliance(model)
dpia_report = gdpr.generate_dpia()  # PDF report ready for auditors

# EU AI Act Risk Assessment
ai_act = AIActCompliance(model, usage='credit_scoring')
risk = ai_act.assess_risk_category()  # Returns 'HIGH', 'MEDIUM', etc.
compliance_report = ai_act.generate_report()  # Full compliance documentation
```

**Supported Regulations:**
- ✅ **GDPR** - Right to explanation, DPIA generation
- ✅ **EU AI Act** - Risk assessment, documentation
- ✅ **HIPAA** - Healthcare compliance
- 🔜 **SOC 2, ISO 27001** (Coming in v1.1)

### 🛡️ Trust & Confidence Evaluation

#### Uncertainty Quantification
Measure prediction confidence with 6 types of uncertainty:

```python
from xplia.explainers.trust import UncertaintyQuantifier

uq = UncertaintyQuantifier(model, explainer)
uncertainty = uq.quantify(X_test)

print(f"Epistemic uncertainty: {uncertainty.epistemic_uncertainty}")
print(f"Aleatoric uncertainty: {uncertainty.aleatoric_uncertainty}")
```

#### Fairwashing Detection (Unique to XPLIA!)
Detect deceptive explanations that hide bias:

```python
from xplia.explainers.trust import FairwashingDetector

detector = FairwashingDetector(model, explainer)
result = detector.detect(X_test, y_test)

if result.detected:
    print(f"⚠️ Fairwashing detected! Types: {result.fairwashing_types}")
    print(f"Severity: {result.severity}")
```

**Detection Types:**
- Feature masking
- Importance shift
- Bias hiding
- Cherry picking
- Threshold manipulation

### 🎨 Advanced Visualizations

```python
from xplia.visualizations import ChartGenerator

generator = ChartGenerator()

# 12+ chart types
generator.create_chart(
    chart_type='waterfall',  # bar, line, pie, heatmap, radar, sankey, etc.
    data=explanation.feature_importance,
    title='Feature Importance',
    theme='dark',  # light, dark, corporate
    export='report.html'  # html, png, pdf, svg
)
```

### 🌐 Multi-Audience Adaptation

Automatic explanation adaptation for different audiences:

```python
from xplia.explainers.calibration import AudienceAdapter

adapter = AudienceAdapter()

# Technical explanation for data scientists
tech_exp = adapter.adapt(explanation, audience='expert')

# Business-friendly explanation for executives
business_exp = adapter.adapt(explanation, audience='basic')

# Public explanation for end users
public_exp = adapter.adapt(explanation, audience='novice')
```

**Audience Levels:**
- 👨‍💼 **Novice** - General public
- 📊 **Basic** - Business stakeholders
- 🔬 **Intermediate** - Analysts
- 🎓 **Advanced** - Data scientists
- 👨‍🔬 **Expert** - ML researchers

---

## 🚀 Installation

### Basic Installation (Lightweight, ~200MB)

```bash
pip install xplia
```

Includes: Core framework, basic visualizations, scikit-learn support

### Full Installation (Recommended, ~2GB)

```bash
pip install xplia[full]
```

Includes everything: All XAI methods, deep learning, boosting, visualizations, APIs, ML Ops

### Custom Installation (Choose what you need)

```bash
# XAI methods only
pip install xplia[xai]

# Deep learning support
pip install xplia[pytorch]  # or tensorflow

# Gradient boosting
pip install xplia[boosting]

# Advanced visualizations
pip install xplia[viz]

# API integrations
pip install xplia[api]

# ML Ops (MLflow, W&B)
pip install xplia[mlops]

# Development tools
pip install xplia[dev]

# Combine multiple
pip install xplia[xai,pytorch,viz,mlops]
```

### From Source (Latest Development)

```bash
git clone https://github.com/nicolasseverino/xplia.git
cd xplia
pip install -e ".[full]"
```

---

## ⚡ Quick Start

### 30-Second Example

```python
from xplia import create_explainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Train a model
X, y = load_iris(return_X_y=True)
model = RandomForestClassifier()
model.fit(X, y)

# Create explainer (auto-detects best method for your model)
explainer = create_explainer(model, method='shap')

# Generate explanations
explanation = explainer.explain(X[:5])

# Access results
print(explanation.feature_importance)
print(explanation.quality_metrics)
```

### Complete Production Example

```python
from xplia import create_explainer, set_config
from xplia.compliance import GDPRCompliance, AIActCompliance
from xplia.explainers.trust import UncertaintyQuantifier, FairwashingDetector

# Configure XPLIA
set_config('verbosity', 'INFO')
set_config('n_jobs', -1)  # Use all CPU cores
set_config('cache_enabled', True)

# Create unified explainer (combines SHAP + LIME + Counterfactuals)
explainer = create_explainer(
    model,
    method='unified',
    methods=['shap', 'lime', 'counterfactual'],
    background_data=X_train
)

# Generate explanations
explanation = explainer.explain(X_test[:10])

# Check regulatory compliance
gdpr = GDPRCompliance(model, model_metadata={
    'name': 'Credit Scoring Model',
    'purpose': 'Loan approval',
    'legal_basis': 'legitimate_interest'
})
gdpr_report = gdpr.generate_dpia()
gdpr_report.export('gdpr_report.pdf')

ai_act = AIActCompliance(model, usage_intent='credit_scoring')
ai_act_report = ai_act.generate_compliance_report()

# Evaluate trust
uq = UncertaintyQuantifier(model, explainer)
uncertainty = uq.quantify(X_test)

detector = FairwashingDetector(model, explainer)
fairwashing = detector.detect(X_test, y_test)

# Generate comprehensive report
from xplia.visualizations import ChartGenerator
chart_gen = ChartGenerator()
chart_gen.create_dashboard(
    explanation,
    uncertainty=uncertainty,
    fairwashing=fairwashing,
    output='complete_report.html'
)
```

---

## 📚 Documentation

### Comprehensive Guides

- **[Installation Guide](docs/INSTALLATION.md)** - Detailed setup for all platforms
- **[Architecture](docs/ARCHITECTURE.md)** - System design and patterns
- **[Plugin Development](docs/PLUGIN_DEVELOPMENT.md)** - Create custom explainers
- **[FAQ](docs/FAQ.md)** - Common questions and troubleshooting
- **[API Reference](https://xplia.readthedocs.io/api/)** - Complete API documentation

### Tutorials

- [Explaining scikit-learn Models](docs/tutorials/sklearn.md)
- [Explaining Deep Learning Models](docs/tutorials/deep_learning.md)
- [GDPR Compliance Workflow](docs/tutorials/gdpr_compliance.md)
- [Production Deployment](docs/tutorials/production.md)

---

## 💡 Examples

### Real-World Use Cases

#### 1. Loan Approval System (Complete Example)

```bash
python examples/loan_approval_system.py
```

Features:
- Model training and evaluation
- Multiple explanation methods
- GDPR and AI Act compliance
- Trust evaluation
- Audit trails
- Production-ready code

#### 2. Healthcare Diagnosis Explainability

```python
# See examples/healthcare_diagnosis.py
```

#### 3. Fraud Detection with Counterfactuals

```python
# See examples/fraud_detection.py
```

### API Integration

#### REST API with FastAPI

```python
from xplia.api import create_api_app

app = create_api_app(models={'my_model': model})

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
```

**Endpoints:**
- `POST /explain` - Generate explanations
- `POST /compliance` - Check compliance
- `POST /trust/evaluate` - Evaluate trust
- `GET /health` - Health check

#### MLflow Integration

```python
from xplia.integrations.mlflow import XPLIAMLflowLogger

with XPLIAMLflowLogger(experiment_name="my_experiment") as logger:
    # Train model
    model.fit(X, y)

    # Log with explanations
    explainer = create_explainer(model)
    explanation = explainer.explain(X_test)
    logger.log_explanation(explanation)
```

#### Weights & Biases Integration

```python
from xplia.integrations.wandb import XPLIAWandBContext

with XPLIAWandBContext(project="my-project") as logger:
    # Train and log
    model.fit(X, y)

    explainer = create_explainer(model)
    explanation = explainer.explain(X_test)
    logger.log_explanation(explanation)
```

---

## 🐳 Deployment

### Docker

```bash
# Build image
docker build -t xplia:latest .

# Run API server
docker run -p 8000:8000 xplia:latest

# Or use docker-compose
docker-compose up
```

### Kubernetes

```bash
# Deploy to Kubernetes
kubectl apply -f kubernetes/deployment.yaml

# Check status
kubectl get pods -l app=xplia

# Access API
kubectl port-forward svc/xplia-api-service 8000:80
```

Features:
- Horizontal Pod Autoscaling
- Health checks
- Persistent volumes
- Load balancing

---

## 🏗️ Architecture

### High-Level Overview

```
User Interface (CLI, API, Notebooks)
            ↓
    Public API Layer
            ↓
     Core Framework
    (Factory, Registry, Config)
            ↓
┌───────────────────────────────────┐
│  Explanation Layer                │
│  (SHAP, LIME, Unified, etc.)     │
├───────────────────────────────────┤
│  Model Adapter Layer              │
│  (sklearn, TF, PyTorch, XGBoost) │
├───────────────────────────────────┤
│  Compliance & Trust Layer         │
│  (GDPR, AI Act, Uncertainty)     │
├───────────────────────────────────┤
│  Visualization Layer              │
│  (Charts, Reports, Dashboards)    │
└───────────────────────────────────┘
```

**Key Design Patterns:**
- **Adapter Pattern** - Unified interface for all ML frameworks
- **Factory Pattern** - Dynamic explainer creation
- **Registry Pattern** - Component discovery and versioning
- **Strategy Pattern** - Runtime algorithm selection

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for complete details.

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=xplia --cov-report=html

# Run specific test suite
pytest tests/explainers/ -v

# Run benchmarks
pytest tests/benchmarks/ -m benchmark
```

**Test Coverage:** 50%+ (6000+ lines of test code)

---

## 🤝 Contributing

We welcome contributions! Please see:

- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute
- **[Code of Conduct](CODE_OF_CONDUCT.md)** - Community standards
- **[Development Setup](CONTRIBUTING.md#development-setup)** - Get started

### Quick Contribution Workflow

```bash
# Fork and clone
git clone https://github.com/YOUR-USERNAME/xplia.git

# Create branch
git checkout -b feature/amazing-feature

# Make changes, add tests, update docs

# Run tests
pytest tests/ -v

# Commit and push
git commit -m "feat: Add amazing feature"
git push origin feature/amazing-feature

# Open Pull Request on GitHub
```

---

## 📊 Comparison with Other Libraries

| Feature | XPLIA | SHAP | LIME | Alibi | InterpretML |
|---------|-------|------|------|-------|-------------|
| Methods | 8+ | 1 | 1 | 5 | 2 |
| GDPR Compliance | ✅ | ❌ | ❌ | ❌ | ❌ |
| Fairwashing Detection | ✅ | ❌ | ❌ | ❌ | ❌ |
| Multi-Audience | ✅ | ❌ | ❌ | ❌ | ❌ |
| REST API | ✅ | ❌ | ❌ | ❌ | ❌ |
| Uncertainty | ✅ | ❌ | ❌ | ✅ | ❌ |
| Production Ready | ✅ | ⚠️ | ⚠️ | ✅ | ⚠️ |
| Test Coverage | 50%+ | 60%+ | 50%+ | 70%+ | 55%+ |

**XPLIA Advantage:** Only library with built-in compliance, fairwashing detection, and production deployment tools.

---

## 🗺️ Roadmap

### v1.1.0 (Q1 2026)
- [ ] Interactive web dashboard (React)
- [ ] Additional compliance (SOC 2, ISO 27001)
- [ ] Automated model monitoring
- [ ] Advanced fairness metrics

### v1.2.0 (Q2 2026)
- [ ] Distributed computing (Spark, Dask)
- [ ] Quantum ML explainability
- [ ] Federated learning support
- [ ] AutoML integration

### v2.0.0 (Q3 2026)
- [ ] Causal inference integration
- [ ] Time series explainability
- [ ] Graph neural network support
- [ ] Multi-language support (R, Julia)

---

## 📜 License

XPLIA is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 📞 Support & Community

- **Documentation**: https://xplia.readthedocs.io
- **GitHub Issues**: https://github.com/nicolasseverino/xplia/issues
- **Discussions**: https://github.com/nicolasseverino/xplia/discussions
- **Email**: contact@xplia.com
- **Twitter**: [@XPLIALib](https://twitter.com/XPLIALib)

---

## 🙏 Acknowledgments

XPLIA is built on the shoulders of giants:
- **SHAP** by Scott Lundberg
- **LIME** by Marco Tulio Ribeiro
- **Alibi** by Seldon
- **InterpretML** by Microsoft

Special thanks to all [contributors](CONTRIBUTORS.md) and the open-source community.

---

## 📖 Citation

If you use XPLIA in your research, please cite:

```bibtex
@software{xplia2025,
  author = {Severino, Nicolas and contributors},
  title = {XPLIA: The Ultimate State-of-the-Art AI Explainability Library},
  url = {https://github.com/nicolasseverino/xplia},
  version = {1.0.0},
  year = {2025},
  doi = {10.5281/zenodo.XXXXXXX}
}
```

---

<div align="center">

**[⭐ Star us on GitHub](https://github.com/nicolasseverino/xplia)** • **[📦 Install Now](https://pypi.org/project/xplia/)** • **[📖 Read the Docs](https://xplia.readthedocs.io)**

Made with ❤️ by the XPLIA Team

</div>
