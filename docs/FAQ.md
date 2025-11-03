# XPLIA Frequently Asked Questions (FAQ)

**Version:** 1.0.0
**Last Updated:** November 2025

---

## General Questions

### What is XPLIA?

XPLIA (eXplainable AI Library) is a production-grade explainability framework for AI systems. It provides:
- Multiple explanation methods (SHAP, LIME, counterfactuals, etc.)
- Regulatory compliance tools (GDPR, AI Act, HIPAA)
- Trust evaluation (uncertainty quantification, fairwashing detection)
- Framework-agnostic support (sklearn, TensorFlow, PyTorch, XGBoost, etc.)

---

### Is XPLIA free to use?

Yes! XPLIA is open-source and released under the MIT License. You can use it freely for commercial and non-commercial projects.

---

### Which Python versions are supported?

XPLIA supports Python 3.8, 3.9, 3.10, and 3.11. Python 3.12 support is experimental.

---

### Can I use XPLIA in production?

Yes! XPLIA v1.0.0 is production-stable with:
- Comprehensive test coverage (50%+)
- Performance optimizations
- Audit trails for compliance
- Scalability for large datasets

---

## Installation Questions

### How do I install XPLIA?

```bash
# Basic installation
pip install xplia

# Full installation (recommended)
pip install xplia[full]

# Custom installation
pip install xplia[xai,pytorch,viz]
```

See [INSTALLATION.md](INSTALLATION.md) for details.

---

### Why is the full installation so large?

The full installation includes:
- TensorFlow (~500 MB)
- PyTorch (~800 MB)
- XGBoost, LightGBM, CatBoost
- SHAP, LIME, Alibi
- All visualization dependencies

Use custom installation to install only what you need:
```bash
pip install xplia[xai,viz]  # Skip deep learning frameworks
```

---

### Can I install XPLIA without TensorFlow/PyTorch?

Yes! By default, XPLIA only installs core dependencies. TensorFlow and PyTorch are optional:

```bash
# No deep learning
pip install xplia

# Only PyTorch
pip install xplia[pytorch]

# Only TensorFlow
pip install xplia[tensorflow]
```

---

### How do I fix "ImportError: No module named 'shap'"?

Install XAI dependencies:
```bash
pip install xplia[xai]
```

---

## Usage Questions

### Which explainer should I use?

| Use Case | Recommended Method | Reason |
|----------|-------------------|--------|
| Fast, any model | `feature_importance` | Fastest, works everywhere |
| Tree models | `shap` | Best accuracy for trees |
| Any model, interpretable | `lime` | Model-agnostic, fast |
| Neural networks | `gradient` | Gradient-based, accurate |
| What-if analysis | `counterfactual` | Actionable insights |
| Comprehensive | `unified` | Combines multiple methods |

---

### How do I explain a scikit-learn model?

```python
from xplia import create_explainer
from sklearn.ensemble import RandomForestClassifier

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Create explainer
explainer = create_explainer(model, method='shap')

# Explain predictions
explanation = explainer.explain(X_test[:10])

# Access feature importance
print(explanation.explanation_data['feature_importance'])
```

---

### How do I explain a PyTorch model?

```python
import torch
from xplia import create_explainer

# Your PyTorch model
class MyModel(torch.nn.Module):
    pass

model = MyModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Create explainer
explainer = create_explainer(
    model,
    method='gradient',
    task='classification'
)

# Explain
explanation = explainer.explain(X_test)
```

---

### How do I generate a compliance report?

```python
from xplia.compliance import ComplianceChecker

checker = ComplianceChecker(regulation='gdpr')
report = checker.generate_report(
    model=model,
    model_metadata={
        'name': 'Credit Scoring Model',
        'purpose': 'Loan approval',
        'data_sources': ['customer_data.csv']
    }
)

# Export to PDF
report.export('compliance_report.pdf', format='pdf')
```

---

### Can I use multiple explanation methods together?

Yes! Use the Unified Explainer:

```python
explainer = create_explainer(
    model,
    method='unified',
    methods=['shap', 'lime', 'counterfactual']
)

explanation = explainer.explain(X_test)

# Access individual explanations
shap_exp = explanation.explanation_data['shap']
lime_exp = explanation.explanation_data['lime']
```

---

## Performance Questions

### SHAP is too slow. How can I speed it up?

1. **Reduce background samples:**
```python
explainer = create_explainer(
    model,
    method='shap',
    n_samples=50  # Default is 100
)
```

2. **Use sampling:**
```python
explainer = create_explainer(
    model,
    method='shap',
    approximate=True
)
```

3. **Enable caching:**
```python
from xplia import set_config
set_config('cache_enabled', True)
```

4. **Use parallel processing:**
```python
set_config('n_jobs', -1)  # Use all cores
```

---

### How do I handle large datasets?

1. **Enable chunked processing:**
```python
set_config('chunk_size', 500)
```

2. **Sample the dataset:**
```python
# Explain a sample, not entire dataset
X_sample = X_test[:100]
explanation = explainer.explain(X_sample)
```

3. **Use faster methods:**
```python
# LIME is faster than SHAP
explainer = create_explainer(model, method='lime')
```

---

### Can I use GPU acceleration?

Yes, if using TensorFlow or PyTorch models:

```python
# PyTorch - model automatically uses GPU if available
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

explainer = create_explainer(model, method='gradient')
```

---

## Compliance Questions

### Does XPLIA help with GDPR compliance?

Yes! XPLIA provides:
- Right to explanation tools
- Data Processing Impact Assessment (DPIA) generation
- Audit trail recording
- Automated compliance reports

```python
from xplia.compliance.gdpr import GDPRCompliance

gdpr = GDPRCompliance(model)
report = gdpr.generate_dpia()
report.export('dpia_report.pdf')
```

---

### What about EU AI Act compliance?

Yes! XPLIA includes AI Act assessment:

```python
from xplia.compliance.ai_act import AIActCompliance

ai_act = AIActCompliance(model, usage_intent='credit_scoring')
risk_category = ai_act.assess_risk_category()  # Returns 'HIGH', etc.
report = ai_act.generate_compliance_report()
```

---

### Can XPLIA detect biased explanations (fairwashing)?

Yes! Use the fairwashing detector:

```python
from xplia.explainers.trust.fairwashing import FairwashingDetector

detector = FairwashingDetector(model, explainer)
result = detector.detect_fairwashing(X_test, y_test)

if result.detected:
    print(f"Fairwashing detected: {result.fairwashing_types}")
    print(f"Severity: {result.severity}")
```

---

## Integration Questions

### Can I use XPLIA with MLflow?

Yes! (Coming in v1.1.0 - integration is included in this release)

```python
from xplia.integrations.mlflow import XPLIAMLflowLogger

logger = XPLIAMLflowLogger()
logger.log_explanation(explanation, run_id='...')
```

---

### How do I deploy XPLIA as a REST API?

Use the FastAPI integration:

```python
from xplia.api import create_api_app

app = create_api_app(
    models={'credit_model': model},
    explainers={'shap': shap_explainer}
)

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
```

---

### Can I use XPLIA in a Jupyter notebook?

Yes! XPLIA works great in notebooks:

```python
import xplia
from xplia import create_explainer

# Create explanations
explainer = create_explainer(model, method='shap')
explanation = explainer.explain(X_test[:5])

# Visualize inline
from xplia.visualizations import ChartGenerator

chart_gen = ChartGenerator()
chart_gen.plot_feature_importance(explanation)
```

---

### Does XPLIA support Docker?

Yes! See [examples/docker/](../examples/docker/) for Dockerfiles.

```bash
docker pull nicolasseverino/xplia:latest
docker run -it nicolasseverino/xplia python
```

---

## Troubleshooting

### I get "MemoryError" with large datasets

**Solution:**
1. Enable chunked processing:
```python
set_config('chunk_size', 500)
```

2. Reduce samples:
```python
explainer = create_explainer(model, method='shap', n_samples=50)
```

3. Use sampling:
```python
X_sampled = X_test.sample(n=1000)
explanation = explainer.explain(X_sampled)
```

---

### Explanations are inconsistent between runs

**Solution:**
1. Set random seed:
```python
explainer = create_explainer(model, method='lime', random_state=42)
```

2. Increase sampling:
```python
explainer = create_explainer(model, method='lime', n_samples=10000)
```

---

### SHAP fails with "Additivity check failed"

**Solution:**
Disable additivity check:
```python
explainer = create_explainer(
    model,
    method='shap',
    check_additivity=False
)
```

---

### Visualizations don't render in Jupyter

**Solution:**
```python
# Enable inline plotting
%matplotlib inline

# Or use plotly for interactive
import plotly.io as pio
pio.renderers.default = 'notebook'
```

---

### "AttributeError: 'Model' object has no attribute 'predict_proba'"

**Solution:**

Your model doesn't support probability predictions. Use regression explainer or add predict_proba method:

```python
class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        # Convert predictions to probabilities
        preds = self.model.predict(X)
        # Implement conversion logic
        return np.column_stack([1-preds, preds])

wrapped_model = ModelWrapper(model)
explainer = create_explainer(wrapped_model, method='lime')
```

---

## Feature Requests

### Can I request a new feature?

Yes! Please:
1. Check existing issues: https://github.com/nicolasseverino/xplia/issues
2. Create a new issue with:
   - Clear description
   - Use case
   - Example code (if applicable)

---

### How do I contribute?

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

---

### Is there a roadmap?

Yes! See [ROADMAP.md](ROADMAP.md) for planned features.

---

## Best Practices

### Should I explain every prediction?

No. Best practices:
- **Development**: Explain samples to understand model behavior
- **Production**: Explain on-demand when users request explanations
- **Monitoring**: Periodically explain random samples to detect drift

---

### How many instances should I explain?

Depends on use case:
- **Individual explanations**: 1 instance
- **Understanding patterns**: 50-100 instances
- **Global understanding**: 500-1000 instances or use `explain_model()`

---

### Should I cache explanations?

Yes, for production:
```python
set_config('cache_enabled', True)
set_config('cache_dir', '/path/to/cache')
```

---

## Still have questions?

- **Documentation**: https://xplia.readthedocs.io
- **GitHub Discussions**: https://github.com/nicolasseverino/xplia/discussions
- **Discord**: [Join our community]
- **Email**: contact@xplia.com
