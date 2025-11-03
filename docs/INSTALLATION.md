# XPLIA Installation Guide

**Version:** 1.0.0
**Last Updated:** November 2025

---

## Table of Contents

1. [Requirements](#requirements)
2. [Installation Methods](#installation-methods)
3. [Optional Dependencies](#optional-dependencies)
4. [Verification](#verification)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)
7. [Platform-Specific Instructions](#platform-specific-instructions)
8. [Development Installation](#development-installation)

---

## Requirements

### Minimum Requirements

- **Python:** 3.8 or higher
- **RAM:** 4 GB minimum, 8 GB recommended
- **Disk Space:** 2 GB for full installation
- **OS:** Linux, macOS, Windows

### Python Version Compatibility

| Python Version | Support Status |
|----------------|----------------|
| 3.8            | ✅ Fully supported |
| 3.9            | ✅ Fully supported |
| 3.10           | ✅ Fully supported |
| 3.11           | ✅ Fully supported |
| 3.12           | ⚠️ Experimental |
| < 3.8          | ❌ Not supported |

---

## Installation Methods

### Method 1: pip (Recommended)

#### Basic Installation (Lightweight)

Installs only core dependencies (~200 MB):

```bash
pip install xplia
```

**What's included:**
- Core framework
- Basic visualizations (matplotlib, plotly)
- Configuration management
- Model adapters (scikit-learn)

---

#### Full Installation (Recommended for most users)

Installs all features (~2 GB):

```bash
pip install xplia[full]
```

**What's included:**
- All core features
- All XAI methods (SHAP, LIME, Alibi, etc.)
- Deep learning support (TensorFlow, PyTorch)
- Gradient boosting (XGBoost, LightGBM, CatBoost)
- Advanced visualizations
- API integrations (FastAPI, Flask)
- ML Ops tools (MLflow, W&B)

---

#### Custom Installation (Choose what you need)

Install only the components you need:

```bash
# Core + XAI methods only
pip install xplia[xai]

# Core + Deep learning (TensorFlow)
pip install xplia[tensorflow]

# Core + Deep learning (PyTorch)
pip install xplia[pytorch]

# Core + Gradient boosting
pip install xplia[boosting]

# Core + Advanced visualizations
pip install xplia[viz]

# Core + API support
pip install xplia[api]

# Core + ML Ops integrations
pip install xplia[mlops]

# Combine multiple
pip install xplia[xai,pytorch,viz,api]
```

---

### Method 2: conda

```bash
# Create a new environment (recommended)
conda create -n xplia python=3.10
conda activate xplia

# Install XPLIA
conda install -c conda-forge xplia
```

---

### Method 3: From Source (Latest Development Version)

```bash
# Clone the repository
git clone https://github.com/nicolasseverino/xplia.git
cd xplia

# Install in development mode
pip install -e .

# Or install with all extras
pip install -e ".[full]"
```

---

## Optional Dependencies

### XAI Methods

```bash
pip install xplia[xai]
```

**Includes:**
- SHAP (≥0.40.0)
- LIME (≥0.2.0)
- Alibi (≥0.7.0)
- InterpretML (≥0.2.7)

**Use when:** You want to use advanced XAI methods beyond basic feature importance.

---

### Deep Learning Frameworks

#### TensorFlow

```bash
pip install xplia[tensorflow]
```

**Includes:**
- TensorFlow (≥2.6.0)
- Transformers (≥4.11.0) for NLP models

**Use when:** Explaining Keras/TensorFlow models, including neural networks, CNNs, RNNs, Transformers.

---

#### PyTorch

```bash
pip install xplia[pytorch]
```

**Includes:**
- PyTorch (≥1.9.0)
- TorchVision (≥0.10.0)
- Transformers (≥4.11.0) for NLP models

**Use when:** Explaining PyTorch models.

---

### Gradient Boosting

```bash
pip install xplia[boosting]
```

**Includes:**
- XGBoost (≥1.5.0)
- LightGBM (≥3.3.0)
- CatBoost (≥1.0.0)

**Use when:** Explaining XGBoost, LightGBM, or CatBoost models.

---

### Visualizations

```bash
pip install xplia[viz]
```

**Includes:**
- Seaborn (≥0.11.0) for statistical plots
- Dash (≥2.0.0) for interactive dashboards
- ReportLab (≥3.6.0) for PDF generation
- Kaleido (≥0.2.0) for static image export

**Use when:** You need advanced visualizations, PDF reports, or interactive dashboards.

---

### API & Web Integration

```bash
pip install xplia[api]
```

**Includes:**
- Flask (≥2.0.0)
- FastAPI (≥0.68.0)
- Uvicorn (≥0.15.0)
- Pydantic (≥1.8.0)

**Use when:** Deploying XPLIA as a REST API or integrating with web applications.

---

### ML Ops Integration

```bash
pip install xplia[mlops]
```

**Includes:**
- MLflow (≥1.20.0)
- Weights & Biases (≥0.12.0)

**Use when:** Integrating with ML experiment tracking platforms.

---

### Development Tools

```bash
pip install xplia[dev]
```

**Includes:**
- pytest, pytest-cov (testing)
- flake8, black, isort (code quality)
- mypy (type checking)
- Sphinx (documentation)
- pre-commit (git hooks)
- bandit (security linting)

**Use when:** Contributing to XPLIA or developing extensions.

---

## Verification

### Verify Installation

```python
import xplia
print(xplia.__version__)  # Should print: 1.0.0
```

---

### Check Installed Components

```bash
xplia version
```

**Expected output:**
```
XPLIA version 1.0.0
The Ultimate State-of-the-Art AI Explainability Library

Installed components:
  ✓ SHAP
  ✓ LIME
  ✓ TensorFlow
  ✓ PyTorch
  ✓ XGBoost
  ✓ LightGBM
  ✓ CatBoost
  ✓ FastAPI
  ✓ MLflow
  ✓ W&B
```

---

### Run Basic Test

```python
from xplia import create_explainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Create simple model
X, y = make_classification(n_samples=100, n_features=10, random_state=42)
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# Create explainer
explainer = create_explainer(model, method='feature_importance')
explanation = explainer.explain(X[:5])

print("✓ XPLIA is working correctly!")
```

---

## Configuration

### Default Configuration

XPLIA works out-of-the-box with sensible defaults. To customize:

```python
from xplia import set_config

set_config('verbosity', 'INFO')  # DEBUG, INFO, WARNING, ERROR
set_config('n_jobs', -1)  # Use all CPU cores
set_config('cache_enabled', True)  # Enable result caching
set_config('audit_trail_enabled', True)  # Enable audit logging
```

---

### Configuration File

Create `~/.xplia/config.yaml`:

```yaml
# XPLIA Configuration

general:
  verbosity: INFO
  n_jobs: -1
  cache_enabled: true
  cache_dir: ~/.xplia/cache

explainers:
  default_method: unified
  shap:
    n_samples: 100
    check_additivity: false
  lime:
    n_samples: 5000
    n_features: 10

compliance:
  default_regulation: gdpr
  audit_trail_enabled: true
  audit_trail_dir: ~/.xplia/audit_logs

performance:
  parallel_enabled: true
  chunk_size: 1000
  memory_limit_gb: 8

visualization:
  default_theme: light
  default_format: html
  dpi: 300
```

---

## Troubleshooting

### Common Issues

#### Issue 1: ImportError for optional dependencies

**Error:**
```
ImportError: No module named 'shap'
```

**Solution:**
```bash
pip install xplia[xai]  # Install XAI dependencies
```

---

#### Issue 2: TensorFlow GPU not detected

**Error:**
```
TensorFlow not compiled with CUDA support
```

**Solution:**
```bash
# Uninstall CPU version
pip uninstall tensorflow

# Install GPU version
pip install tensorflow-gpu>=2.6.0
```

Verify:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

---

#### Issue 3: Memory issues with large datasets

**Error:**
```
MemoryError: Unable to allocate array
```

**Solution:**

1. Enable chunked processing:
```python
set_config('chunk_size', 500)  # Process in smaller batches
```

2. Reduce sampling:
```python
explainer = create_explainer(
    model,
    method='shap',
    n_samples=50  # Reduce from default 100
)
```

3. Use sampling for SHAP:
```python
explainer = create_explainer(
    model,
    method='shap',
    approximate=True,
    max_evals=100
)
```

---

#### Issue 4: Slow performance

**Solution:**

1. Enable parallel processing:
```python
set_config('n_jobs', -1)  # Use all cores
```

2. Enable caching:
```python
set_config('cache_enabled', True)
```

3. Use faster methods:
```python
# LIME is faster than SHAP for large datasets
explainer = create_explainer(model, method='lime')
```

---

#### Issue 5: Permission errors on Windows

**Error:**
```
PermissionError: [WinError 5] Access is denied
```

**Solution:**

Run as administrator or install in user directory:
```bash
pip install --user xplia[full]
```

---

### Getting Help

If you encounter issues not covered here:

1. **Check the FAQ:** [docs/FAQ.md](FAQ.md)
2. **Search existing issues:** https://github.com/nicolasseverino/xplia/issues
3. **Create a new issue:** https://github.com/nicolasseverino/xplia/issues/new
4. **Join our community:** [Discord/Slack link]

When reporting issues, include:
- Python version: `python --version`
- XPLIA version: `xplia version`
- Operating system
- Full error traceback
- Minimal reproducible example

---

## Platform-Specific Instructions

### Linux

#### Ubuntu/Debian

```bash
# Update package list
sudo apt-get update

# Install Python 3.10 (if not installed)
sudo apt-get install python3.10 python3.10-venv python3-pip

# Create virtual environment
python3.10 -m venv xplia-env
source xplia-env/bin/activate

# Install XPLIA
pip install xplia[full]
```

---

#### CentOS/RHEL

```bash
# Install Python 3.10
sudo yum install python310 python310-pip

# Create virtual environment
python3.10 -m venv xplia-env
source xplia-env/bin/activate

# Install XPLIA
pip install xplia[full]
```

---

### macOS

#### Using Homebrew

```bash
# Install Python 3.10
brew install python@3.10

# Create virtual environment
python3.10 -m venv xplia-env
source xplia-env/bin/activate

# Install XPLIA
pip install xplia[full]
```

#### Apple Silicon (M1/M2/M3)

```bash
# Use miniforge for better ARM support
brew install miniforge
conda init zsh  # or bash

# Create environment
conda create -n xplia python=3.10
conda activate xplia

# Install XPLIA
pip install xplia[full]
```

---

### Windows

#### Using Python.org installer

1. Download Python 3.10 from https://python.org
2. During installation, check "Add Python to PATH"
3. Open Command Prompt or PowerShell:

```powershell
# Create virtual environment
python -m venv xplia-env
xplia-env\Scripts\activate

# Install XPLIA
pip install xplia[full]
```

---

#### Using Anaconda

```powershell
# Open Anaconda Prompt

# Create environment
conda create -n xplia python=3.10
conda activate xplia

# Install XPLIA
pip install xplia[full]
```

---

## Development Installation

For contributors and developers:

```bash
# Clone repository
git clone https://github.com/nicolasseverino/xplia.git
cd xplia

# Create development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev,full]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=xplia --cov-report=html
```

---

### IDE Setup

#### VS Code

Install recommended extensions:
- Python
- Pylance
- Python Test Explorer

`.vscode/settings.json`:
```json
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"]
}
```

---

#### PyCharm

1. Open project
2. Configure interpreter: Settings → Project → Python Interpreter
3. Enable pytest: Settings → Tools → Python Integrated Tools → Default test runner → pytest

---

## Docker Installation

### Pull from Docker Hub

```bash
docker pull nicolasseverino/xplia:latest

# Run container
docker run -it nicolasseverino/xplia:latest python
```

---

### Build from Dockerfile

```bash
cd xplia
docker build -t xplia:latest .

# Run with volume mount
docker run -v $(pwd)/data:/data -it xplia:latest
```

---

## Next Steps

After installation:

1. **Quick Start:** Read [docs/QUICKSTART.md](QUICKSTART.md)
2. **Tutorials:** Explore [examples/](../examples/)
3. **API Reference:** See [docs/API_REFERENCE.md](API_REFERENCE.md)
4. **Architecture:** Understand [docs/ARCHITECTURE.md](ARCHITECTURE.md)

---

## Uninstallation

To completely remove XPLIA:

```bash
# Uninstall package
pip uninstall xplia

# Remove configuration and cache
rm -rf ~/.xplia

# Remove virtual environment (if used)
rm -rf xplia-env
```

---

## License

XPLIA is licensed under the MIT License. See [LICENSE](../LICENSE) for details.
