# XPLIA Plugin Development Guide

**Version:** 1.0.0
**Last Updated:** November 2025

---

## Table of Contents

1. [Introduction](#introduction)
2. [Plugin Architecture](#plugin-architecture)
3. [Creating a Custom Explainer](#creating-a-custom-explainer)
4. [Creating a Custom Model Adapter](#creating-a-custom-model-adapter)
5. [Creating a Custom Visualizer](#creating-a-custom-visualizer)
6. [Creating a Custom Compliance Module](#creating-a-custom-compliance-module)
7. [Plugin Registration](#plugin-registration)
8. [Testing Your Plugin](#testing-your-plugin)
9. [Publishing Your Plugin](#publishing-your-plugin)
10. [Best Practices](#best-practices)

---

## Introduction

XPLIA is designed for extensibility. You can create custom plugins for:

- **Explainers**: New explanation methods
- **Model Adapters**: Support for new ML frameworks
- **Visualizers**: Custom visualization types
- **Compliance Modules**: Industry-specific regulations
- **Trust Evaluators**: Custom trust metrics

---

## Plugin Architecture

### Plugin Types

```
┌─────────────────────────────────────────┐
│           XPLIA Core                    │
├─────────────────────────────────────────┤
│  Plugin System                          │
│  ┌───────────────────────────────────┐  │
│  │  Explainer Plugins                │  │
│  │  Model Adapter Plugins            │  │
│  │  Visualizer Plugins               │  │
│  │  Compliance Plugins               │  │
│  │  Trust Evaluator Plugins          │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

### Registration System

Plugins use decorators for automatic registration:

```python
@register_explainer(
    name='my_plugin',
    version='1.0.0',
    description='My custom explainer',
    capabilities={'model_types': ['classification']},
    dependencies={'numpy': '>=1.20.0'}
)
class MyExplainer(ExplainerBase):
    pass
```

---

## Creating a Custom Explainer

### Step 1: Create Plugin File

Create `my_explainer_plugin.py`:

```python
"""
My Custom Explainer Plugin for XPLIA

Author: Your Name
Version: 1.0.0
License: MIT
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from xplia.core.base import ExplainerBase, ExplanationResult
from xplia.core.registry import register_explainer


@register_explainer(
    name='my_custom_explainer',
    version='1.0.0',
    description='A custom explanation method using [your technique]',
    capabilities={
        'model_types': ['classification', 'regression'],
        'data_types': ['tabular', 'image'],
        'global_explanations': True,
        'local_explanations': True,
        'feature_importance': True
    },
    dependencies={
        'numpy': '>=1.20.0',
        'pandas': '>=1.3.0',
        # Add your specific dependencies
    }
)
class MyCustomExplainer(ExplainerBase):
    """
    Custom explainer implementing [your method].

    This explainer provides [description of what it does].

    Parameters
    ----------
    model : object
        The model to explain. Must have predict() method.

    background_data : array-like, optional
        Background dataset for explanation baseline.

    n_samples : int, default=100
        Number of samples to use for explanation.

    feature_names : list of str, optional
        Names of features.

    **kwargs : dict
        Additional parameters.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> model = RandomForestClassifier()
    >>> explainer = MyCustomExplainer(model)
    >>> explanation = explainer.explain(X_test)
    """

    def __init__(
        self,
        model: Any,
        background_data: Optional[np.ndarray] = None,
        n_samples: int = 100,
        feature_names: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(model, **kwargs)

        self.model = model
        self.background_data = background_data
        self.n_samples = n_samples
        self.feature_names = feature_names

        # Initialize your explainer
        self._initialize()

    def _initialize(self):
        """Initialize the explainer."""
        # Your initialization logic
        self.add_audit_record('initialization', {
            'n_samples': self.n_samples,
            'has_background_data': self.background_data is not None
        })

    def explain(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        **kwargs
    ) -> ExplanationResult:
        """
        Generate explanations for predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Instances to explain.

        **kwargs : dict
            Additional parameters.

        Returns
        -------
        ExplanationResult
            Structured explanation result.
        """
        # Convert input to numpy array
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            feature_names = X.columns.tolist()
        else:
            X_array = np.array(X)
            feature_names = self.feature_names or [
                f'feature_{i}' for i in range(X_array.shape[1])
            ]

        # Get model predictions
        predictions = self.model.predict(X_array)

        # Generate explanations (your implementation)
        feature_importance = self._compute_feature_importance(X_array)

        # Record audit trail
        self.add_audit_record('explain', {
            'n_instances': X_array.shape[0],
            'n_features': X_array.shape[1]
        })

        # Return structured result
        return ExplanationResult(
            method='my_custom_explainer',
            explanation_data={
                'feature_importance': feature_importance,
                'feature_names': feature_names,
                'predictions': predictions
            },
            metadata={
                'model_type': type(self.model).__name__,
                'n_samples_used': self.n_samples,
                'timestamp': self._get_timestamp()
            },
            quality_metrics=self._compute_quality_metrics(
                X_array, feature_importance
            )
        )

    def explain_model(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[np.ndarray] = None,
        **kwargs
    ) -> ExplanationResult:
        """
        Generate global model explanations.

        Parameters
        ----------
        X : array-like
            Dataset to analyze.

        y : array-like, optional
            True labels.

        **kwargs : dict
            Additional parameters.

        Returns
        -------
        ExplanationResult
            Global explanation result.
        """
        # Your global explanation logic
        global_importance = self._compute_global_importance(X, y)

        return ExplanationResult(
            method='my_custom_explainer',
            explanation_data={
                'global_feature_importance': global_importance,
                'interaction_effects': self._compute_interactions(X)
            },
            metadata={
                'explanation_type': 'global',
                'n_samples': X.shape[0]
            }
        )

    def _compute_feature_importance(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Compute feature importance scores.

        This is where your custom explanation logic goes.
        """
        # EXAMPLE: Simple gradient-based importance
        # Replace with your actual method

        n_samples, n_features = X.shape
        importance_scores = np.zeros(n_features)

        for i in range(n_features):
            # Your importance computation
            # This is just a placeholder
            X_perturbed = X.copy()
            X_perturbed[:, i] += 0.1

            pred_original = self.model.predict(X)
            pred_perturbed = self.model.predict(X_perturbed)

            importance_scores[i] = np.abs(
                pred_perturbed - pred_original
            ).mean()

        return importance_scores

    def _compute_global_importance(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[np.ndarray]
    ) -> np.ndarray:
        """Compute global feature importance."""
        # Your global importance logic
        if isinstance(X, pd.DataFrame):
            X = X.values

        return self._compute_feature_importance(X)

    def _compute_interactions(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> Dict[str, float]:
        """Compute feature interactions."""
        # Your interaction computation
        return {}

    def _compute_quality_metrics(
        self,
        X: np.ndarray,
        importance: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute quality metrics for the explanation.

        Returns
        -------
        dict
            Quality metrics (fidelity, stability, etc.)
        """
        return {
            'fidelity': self._compute_fidelity(X, importance),
            'stability': self._compute_stability(X),
            'sparsity': np.sum(importance > 0.01) / len(importance)
        }

    def _compute_fidelity(
        self,
        X: np.ndarray,
        importance: np.ndarray
    ) -> float:
        """Measure how well explanation matches model."""
        # Your fidelity computation
        return 0.85  # Placeholder

    def _compute_stability(self, X: np.ndarray) -> float:
        """Measure explanation stability."""
        # Your stability computation
        return 0.90  # Placeholder

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()


# Optional: Add convenience function
def create_my_explainer(model, **kwargs):
    """
    Convenience function to create MyCustomExplainer.

    Parameters
    ----------
    model : object
        Model to explain.

    **kwargs : dict
        Additional parameters passed to MyCustomExplainer.

    Returns
    -------
    MyCustomExplainer
        Configured explainer instance.
    """
    return MyCustomExplainer(model, **kwargs)
```

---

### Step 2: Test Your Explainer

Create `test_my_explainer.py`:

```python
import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from my_explainer_plugin import MyCustomExplainer


class TestMyCustomExplainer:
    @pytest.fixture
    def simple_model(self):
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model, X[:5]

    def test_explainer_creation(self, simple_model):
        model, X = simple_model
        explainer = MyCustomExplainer(model)
        assert explainer is not None

    def test_explain_method(self, simple_model):
        model, X = simple_model
        explainer = MyCustomExplainer(model)
        result = explainer.explain(X)

        assert result is not None
        assert 'feature_importance' in result.explanation_data
        assert len(result.explanation_data['feature_importance']) == X.shape[1]

    def test_explain_model_method(self, simple_model):
        model, X = simple_model
        explainer = MyCustomExplainer(model)
        result = explainer.explain_model(X)

        assert result is not None
        assert 'global_feature_importance' in result.explanation_data
```

Run tests:
```bash
pytest test_my_explainer.py -v
```

---

### Step 3: Package Your Plugin

Create `setup.py`:

```python
from setuptools import setup, find_packages

setup(
    name='xplia-my-explainer',
    version='1.0.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='My custom explainer for XPLIA',
    py_modules=['my_explainer_plugin'],
    install_requires=[
        'xplia>=1.0.0',
        'numpy>=1.20.0',
        'pandas>=1.3.0',
    ],
    entry_points={
        'xplia.plugins': [
            'my_explainer = my_explainer_plugin:MyCustomExplainer',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)
```

---

## Creating a Custom Model Adapter

Example for a hypothetical "MyFramework":

```python
from xplia.core.model_adapters.base import ModelAdapterBase
import numpy as np


class MyFrameworkAdapter(ModelAdapterBase):
    """
    Adapter for MyFramework models.

    Parameters
    ----------
    model : MyFramework.Model
        The model to wrap.

    feature_names : list of str, optional
        Names of input features.

    **kwargs : dict
        Additional parameters.
    """

    def __init__(self, model, feature_names=None, **kwargs):
        super().__init__(model, **kwargs)
        self.model = model
        self._feature_names = feature_names
        self._metadata = self._extract_metadata()

    def predict(self, X):
        """
        Get predictions from the model.

        Parameters
        ----------
        X : array-like
            Input data.

        Returns
        -------
        np.ndarray
            Predictions.
        """
        # Convert to framework format
        X_framework = self._convert_to_framework_format(X)

        # Get predictions
        predictions = self.model.forward(X_framework)

        # Convert back to numpy
        predictions_np = self._convert_from_framework_format(predictions)

        return predictions_np

    def predict_proba(self, X):
        """
        Get probability predictions.

        Parameters
        ----------
        X : array-like
            Input data.

        Returns
        -------
        np.ndarray
            Probability predictions.
        """
        # Your implementation
        X_framework = self._convert_to_framework_format(X)
        probabilities = self.model.predict_proba(X_framework)
        return self._convert_from_framework_format(probabilities)

    def get_feature_names(self):
        """Get feature names."""
        if self._feature_names is not None:
            return self._feature_names

        # Try to extract from model
        if hasattr(self.model, 'feature_names'):
            return self.model.feature_names

        # Default
        n_features = self._get_n_features()
        return [f'feature_{i}' for i in range(n_features)]

    def get_metadata(self):
        """Get model metadata."""
        return self._metadata

    def _extract_metadata(self):
        """Extract metadata from model."""
        return {
            'framework': 'MyFramework',
            'model_type': type(self.model).__name__,
            'n_features': self._get_n_features(),
        }

    def _convert_to_framework_format(self, X):
        """Convert numpy array to framework format."""
        # Your conversion logic
        return X

    def _convert_from_framework_format(self, X_framework):
        """Convert from framework format to numpy."""
        # Your conversion logic
        return np.array(X_framework)

    def _get_n_features(self):
        """Get number of features."""
        # Extract from model
        return self.model.n_features if hasattr(self.model, 'n_features') else None
```

---

## Creating a Custom Visualizer

```python
from xplia.core.registry import register_visualizer
import matplotlib.pyplot as plt


@register_visualizer(
    name='my_custom_chart',
    version='1.0.0',
    description='Custom visualization',
    capabilities={
        'output_formats': ['png', 'html', 'svg'],
        'interactive': True
    }
)
class MyCustomVisualizer:
    """Custom visualizer for explanations."""

    def __init__(self, theme='light', **kwargs):
        self.theme = theme
        self.kwargs = kwargs

    def render(self, explanation_result, output_path=None, **kwargs):
        """
        Render visualization.

        Parameters
        ----------
        explanation_result : ExplanationResult
            The explanation to visualize.

        output_path : str, optional
            Where to save the visualization.

        **kwargs : dict
            Additional rendering options.

        Returns
        -------
        figure : object
            The generated figure.
        """
        # Extract data
        feature_importance = explanation_result.explanation_data['feature_importance']
        feature_names = explanation_result.explanation_data['feature_names']

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot
        ax.barh(feature_names, feature_importance)
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance - My Custom Viz')

        # Apply theme
        if self.theme == 'dark':
            fig.patch.set_facecolor('#2b2b2b')
            ax.set_facecolor('#2b2b2b')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')

        # Save if path provided
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')

        return fig
```

---

## Best Practices

### 1. Follow Naming Conventions

```python
# Good
class GradientBasedExplainer(ExplainerBase):
    pass

# Bad
class explainer(ExplainerBase):  # Not capitalized
    pass
```

---

### 2. Provide Comprehensive Docstrings

```python
def explain(self, X, **kwargs):
    """
    Generate explanations.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input instances.

    **kwargs : dict
        method : str, optional
            Explanation method variant.
        n_samples : int, optional
            Number of samples for approximation.

    Returns
    -------
    ExplanationResult
        Structured explanation with metadata.

    Raises
    ------
    ValueError
        If X has wrong shape or contains NaN.

    Examples
    --------
    >>> explainer = MyExplainer(model)
    >>> result = explainer.explain(X_test[:5])
    >>> print(result.explanation_data['feature_importance'])
    """
```

---

### 3. Handle Errors Gracefully

```python
def explain(self, X, **kwargs):
    # Validate input
    if X.shape[1] != self.n_features:
        raise ValueError(
            f"Expected {self.n_features} features, "
            f"got {X.shape[1]}"
        )

    # Check for NaN
    if np.isnan(X).any():
        raise ValueError("Input contains NaN values")

    try:
        # Your logic
        result = self._compute_explanation(X)
    except Exception as e:
        self.add_audit_record('error', {'exception': str(e)})
        raise RuntimeError(
            f"Explanation failed: {e}\n"
            "Check audit trail for details."
        ) from e

    return result
```

---

### 4. Add Comprehensive Tests

- Unit tests for each method
- Integration tests with real models
- Edge case tests (empty input, single sample, etc.)
- Performance tests for large datasets

---

### 5. Use Semantic Versioning

- **Major** (1.0.0 → 2.0.0): Breaking changes
- **Minor** (1.0.0 → 1.1.0): New features, backward compatible
- **Patch** (1.0.0 → 1.0.1): Bug fixes

---

## Publishing Your Plugin

### 1. To PyPI

```bash
# Build distribution
python setup.py sdist bdist_wheel

# Upload to PyPI
twine upload dist/*
```

---

### 2. To GitHub

```bash
git tag v1.0.0
git push origin v1.0.0
```

---

### 3. Register with XPLIA

Submit your plugin to the official registry:
https://github.com/nicolasseverino/xplia-plugins

---

## Resources

- **Example Plugins:** https://github.com/nicolasseverino/xplia/tree/main/examples/plugins
- **Plugin Template:** https://github.com/nicolasseverino/xplia-plugin-template
- **Community Plugins:** https://github.com/nicolasseverino/xplia-plugins

---

## Support

- **Questions:** [GitHub Discussions](https://github.com/nicolasseverino/xplia/discussions)
- **Issues:** [GitHub Issues](https://github.com/nicolasseverino/xplia/issues)
- **Discord:** [Join our server]
