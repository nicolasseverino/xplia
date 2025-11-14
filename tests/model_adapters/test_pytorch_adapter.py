"""
Comprehensive tests for PyTorch model adapter.
"""

import pytest
import numpy as np
import pandas as pd

# Skip if PyTorch not installed
pytest.importorskip("torch")

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

try:
    from xplia.core.model_adapters.pytorch_adapter import PyTorchAdapter
except ImportError:
    PyTorchAdapter = None
    pytest.skip("PyTorch adapter not found", allow_module_level=True)


class SimpleClassifier(nn.Module):
    """Simple neural network for classification."""

    def __init__(self, input_dim, hidden_dim=64, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleRegressor(nn.Module):
    """Simple neural network for regression."""

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TestPyTorchAdapterClassification:
    """Test PyTorch adapter with classification models."""

    @pytest.fixture
    def classification_model_and_data(self):
        """Create and train a simple classification model."""
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = SimpleClassifier(input_dim=10, output_dim=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Quick training
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)

        model.train()
        for _ in range(50):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        return model, X_test, [f'feature_{i}' for i in range(10)]

    def test_pytorch_prediction(self, classification_model_and_data):
        """Test PyTorch model prediction."""
        model, X_test, feature_names = classification_model_and_data

        adapter = PyTorchAdapter(model, feature_names=feature_names)
        predictions = adapter.predict(X_test)

        assert predictions.shape[0] == X_test.shape[0]
        assert len(np.unique(predictions)) <= 2

    def test_pytorch_predict_proba(self, classification_model_and_data):
        """Test PyTorch probability prediction."""
        model, X_test, feature_names = classification_model_and_data

        adapter = PyTorchAdapter(model, feature_names=feature_names, task='classification')
        probabilities = adapter.predict_proba(X_test)

        assert probabilities.shape[0] == X_test.shape[0]
        assert probabilities.shape[1] == 2
        assert np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.01)

    def test_pytorch_pandas_input(self, classification_model_and_data):
        """Test PyTorch adapter with pandas DataFrame."""
        model, X_test, feature_names = classification_model_and_data

        adapter = PyTorchAdapter(model, feature_names=feature_names)
        df = pd.DataFrame(X_test, columns=feature_names)
        predictions = adapter.predict(df)

        assert predictions.shape[0] == X_test.shape[0]


class TestPyTorchAdapterRegression:
    """Test PyTorch adapter with regression models."""

    @pytest.fixture
    def regression_model_and_data(self):
        """Create and train a simple regression model."""
        X, y = make_regression(n_samples=500, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = SimpleRegressor(input_dim=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Quick training
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)

        model.train()
        for _ in range(50):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        return model, X_test, [f'feature_{i}' for i in range(10)]

    def test_pytorch_regression_prediction(self, regression_model_and_data):
        """Test PyTorch regression prediction."""
        model, X_test, feature_names = regression_model_and_data

        adapter = PyTorchAdapter(model, feature_names=feature_names, task='regression')
        predictions = adapter.predict(X_test)

        assert predictions.shape[0] == X_test.shape[0]
        assert isinstance(predictions[0], (int, float, np.number))


class TestPyTorchAdapterMetadata:
    """Test metadata extraction from PyTorch models."""

    def test_metadata_extraction(self, classification_model_and_data):
        """Test metadata extraction."""
        model, X_test, feature_names = classification_model_and_data

        adapter = PyTorchAdapter(model, feature_names=feature_names)
        metadata = adapter.get_metadata()

        assert isinstance(metadata, dict)

    def test_feature_names_extraction(self, classification_model_and_data):
        """Test feature names extraction."""
        model, X_test, feature_names = classification_model_and_data

        adapter = PyTorchAdapter(model, feature_names=feature_names)
        names = adapter.get_feature_names()

        assert len(names) == 10
        assert all(isinstance(name, str) for name in names)


class TestPyTorchAdapterErrorHandling:
    """Test error handling."""

    def test_wrong_number_of_features(self, classification_model_and_data):
        """Test error with wrong number of features."""
        model, _, feature_names = classification_model_and_data

        adapter = PyTorchAdapter(model, feature_names=feature_names)
        wrong_features = np.random.randn(10, 5)  # Should be 10 features

        with pytest.raises(Exception):
            adapter.predict(wrong_features)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
