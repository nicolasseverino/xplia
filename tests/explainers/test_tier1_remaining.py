"""
Tests for TIER 1 - Remaining Modules (Counterfactuals & Generative)
Tests for Advanced Counterfactuals and Generative Models
"""

import pytest
import numpy as np
from typing import Dict, Any

# Counterfactuals
from xplia.explainers.counterfactuals.advanced_counterfactuals import (
    MinimalCounterfactualGenerator,
    FeasibleCounterfactualGenerator,
    DiverseCounterfactualGenerator,
    ActionableRecourseGenerator,
    CounterfactualExplanation,
)

# Generative
from xplia.explainers.generative.generative_explainer import (
    VAEExplainer,
    GANExplainer,
    StyleGANExplainer,
    GenerativeExplanation,
)


# ===================
# COUNTERFACTUALS
# ===================

class TestMinimalCounterfactualGenerator:
    """Test Minimal Counterfactual Generator."""

    def test_initialization(self):
        """Test initialization."""
        generator = MinimalCounterfactualGenerator()
        assert generator is not None

    def test_generate_minimal(self):
        """Test minimal counterfactual generation."""
        generator = MinimalCounterfactualGenerator()

        instance = np.array([1.0, 2.0, 3.0, 4.0])
        target_class = 1

        cf = generator.generate(instance, target_class)

        assert isinstance(cf, CounterfactualExplanation)
        assert hasattr(cf, 'counterfactual_instance')
        assert hasattr(cf, 'distance')
        assert hasattr(cf, 'changed_features')


class TestFeasibleCounterfactualGenerator:
    """Test Feasible Counterfactual Generator."""

    def test_initialization(self):
        """Test initialization."""
        generator = FeasibleCounterfactualGenerator()
        assert generator is not None

    def test_generate_feasible(self):
        """Test feasible counterfactual with constraints."""
        generator = FeasibleCounterfactualGenerator()

        instance = np.array([1.0, 2.0, 3.0])
        target_class = 1
        constraints = {'feature_0': (0.0, 5.0)}

        cf = generator.generate(instance, target_class, constraints=constraints)

        assert cf is not None
        assert hasattr(cf, 'feasibility_score')


class TestDiverseCounterfactualGenerator:
    """Test Diverse Counterfactual Generator."""

    def test_initialization(self):
        """Test initialization."""
        generator = DiverseCounterfactualGenerator(num_counterfactuals=5)
        assert generator.num_counterfactuals == 5

    def test_generate_diverse(self):
        """Test diverse counterfactual generation."""
        generator = DiverseCounterfactualGenerator(num_counterfactuals=3)

        instance = np.array([1.0, 2.0, 3.0])
        target_class = 1

        cfs = generator.generate(instance, target_class)

        assert len(cfs) == 3
        assert all(isinstance(cf, CounterfactualExplanation) for cf in cfs)


class TestActionableRecourseGenerator:
    """Test Actionable Recourse Generator."""

    def test_initialization(self):
        """Test initialization."""
        generator = ActionableRecourseGenerator()
        assert generator is not None

    def test_generate_actionable(self):
        """Test actionable recourse generation."""
        generator = ActionableRecourseGenerator()

        instance = np.array([1.0, 2.0, 3.0, 4.0])
        target_class = 1

        # Define actionable features and costs
        actionable_features = [0, 1, 2]  # Only first 3 features actionable
        costs = np.array([1.0, 2.0, 3.0, 10.0])  # Feature 3 is expensive

        recourse = generator.generate(
            instance,
            target_class,
            actionable_features=actionable_features,
            feature_costs=costs
        )

        assert recourse is not None
        assert hasattr(recourse, 'recommendations')
        assert hasattr(recourse, 'total_cost')

    def test_rank_by_cost(self):
        """Test ranking recommendations by cost."""
        generator = ActionableRecourseGenerator()

        instance = np.array([1.0, 2.0, 3.0])
        target_class = 1
        costs = np.array([1.0, 5.0, 10.0])

        recourse = generator.generate(instance, target_class, feature_costs=costs)

        # Should prefer lower-cost changes
        assert recourse.total_cost >= 0


# ===================
# GENERATIVE MODELS
# ===================

class TestVAEExplainer:
    """Test VAE Explainer."""

    def test_initialization(self):
        """Test initialization."""
        explainer = VAEExplainer(latent_dim=10)
        assert explainer.latent_dim == 10

    def test_explain_latent_space(self):
        """Test latent space explanation."""
        explainer = VAEExplainer(latent_dim=8)

        # Dummy latent vector
        latent = np.random.randn(8)

        explanation = explainer.explain_latent_space(latent)

        assert isinstance(explanation, GenerativeExplanation)
        assert hasattr(explanation, 'latent_dimensions_importance')
        assert hasattr(explanation, 'disentanglement_score')

    def test_explain_reconstruction(self):
        """Test reconstruction explanation."""
        explainer = VAEExplainer(latent_dim=8)

        original = np.random.rand(28, 28)  # Image
        reconstructed = np.random.rand(28, 28)

        explanation = explainer.explain_reconstruction(original, reconstructed)

        assert explanation is not None
        assert hasattr(explanation, 'reconstruction_error')

    def test_traverse_latent_dimension(self):
        """Test latent dimension traversal."""
        explainer = VAEExplainer(latent_dim=8)

        base_latent = np.random.randn(8)
        dim_to_traverse = 0

        traversal = explainer.traverse_dimension(
            base_latent,
            dimension=dim_to_traverse,
            range_vals=(-3, 3),
            num_steps=7
        )

        assert traversal is not None
        assert len(traversal) == 7


class TestGANExplainer:
    """Test GAN Explainer."""

    def test_initialization(self):
        """Test initialization."""
        explainer = GANExplainer(latent_dim=100)
        assert explainer.latent_dim == 100

    def test_explain_generation(self):
        """Test generation explanation."""
        explainer = GANExplainer(latent_dim=50)

        latent = np.random.randn(50)
        generated = np.random.rand(64, 64, 3)

        explanation = explainer.explain_generation(latent, generated)

        assert isinstance(explanation, GenerativeExplanation)
        assert hasattr(explanation, 'latent_importance')

    def test_identify_important_dimensions(self):
        """Test important dimension identification."""
        explainer = GANExplainer(latent_dim=50)

        latent = np.random.randn(50)

        important_dims = explainer.identify_important_dimensions(latent, top_k=10)

        assert len(important_dims) == 10


class TestStyleGANExplainer:
    """Test StyleGAN Explainer."""

    def test_initialization(self):
        """Test initialization."""
        explainer = StyleGANExplainer(w_dim=512)
        assert explainer.w_dim == 512

    def test_explain_w_space(self):
        """Test W-space explanation."""
        explainer = StyleGANExplainer(w_dim=512)

        w_vector = np.random.randn(512)
        generated = np.random.rand(1024, 1024, 3)

        explanation = explainer.explain_w_space(w_vector, generated)

        assert explanation is not None
        assert hasattr(explanation, 'style_factors')

    def test_identify_disentangled_directions(self):
        """Test disentangled direction identification."""
        explainer = StyleGANExplainer(w_dim=512)

        w_vectors = [np.random.randn(512) for _ in range(10)]

        directions = explainer.identify_disentangled_directions(w_vectors)

        assert directions is not None
        assert len(directions) > 0

    def test_explain_style_mixing(self):
        """Test style mixing explanation."""
        explainer = StyleGANExplainer(w_dim=512)

        w1 = np.random.randn(512)
        w2 = np.random.randn(512)
        mixing_layer = 8

        explanation = explainer.explain_style_mixing(w1, w2, mixing_layer)

        assert explanation is not None
        assert hasattr(explanation, 'coarse_styles')
        assert hasattr(explanation, 'fine_styles')


# Integration Tests
class TestTier1Integration:
    """Integration tests for TIER 1 remaining modules."""

    def test_counterfactual_with_generative(self):
        """Test counterfactual generation with VAE."""
        cf_generator = MinimalCounterfactualGenerator()
        vae_explainer = VAEExplainer(latent_dim=8)

        # Original instance
        instance = np.random.rand(8)
        target_class = 1

        # Generate counterfactual
        cf = cf_generator.generate(instance, target_class)

        # Explain with VAE
        vae_exp = vae_explainer.explain_latent_space(cf.counterfactual_instance)

        assert cf is not None
        assert vae_exp is not None

    def test_diverse_cf_with_actionability(self):
        """Test diverse counterfactuals with actionability constraints."""
        diverse_gen = DiverseCounterfactualGenerator(num_counterfactuals=3)
        actionable_gen = ActionableRecourseGenerator()

        instance = np.array([1.0, 2.0, 3.0, 4.0])
        target_class = 1

        # Generate diverse
        diverse_cfs = diverse_gen.generate(instance, target_class)

        # Check actionability of each
        actionable_features = [0, 1, 2]
        for cf in diverse_cfs:
            # Verify CF only changes actionable features
            changes = cf.changed_features
            # All changes should be in actionable features
            assert all(idx in actionable_features for idx in changes)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
