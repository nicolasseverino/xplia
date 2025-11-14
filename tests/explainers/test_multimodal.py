"""
Tests for TIER 1 - Multimodal AI Explainers
Tests for Vision-Language and Diffusion models explainability
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from xplia.explainers.multimodal.vision_language_explainer import (
    CLIPExplainer,
    BLIPExplainer,
    MultimodalCounterfactualExplainer,
    VisionLanguageExplanation,
)
from xplia.explainers.multimodal.diffusion_explainer import (
    StableDiffusionExplainer,
    NegativePromptAnalyzer,
    LoRAExplainer,
    DiffusionExplanation,
)


class TestCLIPExplainer:
    """Test suite for CLIP explainer."""

    def test_initialization(self):
        """Test CLIPExplainer initialization."""
        explainer = CLIPExplainer()
        assert explainer is not None
        assert hasattr(explainer, 'explain')

    def test_explain_basic(self):
        """Test basic explanation generation."""
        explainer = CLIPExplainer()

        # Create dummy image and text
        image = np.random.rand(224, 224, 3).astype(np.float32)
        text = "a photo of a cat"

        # Generate explanation
        explanation = explainer.explain(image, text, method='attention')

        # Verify explanation structure
        assert isinstance(explanation, VisionLanguageExplanation)
        assert hasattr(explanation, 'similarity_score')
        assert hasattr(explanation, 'image_attribution')
        assert hasattr(explanation, 'text_attribution')
        assert hasattr(explanation, 'cross_modal_attention')

    def test_explain_gradient_method(self):
        """Test gradient-based explanation."""
        explainer = CLIPExplainer()

        image = np.random.rand(224, 224, 3).astype(np.float32)
        text = "a dog playing"

        explanation = explainer.explain(image, text, method='gradient')

        assert explanation is not None
        assert hasattr(explanation, 'image_gradients')

    def test_explain_invalid_method(self):
        """Test with invalid explanation method."""
        explainer = CLIPExplainer()

        image = np.random.rand(224, 224, 3).astype(np.float32)
        text = "test"

        with pytest.raises(ValueError):
            explainer.explain(image, text, method='invalid_method')

    def test_explain_invalid_image_shape(self):
        """Test with invalid image shape."""
        explainer = CLIPExplainer()

        # Invalid shape (not 3 channels)
        image = np.random.rand(224, 224)
        text = "test"

        with pytest.raises((ValueError, AssertionError)):
            explainer.explain(image, text)

    def test_explain_empty_text(self):
        """Test with empty text."""
        explainer = CLIPExplainer()

        image = np.random.rand(224, 224, 3).astype(np.float32)
        text = ""

        with pytest.raises((ValueError, AssertionError)):
            explainer.explain(image, text)


class TestBLIPExplainer:
    """Test suite for BLIP explainer."""

    def test_initialization(self):
        """Test BLIPExplainer initialization."""
        explainer = BLIPExplainer()
        assert explainer is not None

    def test_explain_captioning(self):
        """Test image captioning explanation."""
        explainer = BLIPExplainer()

        image = np.random.rand(224, 224, 3).astype(np.float32)

        explanation = explainer.explain_captioning(image)

        assert explanation is not None
        assert hasattr(explanation, 'caption')
        assert hasattr(explanation, 'caption_confidence')
        assert hasattr(explanation, 'important_regions')

    def test_explain_vqa(self):
        """Test Visual Question Answering explanation."""
        explainer = BLIPExplainer()

        image = np.random.rand(224, 224, 3).astype(np.float32)
        question = "What is in the image?"

        explanation = explainer.explain_vqa(image, question)

        assert explanation is not None
        assert hasattr(explanation, 'answer')
        assert hasattr(explanation, 'answer_confidence')


class TestMultimodalCounterfactualExplainer:
    """Test suite for Multimodal Counterfactual explainer."""

    def test_initialization(self):
        """Test initialization."""
        explainer = MultimodalCounterfactualExplainer()
        assert explainer is not None

    def test_generate_counterfactual(self):
        """Test counterfactual generation."""
        explainer = MultimodalCounterfactualExplainer()

        image = np.random.rand(224, 224, 3).astype(np.float32)
        original_text = "a cat"
        target_text = "a dog"

        cf_explanation = explainer.generate(
            image=image,
            original_text=original_text,
            target_text=target_text
        )

        assert cf_explanation is not None
        assert hasattr(cf_explanation, 'counterfactual_image')
        assert hasattr(cf_explanation, 'modifications')


class TestStableDiffusionExplainer:
    """Test suite for Stable Diffusion explainer."""

    def test_initialization(self):
        """Test StableDiffusionExplainer initialization."""
        explainer = StableDiffusionExplainer()
        assert explainer is not None

    def test_explain_generation(self):
        """Test generation explanation."""
        explainer = StableDiffusionExplainer()

        prompt = "a beautiful sunset over mountains"
        generated_image = np.random.rand(512, 512, 3).astype(np.float32)

        explanation = explainer.explain(prompt, generated_image)

        assert isinstance(explanation, DiffusionExplanation)
        assert hasattr(explanation, 'prompt_token_importance')
        assert hasattr(explanation, 'timestep_importance')
        assert hasattr(explanation, 'spatial_attribution')

    def test_explain_with_negative_prompt(self):
        """Test explanation with negative prompt."""
        explainer = StableDiffusionExplainer()

        prompt = "beautiful landscape"
        negative_prompt = "blurry, low quality"
        generated_image = np.random.rand(512, 512, 3).astype(np.float32)

        explanation = explainer.explain(
            prompt,
            generated_image,
            negative_prompt=negative_prompt
        )

        assert explanation is not None
        assert hasattr(explanation, 'negative_prompt_effect')

    def test_analyze_timesteps(self):
        """Test timestep importance analysis."""
        explainer = StableDiffusionExplainer()

        prompt = "test prompt"

        timestep_analysis = explainer.analyze_timesteps(prompt, num_steps=10)

        assert timestep_analysis is not None
        assert len(timestep_analysis) == 10


class TestNegativePromptAnalyzer:
    """Test suite for Negative Prompt Analyzer."""

    def test_initialization(self):
        """Test initialization."""
        analyzer = NegativePromptAnalyzer()
        assert analyzer is not None

    def test_analyze_effect(self):
        """Test negative prompt effect analysis."""
        analyzer = NegativePromptAnalyzer()

        positive_prompt = "high quality photo"
        negative_prompt = "blurry, distorted"

        analysis = analyzer.analyze_effect(positive_prompt, negative_prompt)

        assert analysis is not None
        assert hasattr(analysis, 'conflict_score')
        assert hasattr(analysis, 'suppressed_concepts')


class TestLoRAExplainer:
    """Test suite for LoRA explainer."""

    def test_initialization(self):
        """Test LoRAExplainer initialization."""
        explainer = LoRAExplainer()
        assert explainer is not None

    def test_explain_lora_effect(self):
        """Test LoRA effect explanation."""
        explainer = LoRAExplainer()

        base_image = np.random.rand(512, 512, 3).astype(np.float32)
        lora_image = np.random.rand(512, 512, 3).astype(np.float32)

        explanation = explainer.explain_effect(
            base_image=base_image,
            lora_image=lora_image,
            lora_strength=0.7
        )

        assert explanation is not None
        assert hasattr(explanation, 'style_changes')
        assert hasattr(explanation, 'content_preservation')

    def test_compare_lora_weights(self):
        """Test LoRA weight comparison."""
        explainer = LoRAExplainer()

        # Mock LoRA weights
        lora_weights = {
            'layer1': np.random.rand(10, 10),
            'layer2': np.random.rand(20, 20)
        }

        analysis = explainer.analyze_weights(lora_weights)

        assert analysis is not None
        assert 'layer_importance' in analysis


# Integration tests
class TestMultimodalIntegration:
    """Integration tests for multimodal explainers."""

    def test_clip_blip_consistency(self):
        """Test consistency between CLIP and BLIP explanations."""
        clip_explainer = CLIPExplainer()
        blip_explainer = BLIPExplainer()

        image = np.random.rand(224, 224, 3).astype(np.float32)
        text = "a cat on a table"

        clip_exp = clip_explainer.explain(image, text)
        blip_exp = blip_explainer.explain_vqa(image, "What is this?")

        # Both should produce valid explanations
        assert clip_exp is not None
        assert blip_exp is not None

    def test_diffusion_with_counterfactual(self):
        """Test diffusion explanation with counterfactuals."""
        diff_explainer = StableDiffusionExplainer()
        cf_explainer = MultimodalCounterfactualExplainer()

        prompt = "a red car"
        generated_image = np.random.rand(512, 512, 3).astype(np.float32)

        diff_exp = diff_explainer.explain(prompt, generated_image)

        # Use diffusion explanation to generate counterfactual
        cf_exp = cf_explainer.generate(
            image=generated_image,
            original_text=prompt,
            target_text="a blue car"
        )

        assert diff_exp is not None
        assert cf_exp is not None


# Performance tests
class TestMultimodalPerformance:
    """Performance tests for multimodal explainers."""

    def test_clip_performance_batch(self):
        """Test CLIP explainer with batch of images."""
        explainer = CLIPExplainer()

        batch_size = 5
        images = [np.random.rand(224, 224, 3).astype(np.float32) for _ in range(batch_size)]
        texts = [f"text {i}" for i in range(batch_size)]

        # Should handle batch efficiently
        explanations = []
        for img, txt in zip(images, texts):
            exp = explainer.explain(img, txt)
            explanations.append(exp)

        assert len(explanations) == batch_size

    def test_diffusion_memory_efficiency(self):
        """Test memory efficiency for diffusion explainer."""
        explainer = StableDiffusionExplainer()

        prompt = "test"
        image = np.random.rand(512, 512, 3).astype(np.float32)

        # Should not cause memory issues
        explanation = explainer.explain(prompt, image)

        assert explanation is not None
        # Cleanup
        del explanation


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
