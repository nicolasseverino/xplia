"""
Diffusion Model Explainability.

Explainability for diffusion models like Stable Diffusion, DALL-E 3, Imagen.

Author: XPLIA Team
License: MIT
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from xplia.core.base import ExplanationResult


@dataclass
class DiffusionExplanation:
    """
    Explanation for diffusion model generation.

    Attributes
    ----------
    prompt_attribution : ndarray
        Attribution for each token in prompt.
    timestep_importance : ndarray
        Importance of each diffusion timestep.
    spatial_attribution : ndarray
        Spatial attribution map for generated image.
    concept_attribution : dict
        Attribution for detected concepts.
    metadata : dict
        Additional metadata.
    """
    prompt_attribution: np.ndarray
    timestep_importance: np.ndarray
    spatial_attribution: np.ndarray
    concept_attribution: Dict[str, float]
    metadata: Dict[str, Any]


class StableDiffusionExplainer:
    """
    Explainer for Stable Diffusion models.

    Explains text-to-image generation through prompt attribution,
    timestep analysis, and spatial attribution.

    Parameters
    ----------
    model : object
        Stable Diffusion model (UNet + VAE + Text Encoder).
    n_inference_steps : int
        Number of diffusion steps.
    guidance_scale : float
        Classifier-free guidance scale.

    Examples
    --------
    >>> explainer = StableDiffusionExplainer(sd_model)
    >>> explanation = explainer.explain(
    ...     prompt="A beautiful sunset over mountains",
    ...     generated_image=image
    ... )
    """

    def __init__(
        self,
        model: Any,
        n_inference_steps: int = 50,
        guidance_scale: float = 7.5
    ):
        self.model = model
        self.n_steps = n_inference_steps
        self.guidance_scale = guidance_scale

    def _explain_prompt_tokens(
        self,
        prompt: str,
        generated_image: np.ndarray
    ) -> np.ndarray:
        """
        Explain contribution of each prompt token to generation.

        Parameters
        ----------
        prompt : str
            Input text prompt.
        generated_image : ndarray
            Generated image.

        Returns
        -------
        token_attribution : ndarray
            Attribution score for each token.
        """
        # In practice:
        # 1. Tokenize prompt
        # 2. For each token, ablate and regenerate
        # 3. Measure image difference
        # 4. Attribution = difference magnitude

        # For demo:
        tokens = prompt.split()
        token_attribution = np.random.rand(len(tokens))
        token_attribution = token_attribution / token_attribution.sum()

        return token_attribution

    def _analyze_timestep_importance(
        self,
        prompt: str,
        latents_history: Optional[List[np.ndarray]] = None
    ) -> np.ndarray:
        """
        Analyze importance of each diffusion timestep.

        Early steps: global structure, composition
        Late steps: fine details, textures

        Parameters
        ----------
        prompt : str
            Input prompt.
        latents_history : list of ndarray, optional
            Latent codes at each timestep.

        Returns
        -------
        timestep_importance : ndarray
            Importance score for each timestep.
        """
        # In practice: measure information gain at each step
        # importance[t] = KL(latent[t+1] || latent[t])

        # For demo: early steps more important for structure
        timestep_importance = np.zeros(self.n_steps)
        for t in range(self.n_steps):
            # Early steps (high t) more important
            timestep_importance[t] = (self.n_steps - t) / self.n_steps

        # Add noise to simulate real importance
        timestep_importance += np.random.rand(self.n_steps) * 0.1
        timestep_importance = timestep_importance / timestep_importance.sum()

        return timestep_importance

    def _compute_spatial_attribution(
        self,
        prompt: str,
        generated_image: np.ndarray,
        method: str = 'attention'
    ) -> np.ndarray:
        """
        Compute spatial attribution map.

        Shows which image regions were most influenced by prompt.

        Parameters
        ----------
        prompt : str
            Input prompt.
        generated_image : ndarray
            Generated image.
        method : str
            Attribution method ('attention', 'gradient').

        Returns
        -------
        spatial_attr : ndarray
            Spatial attribution map.
        """
        if method == 'attention':
            # In practice: aggregate cross-attention maps
            # cross_attn = model.unet.get_cross_attention_maps()
            # spatial_attr = cross_attn.mean(axis=0)  # Average over tokens

            # For demo:
            h, w = 64, 64  # Latent space size
            spatial_attr = np.random.rand(h, w)

        elif method == 'gradient':
            # Gradient-based attribution
            # grad = compute_gradient(latent, prompt)
            # spatial_attr = abs(grad).mean(axis=-1)

            h, w = 64, 64
            spatial_attr = np.random.rand(h, w)

        else:
            raise ValueError(f"Unknown method: {method}")

        return spatial_attr

    def _detect_concepts(
        self,
        prompt: str,
        generated_image: np.ndarray
    ) -> Dict[str, float]:
        """
        Detect and attribute visual concepts in generated image.

        Parameters
        ----------
        prompt : str
            Input prompt.
        generated_image : ndarray
            Generated image.

        Returns
        -------
        concept_attr : dict
            Attribution for each detected concept.
        """
        # In practice: use CLIP or concept detector
        # concepts = detect_concepts(generated_image)
        # for concept in concepts:
        #     attr[concept] = measure_prompt_concept_alignment(prompt, concept)

        # For demo: extract nouns from prompt
        tokens = prompt.lower().split()
        common_concepts = ['sunset', 'mountain', 'forest', 'ocean', 'city', 'sky']

        concept_attr = {}
        for concept in common_concepts:
            if any(concept in token for token in tokens):
                concept_attr[concept] = float(np.random.uniform(0.5, 1.0))
            else:
                concept_attr[concept] = float(np.random.uniform(0.0, 0.3))

        return concept_attr

    def explain(
        self,
        prompt: str,
        generated_image: np.ndarray,
        latents_history: Optional[List[np.ndarray]] = None
    ) -> DiffusionExplanation:
        """
        Explain diffusion model generation.

        Parameters
        ----------
        prompt : str
            Input text prompt.
        generated_image : ndarray
            Generated image.
        latents_history : list of ndarray, optional
            Latent codes at each timestep (if available).

        Returns
        -------
        explanation : DiffusionExplanation
            Complete diffusion explanation.
        """
        # Prompt token attribution
        prompt_attr = self._explain_prompt_tokens(prompt, generated_image)

        # Timestep importance
        timestep_importance = self._analyze_timestep_importance(prompt, latents_history)

        # Spatial attribution
        spatial_attr = self._compute_spatial_attribution(prompt, generated_image)

        # Concept attribution
        concept_attr = self._detect_concepts(prompt, generated_image)

        return DiffusionExplanation(
            prompt_attribution=prompt_attr,
            timestep_importance=timestep_importance,
            spatial_attribution=spatial_attr,
            concept_attribution=concept_attr,
            metadata={
                'prompt': prompt,
                'tokens': prompt.split(),
                'n_steps': self.n_steps,
                'guidance_scale': self.guidance_scale,
                'image_shape': generated_image.shape
            }
        )


class NegativePromptAnalyzer:
    """
    Analyze effect of negative prompts in diffusion models.

    Parameters
    ----------
    model : object
        Diffusion model.

    Examples
    --------
    >>> analyzer = NegativePromptAnalyzer(sd_model)
    >>> analysis = analyzer.analyze(
    ...     positive_prompt="beautiful landscape",
    ...     negative_prompt="ugly, blurry, low quality"
    ... )
    """

    def __init__(self, model: Any):
        self.model = model

    def analyze(
        self,
        positive_prompt: str,
        negative_prompt: str,
        generated_image: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze negative prompt effect.

        Parameters
        ----------
        positive_prompt : str
            Positive text prompt.
        negative_prompt : str
            Negative text prompt.
        generated_image : ndarray
            Generated image.

        Returns
        -------
        analysis : dict
            Negative prompt analysis.
        """
        # Tokenize negative prompt
        neg_tokens = negative_prompt.split(',')
        neg_tokens = [t.strip() for t in neg_tokens]

        # Measure suppression strength for each negative concept
        suppression_scores = {}
        for token in neg_tokens:
            # In practice: measure how well the concept is suppressed
            # score = measure_concept_presence(generated_image, token)
            # suppression_scores[token] = 1.0 - score

            # For demo:
            suppression_scores[token] = float(np.random.uniform(0.6, 0.95))

        return {
            'negative_tokens': neg_tokens,
            'suppression_scores': suppression_scores,
            'avg_suppression': np.mean(list(suppression_scores.values())),
            'metadata': {
                'positive_prompt': positive_prompt,
                'negative_prompt': negative_prompt
            }
        }


class LoRAExplainer:
    """
    Explainer for LoRA (Low-Rank Adaptation) models in Stable Diffusion.

    Explains how LoRA affects generation style/content.

    Parameters
    ----------
    base_model : object
        Base Stable Diffusion model.
    lora_weights : dict
        LoRA weight matrices.

    Examples
    --------
    >>> explainer = LoRAExplainer(sd_model, lora_weights)
    >>> effect = explainer.explain_lora_effect(prompt, image)
    """

    def __init__(
        self,
        base_model: Any,
        lora_weights: Optional[Dict[str, np.ndarray]] = None
    ):
        self.base_model = base_model
        self.lora_weights = lora_weights or {}

    def explain_lora_effect(
        self,
        prompt: str,
        image_with_lora: np.ndarray,
        image_without_lora: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Explain LoRA effect on generation.

        Parameters
        ----------
        prompt : str
            Input prompt.
        image_with_lora : ndarray
            Image generated with LoRA.
        image_without_lora : ndarray, optional
            Image generated without LoRA (for comparison).

        Returns
        -------
        explanation : dict
            LoRA effect explanation.
        """
        # Compute difference if baseline available
        if image_without_lora is not None:
            difference = np.abs(image_with_lora - image_without_lora)
            diff_magnitude = np.mean(difference)
        else:
            difference = None
            diff_magnitude = None

        # Analyze LoRA contribution by layer
        layer_contributions = {}
        for layer_name, weights in self.lora_weights.items():
            # In practice: measure activation change caused by LoRA
            # contribution = measure_lora_activation_change(layer_name)

            # For demo:
            layer_contributions[layer_name] = float(np.random.rand())

        return {
            'difference_magnitude': diff_magnitude,
            'difference_map': difference,
            'layer_contributions': layer_contributions,
            'dominant_layers': sorted(
                layer_contributions.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            'metadata': {
                'prompt': prompt,
                'n_lora_layers': len(self.lora_weights)
            }
        }


class DiffusionProcessVisualizer:
    """
    Visualize and explain the diffusion denoising process.

    Parameters
    ----------
    model : object
        Diffusion model.

    Examples
    --------
    >>> visualizer = DiffusionProcessVisualizer(sd_model)
    >>> timeline = visualizer.create_denoising_timeline(prompt, n_snapshots=10)
    """

    def __init__(self, model: Any):
        self.model = model

    def create_denoising_timeline(
        self,
        prompt: str,
        n_snapshots: int = 10,
        latents_history: Optional[List[np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Create timeline visualization of denoising process.

        Parameters
        ----------
        prompt : str
            Input prompt.
        n_snapshots : int
            Number of snapshots to take during generation.
        latents_history : list of ndarray, optional
            Latent codes history.

        Returns
        -------
        timeline : dict
            Denoising timeline with explanations.
        """
        if latents_history is None:
            # Simulate latent history
            latents_history = [
                np.random.randn(4, 64, 64) for _ in range(n_snapshots)
            ]

        # Select evenly spaced snapshots
        total_steps = len(latents_history)
        snapshot_indices = np.linspace(0, total_steps - 1, n_snapshots, dtype=int)

        timeline = {
            'snapshots': [],
            'descriptions': []
        }

        for idx in snapshot_indices:
            latent = latents_history[idx]
            timestep = idx

            # Describe what happens at this timestep
            if timestep < total_steps * 0.2:
                description = "Early stage: Global composition and layout forming"
            elif timestep < total_steps * 0.5:
                description = "Mid stage: Major objects and structures appearing"
            elif timestep < total_steps * 0.8:
                description = "Late stage: Fine details and textures emerging"
            else:
                description = "Final stage: Polishing and final adjustments"

            timeline['snapshots'].append({
                'timestep': int(timestep),
                'latent_shape': latent.shape,
                'description': description
            })

        return timeline


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Diffusion Model Explainability - Example")
    print("=" * 80)

    # Dummy generated image
    generated_image = np.random.rand(512, 512, 3)
    prompt = "A beautiful sunset over mountains with vibrant colors"

    print("\n1. STABLE DIFFUSION EXPLAINER")
    print("-" * 80)

    class DummySD:
        pass

    sd_model = DummySD()
    sd_explainer = StableDiffusionExplainer(sd_model)

    explanation = sd_explainer.explain(prompt, generated_image)

    print(f"Prompt: {prompt}")
    print(f"\nToken attribution:")
    tokens = prompt.split()
    for token, attr in zip(tokens, explanation.prompt_attribution):
        print(f"  '{token}': {attr:.4f}")

    print(f"\nTimestep importance (first 5 steps):")
    for t in range(5):
        print(f"  Step {t}: {explanation.timestep_importance[t]:.4f}")

    print(f"\nSpatial attribution shape: {explanation.spatial_attribution.shape}")

    print(f"\nConcept attribution:")
    for concept, score in explanation.concept_attribution.items():
        if score > 0.3:
            print(f"  {concept}: {score:.4f}")

    print("\n2. NEGATIVE PROMPT ANALYZER")
    print("-" * 80)

    neg_analyzer = NegativePromptAnalyzer(sd_model)
    neg_analysis = neg_analyzer.analyze(
        positive_prompt=prompt,
        negative_prompt="ugly, blurry, low quality, distorted",
        generated_image=generated_image
    )

    print(f"Negative prompt: {neg_analysis['metadata']['negative_prompt']}")
    print(f"Average suppression strength: {neg_analysis['avg_suppression']:.4f}")
    print(f"\nSuppression scores:")
    for token, score in neg_analysis['suppression_scores'].items():
        print(f"  '{token}': {score:.4f}")

    print("\n3. LORA EXPLAINER")
    print("-" * 80)

    # Simulate LoRA weights
    lora_weights = {
        'unet.down.0': np.random.randn(320, 8),
        'unet.down.1': np.random.randn(640, 8),
        'unet.mid': np.random.randn(1280, 8)
    }

    lora_explainer = LoRAExplainer(sd_model, lora_weights)

    image_without_lora = np.random.rand(512, 512, 3)
    lora_effect = lora_explainer.explain_lora_effect(
        prompt,
        generated_image,
        image_without_lora
    )

    print(f"LoRA effect magnitude: {lora_effect['difference_magnitude']:.4f}")
    print(f"\nTop 3 LoRA layers:")
    for layer_name, contribution in lora_effect['dominant_layers'][:3]:
        print(f"  {layer_name}: {contribution:.4f}")

    print("\n4. DIFFUSION PROCESS VISUALIZER")
    print("-" * 80)

    visualizer = DiffusionProcessVisualizer(sd_model)
    timeline = visualizer.create_denoising_timeline(prompt, n_snapshots=5)

    print(f"Denoising timeline ({len(timeline['snapshots'])} snapshots):")
    for snapshot in timeline['snapshots']:
        print(f"\n  Timestep {snapshot['timestep']}:")
        print(f"    {snapshot['description']}")

    print("\n" + "=" * 80)
    print("Diffusion model explainability demonstration complete!")
    print("=" * 80)
