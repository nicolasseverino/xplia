"""
Vision-Language Model Explainability.

Explainability for multimodal models like CLIP, BLIP, LLaVA, GPT-4V, Gemini.

Author: XPLIA Team
License: MIT
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
import warnings

from xplia.core.base import ExplanationResult


@dataclass
class VisionLanguageExplanation:
    """
    Explanation for vision-language models.

    Attributes
    ----------
    image_attribution : ndarray
        Attribution map for image regions.
    text_attribution : ndarray
        Attribution scores for text tokens.
    cross_modal_attention : ndarray
        Attention matrix between image and text.
    similarity_score : float
        Image-text similarity score.
    metadata : dict
        Additional metadata.
    """
    image_attribution: np.ndarray
    text_attribution: np.ndarray
    cross_modal_attention: np.ndarray
    similarity_score: float
    metadata: Dict[str, Any]


class CLIPExplainer:
    """
    Explainer for CLIP (Contrastive Language-Image Pre-training) models.

    Explains image-text similarity through attention visualization and
    gradient-based attribution.

    Parameters
    ----------
    model : object
        CLIP model with image and text encoders.
    image_size : tuple
        Expected image size (H, W).
    patch_size : int
        Vision transformer patch size.

    Examples
    --------
    >>> explainer = CLIPExplainer(clip_model)
    >>> explanation = explainer.explain(image, text="A dog playing in the park")
    """

    def __init__(
        self,
        model: Any,
        image_size: Tuple[int, int] = (224, 224),
        patch_size: int = 16
    ):
        self.model = model
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_patches = (image_size[0] // patch_size, image_size[1] // patch_size)

    def _get_attention_maps(
        self,
        image: np.ndarray,
        text: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract attention maps from CLIP model.

        Parameters
        ----------
        image : ndarray
            Input image.
        text : str
            Input text.

        Returns
        -------
        image_attention : ndarray
            Self-attention in image encoder.
        text_attention : ndarray
            Self-attention in text encoder.
        """
        # In practice: extract attention from model forward pass
        # image_features, image_attn = model.encode_image(image, output_attentions=True)
        # text_features, text_attn = model.encode_text(text, output_attentions=True)

        # For demo: simulate attention maps
        n_patches_total = self.n_patches[0] * self.n_patches[1]
        image_attention = np.random.rand(n_patches_total, n_patches_total)
        image_attention = image_attention / image_attention.sum(axis=1, keepdims=True)

        # Text attention
        tokens = text.split()
        n_tokens = len(tokens)
        text_attention = np.random.rand(n_tokens, n_tokens)
        text_attention = text_attention / text_attention.sum(axis=1, keepdims=True)

        return image_attention, text_attention

    def _compute_cross_modal_attention(
        self,
        image_features: np.ndarray,
        text_features: np.ndarray
    ) -> np.ndarray:
        """
        Compute cross-modal attention between image and text.

        Parameters
        ----------
        image_features : ndarray of shape (n_patches, d)
            Image patch features.
        text_features : ndarray of shape (n_tokens, d)
            Text token features.

        Returns
        -------
        cross_attention : ndarray of shape (n_patches, n_tokens)
            Cross-modal attention matrix.
        """
        # Compute similarity between image patches and text tokens
        # cross_attention = image_features @ text_features.T
        # cross_attention = softmax(cross_attention / temperature)

        # For demo: simulate
        n_patches = image_features.shape[0]
        n_tokens = text_features.shape[0]
        cross_attention = np.random.rand(n_patches, n_tokens)
        cross_attention = cross_attention / cross_attention.sum(axis=1, keepdims=True)

        return cross_attention

    def _compute_gradient_attribution(
        self,
        image: np.ndarray,
        text: str,
        target: str = 'similarity'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradient-based attribution.

        Parameters
        ----------
        image : ndarray
            Input image.
        text : str
            Input text.
        target : str
            Target to compute gradients for ('similarity', 'image', 'text').

        Returns
        -------
        image_grad : ndarray
            Gradients w.r.t. image.
        text_grad : ndarray
            Gradients w.r.t. text embeddings.
        """
        # In practice: compute gradients using autograd
        # similarity = model(image, text)
        # image_grad = grad(similarity, image)
        # text_grad = grad(similarity, text_embeddings)

        # For demo: simulate gradients
        image_grad = np.random.randn(*image.shape)

        tokens = text.split()
        text_grad = np.random.randn(len(tokens))

        return image_grad, text_grad

    def explain(
        self,
        image: np.ndarray,
        text: str,
        method: str = 'attention'
    ) -> VisionLanguageExplanation:
        """
        Explain CLIP model prediction.

        Parameters
        ----------
        image : ndarray
            Input image.
        text : str
            Input text.
        method : str
            Explanation method ('attention', 'gradient', 'integrated_gradients').

        Returns
        -------
        explanation : VisionLanguageExplanation
            Vision-language explanation.
        """
        # Encode image and text
        # In practice: image_features, text_features = model(image, text)
        # For demo:
        n_patches_total = self.n_patches[0] * self.n_patches[1]
        image_features = np.random.randn(n_patches_total, 512)

        tokens = text.split()
        text_features = np.random.randn(len(tokens), 512)

        # Compute similarity
        # similarity = cosine_similarity(image_features.mean(0), text_features.mean(0))
        similarity = float(np.random.rand())

        if method == 'attention':
            # Attention-based explanation
            image_attn, text_attn = self._get_attention_maps(image, text)

            # Average attention to get attribution
            image_attribution = image_attn.mean(axis=0)
            text_attribution = text_attn.mean(axis=0)

            cross_modal_attn = self._compute_cross_modal_attention(
                image_features, text_features
            )

        elif method == 'gradient':
            # Gradient-based explanation
            image_grad, text_grad = self._compute_gradient_attribution(image, text)

            # Convert gradients to attribution scores
            image_attribution = np.abs(image_grad).mean(axis=(0, 1, 2))  # Per patch
            text_attribution = np.abs(text_grad)

            # Cross-modal attention
            cross_modal_attn = self._compute_cross_modal_attention(
                image_features, text_features
            )

        else:
            raise ValueError(f"Unknown method: {method}")

        return VisionLanguageExplanation(
            image_attribution=image_attribution,
            text_attribution=text_attribution,
            cross_modal_attention=cross_modal_attn,
            similarity_score=similarity,
            metadata={
                'method': method,
                'image_shape': image.shape,
                'text_tokens': tokens,
                'n_patches': n_patches_total
            }
        )


class BLIPExplainer:
    """
    Explainer for BLIP (Bootstrapping Language-Image Pre-training) models.

    Explains image captioning and VQA (Visual Question Answering) tasks.

    Parameters
    ----------
    model : object
        BLIP model.
    task : str
        Task type ('captioning', 'vqa').

    Examples
    --------
    >>> explainer = BLIPExplainer(blip_model, task='captioning')
    >>> explanation = explainer.explain_caption(image)
    """

    def __init__(
        self,
        model: Any,
        task: str = 'captioning'
    ):
        self.model = model
        self.task = task

        valid_tasks = ['captioning', 'vqa']
        if task not in valid_tasks:
            raise ValueError(f"Task must be one of {valid_tasks}")

    def explain_caption(
        self,
        image: np.ndarray,
        caption: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Explain image caption generation.

        Parameters
        ----------
        image : ndarray
            Input image.
        caption : str, optional
            Ground truth or generated caption.

        Returns
        -------
        explanation : dict
            Caption explanation with token attributions.
        """
        # Generate caption if not provided
        if caption is None:
            # caption = model.generate_caption(image)
            caption = "A simulated caption for the image"

        tokens = caption.split()

        # For each token, compute image region importance
        token_attributions = []
        image_attributions = []

        for token in tokens:
            # In practice: compute attention or gradients for this token
            # attn = model.get_cross_attention(image, token)

            # For demo:
            token_attr = np.random.rand()
            image_attr = np.random.rand(14, 14)  # Spatial attribution

            token_attributions.append(token_attr)
            image_attributions.append(image_attr)

        return {
            'caption': caption,
            'tokens': tokens,
            'token_attributions': np.array(token_attributions),
            'image_attributions': np.stack(image_attributions),
            'metadata': {
                'task': 'captioning',
                'n_tokens': len(tokens)
            }
        }

    def explain_vqa(
        self,
        image: np.ndarray,
        question: str,
        answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Explain Visual Question Answering.

        Parameters
        ----------
        image : ndarray
            Input image.
        question : str
            Question about the image.
        answer : str, optional
            Model's answer.

        Returns
        -------
        explanation : dict
            VQA explanation.
        """
        # Generate answer if not provided
        if answer is None:
            # answer = model.answer_question(image, question)
            answer = "simulated answer"

        # Compute attribution for question tokens
        question_tokens = question.split()
        question_attr = np.random.rand(len(question_tokens))

        # Compute image region importance for answer
        image_attr = np.random.rand(14, 14)

        # Answer token attribution
        answer_tokens = answer.split()
        answer_attr = np.random.rand(len(answer_tokens))

        return {
            'question': question,
            'answer': answer,
            'question_tokens': question_tokens,
            'question_attribution': question_attr,
            'answer_tokens': answer_tokens,
            'answer_attribution': answer_attr,
            'image_attribution': image_attr,
            'metadata': {
                'task': 'vqa'
            }
        }


class MultimodalCounterfactualExplainer:
    """
    Generate counterfactual explanations for multimodal models.

    Finds minimal changes to image or text that change the prediction.

    Parameters
    ----------
    model : object
        Multimodal model.
    image_perturbation_budget : float
        Maximum allowed image perturbation (L2 norm).
    text_perturbation_budget : int
        Maximum number of text tokens to change.

    Examples
    --------
    >>> explainer = MultimodalCounterfactualExplainer(clip_model)
    >>> cf = explainer.find_counterfactual(image, text, target_similarity=0.9)
    """

    def __init__(
        self,
        model: Any,
        image_perturbation_budget: float = 0.1,
        text_perturbation_budget: int = 3
    ):
        self.model = model
        self.image_budget = image_perturbation_budget
        self.text_budget = text_perturbation_budget

    def find_counterfactual_image(
        self,
        image: np.ndarray,
        text: str,
        target_similarity: float,
        max_iterations: int = 100
    ) -> Tuple[np.ndarray, float]:
        """
        Find counterfactual image that achieves target similarity.

        Parameters
        ----------
        image : ndarray
            Original image.
        text : str
            Text query.
        target_similarity : float
            Target similarity score.
        max_iterations : int
            Maximum optimization iterations.

        Returns
        -------
        cf_image : ndarray
            Counterfactual image.
        similarity : float
            Achieved similarity.
        """
        # In practice: optimize image using projected gradient descent
        # cf_image = image.copy()
        # for i in range(max_iterations):
        #     grad = compute_gradient(cf_image, text, target_similarity)
        #     cf_image = cf_image - lr * grad
        #     cf_image = project_to_budget(cf_image, image, budget)

        # For demo: add small perturbation
        cf_image = image + np.random.randn(*image.shape) * self.image_budget
        similarity = np.random.uniform(target_similarity - 0.1, target_similarity + 0.1)

        return cf_image, float(similarity)

    def find_counterfactual_text(
        self,
        image: np.ndarray,
        text: str,
        target_similarity: float,
        vocabulary: Optional[List[str]] = None
    ) -> Tuple[str, float]:
        """
        Find counterfactual text that achieves target similarity.

        Parameters
        ----------
        image : ndarray
            Input image.
        text : str
            Original text.
        target_similarity : float
            Target similarity score.
        vocabulary : list of str, optional
            Allowed vocabulary for substitution.

        Returns
        -------
        cf_text : str
            Counterfactual text.
        similarity : float
            Achieved similarity.
        """
        # In practice: search over text perturbations
        # - Token substitution
        # - Token insertion/deletion
        # - Synonym replacement

        # For demo: replace one word
        tokens = text.split()
        if len(tokens) > 0:
            idx = np.random.randint(len(tokens))
            tokens[idx] = "[CHANGED]"

        cf_text = " ".join(tokens)
        similarity = np.random.uniform(target_similarity - 0.1, target_similarity + 0.1)

        return cf_text, float(similarity)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Vision-Language Model Explainability - Example")
    print("=" * 80)

    # Dummy image and text
    image = np.random.rand(224, 224, 3)
    text = "A dog playing in the park"

    print("\n1. CLIP EXPLAINER (Attention-based)")
    print("-" * 80)

    class DummyCLIP:
        pass

    clip_model = DummyCLIP()
    clip_explainer = CLIPExplainer(clip_model)

    explanation = clip_explainer.explain(image, text, method='attention')

    print(f"Text: {text}")
    print(f"Similarity score: {explanation.similarity_score:.4f}")
    print(f"Image attribution shape: {explanation.image_attribution.shape}")
    print(f"Text attribution: {explanation.text_attribution}")
    print(f"Cross-modal attention shape: {explanation.cross_modal_attention.shape}")

    # Show which text tokens attend to which image regions
    print(f"\nTop cross-modal connections:")
    tokens = text.split()
    for token_idx, token in enumerate(tokens[:3]):
        top_patches = np.argsort(explanation.cross_modal_attention[:, token_idx])[-3:][::-1]
        print(f"  '{token}' attends to image patches: {top_patches}")

    print("\n2. BLIP EXPLAINER (Image Captioning)")
    print("-" * 80)

    class DummyBLIP:
        pass

    blip_model = DummyBLIP()
    blip_explainer = BLIPExplainer(blip_model, task='captioning')

    caption_exp = blip_explainer.explain_caption(image)

    print(f"Generated caption: {caption_exp['caption']}")
    print(f"Token attributions: {caption_exp['token_attributions']}")
    print(f"Image attribution shape (per token): {caption_exp['image_attributions'].shape}")

    print("\n3. BLIP VQA EXPLAINER")
    print("-" * 80)

    blip_vqa = BLIPExplainer(blip_model, task='vqa')

    vqa_exp = blip_vqa.explain_vqa(
        image,
        question="What is the dog doing?",
        answer="playing"
    )

    print(f"Question: {vqa_exp['question']}")
    print(f"Answer: {vqa_exp['answer']}")
    print(f"Question token attribution: {vqa_exp['question_attribution']}")
    print(f"Answer token attribution: {vqa_exp['answer_attribution']}")
    print(f"Image attribution shape: {vqa_exp['image_attribution'].shape}")

    print("\n4. MULTIMODAL COUNTERFACTUAL")
    print("-" * 80)

    cf_explainer = MultimodalCounterfactualExplainer(clip_model)

    # Find counterfactual image
    cf_image, cf_sim = cf_explainer.find_counterfactual_image(
        image, text, target_similarity=0.9
    )
    print(f"Counterfactual image similarity: {cf_sim:.4f}")
    print(f"Image perturbation norm: {np.linalg.norm(cf_image - image):.4f}")

    # Find counterfactual text
    cf_text, cf_text_sim = cf_explainer.find_counterfactual_text(
        image, text, target_similarity=0.5
    )
    print(f"\nOriginal text: {text}")
    print(f"Counterfactual text: {cf_text}")
    print(f"Counterfactual similarity: {cf_text_sim:.4f}")

    print("\n" + "=" * 80)
    print("Vision-Language explainability demonstration complete!")
    print("=" * 80)
