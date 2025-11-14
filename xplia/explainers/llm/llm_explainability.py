"""
LLM and RAG Explainability.

Explainability methods for Large Language Models (LLMs) and
Retrieval-Augmented Generation (RAG) systems.

Author: XPLIA Team
License: MIT
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass
import warnings

from xplia.core.base import ExplanationResult


@dataclass
class TokenAttribution:
    """
    Attribution scores for tokens.

    Attributes
    ----------
    tokens : list of str
        Token strings.
    attributions : ndarray
        Attribution score for each token.
    metadata : dict
        Additional metadata.
    """
    tokens: List[str]
    attributions: np.ndarray
    metadata: Dict[str, Any]


@dataclass
class RAGExplanation:
    """
    Explanation for RAG system.

    Attributes
    ----------
    query : str
        User query.
    retrieved_docs : list
        Retrieved documents.
    doc_relevance_scores : ndarray
        Relevance score for each document.
    token_attributions : TokenAttribution
        Token-level attributions in response.
    metadata : dict
        Additional metadata.
    """
    query: str
    retrieved_docs: List[str]
    doc_relevance_scores: np.ndarray
    token_attributions: Optional[TokenAttribution] = None
    metadata: Optional[Dict[str, Any]] = None


class AttentionExplainer:
    """
    Explainer based on attention weights.

    Extracts and visualizes attention patterns in transformer models.

    Parameters
    ----------
    model : object
        Transformer model with attention outputs.
    tokenizer : object
        Tokenizer for the model.
    layer : int, optional
        Which layer's attention to use (-1 for last layer).
    head : int, optional
        Which attention head to use (-1 for averaged).

    Examples
    --------
    >>> explainer = AttentionExplainer(model, tokenizer)
    >>> attribution = explainer.explain("The cat sat on the mat")
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        layer: int = -1,
        head: int = -1
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.head = head

    def explain(
        self,
        text: str,
        **kwargs
    ) -> TokenAttribution:
        """
        Extract attention-based explanations.

        Parameters
        ----------
        text : str
            Input text.
        **kwargs
            Additional arguments.

        Returns
        -------
        attribution : TokenAttribution
            Token attributions based on attention.
        """
        # Tokenize
        tokens = self.tokenizer.tokenize(text) if hasattr(self.tokenizer, 'tokenize') else text.split()

        # In practice: get attention from model
        # attention = self.model(text, output_attentions=True).attentions
        # For demo: simulate attention weights

        n_tokens = len(tokens)
        attention_weights = np.random.rand(n_tokens, n_tokens)
        attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)

        # Aggregate attention as attribution
        # Common approach: sum incoming attention to each token
        attributions = attention_weights.sum(axis=0)
        attributions = attributions / attributions.sum()

        return TokenAttribution(
            tokens=tokens,
            attributions=attributions,
            metadata={
                'layer': self.layer,
                'head': self.head,
                'method': 'attention'
            }
        )


class IntegratedGradientsLLM:
    """
    Integrated Gradients for LLMs.

    Computes attributions by integrating gradients along path from
    baseline to input.

    Parameters
    ----------
    model : object
        LLM model.
    tokenizer : object
        Tokenizer.
    baseline : str, optional
        Baseline text (default: empty or padding).
    n_steps : int
        Number of integration steps.

    Examples
    --------
    >>> explainer = IntegratedGradientsLLM(model, tokenizer, n_steps=50)
    >>> attribution = explainer.explain("The cat sat on the mat")
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        baseline: Optional[str] = None,
        n_steps: int = 50
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.baseline = baseline
        self.n_steps = n_steps

    def explain(
        self,
        text: str,
        target_token: Optional[int] = None,
        **kwargs
    ) -> TokenAttribution:
        """
        Compute integrated gradients for input text.

        Parameters
        ----------
        text : str
            Input text.
        target_token : int, optional
            Target output token to explain. If None, explains max probability token.
        **kwargs
            Additional arguments.

        Returns
        -------
        attribution : TokenAttribution
            Token attributions via integrated gradients.
        """
        # Tokenize
        tokens = self.tokenizer.tokenize(text) if hasattr(self.tokenizer, 'tokenize') else text.split()

        # In practice:
        # 1. Get embeddings for input and baseline
        # 2. Interpolate between baseline and input
        # 3. Compute gradients at each interpolation step
        # 4. Integrate gradients
        #
        # input_embeds = model.get_input_embeddings()(input_ids)
        # baseline_embeds = model.get_input_embeddings()(baseline_ids)
        # for alpha in np.linspace(0, 1, n_steps):
        #     interpolated = baseline_embeds + alpha * (input_embeds - baseline_embeds)
        #     gradients = compute_gradients(model, interpolated, target)
        #     integrated_grads += gradients
        # attributions = (input_embeds - baseline_embeds) * integrated_grads

        # For demo: simulate attributions
        n_tokens = len(tokens)
        attributions = np.random.randn(n_tokens)
        attributions = np.abs(attributions)
        attributions = attributions / attributions.sum()

        return TokenAttribution(
            tokens=tokens,
            attributions=attributions,
            metadata={
                'method': 'integrated_gradients',
                'n_steps': self.n_steps,
                'target_token': target_token
            }
        )


class SHAPForLLM:
    """
    SHAP for LLMs.

    Adapts SHAP to text by treating tokens as features.

    Parameters
    ----------
    model : object
        LLM model.
    tokenizer : object
        Tokenizer.
    background_texts : list of str, optional
        Background dataset for SHAP.

    Examples
    --------
    >>> explainer = SHAPForLLM(model, tokenizer, background_texts=train_texts[:100])
    >>> attribution = explainer.explain("The cat sat on the mat")
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        background_texts: Optional[List[str]] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.background_texts = background_texts or [""]

    def explain(
        self,
        text: str,
        **kwargs
    ) -> TokenAttribution:
        """
        Compute SHAP values for tokens.

        Parameters
        ----------
        text : str
            Input text.
        **kwargs
            Additional arguments.

        Returns
        -------
        attribution : TokenAttribution
            Token SHAP values.
        """
        # Tokenize
        tokens = self.tokenizer.tokenize(text) if hasattr(self.tokenizer, 'tokenize') else text.split()

        # In practice: use SHAP explainer
        # explainer = shap.Explainer(model, background_data)
        # shap_values = explainer([text])

        # For demo: simulate SHAP values
        n_tokens = len(tokens)
        shap_values = np.random.randn(n_tokens)

        return TokenAttribution(
            tokens=tokens,
            attributions=shap_values,
            metadata={
                'method': 'shap',
                'n_background': len(self.background_texts)
            }
        )


class LIMEForLLM:
    """
    LIME for LLMs.

    Explains by perturbing input text and fitting local linear model.

    Parameters
    ----------
    model : object
        LLM model.
    tokenizer : object
        Tokenizer.
    n_samples : int
        Number of perturbations.

    Examples
    --------
    >>> explainer = LIMEForLLM(model, tokenizer, n_samples=1000)
    >>> attribution = explainer.explain("The cat sat on the mat")
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        n_samples: int = 1000
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.n_samples = n_samples

    def _perturb_text(self, tokens: List[str], mask_prob: float = 0.3) -> Tuple[List[str], np.ndarray]:
        """
        Perturb text by masking tokens.

        Parameters
        ----------
        tokens : list of str
            Original tokens.
        mask_prob : float
            Probability of masking each token.

        Returns
        -------
        perturbed_tokens : list of str
            Perturbed tokens.
        mask : ndarray
            Binary mask (1 = kept, 0 = masked).
        """
        mask = (np.random.rand(len(tokens)) > mask_prob).astype(int)
        perturbed_tokens = [token if mask[i] else '[MASK]' for i, token in enumerate(tokens)]
        return perturbed_tokens, mask

    def explain(
        self,
        text: str,
        **kwargs
    ) -> TokenAttribution:
        """
        Compute LIME explanation for text.

        Parameters
        ----------
        text : str
            Input text.
        **kwargs
            Additional arguments.

        Returns
        -------
        attribution : TokenAttribution
            Token attributions via LIME.
        """
        # Tokenize
        tokens = self.tokenizer.tokenize(text) if hasattr(self.tokenizer, 'tokenize') else text.split()
        n_tokens = len(tokens)

        # Generate perturbations
        perturbations = []
        masks = []
        predictions = []

        for _ in range(self.n_samples):
            perturbed_tokens, mask = self._perturb_text(tokens)
            perturbations.append(perturbed_tokens)
            masks.append(mask)

            # In practice: get model prediction on perturbed text
            # pred = model(perturbed_text)
            # For demo: simulate prediction
            pred = np.random.rand()
            predictions.append(pred)

        masks = np.array(masks)
        predictions = np.array(predictions)

        # Fit linear model: predictions ~ masks
        # In practice: use weighted linear regression with distance kernel
        # For demo: simple correlation
        attributions = np.zeros(n_tokens)
        for i in range(n_tokens):
            # Correlation between token presence and prediction
            attributions[i] = np.corrcoef(masks[:, i], predictions)[0, 1]

        # Handle NaN from constant masks
        attributions = np.nan_to_num(attributions)

        return TokenAttribution(
            tokens=tokens,
            attributions=attributions,
            metadata={
                'method': 'lime',
                'n_samples': self.n_samples
            }
        )


class RAGExplainer:
    """
    Explainer for Retrieval-Augmented Generation (RAG) systems.

    Explains both retrieval and generation components.

    Parameters
    ----------
    retriever : object
        Retriever component.
    generator : object
        Generator (LLM) component.
    tokenizer : object
        Tokenizer for generator.

    Examples
    --------
    >>> explainer = RAGExplainer(retriever, generator, tokenizer)
    >>> rag_exp = explainer.explain("What is machine learning?", doc_corpus)
    """

    def __init__(
        self,
        retriever: Any,
        generator: Any,
        tokenizer: Any
    ):
        self.retriever = retriever
        self.generator = generator
        self.tokenizer = tokenizer

    def explain_retrieval(
        self,
        query: str,
        retrieved_docs: List[str],
        doc_embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Explain document retrieval.

        Computes relevance scores for each retrieved document.

        Parameters
        ----------
        query : str
            User query.
        retrieved_docs : list of str
            Retrieved documents.
        doc_embeddings : ndarray, optional
            Precomputed document embeddings.

        Returns
        -------
        relevance_scores : ndarray
            Relevance score for each document.
        """
        # In practice: compute query-document similarity
        # query_embedding = retriever.encode(query)
        # doc_embeddings = retriever.encode(retrieved_docs)
        # relevance = cosine_similarity(query_embedding, doc_embeddings)

        # For demo: simulate relevance scores
        n_docs = len(retrieved_docs)
        relevance_scores = np.random.rand(n_docs)
        relevance_scores = relevance_scores / relevance_scores.sum()

        return relevance_scores

    def explain_generation(
        self,
        context: str,
        response: str
    ) -> TokenAttribution:
        """
        Explain generation given context.

        Shows which tokens in context contributed to response.

        Parameters
        ----------
        context : str
            Retrieved context.
        response : str
            Generated response.

        Returns
        -------
        attribution : TokenAttribution
            Context token attributions.
        """
        # Use attention-based explanation
        explainer = AttentionExplainer(self.generator, self.tokenizer)
        attribution = explainer.explain(context)

        return attribution

    def explain(
        self,
        query: str,
        retrieved_docs: List[str],
        response: str
    ) -> RAGExplanation:
        """
        Full RAG explanation.

        Parameters
        ----------
        query : str
            User query.
        retrieved_docs : list of str
            Retrieved documents.
        response : str
            Generated response.

        Returns
        -------
        rag_explanation : RAGExplanation
            Complete RAG explanation.
        """
        # Explain retrieval
        doc_relevance = self.explain_retrieval(query, retrieved_docs)

        # Explain generation
        context = " ".join(retrieved_docs)
        token_attrs = self.explain_generation(context, response)

        return RAGExplanation(
            query=query,
            retrieved_docs=retrieved_docs,
            doc_relevance_scores=doc_relevance,
            token_attributions=token_attrs,
            metadata={
                'n_docs': len(retrieved_docs),
                'response_length': len(response.split())
            }
        )


class PromptInfluenceAnalyzer:
    """
    Analyze influence of different prompt components.

    Measures how different parts of a prompt affect the output.

    Parameters
    ----------
    model : object
        LLM model.
    tokenizer : object
        Tokenizer.

    Examples
    --------
    >>> analyzer = PromptInfluenceAnalyzer(model, tokenizer)
    >>> influence = analyzer.analyze_prompt(
    ...     "You are a helpful assistant. User: What is AI? Assistant:",
    ...     components=["You are a helpful assistant.", "What is AI?"]
    ... )
    """

    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    def analyze_prompt(
        self,
        full_prompt: str,
        components: List[str]
    ) -> Dict[str, float]:
        """
        Analyze influence of prompt components.

        Parameters
        ----------
        full_prompt : str
            Complete prompt.
        components : list of str
            Prompt components to analyze.

        Returns
        -------
        influence_scores : dict
            Influence score for each component.
        """
        # In practice:
        # 1. Get baseline output with full prompt
        # 2. For each component, remove it and measure output change
        # 3. Influence = difference in output distribution

        # For demo: simulate influence scores
        influence_scores = {}
        for component in components:
            score = np.random.rand()
            influence_scores[component] = float(score)

        # Normalize
        total = sum(influence_scores.values())
        influence_scores = {k: v / total for k, v in influence_scores.items()}

        return influence_scores


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("LLM and RAG Explainability - Example")
    print("=" * 80)

    # Dummy tokenizer
    class SimpleTokenizer:
        def tokenize(self, text):
            return text.split()

    tokenizer = SimpleTokenizer()

    # Dummy model
    class DummyModel:
        pass

    model = DummyModel()

    print("\n1. ATTENTION-BASED EXPLANATION")
    print("-" * 80)
    attention_explainer = AttentionExplainer(model, tokenizer)
    text = "The quick brown fox jumps over the lazy dog"

    attribution = attention_explainer.explain(text)
    print(f"Text: {text}")
    print(f"Tokens: {attribution.tokens}")
    print(f"Attributions: {attribution.attributions}")
    print(f"Method: {attribution.metadata['method']}")

    print("\n2. INTEGRATED GRADIENTS FOR LLM")
    print("-" * 80)
    ig_explainer = IntegratedGradientsLLM(model, tokenizer, n_steps=50)

    attribution_ig = ig_explainer.explain("Machine learning is amazing")
    print(f"Tokens: {attribution_ig.tokens}")
    print(f"IG Attributions: {attribution_ig.attributions}")
    print(f"Top contributing tokens:")
    top_indices = np.argsort(attribution_ig.attributions)[-3:][::-1]
    for idx in top_indices:
        print(f"  {attribution_ig.tokens[idx]}: {attribution_ig.attributions[idx]:.4f}")

    print("\n3. SHAP FOR LLM")
    print("-" * 80)
    shap_explainer = SHAPForLLM(model, tokenizer)

    attribution_shap = shap_explainer.explain("AI will transform healthcare")
    print(f"Tokens: {attribution_shap.tokens}")
    print(f"SHAP values: {attribution_shap.attributions}")

    print("\n4. LIME FOR LLM")
    print("-" * 80)
    lime_explainer = LIMEForLLM(model, tokenizer, n_samples=500)

    attribution_lime = lime_explainer.explain("Natural language processing is powerful")
    print(f"Tokens: {attribution_lime.tokens}")
    print(f"LIME attributions: {attribution_lime.attributions}")
    print(f"Number of perturbations: {attribution_lime.metadata['n_samples']}")

    print("\n5. RAG EXPLAINABILITY")
    print("-" * 80)

    # Dummy retriever and generator
    retriever = DummyModel()
    generator = DummyModel()

    rag_explainer = RAGExplainer(retriever, generator, tokenizer)

    query = "What is machine learning?"
    retrieved_docs = [
        "Machine learning is a subset of AI that learns from data",
        "ML algorithms improve through experience",
        "Deep learning is a type of machine learning"
    ]
    response = "Machine learning is an AI technique that learns patterns from data"

    rag_exp = rag_explainer.explain(query, retrieved_docs, response)

    print(f"Query: {rag_exp.query}")
    print(f"\nDocument relevance scores:")
    for i, (doc, score) in enumerate(zip(rag_exp.retrieved_docs, rag_exp.doc_relevance_scores)):
        print(f"  Doc {i+1} (score={score:.4f}): {doc[:50]}...")

    print(f"\nToken attributions in context:")
    print(f"  Top tokens: {rag_exp.token_attributions.tokens[:5]}")
    print(f"  Attributions: {rag_exp.token_attributions.attributions[:5]}")

    print("\n6. PROMPT INFLUENCE ANALYSIS")
    print("-" * 80)
    prompt_analyzer = PromptInfluenceAnalyzer(model, tokenizer)

    full_prompt = "You are a helpful assistant. Answer concisely. User: What is AI?"
    components = [
        "You are a helpful assistant.",
        "Answer concisely.",
        "What is AI?"
    ]

    influence = prompt_analyzer.analyze_prompt(full_prompt, components)
    print(f"Prompt component influence:")
    for component, score in influence.items():
        print(f"  '{component}': {score:.4f}")

    print("\n" + "=" * 80)
    print("LLM and RAG explainability demonstration complete!")
    print("=" * 80)
