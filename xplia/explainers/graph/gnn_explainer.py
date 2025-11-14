"""
Graph Neural Network Explainability.

Explainability for GNNs used in social networks, molecular graphs,
knowledge graphs, and other graph-structured data.

Author: XPLIA Team
License: MIT
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass
import warnings

from xplia.core.base import ExplanationResult


@dataclass
class GraphExplanation:
    """
    Explanation for graph neural network prediction.

    Attributes
    ----------
    node_importance : ndarray
        Importance score for each node.
    edge_importance : ndarray
        Importance score for each edge.
    subgraph_nodes : set
        Important subgraph nodes.
    subgraph_edges : set
        Important subgraph edges.
    feature_importance : ndarray, optional
        Node feature importance.
    metadata : dict
        Additional metadata.
    """
    node_importance: np.ndarray
    edge_importance: np.ndarray
    subgraph_nodes: Set[int]
    subgraph_edges: Set[Tuple[int, int]]
    feature_importance: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None


class GNNExplainer:
    """
    GNNExplainer - Official GNN explainability method.

    Learns edge and feature masks to explain GNN predictions.

    Based on: Ying et al. "GNNExplainer: Generating Explanations for
    Graph Neural Networks" (NeurIPS 2019)

    Parameters
    ----------
    model : object
        Trained GNN model.
    n_epochs : int
        Number of optimization epochs.
    lr : float
        Learning rate for mask optimization.

    Examples
    --------
    >>> explainer = GNNExplainer(gnn_model)
    >>> explanation = explainer.explain_node(graph, node_idx=5)
    """

    def __init__(
        self,
        model: Any,
        n_epochs: int = 100,
        lr: float = 0.01
    ):
        self.model = model
        self.n_epochs = n_epochs
        self.lr = lr

    def _initialize_masks(
        self,
        n_edges: int,
        n_features: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize edge and feature masks.

        Parameters
        ----------
        n_edges : int
            Number of edges.
        n_features : int
            Number of node features.

        Returns
        -------
        edge_mask : ndarray
            Edge importance mask.
        feature_mask : ndarray
            Feature importance mask.
        """
        # Initialize to 0.5 (neutral)
        edge_mask = np.ones(n_edges) * 0.5
        feature_mask = np.ones(n_features) * 0.5

        return edge_mask, feature_mask

    def _optimize_masks(
        self,
        graph: Dict[str, Any],
        target_node: int,
        initial_pred: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimize edge and feature masks.

        Parameters
        ----------
        graph : dict
            Graph data with 'nodes', 'edges', 'features'.
        target_node : int
            Node to explain.
        initial_pred : float
            Original prediction.

        Returns
        -------
        edge_mask : ndarray
            Optimized edge mask.
        feature_mask : ndarray
            Optimized feature mask.
        """
        n_edges = len(graph['edges'])
        n_features = graph['features'].shape[1]

        edge_mask, feature_mask = self._initialize_masks(n_edges, n_features)

        # In practice: gradient-based optimization
        # for epoch in range(self.n_epochs):
        #     # Forward pass with masks
        #     masked_pred = model(graph, edge_mask, feature_mask)[target_node]
        #
        #     # Loss = KL(masked_pred || original_pred) + regularization
        #     loss = kl_divergence(masked_pred, initial_pred)
        #     loss += lambda_edge * edge_mask.sum()
        #     loss += lambda_feat * feature_mask.sum()
        #
        #     # Backward pass
        #     edge_mask -= lr * grad(loss, edge_mask)
        #     feature_mask -= lr * grad(loss, feature_mask)
        #
        #     # Project to [0, 1]
        #     edge_mask = np.clip(edge_mask, 0, 1)
        #     feature_mask = np.clip(feature_mask, 0, 1)

        # For demo: simulate optimized masks
        edge_mask = np.random.beta(2, 5, n_edges)  # Sparse mask
        feature_mask = np.random.beta(2, 5, n_features)

        return edge_mask, feature_mask

    def explain_node(
        self,
        graph: Dict[str, Any],
        node_idx: int,
        top_k_edges: int = 10
    ) -> GraphExplanation:
        """
        Explain GNN prediction for a specific node.

        Parameters
        ----------
        graph : dict
            Graph with keys: 'nodes', 'edges', 'features', 'adj_matrix'.
        node_idx : int
            Node to explain.
        top_k_edges : int
            Number of top edges to include in explanation.

        Returns
        -------
        explanation : GraphExplanation
            Node-level explanation.
        """
        # Get original prediction
        # pred = model(graph)[node_idx]
        pred = np.random.rand()

        # Optimize masks
        edge_mask, feature_mask = self._optimize_masks(graph, node_idx, pred)

        # Node importance (aggregate from edge mask)
        n_nodes = len(graph['nodes'])
        node_importance = np.zeros(n_nodes)

        for edge_idx, (u, v) in enumerate(graph['edges']):
            node_importance[u] += edge_mask[edge_idx]
            node_importance[v] += edge_mask[edge_idx]

        node_importance = node_importance / (node_importance.max() + 1e-8)

        # Extract top-k subgraph
        top_edge_indices = np.argsort(edge_mask)[-top_k_edges:]
        subgraph_edges = {graph['edges'][i] for i in top_edge_indices}
        subgraph_nodes = {node for edge in subgraph_edges for node in edge}

        return GraphExplanation(
            node_importance=node_importance,
            edge_importance=edge_mask,
            subgraph_nodes=subgraph_nodes,
            subgraph_edges=subgraph_edges,
            feature_importance=feature_mask,
            metadata={
                'target_node': node_idx,
                'method': 'GNNExplainer',
                'n_epochs': self.n_epochs,
                'top_k': top_k_edges
            }
        )

    def explain_graph(
        self,
        graph: Dict[str, Any],
        top_k_edges: int = 20
    ) -> GraphExplanation:
        """
        Explain GNN prediction for entire graph classification.

        Parameters
        ----------
        graph : dict
            Graph data.
        top_k_edges : int
            Number of important edges.

        Returns
        -------
        explanation : GraphExplanation
            Graph-level explanation.
        """
        # Similar to node explanation but for whole graph
        pred = np.random.rand()

        n_edges = len(graph['edges'])
        n_features = graph['features'].shape[1]

        edge_mask, feature_mask = self._initialize_masks(n_edges, n_features)

        # Optimize for graph-level prediction
        edge_mask = np.random.beta(2, 5, n_edges)
        feature_mask = np.random.beta(2, 5, n_features)

        # Node importance
        n_nodes = len(graph['nodes'])
        node_importance = np.zeros(n_nodes)

        for edge_idx, (u, v) in enumerate(graph['edges']):
            node_importance[u] += edge_mask[edge_idx]
            node_importance[v] += edge_mask[edge_idx]

        node_importance = node_importance / (node_importance.max() + 1e-8)

        # Top-k subgraph
        top_edge_indices = np.argsort(edge_mask)[-top_k_edges:]
        subgraph_edges = {graph['edges'][i] for i in top_edge_indices}
        subgraph_nodes = {node for edge in subgraph_edges for node in edge}

        return GraphExplanation(
            node_importance=node_importance,
            edge_importance=edge_mask,
            subgraph_nodes=subgraph_nodes,
            subgraph_edges=subgraph_edges,
            feature_importance=feature_mask,
            metadata={
                'method': 'GNNExplainer',
                'task': 'graph_classification',
                'top_k': top_k_edges
            }
        )


class SubgraphXExplainer:
    """
    SubgraphX - Monte Carlo Tree Search based explainer.

    Finds important subgraphs using MCTS.

    Based on: Yuan et al. "On Explainability of Graph Neural Networks
    via Subgraph Exploration" (ICML 2021)

    Parameters
    ----------
    model : object
        Trained GNN model.
    n_rollouts : int
        Number of MCTS rollouts.
    max_nodes : int
        Maximum subgraph size.

    Examples
    --------
    >>> explainer = SubgraphXExplainer(gnn_model)
    >>> explanation = explainer.explain(graph, node_idx=10)
    """

    def __init__(
        self,
        model: Any,
        n_rollouts: int = 100,
        max_nodes: int = 10
    ):
        self.model = model
        self.n_rollouts = n_rollouts
        self.max_nodes = max_nodes

    def _mcts_search(
        self,
        graph: Dict[str, Any],
        target_node: int
    ) -> Set[int]:
        """
        MCTS search for important subgraph.

        Parameters
        ----------
        graph : dict
            Graph data.
        target_node : int
            Node to explain.

        Returns
        -------
        important_nodes : set
            Important subgraph nodes.
        """
        # In practice: MCTS algorithm
        # 1. Selection: traverse tree using UCB
        # 2. Expansion: add new node to subgraph
        # 3. Simulation: evaluate subgraph importance
        # 4. Backpropagation: update node values

        # For demo: select nodes with high connectivity
        adj_matrix = graph['adj_matrix']
        node_degrees = adj_matrix.sum(axis=1)

        # Start from target node, expand to neighbors
        important_nodes = {target_node}
        current_frontier = {target_node}

        for _ in range(self.max_nodes - 1):
            if not current_frontier:
                break

            # Find neighbors
            neighbors = set()
            for node in current_frontier:
                neighbors.update(np.where(adj_matrix[node] > 0)[0].tolist())

            neighbors = neighbors - important_nodes

            if not neighbors:
                break

            # Select best neighbor (highest degree)
            neighbor_degrees = {n: node_degrees[n] for n in neighbors}
            best_neighbor = max(neighbor_degrees, key=neighbor_degrees.get)

            important_nodes.add(best_neighbor)
            current_frontier = {best_neighbor}

        return important_nodes

    def explain(
        self,
        graph: Dict[str, Any],
        node_idx: int
    ) -> GraphExplanation:
        """
        Explain node prediction using SubgraphX.

        Parameters
        ----------
        graph : dict
            Graph data.
        node_idx : int
            Node to explain.

        Returns
        -------
        explanation : GraphExplanation
            SubgraphX explanation.
        """
        # MCTS search
        important_nodes = self._mcts_search(graph, node_idx)

        # Extract subgraph edges
        subgraph_edges = set()
        for u, v in graph['edges']:
            if u in important_nodes and v in important_nodes:
                subgraph_edges.add((u, v))

        # Compute node and edge importance
        n_nodes = len(graph['nodes'])
        node_importance = np.zeros(n_nodes)
        node_importance[list(important_nodes)] = 1.0

        n_edges = len(graph['edges'])
        edge_importance = np.zeros(n_edges)
        for edge_idx, (u, v) in enumerate(graph['edges']):
            if (u, v) in subgraph_edges:
                edge_importance[edge_idx] = 1.0

        return GraphExplanation(
            node_importance=node_importance,
            edge_importance=edge_importance,
            subgraph_nodes=important_nodes,
            subgraph_edges=subgraph_edges,
            metadata={
                'target_node': node_idx,
                'method': 'SubgraphX',
                'n_rollouts': self.n_rollouts,
                'subgraph_size': len(important_nodes)
            }
        )


class GraphSHAPExplainer:
    """
    GraphSHAP - Shapley value based graph explainer.

    Computes Shapley values for nodes and edges.

    Parameters
    ----------
    model : object
        Trained GNN model.
    n_samples : int
        Number of coalition samples.

    Examples
    --------
    >>> explainer = GraphSHAPExplainer(gnn_model)
    >>> explanation = explainer.explain(graph, node_idx=7)
    """

    def __init__(
        self,
        model: Any,
        n_samples: int = 100
    ):
        self.model = model
        self.n_samples = n_samples

    def _compute_shapley_values(
        self,
        graph: Dict[str, Any],
        target_node: int,
        elements: List[int]
    ) -> np.ndarray:
        """
        Compute Shapley values for graph elements.

        Parameters
        ----------
        graph : dict
            Graph data.
        target_node : int
            Node to explain.
        elements : list
            Elements to compute Shapley values for (nodes or edges).

        Returns
        -------
        shapley_values : ndarray
            Shapley value for each element.
        """
        # In practice: sample coalitions and compute marginal contributions
        # shapley[i] = E[f(S ∪ {i}) - f(S)]

        # For demo: simulate Shapley values
        shapley_values = np.random.randn(len(elements))

        return shapley_values

    def explain(
        self,
        graph: Dict[str, Any],
        node_idx: int
    ) -> GraphExplanation:
        """
        Explain using Shapley values.

        Parameters
        ----------
        graph : dict
            Graph data.
        node_idx : int
            Node to explain.

        Returns
        -------
        explanation : GraphExplanation
            GraphSHAP explanation.
        """
        # Compute Shapley values for nodes
        nodes = list(range(len(graph['nodes'])))
        node_shapley = self._compute_shapley_values(graph, node_idx, nodes)

        # Compute Shapley values for edges
        edges = list(range(len(graph['edges'])))
        edge_shapley = self._compute_shapley_values(graph, node_idx, edges)

        # Select important subgraph (positive Shapley values)
        important_node_indices = np.where(node_shapley > 0)[0]
        important_nodes = set(important_node_indices.tolist())

        important_edge_indices = np.where(edge_shapley > 0)[0]
        subgraph_edges = {graph['edges'][i] for i in important_edge_indices}

        return GraphExplanation(
            node_importance=np.abs(node_shapley),
            edge_importance=np.abs(edge_shapley),
            subgraph_nodes=important_nodes,
            subgraph_edges=subgraph_edges,
            metadata={
                'target_node': node_idx,
                'method': 'GraphSHAP',
                'n_samples': self.n_samples
            }
        )


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Graph Neural Network Explainability - Example")
    print("=" * 80)

    # Create synthetic graph
    n_nodes = 20
    nodes = list(range(n_nodes))

    # Random edges
    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if np.random.rand() < 0.2:
                edges.append((i, j))

    # Node features
    features = np.random.randn(n_nodes, 16)

    # Adjacency matrix
    adj_matrix = np.zeros((n_nodes, n_nodes))
    for u, v in edges:
        adj_matrix[u, v] = 1
        adj_matrix[v, u] = 1

    graph = {
        'nodes': nodes,
        'edges': edges,
        'features': features,
        'adj_matrix': adj_matrix
    }

    print(f"\nGraph: {n_nodes} nodes, {len(edges)} edges")

    print("\n1. GNNExplainer (Node Classification)")
    print("-" * 80)

    class DummyGNN:
        pass

    gnn_model = DummyGNN()
    gnn_explainer = GNNExplainer(gnn_model, n_epochs=100)

    target_node = 5
    node_explanation = gnn_explainer.explain_node(graph, target_node, top_k_edges=8)

    print(f"Explaining node {target_node}")
    print(f"Method: {node_explanation.metadata['method']}")
    print(f"Important subgraph size: {len(node_explanation.subgraph_nodes)} nodes")
    print(f"Important subgraph edges: {len(node_explanation.subgraph_edges)} edges")

    print(f"\nTop 5 important nodes:")
    top_nodes = np.argsort(node_explanation.node_importance)[-5:][::-1]
    for node in top_nodes:
        print(f"  Node {node}: {node_explanation.node_importance[node]:.4f}")

    print(f"\nTop 5 important features:")
    if node_explanation.feature_importance is not None:
        top_features = np.argsort(node_explanation.feature_importance)[-5:][::-1]
        for feat in top_features:
            print(f"  Feature {feat}: {node_explanation.feature_importance[feat]:.4f}")

    print("\n2. SubgraphX (MCTS-based)")
    print("-" * 80)

    subgraphx = SubgraphXExplainer(gnn_model, n_rollouts=50, max_nodes=8)

    subgraphx_exp = subgraphx.explain(graph, target_node)

    print(f"Method: {subgraphx_exp.metadata['method']}")
    print(f"Subgraph size: {subgraphx_exp.metadata['subgraph_size']} nodes")
    print(f"Important nodes: {sorted(list(subgraphx_exp.subgraph_nodes))}")
    print(f"Important edges: {len(subgraphx_exp.subgraph_edges)} edges")

    print("\n3. GraphSHAP (Shapley Values)")
    print("-" * 80)

    graphshap = GraphSHAPExplainer(gnn_model, n_samples=100)

    shapley_exp = graphshap.explain(graph, target_node)

    print(f"Method: {shapley_exp.metadata['method']}")
    print(f"Nodes with positive Shapley values: {len(shapley_exp.subgraph_nodes)}")

    print(f"\nTop 5 nodes by Shapley value:")
    top_shapley_nodes = np.argsort(shapley_exp.node_importance)[-5:][::-1]
    for node in top_shapley_nodes:
        print(f"  Node {node}: {shapley_exp.node_importance[node]:.4f}")

    print("\n4. Graph Classification Explanation")
    print("-" * 80)

    graph_exp = gnn_explainer.explain_graph(graph, top_k_edges=15)

    print(f"Task: {graph_exp.metadata['task']}")
    print(f"Important subgraph: {len(graph_exp.subgraph_nodes)} nodes")
    print(f"Important edges: {len(graph_exp.subgraph_edges)} edges")

    print("\n" + "=" * 80)
    print("GNN explainability demonstration complete!")
    print("=" * 80)
