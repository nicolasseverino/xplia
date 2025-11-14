"""
Tests for TIER 1 - Graph Neural Network Explainers
Tests for GNN and Molecular explainability
"""

import pytest
import numpy as np
from typing import Dict, Any

from xplia.explainers.graph.gnn_explainer import (
    GNNExplainer,
    SubgraphXExplainer,
    GraphSHAPExplainer,
    GraphExplanation,
)
from xplia.explainers.graph.molecular_explainer import (
    MolecularGNNExplainer,
    DrugLikenessExplainer,
    MolecularExplanation,
)


class TestGNNExplainer:
    """Test suite for GNN explainer."""

    def create_dummy_graph(self) -> Dict[str, Any]:
        """Create a dummy graph for testing."""
        num_nodes = 10
        num_edges = 15

        return {
            'node_features': np.random.rand(num_nodes, 5),
            'edge_index': np.random.randint(0, num_nodes, (2, num_edges)),
            'edge_features': np.random.rand(num_edges, 3),
            'num_nodes': num_nodes,
            'num_edges': num_edges
        }

    def test_initialization(self):
        """Test GNNExplainer initialization."""
        explainer = GNNExplainer()
        assert explainer is not None

    def test_explain_node(self):
        """Test node classification explanation."""
        explainer = GNNExplainer()
        graph = self.create_dummy_graph()
        node_idx = 0

        explanation = explainer.explain_node(graph, node_idx)

        assert isinstance(explanation, GraphExplanation)
        assert hasattr(explanation, 'important_nodes')
        assert hasattr(explanation, 'important_edges')
        assert hasattr(explanation, 'node_importance')
        assert hasattr(explanation, 'edge_importance')

    def test_explain_graph(self):
        """Test graph classification explanation."""
        explainer = GNNExplainer()
        graph = self.create_dummy_graph()

        explanation = explainer.explain_graph(graph)

        assert explanation is not None
        assert hasattr(explanation, 'important_subgraph')

    def test_explain_invalid_node(self):
        """Test with invalid node index."""
        explainer = GNNExplainer()
        graph = self.create_dummy_graph()

        with pytest.raises((ValueError, IndexError)):
            explainer.explain_node(graph, node_idx=999)

    def test_explain_empty_graph(self):
        """Test with empty graph."""
        explainer = GNNExplainer()

        empty_graph = {
            'node_features': np.array([]),
            'edge_index': np.array([[], []]),
            'num_nodes': 0,
            'num_edges': 0
        }

        with pytest.raises((ValueError, AssertionError)):
            explainer.explain_graph(empty_graph)


class TestSubgraphXExplainer:
    """Test suite for SubgraphX explainer (MCTS-based)."""

    def create_dummy_graph(self) -> Dict[str, Any]:
        """Create a dummy graph for testing."""
        num_nodes = 15
        num_edges = 25

        return {
            'node_features': np.random.rand(num_nodes, 8),
            'edge_index': np.random.randint(0, num_nodes, (2, num_edges)),
            'num_nodes': num_nodes,
            'num_edges': num_edges
        }

    def test_initialization(self):
        """Test SubgraphX initialization."""
        explainer = SubgraphXExplainer(num_simulations=10)
        assert explainer is not None
        assert explainer.num_simulations == 10

    def test_explain_node(self):
        """Test node explanation with MCTS."""
        explainer = SubgraphXExplainer(num_simulations=5)
        graph = self.create_dummy_graph()

        explanation = explainer.explain_node(graph, node_idx=0)

        assert explanation is not None
        assert hasattr(explanation, 'subgraph_importance')

    def test_explain_with_more_simulations(self):
        """Test with different number of simulations."""
        explainer = SubgraphXExplainer(num_simulations=20)
        graph = self.create_dummy_graph()

        explanation = explainer.explain_node(graph, node_idx=0)

        assert explanation is not None


class TestGraphSHAPExplainer:
    """Test suite for GraphSHAP explainer."""

    def create_dummy_graph(self) -> Dict[str, Any]:
        """Create a dummy graph for testing."""
        num_nodes = 12
        num_edges = 20

        return {
            'node_features': np.random.rand(num_nodes, 6),
            'edge_index': np.random.randint(0, num_nodes, (2, num_edges)),
            'num_nodes': num_nodes,
            'num_edges': num_edges
        }

    def test_initialization(self):
        """Test GraphSHAP initialization."""
        explainer = GraphSHAPExplainer()
        assert explainer is not None

    def test_explain_node(self):
        """Test SHAP-based node explanation."""
        explainer = GraphSHAPExplainer()
        graph = self.create_dummy_graph()

        explanation = explainer.explain_node(graph, node_idx=0)

        assert explanation is not None
        assert hasattr(explanation, 'shapley_values')

    def test_explain_with_background_graphs(self):
        """Test with background graphs."""
        explainer = GraphSHAPExplainer()

        graph = self.create_dummy_graph()
        background_graphs = [self.create_dummy_graph() for _ in range(5)]

        explanation = explainer.explain_node(
            graph,
            node_idx=0,
            background_graphs=background_graphs
        )

        assert explanation is not None


class TestMolecularGNNExplainer:
    """Test suite for Molecular GNN explainer."""

    def create_dummy_molecule(self) -> Dict[str, Any]:
        """Create a dummy molecule graph."""
        num_atoms = 20
        num_bonds = 25

        return {
            'atom_features': np.random.rand(num_atoms, 10),
            'bond_index': np.random.randint(0, num_atoms, (2, num_bonds)),
            'bond_features': np.random.rand(num_bonds, 4),
            'num_atoms': num_atoms,
            'num_bonds': num_bonds,
            'smiles': 'CCO'  # Dummy SMILES string
        }

    def test_initialization(self):
        """Test Molecular GNN explainer initialization."""
        explainer = MolecularGNNExplainer()
        assert explainer is not None

    def test_explain_property(self):
        """Test molecular property explanation."""
        explainer = MolecularGNNExplainer()
        molecule = self.create_dummy_molecule()

        explanation = explainer.explain_property(
            molecule,
            property_name='toxicity'
        )

        assert isinstance(explanation, MolecularExplanation)
        assert hasattr(explanation, 'important_atoms')
        assert hasattr(explanation, 'important_bonds')
        assert hasattr(explanation, 'functional_groups')

    def test_explain_toxicity(self):
        """Test toxicity explanation."""
        explainer = MolecularGNNExplainer()
        molecule = self.create_dummy_molecule()

        explanation = explainer.explain_toxicity(molecule)

        assert explanation is not None
        assert hasattr(explanation, 'toxicophores')
        assert hasattr(explanation, 'structural_alerts')

    def test_explain_activity(self):
        """Test biological activity explanation."""
        explainer = MolecularGNNExplainer()
        molecule = self.create_dummy_molecule()

        explanation = explainer.explain_activity(
            molecule,
            target='protein_binding'
        )

        assert explanation is not None
        assert hasattr(explanation, 'pharmacophore')

    def test_identify_functional_groups(self):
        """Test functional group identification."""
        explainer = MolecularGNNExplainer()
        molecule = self.create_dummy_molecule()

        groups = explainer.identify_functional_groups(molecule)

        assert isinstance(groups, list)


class TestDrugLikenessExplainer:
    """Test suite for Drug-Likeness explainer."""

    def create_dummy_molecule(self) -> Dict[str, Any]:
        """Create a dummy molecule."""
        return {
            'molecular_weight': 250.0,
            'logp': 2.5,
            'num_hbd': 2,  # H-bond donors
            'num_hba': 3,  # H-bond acceptors
            'num_rotatable_bonds': 4,
            'tpsa': 60.0,  # Topological polar surface area
            'smiles': 'CCO'
        }

    def test_initialization(self):
        """Test initialization."""
        explainer = DrugLikenessExplainer()
        assert explainer is not None

    def test_explain_lipinski_rule(self):
        """Test Lipinski's Rule of Five explanation."""
        explainer = DrugLikenessExplainer()
        molecule = self.create_dummy_molecule()

        explanation = explainer.explain_lipinski(molecule)

        assert explanation is not None
        assert hasattr(explanation, 'passes_lipinski')
        assert hasattr(explanation, 'violations')
        assert hasattr(explanation, 'recommendations')

    def test_explain_bioavailability(self):
        """Test oral bioavailability explanation."""
        explainer = DrugLikenessExplainer()
        molecule = self.create_dummy_molecule()

        explanation = explainer.explain_bioavailability(molecule)

        assert explanation is not None
        assert hasattr(explanation, 'bioavailability_score')

    def test_lipinski_violations(self):
        """Test molecule with Lipinski violations."""
        explainer = DrugLikenessExplainer()

        # Create molecule with violations
        bad_molecule = {
            'molecular_weight': 600.0,  # Too high
            'logp': 6.0,  # Too high
            'num_hbd': 6,  # Too many
            'num_hba': 12,  # Too many
            'smiles': 'C' * 100
        }

        explanation = explainer.explain_lipinski(bad_molecule)

        assert explanation is not None
        assert not explanation.passes_lipinski
        assert len(explanation.violations) > 0

    def test_explain_admet(self):
        """Test ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) explanation."""
        explainer = DrugLikenessExplainer()
        molecule = self.create_dummy_molecule()

        explanation = explainer.explain_admet(molecule)

        assert explanation is not None
        assert hasattr(explanation, 'absorption_score')
        assert hasattr(explanation, 'toxicity_alerts')


# Integration tests
class TestGraphIntegration:
    """Integration tests for graph explainers."""

    def create_dummy_graph(self) -> Dict[str, Any]:
        """Create a dummy graph."""
        num_nodes = 10
        return {
            'node_features': np.random.rand(num_nodes, 5),
            'edge_index': np.random.randint(0, num_nodes, (2, 15)),
            'num_nodes': num_nodes,
            'num_edges': 15
        }

    def test_multiple_explainers_consistency(self):
        """Test consistency across different graph explainers."""
        graph = self.create_dummy_graph()
        node_idx = 0

        gnn_explainer = GNNExplainer()
        subgraphx_explainer = SubgraphXExplainer(num_simulations=5)
        graphshap_explainer = GraphSHAPExplainer()

        gnn_exp = gnn_explainer.explain_node(graph, node_idx)
        subgraphx_exp = subgraphx_explainer.explain_node(graph, node_idx)
        graphshap_exp = graphshap_explainer.explain_node(graph, node_idx)

        # All should produce valid explanations
        assert gnn_exp is not None
        assert subgraphx_exp is not None
        assert graphshap_exp is not None

    def test_molecular_workflow(self):
        """Test complete molecular explanation workflow."""
        mol_explainer = MolecularGNNExplainer()
        drug_explainer = DrugLikenessExplainer()

        molecule = {
            'atom_features': np.random.rand(20, 10),
            'bond_index': np.random.randint(0, 20, (2, 25)),
            'bond_features': np.random.rand(25, 4),
            'num_atoms': 20,
            'num_bonds': 25,
            'smiles': 'CCO',
            'molecular_weight': 250.0,
            'logp': 2.5,
            'num_hbd': 2,
            'num_hba': 3,
        }

        # Explain toxicity
        tox_exp = mol_explainer.explain_toxicity(molecule)

        # Explain drug-likeness
        drug_exp = drug_explainer.explain_lipinski(molecule)

        assert tox_exp is not None
        assert drug_exp is not None


# Performance tests
class TestGraphPerformance:
    """Performance tests for graph explainers."""

    def test_large_graph_performance(self):
        """Test performance on larger graphs."""
        explainer = GNNExplainer()

        # Create larger graph
        num_nodes = 100
        num_edges = 300

        large_graph = {
            'node_features': np.random.rand(num_nodes, 10),
            'edge_index': np.random.randint(0, num_nodes, (2, num_edges)),
            'num_nodes': num_nodes,
            'num_edges': num_edges
        }

        # Should handle larger graphs
        explanation = explainer.explain_node(large_graph, node_idx=0)

        assert explanation is not None

    def test_multiple_molecules_batch(self):
        """Test batch processing of molecules."""
        explainer = MolecularGNNExplainer()

        molecules = []
        for _ in range(10):
            molecules.append({
                'atom_features': np.random.rand(15, 10),
                'bond_index': np.random.randint(0, 15, (2, 20)),
                'bond_features': np.random.rand(20, 4),
                'num_atoms': 15,
                'num_bonds': 20,
                'smiles': 'CCO'
            })

        # Process batch
        explanations = []
        for mol in molecules:
            exp = explainer.explain_property(mol, 'toxicity')
            explanations.append(exp)

        assert len(explanations) == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
