"""
Molecular Graph Explainability for Drug Discovery.

Explains GNN predictions on molecular graphs - crucial for drug discovery,
toxicity prediction, and molecular property prediction.

Author: XPLIA Team
License: MIT
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass

@dataclass
class MolecularExplanation:
    """
    Explanation for molecular property prediction.

    Attributes
    ----------
    atom_importance : ndarray
        Importance of each atom.
    bond_importance : ndarray
        Importance of each bond.
    functional_groups : list
        Important functional groups identified.
    pharmacophore : dict
        Pharmacophore features.
    metadata : dict
        Additional metadata.
    """
    atom_importance: np.ndarray
    bond_importance: np.ndarray
    functional_groups: List[str]
    pharmacophore: Dict[str, Any]
    metadata: Dict[str, Any]


class MolecularGNNExplainer:
    """
    Explainer for molecular property prediction using GNNs.

    Identifies important atoms, bonds, and functional groups.

    Parameters
    ----------
    model : object
        Molecular GNN model.
    atom_vocab : dict, optional
        Mapping of atom indices to types (C, N, O, etc.).

    Examples
    --------
    >>> explainer = MolecularGNNExplainer(mol_gnn)
    >>> explanation = explainer.explain_molecule(mol_graph, property='toxicity')
    """

    def __init__(
        self,
        model: Any,
        atom_vocab: Optional[Dict[int, str]] = None
    ):
        self.model = model
        self.atom_vocab = atom_vocab or {}

    def _identify_functional_groups(
        self,
        molecule: Dict[str, Any],
        atom_importance: np.ndarray
    ) -> List[str]:
        """
        Identify important functional groups.

        Parameters
        ----------
        molecule : dict
            Molecular graph with atoms and bonds.
        atom_importance : ndarray
            Importance of each atom.

        Returns
        -------
        functional_groups : list
            List of important functional groups.
        """
        # Common functional groups patterns
        functional_groups = []

        # In practice: pattern matching on molecular graph
        # - Hydroxyl (-OH)
        # - Carbonyl (C=O)
        # - Amino (-NH2)
        # - Carboxyl (-COOH)
        # - Benzene ring
        # etc.

        # For demo: simulate detection
        possible_groups = [
            'hydroxyl', 'carbonyl', 'amino', 'carboxyl',
            'benzene_ring', 'methyl', 'ester', 'amide'
        ]

        # Select groups with high atom importance
        for group in possible_groups:
            if np.random.rand() > 0.7:  # Simulate detection
                functional_groups.append(group)

        return functional_groups

    def _extract_pharmacophore(
        self,
        molecule: Dict[str, Any],
        atom_importance: np.ndarray
    ) -> Dict[str, Any]:
        """
        Extract pharmacophore features.

        Pharmacophore: 3D arrangement of features important for activity.

        Parameters
        ----------
        molecule : dict
            Molecular graph.
        atom_importance : ndarray
            Atom importance scores.

        Returns
        -------
        pharmacophore : dict
            Pharmacophore features.
        """
        # Pharmacophore features:
        # - Hydrogen bond donors/acceptors
        # - Hydrophobic regions
        # - Aromatic rings
        # - Charged groups

        # In practice: analyze 3D structure and interactions
        # For demo: simulate

        important_atoms = np.where(atom_importance > 0.5)[0]

        pharmacophore = {
            'h_bond_donors': int(np.sum(np.random.rand(len(important_atoms)) > 0.7)),
            'h_bond_acceptors': int(np.sum(np.random.rand(len(important_atoms)) > 0.6)),
            'hydrophobic_regions': int(np.sum(np.random.rand(len(important_atoms)) > 0.8)),
            'aromatic_rings': int(np.random.choice([0, 1, 2])),
            'charged_groups': int(np.random.choice([0, 1]))
        }

        return pharmacophore

    def explain_molecule(
        self,
        molecule: Dict[str, Any],
        property_name: str = 'activity'
    ) -> MolecularExplanation:
        """
        Explain molecular property prediction.

        Parameters
        ----------
        molecule : dict
            Molecular graph with 'atoms', 'bonds', 'coords' (optional).
        property_name : str
            Property being predicted (e.g., 'toxicity', 'activity', 'solubility').

        Returns
        -------
        explanation : MolecularExplanation
            Molecular explanation.
        """
        n_atoms = len(molecule['atoms'])
        n_bonds = len(molecule['bonds'])

        # In practice: use GNNExplainer or similar on molecular graph
        # atom_importance = gnn_explainer.explain(molecule)

        # For demo: simulate importance scores
        atom_importance = np.random.beta(2, 5, n_atoms)
        bond_importance = np.random.beta(2, 5, n_bonds)

        # Identify functional groups
        functional_groups = self._identify_functional_groups(molecule, atom_importance)

        # Extract pharmacophore
        pharmacophore = self._extract_pharmacophore(molecule, atom_importance)

        return MolecularExplanation(
            atom_importance=atom_importance,
            bond_importance=bond_importance,
            functional_groups=functional_groups,
            pharmacophore=pharmacophore,
            metadata={
                'property': property_name,
                'n_atoms': n_atoms,
                'n_bonds': n_bonds,
                'method': 'MolecularGNNExplainer'
            }
        )

    def explain_toxicity(
        self,
        molecule: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Explain toxicity prediction.

        Identifies structural alerts (toxicophores).

        Parameters
        ----------
        molecule : dict
            Molecular graph.

        Returns
        -------
        toxicity_explanation : dict
            Toxicity explanation with structural alerts.
        """
        explanation = self.explain_molecule(molecule, property_name='toxicity')

        # Identify structural alerts (toxicophores)
        # Common alerts: aromatic amines, nitro groups, aldehydes, etc.
        structural_alerts = []

        if 'amino' in explanation.functional_groups:
            structural_alerts.append({
                'alert': 'aromatic_amine',
                'severity': 'high',
                'description': 'Potential genotoxicity risk'
            })

        if 'carbonyl' in explanation.functional_groups:
            structural_alerts.append({
                'alert': 'aldehyde',
                'severity': 'medium',
                'description': 'Potential reactivity with proteins'
            })

        return {
            'atom_importance': explanation.atom_importance,
            'bond_importance': explanation.bond_importance,
            'structural_alerts': structural_alerts,
            'toxicophores': [alert['alert'] for alert in structural_alerts],
            'overall_risk': 'high' if len(structural_alerts) > 1 else 'medium' if structural_alerts else 'low'
        }


class DrugLikenessExplainer:
    """
    Explain drug-likeness predictions (Lipinski's Rule of Five, etc.).

    Parameters
    ----------
    model : object
        Drug-likeness prediction model.

    Examples
    --------
    >>> explainer = DrugLikenessExplainer(model)
    >>> explanation = explainer.explain_lipinski(molecule)
    """

    def __init__(self, model: Any):
        self.model = model

    def explain_lipinski(
        self,
        molecule: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Explain Lipinski's Rule of Five violations.

        Lipinski's Rule:
        - Molecular weight < 500 Da
        - LogP < 5
        - H-bond donors < 5
        - H-bond acceptors < 10

        Parameters
        ----------
        molecule : dict
            Molecular data.

        Returns
        -------
        explanation : dict
            Lipinski rule explanation.
        """
        # Compute molecular properties
        # In practice: use RDKit or similar
        # For demo: simulate

        mol_weight = float(np.random.uniform(200, 600))
        logp = float(np.random.uniform(-2, 6))
        h_donors = int(np.random.choice([0, 1, 2, 3, 5, 7]))
        h_acceptors = int(np.random.choice([1, 3, 5, 8, 12]))

        # Check violations
        violations = []

        if mol_weight > 500:
            violations.append({
                'rule': 'molecular_weight',
                'value': mol_weight,
                'threshold': 500,
                'severity': 'high'
            })

        if logp > 5:
            violations.append({
                'rule': 'logP',
                'value': logp,
                'threshold': 5,
                'severity': 'high'
            })

        if h_donors > 5:
            violations.append({
                'rule': 'h_bond_donors',
                'value': h_donors,
                'threshold': 5,
                'severity': 'medium'
            })

        if h_acceptors > 10:
            violations.append({
                'rule': 'h_bond_acceptors',
                'value': h_acceptors,
                'threshold': 10,
                'severity': 'medium'
            })

        return {
            'properties': {
                'molecular_weight': mol_weight,
                'logP': logp,
                'h_bond_donors': h_donors,
                'h_bond_acceptors': h_acceptors
            },
            'violations': violations,
            'drug_like': len(violations) == 0,
            'score': max(0, 100 - len(violations) * 25)
        }


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Molecular Graph Explainability - Example")
    print("=" * 80)

    # Simulate molecule (e.g., aspirin-like structure)
    molecule = {
        'atoms': ['C'] * 9 + ['O'] * 4,  # 9 carbons, 4 oxygens
        'bonds': [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),  # Benzene ring
                  (2, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12)],
        'coords': None  # 3D coordinates if available
    }

    print(f"\nMolecule: {len(molecule['atoms'])} atoms, {len(molecule['bonds'])} bonds")

    print("\n1. MOLECULAR GNN EXPLAINER (Activity Prediction)")
    print("-" * 80)

    class DummyMolGNN:
        pass

    mol_gnn = DummyMolGNN()
    mol_explainer = MolecularGNNExplainer(mol_gnn)

    activity_exp = mol_explainer.explain_molecule(molecule, property_name='activity')

    print(f"Property: {activity_exp.metadata['property']}")
    print(f"Method: {activity_exp.metadata['method']}")

    print(f"\nTop 5 important atoms:")
    top_atoms = np.argsort(activity_exp.atom_importance)[-5:][::-1]
    for atom_idx in top_atoms:
        atom_type = molecule['atoms'][atom_idx]
        importance = activity_exp.atom_importance[atom_idx]
        print(f"  Atom {atom_idx} ({atom_type}): {importance:.4f}")

    print(f"\nImportant functional groups:")
    for group in activity_exp.functional_groups:
        print(f"  - {group}")

    print(f"\nPharmacophore features:")
    for feature, count in activity_exp.pharmacophore.items():
        print(f"  {feature}: {count}")

    print("\n2. TOXICITY PREDICTION EXPLANATION")
    print("-" * 80)

    toxicity_exp = mol_explainer.explain_toxicity(molecule)

    print(f"Overall toxicity risk: {toxicity_exp['overall_risk']}")

    if toxicity_exp['structural_alerts']:
        print(f"\nStructural alerts detected:")
        for alert in toxicity_exp['structural_alerts']:
            print(f"  - {alert['alert']} (severity: {alert['severity']})")
            print(f"    {alert['description']}")
    else:
        print("\nNo structural alerts detected")

    print(f"\nToxicophores: {', '.join(toxicity_exp['toxicophores']) if toxicity_exp['toxicophores'] else 'None'}")

    print("\n3. DRUG-LIKENESS (Lipinski's Rule)")
    print("-" * 80)

    drug_like_explainer = DrugLikenessExplainer(mol_gnn)
    lipinski_exp = drug_like_explainer.explain_lipinski(molecule)

    print(f"Molecular properties:")
    for prop, value in lipinski_exp['properties'].items():
        print(f"  {prop}: {value:.2f}" if isinstance(value, float) else f"  {prop}: {value}")

    print(f"\nDrug-like: {lipinski_exp['drug_like']}")
    print(f"Drug-likeness score: {lipinski_exp['score']}/100")

    if lipinski_exp['violations']:
        print(f"\nLipinski violations:")
        for violation in lipinski_exp['violations']:
            print(f"  - {violation['rule']}: {violation['value']:.2f} > {violation['threshold']} "
                  f"(severity: {violation['severity']})")
    else:
        print("\nNo Lipinski violations - molecule is drug-like!")

    print("\n" + "=" * 80)
    print("Molecular explainability demonstration complete!")
    print("Use cases: Drug discovery, toxicity prediction, ADMET optimization")
    print("=" * 80)
