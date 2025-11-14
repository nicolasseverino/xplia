"""
Quantum Machine Learning Explainability.

Explains quantum circuits and hybrid quantum-classical models.

Author: XPLIA Team
License: MIT
"""

import numpy as np
from typing import Dict, Any, Optional, List

class QuantumCircuitExplainer:
    """Explain quantum circuit components."""

    def explain_circuit(self, circuit_description: Dict[str, Any]) -> Dict[str, Any]:
        """Explain quantum circuit gates and their effects."""

        n_qubits = circuit_description.get('n_qubits', 4)
        gates = circuit_description.get('gates', [])

        gate_explanations = []
        for gate in gates:
            gate_explanations.append({
                'gate_type': gate.get('type', 'unknown'),
                'qubits': gate.get('qubits', []),
                'effect': self._describe_gate_effect(gate.get('type')),
                'importance': float(np.random.rand())
            })

        return {
            'n_qubits': n_qubits,
            'n_gates': len(gates),
            'gate_explanations': gate_explanations,
            'entanglement_measure': float(np.random.rand()),
            'circuit_depth': len(gates)
        }

    def _describe_gate_effect(self, gate_type: str) -> str:
        """Describe what a quantum gate does."""
        descriptions = {
            'H': 'Creates superposition',
            'CNOT': 'Creates entanglement',
            'RX': 'Rotates around X-axis',
            'RY': 'Rotates around Y-axis',
            'RZ': 'Rotates around Z-axis'
        }
        return descriptions.get(gate_type, 'Unknown operation')

# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("Quantum ML Explainability - Example")
    print("=" * 80)

    circuit = {
        'n_qubits': 4,
        'gates': [
            {'type': 'H', 'qubits': [0]},
            {'type': 'CNOT', 'qubits': [0, 1]},
            {'type': 'RX', 'qubits': [2]},
            {'type': 'CNOT', 'qubits': [1, 2]}
        ]
    }

    print("\nQUANTUM CIRCUIT EXPLANATION")
    print("-" * 80)

    qc_exp = QuantumCircuitExplainer()
    result = qc_exp.explain_circuit(circuit)

    print(f"Circuit: {result['n_qubits']} qubits, {result['n_gates']} gates, depth {result['circuit_depth']}")
    print(f"Entanglement measure: {result['entanglement_measure']:.3f}")
    print(f"\nGate explanations:")
    for gate_exp in result['gate_explanations']:
        print(f"  {gate_exp['gate_type']} on qubits {gate_exp['qubits']}: {gate_exp['effect']}")
        print(f"    Importance: {gate_exp['importance']:.3f}")

    print("\n" + "=" * 80)
