"""Transverse-field Ising model and layered or QAOA-style ansatz."""
from __future__ import annotations

from math import pi

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

from vqe_lab.models import Workload


def hamiltonian(*, n: int, j: float, h: float) -> SparsePauliOp:
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer.")
    terms = [("ZZ", [qubit, qubit + 1], -j) for qubit in range(n - 1)]
    terms.extend(("X", [qubit], -h) for qubit in range(n))
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n).simplify()


def ansatz(
    *,
    n: int,
    p: int,
    kind: str = "qaoa",
    parametric: bool = True,
) -> QuantumCircuit:
    if not isinstance(n, int) or n <= 0 or not isinstance(p, int) or p <= 0:
        raise ValueError("n and p must be positive integers.")
    if kind not in {"qaoa", "layered"}:
        raise ValueError("kind must be 'qaoa' or 'layered'.")
    if kind == "qaoa":
        return _qaoa(n, p, parametric)
    return _layered(n, p, parametric)


def _qaoa(n: int, p: int, parametric: bool) -> QuantumCircuit:
    theta = ParameterVector("θ", 2 * p)
    circuit = QuantumCircuit(n)
    circuit.h(range(n))
    for layer in range(p):
        gamma, beta = theta[2 * layer : 2 * layer + 2]
        for qubit in range(n - 1):
            _rzz_block(circuit, 2 * gamma, qubit, qubit + 1, parametric)
        for qubit in range(n):
            circuit.rx(2 * beta, qubit)
    effective = list(range(0, 2 * p, 2))
    circuit.metadata = {
        "two_qubit_reference_angle": pi / 2,
        "two_qubit_angle_period": pi,
        "two_qubit_parameters": (
            [
                {"index": index, "angle_scale": 2.0, "gate_count": n - 1}
                for index in effective
            ]
            if parametric
            else []
        ),
    }
    return circuit


def _layered(n: int, p: int, parametric: bool) -> QuantumCircuit:
    edges = [(qubit, qubit + 1) for qubit in range(n - 1)]
    theta = ParameterVector("θ", p * (n + len(edges)))
    circuit = QuantumCircuit(n)
    two_qubit: list[int] = []
    index = 0
    for _ in range(p):
        for qubit in range(n):
            circuit.ry(theta[index], qubit)
            index += 1
        for left, right in edges:
            if parametric:
                two_qubit.append(index)
            _rzz_block(circuit, theta[index], left, right, parametric)
            index += 1
    circuit.metadata = {
        "two_qubit_reference_angle": pi / 2,
        "two_qubit_angle_period": pi,
        "two_qubit_parameters": [
            {"index": index, "angle_scale": 1.0, "gate_count": 1}
            for index in two_qubit
        ],
    }
    return circuit


def _rzz_block(circuit: QuantumCircuit, angle: object, left: int, right: int, parametric: bool) -> None:
    if parametric:
        circuit.rzz(angle, left, right)
        return
    circuit.rx(pi / 2, right)
    circuit.rzz(pi / 2, left, right)
    circuit.rx(angle, right)
    circuit.rzz(-pi / 2, left, right)
    circuit.rx(-pi / 2, right)


WORKLOAD = Workload(name="tfim", hamiltonian=hamiltonian, ansatz=ansatz)
