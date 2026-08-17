"""Staggered-fermion lattice Schwinger model with a charge-preserving ansatz."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import SparsePauliOp

from vqe_lab.models import Workload


def hamiltonian(
    *,
    N: int,
    F: int,
    m: float | Sequence[float],
    x: float,
    k: float | Sequence[float],
    theta: float = 0.0,
) -> SparsePauliOp:
    if not isinstance(N, int) or N <= 0 or not isinstance(F, int) or F <= 0:
        raise ValueError("N and F must be positive integers.")
    if x < 0:
        raise ValueError("x must be non-negative.")
    masses, chemical = _flavours(m, F, "m"), _flavours(k, F, "k")
    mu = 2 * np.sqrt(x) * np.asarray(masses)
    nu = 2 * np.sqrt(x) * np.asarray(chemical)
    qubits = N * F
    terms: list[tuple[str, complex]] = []

    def add(label: list[str], coefficient: complex) -> None:
        if not np.isclose(coefficient, 0.0):
            terms.append(("".join(label[::-1]), coefficient))

    hopping_sign = (-1) ** (F // 2 - 1) if F % 2 == 0 else (-1) ** ((F + 1) // 2)
    for left in range((N - 1) * F):
        right = left + F
        z_string = range(left + 1, right)
        if F % 2 == 0:
            paulis = (("X", "X", hopping_sign * x / 2), ("Y", "Y", hopping_sign * x / 2))
        else:
            paulis = (("Y", "X", hopping_sign * x / 2), ("X", "Y", -hopping_sign * x / 2))
        for left_pauli, right_pauli, coefficient in paulis:
            label = ["I"] * qubits
            label[left], label[right] = left_pauli, right_pauli
            for index in z_string:
                label[index] = "Z"
            add(label, coefficient)

    identity = N / 2 * float(nu.sum()) + qubits / 8 * (N + F - 1 + 4 * theta)
    add(["I"] * qubits, identity)
    for site in range(N):
        for flavour in range(F):
            coefficient = 0.5 * (mu[flavour] * (-1) ** site + nu[flavour])
            if site < N - 1:
                coefficient += (N - 1 - site) * theta + qubits / 4 - (site + 1) // 2 * F / 2
            label = ["I"] * qubits
            label[site * F + flavour] = "Z"
            add(label, coefficient)
    for left in range((N - 1) * F):
        for right in range(left + 1, (N - 1) * F):
            label = ["I"] * qubits
            label[left] = label[right] = "Z"
            add(label, 0.5 * (N - 1 - right // F))
    return SparsePauliOp.from_list(terms).simplify() if terms else SparsePauliOp.from_list([("I" * qubits, 0.0)])


def ansatz(*, N: int, F: int, reps: int, add_barriers: bool = False) -> QuantumCircuit:
    if not isinstance(reps, int) or reps <= 0:
        raise ValueError("reps must be a positive integer.")
    qubits = N * F
    if qubits < 2:
        raise ValueError("The ansatz needs at least two qubits.")
    pairs = [(index, index + 1) for index in range(0, qubits - 1, 2)]
    pairs += [(index, index + 1) for index in range(1, qubits - 1, 2)]
    parameters_per_rep = len(pairs) + qubits
    theta = ParameterVector("θ", reps * parameters_per_rep)
    circuit = QuantumCircuit(qubits)
    circuit.x(range(1, qubits, 2))
    entangling: list[int] = []
    index = 0
    for _ in range(reps):
        for left, right in pairs:
            circuit.append(XXPlusYYGate(theta[index]), [left, right])
            entangling.append(index)
            index += 1
        if add_barriers:
            circuit.barrier()
        for qubit in range(qubits):
            circuit.rz(theta[index], qubit)
            index += 1
    circuit.metadata = {
        "two_qubit_reference_angle": np.pi / 2,
        "two_qubit_angle_period": 2 * np.pi,
        "two_qubit_parameters": [
            {"index": index, "angle_scale": 1.0, "gate_count": 1}
            for index in entangling
        ],
    }
    return circuit


def _flavours(value: float | Sequence[float], count: int, name: str) -> list[float]:
    if np.isscalar(value):
        return [float(value)] * count
    values = [float(entry) for entry in value]
    if len(values) != count:
        raise ValueError(f"{name} must be scalar or contain F values.")
    return values


WORKLOAD = Workload(name="schwinger", hamiltonian=hamiltonian, ansatz=ansatz)
