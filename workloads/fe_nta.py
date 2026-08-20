"""Fe-NTA tensor-data workload with compatible Givens ansatz."""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Literal

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector
from qiskit.quantum_info import SparsePauliOp

from vqe_lab.models import Workload

SpinState = Literal["LS", "IS"]
Entanglement = Literal["full", "linear"]


def hamiltonian(
    *,
    molecule_dir: str | Path,
    spin_state: SpinState = "LS",
    penalty_strength: float = 0.01,
    apply_spin_penalty: bool = True,
) -> SparsePauliOp:
    """Build the spin-penalized Jordan-Wigner Hamiltonian from Fe-NTA tensors."""

    state = _spin_state(spin_state)
    if penalty_strength < 0:
        raise ValueError("penalty_strength must be non-negative.")
    folder = Path(molecule_dir)
    with (folder / "data_file.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    one_body = np.load(folder / "one_body.npy")
    two_body = np.load(folder / "two_body.npy")
    core_energy = float(np.load(folder / "core_energy.npy"))
    qubits = int(metadata.get("num_qubits", one_body.shape[0]))
    if one_body.shape != (qubits, qubits) or two_body.shape != (qubits,) * 4:
        raise ValueError("Fe-NTA tensor dimensions do not match num_qubits.")

    result = _identity(qubits, core_energy)
    for i, j in np.argwhere(np.abs(one_body) > 1e-12):
        result += one_body[i, j] * _fermion_product(
            qubits, [(int(i), True), (int(j), False)]
        )
    for i, j, k, l in np.argwhere(np.abs(two_body) > 1e-12):
        result += two_body[i, j, k, l] * _fermion_product(
            qubits,
            [(int(i), True), (int(j), True), (int(k), False), (int(l), False)],
        )
    if apply_spin_penalty and penalty_strength > 0:
        desired_s2 = float(metadata.get("s_squared", 0.75 if state == "LS" else 3.75))
        penalty = _spin_squared(qubits) - desired_s2 * _identity(qubits, 1)
        result += penalty_strength * (penalty @ penalty)
    return result.simplify(atol=1e-10)


def ansatz(
    *,
    num_qubits: int = 10,
    num_electrons: int = 5,
    reps: int = 1,
    spin_state: SpinState = "LS",
    parametric: bool = True,
    entanglement: Entanglement = "full",
    add_barriers: bool = False,
) -> QuantumCircuit:
    """Prepare spin sectors with Givens-rotation layers."""

    state = _spin_state(spin_state)
    if num_qubits <= 0 or num_qubits % 2 or reps < 0:
        raise ValueError("num_qubits must be positive and even; reps must be non-negative.")
    orbitals = num_qubits // 2
    up, down = _spin_counts(state, num_electrons)
    if min(up, down) < 0 or max(up, down) > orbitals:
        raise ValueError("Electron count is incompatible with the requested active space.")
    if entanglement == "full":
        pairs = list(combinations(range(orbitals), 2))
    elif entanglement == "linear":
        pairs = [(index, index + 1) for index in range(orbitals - 1)]
    else:
        raise ValueError("entanglement must be 'full' or 'linear'.")

    parameters_per_rep = 2 * len(pairs)
    theta = ParameterVector("θ", reps * parameters_per_rep)
    circuit = QuantumCircuit(num_qubits)
    for orbital in range(up):
        circuit.x(2 * orbital)
    for orbital in range(down):
        circuit.x(2 * orbital + 1)
    if add_barriers:
        circuit.barrier()

    entangling: list[int] = []
    index = 0
    for layer in range(reps):
        for spin in (0, 1):
            for left, right in pairs:
                _givens(
                    circuit,
                    theta[index],
                    2 * left + spin,
                    2 * right + spin,
                    parametric=parametric,
                )
                if parametric:
                    entangling.append(index)
                index += 1
            if add_barriers:
                circuit.barrier()
        for orbital in range(orbitals):
            circuit.cz(2 * orbital, 2 * orbital + 1)
        if add_barriers and layer != reps - 1:
            circuit.barrier()

    circuit.metadata = {
        "two_qubit_reference_angle": np.pi / 4,
        "two_qubit_angle_period": np.pi,
        "two_qubit_parameters": [
            {"index": parameter, "angle_scale": 1.0, "gate_count": 2}
            for parameter in entangling
        ],
    }
    return circuit


def _identity(qubits: int, coefficient: complex) -> SparsePauliOp:
    return SparsePauliOp.from_list([("I" * qubits, coefficient)])


def _ladder(qubits: int, index: int, dagger: bool) -> SparsePauliOp:
    local = SparsePauliOp.from_sparse_list(
        [("X", [index], 0.5), ("Y", [index], -0.5j if dagger else 0.5j)],
        num_qubits=qubits,
    )
    if index == 0:
        return local
    parity = SparsePauliOp.from_sparse_list(
        [("Z" * index, list(range(index)), 1)], num_qubits=qubits
    )
    return parity @ local


def _fermion_product(qubits: int, operators: list[tuple[int, bool]]) -> SparsePauliOp:
    result = _identity(qubits, 1)
    for index, dagger in operators:
        result = result @ _ladder(qubits, index, dagger)
    return result.simplify(atol=1e-12)


def _number_operator(qubits: int, index: int) -> SparsePauliOp:
    return _fermion_product(qubits, [(index, True), (index, False)])


def _spin_squared(qubits: int) -> SparsePauliOp:
    if qubits % 2:
        raise ValueError("spin sectors require an even number of qubits.")
    result = _identity(qubits, 0)
    for left in range(qubits // 2):
        for right in range(qubits // 2):
            result += _fermion_product(
                qubits,
                [
                    (2 * left, False),
                    (2 * right, True),
                    (2 * left + 1, True),
                    (2 * right + 1, False),
                ],
            )
    up = _identity(qubits, 0)
    down = _identity(qubits, 0)
    for index in range(0, qubits, 2):
        up += _number_operator(qubits, index)
    for index in range(1, qubits, 2):
        down += _number_operator(qubits, index)
    sz = 0.5 * (up - down)
    return (result + sz + sz @ sz).simplify(atol=1e-10)


def _spin_state(state: str) -> SpinState:
    normalized = state.upper()
    if normalized not in {"LS", "IS"}:
        raise ValueError("spin_state must be 'LS' or 'IS'.")
    return normalized  # type: ignore[return-value]


def _spin_counts(state: SpinState, electrons: int) -> tuple[int, int]:
    if state == "LS":
        return (electrons + 1) // 2, electrons // 2
    return (electrons + 3) // 2, (electrons - 3) // 2


def _xxplusyy_as_rzz(
    circuit: QuantumCircuit,
    angle: ParameterExpression | float,
    left: int,
    right: int,
    *,
    beta: float,
) -> None:
    if beta:
        circuit.rz(beta, left)
    circuit.h(left)
    circuit.h(right)
    circuit.rzz(angle / 2, left, right)
    circuit.h(left)
    circuit.h(right)
    circuit.rx(np.pi / 2, left)
    circuit.rx(np.pi / 2, right)
    circuit.rzz(angle / 2, left, right)
    circuit.rx(-np.pi / 2, left)
    circuit.rx(-np.pi / 2, right)
    if beta:
        circuit.rz(-beta, left)


def _givens(
    circuit: QuantumCircuit,
    angle: ParameterExpression,
    left: int,
    right: int,
    *,
    parametric: bool,
) -> None:
    if parametric:
        _xxplusyy_as_rzz(circuit, 2 * angle, left, right, beta=-np.pi / 2)
        return
    _xxplusyy_as_rzz(circuit, np.pi / 2, left, right, beta=0.0)
    circuit.rz(angle, left)
    circuit.rz(-angle, right)
    _xxplusyy_as_rzz(circuit, -np.pi / 2, left, right, beta=0.0)


WORKLOAD = Workload(
    name="fe_nta",
    hamiltonian=hamiltonian,
    ansatz=ansatz,
    metadata={"source": "IndustriallyRelevantVQA Fe(III)-NTA tensor data"},
)
