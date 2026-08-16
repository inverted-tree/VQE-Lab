"""Build, run, and collect exact and variational Qiskit experiments."""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Iterable, Literal

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import BackendEstimatorV2, StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B, SPSA

from .models import Workload

Backend = Literal["statevector", "aer"]


@dataclass(frozen=True)
class ExactResult:
    energy: float
    eigenvalues: np.ndarray
    eigenvector: np.ndarray | None
    elapsed_time: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class VQEResult:
    energy: float
    energy_history: list[float]
    parameter_history: list[np.ndarray]
    initial_point: np.ndarray
    initial_point_seed: int | None
    entropy_history: list[float]
    elapsed_time: float
    metadata: dict[str, Any]
    raw_result: Any | None = None


@dataclass(frozen=True)
class FailedRun:
    configuration: dict[str, Any]
    error: Exception


@dataclass(frozen=True)
class ExperimentResults:
    runs: list[VQEResult]
    failures: list[FailedRun] = field(default_factory=list)


def make_initial_point(
    num_parameters: int,
    *,
    point: np.ndarray | list[float] | None = None,
    seed: int | None = None,
    bounds: tuple[float, float] = (-np.pi, np.pi),
) -> np.ndarray:
    """Return an explicit, seeded-random, or zero initial parameter vector."""

    if not isinstance(num_parameters, int) or num_parameters < 0:
        raise ValueError("num_parameters must be a non-negative integer.")
    if point is not None and seed is not None:
        raise ValueError("Pass either point or seed, not both.")
    if not isinstance(bounds, tuple) or len(bounds) != 2:
        raise ValueError("bounds must be a (low, high) tuple.")
    low, high = map(float, bounds)
    if not low < high:
        raise ValueError("bounds must satisfy low < high.")
    if point is not None:
        values = np.asarray(point, dtype=float)
    elif seed is not None:
        values = np.random.default_rng(seed).uniform(low, high, num_parameters)
    else:
        values = np.zeros(num_parameters, dtype=float)
    if values.shape != (num_parameters,):
        raise ValueError(
            f"Initial point must have shape {(num_parameters,)}, got {values.shape}."
        )
    return values.copy()


class Experiment:
    """A reusable workload plus Hamiltonian and ansatz keyword arguments."""

    def __init__(
        self,
        workload: Workload,
        *,
        hamiltonian: dict[str, Any],
        ansatz: dict[str, Any] | None = None,
    ) -> None:
        self.workload = workload
        self.hamiltonian_kwargs = dict(hamiltonian)
        self.ansatz_kwargs = dict(ansatz or {})

    def exact(self, *, return_eigenvector: bool = False) -> ExactResult:
        """Dense-diagonalize the workload Hamiltonian."""

        start = time.perf_counter()
        operator = self._hamiltonian()
        matrix = np.asarray(operator.to_matrix(sparse=False), dtype=complex)
        if return_eigenvector:
            values, vectors = np.linalg.eigh(matrix)
            vector: np.ndarray | None = vectors[:, 0]
        else:
            values = np.linalg.eigvalsh(matrix)
            vector = None
        return ExactResult(
            energy=float(np.real(values[0])),
            eigenvalues=np.asarray(values, dtype=float),
            eigenvector=vector,
            elapsed_time=time.perf_counter() - start,
            metadata=self._metadata(),
        )

    def vqe(
        self,
        *,
        optimizer: str | Any = "cobyla",
        maxiter: int = 100,
        backend: Backend = "statevector",
        shots: int | None = None,
        seed: int | None = None,
        initial_point: np.ndarray | list[float] | None = None,
        initial_point_seed: int | None = None,
        initial_point_bounds: tuple[float, float] = (-np.pi, np.pi),
        optimizer_options: dict[str, Any] | None = None,
        backend_options: dict[str, Any] | None = None,
        entropy_partition: int | Iterable[int] | None = None,
    ) -> VQEResult:
        """Optimize the ansatz with a Qiskit estimator primitive."""
        
        start = time.perf_counter()
        operator, circuit = self._built_problem()
        point = make_initial_point(
            circuit.num_parameters,
            point=initial_point,
            seed=initial_point_seed,
            bounds=initial_point_bounds,
        )
        estimator, precision = _make_estimator(
            backend=backend,
            shots=shots,
            seed=seed,
            backend_options=backend_options or {},
        )
        energies: list[float] = []
        parameters: list[np.ndarray] = []
        entropies: list[float] = []

        def objective(values: np.ndarray) -> float:
            params = np.asarray(values, dtype=float).reshape(-1)
            energy = _expectation(estimator, circuit, operator, params, precision)
            parameters.append(params.copy())
            energies.append(energy)
            entropies.append(
                entropy(circuit, params, partition=entropy_partition)
            )
            return energy

        built_optimizer = _make_optimizer(
            optimizer, maxiter, optimizer_options or {}, seed
        )
        raw_result = built_optimizer.minimize(fun=objective, x0=point)
        if not energies:
            objective(np.asarray(getattr(raw_result, "x", point), dtype=float))
        metadata = self._metadata()
        metadata.update(
            {
                "backend": backend,
                "shots": shots,
                "optimizer": type(built_optimizer).__name__,
                "maxiter": maxiter,
                **_two_qubit_metadata(circuit),
            }
        )
        return VQEResult(
            energy=float(min(energies)),
            energy_history=energies,
            parameter_history=parameters,
            initial_point=point,
            initial_point_seed=initial_point_seed,
            entropy_history=entropies,
            elapsed_time=time.perf_counter() - start,
            metadata=metadata,
            raw_result=raw_result,
        )

    def with_configuration(
        self,
        *,
        hamiltonian: dict[str, Any] | None = None,
        ansatz: dict[str, Any] | None = None,
    ) -> Experiment:
        """Return a copy updated with selected factory keyword arguments."""

        return Experiment(
            self.workload,
            hamiltonian={**self.hamiltonian_kwargs, **(hamiltonian or {})},
            ansatz={**self.ansatz_kwargs, **(ansatz or {})},
        )

    def _hamiltonian(self) -> SparsePauliOp:
        operator = self.workload.hamiltonian(**self.hamiltonian_kwargs)
        if not isinstance(operator, SparsePauliOp):
            raise TypeError("Workload hamiltonian factory must return SparsePauliOp.")
        return operator

    def _built_problem(self) -> tuple[SparsePauliOp, QuantumCircuit]:
        operator = self._hamiltonian()
        circuit = self.workload.ansatz(**self.ansatz_kwargs)
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError("Workload ansatz factory must return QuantumCircuit.")
        if circuit.num_qubits != operator.num_qubits:
            raise ValueError("Ansatz and Hamiltonian have different qubit counts.")
        return operator, circuit

    def _metadata(self) -> dict[str, Any]:
        return {
            **self.workload.metadata,
            "workload": self.workload.name,
            "hamiltonian": dict(self.hamiltonian_kwargs),
            "ansatz": dict(self.ansatz_kwargs),
        }


def run_grid(
    workload: Workload,
    *,
    hamiltonian: dict[str, Any],
    ansatz: dict[str, Any],
    hamiltonian_grid: dict[str, Iterable[Any]] | None = None,
    ansatz_grid: dict[str, Iterable[Any]] | None = None,
    repeats: int = 1,
    seed: int | None = None,
    **vqe_kwargs: Any,
) -> ExperimentResults:
    """Run the Cartesian product of naturally named Hamiltonian and ansatz grids."""

    if not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("repeats must be a positive integer.")
    configurations = itertools.product(
        _grid_configurations(hamiltonian_grid),
        _grid_configurations(ansatz_grid),
        range(repeats),
    )
    runs: list[VQEResult] = []
    failures: list[FailedRun] = []
    for index, (h_update, a_update, repeat) in enumerate(configurations):
        experiment = Experiment(
            workload,
            hamiltonian={**hamiltonian, **h_update},
            ansatz={**ansatz, **a_update},
        )
        configuration = {
            "hamiltonian": experiment.hamiltonian_kwargs,
            "ansatz": experiment.ansatz_kwargs,
            "repeat": repeat,
        }
        try:
            run_kwargs = dict(vqe_kwargs)
            if (
                "initial_point" not in run_kwargs
                and "initial_point_seed" not in run_kwargs
            ):
                run_kwargs["initial_point_seed"] = (
                    None if seed is None else seed + index
                )
            runs.append(experiment.vqe(**run_kwargs))
        except Exception as error:
            failures.append(FailedRun(configuration, error))
    return ExperimentResults(runs=runs, failures=failures)


def entropy(
    state: QuantumCircuit | np.ndarray | list[complex],
    parameters: np.ndarray | list[float] | None = None,
    *,
    partition: int | Iterable[int] | None = None,
    tolerance: float = 1e-12,
) -> float:
    """Compute pure-state bipartite von Neumann entropy in bits.

    ``partition`` identifies the qubits in one subsystem; its complement forms
    the other. An integer selects the first number of qubits. When omitted, the
    first half of the register is used.
    """

    if isinstance(state, QuantumCircuit):
        bound = state
        if parameters is not None:
            values = np.asarray(parameters, dtype=float)
            if values.shape != (state.num_parameters,):
                raise ValueError("Parameter vector has the wrong shape.")
            bound = state.assign_parameters(values, inplace=False)
        vector = np.asarray(Statevector.from_instruction(bound).data, dtype=complex)
    else:
        if parameters is not None:
            raise ValueError("parameters are only valid when state is a circuit.")
        vector = np.asarray(state, dtype=complex)
    if vector.ndim != 1 or vector.size == 0 or vector.size & (vector.size - 1):
        raise ValueError("state must be a non-empty vector of length 2**n.")

    qubits = vector.size.bit_length() - 1
    if partition is None:
        subsystem = list(range(qubits // 2))
    elif isinstance(partition, int):
        if not 0 <= partition <= qubits:
            raise ValueError("partition must be between zero and the number of qubits.")
        subsystem = list(range(partition))
    else:
        try:
            subsystem = list(partition)
        except TypeError as error:
            raise ValueError("partition must contain qubit indices.") from error
    if any(not isinstance(qubit, int) or not 0 <= qubit < qubits for qubit in subsystem):
        raise ValueError("partition must contain qubit indices in the register.")
    if len(set(subsystem)) != len(subsystem):
        raise ValueError("partition must not contain duplicate qubit indices.")
    axes = [qubits - 1 - qubit for qubit in subsystem]
    axes.extend(qubits - 1 - qubit for qubit in range(qubits) if qubit not in subsystem)
    norm = np.linalg.norm(vector)
    if np.isclose(norm, 0.0):
        raise ValueError("state must not be the zero vector.")
    singular_values = np.linalg.svd(
        np.transpose((vector / norm).reshape((2,) * qubits), axes).reshape(
            2**len(subsystem), 2 ** (qubits - len(subsystem))
        ),
        compute_uv=False,
    )
    probabilities = singular_values**2
    probabilities = probabilities[probabilities > tolerance]
    return (
        float(-np.sum(probabilities * np.log2(probabilities)))
        if probabilities.size
        else 0.0
    )


def _grid_configurations(
    grid: dict[str, Iterable[Any]] | None,
) -> Iterable[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid)
    values = [list(grid[key]) for key in keys]
    return [
        dict(zip(keys, combination, strict=True))
        for combination in itertools.product(*values)
    ]


def _make_optimizer(
    optimizer: str | Any, maxiter: int, options: dict[str, Any], seed: int | None
) -> Any:
    if not isinstance(optimizer, str):
        return optimizer
    name = optimizer.lower().replace("_", "-")
    if name == "cobyla":
        return COBYLA(maxiter=maxiter, **options)
    if name in {"lbfgsb", "l-bfgs-b"}:
        return L_BFGS_B(maxiter=maxiter, **options)
    if name == "spsa":
        if seed is not None:
            from qiskit_algorithms.utils import algorithm_globals

            algorithm_globals.random_seed = seed
        return SPSA(maxiter=maxiter, **options)
    raise ValueError("optimizer must be cobyla, spsa, lbfgsb, or a Qiskit optimizer.")


def _make_estimator(
    *,
    backend: Backend,
    shots: int | None,
    seed: int | None,
    backend_options: dict[str, Any],
) -> tuple[Any, float | None]:
    if backend == "statevector":
        if backend_options:
            raise ValueError("backend_options are only valid for backend='aer'.")
        return StatevectorEstimator(seed=seed), None
    if backend == "aer":
        if not isinstance(shots, int) or shots <= 0:
            raise ValueError("shots must be a positive integer for backend='aer'.")
        simulator = AerSimulator(**backend_options)
        return BackendEstimatorV2(
            backend=simulator, options={"seed_simulator": seed}
        ), 1 / sqrt(shots)
    raise ValueError("backend must be 'statevector' or 'aer'.")


def _expectation(
    estimator: Any,
    circuit: QuantumCircuit,
    operator: SparsePauliOp,
    parameters: np.ndarray,
    precision: float | None,
) -> float:
    pub = (circuit, operator, [parameters])
    job = (
        estimator.run([pub])
        if precision is None
        else estimator.run([pub], precision=precision)
    )
    result = job.result()
    pub_result = result[0] if hasattr(result, "__getitem__") else result
    for holder in (getattr(pub_result, "data", None), pub_result):
        for name in ("evs", "values", "value"):
            value = getattr(holder, name, None)
            if value is not None and np.asarray(value).size == 1:
                return float(np.asarray(value, dtype=float).reshape(-1)[0])
    raise TypeError(
        f"Could not extract an expectation value from {type(result).__name__}."
    )


def _two_qubit_metadata(circuit: QuantumCircuit) -> dict[str, Any]:
    metadata = circuit.metadata or {}
    keys = (
        "two_qubit_reference_angle",
        "two_qubit_angle_period",
        "two_qubit_parameters",
    )
    return {key: metadata[key] for key in keys if key in metadata}
