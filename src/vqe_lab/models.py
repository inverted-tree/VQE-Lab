"""Load self-contained workload definitions from Python files."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

HamiltonianFactory = Callable[..., SparsePauliOp]
AnsatzFactory = Callable[..., QuantumCircuit]


@dataclass(frozen=True)
class Workload:
    """A Hamiltonian and variational-circuit factory pair."""

    name: str
    hamiltonian: HamiltonianFactory
    ansatz: AnsatzFactory
    metadata: dict[str, Any] = field(default_factory=dict)


def load_workload(path: str | Path) -> Workload:
    """Load the ``WORKLOAD`` object exported by a Python file."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Workload file does not exist: {source}")

    module_name = f"_vqe_lab_workload_{source.stem}_{abs(hash(source))}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import specification for {source}.")
    module: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    workload = getattr(module, "WORKLOAD", None)
    if not isinstance(workload, Workload):
        raise TypeError(f"{source} must export WORKLOAD = Workload(...).")
    return workload
