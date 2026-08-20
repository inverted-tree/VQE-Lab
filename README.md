<div align="center">
    <h1>VQE Lab</h1>
    <p>A compact, notebook-centric library for small VQE studies.<p>
    <a href="https://www.python.org/">
        <img src="https://upload.wikimedia.org/wikipedia/commons/0/0a/Python.svg" alt="Python" width=64>
    </a>
    &nbsp;
    &nbsp;
    <a href="https://www.ibm.com/quantum/qiskit">
        <img src="https://upload.wikimedia.org/wikipedia/commons/5/51/Qiskit-Logo.svg" alt="Qiskit" width=64>
    </a>
    &nbsp;
    &nbsp;
    <a href="https://matplotlib.org/">
        <img src="https://upload.wikimedia.org/wikipedia/commons/8/84/Matplotlib_icon.svg" alt="Matplotlib" width=64>
    </a>
</div>
<br>
<br>

## What This Library Does
This library works by providing a hamiltonian and ansatz implementation for a physical model of interest and using the built in convenience functions to run a full VQE analysis with multiple parameters.

I used this library for analyzing entangler parametrizations for VQE ansätze my masters thesis.

## Getting Started
Inside an analysis notebook, set up the model with the expected parameters:


```python
import vqe_lab as vqe

model = vqe.load_workload("workloads/my-workload.py")
experiment = vqe.Experiment(
    model,
    hamiltonian={"n": 8, "j": 1.0, "h": 1.0},
    ansatz={"n": 8, "p": 2, "kind": "qaoa"},
)
```

You can calculate a reference energy and a VQE estimate and quickly compare the results:

```python
exact = experiment.exact()
run = experiment.vqe(optimizer="cobyla", maxiter=200, initial_point_seed=42)
vqe.plot_convergence(run, exact=exact)
```

After your model produces satisfying results, run parameter sweeps with naturally named grids:

```python
results = vqe.run_grid(
    model,
    hamiltonian={"n": 8, "j": 1.0, "h": 1.0},
    ansatz={"n": 8, "kind": "qaoa"},
    ansatz_grid={"p": [1, 2, 3], "parametric": [True, False]},
    repeats=5,
    seed=100,
    optimizer="spsa",
    maxiter=300,
)
vqe.plot_convergence(results.runs, exact=exact)
```

A workload is a Python file that exports one `WORKLOAD` object:

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from vqe_lab.models import Workload

def hamiltonian(*, n: int) -> SparsePauliOp:
    return SparsePauliOp.from_list([("Z" * n, 1.0)])

def ansatz(*, n: int) -> QuantumCircuit:
    return QuantumCircuit(n)

WORKLOAD = Workload("example", hamiltonian, ansatz)
```

Bundled examples are in `workloads/`: `tfim.py`, `schwinger.py`, and `fe_nta.py`.
The Fe-NTA workload accepts a released tensor-data directory through `molecule_dir`.
I have used the reference values from [this paper](https://doi.org/10.1088/2058-9565/ad9ed3), provided in [this repository](https://github.com/ludwig-831/IndustriallyRelevantVQA).
