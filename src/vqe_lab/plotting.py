"""Summarize and visualize VQE result histories in notebooks."""

from __future__ import annotations

from collections.abc import Sequence
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit

from .experiment import ExactResult, VQEResult

ResultGroups = (
    VQEResult
    | Sequence[VQEResult]
    | tuple[str, Sequence[VQEResult]]
    | Sequence[tuple[str, Sequence[VQEResult]]]
)


def draw_ansatz(
    circuit: QuantumCircuit, *, output: str = "mpl", **kwargs: object
) -> object:
    """Draw a circuit with notebook-friendly defaults."""

    if output == "mpl":
        kwargs.setdefault("fold", 40)
        kwargs.setdefault("idle_wires", False)
    return circuit.draw(output=output, **kwargs)


def plot_convergence(
    results: ResultGroups, *, exact: ExactResult | float | None = None
) -> plt.Axes:
    """Plot best-so-far energy histories, including group mean envelopes."""

    figure, axis = plt.subplots()
    for label, group in _groups(results):
        _plot_history(
            axis,
            [np.minimum.accumulate(run.energy_history) for run in group],
            label,
        )
    if exact is not None:
        axis.axhline(_exact_energy(exact), color="black", linestyle="--", label="exact")
    axis.set(xlabel="Objective evaluation", ylabel="Energy", title="VQE convergence")
    axis.legend()
    figure.tight_layout()
    return axis


def plot_energy_error(results: ResultGroups, exact: ExactResult | float) -> plt.Axes:
    """Plot energy histories relative to an exact reference."""

    figure, axis = plt.subplots()
    reference = _exact_energy(exact)
    for label, group in _groups(results):
        _plot_history(
            axis,
            [[energy - reference for energy in run.energy_history] for run in group],
            label,
        )
    axis.axhline(0.0, color="black", linestyle="--", label="exact")
    axis.set(
        xlabel="Objective evaluation", ylabel="Energy error", title="VQE energy error"
    )
    axis.legend()
    figure.tight_layout()
    return axis


def plot_entropy(
    results: ResultGroups, *, exact: float | None = None
) -> plt.Axes:
    """Plot bipartite entropy histories against an optional exact reference."""

    figure, axis = plt.subplots()
    for label, group in _groups(results):
        _plot_history(axis, [run.entropy_history for run in group], label)
    if exact is not None:
        axis.axhline(float(exact), color="black", linestyle="--", label="exact")
    axis.set(
        xlabel="Objective evaluation",
        ylabel="Entropy [bits]",
        title="Entanglement entropy",
    )
    axis.legend()
    figure.tight_layout()
    return axis


def plot_parameters(
    results: ResultGroups,
    *,
    absolute: bool = False,
    total_angle: bool = False,
    heatmap: bool = False,
) -> plt.Axes:
    """Plot effective two-qubit gate angles from workload metadata.

    By default, the y-axis is the normalized average angle per physical gate.
    ``total_angle`` instead plots the non-normalized aggregate angle in radians.
    ``heatmap`` plots the best-energy normalized angle of every parameter instead.
    """

    if heatmap:
        if total_angle:
            raise ValueError("total_angle cannot be combined with heatmap.")
        return _plot_final_angle_heatmap(results, absolute=absolute)

    figure, axis = plt.subplots()
    for group_label, group in _groups(results):
        for run_index, run in enumerate(group, start=1):
            history = _two_qubit_angle_history(
                run, absolute=absolute, total_angle=total_angle
            )
            if history is not None:
                axis.plot(
                    np.arange(1, len(history) + 1),
                    history,
                    label=f"{group_label} {run_index}",
                )
    limit = 1.0
    if total_angle:
        limits = []
        for _, group in _groups(results):
            for run in group:
                specification = _two_qubit_specification(run)
                if specification is not None:
                    reference, _, _, _, counts, _ = specification
                    limits.append(reference * counts.sum())
        if limits:
            limit = float(max(limits))
    axis.set(
        xlabel="Objective evaluation",
        ylabel="Total effective angle [rad]" if total_angle else "Normalized angle per gate",
        title="Effective two-qubit gate angles",
        ylim=(0.0 if absolute else -limit, limit),
    )
    axis.legend()
    figure.tight_layout()
    return axis


def _plot_final_angle_heatmap(
    results: ResultGroups, *, absolute: bool
) -> plt.Axes:
    """Plot best-energy normalized two-qubit angles for all runs as a heatmap."""

    rows: list[np.ndarray] = []
    row_labels: list[str] = []
    specification: list[tuple[int, float]] | None = None
    for group_label, group in _groups(results):
        for run_index, run in enumerate(group, start=1):
            final = _final_two_qubit_angles(run, absolute=absolute, normalized=True)
            if final is None:
                continue
            values, current_specification = final
            if specification is None:
                specification = current_specification
            elif current_specification != specification:
                raise ValueError("All plotted runs must use the same two-qubit gates.")
            rows.append(values)
            row_labels.append(f"{group_label} {run_index}")
    if not rows or specification is None:
        raise ValueError("At least one run with parameterized two-qubit gates is required.")

    values = np.asarray(rows)
    limit = max(1.0, float(np.abs(values).max()))
    figure, axis = plt.subplots()
    image = axis.imshow(
        values,
        aspect="auto",
        cmap="viridis" if absolute else "coolwarm",
        vmin=0.0 if absolute else -limit,
        vmax=limit,
    )
    axis.set(
        xticks=range(len(specification)),
        xticklabels=[_two_qubit_label(parameter) for parameter in specification],
        yticks=range(len(row_labels)),
        yticklabels=row_labels,
        xlabel="Two-qubit parameter",
        ylabel="VQE run",
        title="Best-energy two-qubit angles",
    )
    figure.colorbar(image, ax=axis, label="Normalized angle")
    figure.tight_layout()
    return axis


def _groups(results: ResultGroups) -> list[tuple[str, list[VQEResult]]]:
    if isinstance(results, VQEResult):
        return [("VQE", [results])]
    if isinstance(results, tuple) and len(results) == 2 and isinstance(results[0], str):
        return [(results[0], list(results[1]))]
    values = list(results)
    if not values:
        raise ValueError("At least one VQE result is required.")
    if isinstance(values[0], VQEResult):
        return [("VQE", values)]
    return [(label, list(group)) for label, group in values]


def _plot_history(
    axis: plt.Axes, histories: Sequence[Sequence[float]], label: str
) -> None:
    values = _pad_series([np.asarray(history, dtype=float) for history in histories])
    mean = values.mean(axis=0)
    x = np.arange(1, len(mean) + 1)
    line = axis.plot(x, mean, label=label)[0]
    if len(histories) > 1:
        axis.fill_between(
            x,
            values.min(axis=0),
            values.max(axis=0),
            color=line.get_color(),
            alpha=0.15,
        )


def _two_qubit_angle_history(
    result: VQEResult, *, absolute: bool, total_angle: bool
) -> np.ndarray | None:
    specification = _two_qubit_specification(result)
    if specification is None:
        return None
    reference, period, indices, scales, counts, _ = specification
    values = _canonical_two_qubit_angles(
        _pad_parameters(result.parameter_history)[:, indices] * scales,
        period=period,
        absolute=absolute,
    )
    total = (values * counts).sum(axis=1)
    return total if total_angle else total / counts.sum() / reference


def _final_two_qubit_angles(
    result: VQEResult, *, absolute: bool, normalized: bool
) -> tuple[np.ndarray, list[tuple[int, float]]] | None:
    specification = _two_qubit_specification(result)
    if specification is None:
        return None
    reference, period, indices, scales, counts, labels = specification
    if len(result.energy_history) != len(result.parameter_history):
        raise ValueError("Energy and parameter histories must have the same length.")
    best_index = int(np.argmin(result.energy_history))
    values = _canonical_two_qubit_angles(
        result.parameter_history[best_index][indices] * scales,
        period=period,
        absolute=absolute,
    )
    return (values / reference if normalized else values), labels


def _two_qubit_specification(
    result: VQEResult,
) -> tuple[float, float, list[int], np.ndarray, np.ndarray, list[tuple[int, float]]] | None:
    metadata = result.metadata
    reference = metadata.get("two_qubit_reference_angle")
    period = metadata.get("two_qubit_angle_period")
    parameters = metadata.get("two_qubit_parameters")
    if reference is None:
        raise ValueError("Workload metadata must define two_qubit_reference_angle.")
    if period is None:
        raise ValueError("Workload metadata must define two_qubit_angle_period.")
    if not isinstance(reference, (int, float)) or reference <= 0:
        raise ValueError("two_qubit_reference_angle must be positive.")
    if not isinstance(period, (int, float)) or period <= 0:
        raise ValueError("two_qubit_angle_period must be positive.")
    if not isinstance(parameters, list):
        raise ValueError("Workload metadata must define two_qubit_parameters.")
    if not parameters:
        return None

    indices = [parameter.get("index") for parameter in parameters]
    scales = np.asarray(
        [parameter.get("angle_scale") for parameter in parameters], dtype=float
    )
    counts = np.asarray(
        [parameter.get("gate_count") for parameter in parameters], dtype=float
    )
    if (
        any(not isinstance(index, int) or index < 0 for index in indices)
        or not np.all(np.isfinite(scales))
        or not np.all(np.isfinite(counts))
        or np.any(counts <= 0)
    ):
        raise ValueError("two_qubit_parameters contains invalid metadata.")
    return (
        float(reference),
        float(period),
        indices,
        scales,
        counts,
        list(zip(indices, counts.tolist(), strict=True)),
    )


def _canonical_two_qubit_angles(
    values: np.ndarray, *, period: float, absolute: bool
) -> np.ndarray:
    canonical = (values + period / 2) % period - period / 2
    return np.abs(canonical) if absolute else canonical


def _two_qubit_label(parameter: tuple[int, float]) -> str:
    index, count = parameter
    return f"θ{index} (×{count:g})"


def _pad_series(series: Sequence[np.ndarray]) -> np.ndarray:
    width = max(len(values) for values in series)
    padded = []
    for values in series:
        if len(values) == 0:
            raise ValueError("Result histories must not be empty.")
        tail = np.repeat(values[-1][None, ...], width - len(values), axis=0)
        padded.append(np.concatenate((values, tail), axis=0))
    return np.asarray(padded)


def _pad_parameters(history: Sequence[np.ndarray]) -> np.ndarray:
    if not history:
        raise ValueError("Parameter history must not be empty.")
    return np.asarray(history, dtype=float)


def _exact_energy(exact: ExactResult | float) -> float:
    return exact.energy if isinstance(exact, ExactResult) else float(exact)
