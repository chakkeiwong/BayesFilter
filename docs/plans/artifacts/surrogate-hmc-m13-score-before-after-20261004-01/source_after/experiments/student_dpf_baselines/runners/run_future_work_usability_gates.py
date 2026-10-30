"""Run bounded usability gates for student DPF future-work surfaces.

This runner is intentionally conservative.  It records contract/artifact
readiness first, then executes only small CPU-local probes that can be run
without vendored edits, production imports, training, notebooks, HMC, network,
or GPU requirements.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import inspect
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable

import numpy as np

from experiments.student_dpf_baselines.adapters.advanced_particle_filter_adapter import (
    SOURCE_COMMIT as ADVANCED_COMMIT,
    VENDOR_ROOT as ADVANCED_VENDOR_ROOT,
)
from experiments.student_dpf_baselines.adapters.common import prepend_sys_path, write_json
from experiments.student_dpf_baselines.adapters.mlcoe_adapter import (
    SNAPSHOT_ROOT as MLCOE_ROOT,
    SOURCE_COMMIT as MLCOE_COMMIT,
)
from experiments.student_dpf_baselines.fixtures.nonlinear_fixtures import (
    make_nonlinear_fixture,
)
from experiments.student_dpf_baselines.runners.run_nonlinear_reference_panel import (
    _make_advanced_model,
    _trajectory_metrics,
)


DATE = "2026-05-15"
SCHEMA_VERSION = "student_dpf_future_work_usability_gates.v1"
COMMAND = (
    "python -m "
    "experiments.student_dpf_baselines.runners.run_future_work_usability_gates"
)
OUTPUT_PATH = Path(
    "experiments/student_dpf_baselines/reports/outputs/"
    "future_work_usability_gates_2026-05-15.json"
)
SUMMARY_PATH = Path(
    "experiments/student_dpf_baselines/reports/outputs/"
    "future_work_usability_gates_summary_2026-05-15.json"
)
REPORT_PATH = Path(
    "experiments/student_dpf_baselines/reports/"
    "student-dpf-baseline-future-work-usability-gates-result-2026-05-15.md"
)

SEED = 20260515
RUNTIME_WARNING_SECONDS = 10.0
ARTIFACT_SIZE_CAP_BYTES = 1_000_000


@dataclass(frozen=True, slots=True)
class PlannedProbe:
    gate_phase: str
    gate_name: str
    family: str
    probe_id: str
    implementation_name: str
    source_commit: str
    planned_probe: str
    requires_contract: bool
    requires_artifact: bool = False
    artifact_path: str | None = None
    neural_training_status: str = "not_neural"
    command: str = COMMAND


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        _validate_existing_outputs()
        return

    planned = _planned_probes()
    records = []
    contract_records: dict[str, dict[str, Any]] = {}
    for probe in planned:
        record = _contract_record(probe)
        records.append(record)
        contract_records[probe.probe_id] = record

    runners: dict[str, Callable[[PlannedProbe, dict[str, Any]], dict[str, Any]]] = {
        "advanced_soft_resampler": _run_advanced_soft_resampler,
        "advanced_sinkhorn_resampler": _run_advanced_sinkhorn_resampler,
        "mlcoe_soft_resampler": _run_mlcoe_soft_resampler,
        "mlcoe_sinkhorn_resampler": _run_mlcoe_sinkhorn_resampler,
        "advanced_amortized_ot_resampler": _run_advanced_amortized_ot,
        "mlcoe_transformer_resampler": _run_mlcoe_transformer_resampler,
        "advanced_stochastic_pff_flow": _run_advanced_stochastic_pff_flow,
        "advanced_stochastic_pfpf": _run_advanced_stochastic_pfpf,
        "mlcoe_stochastic_flow": _run_blocked_contract_only,
        "advanced_tf_dpf_soft": _run_advanced_tf_dpf_soft,
        "advanced_tf_dpf_sinkhorn": _run_advanced_tf_dpf_sinkhorn,
        "advanced_tf_dpf_amortized": _run_advanced_tf_dpf_amortized,
        "mlcoe_dpf_soft": _run_mlcoe_dpf_soft,
        "mlcoe_dpf_sinkhorn": _run_mlcoe_dpf_sinkhorn,
        "mlcoe_dpfpf": _run_blocked_contract_only,
    }

    for probe in planned:
        contract = contract_records[probe.probe_id]
        if probe.gate_phase == "FW1":
            continue
        records.append(runners[probe.probe_id](probe, contract))

    summary = _summarize(planned, records)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "date": DATE,
        "command": COMMAND,
        "working_directory": str(Path.cwd()),
        "environment": _environment_record(),
        "planned_probes": [_probe_to_dict(probe) for probe in planned],
        "records": records,
        "summary": summary,
    }
    _write_outputs_with_stable_artifact_size(payload)
    _validate_payload(payload)


def _planned_probes() -> list[PlannedProbe]:
    advanced_ckpt = (
        ADVANCED_VENDOR_ROOT
        / "advanced_particle_filter"
        / "dpf_pretrained"
        / "mgn_ot_operator"
        / "checkpoints_option_b"
    )
    return [
        PlannedProbe(
            "FW2",
            "non_neural_differentiable_resampling",
            "differentiable_resampling",
            "advanced_soft_resampler",
            "advanced_particle_filter",
            ADVANCED_COMMIT,
            "advanced tf_utils.soft_resampler.soft_resample",
            requires_contract=False,
        ),
        PlannedProbe(
            "FW2",
            "non_neural_differentiable_resampling",
            "differentiable_resampling",
            "advanced_sinkhorn_resampler",
            "advanced_particle_filter",
            ADVANCED_COMMIT,
            "advanced tf_utils.sinkhorn.sinkhorn_resample",
            requires_contract=False,
        ),
        PlannedProbe(
            "FW2",
            "non_neural_differentiable_resampling",
            "differentiable_resampling",
            "mlcoe_soft_resampler",
            "2026MLCOE",
            MLCOE_COMMIT,
            "MLCOE SoftResampler",
            requires_contract=False,
        ),
        PlannedProbe(
            "FW2",
            "non_neural_differentiable_resampling",
            "differentiable_resampling",
            "mlcoe_sinkhorn_resampler",
            "2026MLCOE",
            MLCOE_COMMIT,
            "MLCOE SinkhornResampler",
            requires_contract=False,
        ),
        PlannedProbe(
            "FW3",
            "neural_ot_and_neural_resampling",
            "neural_ot",
            "advanced_amortized_ot_resampler",
            "advanced_particle_filter",
            ADVANCED_COMMIT,
            "advanced AmortizedOTResampler with bundled checkpoint",
            requires_contract=False,
            requires_artifact=True,
            artifact_path=str(advanced_ckpt),
            neural_training_status="checkpoint_dependent",
        ),
        PlannedProbe(
            "FW3",
            "neural_ot_and_neural_resampling",
            "neural_resampling",
            "mlcoe_transformer_resampler",
            "2026MLCOE",
            MLCOE_COMMIT,
            "MLCOE TransformerResampler untrained API smoke",
            requires_contract=False,
            neural_training_status="untrained_api_only",
        ),
        PlannedProbe(
            "FW4",
            "stochastic_flow_reduced",
            "stochastic_flow",
            "advanced_stochastic_pff_flow",
            "advanced_particle_filter",
            ADVANCED_COMMIT,
            "advanced StochasticPFFlow reduced fixture",
            requires_contract=True,
        ),
        PlannedProbe(
            "FW4",
            "stochastic_flow_reduced",
            "stochastic_flow",
            "advanced_stochastic_pfpf",
            "advanced_particle_filter",
            ADVANCED_COMMIT,
            "advanced StochasticPFParticleFilter reduced fixture",
            requires_contract=True,
        ),
        PlannedProbe(
            "FW4",
            "stochastic_flow_reduced",
            "stochastic_flow",
            "mlcoe_stochastic_flow",
            "2026MLCOE",
            MLCOE_COMMIT,
            "MLCOE stochastic-flow surfaces",
            requires_contract=True,
        ),
        PlannedProbe(
            "FW5",
            "dpf_dpfpf_contract_conditional",
            "dpf",
            "advanced_tf_dpf_soft",
            "advanced_particle_filter",
            ADVANCED_COMMIT,
            "advanced TFDifferentiableParticleFilter soft",
            requires_contract=True,
        ),
        PlannedProbe(
            "FW5",
            "dpf_dpfpf_contract_conditional",
            "dpf",
            "advanced_tf_dpf_sinkhorn",
            "advanced_particle_filter",
            ADVANCED_COMMIT,
            "advanced TFDifferentiableParticleFilter sinkhorn",
            requires_contract=True,
        ),
        PlannedProbe(
            "FW5",
            "dpf_dpfpf_contract_conditional",
            "dpf",
            "advanced_tf_dpf_amortized",
            "advanced_particle_filter",
            ADVANCED_COMMIT,
            "advanced TFDifferentiableParticleFilter amortized",
            requires_contract=True,
            requires_artifact=True,
            artifact_path=str(advanced_ckpt),
            neural_training_status="checkpoint_dependent",
        ),
        PlannedProbe(
            "FW5",
            "dpf_dpfpf_contract_conditional",
            "dpf",
            "mlcoe_dpf_soft",
            "2026MLCOE",
            MLCOE_COMMIT,
            "MLCOE DPF method soft",
            requires_contract=True,
        ),
        PlannedProbe(
            "FW5",
            "dpf_dpfpf_contract_conditional",
            "dpf",
            "mlcoe_dpf_sinkhorn",
            "2026MLCOE",
            MLCOE_COMMIT,
            "MLCOE DPF method sinkhorn",
            requires_contract=True,
        ),
        PlannedProbe(
            "FW5",
            "dpf_dpfpf_contract_conditional",
            "dpfpf",
            "mlcoe_dpfpf",
            "2026MLCOE",
            MLCOE_COMMIT,
            "MLCOE DifferentiablePFPF",
            requires_contract=True,
        ),
    ]


def _base_record(
    probe: PlannedProbe,
    *,
    phase: str | None = None,
    status: str,
    classification: str,
    runtime_seconds: float | None = None,
    blocker_class: str | None = None,
    blocker_message: str | None = None,
    artifact_dependency_status: str = "not_required",
    model_contract_status: str = "not_required",
    contract_satisfiable_without_vendor_edits: str = "not_required",
    execution_attempted: bool = False,
    metrics: dict[str, Any] | None = None,
    assumptions: list[str] | None = None,
    next_decision: str = "not_decided",
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "date": DATE,
        "phase": phase or probe.gate_phase,
        "gate_phase": probe.gate_phase,
        "gate_name": probe.gate_name,
        "family": probe.family,
        "probe_id": probe.probe_id,
        "planned_probe": probe.planned_probe,
        "implementation_name": probe.implementation_name,
        "source_commit": probe.source_commit,
        "command": probe.command,
        "working_directory": str(Path.cwd()),
        "fixture_or_input": None,
        "seed": SEED,
        "runtime_seconds": runtime_seconds,
        "status": status,
        "classification": classification,
        "blocker_class": blocker_class,
        "blocker_message": blocker_message,
        "artifact_dependency_status": artifact_dependency_status,
        "model_contract_status": model_contract_status,
        "contract_satisfiable_without_vendor_edits": (
            contract_satisfiable_without_vendor_edits
        ),
        "execution_attempted": execution_attempted,
        "prohibited_actions_avoided": [
            "vendored_edits",
            "production_edits",
            "monograph_edits",
            "network",
            "gpu_required",
            "training",
            "hmc",
            "notebook_conversion",
        ],
        "metrics": metrics or {},
        "assumptions": assumptions or [],
        "provenance": {
            "date": DATE,
            "lane": "student_dpf_experimental_baseline",
            "comparison_only": True,
        },
        "next_decision": next_decision,
    }


def _contract_record(probe: PlannedProbe) -> dict[str, Any]:
    artifact_status = "not_required"
    if probe.requires_artifact:
        artifact = Path(probe.artifact_path or "")
        artifact_status = "present" if artifact.exists() else "missing"
    missing_fields: list[str] = []
    verdict = "yes"
    model_status = "satisfiable_without_vendor_edits"
    if probe.probe_id in {"mlcoe_stochastic_flow", "mlcoe_dpfpf"}:
        verdict = "no"
        model_status = "missing_adapter_contract"
        missing_fields = [
            "complete model object with exact flow/dPF field semantics",
            "validated covariance and observation contract",
        ]
    if probe.requires_artifact and artifact_status == "missing":
        verdict = "no"
        missing_fields.append(f"missing artifact path: {probe.artifact_path}")
    return _base_record(
        probe,
        phase="FW1",
        status="ok" if verdict == "yes" else "blocked",
        classification="api_smoke_only" if verdict == "yes" else "blocked_missing_assumption",
        artifact_dependency_status=artifact_status,
        model_contract_status=model_status,
        contract_satisfiable_without_vendor_edits=verdict,
        execution_attempted=False,
        blocker_class=None if verdict == "yes" else "blocked_missing_assumption",
        blocker_message=None if verdict == "yes" else "; ".join(missing_fields),
        metrics=_contract_metrics(probe),
        assumptions=[
            "contract audit only; no filter or resampler execution",
            "student source remains comparison-only",
        ],
        next_decision="execution_probe_allowed" if verdict == "yes" else "blocked_without_execution",
    )


def _contract_metrics(probe: PlannedProbe) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "requires_contract": probe.requires_contract,
        "requires_artifact": probe.requires_artifact,
        "artifact_path": probe.artifact_path,
        "neural_training_status": probe.neural_training_status,
    }
    try:
        with _vendor_context(probe.implementation_name):
            obj = _resolve_probe_object(probe.probe_id)
        metrics["object_signature"] = _signature(obj)
        call = getattr(obj, "__call__", None)
        if call is not None:
            metrics["call_signature"] = _signature(call)
        for method_name in ("filter", "filter_step", "step", "resample"):
            method = getattr(obj, method_name, None)
            if method is not None:
                metrics[f"{method_name}_signature"] = _signature(method)
    except Exception as exc:
        metrics["contract_probe_error"] = f"{type(exc).__name__}: {exc}"
    return metrics


def _resolve_probe_object(probe_id: str) -> Any:
    if probe_id == "advanced_soft_resampler":
        from advanced_particle_filter.tf_utils.soft_resampler import soft_resample

        return soft_resample
    if probe_id == "advanced_sinkhorn_resampler":
        from advanced_particle_filter.tf_utils.sinkhorn import sinkhorn_resample

        return sinkhorn_resample
    if probe_id in {"advanced_amortized_ot_resampler", "advanced_tf_dpf_amortized"}:
        from advanced_particle_filter.tf_utils.amortized_resampler import (
            AmortizedOTResampler,
        )

        return AmortizedOTResampler
    if probe_id in {"advanced_stochastic_pff_flow"}:
        from advanced_particle_filter.filters import StochasticPFFlow

        return StochasticPFFlow
    if probe_id == "advanced_stochastic_pfpf":
        from advanced_particle_filter.filters import StochasticPFParticleFilter

        return StochasticPFParticleFilter
    if probe_id in {
        "advanced_tf_dpf_soft",
        "advanced_tf_dpf_sinkhorn",
    }:
        from advanced_particle_filter.tf_filters.differentiable_particle import (
            TFDifferentiableParticleFilter,
        )

        return TFDifferentiableParticleFilter
    if probe_id == "mlcoe_soft_resampler":
        from src.filters.resampling.soft import SoftResampler

        return SoftResampler
    if probe_id == "mlcoe_sinkhorn_resampler":
        from src.filters.resampling.optimal_transport import SinkhornResampler

        return SinkhornResampler
    if probe_id == "mlcoe_transformer_resampler":
        from src.filters.resampling.transformer import TransformerResampler

        return TransformerResampler
    if probe_id in {"mlcoe_dpf_soft", "mlcoe_dpf_sinkhorn"}:
        from src.filters.DPF import DPF

        return DPF
    if probe_id == "mlcoe_dpfpf":
        from src.filters.dpfpf import DifferentiablePFPF

        return DifferentiablePFPF
    if probe_id == "mlcoe_stochastic_flow":
        from src.filters.stochastic_flow_filters import SPF

        return SPF
    raise KeyError(probe_id)


def _signature(obj: Any) -> str:
    try:
        return str(inspect.signature(obj))
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


def _vendor_context(implementation_name: str):
    if implementation_name == "advanced_particle_filter":
        return prepend_sys_path(ADVANCED_VENDOR_ROOT)
    return prepend_sys_path(MLCOE_ROOT)


def _tiny_cloud_tf(dtype: Any | None = None) -> tuple[Any, Any]:
    import tensorflow as tf  # type: ignore

    dtype = dtype or tf.float64
    particles_np, weights_np = _tiny_cloud_np()
    particles = tf.constant(particles_np[None, :, :], dtype=dtype)
    log_w = tf.math.log(tf.constant(weights_np[None, :], dtype=dtype))
    return particles, log_w


def _tiny_cloud_np() -> tuple[np.ndarray, np.ndarray]:
    particles = np.array(
        [
            [-1.0, -0.5],
            [-0.7, 0.2],
            [-0.2, 0.9],
            [0.0, -0.1],
            [0.35, 0.5],
            [0.8, -0.3],
            [1.2, 0.75],
            [1.5, -0.8],
        ],
        dtype=float,
    )
    weights = np.array([0.04, 0.07, 0.11, 0.16, 0.22, 0.18, 0.14, 0.08], dtype=float)
    weights /= weights.sum()
    return particles, weights


def _run_with_guard(
    probe: PlannedProbe,
    contract: dict[str, Any],
    fn: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    if contract["contract_satisfiable_without_vendor_edits"] != "yes":
        return _base_record(
            probe,
            status="blocked",
            classification=contract["classification"],
            blocker_class=contract["blocker_class"],
            blocker_message=contract["blocker_message"],
            artifact_dependency_status=contract["artifact_dependency_status"],
            model_contract_status=contract["model_contract_status"],
            contract_satisfiable_without_vendor_edits="no",
            execution_attempted=False,
            assumptions=["execution forbidden by FW1 contract audit"],
            next_decision="defer_until_artifacts_or_assumptions",
        )
    start = time.perf_counter()
    try:
        record = fn()
        if record.get("runtime_seconds") is None:
            record["runtime_seconds"] = time.perf_counter() - start
        return record
    except Exception as exc:
        return _base_record(
            probe,
            status="blocked",
            classification="blocked_environment_drift",
            blocker_class="blocked_environment_drift",
            blocker_message=f"{type(exc).__name__}: {exc}",
            artifact_dependency_status=contract["artifact_dependency_status"],
            model_contract_status=contract["model_contract_status"],
            contract_satisfiable_without_vendor_edits="yes",
            execution_attempted=True,
            runtime_seconds=time.perf_counter() - start,
            assumptions=["exception converted to structured blocker"],
            next_decision="debug_gate_next",
        )


def _run_advanced_soft_resampler(
    probe: PlannedProbe, contract: dict[str, Any]
) -> dict[str, Any]:
    def run() -> dict[str, Any]:
        import tensorflow as tf  # type: ignore

        start = time.perf_counter()
        with prepend_sys_path(ADVANCED_VENDOR_ROOT):
            from advanced_particle_filter.tf_utils.soft_resampler import soft_resample

            particles, log_w = _tiny_cloud_tf()
            rng = tf.random.Generator.from_seed(SEED)
            with tf.GradientTape() as tape:
                tape.watch(particles)
                out, new_log_w = soft_resample(
                    particles, log_w, tf.constant(0.5, dtype=tf.float64), rng
                )
                loss = tf.reduce_sum(out)
            grad = tape.gradient(loss, particles)
        return _success_record(
            probe,
            contract,
            runtime_seconds=time.perf_counter() - start,
            classification="usable_component_only",
            metrics=_tf_resampler_metrics(out, new_log_w, grad),
            next_decision="component_spec_next",
        )

    return _run_with_guard(probe, contract, run)


def _run_advanced_sinkhorn_resampler(
    probe: PlannedProbe, contract: dict[str, Any]
) -> dict[str, Any]:
    def run() -> dict[str, Any]:
        import tensorflow as tf  # type: ignore

        start = time.perf_counter()
        with prepend_sys_path(ADVANCED_VENDOR_ROOT):
            from advanced_particle_filter.tf_utils.sinkhorn import sinkhorn_resample

            particles, log_w = _tiny_cloud_tf()
            with tf.GradientTape() as tape:
                tape.watch(particles)
                out, new_log_w = sinkhorn_resample(
                    particles, log_w, tf.constant(0.2, dtype=tf.float64), 5
                )
                loss = tf.reduce_sum(out)
            grad = tape.gradient(loss, particles)
        return _success_record(
            probe,
            contract,
            runtime_seconds=time.perf_counter() - start,
            classification="usable_component_only",
            metrics=_tf_resampler_metrics(out, new_log_w, grad),
            next_decision="component_spec_next",
        )

    return _run_with_guard(probe, contract, run)


def _run_mlcoe_soft_resampler(
    probe: PlannedProbe, contract: dict[str, Any]
) -> dict[str, Any]:
    def run() -> dict[str, Any]:
        import tensorflow as tf  # type: ignore

        start = time.perf_counter()
        with prepend_sys_path(MLCOE_ROOT):
            from src.filters.classical import DTYPE
            from src.filters.resampling.soft import SoftResampler

            particles_np, weights_np = _tiny_cloud_np()
            particles = tf.constant(particles_np, dtype=DTYPE)
            weights = tf.constant(weights_np, dtype=DTYPE)
            hidden = [tf.zeros((particles_np.shape[0], 1), dtype=DTYPE)]
            with tf.GradientTape() as tape:
                tape.watch(particles)
                out, _, new_log_w = SoftResampler(particles_np.shape[0], alpha=0.5)(
                    particles, hidden, weights
                )
                loss = tf.reduce_sum(out)
            grad = tape.gradient(loss, particles)
        return _success_record(
            probe,
            contract,
            runtime_seconds=time.perf_counter() - start,
            classification="usable_component_only",
            metrics=_tf_resampler_metrics(out, new_log_w, grad),
            next_decision="component_spec_next",
        )

    return _run_with_guard(probe, contract, run)


def _run_mlcoe_sinkhorn_resampler(
    probe: PlannedProbe, contract: dict[str, Any]
) -> dict[str, Any]:
    def run() -> dict[str, Any]:
        import tensorflow as tf  # type: ignore

        start = time.perf_counter()
        with prepend_sys_path(MLCOE_ROOT):
            from src.filters.classical import DTYPE
            from src.filters.resampling.optimal_transport import SinkhornResampler

            particles_np, weights_np = _tiny_cloud_np()
            particles = tf.constant(particles_np, dtype=DTYPE)
            log_w = tf.math.log(tf.constant(weights_np, dtype=DTYPE))
            with tf.GradientTape() as tape:
                tape.watch(particles)
                out, _ = SinkhornResampler(epsilon=0.2, n_iter=5).resample(
                    particles, log_w
                )
                loss = tf.reduce_sum(out)
            grad = tape.gradient(loss, particles)
        return _success_record(
            probe,
            contract,
            runtime_seconds=time.perf_counter() - start,
            classification="usable_component_only",
            metrics=_tf_resampler_metrics(out, None, grad),
            next_decision="component_spec_next",
        )

    return _run_with_guard(probe, contract, run)


def _run_advanced_amortized_ot(
    probe: PlannedProbe, contract: dict[str, Any]
) -> dict[str, Any]:
    def run() -> dict[str, Any]:
        import tensorflow as tf  # type: ignore

        start = time.perf_counter()
        with prepend_sys_path(ADVANCED_VENDOR_ROOT):
            from advanced_particle_filter.tf_utils.amortized_resampler import (
                AmortizedOTResampler,
            )

            particles, log_w = _tiny_cloud_tf()
            with tf.GradientTape() as tape:
                tape.watch(particles)
                out, new_log_w = AmortizedOTResampler(d=2, N=8, dtype=tf.float64)(
                    particles, log_w
                )
                loss = tf.reduce_sum(out)
            grad = tape.gradient(loss, particles)
        return _success_record(
            probe,
            contract,
            runtime_seconds=time.perf_counter() - start,
            classification="usable_component_only",
            metrics=_tf_resampler_metrics(out, new_log_w, grad),
            next_decision="component_spec_next",
        )

    return _run_with_guard(probe, contract, run)


def _run_mlcoe_transformer_resampler(
    probe: PlannedProbe, contract: dict[str, Any]
) -> dict[str, Any]:
    def run() -> dict[str, Any]:
        import tensorflow as tf  # type: ignore

        start = time.perf_counter()
        with prepend_sys_path(MLCOE_ROOT):
            from src.filters.classical import DTYPE
            from src.filters.resampling.transformer import TransformerResampler

            particles_np, weights_np = _tiny_cloud_np()
            particles = tf.constant(particles_np, dtype=DTYPE)
            weights = tf.constant(weights_np, dtype=DTYPE)
            hidden = [tf.zeros((particles_np.shape[0], 1), dtype=DTYPE)]
            out, _, new_log_w = TransformerResampler(
                particles_np.shape[0], latent_dim=16, num_heads=2
            )(particles, hidden, weights)
        return _success_record(
            probe,
            contract,
            runtime_seconds=time.perf_counter() - start,
            classification="api_smoke_only",
            metrics=_tf_resampler_metrics(out, new_log_w, None),
            assumptions=["untrained neural resampler; API smoke only"],
            next_decision="defer_until_artifacts_or_assumptions",
        )

    return _run_with_guard(probe, contract, run)


def _run_advanced_stochastic_pff_flow(
    probe: PlannedProbe, contract: dict[str, Any]
) -> dict[str, Any]:
    def run() -> dict[str, Any]:
        start = time.perf_counter()
        fixture = _reduced_fixture()
        with prepend_sys_path(ADVANCED_VENDOR_ROOT):
            from advanced_particle_filter.filters import StochasticPFFlow

            model = _make_advanced_model(fixture)
            filt = StochasticPFFlow(
                n_particles=16,
                n_flow_steps=3,
                Q_flow_mode="fixed",
                Q_flow_fixed=np.eye(fixture.state_dim) * 1e-8,
                beta_schedule="linear",
                deterministic_dynamics=False,
                integration_method="euler",
                seed=SEED,
            )
            result = filt.filter(
                model,
                fixture.observations,
                return_diagnostics=True,
                rng=np.random.default_rng(SEED),
            )
        metrics = _trajectory_metrics(result.means, fixture)
        metrics.update(
            {
                "finite_means": bool(np.all(np.isfinite(result.means))),
                "finite_covariances": bool(np.all(np.isfinite(result.covariances))),
                "average_ess": float(np.mean(result.ess)),
            }
        )
        return _success_record(
            probe,
            contract,
            runtime_seconds=time.perf_counter() - start,
            classification="usable_for_clean_room_spec",
            metrics=metrics,
            fixture_or_input=fixture.name,
            assumptions=["linear beta schedule", "fixed tiny diffusion", "horizon 4"],
            next_decision="clean_room_spec_next",
        )

    return _run_with_guard(probe, contract, run)


def _run_advanced_stochastic_pfpf(
    probe: PlannedProbe, contract: dict[str, Any]
) -> dict[str, Any]:
    def run() -> dict[str, Any]:
        start = time.perf_counter()
        fixture = _reduced_fixture()
        with prepend_sys_path(ADVANCED_VENDOR_ROOT):
            from advanced_particle_filter.filters import StochasticPFParticleFilter

            model = _make_advanced_model(fixture)
            filt = StochasticPFParticleFilter(
                n_particles=16,
                n_flow_steps=3,
                Q_flow_mode="fixed",
                Q_flow_fixed=np.eye(fixture.state_dim) * 1e-8,
                beta_schedule="linear",
                deterministic_dynamics=False,
                integration_method="euler",
                resample_criterion="ess",
                ess_threshold=0.5,
                seed=SEED,
            )
            result = filt.filter(
                model,
                fixture.observations,
                return_diagnostics=True,
                rng=np.random.default_rng(SEED),
            )
        metrics = _trajectory_metrics(result.means, fixture)
        ess = np.asarray(result.ess, dtype=float)
        metrics.update(
            {
                "finite_means": bool(np.all(np.isfinite(result.means))),
                "finite_covariances": bool(np.all(np.isfinite(result.covariances))),
                "average_ess": float(np.mean(ess)),
                "min_ess": float(np.min(ess)),
                "resampling_count": (
                    None
                    if result.resampled is None
                    else int(np.sum(np.asarray(result.resampled, dtype=bool)))
                ),
            }
        )
        return _success_record(
            probe,
            contract,
            runtime_seconds=time.perf_counter() - start,
            classification="usable_for_clean_room_spec",
            metrics=metrics,
            fixture_or_input=fixture.name,
            assumptions=["linear beta schedule", "fixed tiny diffusion", "horizon 4"],
            next_decision="clean_room_spec_next",
        )

    return _run_with_guard(probe, contract, run)


def _run_advanced_tf_dpf_soft(
    probe: PlannedProbe, contract: dict[str, Any]
) -> dict[str, Any]:
    return _run_advanced_tf_dpf(probe, contract, resampler="soft")


def _run_advanced_tf_dpf_sinkhorn(
    probe: PlannedProbe, contract: dict[str, Any]
) -> dict[str, Any]:
    return _run_advanced_tf_dpf(probe, contract, resampler="sinkhorn")


def _run_advanced_tf_dpf_amortized(
    probe: PlannedProbe, contract: dict[str, Any]
) -> dict[str, Any]:
    return _run_advanced_tf_dpf(probe, contract, resampler="amortized")


def _run_advanced_tf_dpf(
    probe: PlannedProbe, contract: dict[str, Any], *, resampler: str
) -> dict[str, Any]:
    def run() -> dict[str, Any]:
        import tensorflow as tf  # type: ignore

        start = time.perf_counter()
        with prepend_sys_path(ADVANCED_VENDOR_ROOT):
            from advanced_particle_filter.tf_filters.differentiable_particle import (
                TFDifferentiableParticleFilter,
            )
            from advanced_particle_filter.tf_models.svssm import SVSSMParams

            dtype = tf.float64
            params = SVSSMParams(
                mu=tf.constant([[0.0, 0.0]], dtype=dtype),
                Phi=tf.constant([[[0.9, 0.0], [0.0, 0.85]]], dtype=dtype),
                Sigma_eta_chol=tf.constant([[[0.1, 0.0], [0.0, 0.08]]], dtype=dtype),
            )
            observations = tf.constant([[0.1, -0.2], [0.05, 0.15]], dtype=dtype)
            kwargs: dict[str, Any] = {"n_particles": 8, "resampler": resampler, "dtype": dtype}
            if resampler == "sinkhorn":
                kwargs.update({"epsilon": 0.2, "sinkhorn_iters": 5})
            if resampler == "amortized":
                kwargs.update({"amortized_d": 2, "amortized_eps": 0.5})
            dpf = TFDifferentiableParticleFilter(**kwargs)
            rng = tf.random.Generator.from_seed(SEED)
            with tf.GradientTape() as tape:
                tape.watch(params.mu)
                result = dpf.filter(params, observations, rng)
                loss = tf.reduce_sum(result.log_evidence)
            grad = tape.gradient(loss, params.mu)
        metrics = {
            "log_evidence": _to_float_list(result.log_evidence),
            "finite_log_evidence": bool(
                np.all(np.isfinite(np.asarray(result.log_evidence.numpy())))
            ),
            "finite_particles": bool(
                np.all(np.isfinite(np.asarray(result.final_particles.numpy())))
            ),
            "finite_log_weights": bool(
                np.all(np.isfinite(np.asarray(result.final_log_w.numpy())))
            ),
            "gradient_finite": _gradient_finite(grad),
            "resampler": resampler,
        }
        return _success_record(
            probe,
            contract,
            runtime_seconds=time.perf_counter() - start,
            classification="usable_for_clean_room_spec",
            metrics=metrics,
            fixture_or_input="tiny_svssm_T2_B1_N8",
            assumptions=["tiny SVSSM smoke only", "no HMC"],
            next_decision="clean_room_spec_next",
        )

    return _run_with_guard(probe, contract, run)


def _run_mlcoe_dpf_soft(probe: PlannedProbe, contract: dict[str, Any]) -> dict[str, Any]:
    return _run_mlcoe_dpf(probe, contract, method="soft")


def _run_mlcoe_dpf_sinkhorn(
    probe: PlannedProbe, contract: dict[str, Any]
) -> dict[str, Any]:
    return _run_mlcoe_dpf(probe, contract, method="sinkhorn")


def _run_mlcoe_dpf(
    probe: PlannedProbe, contract: dict[str, Any], *, method: str
) -> dict[str, Any]:
    def run() -> dict[str, Any]:
        import tensorflow as tf  # type: ignore

        start = time.perf_counter()
        with prepend_sys_path(MLCOE_ROOT):
            from src.filters.DPF import DPF
            from src.filters.classical import DTYPE

            class Transition(tf.Module):
                def __call__(self, x_prev: Any, h_prev: Any, noise: Any) -> tuple[Any, Any]:
                    return 0.85 * x_prev + 0.1 * noise, h_prev

            class Observation(tf.Module):
                def __call__(self, x: Any, y: Any) -> Any:
                    return -0.5 * tf.reduce_sum(tf.square(y - x), axis=-1) / tf.cast(
                        0.04, DTYPE
                    )

            dpf = DPF(Transition(), Observation(), num_particles=8)
            x = tf.random.stateless_normal((8, 2), seed=[SEED % 1000, 17], dtype=DTYPE)
            h = [tf.zeros((8, 1), dtype=DTYPE)]
            log_w = tf.fill((8,), -tf.math.log(tf.cast(8, DTYPE)))
            y = tf.constant([0.1, -0.2], dtype=DTYPE)
            with tf.GradientTape() as tape:
                tape.watch(x)
                out_x, out_h, out_log_w = dpf.filter_step(
                    x, h, log_w, y, method=method, alpha=0.5
                )
                loss = tf.reduce_sum(out_x)
            grad = tape.gradient(loss, x)
        metrics = {
            "finite_particles": bool(np.all(np.isfinite(np.asarray(out_x.numpy())))),
            "finite_log_weights": bool(
                np.all(np.isfinite(np.asarray(out_log_w.numpy())))
            ),
            "output_shape": list(out_x.shape),
            "hidden_shapes": [list(item.shape) for item in out_h],
            "gradient_finite": _gradient_finite(grad),
            "method": method,
        }
        return _success_record(
            probe,
            contract,
            runtime_seconds=time.perf_counter() - start,
            classification="usable_for_clean_room_spec",
            metrics=metrics,
            fixture_or_input="tiny_synthetic_single_step_N8_D2",
            assumptions=["single filter_step smoke", "no training", "no HMC"],
            next_decision="clean_room_spec_next",
        )

    return _run_with_guard(probe, contract, run)


def _run_blocked_contract_only(
    probe: PlannedProbe, contract: dict[str, Any]
) -> dict[str, Any]:
    return _run_with_guard(
        probe,
        contract,
        lambda: _base_record(
            probe,
            status="blocked",
            classification="blocked_missing_assumption",
            blocker_class="blocked_missing_assumption",
            blocker_message="contract-only path unexpectedly reached execution",
            contract_satisfiable_without_vendor_edits="no",
            execution_attempted=False,
            next_decision="defer_until_artifacts_or_assumptions",
        ),
    )


def _success_record(
    probe: PlannedProbe,
    contract: dict[str, Any],
    *,
    runtime_seconds: float,
    classification: str,
    metrics: dict[str, Any],
    next_decision: str,
    fixture_or_input: str | None = "tiny_weighted_cloud_B1_N8_D2",
    assumptions: list[str] | None = None,
) -> dict[str, Any]:
    record = _base_record(
        probe,
        status="ok",
        classification=classification,
        runtime_seconds=runtime_seconds,
        artifact_dependency_status=contract["artifact_dependency_status"],
        model_contract_status=contract["model_contract_status"],
        contract_satisfiable_without_vendor_edits="yes",
        execution_attempted=True,
        metrics=metrics,
        assumptions=assumptions or ["bounded smoke only", "comparison-only evidence"],
        next_decision=next_decision,
    )
    record["fixture_or_input"] = fixture_or_input
    if runtime_seconds > RUNTIME_WARNING_SECONDS:
        record["metrics"]["runtime_warning"] = True
    return record


def _tf_resampler_metrics(out: Any, new_log_w: Any | None, grad: Any | None) -> dict[str, Any]:
    out_np = np.asarray(out.numpy(), dtype=float)
    metrics = {
        "output_shape": list(out_np.shape),
        "finite_outputs": bool(np.all(np.isfinite(out_np))),
        "output_mean": np.mean(out_np.reshape(-1, out_np.shape[-1]), axis=0).tolist(),
        "gradient_finite": _gradient_finite(grad),
    }
    if new_log_w is not None:
        log_w_np = np.asarray(new_log_w.numpy(), dtype=float)
        metrics.update(
            {
                "log_weight_shape": list(log_w_np.shape),
                "finite_log_weights": bool(np.all(np.isfinite(log_w_np))),
                "weight_sum": float(np.sum(np.exp(log_w_np))),
            }
        )
    return metrics


def _gradient_finite(grad: Any | None) -> bool:
    if grad is None:
        return False
    if isinstance(grad, (list, tuple)):
        return all(_gradient_finite(item) for item in grad if item is not None)
    if hasattr(grad, "numpy"):
        arr = np.asarray(grad.numpy(), dtype=float)
    else:
        arr = np.asarray(grad, dtype=float)
    return bool(arr.size and np.all(np.isfinite(arr)))


def _to_float_list(value: Any) -> list[float]:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return [float(item) for item in np.asarray(value).reshape(-1)]


def _reduced_fixture() -> Any:
    from dataclasses import replace

    fixture = make_nonlinear_fixture("range_bearing_gaussian_moderate")
    horizon = 4
    return replace(
        fixture,
        name=f"{fixture.name}_future_work_h{horizon}",
        states=fixture.states[: horizon + 1].copy(),
        observations=fixture.observations[:horizon].copy(),
    )


def _environment_record() -> dict[str, Any]:
    record: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        import tensorflow as tf  # type: ignore

        record["tensorflow"] = tf.__version__
        record["tensorflow_devices"] = [
            {"name": device.name, "type": device.device_type}
            for device in tf.config.list_physical_devices()
        ]
    except Exception as exc:
        record["tensorflow_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import tensorflow_probability as tfp  # type: ignore

        record["tensorflow_probability"] = tfp.__version__
    except Exception as exc:
        record["tensorflow_probability_error"] = f"{type(exc).__name__}: {exc}"
    return record


def _probe_to_dict(probe: PlannedProbe) -> dict[str, Any]:
    return {
        "gate_phase": probe.gate_phase,
        "gate_name": probe.gate_name,
        "family": probe.family,
        "probe_id": probe.probe_id,
        "implementation_name": probe.implementation_name,
        "source_commit": probe.source_commit,
        "planned_probe": probe.planned_probe,
        "requires_contract": probe.requires_contract,
        "requires_artifact": probe.requires_artifact,
        "artifact_path": probe.artifact_path,
        "neural_training_status": probe.neural_training_status,
        "command": probe.command,
    }


def _summarize(planned: list[PlannedProbe], records: list[dict[str, Any]]) -> dict[str, Any]:
    expected_inventory = {
        probe.probe_id: {
            "family": probe.family,
            "gate_phase": probe.gate_phase,
            "implementation_name": probe.implementation_name,
        }
        for probe in planned
    }
    execution_records = [
        record for record in records if record["phase"] != "FW1"
    ]
    observed_inventory = {
        record["probe_id"]: {
            "family": record["family"],
            "gate_phase": record["gate_phase"],
            "implementation_name": record["implementation_name"],
        }
        for record in execution_records
    }
    missing = sorted(set(expected_inventory) - set(observed_inventory))
    status_counts = Counter(record["status"] for record in records)
    classification_counts = Counter(record["classification"] for record in records)
    family_decisions = _family_decisions(execution_records)
    artifact_size_bytes = _artifact_size_bytes()
    final_label = (
        "future_work_usability_gates_complete"
        if not missing and all(record["classification"] for record in records)
        else "future_work_usability_gates_needs_revision"
    )
    if artifact_size_bytes > ARTIFACT_SIZE_CAP_BYTES:
        final_label = "future_work_usability_gates_blocked_artifact_size"
    return {
        "records_expected": len(planned),
        "execution_records_observed": len(execution_records),
        "total_records_observed": len(records),
        "status_counts": dict(status_counts),
        "classification_counts": dict(classification_counts),
        "planned_probe_inventory": expected_inventory,
        "observed_record_inventory": observed_inventory,
        "missing_planned_probes": missing,
        "family_decisions": family_decisions,
        "artifact_size_bytes": artifact_size_bytes,
        "artifact_size_cap_bytes": ARTIFACT_SIZE_CAP_BYTES,
        "veto_diagnostic_fired": bool(missing or artifact_size_bytes > ARTIFACT_SIZE_CAP_BYTES),
        "final_label": final_label,
    }


def _family_decisions(records: list[dict[str, Any]]) -> dict[str, str]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["family"], []).append(record)
    decisions = {}
    for family, group in sorted(grouped.items()):
        classifications = {record["classification"] for record in group}
        if "usable_for_clean_room_spec" in classifications:
            decisions[family] = "clean_room_spec_next"
        elif "usable_component_only" in classifications:
            decisions[family] = "component_spec_next"
        elif classifications == {"api_smoke_only"}:
            decisions[family] = "defer_until_artifacts_or_assumptions"
        elif any(c.startswith("blocked_") for c in classifications):
            decisions[family] = "debug_gate_next"
        else:
            decisions[family] = "defer_until_artifacts_or_assumptions"
    return decisions


def _artifact_size_bytes() -> int:
    total = 0
    for path in (OUTPUT_PATH, SUMMARY_PATH, REPORT_PATH):
        if path.exists():
            total += path.stat().st_size
    return total


def _write_outputs_with_stable_artifact_size(payload: dict[str, Any]) -> None:
    """Write outputs, then update artifact-size metadata until stable."""

    for _ in range(3):
        write_json(OUTPUT_PATH, payload)
        write_json(SUMMARY_PATH, payload["summary"])
        REPORT_PATH.write_text(_render_report(payload["summary"]), encoding="utf-8")
        size = _artifact_size_bytes()
        payload["summary"]["artifact_size_bytes"] = size
        payload["summary"]["veto_diagnostic_fired"] = bool(
            payload["summary"].get("missing_planned_probes")
            or size > ARTIFACT_SIZE_CAP_BYTES
        )
        payload["summary"]["final_label"] = (
            "future_work_usability_gates_complete"
            if not payload["summary"]["veto_diagnostic_fired"]
            else "future_work_usability_gates_needs_revision"
        )
    write_json(OUTPUT_PATH, payload)
    write_json(SUMMARY_PATH, payload["summary"])
    REPORT_PATH.write_text(_render_report(payload["summary"]), encoding="utf-8")


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Student DPF future-work usability gates result",
        "",
        "## Status",
        "",
        f"Final label: `{summary['final_label']}`.",
        "",
        "This report is student-lane, comparison-only evidence.  It does not",
        "promote student code, the clean-room prototype, or any neural/artifact",
        "surface into production.",
        "",
        "## Counts",
        "",
        f"- planned execution probes: `{summary['records_expected']}`",
        f"- observed execution records: `{summary['execution_records_observed']}`",
        f"- total records including contract audit: `{summary['total_records_observed']}`",
        f"- missing planned probes: `{summary['missing_planned_probes']}`",
        f"- artifact size bytes: `{summary['artifact_size_bytes']}`",
        "",
        "## Status Counts",
        "",
    ]
    for key, value in sorted(summary["status_counts"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Classification Counts", ""])
    for key, value in sorted(summary["classification_counts"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Family Decisions", ""])
    for key, value in sorted(summary["family_decisions"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Execution Records", ""])
    lines.extend(
        [
            "| Probe | Family | Implementation | Status | Classification | Next decision |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    records = _read_records_for_report()
    for record in records:
        if record["phase"] == "FW1":
            continue
        lines.append(
            "| "
            f"`{record['probe_id']}` | "
            f"`{record['family']}` | "
            f"`{record['implementation_name']}` | "
            f"`{record['status']}` | "
            f"`{record['classification']}` | "
            f"`{record['next_decision']}` |"
        )
    blockers = [
        record
        for record in records
        if record["phase"] != "FW1" and record["status"] != "ok"
    ]
    lines.extend(["", "## Blockers", ""])
    if blockers:
        lines.extend(
            [
                "| Probe | Classification | Execution attempted | Blocker |",
                "| --- | --- | ---: | --- |",
            ]
        )
        for record in blockers:
            message = str(record.get("blocker_message") or "").replace("\n", " ")
            if len(message) > 180:
                message = message[:177] + "..."
            lines.append(
                "| "
                f"`{record['probe_id']}` | "
                f"`{record['classification']}` | "
                f"`{record['execution_attempted']}` | "
                f"{message} |"
            )
    else:
        lines.append("No execution blockers were recorded.")
    lines.extend(["", "## Successful Probe Highlights", ""])
    highlights = [
        record
        for record in records
        if record["phase"] != "FW1" and record["status"] == "ok"
    ]
    if highlights:
        lines.extend(
            [
                "| Probe | Runtime seconds | Key finite/gradient evidence |",
                "| --- | ---: | --- |",
            ]
        )
        for record in highlights:
            metrics = record.get("metrics", {})
            evidence = []
            for key in (
                "gradient_finite",
                "finite_outputs",
                "finite_means",
                "finite_particles",
                "finite_log_evidence",
            ):
                if key in metrics:
                    evidence.append(f"{key}={metrics[key]}")
            if "position_rmse" in metrics:
                evidence.append(f"position_rmse={metrics['position_rmse']:.6g}")
            if "average_ess" in metrics:
                evidence.append(f"average_ess={metrics['average_ess']:.6g}")
            lines.append(
                "| "
                f"`{record['probe_id']}` | "
                f"{_fmt_float(record.get('runtime_seconds'))} | "
                f"{'; '.join(evidence)} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Non-neural differentiable resampling and any successful filter-level",
            "smokes can inform later clean-room specifications.  Neural paths remain",
            "artifact- or training-semantics dependent unless classified otherwise in",
            "the machine-readable records.  Blocked paths are evidence of current",
            "readiness limits, not failures of the future research direction.",
            "",
        ]
    )
    return "\n".join(lines)


def _read_records_for_report() -> list[dict[str, Any]]:
    """Read existing records when stabilizing the report, otherwise return empty."""

    if not OUTPUT_PATH.exists():
        return []
    try:
        import json

        payload = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        return list(payload.get("records", []))
    except Exception:
        return []


def _fmt_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6g}"


def _validate_existing_outputs() -> None:
    import json

    if not OUTPUT_PATH.exists() or not SUMMARY_PATH.exists():
        raise SystemExit("future-work usability outputs are missing")
    payload = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    _validate_payload(payload)


def _validate_payload(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise SystemExit("schema_version mismatch")
    planned = {probe["probe_id"] for probe in payload["planned_probes"]}
    planned_by_id = {probe["probe_id"]: probe for probe in payload["planned_probes"]}
    execution_records = [
        record for record in payload["records"] if record["phase"] != "FW1"
    ]
    observed = {record["probe_id"] for record in execution_records}
    missing = sorted(planned - observed)
    if missing:
        raise SystemExit(f"missing execution records: {missing}")
    for record in payload["records"]:
        required = [
            "schema_version",
            "gate_phase",
            "gate_name",
            "planned_probe",
            "contract_satisfiable_without_vendor_edits",
            "execution_attempted",
            "classification",
        ]
        for key in required:
            if key not in record:
                raise SystemExit(f"record {record.get('probe_id')} missing {key}")
        if record["status"] != "ok" and not (
            record.get("blocker_class") and record.get("blocker_message")
        ):
            raise SystemExit(f"non-ok record lacks blocker: {record['probe_id']}")
        if (
            planned_by_id[record["probe_id"]].get("neural_training_status")
            == "untrained_api_only"
            and record["classification"].startswith("usable_")
        ):
            raise SystemExit(
                f"untrained neural path got usable classification: {record['probe_id']}"
            )
    artifact_size = payload["summary"].get("artifact_size_bytes", 0)
    if artifact_size <= 0:
        raise SystemExit(f"artifact size was not recorded: {artifact_size}")
    if artifact_size > ARTIFACT_SIZE_CAP_BYTES:
        raise SystemExit(f"artifact size too large: {artifact_size}")


if __name__ == "__main__":
    main()
