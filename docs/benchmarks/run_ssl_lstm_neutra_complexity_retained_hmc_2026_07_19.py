#!/usr/bin/env python3
"""Q-general immutable retained HMC with sequential diagnostic checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _configure_visibility_before_tensorflow_import() -> str:
    mode = None
    if "--mode" in sys.argv:
        index = sys.argv.index("--mode")
        if index + 1 < len(sys.argv):
            mode = sys.argv[index + 1]
    if mode == "contract-smoke":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return "cpu-hidden-contract-smoke"
    if mode not in {"acquire"} and os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        return str(os.environ["CUDA_VISIBLE_DEVICES"])
    probe = subprocess.run(
        ("nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"),
        check=True,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    available = {
        int(line.strip())
        for line in probe.stdout.splitlines()
        if line.strip().isdigit()
    }
    selected = "1" if 1 in available else ("0" if 0 in available else "")
    if not selected:
        raise RuntimeError("no physical GPU 1 or GPU 0 is available")
    os.environ["CUDA_VISIBLE_DEVICES"] = selected
    return selected


SELECTED_GPU = _configure_visibility_before_tensorflow_import()

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter  # noqa: E402
from bayesfilter.inference.hmc import (  # noqa: E402
    RetainedSampleHMCArchiveConfig,
    build_retained_sample_hmc_archive_runner,
)
from bayesfilter.inference.hmc_posterior_diagnostics import (  # noqa: E402
    compute_coordinate_diagnostics,
    posterior_mean_diagnostics,
)
from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact  # noqa: E402
from bayesfilter.inference.posterior_adapter import ValueScoreCapability  # noqa: E402
from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import (  # noqa: E402
    complexity_posterior_target,
)


SCHEMA = "bayesfilter.ssl_lstm.neutra_complexity_retained_hmc.v1"
PHASE4_SCHEMA = "bayesfilter.ssl_lstm.neutra_complexity_hmc_tuning.v1"
PLAN = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-neutra-hmc-state-complexity-ladder-plan-2026-07-19.md"
)
SCRIPT = Path(__file__).resolve().relative_to(ROOT)
Q_VALUES = (1, 2, 5, 10, 20)
CHARTS = ("chart-a", "chart-b")
INITIAL_Z = (
    (0.0, 0.0, 0.0, 0.0),
    (0.5, -0.5, 0.5, -0.5),
    (-0.5, 0.5, -0.5, 0.5),
    (0.5, 0.5, -0.5, -0.5),
)
ROOT_SEED = 20260719
SEGMENT_RESULTS = 256
INITIAL_BURNIN = 256
CHECKPOINT_DRAWS_PER_CHAIN = (512, 1024, 2048, 4096)
MAX_SEGMENTS = CHECKPOINT_DRAWS_PER_CHAIN[-1] // SEGMENT_RESULTS
R_HAT_MAX = 1.01
BULK_ESS_MIN = 400.0
TAIL_ESS_MIN = 400.0
MCSE_SD_RATIO_MAX = 0.05
CROSS_REPLICATION_Z_MAX = 3.0
HOST_RAM_CAP_BYTES = 64 * 1024**3
FIRST_COMPILED_SEGMENT_RESERVE_SECONDS = 900.0
SEGMENT_COST_MARGIN = 1.50
AUDIT_BATCH_SIZE = 4
VALUE_MATCH_ABS = 1.0e-10
SECOND_MOMENT_INDICES = tuple(
    (left, right) for left in range(4) for right in range(left, 4)
)


class RetainedHMCError(RuntimeError):
    """Raised when a Phase 5 binding or evidence invariant fails."""


class ResourceStop(RetainedHMCError):
    """Raised after the cumulative retained-HMC cap is exhausted."""


class HostMemoryVeto(RetainedHMCError):
    """Raised if the retained-HMC parent exceeds 64 GiB RSS."""


class Budget:
    def __init__(self, seconds: float, *, prior_seconds: float = 0.0) -> None:
        self.seconds = float(seconds)
        self.prior_seconds = float(prior_seconds)
        self.started = time.perf_counter()
        self.observed_seconds_per_transition_leapfrog: list[float] = []

    @property
    def elapsed(self) -> float:
        return self.prior_seconds + time.perf_counter() - self.started

    def require(self, reserve_seconds: float) -> None:
        if self.elapsed + float(reserve_seconds) >= self.seconds:
            raise ResourceStop("declared retained-HMC cap exhausted")

    def observe(self, seconds_per_transition_leapfrog: float) -> None:
        value = float(seconds_per_transition_leapfrog)
        if math.isfinite(value) and value > 0.0:
            self.observed_seconds_per_transition_leapfrog.append(value)

    def segment_reserve(
        self,
        *,
        transitions: int,
        leapfrog_steps: int,
        cold_runner: bool,
    ) -> float:
        execution = 0.0
        if self.observed_seconds_per_transition_leapfrog:
            execution = (
                max(self.observed_seconds_per_transition_leapfrog)
                * int(transitions)
                * int(leapfrog_steps)
                * SEGMENT_COST_MARGIN
            )
        compile_reserve = FIRST_COMPILED_SEGMENT_RESERVE_SECONDS if cold_runner else 0.0
        return max(60.0, execution + compile_reserve)


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "numpy"):
        return json_safe(value.numpy())
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def canonical(payload: Any) -> bytes:
    return (
        json.dumps(
            json_safe(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical(payload)).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RetainedHMCError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, payload: Any, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise RetainedHMCError(f"output already exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(payload))
    temporary.replace(path)


def repo_path(path: Path, *, label: str) -> Path:
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise RetainedHMCError(f"{label} must remain inside the repository")
    return resolved


def git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


class TargetBridge:
    def __init__(self, target: Any, *, evidence_path: str) -> None:
        self.target = target
        self.parameter_dim = int(target.parameter_dim)
        self.parameter_names = tuple(target.parameter_names)
        self.target_scope = f"{target.target_scope}:phase5_retained_hmc"
        self.evidence_path = str(evidence_path)

    def adapter_signature(self) -> str:
        return hashlib.sha256(
            (self.target.adapter_signature() + ":phase5-retained-hmc").encode("ascii")
        ).hexdigest()

    def target_signature(self) -> str:
        return self.target.target_signature()

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
            runtime_backend="ssl_lstm_complexity_phase5_target_bridge",
            evidence_path=self.evidence_path,
            target_scope=self.target_scope,
            nonclaims=(
                "q-specific retained-HMC authority only",
                "fixed Phase 4 kernel only",
                "no posterior oracle or stationarity proof",
            ),
        )

    def log_prob_and_grad(self, values: Any) -> tuple[tf.Tensor, tf.Tensor]:
        tensor = tf.convert_to_tensor(values, tf.float64)
        if tensor.shape.rank == 1:
            return self.target.value_and_score(tensor)
        if tensor.shape.rank == 2:
            return self.target.batch_value_and_score(tensor)
        raise ValueError("target bridge requires rank-one or rank-two positions")


def execution_source_signature() -> str:
    paths = (
        SCRIPT,
        Path("bayesfilter/nonlinear/ssl_lstm_complexity_target_tf.py"),
        Path("bayesfilter/inference/batched_value_score.py"),
        Path("bayesfilter/inference/neutra_artifacts.py"),
        Path("bayesfilter/inference/hmc.py"),
        Path("bayesfilter/inference/hmc_posterior_diagnostics.py"),
    )
    return payload_sha256({path.as_posix(): sha256(ROOT / path) for path in paths})


def load_phase4_contract(
    q: int, summary_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = repo_path(summary_path, label="Phase 4 summary")
    payload = strict_json(path)
    if payload.get("schema") != PHASE4_SCHEMA:
        raise RetainedHMCError("Phase 4 summary schema mismatch")
    if payload.get("status") != "KERNELS_FROZEN":
        raise RetainedHMCError("Phase 4 did not freeze both kernels")
    if int(payload.get("q", -1)) != q:
        raise RetainedHMCError("Phase 4 q mismatch")
    bindings = payload.get("bindings")
    tuning = payload.get("tuning")
    if not isinstance(bindings, Mapping) or not isinstance(tuning, Mapping):
        raise RetainedHMCError("Phase 4 bindings/tuning are missing")
    if set(bindings) != set(CHARTS) or set(tuning) != set(CHARTS):
        raise RetainedHMCError("Phase 4 chart set mismatch")
    recorded_sources = payload.get("source_bindings", {}).get("source_sha256", {})
    current_sources = {
        "target": sha256(ROOT / "bayesfilter/nonlinear/ssl_lstm_complexity_target_tf.py"),
        "adapter": sha256(ROOT / "bayesfilter/inference/batched_value_score.py"),
        "artifact_loader": sha256(ROOT / "bayesfilter/inference/neutra_artifacts.py"),
        "hmc": sha256(ROOT / "bayesfilter/inference/hmc.py"),
    }
    if not isinstance(recorded_sources, Mapping) or any(
        recorded_sources.get(key) != value for key, value in current_sources.items()
    ):
        raise RetainedHMCError("shared runtime source drift since Phase 4 tuning")
    result = {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
        "q": q,
        "phase4_execution_source_signature": payload.get(
            "execution_source_signature"
        ),
    }
    return payload, result


def phase4_cost_observations(phase4: Mapping[str, Any]) -> tuple[float, ...]:
    observations = []
    for chart in CHARTS:
        row = phase4["tuning"][chart]
        arms = [
            *row.get("scale_rows", []),
            *row.get("trajectory_rows", []),
        ]
        for key in ("confirmation", "adjacent_repair"):
            if isinstance(row.get(key), Mapping):
                arms.append(row[key])
        for arm in arms:
            value = arm.get("timing", {}).get("seconds_per_transition_leapfrog")
            if value is not None and math.isfinite(float(value)) and float(value) > 0.0:
                observations.append(float(value))
    if not observations:
        raise RetainedHMCError("Phase 4 contains no usable HMC cost observations")
    return tuple(observations)


def load_binding(
    q: int,
    label: str,
    phase4: Mapping[str, Any],
    phase4_path: Path,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    binding = phase4["bindings"][label]
    tuning = phase4["tuning"][label]
    if tuning.get("status") != "ADMITTED":
        raise RetainedHMCError(f"{label} Phase 4 tuning is not ADMITTED")
    kernel = tuning.get("selected_kernel")
    if not isinstance(kernel, Mapping):
        raise RetainedHMCError(f"{label} selected kernel is missing")
    if kernel.get("mass_matrix") != "identity":
        raise RetainedHMCError(f"{label} non-identity mass is outside Phase 5")
    step_size = float(kernel.get("step_size", float("nan")))
    leapfrog_steps = int(kernel.get("num_leapfrog_steps", 0))
    if not math.isfinite(step_size) or step_size <= 0.0 or leapfrog_steps <= 0:
        raise RetainedHMCError(f"{label} selected kernel is invalid")
    payload_path = repo_path(Path(str(binding["payload_path"])), label=f"{label} payload")
    if sha256(payload_path) != binding.get("payload_sha256"):
        raise RetainedHMCError(f"{label} payload hash mismatch")
    target = complexity_posterior_target(q, jit_compile=True)
    if target.target_signature() != binding.get("target_signature"):
        raise RetainedHMCError(f"{label} target signature mismatch")
    artifact = load_frozen_neutra_artifact(
        strict_json(payload_path),
        expected_target_signature=target.target_signature(),
    )
    if artifact.artifact_signature != binding.get("artifact_signature"):
        raise RetainedHMCError(f"{label} artifact signature mismatch")
    bridge = TargetBridge(
        target,
        evidence_path=repo_path(phase4_path, label="Phase 4 summary")
        .relative_to(ROOT)
        .as_posix(),
    )
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=bridge,
        transport=artifact.transport,
        target_scope=f"{bridge.target_scope}:{label}",
        runtime_backend="ssl_lstm_complexity_phase5_fixed_transport_hmc",
        evidence_path=bridge.evidence_path,
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
        require_batch_native=True,
        nonclaims=(
            "immutable fixed Phase 3 transport",
            "identity-mass Phase 4 kernel",
            "finite-sample retained admission only",
        ),
    )
    if not adapter.value_score_capability().is_accepted_full_chain_xla_diagnostic_authority:
        raise RetainedHMCError(f"{label} retained-HMC authority did not bind")
    return adapter, {
        "label": label,
        "phase3_binding_signature": binding["binding_signature"],
        "payload_path": binding["payload_path"],
        "payload_sha256": binding["payload_sha256"],
        "artifact_signature": binding["artifact_signature"],
        "target_signature": binding["target_signature"],
        "scoped_adapter_signature": adapter.adapter_signature(),
        "scoped_target_scope": adapter.target_scope,
    }, {
        "mass_matrix": "identity",
        "step_size": step_size,
        "num_leapfrog_steps": leapfrog_steps,
        "trajectory_length": step_size * leapfrog_steps,
        "phase4_confirmation_seed": kernel.get("confirmation_seed"),
    }


def acquisition_seed(label: str, segment_index: int, q: int) -> tuple[int, int]:
    chart_offset = 0 if label == "chart-a" else 1000
    base = 30000 + 100 * q + chart_offset + 10 * int(segment_index)
    return ROOT_SEED, base


def parse_tensor(path: Path, dtype: tf.DType = tf.float64) -> tf.Tensor:
    return tf.io.parse_tensor(path.read_bytes(), out_type=dtype)


def read_archive(archive_dir: Path, archive_label: str) -> dict[str, Any]:
    manifest_path = archive_dir / f"{archive_label}_private_manifest.json"
    manifest = strict_json(manifest_path)
    if manifest.get("artifact_type") != "bayesfilter_private_retained_sample_hmc_archive":
        raise RetainedHMCError("unexpected retained archive type")
    shards = manifest.get("sample_shards")
    sidecars = manifest.get("sidecars")
    if not isinstance(shards, list) or len(shards) != 1 or not isinstance(sidecars, Mapping):
        raise RetainedHMCError("retained archive descriptor mismatch")
    sample_row = shards[0]
    state_row = sidecars.get("final_state")
    target_row = sidecars.get("final_target_log_prob")
    if not isinstance(state_row, Mapping) or not isinstance(target_row, Mapping):
        raise RetainedHMCError("retained archive sidecars missing")
    rows = (sample_row, state_row, target_row)
    for row in rows:
        row_path = repo_path(Path(str(row["path"])), label="retained tensor")
        if sha256(row_path) != row.get("sha256"):
            raise RetainedHMCError("retained tensor hash mismatch")
    samples = parse_tensor(repo_path(Path(str(sample_row["path"])), label="samples"))
    final_state = parse_tensor(
        repo_path(Path(str(state_row["path"])), label="final state")
    )
    final_target = parse_tensor(
        repo_path(Path(str(target_row["path"])), label="final target")
    )
    if tuple(samples.shape) != (SEGMENT_RESULTS, 4, 4):
        raise RetainedHMCError("retained sample shape mismatch")
    if tuple(final_state.shape) != (4, 4) or tuple(final_target.shape) != (4,):
        raise RetainedHMCError("retained sidecar shape mismatch")
    return {
        "samples": samples,
        "final_state": final_state,
        "final_target_log_prob": final_target,
        "manifest": manifest,
        "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256(manifest_path),
        "sample_sha256": str(sample_row["sha256"]),
        "final_state_sha256": str(state_row["sha256"]),
        "final_target_log_prob_sha256": str(target_row["sha256"]),
    }


def build_audit_batch(adapter: Any) -> Any:
    @tf.function(
        input_signature=(
            tf.TensorSpec((AUDIT_BATCH_SIZE, 4), tf.float64),
        ),
        jit_compile=True,
        reduce_retracing=True,
    )
    def audit_batch(z: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        values, scores = adapter.log_prob_and_grad_batch(z)
        theta = adapter.latent_to_position(z)
        return values, scores, theta

    return audit_batch


def audit_segment(audit_batch: Any, archive: Mapping[str, Any]) -> tuple[tf.Tensor, dict[str, Any]]:
    samples = tf.convert_to_tensor(archive["samples"], tf.float64)
    flat = tf.reshape(samples, (-1, 4))
    all_finite = bool(tf.reduce_all(tf.math.is_finite(samples)).numpy())
    final_value_residual = float("nan")
    theta_chunks = []
    for start in range(0, int(flat.shape[0]), AUDIT_BATCH_SIZE):
        batch = flat[start : start + AUDIT_BATCH_SIZE]
        values, scores, theta = audit_batch(batch)
        theta_chunks.append(theta)
        all_finite = all_finite and bool(
            tf.reduce_all(tf.math.is_finite(values)).numpy()
            and tf.reduce_all(tf.math.is_finite(scores)).numpy()
        )
        if start + AUDIT_BATCH_SIZE == int(flat.shape[0]):
            final_values = tf.reshape(values, (-1, 4))[-1]
            final_value_residual = float(
                tf.reduce_max(
                    tf.abs(final_values - archive["final_target_log_prob"])
                ).numpy()
            )
    theta = tf.reshape(tf.concat(theta_chunks, axis=0), tf.shape(samples))
    theta_finite = bool(tf.reduce_all(tf.math.is_finite(theta)).numpy())
    trace_count_fn = getattr(audit_batch, "experimental_get_tracing_count", None)
    trace_count = None if trace_count_fn is None else int(trace_count_fn())
    return theta, {
        "audit_batch_size": AUDIT_BATCH_SIZE,
        "jit_compile": True,
        "compile_trace_count": trace_count,
        "all_samples_values_scores_finite": all_finite,
        "all_mapped_theta_finite": theta_finite,
        "final_target_log_prob_max_abs_residual": final_value_residual,
        "value_match_threshold": VALUE_MATCH_ABS,
        "passed": (
            all_finite
            and theta_finite
            and final_value_residual <= VALUE_MATCH_ABS
            and trace_count == 1
        ),
    }


def cumulative_movement(samples: tf.Tensor, initial_state: tf.Tensor) -> list[bool]:
    previous = tf.concat((initial_state[tf.newaxis, ...], samples[:-1]), axis=0)
    moved = tf.reduce_any(tf.not_equal(samples, previous), axis=(0, 2))
    return [bool(value) for value in moved.numpy().tolist()]


def coordinate_screen(chain_major: tf.Tensor) -> tuple[dict[str, Any], list[str]]:
    values = json_safe(compute_coordinate_diagnostics(chain_major))
    failures = []
    rhat = values["rank_normalized_split_rhat"]["maximum"]
    bulk = values["rank_normalized_ess"]["bulk"]
    tail = values["rank_normalized_ess"]["tail"]
    ratio = values["mean"]["mcse_sd_ratio"]
    arrays = (rhat, bulk, tail, ratio)
    if not all(math.isfinite(float(item)) for row in arrays for item in row):
        return values, ["nonfinite_rank_ess_or_mcse"]
    if max(float(item) for item in rhat) > R_HAT_MAX:
        failures.append("rank_normalized_split_rhat_above_threshold")
    if min(float(item) for item in bulk) < BULK_ESS_MIN:
        failures.append("bulk_ess_below_threshold")
    if min(float(item) for item in tail) < TAIL_ESS_MIN:
        failures.append("tail_ess_below_threshold")
    if max(float(item) for item in ratio) > MCSE_SD_RATIO_MAX:
        failures.append("mcse_sd_ratio_above_threshold")
    return values, failures


def cumulative_admission(
    *,
    z_draw_major: tf.Tensor,
    theta_draw_major: tf.Tensor,
    initial_state: tf.Tensor,
    segment_manifests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    z = tf.convert_to_tensor(z_draw_major, tf.float64)
    theta = tf.convert_to_tensor(theta_draw_major, tf.float64)
    if tuple(z.shape[1:]) != (4, 4) or theta.shape != z.shape:
        raise RetainedHMCError("cumulative z/theta shape mismatch")
    hard_vetoes = []
    promotion_vetoes = []
    if not bool(tf.reduce_all(tf.math.is_finite(z)).numpy()) or not bool(
        tf.reduce_all(tf.math.is_finite(theta)).numpy()
    ):
        hard_vetoes.append("nonfinite_z_or_theta")
    moved = cumulative_movement(z, initial_state)
    if not all(moved):
        hard_vetoes.append("unmoved_chain")
    divergence_statuses = []
    acceptance_counts = tf.zeros((4,), tf.int64)
    acceptance_totals = tf.zeros((4,), tf.int64)
    for manifest in segment_manifests:
        diagnostics = manifest["diagnostics_private_metadata"]
        health = diagnostics["sampler_health_diagnostics"]
        if int(health["log_accept_ratio"]["nonfinite_count"]) > 0:
            hard_vetoes.append("nonfinite_log_accept_ratio")
        if int(health["target_log_prob"]["nonfinite_count"]) > 0:
            hard_vetoes.append("nonfinite_target_log_prob")
        divergence_statuses.append(str(diagnostics["native_divergence_status"]))
        if diagnostics.get("divergence_count") not in (None, 0):
            hard_vetoes.append("positive_native_divergence")
        draws = int(manifest["retained_sample_count"])
        rates = [float(value) for value in health["acceptance_rate_by_chain"]]
        counts = [int(round(draws * value)) for value in rates]
        if any(
            abs(draws * value - count) > 1.0e-9
            for value, count in zip(rates, counts, strict=True)
        ):
            raise RetainedHMCError("acceptance rates do not reconstruct exact counts")
        acceptance_counts += tf.constant(counts, tf.int64)
        acceptance_totals += tf.fill((4,), tf.cast(draws, tf.int64))
    if hard_vetoes:
        coordinates: Mapping[str, Any] = {"status": "not_computed_hard_veto"}
    else:
        z_diag, z_failures = coordinate_screen(tf.transpose(z, (1, 0, 2)))
        theta_diag, theta_failures = coordinate_screen(tf.transpose(theta, (1, 0, 2)))
        promotion_vetoes.extend(f"z:{item}" for item in z_failures)
        promotion_vetoes.extend(f"theta:{item}" for item in theta_failures)
        coordinates = {"z": z_diag, "theta": theta_diag}
    admitted = not hard_vetoes and not promotion_vetoes
    return {
        "admitted": admitted,
        "decision": (
            "ADMITTED"
            if admitted
            else "HARD_VETO_STOP"
            if hard_vetoes
            else "EXTEND_TO_NEXT_FROZEN_CHECKPOINT"
        ),
        "draw_count_per_chain": int(z.shape[0]),
        "chain_moved": moved,
        "acceptance_rate_by_chain": json_safe(
            tf.cast(acceptance_counts, tf.float64)
            / tf.cast(acceptance_totals, tf.float64)
        ),
        "native_divergence_statuses": divergence_statuses,
        "hard_vetoes": list(dict.fromkeys(hard_vetoes)),
        "promotion_vetoes": list(dict.fromkeys(promotion_vetoes)),
        "coordinate_diagnostics": coordinates,
        "thresholds": {
            "rhat_max": R_HAT_MAX,
            "bulk_ess_min": BULK_ESS_MIN,
            "tail_ess_min": TAIL_ESS_MIN,
            "mcse_sd_ratio_max": MCSE_SD_RATIO_MAX,
        },
    }


def functional_draws(theta_chain_major: tf.Tensor) -> tf.Tensor:
    theta = tf.convert_to_tensor(theta_chain_major, tf.float64)
    seconds = tf.stack(
        [theta[..., left] * theta[..., right] for left, right in SECOND_MOMENT_INDICES],
        axis=-1,
    )
    return tf.concat((theta, seconds), axis=-1)


def cross_replication_stability(
    theta_a_chain_major: tf.Tensor,
    theta_b_chain_major: tf.Tensor,
) -> dict[str, Any]:
    functionals = {
        "chart-a": functional_draws(theta_a_chain_major),
        "chart-b": functional_draws(theta_b_chain_major),
    }
    diagnostics = {
        label: posterior_mean_diagnostics(values)
        for label, values in functionals.items()
    }
    means = {
        label: tf.convert_to_tensor(row["pooled_mean"], tf.float64)
        for label, row in diagnostics.items()
    }
    mcse = {
        label: tf.convert_to_tensor(row["mean_mcse"], tf.float64)
        for label, row in diagnostics.items()
    }
    denominator = tf.sqrt(tf.square(mcse["chart-a"]) + tf.square(mcse["chart-b"]))
    standardized = tf.where(
        denominator > 0.0,
        tf.abs(means["chart-a"] - means["chart-b"]) / denominator,
        tf.fill(tf.shape(denominator), tf.constant(float("nan"), tf.float64)),
    )
    finite = bool(tf.reduce_all(tf.math.is_finite(standardized)).numpy())
    maximum = float(tf.reduce_max(standardized).numpy()) if finite else float("nan")
    passed = finite and maximum <= CROSS_REPLICATION_Z_MAX
    names = [
        *(f"mean_theta_{index}" for index in range(4)),
        *(f"raw_second_theta_{left}_{right}" for left, right in SECOND_MOMENT_INDICES),
    ]
    return {
        "passed": passed,
        "decision": "STABILITY_SCREEN_PASSED" if passed else "STABILITY_PROMOTION_VETO",
        "functional_names": names,
        "standardized_absolute_difference": json_safe(standardized),
        "maximum_standardized_absolute_difference": json_safe(maximum),
        "threshold": CROSS_REPLICATION_Z_MAX,
        "means": {label: json_safe(value) for label, value in means.items()},
        "mean_mcse": {label: json_safe(value) for label, value in mcse.items()},
        "nonclaims": [
            "not a formal equivalence test",
            "neither chart is an oracle",
            "no sampler or transport ranking",
        ],
    }


def build_runner(
    adapter: Any,
    initial_state: tf.Tensor,
    kernel: Mapping[str, Any],
    *,
    burnin: int,
    seed: tuple[int, int],
) -> Any:
    config = RetainedSampleHMCArchiveConfig(
        num_results=SEGMENT_RESULTS,
        num_burnin_steps=burnin,
        step_size=float(kernel["step_size"]),
        num_leapfrog_steps=int(kernel["num_leapfrog_steps"]),
        seed=seed,
        use_xla=True,
        target_scope=adapter.target_scope,
        chain_execution_mode="tf_function",
    )
    return build_retained_sample_hmc_archive_runner(adapter, initial_state, config)


def segment_label(chart: str, index: int) -> str:
    return f"{chart}-retained-segment-{index:03d}"


def run_segment(
    *,
    runner: Any,
    audit_batch: Any,
    archive_dir: Path,
    chart: str,
    index: int,
    current_state: tf.Tensor,
    seed: tuple[int, int],
    previous: Mapping[str, Any] | None,
    kernel: Mapping[str, Any],
    phase4_sha256: str,
    source_signature: str,
    budget: Budget,
) -> tuple[dict[str, Any], dict[str, Any], tf.Tensor]:
    segment_started = time.perf_counter()
    transitions = SEGMENT_RESULTS + (INITIAL_BURNIN if index == 0 else 0)
    reserve = budget.segment_reserve(
        transitions=transitions,
        leapfrog_steps=int(kernel["num_leapfrog_steps"]),
        cold_runner=int(getattr(runner, "_call_count", 0)) == 0,
    )
    budget.require(reserve)
    label = segment_label(chart, index)
    result = runner.run(
        archive_dir=archive_dir,
        archive_label=label,
        current_state=current_state,
        seed=seed,
        step_size=float(kernel["step_size"]),
        metadata={
            "schema": SCHEMA,
            "chart": chart,
            "segment_index": index,
            "segment_seed": list(seed),
            "phase4_summary_sha256": phase4_sha256,
            "execution_source_signature": source_signature,
            "previous_manifest_sha256": (
                None if previous is None else previous["manifest_sha256"]
            ),
            "previous_final_state_sha256": (
                None if previous is None else previous["final_state_sha256"]
            ),
        },
        overwrite=False,
    )
    archive = read_archive(archive_dir, label)
    caller = archive["manifest"]["metadata"]["caller_metadata"]
    expected_manifest = None if previous is None else previous["manifest_sha256"]
    expected_state = None if previous is None else previous["final_state_sha256"]
    if caller.get("previous_manifest_sha256") != expected_manifest:
        raise RetainedHMCError("retained manifest lineage mismatch")
    if caller.get("previous_final_state_sha256") != expected_state:
        raise RetainedHMCError("retained state lineage mismatch")
    theta, audit = audit_segment(audit_batch, archive)
    diagnostics = json_safe(result.diagnostics)
    hard_vetoes = []
    if not audit["passed"]:
        hard_vetoes.append("post_archive_value_score_audit_failed")
    if not diagnostics.get("retained_samples_all_finite"):
        hard_vetoes.append("nonfinite_retained_samples")
    if diagnostics.get("divergence_count") not in (None, 0):
        hard_vetoes.append("positive_native_divergence")
    health = diagnostics.get("sampler_health_diagnostics", {})
    if health.get("log_accept_ratio", {}).get("nonfinite_count"):
        hard_vetoes.append("nonfinite_log_accept_ratio")
    if health.get("target_log_prob", {}).get("nonfinite_count"):
        hard_vetoes.append("nonfinite_target_log_prob")
    devices = sorted(
        {str(result.final_state.device), str(result.final_target_log_prob.device)}
    )
    if not devices or not all("GPU:" in value for value in devices):
        hard_vetoes.append("trusted_gpu_output_placement_missing")
    runner_wall = float(result.metadata["call_s"])
    segment_wall = time.perf_counter() - segment_started
    normalized = segment_wall / (
        transitions * int(kernel["num_leapfrog_steps"])
    )
    budget.observe(normalized)
    host_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    if host_rss > HOST_RAM_CAP_BYTES:
        raise HostMemoryVeto("retained-HMC parent RSS exceeded 64 GiB")
    public = {
        "label": label,
        "chart": chart,
        "segment_index": index,
        "seed": list(seed),
        "archive_hashes": {
            "private_manifest_sha256": archive["manifest_sha256"],
            "sample_sha256": archive["sample_sha256"],
            "final_state_sha256": archive["final_state_sha256"],
            "final_target_log_prob_sha256": archive[
                "final_target_log_prob_sha256"
            ],
        },
        "diagnostics": diagnostics,
        "post_archive_value_score_audit": audit,
        "runner_metadata": json_safe(result.metadata),
        "evidence_output_devices": devices,
        "runner_call_seconds": runner_wall,
        "segment_plus_audit_seconds": segment_wall,
        "resource_reserve_seconds_before_launch": reserve,
        "seconds_per_transition_leapfrog": normalized,
        "hard_vetoes": hard_vetoes,
        "passed": not hard_vetoes,
    }
    return archive, public, theta


def material_contract(
    args: argparse.Namespace,
    phase4_binding: Mapping[str, Any],
    source_signature: str,
) -> dict[str, Any]:
    return {
        "q": int(args.q),
        "phase4_summary": dict(phase4_binding),
        "execution_source_signature": source_signature,
        "segment_results": SEGMENT_RESULTS,
        "initial_burnin": INITIAL_BURNIN,
        "continuation_burnin": 0,
        "checkpoint_draws_per_chain": list(CHECKPOINT_DRAWS_PER_CHAIN),
        "selected_hmc_topology": "retained_archive_batched_four_chain_xla",
    }


def source_bindings() -> dict[str, Any]:
    paths = {
        "runner": SCRIPT,
        "plan": PLAN,
        "target": Path("bayesfilter/nonlinear/ssl_lstm_complexity_target_tf.py"),
        "adapter": Path("bayesfilter/inference/batched_value_score.py"),
        "artifact_loader": Path("bayesfilter/inference/neutra_artifacts.py"),
        "hmc": Path("bayesfilter/inference/hmc.py"),
        "diagnostics": Path("bayesfilter/inference/hmc_posterior_diagnostics.py"),
    }
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "source_paths": {key: value.as_posix() for key, value in paths.items()},
        "source_sha256": {key: sha256(ROOT / value) for key, value in paths.items()},
    }


def run_manifest(args: argparse.Namespace, budget: Budget) -> dict[str, Any]:
    try:
        gpu_memory = json_safe(tf.config.experimental.get_memory_info("GPU:0"))
    except (ValueError, RuntimeError):
        gpu_memory = {"status": "unavailable"}
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "command": " ".join(sys.argv),
        "environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
        "python": sys.version.split()[0],
        "tensorflow": tf.__version__,
        "selected_physical_gpu": SELECTED_GPU,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_gpus": [device.name for device in tf.config.list_logical_devices("GPU")],
        "jit_compile": True,
        "tf32": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        "charged_seconds": budget.elapsed,
        "cap_seconds": budget.seconds,
        "random_seeds": {
            chart: [list(acquisition_seed(chart, index, args.q)) for index in range(MAX_SEGMENTS)]
            for chart in CHARTS
        },
        "host_ru_maxrss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
        "host_ram_cap_bytes": HOST_RAM_CAP_BYTES,
        "gpu_allocator_memory": gpu_memory,
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        "plan": PLAN.as_posix(),
        "output_root": args.output_root.as_posix(),
    }


def configure_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RetainedHMCError("retained-HMC acquisition requires a visible GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            if "cannot be modified after being initialized" not in str(exc):
                raise
    tf.config.experimental.enable_tensor_float_32_execution(True)
    tf.config.set_soft_device_placement(False)
    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
    except (ValueError, RuntimeError):
        pass


def run_acquisition(args: argparse.Namespace) -> dict[str, Any]:
    output = repo_path(args.output_root, label="output root")
    output.mkdir(parents=True, exist_ok=args.resume)
    summary_path = output / "summary.json"
    phase4, phase4_binding = load_phase4_contract(args.q, args.phase4_summary)
    source_signature = execution_source_signature()
    contract = material_contract(args, phase4_binding, source_signature)
    prior_seconds = 0.0
    previous_summary: Mapping[str, Any] = {}
    if args.resume:
        if not summary_path.is_file():
            raise RetainedHMCError("resume requires summary.json")
        previous_summary = strict_json(summary_path)
        if previous_summary.get("material_contract") != contract:
            raise RetainedHMCError("resume material contract mismatch")
        prior_seconds = float(previous_summary["run_manifest"]["charged_seconds"])
    budget = Budget(args.cap_seconds, prior_seconds=prior_seconds)
    for observation in phase4_cost_observations(phase4):
        budget.observe(observation)
    chart_receipts = {}
    admitted_theta = {}
    resource_stop = None
    hard_veto = None
    for chart in CHARTS:
        if resource_stop or hard_veto:
            break
        adapter, binding, kernel = load_binding(
            args.q, chart, phase4, args.phase4_summary
        )
        initial_state = tf.constant(INITIAL_Z, tf.float64)
        initial_runner = build_runner(
            adapter,
            initial_state,
            kernel,
            burnin=INITIAL_BURNIN,
            seed=acquisition_seed(chart, 0, args.q),
        )
        continuation_runner = build_runner(
            adapter,
            initial_state,
            kernel,
            burnin=0,
            seed=acquisition_seed(chart, 1, args.q),
        )
        audit_batch = build_audit_batch(adapter)
        archive_dir = output / "retained-private" / chart
        archives = []
        public_segments = []
        theta_segments = []
        checkpoints = []
        current = initial_state
        final_admission = {"admitted": False, "decision": "NOT_YET_EVALUATED"}
        try:
            for index in range(MAX_SEGMENTS):
                label = segment_label(chart, index)
                manifest_path = archive_dir / f"{label}_private_manifest.json"
                if args.resume and manifest_path.is_file():
                    archive = read_archive(archive_dir, label)
                    caller = archive["manifest"]["metadata"]["caller_metadata"]
                    if caller.get("phase4_summary_sha256") != phase4_binding["sha256"]:
                        raise RetainedHMCError("resumed segment Phase 4 binding mismatch")
                    if caller.get("execution_source_signature") != source_signature:
                        raise RetainedHMCError("resumed segment source signature mismatch")
                    expected_previous = None if index == 0 else archives[-1]["manifest_sha256"]
                    expected_state = None if index == 0 else archives[-1]["final_state_sha256"]
                    if caller.get("previous_manifest_sha256") != expected_previous:
                        raise RetainedHMCError("resumed segment lineage mismatch")
                    if caller.get("previous_final_state_sha256") != expected_state:
                        raise RetainedHMCError("resumed segment state lineage mismatch")
                    audit_started = time.perf_counter()
                    theta, audit = audit_segment(audit_batch, archive)
                    audit_seconds = time.perf_counter() - audit_started
                    if not audit["passed"]:
                        raise RetainedHMCError("resumed segment audit failed")
                    diagnostics = archive["manifest"]["diagnostics_private_metadata"]
                    health = diagnostics.get("sampler_health_diagnostics", {})
                    if diagnostics.get("divergence_count") not in (None, 0):
                        raise RetainedHMCError("resumed segment has positive divergence")
                    if health.get("log_accept_ratio", {}).get("nonfinite_count"):
                        raise RetainedHMCError("resumed segment has nonfinite log acceptance")
                    if health.get("target_log_prob", {}).get("nonfinite_count"):
                        raise RetainedHMCError("resumed segment has nonfinite target value")
                    runner_metadata = archive["manifest"]["metadata"]["runner_metadata"]
                    transitions = SEGMENT_RESULTS + (INITIAL_BURNIN if index == 0 else 0)
                    segment_plus_audit = float(runner_metadata["call_s"]) + audit_seconds
                    normalized = segment_plus_audit / (
                        transitions * int(kernel["num_leapfrog_steps"])
                    )
                    budget.observe(normalized)
                    public = {
                        "label": label,
                        "chart": chart,
                        "segment_index": index,
                        "seed": list(acquisition_seed(chart, index, args.q)),
                        "archive_hashes": {
                            "private_manifest_sha256": archive["manifest_sha256"],
                            "sample_sha256": archive["sample_sha256"],
                            "final_state_sha256": archive["final_state_sha256"],
                            "final_target_log_prob_sha256": archive[
                                "final_target_log_prob_sha256"
                            ],
                        },
                        "post_archive_value_score_audit": audit,
                        "diagnostics": diagnostics,
                        "runner_metadata": runner_metadata,
                        "runner_call_seconds": float(runner_metadata["call_s"]),
                        "resume_reaudit_seconds": audit_seconds,
                        "segment_plus_resume_reaudit_seconds": segment_plus_audit,
                        "seconds_per_transition_leapfrog": normalized,
                        "resumed": True,
                        "hard_vetoes": [],
                        "passed": True,
                    }
                else:
                    if execution_source_signature() != source_signature:
                        raise RetainedHMCError("execution source drift during acquisition")
                    archive, public, theta = run_segment(
                        runner=initial_runner if index == 0 else continuation_runner,
                        audit_batch=audit_batch,
                        archive_dir=archive_dir,
                        chart=chart,
                        index=index,
                        current_state=current,
                        seed=acquisition_seed(chart, index, args.q),
                        previous=None if index == 0 else archives[-1],
                        kernel=kernel,
                        phase4_sha256=phase4_binding["sha256"],
                        source_signature=source_signature,
                        budget=budget,
                    )
                archives.append(archive)
                public_segments.append(public)
                theta_segments.append(theta)
                current = archive["final_state"]
                if public.get("hard_vetoes"):
                    final_admission = {
                        "admitted": False,
                        "decision": "HARD_VETO_STOP",
                        "hard_vetoes": public["hard_vetoes"],
                        "promotion_vetoes": [],
                        "draw_count_per_chain": (index + 1) * SEGMENT_RESULTS,
                    }
                    hard_veto = ";".join(public["hard_vetoes"])
                    break
                draw_count = (index + 1) * SEGMENT_RESULTS
                if draw_count in CHECKPOINT_DRAWS_PER_CHAIN:
                    admission = cumulative_admission(
                        z_draw_major=tf.concat(
                            [item["samples"] for item in archives], axis=0
                        ),
                        theta_draw_major=tf.concat(theta_segments, axis=0),
                        initial_state=initial_state,
                        segment_manifests=[item["manifest"] for item in archives],
                    )
                    checkpoints.append(admission)
                    final_admission = admission
                    if admission["hard_vetoes"]:
                        hard_veto = ";".join(admission["hard_vetoes"])
                        break
                    if admission["admitted"]:
                        admitted_theta[chart] = tf.transpose(
                            tf.concat(theta_segments, axis=0), (1, 0, 2)
                        )
                        break
        except ResourceStop as exc:
            resource_stop = str(exc)
        except HostMemoryVeto as exc:
            hard_veto = str(exc)
        except RetainedHMCError as exc:
            hard_veto = str(exc)
        if (
            final_admission.get("admitted") is not True
            and not final_admission.get("hard_vetoes")
            and len(archives) == MAX_SEGMENTS
        ):
            final_admission = dict(final_admission)
            final_admission["decision"] = "MAXIMUM_OPPORTUNITY_EXHAUSTED_NOT_ADMITTED"
        chart_receipts[chart] = {
            "binding": binding,
            "kernel": kernel,
            "segments": public_segments,
            "checkpoints": checkpoints,
            "final_admission": final_admission,
            "executed_segment_count": len(archives),
            "executed_draws_per_chain": len(archives) * SEGMENT_RESULTS,
        }
        partial = {
            "schema": SCHEMA,
            "mode": "acquire",
            "status": "RUNNING",
            "q": args.q,
            "material_contract": contract,
            "charts": chart_receipts,
            "resource_stop": resource_stop,
            "hard_veto": hard_veto,
            "source_bindings": source_bindings(),
            "run_manifest": run_manifest(args, budget),
        }
        write_json(summary_path, partial, replace=True)
    both_admitted = set(admitted_theta) == set(CHARTS)
    stability = (
        cross_replication_stability(
            admitted_theta["chart-a"], admitted_theta["chart-b"]
        )
        if both_admitted
        else {
            "passed": False,
            "decision": "NOT_REACHED_BOTH_CHARTS_NOT_ADMITTED",
        }
    )
    status = (
        "HARD_VETO"
        if hard_veto
        else "RESOURCE_STOP"
        if resource_stop
        else "ADMITTED"
        if both_admitted and stability["passed"]
        else "REPLICATION_STABILITY_VETO"
        if both_admitted
        else "MAXIMUM_OPPORTUNITY_NOT_ADMITTED"
    )
    payload = {
        "schema": SCHEMA,
        "mode": "acquire",
        "status": status,
        "q": args.q,
        "material_contract": contract,
        "charts": chart_receipts,
        "both_charts_admitted": both_admitted,
        "cross_replication_stability": stability,
        "resource_stop": resource_stop,
        "hard_veto": hard_veto,
        "candidate_rejection_is_not_research_direction_rejection": True,
        "scientific_interpretation": "finite_sample_replication_screen_only",
        "source_bindings": source_bindings(),
        "run_manifest": run_manifest(args, budget),
        "inference_status": {
            "hard_veto_screen": "failed" if hard_veto else "passed",
            "statistically_supported_ranking": "none",
            "descriptive_only_differences": [
                "acceptance",
                "continuous R-hat/ESS/MCSE values",
                "runtime and memory below cap",
            ],
            "default_readiness": "not_assessed",
            "next_evidence_needed": (
                "Phase 6 recovery and predictive moments"
                if status == "ADMITTED"
                else "repair named by retained-HMC result"
            ),
        },
        "nonclaims": [
            "finite-sample retained admission only",
            "no posterior oracle or stationarity proof",
            "no sampler or transport ranking",
            "no predictive or scientific-validity claim",
        ],
    }
    write_json(summary_path, payload, replace=True)
    return payload


def validate_args(args: argparse.Namespace) -> None:
    if args.mode == "contract-smoke":
        return
    if not args.authorize_material_run:
        raise RetainedHMCError("acquire requires --authorize-material-run")
    if args.cap_seconds is None or args.cap_seconds <= 0.0:
        raise RetainedHMCError("acquire requires a positive cumulative cap")
    if args.phase4_summary is None:
        raise RetainedHMCError("acquire requires --phase4-summary")
    if args.output_root is None:
        raise RetainedHMCError("acquire requires an explicit output root")
    repo_path(args.phase4_summary, label="Phase 4 summary")
    repo_path(args.output_root, label="output root")


def contract_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "mode": "contract-smoke",
        "status": "PASSED",
        "q": args.q,
        "segment_results": SEGMENT_RESULTS,
        "initial_burnin": INITIAL_BURNIN,
        "continuation_burnin": 0,
        "checkpoint_draws_per_chain": list(CHECKPOINT_DRAWS_PER_CHAIN),
        "thresholds": {
            "rhat_max": R_HAT_MAX,
            "bulk_ess_min": BULK_ESS_MIN,
            "tail_ess_min": TAIL_ESS_MIN,
            "mcse_sd_ratio_max": MCSE_SD_RATIO_MAX,
            "cross_replication_combined_mcse_multiplier": CROSS_REPLICATION_Z_MAX,
        },
        "diagnostic_coordinates": ["neutra_z", "common_theta"],
        "material_execution_authorized": False,
        "samples_retained_as_posterior_evidence": True,
        "source_bindings": source_bindings(),
        "nonclaims": [
            "contract/import smoke only",
            "no transport loading or HMC execution",
            "no convergence or posterior-correctness claim",
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("contract-smoke", "acquire"), required=True)
    parser.add_argument("--q", type=int, choices=Q_VALUES, required=True)
    parser.add_argument("--phase4-summary", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cap-seconds", type=float)
    parser.add_argument("--authorize-material-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.cap_seconds is not None and (
        not math.isfinite(args.cap_seconds) or args.cap_seconds <= 0.0
    ):
        parser.error("--cap-seconds must be finite and positive")
    if args.mode == "contract-smoke" and args.resume:
        parser.error("contract smoke cannot resume")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    if args.mode == "contract-smoke":
        payload = contract_payload(args)
    else:
        configure_gpu()
        payload = run_acquisition(args)
    print(
        "JSON_SUMMARY "
        + json.dumps(
            {"mode": payload["mode"], "status": payload["status"], "q": payload["q"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
