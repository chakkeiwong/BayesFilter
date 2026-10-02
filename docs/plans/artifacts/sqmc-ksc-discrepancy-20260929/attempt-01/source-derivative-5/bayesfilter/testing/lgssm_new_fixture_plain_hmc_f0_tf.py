"""Bounded plain-HMC comparator for the NeuTra F0 new LGSSM fixture."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.neutra_hmc import (
    NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
    SequentialNeuTraHMCConfig,
    TensorHMCConfig,
    run_batched_hmc,
    run_sequential_neutra_hmc,
)
from bayesfilter.testing import lgssm_neutra_gap_closure_tf as reference
from bayesfilter.testing.deterministic_lgssm_exact_target_tf import (
    load_deterministic_lgssm_exact_target,
    stable_config_hash,
)


ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = ROOT / "docs/plans/bayesfilter-neutra-hmc-robustness-f0-subplan-2026-07-15.md"
F0_ROOT = ROOT / (
    "docs/plans/artifacts/neutra-hmc-core-consolidation-and-robustness-"
    "2026-07-15/f0"
)
CONFIG_PATH = F0_ROOT / "config.json"
FIXTURE_PATH = F0_ROOT / "plain-hmc/fixture_T120_seed20260715_701.json"
MASS_PATH = F0_ROOT / "plain-hmc/mass.json"
FAILED_COMPARATOR_ROOT = F0_ROOT / "plain-hmc/comparator"
COMPARATOR_ROOT = F0_ROOT / "plain-hmc/comparator-repair-attempt-02"
EXPECTED_TARGET_SIGNATURE = (
    "312d2f4ceb5d65bf18251fa53ae1276781c62fd2daefaba0bda8dc3d46a5d283"
)
EXPECTED_MASS_ARTIFACT_HASH = (
    "sha256:c1c3bd0f6e57ae6a4165c6c85b757c3806b0cbfb700aad4b5b31a6d30f8b5145"
)
STEP_SIZES = (0.225, 0.25, 0.275, 0.3, 0.325, 0.35)
NUM_LEAPFROG_STEPS = 10
PROBE_SEED = (20260715, 7101)
WARMUP_SEED = (20260715, 7201)
RETAINED_SEED = (20260715, 7301)

NONCLAIMS = (
    "single deterministic new-fixture plain-HMC comparator only",
    "short-probe acceptance nominates but does not establish convergence",
    "no NeuTra quality, sampler superiority, calibration, or broad robustness claim",
    "no production or default-readiness claim",
)


class F0PlainHMCError(RuntimeError):
    """Raised when F0 comparator identity or evidence fails closed."""


class TensorAffineTargetAdapter:
    """TensorFlow affine coordinates for a batch-native exact target."""

    def __init__(
        self,
        *,
        base_adapter: Any,
        center: Any,
        factor: Any,
        target_signature: str,
        mass_artifact_hash: str,
    ) -> None:
        self.base_adapter = base_adapter
        self.center = tf.convert_to_tensor(center, tf.float64)
        self.factor = tf.convert_to_tensor(factor, tf.float64)
        if self.center.shape != (18,) or self.factor.shape != (18, 18):
            raise F0PlainHMCError("affine mass transform shape mismatch")
        sign, log_abs_det = tf.linalg.slogdet(self.factor)
        if float(sign.numpy()) == 0.0 or not bool(tf.math.is_finite(log_abs_det).numpy()):
            raise F0PlainHMCError("affine mass factor must be nonsingular")
        self.log_abs_det = log_abs_det
        self.parameter_dim = 18
        self.target_signature = str(target_signature)
        self.mass_artifact_hash = str(mass_artifact_hash)

    def forward(self, z: Any) -> tf.Tensor:
        values = tf.convert_to_tensor(z, tf.float64)
        return self.center + tf.tensordot(values, self.factor, axes=[[-1], [1]])

    def log_prob_and_grad(self, z: Any) -> tuple[tf.Tensor, tf.Tensor]:
        values = tf.convert_to_tensor(z, tf.float64)
        theta = self.forward(values)
        raw_value, raw_score = self.base_adapter.log_prob_and_grad(theta)
        latent_score = tf.tensordot(raw_score, self.factor, axes=[[-1], [0]])
        return raw_value + self.log_abs_det, latent_score

    def target_status_telemetry(self, z: Any) -> Mapping[str, tf.Tensor]:
        return self.base_adapter.target_status_telemetry(self.forward(z))

    def adapter_signature(self) -> str:
        return stable_config_hash(
            {
                "schema": "bayesfilter.f0_plain_hmc_affine_target.v1",
                "target_signature": self.target_signature,
                "mass_artifact_hash": self.mass_artifact_hash,
                "center": self.center,
                "factor": self.factor,
                "coordinate_program": "theta=center+z@factor.T",
                "score_program": "grad_z=grad_theta@factor",
                "jacobian": "constant_log_abs_det_factor_included",
            }
        )


def run_f0_plain_hmc_comparator() -> Mapping[str, Any]:
    """Nominate then validate a new-fixture plain-HMC comparator."""

    _require_cpu_hidden()
    if COMPARATOR_ROOT.exists():
        raise FileExistsError(f"F0 comparator root exists: {COMPARATOR_ROOT}")
    bundle, mass, adapter = _load_inputs()
    started = time.monotonic()
    initial_state = tf.zeros((4, 18), tf.float64)
    rows = []
    configuration_vetoes = []
    for index, step_size in enumerate(STEP_SIZES):
        seed = (PROBE_SEED[0], PROBE_SEED[1] + 10_000 + index)
        run = run_batched_hmc(
            adapter=adapter,
            initial_state=initial_state,
            config=TensorHMCConfig(
                num_results=64,
                num_burnin_steps=128,
                step_size=step_size,
                num_leapfrog_steps=NUM_LEAPFROG_STEPS,
                seed=seed,
            ),
            target_status_summary_fn=reference._lgssm_target_status_summary,
        )
        diagnostics = dict(run["diagnostics"])
        if diagnostics["health_passed"] is not True:
            configuration_vetoes.append(f"probe_health_failed:{index}")
        rows.append(
            {
                "grid_index": index,
                "step_size": step_size,
                "num_leapfrog_steps": NUM_LEAPFROG_STEPS,
                "trajectory_length": step_size * NUM_LEAPFROG_STEPS,
                "seed": seed,
                "acceptance_rate": diagnostics["acceptance_rate"],
                "health_passed": diagnostics["health_passed"],
                "diagnostics": diagnostics,
            }
        )
    selected = reference.select_tuning_candidate(rows)
    sequential = None
    if selected is not None:
        sequential = run_sequential_neutra_hmc(
            adapter=adapter,
            initial_state=initial_state,
            model_transform=adapter.forward,
            parameter_names=bundle.parameter_names,
            config=SequentialNeuTraHMCConfig(
                step_size=float(selected["step_size"]),
                num_leapfrog_steps=NUM_LEAPFROG_STEPS,
                warmup_seed=WARMUP_SEED,
                retained_seed=RETAINED_SEED,
                warmup_chunk_results=1000,
                warmup_min_results=2000,
                warmup_check_window_results=1000,
                warmup_max_results=10000,
                warmup_rhat_max=1.05,
                retained_chunk_results=2000,
                retained_min_results=4000,
                retained_max_results=10000,
                retained_rhat_max=1.01,
            ),
            retained_diagnostic_fn=lambda draws: reference.full_convergence_diagnostics(
                draws, parameter_names=bundle.parameter_names
            ),
            archive_callback=_archive_callback(COMPARATOR_ROOT / "samples"),
            target_status_summary_fn=reference._lgssm_target_status_summary,
        )
    convergence = (
        sequential["retained_checks"][-1].get("full_convergence")
        if isinstance(sequential, Mapping) and sequential["retained_checks"]
        else None
    )
    posterior = (
        _posterior_summary(
            samples=sequential["private_retained_raw"],
            parameter_names=bundle.parameter_names,
            truth=bundle.raw_truth,
        )
        if isinstance(sequential, Mapping)
        and sequential["retained_results_per_chain"] > 0
        else None
    )
    passed = bool(
        selected is not None
        and isinstance(sequential, Mapping)
        and sequential.get("passed") is True
        and isinstance(convergence, Mapping)
        and convergence.get("passed") is True
        and isinstance(posterior, Mapping)
        and posterior.get("all_finite") is True
        and posterior.get("recovery_passed") is True
    )
    public_sequential = (
        None
        if sequential is None
        else {
            key: value
            for key, value in sequential.items()
            if not key.startswith("private_")
        }
    )
    result_path = COMPARATOR_ROOT / "result.json"
    result = reference._with_artifact_hash(
        {
            "schema": "bayesfilter.neutra_robustness_f0_plain_hmc_comparator.v1",
            "passed": passed,
            "decision": (
                "ADMIT_F0_NEW_FIXTURE_PLAIN_HMC_COMPARATOR"
                if passed
                else "REJECT_F0_NEW_FIXTURE_PLAIN_HMC_COMPARATOR"
            ),
            "policy_id": NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
            "config": _file_reference(CONFIG_PATH),
            "fixture": _file_reference(FIXTURE_PATH),
            "mass": _file_reference(MASS_PATH),
            "target_signature": bundle.target_signature,
            "adapter_signature": adapter.adapter_signature(),
            "mass_artifact_hash": mass["artifact_hash"],
            "affine_coordinate_program": {
                "theta": "center + z @ factor.T",
                "latent_score": "raw_score @ factor",
                "constant_log_abs_det_included": True,
                "center_role": mass["center_role"],
            },
            "probe_rows": tuple(rows),
            "acceptance_role": "nomination_only",
            "selected_probe": selected,
            "configuration_vetoes": tuple(configuration_vetoes),
            "configuration_veto_scope": (
                "each health veto rejects only its fixed step-size configuration"
            ),
            "failed_wide_grid_attempt": _file_reference(
                FAILED_COMPARATOR_ROOT / "result.json"
            ),
            "sequential_run": public_sequential,
            "final_full_convergence": convergence,
            "posterior_summary": posterior,
            "runtime_manifest": _runtime_manifest(
                elapsed_seconds=time.monotonic() - started,
                output_paths=(result_path, COMPARATOR_ROOT / "samples"),
            ),
            "evidence_role": "new_fixture_tuned_plain_hmc_comparator_repair_attempt",
            "nonclaims": NONCLAIMS,
        }
    )
    reference._write_new_json(result_path, result)
    return result


def _load_inputs():
    xla = _read_mapping(F0_ROOT / "plain-hmc/xla_compile_gate.json")
    geometry = _read_mapping(F0_ROOT / "plain-hmc/geometry.json")
    mass = _read_mapping(MASS_PATH)
    if any(payload.get("passed") is not True or payload.get("vetoes") for payload in (xla, geometry, mass)):
        raise F0PlainHMCError("F0 XLA/geometry/mass precondition failed")
    if mass.get("artifact_hash") != EXPECTED_MASS_ARTIFACT_HASH:
        raise F0PlainHMCError("F0 mass artifact identity mismatch")
    bundle = load_deterministic_lgssm_exact_target(
        config_path=CONFIG_PATH,
        fixture_path=FIXTURE_PATH,
        expected_target_signature=EXPECTED_TARGET_SIGNATURE,
    )
    adapter = TensorAffineTargetAdapter(
        base_adapter=bundle.adapter,
        center=mass["center"],
        factor=mass["factor"],
        target_signature=bundle.target_signature,
        mass_artifact_hash=mass["artifact_hash"],
    )
    return bundle, mass, adapter


def _posterior_summary(
    *, samples: Any, parameter_names: Sequence[str], truth: Any
) -> Mapping[str, Any]:
    values = tf.convert_to_tensor(samples, tf.float64)
    names = tuple(str(item) for item in parameter_names)
    pooled = tf.reshape(values, (-1, 18))
    mean = tf.reduce_mean(pooled, axis=0)
    sd = tf.math.reduce_std(pooled, axis=0)
    half = int(values.shape[0]) // 2
    split = tf.reshape(
        tf.stack((values[:half], values[-half:]), axis=2),
        (half, 2 * int(values.shape[1]), 18),
    )
    mean_ess = tfp.mcmc.effective_sample_size(
        split, filter_beyond_positive_pairs=True, cross_chain_dims=1
    )
    mean_mcse = sd / tf.sqrt(mean_ess)
    truth_tensor = tf.convert_to_tensor(truth, tf.float64)
    recovery = tf.abs(mean - truth_tensor) / sd
    quantiles = tfp.stats.percentile(
        pooled, (5.0, 50.0, 95.0), axis=0, interpolation="linear"
    )
    all_finite = bool(
        tf.reduce_all(
            tf.math.is_finite(
                tf.concat((mean, sd, mean_ess, mean_mcse, recovery), axis=0)
            )
        ).numpy()
    )
    return {
        "all_finite": all_finite,
        "parameter_names": names,
        "truth": tuple(float(item) for item in truth_tensor.numpy().tolist()),
        "posterior_mean": tuple(float(item) for item in mean.numpy().tolist()),
        "posterior_sd": tuple(float(item) for item in sd.numpy().tolist()),
        "mean_ess": tuple(float(item) for item in mean_ess.numpy().tolist()),
        "mean_mcse": tuple(float(item) for item in mean_mcse.numpy().tolist()),
        "q05": tuple(float(item) for item in quantiles[0].numpy().tolist()),
        "q50": tuple(float(item) for item in quantiles[1].numpy().tolist()),
        "q95": tuple(float(item) for item in quantiles[2].numpy().tolist()),
        "recovery_passed": bool(
            all_finite and tf.reduce_all(recovery <= 3.0).numpy()
        ),
        "max_abs_mean_minus_truth_over_sd": float(tf.reduce_max(recovery).numpy()),
        "mean_mcse_definition": "posterior_sd / sqrt(split-chain cross-chain ESS)",
        "nonclaims": NONCLAIMS,
    }


def _archive_callback(root: Path):
    def archive(
        *, stage: str, chunk_index: int | None, latent_samples: tf.Tensor,
        model_samples: tf.Tensor, seed: tuple[int, int] | None, cumulative: bool,
    ) -> Mapping[str, Any]:
        suffix = "all" if cumulative else f"chunk_{int(chunk_index) + 1:04d}"
        metadata = {
            "stage": stage,
            "chunk_index": chunk_index,
            "seed": seed,
            "cumulative": cumulative,
        }
        return {
            "z": reference.write_tensor_archive(
                root / stage / f"{suffix}_z.tftensor",
                latent_samples,
                metadata={**metadata, "coordinate_system": "affine_latent_z"},
            ),
            "raw": reference.write_tensor_archive(
                root / stage / f"{suffix}_raw.tftensor",
                model_samples,
                metadata={**metadata, "coordinate_system": "raw_parameters"},
            ),
        }

    return archive


def _read_mapping(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise F0PlainHMCError(f"artifact must be a mapping: {path}")
    return value


def _file_reference(path: Path) -> Mapping[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "file_sha256": reference._file_sha256(path),
        "byte_count": path.stat().st_size,
    }


def _require_cpu_hidden() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise F0PlainHMCError("F0 comparator requires CUDA_VISIBLE_DEVICES=-1")


def _runtime_manifest(
    *, elapsed_seconds: float, output_paths: Sequence[Path]
) -> Mapping[str, Any]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return {
        "git_commit": commit,
        "command": (
            sys.executable,
            "docs/benchmarks/run_lgssm_new_fixture_plain_hmc_f0_2026_07_15.py",
        ),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "tensorflow_probability_version": tfp.__version__,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
        "hardware": "CPU with CUDA devices intentionally hidden",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "jit_compile": True,
        "dtype": "float64",
        "fixture_seed": (20260715, 701),
        "hmc_seeds": {
            "probe": PROBE_SEED,
            "warmup": WARMUP_SEED,
            "retained": RETAINED_SEED,
        },
        "elapsed_seconds": float(elapsed_seconds),
        "output_paths": tuple(str(path.relative_to(ROOT)) for path in output_paths),
        "plan_file": str(PLAN_PATH.relative_to(ROOT)),
        "result_file": str(output_paths[0].relative_to(ROOT)),
    }
