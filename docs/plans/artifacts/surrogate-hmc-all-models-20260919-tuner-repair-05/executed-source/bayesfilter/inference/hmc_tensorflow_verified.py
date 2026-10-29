"""Fresh verification and durable handoff owned by the public ordinary tuner.

The nomination remains a diagnostic result. This separate schema adds a frozen,
fresh four-chain verification and cannot upgrade old diagnostic artifacts.
Numerics use TensorFlow; JSON and tensor materialization are artifact boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any

import tensorflow as tf

from . import hmc_tensorflow_tuning as stage
from .hmc_posterior_diagnostics import rank_normalized_split_rhat
from .tuning_contract import HMC_TUNING_ORDINARY_RHAT_THRESHOLD, HMCTuningRunnerBinding

VERIFIED_TENSORFLOW_HMC_SCHEMA = "bayesfilter.verified_tensorflow_hmc.v1"
VERIFIED_TENSORFLOW_ARCHIVE_SCHEMA = "bayesfilter.verified_tensorflow_hmc_archive.v1"


@dataclass(frozen=True)
class VerifiedTensorFlowHMCConfig:
    nomination: stage.TensorFlowHMCKernelTuningConfig
    initial_chain_positions: tuple[tuple[float, ...], ...]
    initialization_provenance: str
    warmup_results: int
    verification_results: int
    fresh_seed: tuple[int, int]
    energy_error_max: float

    def __post_init__(self):
        cfg = self.nomination
        if not isinstance(cfg, stage.TensorFlowHMCKernelTuningConfig):
            raise TypeError("nomination must be a TensorFlow tuning config")
        if cfg.evidence_role != "diagnostic_candidate_screen" or not cfg.use_xla:
            raise ValueError("verified tuning requires XLA candidate nomination")
        for name in ("warmup_results", "verification_results"):
            object.__setattr__(self, name, stage._positive_int(getattr(self, name), name))
        if self.verification_results < 1000 or self.verification_results % 2:
            raise ValueError("fresh verification requires at least 1000 even draws")
        stage._nonempty(self.initialization_provenance, "initialization_provenance")
        stage._positive_finite(self.energy_error_max, "energy_error_max")
        positions = tuple(tuple(float(x) for x in row) for row in self.initial_chain_positions)
        if len(positions) != 4 or any(len(row) != cfg.parameter_dimension for row in positions):
            raise ValueError("initial_chain_positions must have shape [4, dimension]")
        if not all(math.isfinite(x) for row in positions for x in row):
            raise ValueError("initial_chain_positions must be finite")
        if any(len({row[j] for row in positions}) < 2 for j in range(cfg.parameter_dimension)):
            raise ValueError("fresh initialization must be dispersed in every coordinate")
        seed = tuple(self.fresh_seed)
        if len(seed) != 2 or any(type(x) is not int or not -(2**31) <= x < 2**31 for x in seed):
            raise ValueError("fresh_seed must contain two int32 values")
        if seed == cfg.seed:
            raise ValueError("fresh verification requires a separate seed")
        object.__setattr__(self, "initial_chain_positions", positions)
        object.__setattr__(self, "fresh_seed", seed)

    def payload(self):
        return dict(nomination=self.nomination.payload(),
                    initial_chain_positions=self.initial_chain_positions,
                    initialization_provenance=self.initialization_provenance,
                    warmup_results=self.warmup_results, verification_results=self.verification_results,
                    fresh_seed=self.fresh_seed, energy_error_max=self.energy_error_max)

    @classmethod
    def from_payload(cls, value):
        return cls(**{**value, "nomination": stage.TensorFlowHMCKernelTuningConfig.from_payload(value["nomination"])})


def _source_hashes():
    root = Path(__file__).parent
    return {name: stage._sha256(root / name) for name in (
        "hmc_tensorflow_verified.py", "hmc_tensorflow_tuning.py", "hmc_tuning_dispatch.py",
        "hmc_posterior_diagnostics.py", "tuning_contract.py", "neural_force_hmc.py")}


def _affine(nomination):
    return stage._TensorFlowAffineAdapter(
        stage._TensorFlowAffineTransform(nomination.center, nomination.factor),
        nomination.adapter_signature, nomination.config.target_scope)


def _chunk_graph(nomination, binding, count):
    kernel = stage._fixed_kernel(binding=binding, adapter=_affine(nomination),
                                step_size=nomination.step_size,
                                leapfrog_steps=nomination.num_leapfrog_steps,
                                config=nomination.config)
    @tf.function(input_signature=[tf.TensorSpec([4, nomination.config.parameter_dimension], tf.float64),
                                  tf.TensorSpec([2], tf.int32)], autograph=False, jit_compile=True)
    def run(state, seed):
        return stage._run_tensor_steps(kernel=kernel, initial_state=state, num_results=count,
                                       seed=seed, adaptive=False)
    return run


def _health(chunk, start, energy_max):
    """Host-invoked TF checks; XLA assertions are never admission guards."""
    finite = tf.reduce_all(tf.stack([
        tf.reduce_all(tf.math.is_finite(x)) for x in
        (chunk.states, chunk.log_accept_ratio, chunk.delta_h, chunk.final_state)]))
    # A declared support rejection is legitimate. Numerical endpoint failure is not.
    endpoint_valid = tf.logical_or(chunk.endpoint_status[..., 0] != 0,
                                   chunk.endpoint_status[..., 2] != 0)
    endpoint_ok = tf.reduce_all(tf.logical_and(endpoint_valid, chunk.endpoint_status[..., 8] == 0))
    moved = tf.reduce_all(tf.reduce_any(tf.not_equal(chunk.states, start[None, ...]), axis=[0, 2]))
    energy_ok = tf.reduce_all(tf.abs(chunk.delta_h) <= energy_max)
    return dict(finite=finite, endpoint_valid=endpoint_ok, all_chains_moved=moved,
                energy_valid=energy_ok, no_divergences=~tf.reduce_any(chunk.divergence))


@lru_cache(maxsize=16)
def _rhat_graph(count, dimension):
    # Rank ties have data-dependent segment counts; keep this diagnostic off XLA.
    @tf.function(input_signature=[tf.TensorSpec([count, 4, dimension], tf.float64)], autograph=False)
    def run(samples):
        return rank_normalized_split_rhat(tf.transpose(samples, [1, 0, 2]))
    return run


def _all_checks(checks):
    return all(bool(value) for value in checks.values())


def _save_chunk(root, prefix, chunk):
    return {name: {**stage._write_tensor(root / f"{prefix}-{name}.tfbin", value), "dtype": value.dtype.name}
            for name, value in zip(chunk._fields, chunk)}


def _read_chunk(root, records):
    values = {name: stage._read_tensor_record(root, record, tf.as_dtype(record["dtype"]))
              for name, record in records.items()}
    return stage._TensorChunk(**values)


def _fresh_checks(config, nomination, warmup, verification):
    affine = _affine(nomination)
    initial = affine.position_to_latent(tf.constant(config.initial_chain_positions, tf.float64))
    checks = {f"warmup_{name}": value for name, value in _health(warmup, initial, config.energy_error_max).items()}
    checks.update({f"verification_{name}": value for name, value in
                   _health(verification, warmup.final_state, config.energy_error_max).items()})
    checks["nomination_passed"] = nomination.heuristic_screen_passed
    samples = affine.latent_to_position(verification.states)
    finite_samples = bool(tf.reduce_all(tf.math.is_finite(samples)))
    if finite_samples:
        rhat = _rhat_graph(config.verification_results, config.nomination.parameter_dimension)(samples)
    else:
        rhat = {key: tf.fill([config.nomination.parameter_dimension], tf.constant(float("inf"), tf.float64))
                for key in ("bulk", "folded", "maximum")}
    checks["fresh_rhat"] = tf.reduce_all(tf.logical_and(tf.math.is_finite(rhat["maximum"]),
                                             rhat["maximum"] <= HMC_TUNING_ORDINARY_RHAT_THRESHOLD))
    return checks, rhat


def _validate_nomination(nomination):
    """Guard numerics outside XLA before constructing a fresh kernel."""
    for name in ("center", "factor", "covariance", "metric_eigenvalues", "step_size"):
        tf.debugging.assert_all_finite(getattr(nomination, name), f"nomination {name}")
    tf.debugging.assert_positive(nomination.step_size)
    tf.debugging.assert_positive(nomination.metric_eigenvalues)
    tf.debugging.assert_positive(tf.linalg.diag_part(nomination.factor))
    tf.debugging.assert_near(nomination.factor, tf.linalg.band_part(nomination.factor, -1, 0))
    tf.debugging.assert_near(nomination.covariance,
                            tf.linalg.matmul(nomination.factor, nomination.factor, transpose_b=True))
    if int(nomination.num_leapfrog_steps) not in nomination.config.trajectory_candidates:
        raise ValueError("selected trajectory is outside the declared grid")


@dataclass(frozen=True)
class VerifiedTensorFlowHMCResult:
    config: VerifiedTensorFlowHMCConfig
    nomination: stage.TensorFlowHMCKernelTuningResult
    runner_binding: HMCTuningRunnerBinding
    artifact_manifest_path: str
    passed: bool
    checks: dict

    @property
    def admission_supported(self):
        return True

    @property
    def handoff_eligible(self):
        return self.passed


def _run_verified_tensorflow_tuning(*, adapter, initial_position, config, output_dir,
                                    parameter_scales, runner_binding):
    if output_dir is None:
        raise ValueError("verified tuning requires a durable output directory")
    root = Path(output_dir).resolve()
    if root.exists():
        raise FileExistsError("verified tuning output directory must be new")
    root.mkdir(parents=True)
    # The same public tuner owns nomination and fresh verification.
    nomination = stage._run_tensorflow_hmc_tuning(
        adapter=adapter, initial_position=initial_position, config=config.nomination,
        output_dir=root / "nomination", parameter_scales=parameter_scales, runner_binding=runner_binding)
    if not bool(nomination.heuristic_screen_passed):
        stage._write_json(root / "rejected.json", dict(reason="nomination_failed", handoff_eligible=False))
        raise ValueError("TensorFlow nomination failed; preserved diagnostic cannot issue handoff")
    _validate_nomination(nomination)
    initial = _affine(nomination).position_to_latent(tf.constant(config.initial_chain_positions, tf.float64))
    warmup = _chunk_graph(nomination, runner_binding, config.warmup_results)(initial, tf.constant(config.fresh_seed))
    warm_records = _save_chunk(root, "warmup", warmup)
    verification_seed = tf.random.experimental.stateless_fold_in(tf.constant(config.fresh_seed), 1)
    verification = _chunk_graph(nomination, runner_binding, config.verification_results)(warmup.final_state, verification_seed)
    verification_records = _save_chunk(root, "verification", verification)
    checks, rhat = _fresh_checks(config, nomination, warmup, verification)
    rhat_records = {key: stage._write_tensor(root / f"rhat-{key}.tfbin", value) for key, value in rhat.items()}
    passed = _all_checks(checks)
    manifest = root / "manifest.json"
    stage._write_json(manifest, dict(
        schema=VERIFIED_TENSORFLOW_HMC_SCHEMA, artifact_authority=True,
        handoff_eligible=passed, passed=passed, config=config.payload(),
        adapter_signature=nomination.adapter_signature, runner_binding=runner_binding.payload(),
        source_hashes=_source_hashes(), numerical_backend="tensorflow_only", jit_compile=True,
        rhat_jit_compile=False, rhat_threshold=HMC_TUNING_ORDINARY_RHAT_THRESHOLD,
        nomination_manifest_sha256=stage._sha256(Path(nomination.artifact_manifest_path)),
        warmup=warm_records, verification=verification_records, rhat=rhat_records,
        checks={k: bool(v) for k, v in checks.items()}, posterior_convergence_claim=False))
    return VerifiedTensorFlowHMCResult(config, nomination, runner_binding, str(manifest), passed,
                                       {k: bool(v) for k, v in checks.items()})


def load_verified_tensorflow_hmc_result(manifest_path, *, adapter, runner_binding):
    path = Path(manifest_path).resolve()
    payload = json.loads(path.read_text())
    if payload.get("schema") != VERIFIED_TENSORFLOW_HMC_SCHEMA:
        raise ValueError("unsupported verified tuning schema")
    if (payload.get("artifact_authority") is not True
            or payload.get("jit_compile") is not True
            or payload.get("numerical_backend") != "tensorflow_only"
            or payload.get("rhat_threshold") != HMC_TUNING_ORDINARY_RHAT_THRESHOLD):
        raise ValueError("invalid verified tuning authority or diagnostic policy")
    if payload.get("source_hashes") != _source_hashes():
        raise ValueError("verified tuning source drift")
    if payload.get("runner_binding") != json.loads(json.dumps(runner_binding.payload())):
        raise ValueError("verified tuning binding mismatch")
    if payload.get("adapter_signature") != stage._adapter_signature(adapter):
        raise ValueError("verified tuning adapter mismatch")
    nomination_path = path.parent / "nomination" / "manifest.json"
    if stage._sha256(nomination_path) != payload["nomination_manifest_sha256"]:
        raise ValueError("nomination artifact checksum mismatch")
    config = VerifiedTensorFlowHMCConfig.from_payload(payload["config"])
    nomination = stage.load_tensorflow_hmc_tuning_result(nomination_path, adapter=adapter, runner_binding=runner_binding)
    _validate_nomination(nomination)
    if nomination.config.payload() != config.nomination.payload():
        raise ValueError("nomination configuration mismatch")
    warmup = _read_chunk(path.parent, payload["warmup"])
    verification = _read_chunk(path.parent, payload["verification"])
    if warmup.states.shape != (config.warmup_results, 4, config.nomination.parameter_dimension):
        raise ValueError("warmup archive shape mismatch")
    if verification.states.shape != (config.verification_results, 4, config.nomination.parameter_dimension):
        raise ValueError("verification archive shape mismatch")
    for chunk in (warmup, verification):
        tf.debugging.assert_equal(chunk.final_state, chunk.states[-1], message="chunk final-state mismatch")
    checks, rhat = _fresh_checks(config, nomination, warmup, verification)
    for key, value in rhat.items():
        archived = stage._read_tensor_record(path.parent, payload["rhat"][key], tf.float64)
        tf.debugging.assert_near(archived, value, atol=1e-12, rtol=1e-12)
    checked = {k: bool(v) for k, v in checks.items()}
    passed = _all_checks(checks)
    if payload.get("checks") != checked or payload.get("passed") != passed or payload.get("handoff_eligible") != passed:
        raise ValueError("verified admission decision mismatch")
    return VerifiedTensorFlowHMCResult(config, nomination, runner_binding, str(path), passed, checked)


@dataclass(frozen=True)
class VerifiedTensorFlowHMCRunner:
    tuning_result: VerifiedTensorFlowHMCResult
    runner_binding: HMCTuningRunnerBinding

    def run(self, config: stage.BoundRetainedHMCArchiveConfig):
        original = self.tuning_result
        class Adapter:
            target_scope = original.config.nomination.target_scope
            def adapter_signature(self):
                return original.nomination.adapter_signature
        tuning = load_verified_tensorflow_hmc_result(original.artifact_manifest_path,
                                                     adapter=Adapter(), runner_binding=self.runner_binding)
        if not tuning.passed:
            raise ValueError("fresh verification did not pass")
        manifest = json.loads(Path(tuning.artifact_manifest_path).read_text())
        verification = _read_chunk(Path(tuning.artifact_manifest_path).parent, manifest["verification"])
        state = verification.final_state
        used_seeds = [list(tuning.config.nomination.seed), list(tuning.config.fresh_seed)]
        used_seeds.append(tf.random.experimental.stateless_fold_in(tf.constant(tuning.config.fresh_seed), 1).numpy().tolist())
        parent = None
        if config.continuation_manifest is not None:
            parent = Path(config.continuation_manifest).resolve()
            previous = json.loads(parent.read_text())
            if (previous.get("schema") != VERIFIED_TENSORFLOW_ARCHIVE_SCHEMA
                    or previous.get("tuning_manifest_sha256") != stage._sha256(Path(tuning.artifact_manifest_path))
                    or previous.get("binding_hash") != self.runner_binding.binding_hash
                    or not previous.get("health_passed")):
                raise ValueError("invalid retained continuation")
            prior = _read_chunk(parent.parent, previous["chunk"])
            prior_initial = stage._read_tensor_record(parent.parent, previous["initial_state"], tf.float64)
            prior_checks = _health(prior, prior_initial, tuning.config.energy_error_max)
            if not _all_checks(prior_checks) or previous.get("checks") != {k:bool(v) for k,v in prior_checks.items()}:
                raise ValueError("retained continuation health mismatch")
            tf.debugging.assert_equal(prior.final_state, prior.states[-1], message="continuation final-state mismatch")
            state = prior.final_state
            used_seeds = previous["used_seeds"]
        if list(config.seed) in used_seeds:
            raise ValueError("retained seed already used")
        root = Path(config.output_dir).resolve()
        if root.exists():
            raise FileExistsError("retained output must be new")
        root.mkdir(parents=True)
        chunk = _chunk_graph(tuning.nomination, self.runner_binding, config.num_results)(state, tf.constant(config.seed))
        checks = _health(chunk, state, tuning.config.energy_error_max)
        raw = _affine(tuning.nomination).latent_to_position(chunk.states)
        records = _save_chunk(root, "retained", chunk)
        stage._write_json(root / "manifest.json", dict(
            schema=VERIFIED_TENSORFLOW_ARCHIVE_SCHEMA, chunk=records,
            raw_samples=stage._write_tensor(root / "samples.tfbin", raw),
            initial_state=stage._write_tensor(root / "initial-state.tfbin", state),
            tuning_manifest=str(tuning.artifact_manifest_path),
            tuning_manifest_sha256=stage._sha256(Path(tuning.artifact_manifest_path)),
            binding_hash=self.runner_binding.binding_hash, config=config.payload(),
            used_seeds=used_seeds + [list(config.seed)], continuation_manifest=str(parent) if parent else None,
            health_passed=_all_checks(checks), checks={k: bool(v) for k, v in checks.items()},
            jit_compile=True, numerical_backend="tensorflow_only", posterior_convergence_claim=False))
        return dict(archive_manifest_path=str(root / "manifest.json"), health_passed=_all_checks(checks))


def _build_verified_runner(tuning_result, runner_binding):
    if tuning_result.runner_binding is not runner_binding:
        raise ValueError("retained runner requires the verified binding")
    if not tuning_result.handoff_eligible:
        raise ValueError("fresh verification did not pass")
    return VerifiedTensorFlowHMCRunner(tuning_result, runner_binding)
