"""Repository TF/TFP observations and frozen geometry for candidate-set HMC.

The host owns scheduling, evidence and persistence. Numerical transitions use
the same independent-chain TensorFlow runner during tuning and retained replay.
No caller observation callback can supply this module's executable binding.
"""
from __future__ import annotations

import base64
import hashlib
import inspect
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

from bayesfilter.inference.hmc_candidate_set_tuning import (
    HMCCandidateSetScope, HMCTuningCandidateRecord, HMCWorkItem, _sha256,
    HMCInfrastructureFailure, HMCSharedInvalidity,
)
from bayesfilter.inference.hmc_verification import HMCAcceptancePolicy
from bayesfilter.inference.hmc_candidate_decisions import HMCCandidateDecision, HMCCandidateExecutionFailure
from bayesfilter.inference.hmc_candidate_runtime import chunk_seed, before_numerical_chunk

EXECUTION_SCHEMA = "bayesfilter.hmc_candidate_execution.v1"
_ISSUER = object()


def _report_rhat(samples: Any) -> Mapping[str, Any]:
    """Optional diagnostics never control tuning or erase numerical evidence."""
    import math
    from bayesfilter.inference.hmc import _rhat_summary_from_retained_samples
    from bayesfilter.inference.tuning_contract import HMC_TUNING_ORDINARY_RHAT_THRESHOLD
    try:
        result = _rhat_summary_from_retained_samples(samples, threshold=HMC_TUNING_ORDINARY_RHAT_THRESHOLD)
        def finite(value):
            if isinstance(value, Mapping):
                return {k: finite(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [finite(v) for v in value]
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value
        return {**finite(result), "role": "reporting_only"}
    except Exception as exc:
        return {"role": "reporting_only", "unavailable_reason": type(exc).__name__ + ": " + str(exc)}


def _runtime_policy() -> Mapping[str, Any]:
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    devices = tf.config.list_physical_devices("GPU")
    if devices and os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
        raise ValueError("GPU candidate execution requires TF_FORCE_GPU_ALLOW_GROWTH=true before import")
    policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=False)
    return {"device_type": "GPU" if devices else "CPU", "memory_policy": policy,
            "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def _tensor_payload(value: Any) -> Mapping[str, Any]:
    import tensorflow as tf

    tensor = tf.convert_to_tensor(value)
    with tf.device("/CPU:0"):
        encoded = tf.io.serialize_tensor(tensor).numpy()
    return {"dtype": tensor.dtype.name, "shape": tensor.shape.as_list(),
            "tensor": base64.b64encode(encoded).decode("ascii"),
            "sha256": hashlib.sha256(encoded).hexdigest()}


def _tensor_from_payload(payload: Mapping[str, Any]) -> Any:
    import tensorflow as tf

    encoded = base64.b64decode(payload["tensor"], validate=True)
    if hashlib.sha256(encoded).hexdigest() != payload["sha256"]:
        raise ValueError("tensor checksum mismatch")
    with tf.device("/CPU:0"):
        tensor = tf.io.parse_tensor(encoded, out_type=tf.as_dtype(payload["dtype"]))
    if tensor.shape.as_list() != payload["shape"]:
        raise ValueError("tensor shape mismatch")
    return tensor


def _trace_payload(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    return {key: _trace_payload(value) if isinstance(value, Mapping)
            else _tensor_payload(value) for key, value in trace.items()}


def _trace_from_payload(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    return {key: _tensor_from_payload(value) if "tensor" in value
            else _trace_from_payload(value) for key, value in trace.items()}


def _positive_int(value: Any, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _seed(value: Any) -> tuple[int, int]:
    value = tuple(value)
    if len(value) != 2 or any(type(i) is not int or not -(2**31) <= i < 2**31 for i in value):
        raise ValueError("seed must contain two int32 integers")
    return value


@dataclass(frozen=True)
class HMCCandidateExecutionConfig:
    """Explicit evidence budgets; XLA is the normal numerical execution mode."""

    measurement_num_results: int
    verification_num_results: int
    num_warmup_steps: int
    seed: tuple[int, int]
    acceptance_policy: HMCAcceptancePolicy
    target_status_trace_policy: str
    use_xla: bool = True
    non_xla_reason: str | None = None
    chain_mode: str = "serial"
    chunk_max_results: int = 256
    preparation_elapsed_seconds: float = 0.0
    pilot_num_results: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.acceptance_policy, HMCAcceptancePolicy):
            raise TypeError("acceptance_policy must be HMCAcceptancePolicy")
        for name in ("measurement_num_results", "verification_num_results"):
            _positive_int(getattr(self, name), name, self.acceptance_policy.min_decisions_per_chain)
        if self.pilot_num_results is not None:
            _positive_int(self.pilot_num_results, "pilot_num_results", self.acceptance_policy.min_decisions_per_chain)
        _positive_int(self.num_warmup_steps, "num_warmup_steps", 0)
        _positive_int(self.chunk_max_results, "chunk_max_results")
        import math
        if not math.isfinite(self.preparation_elapsed_seconds) or self.preparation_elapsed_seconds < 0:
            raise ValueError("preparation_elapsed_seconds must be finite and nonnegative")
        object.__setattr__(self, "seed", _seed(self.seed))
        if type(self.use_xla) is not bool:
            raise TypeError("use_xla must be boolean")
        if not self.use_xla and not str(self.non_xla_reason or "").strip():
            raise ValueError("non-XLA execution requires an explicit non_xla_reason")
        if self.target_status_trace_policy not in {"none", "per_chain_step"}:
            raise ValueError("declare none or per_chain_step target status")
        if self.chain_mode not in {"serial", "threaded"}:
            raise ValueError("chain_mode must be serial or threaded")

    def payload(self) -> Mapping[str, Any]:
        payload = {**asdict(self), "acceptance_policy": self.acceptance_policy.payload()}
        if self.pilot_num_results is None:
            payload.pop("pilot_num_results")
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "HMCCandidateExecutionConfig":
        values = dict(payload)
        policy = values.pop("acceptance_policy")
        values["acceptance_policy"] = HMCAcceptancePolicy(**{
            f.name: policy[f.name] for f in fields(HMCAcceptancePolicy)
        })
        result = cls(**values)
        if _json_copy(result.payload()) != _json_copy(payload):
            raise ValueError("execution policy metadata mismatch")
        return result


def _source_closure(adapter: Any, source_paths: Sequence[str | Path]) -> Mapping[str, str]:
    # Include numerical, coordinate, evidence and replay implementations. The
    # consumer must include data/preparation dependencies beyond its adapter file.
    names = ("hmc", "hmc_budget_ladder", "hmc_kernel_tuning", "hmc_artifact_identity",
             "mass_matrix", "hmc_tuning", "fixed_l_finite_bracket", "hmc_coordinates",
             "hmc_warmup", "hmc_tuning_state", "hmc_kernel_selection", "hmc_diagnostics",
             "hmc_candidate_set_execution", "hmc_candidate_set_retained",
             "hmc_candidate_set_adapters", "hmc_candidate_set_tuning",
             "hmc_candidate_set_checkpoint", "hmc_candidate_set_public", "hmc_candidate_set_position_field",
             "hmc_candidate_decisions", "hmc_candidate_proposals", "hmc_candidate_runtime", "hmc_preparation",
             "hmc_candidate_set_artifacts", "hmc_verification", "hmc_convergence",
             "hmc_diagnostic_math", "hmc_posterior_diagnostics", "hmc_precision", "hmc_posterior_assessment", "neutra_hmc",
             "tuning_contract", "hmc_tuning_dispatch", "fixed_transport_hmc_tuning_tf",
             "posterior_adapter", "batched_value_score",
             "neutra_artifacts", "fixed_transport_hmc_mechanics_tf")
    paths = {Path(__file__).with_name(name + ".py").resolve() for name in names}
    paths.add(Path(__file__).parents[1] / "runtime" / "gpu_memory_policy.py")
    paths.update(Path(__file__).parents[1] / (name + ".py") for name in (
        "hmc_route_contract", "hmc_ordinary_selection_policy", "hmc_budget_contract",
        "runtime/runner", "runtime/selection"))
    if not source_paths:
        raise ValueError("source_paths must include the target's actual source dependencies")
    paths.update(Path(path).resolve() for path in source_paths)
    adapter_file = inspect.getsourcefile(type(adapter))
    if not adapter_file:
        raise ValueError("the target adapter must have inspectable source")
    paths.add(Path(adapter_file).resolve())
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def _check_sources(closure: Mapping[str, str]) -> None:
    for path, digest in closure.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"execution source changed: {path}")


def _rebuild_geometry(adapter: Any, layers: Sequence[Mapping[str, Any]], target_scope: str) -> tuple[Any, tuple[Any, ...]]:
    from bayesfilter.inference.hmc import PrecomputedMassArtifact, stable_adapter_signature
    from bayesfilter.inference.hmc_artifact_identity import mass_artifact_signature
    from bayesfilter.inference.hmc_budget_ladder import build_fixed_mass_hmc_adapter

    transforms = []
    for layer in layers:
        kind = layer["kind"]
        if kind in {"fixed_mass", "bootstrap_mass"}:
            mass = PrecomputedMassArtifact.from_payload(
                layer["artifact"], expected_adapter_signature=stable_adapter_signature(adapter))
            if kind == "bootstrap_mass":
                from bayesfilter.inference.hmc_kernel_tuning import _build_bootstrap_fixed_mass_adapter
                adapter = _build_bootstrap_fixed_mass_adapter(
                    adapter=adapter, mass_artifact=mass,
                    mass_signature=mass_artifact_signature(mass), target_scope=target_scope)
            else:
                adapter = build_fixed_mass_hmc_adapter(
                    adapter=adapter, mass_artifact=mass, target_scope=target_scope)
        elif kind == "frozen_transport" and len(layers) == 1:
            from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
            from bayesfilter.inference.fixed_transport_hmc_mechanics_tf import build_fixed_transport_value_score_adapter
            from bayesfilter.inference.posterior_adapter import value_score_capability
            loaded = load_frozen_neutra_artifact(
                layer["artifact"], expected_target_signature=stable_adapter_signature(adapter))
            capability = value_score_capability(adapter)
            adapter = build_fixed_transport_value_score_adapter(
                base_adapter=adapter, fixed_transport=loaded.transport, target_scope=target_scope,
                evidence_path=capability.evidence_path,
                xla_hmc_ready=capability.is_accepted_xla_hmc_authority,
                full_chain_xla_diagnostic_ready=capability.is_accepted_full_chain_xla_diagnostic_authority)
        else:
            raise ValueError("unsupported frozen geometry")
        transforms.append(adapter)
    if not transforms:
        raise ValueError("explicit frozen geometry is required")
    return adapter, tuple(transforms)


def _probe(adapter: Any, state: Any, *, target_status: bool = False) -> Mapping[str, Any]:
    import tensorflow as tf

    values, scores = zip(*(adapter.log_prob_and_grad(row) for row in tf.unstack(state)))
    values, scores = tf.stack(values), tf.stack(scores)
    if values.shape != (4,) or scores.shape != state.shape:
        raise ValueError("target value/score shape mismatch")
    if not bool(tf.reduce_all(tf.math.is_finite(values))) or not bool(tf.reduce_all(tf.math.is_finite(scores))):
        raise ValueError("initial target value/score must be finite")
    result = {"values": _tensor_payload(values), "scores": _tensor_payload(scores)}
    if target_status:
        from bayesfilter.inference.hmc_verification import target_status_telemetry_has_failure
        telemetry = adapter.target_status_telemetry(state)
        if target_status_telemetry_has_failure(telemetry, expected_shape=(4,)):
            raise ValueError("initial target status failed")
        result["target_status"] = _trace_payload({key: value for key, value in telemetry.items() if tf.is_tensor(value)})
    return result


class HMCCandidateExecutionBinding:
    """Issued numerical evaluator, frozen state and durable evidence owner.

    Use the factories below. ``typed_adapter`` enters the shared controller;
    the binding itself is passed to the retained builders. Callable labels or
    controller flags alone never grant numerical replay authority.
    """

    def __init__(self, token: Any, *, adapter: Any, spec: Mapping[str, Any]) -> None:
        if token is not _ISSUER:
            raise ValueError("execution bindings must be repository-issued")
        import tensorflow as tf
        import tensorflow_probability as tfp
        from bayesfilter.inference.hmc import stable_adapter_signature
        from bayesfilter.inference.posterior_adapter import value_score_capability

        self._base_adapter = adapter
        self._runtime = _json_copy(_runtime_policy())
        self._spec = _json_copy(spec)
        self.config = HMCCandidateExecutionConfig.from_payload(spec["config"])
        self.scope = HMCCandidateSetScope.from_payload(spec["scope"])
        self._active_adapter, self._transforms = _rebuild_geometry(adapter, spec["layers"], spec["target_scope"])
        self._geometry_state_hash = _sha256({"geometry": self._geometry_state()})
        capability = value_score_capability(self._active_adapter)
        if capability.value_score_authority not in {
            "graph_native", "analytical_manual", "reviewed_gradient_tape_xla_exception",
        }:
            raise ValueError("candidate execution requires an exact TensorFlow value/score authority")
        if capability.target_scope is not None and capability.target_scope != spec["target_scope"]:
            raise ValueError("target scope mismatch")
        if self.config.use_xla and not capability.is_accepted_full_chain_xla_diagnostic_authority:
            raise ValueError("target lacks full-chain XLA qualification")
        self.initial_active_state = _tensor_from_payload(spec["initial_active_state"])
        if (self.initial_active_state.dtype != tf.float64 or self.initial_active_state.shape.rank != 2
                or self.initial_active_state.shape[0] != 4 or self.initial_active_state.shape[1] < 1
                or not bool(tf.reduce_all(tf.math.is_finite(self.initial_active_state)))):
            raise ValueError("initial state must be a finite float64 [4, parameter] bank")
        if spec["versions"] != {"tensorflow": tf.__version__, "tensorflow_probability": tfp.__version__}:
            raise ValueError("TensorFlow/TFP version mismatch")
        if self._runtime != self._spec["runtime_policy"]:
            raise ValueError("execution device or numerical policy mismatch")
        if stable_adapter_signature(adapter) != self.scope.target_signature:
            raise ValueError("base adapter identity mismatch")
        if stable_adapter_signature(self._active_adapter) != self.scope.adapter_signature:
            raise ValueError("active adapter identity mismatch")
        if asdict(capability) != spec["capability"] and _json_copy(asdict(capability)) != spec["capability"]:
            raise ValueError("target capability mismatch")
        self.binding_hash = _sha256(self._spec)
        self._evidence: dict[str, Mapping[str, Any]] = {}
        self._analysis_cache: dict[str, Mapping[str, Any]] = {}
        self._persisted_files: dict[str, tuple[int, int]] = {}
        self._partial: dict[str, list[Mapping[str, Any]]] = {}
        self._checkpoint_callback = None
        self._deadline = None
        self._runners: dict[tuple[int, int], Any] = {}
        self.validate()

    @property
    def fixed_transport(self) -> Any:
        """The reconstructed transport for the fixed-transport public dispatcher."""
        if self._spec["adapter_kind"] != "fixed_transport":
            raise ValueError("ordinary execution has no nonlinear transport")
        return self._active_adapter.transport

    def validate_transport(self, transport: Any) -> None:
        if not callable(getattr(transport, "manifest_payload", None)) or _json_copy(
                transport.manifest_payload()) != _json_copy(self.fixed_transport.manifest_payload()):
            raise ValueError("dispatcher transport differs from execution binding")

    @property
    def typed_adapter(self) -> Any:
        from bayesfilter.inference.hmc_candidate_set_adapters import issue_hmc_candidate_set_adapter

        return replace(issue_hmc_candidate_set_adapter(
            scope=self.scope, adapter_kind=self._spec["adapter_kind"], observe=self.observe,
            source_dependency_closure=self._spec["source_closure"],
            target_preparation_identity=self.scope.target_preparation_identity,
            transition_identity=self.scope.transition_identity,
            qualification_status="qualified"), _execution_binding=self)

    def validate(self) -> None:
        from bayesfilter.inference.hmc import stable_adapter_signature
        from bayesfilter.inference.posterior_adapter import value_score_capability

        if _sha256(self._spec) != self.binding_hash:
            raise ValueError("execution binding mutated")
        if _json_copy(self.scope.payload()) != self._spec["scope"] or _json_copy(self.config.payload()) != self._spec["config"]:
            raise ValueError("execution scope or policy mutated")
        _check_sources(self._spec["source_closure"])
        if _sha256({"geometry": self._geometry_state()}) != self._geometry_state_hash:
            raise ValueError("frozen execution geometry changed")
        if stable_adapter_signature(self._base_adapter) != self.scope.target_signature:
            raise ValueError("target adapter changed")
        if _json_copy(asdict(value_score_capability(self._active_adapter))) != self._spec["capability"]:
            raise ValueError("target capability changed")
        if _tensor_payload(self.initial_active_state) != self._spec["initial_active_state"]:
            raise ValueError("initial state changed")
        if _json_copy(_runtime_policy()) != self._runtime:
            raise ValueError("execution device or numerical policy changed")
        if _probe(self._active_adapter, self.initial_active_state,
                  target_status=self.config.target_status_trace_policy == "per_chain_step") != self._spec["probe"]:
            raise ValueError("target or geometry changed at the bound start bank")

    def _geometry_state(self) -> Any:
        """Snapshot repository-owned geometry attributes, including actual tensors.

        This detects accidental mutation hidden by cached manifest labels. It
        deliberately does not traverse or serialize arbitrary target callables.
        """
        import tensorflow as tf
        def encode(value: Any) -> Any:
            if tf.is_tensor(value) or isinstance(value, tf.Variable) or (hasattr(value, "dtype") and hasattr(value, "shape")):
                return _tensor_payload(value)
            if isinstance(value, Mapping):
                return {str(k): encode(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [encode(v) for v in value]
            if hasattr(value, "__dict__"):
                return {"class": type(value).__qualname__, **{k: encode(v) for k, v in vars(value).items()}}
            return value
        return [encode(layer.transform if hasattr(layer, "transform") else layer.transport)
                for layer in self._transforms]

    def validate_dispatch_inputs(self, adapter: Any, initial_position: Any) -> None:
        from bayesfilter.inference.hmc import stable_adapter_signature

        self.validate()
        if stable_adapter_signature(adapter) != self.scope.target_signature:
            raise ValueError("dispatcher target differs from execution binding")
        if _tensor_payload(initial_position) != self._spec["initial_active_state"]:
            raise ValueError("typed dispatcher requires the binding's initial_active_state")

    def position_samples(self, active_samples: Any) -> Any:
        import tensorflow as tf

        state = tf.convert_to_tensor(active_samples, dtype=tf.float64)
        shape = state.shape
        state = tf.reshape(state, [-1, shape[-1]])
        for transform in reversed(self._transforms):
            state = transform.latent_to_position(state)
        return tf.reshape(state, shape)

    def work_seed(self, work: HMCWorkItem) -> tuple[int, int]:
        digest = hashlib.sha256(json.dumps({
            "seed": self.config.seed, "scope": self.scope.payload(),
            "work_id": work.work_item_id, "stage": work.stage,
            "candidate_hash": work.candidate_record_hash,
        }, sort_keys=True).encode()).digest()
        return tuple(int.from_bytes(digest[i:i+4], "big") & 0x7fffffff for i in (0, 4))

    def _run(self, candidate: HMCTuningCandidateRecord, state: Any, count: int, seed: tuple[int, int]) -> Any:
        from bayesfilter.inference.hmc import FullChainHMCConfig, build_independent_chain_tfp_hmc_runner

        HMCTuningCandidateRecord.from_payload(self.scope, candidate.payload())
        key = (candidate.leapfrog_steps, count)
        if key not in self._runners:
            config = FullChainHMCConfig(
                num_results=count, num_burnin_steps=0, step_size=candidate.epsilon,
                num_leapfrog_steps=candidate.leapfrog_steps, seed=seed,
                use_xla=self.config.use_xla, target_scope=self._spec["target_scope"],
                target_status_trace_policy=self.config.target_status_trace_policy,
                capture_candidate_health=True)
            self._runners[key] = build_independent_chain_tfp_hmc_runner(
                self._active_adapter, self.initial_active_state, config)
        return self._runners[key].run(current_state=state, root_seed=seed,
                                      step_size=candidate.epsilon, mode=self.config.chain_mode)

    def health_failures(self, initial: Any, samples: Any, trace: Mapping[str, Any]) -> tuple[str, ...]:
        import tensorflow as tf
        from bayesfilter.inference.hmc_verification import target_status_telemetry_has_failure

        shape = tuple(samples.shape[:2])
        if samples.dtype != tf.float64 or samples.shape.rank != 3 or tuple(samples.shape[1:]) != tuple(self.initial_active_state.shape):
            raise ValueError("candidate sample shape/dtype mismatch")
        failures = []
        for key in ("is_accepted", "log_accept_ratio", "target_log_prob", "proposed_target_log_prob", "target_score_finite"):
            if key not in trace or tuple(trace[key].shape) != shape:
                raise ValueError(f"missing or misaligned health trace: {key}")
        if trace["is_accepted"].dtype != tf.bool or trace["target_score_finite"].dtype != tf.bool:
            raise ValueError("health and acceptance bits must be boolean")
        if "proposed_state" not in trace or trace["proposed_state"].shape != samples.shape:
            raise ValueError("missing or misaligned proposed states")
        previous = tf.concat([initial[None], samples[:-1]], axis=0)
        arrays = {"state": samples, "initial_state": initial, "proposal": trace["proposed_state"],
                  "proposal_displacement": trace["proposed_state"] - previous,
                  **{key: trace[key] for key in ("log_accept_ratio", "target_log_prob", "proposed_target_log_prob")}}
        for key in ("initial_momentum", "final_momentum"):
            if key not in trace or trace[key].shape != samples.shape:
                raise ValueError("missing or misaligned momentum health trace")
            arrays[key] = trace[key]
        if "log_acceptance_correction" in trace:
            arrays["log_acceptance_correction"] = trace["log_acceptance_correction"]
        for key, array in arrays.items():
            if not bool(tf.reduce_all(tf.math.is_finite(array))):
                failures.append("nonfinite_" + key)
        if not bool(tf.reduce_all(trace["target_score_finite"])):
            failures.append("nonfinite_target_score")
        # This invariant also checks the first transition and rejected proposals.
        expected = tf.where(trace["is_accepted"][..., None], trace["proposed_state"], previous)
        if not bool(tf.reduce_all(tf.equal(expected, samples))):
            failures.append("metropolis_state_mismatch")
        if "divergence" in trace:
            if tuple(trace["divergence"].shape) != shape or trace["divergence"].dtype != tf.bool:
                raise ValueError("invalid native divergence trace")
            if bool(tf.reduce_any(trace["divergence"])):
                failures.append("native_divergence_positive")
        if self.config.target_status_trace_policy == "per_chain_step":
            for key in ("target_status_telemetry", "proposed_target_status_telemetry"):
                if target_status_telemetry_has_failure(trace.get(key, {}), expected_shape=shape):
                    failures.append(key + "_failed")
        return tuple(failures)

    def analyze(self, initial: Any, samples: Any, trace: Mapping[str, Any]) -> Mapping[str, Any]:
        import tensorflow as tf
        from bayesfilter.inference.hmc_verification import evaluate_hmc_acceptance_evidence

        failures = self.health_failures(initial, samples, trace)
        warmup = self.config.num_warmup_steps
        divergence = trace.get("divergence")
        evidence = evaluate_hmc_acceptance_evidence(
            samples=samples[warmup:], log_accept_ratio=trace["log_accept_ratio"][warmup:],
            is_accepted=trace["is_accepted"][warmup:], policy=self.config.acceptance_policy,
            target_log_prob=trace["target_log_prob"][warmup:],
            native_divergence_status="available" if divergence is not None else "not_exposed_by_kernel",
            native_divergence_count=int(tf.reduce_sum(tf.cast(divergence, tf.int32))) if divergence is not None else None)
        decision = HMCCandidateDecision.from_evidence(evidence,
            hard_vetoes=tuple(reason for reason in failures if reason != "native_divergence_positive"),
            shared_invalidity="metropolis_state_mismatch" in failures)
        return {**decision.payload(), "acceptance": evidence.pooled_mean,
                "acceptance_evidence": evidence.payload(),
                "engineering_invalidity_reasons": evidence.engineering_invalidity_reasons,
                "diagnostic_alerts": evidence.payload().get("diagnostic_alerts", ())}

    def work_count(self, work: HMCWorkItem) -> int:
        if work.stage == "pilot" and self.config.pilot_num_results is not None:
            return self.config.pilot_num_results * work.evidence_multiplier
        stage = "measurement" if work.stage == "pilot" else work.stage
        return getattr(self.config, stage + "_num_results") * work.evidence_multiplier

    def evidence_analysis(self, numerical: Mapping[str, Any]) -> Mapping[str, Any]:
        """Recompute successful evidence or validate a typed native failure."""
        # Numerical evidence is host JSON and can be mutated by a caller.
        # Hash its current content before using a cached numerical analysis.
        digest = _sha256(numerical)
        if digest in self._analysis_cache:
            return _json_copy(self._analysis_cache[digest])
        if "execution_failure" in numerical:
            failure = HMCCandidateExecutionFailure.from_payload(numerical["execution_failure"])
            if numerical.get("samples") is not None or numerical.get("trace") is not None:
                raise ValueError("failed execution cannot fabricate complete samples")
            work = HMCWorkItem.from_payload(numerical["work"])
            chunks = numerical["chunks"]
            root = self.work_seed(work)
            total = self.work_count(work) + self.config.num_warmup_steps
            if (failure.chunk_index != len(chunks) or tuple(numerical["seed"]) != root
                    or tuple(numerical["attempted_seed"]) != chunk_seed(root, len(chunks))
                    or sum(c["count"] for c in chunks) >= total
                    or any(type(c["count"]) is not int or c["count"] != self.config.chunk_max_results
                           or tuple(c["seed"]) != chunk_seed(root, i) for i, c in enumerate(chunks))):
                raise ValueError("failed execution chunk/seed inventory mismatch")
            analysis = failure.analysis()
        else:
            analysis = self.analyze(_tensor_from_payload(numerical["initial_state"]),
                                   _tensor_from_payload(numerical["samples"]),
                                   _trace_from_payload(numerical["trace"]))
        self._analysis_cache[digest] = _json_copy(analysis)
        return analysis

    def _target_failure(self, exc: Exception, work: HMCWorkItem,
                        candidate: HMCTuningCandidateRecord, chunks: list, seed: tuple[int, int]):
        classifier = getattr(self._base_adapter, "classify_target_exception", None)
        if not callable(classifier):
            return None
        declared = classifier(exc)
        if type(declared) is not bool:
            raise TypeError("classify_target_exception must return a boolean")
        if not declared:
            return None
        failure = HMCCandidateExecutionFailure(type(exc).__name__, str(exc), len(chunks))
        analysis = failure.analysis()
        numerical = {"schema": "bayesfilter.hmc_candidate_numerical_failure.v1",
            "binding_hash": self.binding_hash, "candidate": candidate.payload(), "work": work.payload(),
            "seed": seed, "initial_state": _tensor_payload(self.initial_active_state),
            "execution_failure": failure.payload(), "analysis": analysis,
            "chunks": [{"count": c["count"], "seed": c["seed"]} for c in chunks],
            "attempted_seed": chunk_seed(seed, len(chunks)), "samples_device": "unavailable"}
        digest = _sha256(numerical)
        self._evidence[digest] = _json_copy(numerical)
        self._partial.pop(work.work_item_id, None)
        return {**analysis, "numerical_evidence_hash": digest,
                "stream_id": work.work_item_id + ":" + _sha256({"seed": seed}),
                "draw_range": (0, 0), "seed_lineage": seed}

    def work_cost(self, work: HMCWorkItem, candidate: HMCTuningCandidateRecord, *, remaining=True) -> Mapping[str, Any]:
        count = self.work_count(work) + self.config.num_warmup_steps
        if remaining:
            count -= sum(chunk["count"] for chunk in self._partial.get(work.work_item_id, ()))
        transitions = count * int(self.initial_active_state.shape[0])
        return {"transitions": transitions, "gradient_work": transitions * (candidate.leapfrog_steps + 1),
                "cost_basis": "conservative_endpoint_gradient_work_estimate", "leapfrog_steps": candidate.leapfrog_steps,
                "charge_mode": "chunk"}

    def observe(self, work: HMCWorkItem, candidate: HMCTuningCandidateRecord) -> Mapping[str, Any]:
        import tensorflow as tf

        try:
            self.validate()
        except ValueError as exc:
            raise HMCSharedInvalidity(str(exc)) from exc
        if (work.candidate_id, work.candidate_record_hash) != (candidate.candidate_id, candidate.candidate_record_hash):
            raise ValueError("work item candidate mismatch")
        if work.stage not in {"pilot", "measurement", "verification"}:
            raise ValueError("unsupported numerical work stage")
        count = self.work_count(work)
        seed = self.work_seed(work)
        total = count + self.config.num_warmup_steps
        chunks = self._partial.setdefault(work.work_item_id, [])
        done = sum(chunk["count"] for chunk in chunks)
        state = (_tensor_from_payload(chunks[-1]["samples"])[-1] if chunks else self.initial_active_state)
        started = time.monotonic()
        while done < total:
            take = min(total - done, self.config.chunk_max_results)
            before_numerical_chunk(self, work, candidate, count=take,
                                   chains=int(self.initial_active_state.shape[0]), index=len(chunks))
            current_seed = chunk_seed(seed, len(chunks))
            chunk_started = time.monotonic()
            key = (candidate.leapfrog_steps, take)
            first_runner_call = key not in self._runners
            try:
                result = self._run(candidate, state, take, current_seed)
            except (tf.errors.ResourceExhaustedError, tf.errors.UnavailableError,
                    tf.errors.DeadlineExceededError, tf.errors.AbortedError) as exc:
                raise HMCInfrastructureFailure(type(exc).__name__ + ": " + str(exc)) from exc
            except Exception as exc:
                failure = self._target_failure(exc, work, candidate, chunks, seed)
                if failure is None:
                    raise
                return failure
            chunk = {"count": take, "seed": current_seed, "initial_state": _tensor_payload(state),
                     "work": work.payload(), "binding_hash": self.binding_hash,
                     "samples": _tensor_payload(result.samples), "trace": _trace_payload(result.trace),
                     "samples_device": result.samples.device, "runtime": result.metadata,
                     "elapsed_seconds": time.monotonic() - chunk_started,
                     "includes_first_runner_trace_or_compilation": first_runner_call}
            chunks.append(_json_copy(chunk))
            state = result.samples[-1]
            done += take
            if self._checkpoint_callback is not None:
                self._checkpoint_callback()
        samples = tf.concat([_tensor_from_payload(chunk["samples"]) for chunk in chunks], axis=0)
        traces = [_trace_from_payload(chunk["trace"]) for chunk in chunks]
        trace = tf.nest.map_structure(lambda *parts: tf.concat(parts, axis=0), *traces)
        analysis = self.analyze(self.initial_active_state, samples, trace)
        numerical = {
            "schema": "bayesfilter.hmc_candidate_numerical_evidence.v1",
            "binding_hash": self.binding_hash, "candidate": candidate.payload(), "work": work.payload(),
            "seed": seed, "initial_state": _tensor_payload(self.initial_active_state),
            "samples": _tensor_payload(samples), "trace": _trace_payload(trace),
            "analysis": analysis, "samples_device": chunks[-1]["samples_device"],
            "runtime": [chunk["runtime"] for chunk in chunks],
            "elapsed_seconds": sum(chunk["elapsed_seconds"] for chunk in chunks),
            "resume_call_elapsed_seconds": time.monotonic() - started,
            "cost": self.work_cost(work, candidate, remaining=False),
            "chunks": [{key: chunk[key] for key in ("count", "seed", "elapsed_seconds",
                        "includes_first_runner_trace_or_compilation")} for chunk in chunks],
            "rhat_reporting_only": _report_rhat(samples[self.config.num_warmup_steps:]),
        }
        evidence_hash = _sha256(numerical)
        self._evidence[evidence_hash] = _json_copy(numerical)
        self._partial.pop(work.work_item_id, None)
        return {**analysis, "numerical_evidence_hash": evidence_hash,
                "stream_id": work.work_item_id + ":" + _sha256({"seed": seed}),
                "draw_range": (self.config.num_warmup_steps, self.config.num_warmup_steps + count),
                "seed_lineage": seed, "rhat_reporting_only": numerical["rhat_reporting_only"]}


def _issue_binding(*, adapter: Any, layers: Sequence[Mapping[str, Any]], initial_active_state: Any,
                   target_scope: str, target_lineage: Mapping[str, Any], preparation: Mapping[str, Any],
                   config: HMCCandidateExecutionConfig, source_paths: Sequence[str | Path],
                   scope_id: str, search_id: str, epsilon_domain: tuple[float, float],
                   repair_factor: float, max_repairs_per_family: int) -> HMCCandidateExecutionBinding:
    import tensorflow as tf
    import tensorflow_probability as tfp
    from bayesfilter.inference.hmc import stable_adapter_signature
    from bayesfilter.inference.posterior_adapter import value_score_capability

    if not isinstance(config, HMCCandidateExecutionConfig):
        raise TypeError("config must be HMCCandidateExecutionConfig")
    if not target_lineage or not str(target_scope).strip():
        raise ValueError("explicit target scope and target/data/prior lineage are required")
    active, _ = _rebuild_geometry(adapter, layers, target_scope)
    starts = tf.convert_to_tensor(initial_active_state, dtype=tf.float64)
    closure = _source_closure(adapter, source_paths)
    kind = "fixed_transport" if layers[0]["kind"] == "frozen_transport" else "ordinary"
    preparation_identity = _sha256({"layers": layers, "lineage": target_lineage, "preparation": preparation})
    numerical_policy = dict(config.payload())
    numerical_policy.pop("preparation_elapsed_seconds")
    transition_identity = _sha256({"runner": "independent_chain_tfp_fixed_hmc",
                                    "config": numerical_policy, "source": closure})
    scope = HMCCandidateSetScope(
        scope_id=scope_id, search_id=search_id, target_signature=stable_adapter_signature(adapter),
        mass_signature=_sha256({"layers": layers}), coordinate_system=kind,
        start_bank_signature=_sha256(_tensor_payload(starts)), warmup_protocol=_sha256(numerical_policy),
        adapter_signature=stable_adapter_signature(active), source_dependency_hash=_sha256(closure),
        target_preparation_identity=preparation_identity, transition_identity=transition_identity,
        epsilon_domain=epsilon_domain, repair_factor=repair_factor, max_repairs_per_family=max_repairs_per_family,
        use_xla=config.use_xla)
    spec = {"schema": EXECUTION_SCHEMA, "scope": scope.payload(), "config": config.payload(),
            "target_scope": target_scope, "target_lineage": target_lineage, "preparation": preparation,
            "adapter_kind": kind, "layers": layers, "initial_active_state": _tensor_payload(starts),
            "source_closure": closure, "probe": _probe(active, starts,
                target_status=config.target_status_trace_policy == "per_chain_step"),
            "capability": asdict(value_score_capability(active)),
            "runtime_policy": _runtime_policy(),
            "versions": {"tensorflow": tf.__version__, "tensorflow_probability": tfp.__version__}}
    return HMCCandidateExecutionBinding(_ISSUER, adapter=adapter, spec=spec)


def bind_hmc_candidate_set_execution(*, adapter: Any, initial_position: Any,
        target_scope: str, target_lineage: Mapping[str, Any], config: HMCCandidateExecutionConfig,
        source_paths: Sequence[str | Path], scope_id: str, search_id: str,
        epsilon_domain: tuple[float, float], repair_factor: float, max_repairs_per_family: int,
        mass_artifact: Any = None, frozen_transport_payload: Mapping[str, Any] | None = None,
        start_coordinates: str = "position") -> HMCCandidateExecutionBinding:
    """Bind an explicit mass in ordinary coordinates, or a supported transport.

    For a nonlinear transport pass starts in active (latent) coordinates and
    ``start_coordinates='active'``; no nonlinear inverse is inferred.
    """
    import tensorflow as tf
    from bayesfilter.inference.hmc import PrecomputedMassArtifact

    _runtime_policy()
    if (mass_artifact is None) == (frozen_transport_payload is None):
        raise ValueError("supply exactly one explicit mass or frozen transport")
    if start_coordinates not in {"position", "active"}:
        raise ValueError("start_coordinates must be position or active")
    if mass_artifact is not None:
        if not isinstance(mass_artifact, PrecomputedMassArtifact):
            raise TypeError("mass_artifact must be PrecomputedMassArtifact")
        mass_artifact.validate_for_adapter(adapter)
        layers = [{"kind": "fixed_mass", "artifact": mass_artifact.to_payload(include_arrays=True)}]
        starts = (mass_artifact.build_latent_transform().position_to_latent(initial_position)
                  if start_coordinates == "position" else initial_position)
    else:
        if start_coordinates != "active":
            raise ValueError("frozen transport requires explicit active-coordinate starts")
        layers = [{"kind": "frozen_transport", "artifact": frozen_transport_payload}]
        starts = initial_position
    return _issue_binding(adapter=adapter, layers=layers, initial_active_state=tf.convert_to_tensor(starts, tf.float64),
        target_scope=target_scope, target_lineage=target_lineage, preparation={"source": "explicit_frozen_geometry"},
        config=config, source_paths=source_paths, scope_id=scope_id, search_id=search_id,
        epsilon_domain=epsilon_domain, repair_factor=repair_factor, max_repairs_per_family=max_repairs_per_family)


def bind_hmc_candidate_set_execution_from_preparation(*, adapter: Any, preparation: Mapping[str, Any],
        target_lineage: Mapping[str, Any], config: HMCCandidateExecutionConfig,
        source_paths: Sequence[str | Path], scope_id: str, search_id: str,
        epsilon_domain: tuple[float, float], repair_factor: float,
        max_repairs_per_family: int) -> HMCCandidateExecutionBinding:
    """Revalidate an operational windowed handoff and preserve both affine layers."""
    from bayesfilter.inference.hmc_kernel_tuning import (
        build_operational_fixed_mass_hmc_adapter, _fixed_mass_step_upper_bound,
    )

    _runtime_policy()
    geometry, windowed = preparation["geometry"], preparation["windowed_stage"]
    checked = build_operational_fixed_mass_hmc_adapter(
        adapter=adapter, geometry=geometry, windowed_stage=windowed, target_scope=preparation["target_scope"])
    for key in ("final_adapter_signature", "adapted_mass_artifact_signature", "start_lineage"):
        if _json_copy(checked[key]) != _json_copy(preparation[key]):
            raise ValueError("preparation handoff identity mismatch: " + key)
    if _tensor_payload(checked["initial_position"]) != _tensor_payload(preparation["initial_position"]):
        raise ValueError("preparation start bank mismatch")
    upper = _fixed_mass_step_upper_bound(windowed)
    if upper is None:
        raise ValueError("operational preparation requires a final-metric epsilon bound")
    domain = (epsilon_domain[0], min(epsilon_domain[1], upper))
    final = windowed.operational_warmup_result.final_kernel_state
    binding = _issue_binding(adapter=adapter,
        layers=[{"kind": "bootstrap_mass", "artifact": geometry.mass_artifact.to_payload(include_arrays=True)},
                {"kind": "fixed_mass", "artifact": checked["adapted_mass_artifact"].to_payload(include_arrays=True)}],
        initial_active_state=checked["initial_position"], target_scope=checked["target_scope"],
        target_lineage=target_lineage,
        preparation={"source": "operational_windowed_handoff", "geometry_hash": geometry.artifact_hash,
                     "start_lineage": checked["start_lineage"], "final_adapter_signature": checked["final_adapter_signature"],
                     "epsilon_proposal_bound": {"upper": upper, "role": "proposal_safety_only",
                         "coordinate_signature": final.transform.signature,
                         "metric_signature": final.momentum_metric.signature}},
        config=config, source_paths=source_paths, scope_id=scope_id, search_id=search_id,
        epsilon_domain=domain, repair_factor=repair_factor, max_repairs_per_family=max_repairs_per_family)
    if binding.scope.adapter_signature != checked["final_adapter_signature"]:
        raise ValueError("reconstructed preparation adapter mismatch")
    return binding
