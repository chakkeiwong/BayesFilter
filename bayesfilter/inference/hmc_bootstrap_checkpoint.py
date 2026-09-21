"""Checkpointed fixed-kernel bootstrap execution; no tuning authority.

Host orchestration surrounds small, stable-signature TF/TFP chain calls.
Completed chunks include discarded startup transitions and their health traces.
"""
from __future__ import annotations

import base64
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time

POLICY = "bayesfilter.hmc_checkpointed_bootstrap.v1"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _encode(value):
    import tensorflow as tf
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    tensor = tf.convert_to_tensor(value)
    return {"dtype": tensor.dtype.name, "shape": tensor.shape.as_list(),
            "tensor": base64.b64encode(tf.io.serialize_tensor(tensor).numpy()).decode("ascii")}


def _decode(value):
    import tensorflow as tf
    if "tensor" not in value:
        return {key: _decode(item) for key, item in value.items()}
    tensor = tf.io.parse_tensor(base64.b64decode(value["tensor"], validate=True),
                                out_type=tf.as_dtype(value["dtype"]))
    return tf.ensure_shape(tensor, value["shape"])


def require_chunk_health(samples, trace):
    """Inspect discarded and measured transitions, including rejected proposals."""
    import tensorflow as tf
    from bayesfilter.inference.hmc_preparation import HMCPreparationFailure
    required = {"log_accept_ratio", "target_log_prob", "proposed_target_log_prob",
                "target_score_finite", "proposed_state", "initial_momentum", "final_momentum",
                "target_status_telemetry", "proposed_target_status_telemetry", "is_accepted"}
    failures = ["missing:" + key for key in sorted(required - trace.keys())]
    for index, value in enumerate(tf.nest.flatten({"samples": samples, "trace": trace})):
        if value.dtype.is_floating and not bool(tf.reduce_all(tf.math.is_finite(value)).numpy()):
            failures.append("nonfinite_tensor:" + str(index))
    if "target_score_finite" in trace and not bool(tf.reduce_all(trace["target_score_finite"]).numpy()):
        failures.append("nonfinite_score")
    for name in ("target_status_telemetry", "proposed_target_status_telemetry"):
        if name not in trace:
            continue
        status = trace[name]
        fields = {"status_code", "valid_pre_regularized_score", "floor_count_value"}
        if fields - status.keys():
            failures.append(name + ":incomplete")
        elif not bool(tf.reduce_all((status["status_code"] == 0) &
                status["valid_pre_regularized_score"] & (status["floor_count_value"] >= 0)).numpy()):
            failures.append(name + ":invalid")
    if "divergence" in trace and bool(tf.reduce_any(trace["divergence"]).numpy()):
        failures.append("native_divergence")
    if failures:
        raise HMCPreparationFailure("bootstrap chunk health veto",
                                    details={"stage": "bootstrap_chunk", "hard_vetoes": failures})


class CheckpointedBootstrapRunner:
    def __init__(self, output_dir, *, sources, progress, chunk_size=4,
                 seconds_per_transition=None, safety_factor=2., resume_from=None):
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("bootstrap chunk size must be a positive integer")
        if not sources:
            raise ValueError("bootstrap checkpoints require source identity")
        if seconds_per_transition is not None and (not math.isfinite(seconds_per_transition) or seconds_per_transition <= 0):
            raise ValueError("bootstrap requires a positive measured cost estimate")
        if not math.isfinite(safety_factor) or safety_factor < 1:
            raise ValueError("bootstrap forecast factor must be finite and at least one")
        self.root = Path(output_dir)
        self.root.mkdir(parents=True, exist_ok=False)
        self.previous = None if resume_from is None else Path(resume_from)
        if self.previous is not None and (not self.previous.is_dir() or self.previous.resolve() == self.root.resolve()):
            raise ValueError("bootstrap resume requires a separate existing checkpoint directory")
        self.sources, self.progress = sources, progress
        self.chunk_size, self.safety_factor = chunk_size, safety_factor
        self.seconds_per_transition = seconds_per_transition
        self.runners, self.completed_chunks = {}, 0

    def observe_initialization(self, geometry):
        record = geometry.formula_report.get("bootstrap_initialization", {})
        rows = record.get("rounds", [])
        if self.seconds_per_transition is None:
            if not rows or record.get("status") != "startup_nominated":
                raise ValueError("checkpointed startup requires a measured initializer or prior cost")
            # A batch of four independent proposals prices the first scalar
            # transition conservatively; later scalar chunks refine the forecast.
            self.seconds_per_transition = rows[-1]["wall_seconds"]

    def admit(self, stage, transitions):
        if self.seconds_per_transition is None:
            raise ValueError("bootstrap cost has not been measured")
        estimate = transitions * self.seconds_per_transition * self.safety_factor
        self.progress.admit_work(stage, estimated_seconds=estimate,
            details={"transitions": transitions, "seconds_per_transition": self.seconds_per_transition,
                     "safety_factor": self.safety_factor, "checkpoint_dir": str(self.root),
                     "estimate_role": "scheduling_hypothesis_not_completed_preparation_price"})

    def __call__(self, adapter, initial, config):
        import tensorflow as tf
        from bayesfilter.inference.hmc import (
            FullChainHMCRunResult, build_reusable_full_chain_tfp_hmc_runner,
            _full_chain_hmc_diagnostics, stable_adapter_signature,
        )
        initial = tf.convert_to_tensor(initial, tf.float64)
        contract = {"policy": POLICY, "sources": self.sources,
            "adapter": stable_adapter_signature(adapter), "config": config.signature_payload(),
            "initial": _encode(initial), "chunk_size": self.chunk_size}
        # Normalize tuples before comparing JSON-loaded receipts.
        contract = json.loads(json.dumps(contract))
        key = _digest(config.signature_payload())
        folder = self.root / key
        folder.mkdir(exist_ok=True)
        prior = None if self.previous is None else self.previous / key
        if prior is not None and not prior.exists() and self.completed_chunks == 0:
            raise ValueError("bootstrap resume config does not match the saved first round")
        if prior is not None and prior.exists():
            if json.loads((prior / "contract.json").read_text()) != contract:
                raise ValueError("bootstrap checkpoint source/target/start/config mismatch")
        _write(folder / "contract.json", contract)
        total = config.num_burnin_steps + config.num_results
        samples, traces, current, elapsed = [], [], initial, 0.
        for offset in range(0, total, self.chunk_size):
            count = min(self.chunk_size, total-offset)
            raw = hashlib.sha256(json.dumps([POLICY, list(config.seed), offset]).encode()).digest()
            seed = tuple(int.from_bytes(raw[i:i+4], "big") & 0x7fffffff for i in (0, 4))
            path = folder / f"chunk-{offset:06d}.json"
            existing = path if path.exists() else None if prior is None else prior / path.name
            expected = {"contract_hash": _digest(contract), "offset": offset, "count": count,
                        "seed": list(seed), "initial": _encode(current)}
            if existing is not None and existing.exists():
                payload = json.loads(existing.read_text())
                checksum = payload.pop("checksum")
                if checksum != _digest(payload) or any(payload.get(k) != v for k, v in expected.items()):
                    raise ValueError("bootstrap checkpoint checksum or continuity mismatch")
                chunk_samples, chunk_trace = _decode(payload["samples"]), _decode(payload["trace"])
                reused = True
            else:
                self.admit("bootstrap_chunk_start", count)
                chunk_config = replace(config, num_results=count, num_burnin_steps=0,
                                       capture_candidate_health=True, capture_first_failure=False)
                static = _digest({**chunk_config.signature_payload(), "step_size": None, "seed": None})
                if static not in self.runners:
                    self.runners[static] = build_reusable_full_chain_tfp_hmc_runner(adapter, initial, chunk_config)
                began = time.monotonic()
                result = self.runners[static].run(current_state=current, seed=seed, step_size=config.step_size)
                # Materialization in _encode synchronizes device work before timing ends.
                encoded_samples, encoded_trace = _encode(result.samples), _encode(dict(result.trace))
                payload = {**expected, "samples": encoded_samples, "trace": encoded_trace,
                           "wall_seconds": time.monotonic()-began}
                chunk_samples, chunk_trace = result.samples, result.trace
                reused = False
            _write(path, {**payload, "checksum": _digest(payload)})
            require_chunk_health(chunk_samples, chunk_trace)
            if chunk_samples.shape != (count, *initial.shape):
                raise ValueError("bootstrap checkpoint sample shape mismatch")
            self.seconds_per_transition = max(self.seconds_per_transition, payload["wall_seconds"]/count)
            elapsed += payload["wall_seconds"]
            current = chunk_samples[-1]
            samples.append(chunk_samples)
            traces.append(chunk_trace)
            self.completed_chunks += 1
            self.progress.phase("bootstrap_chunk_complete", {"offset": offset, "count": count,
                "round_seed": list(config.seed), "seed": list(seed), "resumed": reused,
                "wall_seconds": payload["wall_seconds"], "checkpoint": str(path)})
        burn = config.num_burnin_steps
        retained = tf.concat(samples, axis=0)[burn:]
        trace = tf.nest.map_structure(lambda *values: tf.concat(values, axis=0)[burn:], *traces)
        return FullChainHMCRunResult(samples=retained, trace=trace,
            diagnostics=_full_chain_hmc_diagnostics(retained, trace, trace_policy="standard"),
            metadata={"sample_chain_call_s": elapsed, "use_xla": config.use_xla,
                "jit_compile": config.use_xla, "bootstrap_chunk_policy": POLICY,
                "all_startup_health_checked": True, "checkpoint_dir": str(folder),
                "burnin_transitions": burn, "total_transitions": total})
