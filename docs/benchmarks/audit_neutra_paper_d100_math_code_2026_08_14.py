#!/usr/bin/env python3
"""Audit the frozen paper-d100 NeuTra value, Jacobian, and HMC force.

This is a CPU-only diagnostic.  It does not train a transport or run HMC.  The
manual transformed score is compared with TensorFlow autodiff of the exact
finite program used to define the transformed target.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARTIFACT_ROOT = ROOT / (
    "docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/paper-d100"
)
SOURCE_ROOT = ARTIFACT_ROOT / "source-r1"
AUDIT_PLAN = ROOT / (
    "docs/plans/bayesfilter-weighted-forward-kl-paper-d100-math-code-audit-plan-2026-08-14.md"
)
RUNNER_PATH = ROOT / "docs/benchmarks/run_neutra_paper_d100_hmc_2026_08_13.py"

FROZEN_CASES = (
    ("gaussian-reverse-r1", "paper_ill_cond_gaussian", "reverse_kl"),
    ("gaussian-forward-r1", "paper_ill_cond_gaussian", "forward_kl"),
    ("funnel-reverse-r1", "paper_funnel", "reverse_kl"),
    ("funnel-forward-r1", "paper_funnel", "forward_kl"),
)

HMC_ARCHIVE_ROOTS = {
    "gaussian-reverse-r1": "gaussian-reverse-hmc-r2-repair",
    "gaussian-forward-r1": "gaussian-forward-hmc-r1",
    "funnel-reverse-r1": "funnel-reverse-hmc-r1",
    "funnel-forward-r1": "funnel-forward-hmc-r1",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_ready(item) for item in value]
    if hasattr(value, "numpy"):
        return _ready(value.numpy().tolist())
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "item"):
        return value.item()
    return value


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(_ready(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("paper_d100_hmc_audit_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _max_abs(value: Any) -> float:
    return float(tf.reduce_max(tf.abs(value)).numpy())


def _max_rel(value: Any, reference: Any) -> float:
    denominator = tf.maximum(tf.abs(reference), tf.constant(1.0e-12, tf.float64))
    return float(tf.reduce_max(tf.abs(value) / denominator).numpy())


def _tensor_summary(value: Any) -> Mapping[str, float]:
    return {
        "min": float(tf.reduce_min(value).numpy()),
        "max": float(tf.reduce_max(value).numpy()),
        "mean": float(tf.reduce_mean(value).numpy()),
        "max_abs": _max_abs(value),
    }


def _scale_diagnostics(transport: Any, z: Any) -> Mapping[str, Any]:
    current = z
    stages = []
    for index, stage in enumerate(transport.stages):
        scale, _shift = stage._network(current)
        bound = float(stage.s_max)
        near_bound = tf.abs(scale) >= tf.constant(0.99 * bound, tf.float64)
        stages.append(
            {
                "stage": index,
                "input_summary": _tensor_summary(current),
                "scale_summary": _tensor_summary(scale),
                "s_max": bound,
                "fraction_abs_scale_ge_0.99_s_max": float(tf.reduce_mean(tf.cast(near_bound, tf.float64)).numpy()),
                "fraction_abs_scale_ge_0.999_s_max": float(
                    tf.reduce_mean(tf.cast(tf.abs(scale) >= tf.constant(0.999 * bound, tf.float64), tf.float64)).numpy()
                ),
            }
        )
        current, _ = stage.forward_and_logdet(current)
        if index + 1 < len(transport.stages):
            current = tf.reverse(current, axis=(-1,))
    return {"stages": stages, "final_physical_summary": _tensor_summary(current)}


def _verify_hash_ledger(root: Path) -> str:
    ledger_path = root / "artifact_hashes.json"
    if not ledger_path.is_file():
        raise RuntimeError(f"HMC artifact hash ledger is missing: {ledger_path}")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    artifacts = ledger.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise RuntimeError("HMC artifact hash ledger is malformed")
    for relative, expected in artifacts.items():
        path = root / str(relative)
        if not path.is_file() or _sha256(path) != str(expected):
            raise RuntimeError(f"HMC artifact hash mismatch: {path}")
    return _sha256(ledger_path)


def _load_retained_latent(root: Path) -> Any:
    tensors = _load_stage_latent(root, "retained")
    return tf.concat(tensors, axis=0)


def _load_stage_latent(root: Path, stage: str) -> list[Any]:
    paths = sorted((root / "archive" / stage).glob("*-samples.tftensor"))
    if not paths:
        raise RuntimeError(f"HMC {stage} latent archive is missing: {root}")
    tensors = [tf.io.parse_tensor(path.read_bytes(), out_type=tf.float64) for path in paths]
    if any(tensor.shape.rank != 3 or tensor.shape[1:] != (4, 100) for tensor in tensors):
        raise RuntimeError(f"HMC {stage} latent archive has an invalid shape")
    return tensors


def _gaussian_reference_geometry(spec: Any) -> Mapping[str, float]:
    cholesky = tf.constant(spec.cholesky, tf.float64)
    covariance = tf.constant(spec.covariance, tf.float64)
    diagonal = tf.linalg.diag_part(cholesky)
    singular_values = tf.linalg.svd(covariance, compute_uv=False)
    return {
        "single_triangular_map_logdet": float(tf.reduce_sum(tf.math.log(diagonal)).numpy()),
        "single_triangular_map_min_log_diagonal": float(tf.reduce_min(tf.math.log(diagonal)).numpy()),
        "single_triangular_map_max_log_diagonal": float(tf.reduce_max(tf.math.log(diagonal)).numpy()),
        "covariance_log_condition_number": float(
            tf.math.log(tf.reduce_max(singular_values) / tf.reduce_min(singular_values)).numpy()
        ),
    }


def _proposal_diagnostics(
    transport: Any, base: Any, spec: Any, target_name: str, seed_offset: int
) -> Mapping[str, Any]:
    count = 32768
    z = tf.random.stateless_normal(
        (count, 100), seed=(20260814, 89000 + seed_offset), dtype=tf.float64
    )
    physical, logdet = transport.forward_and_logdet(z)
    target_value = base.log_prob(physical)
    log_base = -tf.constant(0.5, tf.float64) * (
        tf.reduce_sum(tf.square(z), axis=1)
        + tf.constant(100.0 * math.log(2.0 * math.pi), tf.float64)
    )
    log_ratio = target_value + logdet - log_base
    normalized = tf.nn.softmax(log_ratio)
    output: dict[str, Any] = {
        "sample_count": count,
        "source": "iid_standard_normal_pushed_through_frozen_transport",
        "log_target_to_proposal_ratio_stddev": float(tf.math.reduce_std(log_ratio).numpy()),
        "self_normalized_importance_ess_fraction": float(
            (tf.math.reciprocal(tf.reduce_sum(tf.square(normalized))) / tf.cast(count, tf.float64)).numpy()
        ),
        "maximum_normalized_importance_weight": float(tf.reduce_max(normalized).numpy()),
        "proposal_scale_diagnostics": _scale_diagnostics(transport, z),
    }
    if target_name == "paper_ill_cond_gaussian":
        centered = physical - tf.constant(spec.mean, tf.float64)[tf.newaxis, :]
        whitened = tf.transpose(
            tf.linalg.triangular_solve(
                tf.constant(spec.cholesky, tf.float64), tf.transpose(centered), lower=True
            )
        )
        output["gaussian_whitened"] = {
            "projection_mean_first4": _ready(tf.reduce_mean(whitened[:, :4], axis=0)),
            "projection_second_moment_first4": _ready(
                tf.reduce_mean(tf.square(whitened[:, :4]), axis=0)
            ),
            "grand_mean": float(tf.reduce_mean(whitened).numpy()),
            "grand_second_moment": float(tf.reduce_mean(tf.square(whitened)).numpy()),
        }
    else:
        y = physical[:, 0]
        output["funnel_y"] = {
            "mean": float(tf.reduce_mean(y).numpy()),
            "second_moment": float(tf.reduce_mean(tf.square(y)).numpy()),
            "probability_below_minus2": float(tf.reduce_mean(tf.cast(y < -2.0, tf.float64)).numpy()),
            "probability_above_plus2": float(tf.reduce_mean(tf.cast(y > 2.0, tf.float64)).numpy()),
        }
    return output


def _archive_diagnostics(
    hmc_runner: Any, transport: Any, spec: Any, target_name: str, root_name: str
) -> Mapping[str, Any]:
    archive_name = HMC_ARCHIVE_ROOTS[root_name]
    archive_root = ARTIFACT_ROOT / archive_name
    ledger_sha256 = _verify_hash_ledger(archive_root)
    latent = _load_retained_latent(archive_root)
    warmup_chunks = _load_stage_latent(archive_root, "warmup")
    retained_chunks = _load_stage_latent(archive_root, "retained")
    flat = tf.reshape(latent, (-1, 100))
    run_manifest = json.loads((archive_root / "run_manifest.json").read_text(encoding="utf-8"))
    initial_z = tf.constant(run_manifest["initial_state"], tf.float64)
    output: dict[str, Any] = {
        "hmc_archive_root": archive_root.as_posix(),
        "artifact_hash_ledger_sha256": ledger_sha256,
        "retained_shape": list(latent.shape),
        "retained_latent_scale_diagnostics": _scale_diagnostics(transport, flat),
        "trajectory_chunk_counts": {
            "warmup": len(warmup_chunks),
            "retained": len(retained_chunks),
        },
    }
    physical = tf.reshape(transport.forward_batch(flat), tf.shape(latent))
    if target_name == "paper_ill_cond_gaussian":
        def whiten(values: Any) -> Any:
            rank = values.shape.rank
            flat_values = tf.reshape(values, (-1, 100))
            centered_values = flat_values - tf.constant(spec.mean, tf.float64)[tf.newaxis, :]
            transformed = tf.transpose(
                tf.linalg.triangular_solve(
                    tf.constant(spec.cholesky, tf.float64),
                    tf.transpose(centered_values),
                    lower=True,
                )
            )
            return tf.reshape(transformed, tf.shape(values)) if rank == 3 else transformed

        whitened = whiten(physical)
        projection = whitened[:, :, 2]
        initial_projection = whiten(transport.forward_batch(initial_z))[:, 2]

        def chunk_projection(chunk: Any) -> Mapping[str, Any]:
            projection_chunk = whiten(transport.forward_batch(tf.reshape(chunk, (-1, 100))))
            projection_chunk = tf.reshape(projection_chunk[:, 2], tf.shape(chunk)[:2])
            return {
                "mean": float(tf.reduce_mean(projection_chunk).numpy()),
                "mean_by_chain": _ready(tf.reduce_mean(projection_chunk, axis=0)),
            }

        output["projection_2"] = {
            "mean": float(tf.reduce_mean(projection).numpy()),
            "mean_by_chain": _ready(tf.reduce_mean(projection, axis=0)),
            "initial_value_by_chain": _ready(initial_projection),
            "warmup_chunk_trajectory": [chunk_projection(chunk) for chunk in warmup_chunks],
            "retained_chunk_trajectory": [chunk_projection(chunk) for chunk in retained_chunks],
            "batch_means_mcse_by_batch_count": {
                str(batch_count): float(
                    hmc_runner._batch_means_mcse(
                        tf, projection[:, :, tf.newaxis], batch_count=batch_count
                    )[0].numpy()
                )
                for batch_count in (4, 5, 8, 10, 20)
            },
        }
        output["gaussian_reference_geometry"] = _gaussian_reference_geometry(spec)
    else:
        y = physical[:, :, 0]
        low = tf.cast(y < -2.0, tf.float64)
        high = tf.cast(y > 2.0, tf.float64)
        output["funnel_y_tail_coverage"] = {
            "y_second_moment": float(tf.reduce_mean(tf.square(y)).numpy()),
            "probability_below_minus2": float(tf.reduce_mean(low).numpy()),
            "probability_above_plus2": float(tf.reduce_mean(high).numpy()),
            "probability_below_minus2_by_chain": _ready(tf.reduce_mean(low, axis=0)),
            "probability_above_plus2_by_chain": _ready(tf.reduce_mean(high, axis=0)),
        }
        initial_y = transport.forward_batch(initial_z)[:, 0]

        def chunk_funnel(chunk: Any) -> Mapping[str, float]:
            rows = transport.forward_batch(tf.reshape(chunk, (-1, 100)))
            y_chunk = tf.reshape(rows[:, 0], tf.shape(chunk)[:2])
            return {
                "y_second_moment": float(tf.reduce_mean(tf.square(y_chunk)).numpy()),
                "probability_below_minus2": float(
                    tf.reduce_mean(tf.cast(y_chunk < -2.0, tf.float64)).numpy()
                ),
                "probability_above_plus2": float(
                    tf.reduce_mean(tf.cast(y_chunk > 2.0, tf.float64)).numpy()
                ),
            }

        output["funnel_y_tail_coverage"].update(
            {
                "initial_y_by_chain": _ready(initial_y),
                "warmup_chunk_trajectory": [chunk_funnel(chunk) for chunk in warmup_chunks],
                "retained_chunk_trajectory": [chunk_funnel(chunk) for chunk in retained_chunks],
            }
        )
    return output


def _audit_case(hmc_runner: Any, target_name: str, objective: str, root_name: str) -> Mapping[str, Any]:
    from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter
    from bayesfilter.inference.neutra_paper_d100_target import (
        PaperD100ValueScoreAdapter,
        load_paper_gaussian_spec,
        make_paper_funnel_spec,
    )

    training_root = ARTIFACT_ROOT / root_name
    state_path = training_root / "trainer_state.json"
    manifest_path = training_root / "run_manifest.json"
    hashes_path = training_root / "artifact_hashes.json"
    for path in (state_path, manifest_path, hashes_path):
        if not path.is_file():
            raise RuntimeError(f"missing frozen-state artifact: {path}")
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
    artifacts = hashes.get("artifacts", {})
    for path in (state_path, manifest_path):
        if artifacts.get(path.name) != _sha256(path):
            raise RuntimeError(f"frozen artifact hash mismatch: {path}")

    constants = SOURCE_ROOT / "paper_ill_cond_gaussian_d100_constants.json"
    if target_name == "paper_funnel":
        spec = make_paper_funnel_spec()
    else:
        spec = load_paper_gaussian_spec(constants)
    transport, config, frozen = hmc_runner._load_frozen_transport(
        tf, training_root, target_name, objective
    )
    base = PaperD100ValueScoreAdapter(spec)
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=base,
        transport=transport,
        target_scope=f"audit:{target_name}:{objective}",
        runtime_backend="tensorflow_cpu_d100_math_audit",
        evidence_path=AUDIT_PLAN.as_posix(),
        require_batch_native=True,
    )

    # Include the origin and moderate deterministic perturbations.  This tests
    # both zero-bias initialization behavior and nontrivial nonlinear paths.
    z = tf.random.stateless_normal((8, 100), seed=(20260814, 88100 + len(root_name)), dtype=tf.float64)
    z = tf.concat((tf.zeros((1, 100), tf.float64), 0.25 * z[:3], z[3:]), axis=0)
    with tf.GradientTape() as tape:
        tape.watch(z)
        theta = transport.forward_batch(z)
        base_value, _base_score = base.log_prob_and_grad(theta)
        logdet = transport.log_abs_det_jacobian_batch(z)
        direct_value = base_value + logdet
        direct_total = tf.reduce_sum(direct_value)
    direct_score = tape.gradient(direct_total, z)
    manual_value, manual_score = adapter.log_prob_and_grad_batch(z)

    recovered, inverse_logdet = transport.inverse_and_forward_logdet(theta)
    roundtrip_error = _max_abs(recovered - z)
    logdet_error = _max_abs(inverse_logdet - logdet)
    value_error = _max_abs(manual_value - direct_value)
    value_relative_error = _max_rel(manual_value - direct_value, direct_value)
    score_error = _max_abs(manual_score - direct_score)
    score_relative_error = _max_rel(manual_score - direct_score, direct_score)

    # Separately test the two explicit transport score terms against autodiff.
    with tf.GradientTape() as tape:
        tape.watch(z)
        pullback_total = tf.reduce_sum(
            transport.forward_batch(z) * _base_score
        )
    expected_pullback = tape.gradient(pullback_total, z)
    with tf.GradientTape() as tape:
        tape.watch(z)
        logdet_total = tf.reduce_sum(transport.log_abs_det_jacobian_batch(z))
    expected_logdet_score = tape.gradient(logdet_total, z)
    actual_pullback = transport.pullback_score_batch(z, _base_score)
    actual_logdet_score = transport.log_abs_det_jacobian_score_batch(z)

    result = {
        "case": root_name,
        "target": target_name,
        "objective": objective,
        "training_state_sha256": frozen["state_sha256"],
        "training_state_hash": frozen["state_hash"],
        "selected_update": int(json.loads(state_path.read_text(encoding="utf-8")).get("selected_update", -1)),
        "config": config.manifest_payload(),
        "sample_shape": list(z.shape),
        "value_parity": {
            "max_abs_error": value_error,
            "max_relative_error": value_relative_error,
            "passed": bool(value_error <= 1.0e-11),
        },
        "transformed_score_parity": {
            "max_abs_error": score_error,
            "max_relative_error": score_relative_error,
            "passed": bool(score_error <= 1.0e-10),
        },
        "pullback_score_parity": {
            "max_abs_error": _max_abs(actual_pullback - expected_pullback),
            "max_relative_error": _max_rel(actual_pullback - expected_pullback, expected_pullback),
        },
        "logdet_score_parity": {
            "max_abs_error": _max_abs(actual_logdet_score - expected_logdet_score),
            "max_relative_error": _max_rel(actual_logdet_score - expected_logdet_score, expected_logdet_score),
        },
        "inverse_roundtrip": {
            "max_abs_coordinate_error": roundtrip_error,
            "max_abs_logdet_error": logdet_error,
            "passed": bool(roundtrip_error <= 1.0e-10 and logdet_error <= 1.0e-10),
        },
        "scale_diagnostics": _scale_diagnostics(transport, z),
        "proposal_diagnostics": _proposal_diagnostics(
            transport, base, spec, target_name, len(root_name)
        ),
        "archive_diagnostics": _archive_diagnostics(
            hmc_runner, transport, spec, target_name, root_name
        ),
    }
    result["passed"] = bool(
        result["value_parity"]["passed"]
        and result["transformed_score_parity"]["passed"]
        and result["inverse_roundtrip"]["passed"]
        and result["pullback_score_parity"]["max_abs_error"] <= 1.0e-10
        and result["logdet_score_parity"]["max_abs_error"] <= 1.0e-10
    )
    return result


def _markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Paper d100 NeuTra math/code audit result (2026-08-14)",
        "",
        f"Plan: `{AUDIT_PLAN.relative_to(ROOT).as_posix()}`",
        "",
        "This is a CPU-only diagnostic. It does not rerun training or HMC.",
        "The transformed force is checked against autodiff of the exact finite value program.",
        "",
        "## Parity results",
        "",
        "| Case | value | transformed force | roundtrip | decision |",
        "|---|---:|---:|---:|---|",
    ]
    for case in result["cases"]:
        lines.append(
            f"| {case['case']} | {case['value_parity']['max_abs_error']:.3e} | "
            f"{case['transformed_score_parity']['max_abs_error']:.3e} | "
            f"{case['inverse_roundtrip']['max_abs_coordinate_error']:.3e} | "
            f"{'PASS' if case['passed'] else 'FAIL'} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "All parity checks passing rules out an omitted target-score term, a missing `grad log|J|` term, a pullback orientation error, and an inverse/logdet sign error for these frozen states. It does not establish HMC convergence or transport quality.",
        "",
        "The remaining failure hypotheses are therefore empirical rather than an established code mismatch:",
        "",
        "1. Reverse KL is mode-seeking in the relevant geometry. Its objective `E_q[log q-log p]` penalizes placing q mass in low-density tails, while it does not directly penalize missing p-tail mass; this predicts the observed funnel tail compression.",
        "2. Forward KL uses exact replay and directly penalizes low q density at replayed target-tail rows. Its funnel pass is consistent with that mechanism, while the Gaussian projection-2 drift remains a direction-specific finite-training or finite-chain issue, not evidence of a wrong force.",
        "3. The d100 config bounds every conditional log-scale to `[-1,1]` per stage (`s_max=1`, three stages). This is a capacity/conditioning hypothesis. The scale saturation table is descriptive and is not a promotion criterion.",
        "4. The Gaussian projection failure could still be a common initialization or finite-chain effect. The four chains share the same small initial-state construction in the HMC runner, so chainwise replication with independently dispersed starts is needed before attributing it to the learned transport.",
        "",
        "These hypotheses do not rank objectives and do not establish default readiness.",
        "",
        "## Archived-chain geometry",
        "",
        "| Case | retained scale contact at 99.9% cap by stage | targeted retained diagnostic |",
        "|---|---|---|",
    ]
    for case in result["cases"]:
        saturation = ", ".join(
            f"{stage['fraction_abs_scale_ge_0.999_s_max']:.3f}"
            for stage in case["archive_diagnostics"]["retained_latent_scale_diagnostics"]["stages"]
        )
        diagnostic = case["archive_diagnostics"].get("projection_2")
        if diagnostic is not None:
            target = f"projection-2={diagnostic['mean']:.5f}; chain means={diagnostic['mean_by_chain']}"
        else:
            tails = case["archive_diagnostics"]["funnel_y_tail_coverage"]
            target = (
                f"E[y^2]={tails['y_second_moment']:.5f}; "
                f"P(y<-2)={tails['probability_below_minus2']:.5f}; "
                f"P(y>2)={tails['probability_above_plus2']:.5f}"
            )
        lines.append(f"| {case['case']} | {saturation} | {target} |")
    lines += [
        "",
        "The scale entries are fractions in stages 0, 1, and 2. They show whether the tanh-bounded conditional scales are active on the actual retained HMC states. They are explanatory diagnostics only: a scale cap can make the transport geometry poor, but it cannot bias the invariant transformed target once the exact Metropolis correction and audited force are used.",
        "",
        "For the Gaussian, batch-means MCSE is reported for several block counts in `result.json`; sensitivity to block size is evidence about uncertainty estimation, not a proof of stationarity.",
        "",
        "## Learned proposal",
        "",
        "| Case | proposal diagnostic | log target/proposal SD | importance ESS fraction |",
        "|---|---|---:|---:|",
    ]
    for case in result["cases"]:
        proposal = case["proposal_diagnostics"]
        if "gaussian_whitened" in proposal:
            gaussian = proposal["gaussian_whitened"]
            diagnostic = (
                f"whitened means={gaussian['projection_mean_first4']}; "
                f"seconds={gaussian['projection_second_moment_first4']}"
            )
        else:
            funnel = proposal["funnel_y"]
            diagnostic = (
                f"E_q[y^2]={funnel['second_moment']:.5f}; "
                f"tails={funnel['probability_below_minus2']:.5f}/{funnel['probability_above_plus2']:.5f}"
            )
        lines.append(
            f"| {case['case']} | {diagnostic} | "
            f"{proposal['log_target_to_proposal_ratio_stddev']:.3f} | "
            f"{proposal['self_normalized_importance_ess_fraction']:.3e} |"
        )
    lines += [
        "",
        "These are proposal-quality diagnostics from iid `z~N(0,I)` draws, not posterior samples. The target/proposal ratio is unnormalized, but its dispersion and normalized-weight ESS are invariant to the missing target normalizing constant. A poor value explains difficult transformed geometry; it does not change HMC's exact invariant density.",
        "",
        "## Decision table",
        "",
        "| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Non-claim |",
        "|---|---|---|---|---|---|",
        "| Frozen transformed-density implementation | direct value/force, Jacobian, and roundtrip parity | see case table | numerical tolerance and finite probe set | retain code; investigate learned geometry and chain-start effects | no posterior correctness claim |",
        "",
        "## Inference status",
        "",
        "| Evidence class | Status |",
        "|---|---|",
        "| Hard veto screen | No parity veto if all cases pass |",
        "| Statistically supported ranking | None |",
        "| Descriptive-only differences | scale saturation and prior HMC diagnostics |",
        "| Default-readiness | Not assessed |",
        "| Next evidence needed | independent HMC starts and transport-capacity/scale ablation under a new reviewed plan |",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    global tf
    import tensorflow as tf  # noqa: PLC0415

    hmc_runner = _load_module(RUNNER_PATH)
    cases = [_audit_case(hmc_runner, target, objective, root) for root, target, objective in FROZEN_CASES]
    payload = {
        "schema": "bayesfilter.neutra.paper_d100_math_code_audit.v1",
        "plan": AUDIT_PLAN.as_posix(),
        "runner": RUNNER_PATH.as_posix(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tensorflow_version": tf.__version__,
        "cases": cases,
        "all_passed": all(bool(case["passed"]) for case in cases),
        "research_question": "Does the frozen target-to-transport-to-HMC score compute the stated transformed density?",
        "nonclaims": [
            "parity does not establish HMC convergence",
            "parity does not establish objective superiority",
            "scale diagnostics do not establish transport promotion",
        ],
    }
    output = ARTIFACT_ROOT / "math-code-audit-r1"
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(_ready(payload), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (output / "result.md").write_text(_markdown(payload), encoding="utf-8")
    manifest = {
        "schema": "bayesfilter.neutra.paper_d100_math_code_audit_manifest.v1",
        "plan": AUDIT_PLAN.as_posix(),
        "command": "CUDA_VISIBLE_DEVICES=-1 python docs/benchmarks/audit_neutra_paper_d100_math_code_2026_08_14.py",
        "tensorflow_version": tf.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "frozen_cases": [case[0] for case in FROZEN_CASES],
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    hashes = {
        "schema": "bayesfilter.neutra.paper_d100_math_code_audit_hashes.v1",
        "artifacts": {path.name: _sha256(path) for path in sorted(output.iterdir()) if path.is_file() and path.name != "artifact_hashes.json"},
    }
    (output / "artifact_hashes.json").write_text(json.dumps(hashes, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_passed": payload["all_passed"], "output": output.as_posix()}, sort_keys=True))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
