#!/usr/bin/env python3
"""Compute bounded retrospective truth-tail diagnostics from frozen draws."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = (
    "docs/plans/bayesfilter-neutra-retrospective-truth-tail-diagnostic-plan-"
    "2026-07-17.md"
)
PASS_THRESHOLD = 0.05
SEVERE_THRESHOLD = 0.003
RHAT_MAX = 1.01
BULK_ESS_MIN = 1000.0
TAIL_ESS_MIN = 400.0


@dataclass(frozen=True)
class CellSpec:
    cell_id: str
    result_path: str
    result_sha256: str
    sample_path: str
    sample_sha256: str
    sample_shape: tuple[int, int, int]
    parameter_names: tuple[str, ...]
    truth: tuple[float, ...]
    prior_center: tuple[float, ...]
    central_in_declared_prior_coordinates: tuple[bool, ...]
    truth_provenance: tuple[str, ...]
    target_signature: str
    coordinate_system: str
    transform: Callable[[Any, Any], Any]
    result_convergence_key: str


def _identity(_tf: Any, values: Any) -> Any:
    return values


def _lgssm_physical(tf: Any, values: Any) -> Any:
    return tf.concat(
        (
            0.85 * tf.math.tanh(values[..., :4]),
            0.35 * tf.math.tanh(values[..., 4:10]),
            tf.math.exp(values[..., 10:18]),
        ),
        axis=-1,
    )


def _predator_prey_physical(tf: Any, values: Any) -> Any:
    lower = tf.constant((0.1, 110.0, 20.0, 0.1, 0.0, 0.0), tf.float64)
    upper = tf.constant((1.1, 130.0, 30.0, 1.1, 1.0, 1.0), tf.float64)
    probability = 0.5 * (
        1.0 + tf.math.erf(values / tf.math.sqrt(tf.constant(2.0, tf.float64)))
    )
    return lower + (upper - lower) * probability


def _sir_physical(tf: Any, values: Any) -> Any:
    return tf.constant((0.1, 18.0, 10.0), tf.float64) * tf.math.exp(values)


LGSSM_TRUTH_RAW = (
    0.9274691854247932,
    0.6397156077887646,
    0.3687994715653896,
    0.19050700612200003,
    0.5685392847479529,
    -0.29389333245105953,
    0.42364893019360195,
    0.1731381183589169,
    -0.23268162484461669,
    0.3252937830705747,
    -1.203972804325936,
    -1.3470736479666092,
    -1.5141277326297755,
    -1.7147984280919266,
    -2.120263536200091,
    -2.2072749131897207,
    -2.3025850929940455,
    -2.407945608651872,
)
LGSSM_TRUTH_PHYSICAL = (
    0.62,
    0.48,
    0.30,
    0.16,
    0.18,
    -0.10,
    0.14,
    0.06,
    -0.08,
    0.11,
    0.30,
    0.26,
    0.22,
    0.18,
    0.12,
    0.11,
    0.10,
    0.09,
)


def _cell_specs() -> tuple[CellSpec, ...]:
    multimodel = "docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715"
    pp_names = ("r", "K", "a", "s", "u", "v")
    pp_truth = (0.6, 114.0, 25.0, 0.3, 0.5, 0.5)
    pp_prior = (0.6, 120.0, 25.0, 0.6, 0.5, 0.5)
    pp_central = (True, False, True, False, True, True)
    pp_truth_provenance = (
        "bayesfilter/testing/predator_prey_ukf_neutra_target_tf.py:PP_TRUTH_PHYSICAL",
        "Bayesian target prior: independent Uniform physical parameter boxes",
        "dataset seed 81104, observation sha256:dc63294b6e77913aef0c92796dd2d3c7a1721a766f976fcc392cd02a70754387",
    )
    pp_ukf_root = f"{multimodel}/phase-p4/PP-UKF/neutra-confirmation/attempt-03"
    pp_sgqf_root = f"{multimodel}/phase-p4/PP-SGQF/neutra-confirmation/attempt-02"
    sir_root = f"{multimodel}/phase-p6/SIR-SGQF/neutra-confirmation/attempt-01"
    lgssm_root = (
        "docs/plans/artifacts/lgssm-neutra-gap-closure-2026-07-15/"
        "sequential-repair-attempt-01/confirmation-attempt-01/dense_seed1201"
    )
    return (
        CellSpec(
            cell_id="LGSSM-EXACT",
            result_path=f"{lgssm_root}/result.json",
            result_sha256="135ba4836f0b439e386e516979d6d43109021a6f0995512c028f672736f43675",
            sample_path=f"{lgssm_root}/samples/retained/all_raw.tftensor",
            sample_sha256="6370ecd56f721b039c3beb180e3e2e46e4f50a61cb43ed9e8d3c82ea35258d2f",
            sample_shape=(4000, 4, 18),
            parameter_names=(
                "a11", "a22", "a33", "a44", "a21", "a31", "a32",
                "a41", "a42", "a43", "q1", "q2", "q3", "q4",
                "r1", "r2", "r3", "r4",
            ),
            truth=LGSSM_TRUTH_PHYSICAL,
            prior_center=LGSSM_TRUTH_PHYSICAL,
            central_in_declared_prior_coordinates=(True,) * 18,
            truth_provenance=(
                "docs/plans/artifacts/multidim-triangular-lgssm-neutra-hmc-2026-07-08/lower_triangular_lgssm_contract_v1.json:truth_template",
                "docs/benchmarks/configs/multidim_lgssm_full_estimation_rerun_2026_07_13.json:truth_policy=prior_mean_raw_coordinates_from_source_contract",
                "dataset seed [20260709,301]",
            ),
            target_signature="f47619320ded5f70259c6932eb2436642a02834c7a0249c7c52c20a5a2302f30",
            coordinate_system="physical_lgssm_model_parameters",
            transform=_lgssm_physical,
            result_convergence_key="final_full_convergence",
        ),
        CellSpec(
            cell_id="PP-UKF",
            result_path=f"{pp_ukf_root}/result.json",
            result_sha256="d9b4f603b28acb06154ab554f41f745c5f544e2516ba4969c6b21d9e5268bacf",
            sample_path=f"{pp_ukf_root}/samples/retained/cumulative/model.tensor",
            sample_sha256="a856d11ac7425ce87e34839b87764157d7c69e3fc0e308a69b462243ff75b67a",
            sample_shape=(4000, 4, 6),
            parameter_names=pp_names,
            truth=pp_truth,
            prior_center=pp_prior,
            central_in_declared_prior_coordinates=pp_central,
            truth_provenance=pp_truth_provenance,
            target_signature="036948f0faaf028d159d7b70337214f01514d732112c2d10e9f7eea1e13b8e30",
            coordinate_system="physical_predator_prey_parameters",
            transform=_predator_prey_physical,
            result_convergence_key="final_joint_diagnostic",
        ),
        CellSpec(
            cell_id="PP-SGQF",
            result_path=f"{pp_sgqf_root}/result.json",
            result_sha256="a77d5edf2b8129d6ff95844e9c5d4bb94b7125c9997777b517f36b830fbda9c4",
            sample_path=f"{pp_sgqf_root}/samples/retained/cumulative/model.tensor",
            sample_sha256="35a8d14d1517b693a07da735bd43b162cee47950ea4539288fb36cfc2588ed23",
            sample_shape=(4000, 4, 6),
            parameter_names=pp_names,
            truth=pp_truth,
            prior_center=pp_prior,
            central_in_declared_prior_coordinates=pp_central,
            truth_provenance=pp_truth_provenance,
            target_signature="8e0a9582fd30643b2e77e7615a21c0d44cc6c1827865ea52c841cc6dbfdde1ad",
            coordinate_system="physical_predator_prey_parameters",
            transform=_predator_prey_physical,
            result_convergence_key="final_joint_diagnostic",
        ),
        CellSpec(
            cell_id="SIR-SGQF",
            result_path=f"{sir_root}/result.json",
            result_sha256="e8b6c159648ade9f2919d97674ffc50a8b55d75d591a256291c3abfdcd4dbcce",
            sample_path=f"{sir_root}/samples/retained/cumulative/model.tensor",
            sample_sha256="3848596cb1428b4800abab17c790edb1a9fc6d3909aae12bfe7198cbbbdb5092",
            sample_shape=(4000, 4, 3),
            parameter_names=("kappa", "nu", "observation_sd_scale"),
            truth=(0.1, 18.0, 10.0),
            prior_center=(0.1, 18.0, 10.0),
            central_in_declared_prior_coordinates=(True, True, True),
            truth_provenance=(
                "bayesfilter/testing/sir_filter_neutra_target_design_tf.py:theta_truth=(0,0,0)",
                "declared prior Normal log-scale mean=(0,0,0)",
                "dataset seed 81120, observation sha256:cd794ad6e90a74f7cf6dc06b33550bff4bef6fbf66bb0917846d0691b5910f07",
            ),
            target_signature="0e7921dbd1a2c9a943674b16fd10ccd8b68e1c889e9ae8269a06e0359a750fbc",
            coordinate_system="physical_sir_scale_parameters",
            transform=_sir_physical,
            result_convergence_key="final_joint_diagnostic",
        ),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def _read_json(relative_path: str) -> Mapping[str, Any]:
    payload = json.loads((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {relative_path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _self_check(tf: Any) -> Mapping[str, Any]:
    values = tf.constant((0.0, 1.0, 2.0, 3.0), tf.float64)
    truth = tf.constant(1.5, tf.float64)
    less = tf.reduce_sum(tf.cast(values < truth, tf.float64))
    equal = tf.reduce_sum(tf.cast(values == truth, tf.float64))
    cdf = (less + 0.5 * equal + 0.5) / (tf.cast(tf.size(values), tf.float64) + 1.0)
    p_truth = 2.0 * tf.minimum(cdf, 1.0 - cdf)
    expected = 1.0
    if not math.isclose(float(p_truth.numpy()), expected, rel_tol=0.0, abs_tol=1e-15):
        raise AssertionError("smoothed ECDF self-check failed")
    return {
        "passed": True,
        "case": [0.0, 1.0, 2.0, 3.0],
        "truth": 1.5,
        "expected_p_truth": expected,
        "observed_p_truth": float(p_truth.numpy()),
    }


def _load_samples(tf: Any, spec: CellSpec) -> Any:
    path = REPO_ROOT / spec.sample_path
    tensor = tf.io.parse_tensor(tf.io.read_file(str(path)), out_type=tf.float64)
    tensor = tf.ensure_shape(tensor, spec.sample_shape)
    if not bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy()):
        raise ValueError(f"nonfinite retained samples: {spec.cell_id}")
    return tensor


def _validate_sampler(spec: CellSpec, result: Mapping[str, Any]) -> Mapping[str, Any]:
    if _sha256(REPO_ROOT / spec.result_path) != spec.result_sha256:
        raise ValueError(f"source result hash mismatch: {spec.cell_id}")
    if _sha256(REPO_ROOT / spec.sample_path) != spec.sample_sha256:
        raise ValueError(f"retained sample hash mismatch: {spec.cell_id}")
    if result.get("passed") is not True:
        raise ValueError(f"source result did not pass: {spec.cell_id}")
    target_signature = result.get("target_signature")
    if target_signature is None:
        target_signature = result.get("target_identity", {}).get("target_signature")
    if str(target_signature) != spec.target_signature:
        raise ValueError(f"target signature mismatch: {spec.cell_id}")
    convergence_payload = result.get(spec.result_convergence_key)
    if not isinstance(convergence_payload, Mapping):
        raise ValueError(f"missing convergence payload: {spec.cell_id}")
    convergence = convergence_payload.get("convergence", convergence_payload)
    if convergence.get("passed") is not True or convergence.get("hard_vetoes"):
        raise ValueError(f"preserved convergence veto: {spec.cell_id}")
    max_rhat = float(convergence["max_rhat"])
    min_bulk_ess = float(convergence["min_bulk_ess"])
    min_tail_ess = float(convergence["min_tail_ess"])
    if max_rhat > RHAT_MAX or min_bulk_ess < BULK_ESS_MIN or min_tail_ess < TAIL_ESS_MIN:
        raise ValueError(f"preserved convergence threshold failure: {spec.cell_id}")
    definitions = convergence.get("definitions", {})
    expected_rhat = "max(rank-normalized split R-hat, folded rank-normalized split R-hat)"
    if definitions.get("rhat") != expected_rhat:
        raise ValueError(f"non-modern preserved R-hat definition: {spec.cell_id}")

    sequential = result.get("sequential_run")
    if not isinstance(sequential, Mapping):
        raise ValueError(f"missing sequential run: {spec.cell_id}")
    if (
        sequential.get("passed") is not True
        or sequential.get("retained_passed") is not True
        or sequential.get("hard_vetoes")
        or sequential.get("warmup_excluded_from_posterior") is not True
        or sequential.get("warmup_samples_retained") is not True
    ):
        raise ValueError(f"sequential/warm-up validity failure: {spec.cell_id}")
    archive_key = "retained_raw" if spec.cell_id == "LGSSM-EXACT" else "retained"
    archive = sequential.get("cumulative_archives", {}).get(archive_key)
    if not isinstance(archive, Mapping):
        raise ValueError(f"missing cumulative archive binding: {spec.cell_id}")
    archive_path = archive.get("tensor_path", archive.get("model_path"))
    if Path(str(archive_path)).resolve() != (REPO_ROOT / spec.sample_path).resolve():
        raise ValueError(f"sample path binding mismatch: {spec.cell_id}")
    if tuple(int(item) for item in archive["shape" if "shape" in archive else "sample_shape"]) != spec.sample_shape:
        raise ValueError(f"sample shape binding mismatch: {spec.cell_id}")
    if archive.get("target_signature", spec.target_signature) != spec.target_signature:
        raise ValueError(f"archive target signature mismatch: {spec.cell_id}")

    final_check = sequential.get("retained_checks", [])[-1]
    health = final_check.get("health", {})
    telemetry = health.get("target_status_telemetry", {})
    if (
        health.get("health_passed") is not True
        or health.get("samples_all_finite") is not True
        or health.get("target_log_prob_all_finite") is not True
        or health.get("log_accept_ratio_all_finite") is not True
        or telemetry.get("all_status_valid") is not True
        or (
            health.get("native_divergence_status") == "available"
            and int(health.get("native_divergence_count", 1)) != 0
        )
    ):
        raise ValueError(f"preserved health/status veto: {spec.cell_id}")
    return {
        "passed": True,
        "max_modern_rhat": max_rhat,
        "min_bulk_ess": min_bulk_ess,
        "min_tail_ess": min_tail_ess,
        "draws_per_chain": int(convergence["draw_count_per_chain"]),
        "chain_count": int(convergence["chain_count"]),
        "warmup_results_per_chain": int(sequential["warmup_results_per_chain"]),
        "retained_results_per_chain": int(sequential["retained_results_per_chain"]),
        "warmup_separate_and_excluded": True,
        "extreme_log_accept_count": int(
            health.get(
                "extreme_log_accept_count",
                health.get("energy_error_divergence_count", 0),
            )
        ),
        "extreme_log_accept_role": "explanatory_only_not_a_veto_or_divergence",
        "native_divergence_status": health["native_divergence_status"],
        "target_status_all_valid": True,
    }


def _parameter_statistics(tf: Any, tfp: Any, values: Any, spec: CellSpec) -> tuple[Mapping[str, Any], ...]:
    draw_count, chain_count, dimension = spec.sample_shape
    half = draw_count // 2
    split = tf.reshape(
        tf.stack((values[:half], values[-half:]), axis=2),
        (half, 2 * chain_count, dimension),
    )
    mean_ess = tfp.mcmc.effective_sample_size(
        split, filter_beyond_positive_pairs=True, cross_chain_dims=1
    )
    pooled = tf.reshape(values, (-1, dimension))
    means = tf.reduce_mean(pooled, axis=0)
    centered = pooled - means[None, :]
    sds = tf.sqrt(
        tf.reduce_sum(tf.square(centered), axis=0)
        / tf.cast(tf.shape(pooled)[0] - 1, tf.float64)
    )
    intervals = tfp.stats.percentile(
        pooled, (2.5, 97.5), axis=0, interpolation="linear"
    )
    truth = tf.constant(spec.truth, tf.float64)
    less = tf.reduce_sum(tf.cast(pooled < truth[None, :], tf.float64), axis=0)
    equal = tf.reduce_sum(tf.cast(pooled == truth[None, :], tf.float64), axis=0)
    total = tf.cast(tf.shape(pooled)[0], tf.float64)
    cdf = (less + 0.5 * equal + 0.5) / (total + 1.0)
    p_truth = 2.0 * tf.minimum(cdf, 1.0 - cdf)
    finite = tf.reduce_all(
        tf.stack(
            (
                tf.reduce_all(tf.math.is_finite(means)),
                tf.reduce_all(tf.math.is_finite(sds)),
                tf.reduce_all(tf.math.is_finite(mean_ess)),
                tf.reduce_all(tf.math.is_finite(intervals)),
                tf.reduce_all(tf.math.is_finite(p_truth)),
            )
        )
    )
    if not bool(finite.numpy()) or not bool(tf.reduce_all(mean_ess > 0.0).numpy()):
        raise ValueError(f"invalid posterior summaries: {spec.cell_id}")
    rows = []
    for index, name in enumerate(spec.parameter_names):
        current_p = float(p_truth[index].numpy())
        if current_p < SEVERE_THRESHOLD:
            status = "SEVERE"
        elif current_p < PASS_THRESHOLD:
            status = "MARGINAL"
        else:
            status = "PASS"
        rows.append(
            {
                "parameter": name,
                "truth": float(spec.truth[index]),
                "prior_center": float(spec.prior_center[index]),
                "truth_at_prior_center": bool(spec.central_in_declared_prior_coordinates[index]),
                "posterior_mean": float(means[index].numpy()),
                "posterior_sd": float(sds[index].numpy()),
                "credible_interval_95": [
                    float(intervals[0, index].numpy()),
                    float(intervals[1, index].numpy()),
                ],
                "truth_in_empirical_95_interval": bool(
                    tf.logical_and(
                        intervals[0, index] <= truth[index],
                        truth[index] <= intervals[1, index],
                    ).numpy()
                ),
                "mean_ess": float(mean_ess[index].numpy()),
                "pooled_draw_count": int(draw_count * chain_count),
                "count_less_than_truth": int(less[index].numpy()),
                "count_equal_to_truth": int(equal[index].numpy()),
                "smoothed_ecdf_at_truth": float(cdf[index].numpy()),
                "p_truth": current_p,
                "tail_status": status,
            }
        )
    return tuple(rows)


def _classify(rows: Sequence[Mapping[str, Any]], fully_central: bool) -> str:
    minimum = min(float(row["p_truth"]) for row in rows)
    if minimum < SEVERE_THRESHOLD:
        return "ONE_SEED_DIAGNOSTIC_FAILURE"
    if minimum < PASS_THRESHOLD:
        return "MARGINAL_ONE_SEED"
    if fully_central:
        return "ONE_SEED_DIAGNOSTIC_PASS"
    return "RETROSPECTIVE_ONE_SEED_TAIL_PASS_NONCENTRAL_TRUTH"


def _analyze_cell(tf: Any, tfp: Any, spec: CellSpec) -> Mapping[str, Any]:
    result_path = REPO_ROOT / spec.result_path
    sample_path = REPO_ROOT / spec.sample_path
    result = _read_json(spec.result_path)
    sampler = _validate_sampler(spec, result)
    raw_samples = _load_samples(tf, spec)
    displayed_samples = spec.transform(tf, raw_samples)
    displayed_samples = tf.ensure_shape(displayed_samples, spec.sample_shape)
    rows = _parameter_statistics(tf, tfp, displayed_samples, spec)
    fully_central = all(spec.central_in_declared_prior_coordinates)
    classification = _classify(rows, fully_central)
    return {
        "cell_id": spec.cell_id,
        "classification": classification,
        "sampler_valid": True,
        "fully_central_truth": fully_central,
        "central_truth_parameter_count": sum(spec.central_in_declared_prior_coordinates),
        "parameter_count": len(spec.parameter_names),
        "coordinate_system": spec.coordinate_system,
        "truth_provenance": list(spec.truth_provenance),
        "source_result": {
            "path": spec.result_path,
            "sha256": _sha256(result_path),
            "expected_sha256": spec.result_sha256,
        },
        "retained_sample_archive": {
            "path": spec.sample_path,
            "sha256": _sha256(sample_path),
            "expected_sha256": spec.sample_sha256,
            "byte_count": sample_path.stat().st_size,
            "shape": list(spec.sample_shape),
            "dtype": "float64",
        },
        "target_signature": spec.target_signature,
        "sampler_diagnostics": sampler,
        "thresholds": {
            "pass_minimum_p_truth": PASS_THRESHOLD,
            "severe_below_p_truth": SEVERE_THRESHOLD,
        },
        "minimum_p_truth": min(float(row["p_truth"]) for row in rows),
        "parameters_below_0_05": [
            row["parameter"] for row in rows if float(row["p_truth"]) < PASS_THRESHOLD
        ],
        "parameters_below_0_003": [
            row["parameter"] for row in rows if float(row["p_truth"]) < SEVERE_THRESHOLD
        ],
        "parameter_rows": list(rows),
        "next_action": (
            "stop_no_second_seed_needed"
            if classification.endswith("PASS") or classification.endswith("NONCENTRAL_TRUTH")
            else "plan_one_fresh_dataset_seed"
            if classification == "MARGINAL_ONE_SEED"
            else "stop_and_investigate"
        ),
    }


def _git_commit() -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    )
    return completed.stdout.strip()


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# NeuTra Retrospective Truth-Tail Diagnostic",
        "",
        f"Decision: `{payload['decision']}`",
        "",
        "| Cell | Central truth | Min p_truth | Classification | Next action |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for cell in payload["cells"]:
        lines.append(
            f"| `{cell['cell_id']}` | `{str(cell['fully_central_truth']).lower()}` | "
            f"`{cell['minimum_p_truth']:.6g}` | `{cell['classification']}` | "
            f"`{cell['next_action']}` |"
        )
    for cell in payload["cells"]:
        lines.extend(
            (
                "",
                f"## {cell['cell_id']}",
                "",
                "| Parameter | Truth | Prior center | Mean | SD | 95% interval | Mean ESS | p_truth | Status |",
                "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
            )
        )
        for row in cell["parameter_rows"]:
            interval = row["credible_interval_95"]
            lines.append(
                f"| `{row['parameter']}` | `{row['truth']:.8g}` | `{row['prior_center']:.8g}` | "
                f"`{row['posterior_mean']:.8g}` | `{row['posterior_sd']:.8g}` | "
                f"`[{interval[0]:.8g}, {interval[1]:.8g}]` | `{row['mean_ess']:.2f}` | "
                f"`{row['p_truth']:.6g}` | `{row['tail_status']}` |"
            )
    lines.extend(
        (
            "",
            "## Interpretation Boundary",
            "",
            "The posterior-tail quantity is not a frequentist p-value. A pass is a one-seed "
            "diagnostic only. Native TFP HMC divergence flags were unavailable; sampler "
            "validity uses each preserved run's declared energy-error divergence screen.",
            "",
            "This result does not establish calibration, coverage, universal reliability, "
            "filter exactness, method ranking, production readiness, or default readiness.",
            "",
        )
    )
    return "\n".join(lines)


def main() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("this reporting diagnostic requires CUDA_VISIBLE_DEVICES=-1")
    args = _parse_args()
    output_root = (REPO_ROOT / args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"output root must be fresh: {output_root}")
    output_root.mkdir(parents=True)
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()

    import tensorflow as tf
    import tensorflow_probability as tfp

    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("GPU unexpectedly visible during CPU-only diagnostic")
    self_check = _self_check(tf)
    cells = tuple(_analyze_cell(tf, tfp, spec) for spec in _cell_specs())
    marginal = [cell["cell_id"] for cell in cells if cell["classification"] == "MARGINAL_ONE_SEED"]
    severe = [cell["cell_id"] for cell in cells if cell["classification"] == "ONE_SEED_DIAGNOSTIC_FAILURE"]
    if severe:
        decision = "RETROSPECTIVE_DIAGNOSTIC_FAILURE_REQUIRES_INVESTIGATION"
    elif marginal:
        decision = "RETROSPECTIVE_DIAGNOSTIC_MARGINAL_SECOND_SEED_NOMINATED"
    else:
        decision = "RETROSPECTIVE_DIAGNOSTIC_COMPLETE_NO_SECOND_SEED_NEEDED"
    completed_at = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "schema": "bayesfilter.neutra_retrospective_truth_tail.v1",
        "decision": decision,
        "completed": True,
        "self_check": self_check,
        "definition": {
            "smoothed_ecdf": "(count_less + 0.5*count_equal + 0.5)/(N+1)",
            "p_truth": "2*min(F_truth,1-F_truth)",
            "interpretation": "posterior-tail diagnostic, not a frequentist p-value",
        },
        "thresholds": {"pass": PASS_THRESHOLD, "severe": SEVERE_THRESHOLD},
        "cells": list(cells),
        "counts": {
            "configurations": len(cells),
            "fully_central_truth_configurations": sum(bool(cell["fully_central_truth"]) for cell in cells),
            "parameters": sum(int(cell["parameter_count"]) for cell in cells),
            "marginal_configurations": len(marginal),
            "severe_failure_configurations": len(severe),
        },
        "second_seed_nominated": marginal,
        "nonclaims": [
            "one preserved dataset seed per configuration only",
            "not calibrated coverage evidence",
            "not a frequentist hypothesis-test p-value",
            "not filter exactness or ranking evidence",
            "not sampler superiority evidence",
            "not universal reliability, production, or default-readiness evidence",
        ],
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "elapsed_seconds": time.monotonic() - started,
    }
    payload["artifact_hash"] = f"sha256:{_stable_hash(payload)}"
    result_path = output_root / "result.json"
    markdown_path = output_root / "result.md"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(payload), encoding="utf-8")

    manifest = {
        "schema": "bayesfilter.neutra_retrospective_truth_tail_run_manifest.v1",
        "git_commit": _git_commit(),
        "git_worktree_dirty": bool(
            subprocess.run(("git", "status", "--porcelain"), cwd=REPO_ROOT, check=True, capture_output=True).stdout
        ),
        "command": (
            "CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/bin/conda run -n tf-gpu "
            "python docs/benchmarks/compute_neutra_retrospective_truth_tail.py "
            f"--output-root {args.output_root}"
        ),
        "environment": "tf-gpu conda environment",
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "tensorflow_probability_version": tfp.__version__,
        "cpu_gpu_status": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_intentionally_hidden": True,
            "visible_physical_gpus": [],
            "execution_role": "reporting_only_no_target_training_or_hmc",
        },
        "data_version": "preserved source result and retained archive hashes in result.json",
        "random_seeds": "none; deterministic replay of retained samples",
        "wall_time_seconds": payload["elapsed_seconds"],
        "output_artifacts": {
            "result_json": str(result_path.relative_to(REPO_ROOT)),
            "result_json_sha256": _sha256(result_path),
            "result_markdown": str(markdown_path.relative_to(REPO_ROOT)),
            "result_markdown_sha256": _sha256(markdown_path),
        },
        "plan_file": PLAN_PATH,
        "result_file": "docs/plans/bayesfilter-neutra-retrospective-truth-tail-diagnostic-result-2026-07-17.md",
    }
    manifest_path = output_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "result": str(result_path), "cells": [
        {"cell_id": cell["cell_id"], "classification": cell["classification"], "minimum_p_truth": cell["minimum_p_truth"]}
        for cell in cells
    ]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
