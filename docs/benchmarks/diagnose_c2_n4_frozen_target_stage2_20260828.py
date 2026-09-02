"""Formal C2 n=4 frozen-state decomposition (Stage 2, 2026-08-28)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time
from typing import Mapping, Sequence

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "docs/benchmarks"
sys.path.insert(0, str(BENCHMARK_DIR))
import diagnose_c2_n4_frozen_target_20260828 as stage1_support

PLAN = ROOT / "docs/plans/bayesfilter-n4-root-cause-diagnostic-plan-2026-08-28.md"
REFERENCE = (
    ROOT
    / "docs/benchmarks/artifacts/c2_completion_20260824/attempt05/reference_n4_s42.json"
)
ENGINE_SOURCE = ROOT / "bayesfilter/highdim/squared_tt_engine_gaussian_xla_tf.py"
FIXTURE_SOURCE = ROOT / "docs/benchmarks/sv_fixture_c2_20260826.py"
STAGE0_TEST = ROOT / "tests/highdim/test_c2_gaussian_frozen_target_diagnostics.py"

N = 4
DEGREE = 6
RANK = 6
SWEEPS = 32
RIDGE = 1e-10
CONFIGURED_TAU = 1e-6
MODEL_SEED = 52
OBS_SEED = 42
CONFIG_SEED = 98000 + 100 * N + 10 * DEGREE + RANK
CAPTURE_STEPS = (2, 3, 4)
RUN_HORIZON = 5
BASE_ROWS = 8192
ROW_LADDER = (8192, 16384, 32768)
SCRAMBLES = 4
QMC_T_CRITICAL_95 = 3.182446305284263
PF_T_CRITICAL_95 = 2.2621571627409915
QMC_HALF_WIDTH_LIMIT = 0.00125
SCIENTIFIC_BAR = 0.0025
ALPHA_MAX = 0.8
NU_MARGIN_CAP = 12.0
TRUST_BASIS = "owner_designated_managed_session_visible_gpu_trusted"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", choices=("formal", "pf-reference"), default="formal")
    parser.add_argument("--jit-compile", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _pf_reference(output_root: Path) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(BENCHMARK_DIR))
    import sv_fixture_c2_20260826 as sv

    output = output_root / "pf_per_step_reference.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    started = time.perf_counter()
    model = sv.sv_model(N, MODEL_SEED)
    observations = sv.sv_simulate(model, 20, OBS_SEED)[:RUN_HORIZON]
    result = sv.sv_particle_reference(
        model,
        observations,
        n_particles=800_000,
        replicates=10,
        seed=31 + OBS_SEED,
    )
    result.update(
        {
            "schema_id": "c2_n4_pf_per_step_uncertainty_v1",
            "classification": "cpu_only_independent_diagnostic_reference",
            "cuda_visible_devices": "-1",
            "n": N,
            "model_seed": MODEL_SEED,
            "observation_seed": OBS_SEED,
            "horizon": RUN_HORIZON,
            "seed": 31 + OBS_SEED,
            "wall_seconds": time.perf_counter() - started,
        }
    )
    stage1_support._write_json(output, result)
    print(json.dumps({"per_step_mean": result["per_step_mean"], "per_step_se": result["per_step_se"]}))


def _run_pf_subprocess(output_root: Path) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--output-root",
        str(output_root),
        "--mode",
        "pf-reference",
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    environment["MPLCONFIGDIR"] = "/tmp/matplotlib-c2-stage2-pf"
    started = time.perf_counter()
    process = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    wall = time.perf_counter() - started
    log = process.stdout + process.stderr
    (output_root / "pf_reference.log").write_text(log, encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError("PF uncertainty subprocess failed; see pf_reference.log")
    record = json.loads(
        (output_root / "pf_per_step_reference.json").read_text(encoding="utf-8")
    )
    record["subprocess_wall_seconds"] = wall
    record["subprocess_command"] = shlex.join(command)
    return record


def _sample_stats(values: Sequence[float]) -> dict:
    values = [float(value) for value in values]
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    standard_error = standard_deviation / math.sqrt(len(values))
    half_width = QMC_T_CRITICAL_95 * standard_error
    return {
        "n": len(values),
        "values": values,
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "half_width_95": half_width,
        "ci95": [mean - half_width, mean + half_width],
    }


def _rotated_sobol_uniform(tf, count: int, dimension: int, seed: tuple[int, int]):
    base = tf.math.sobol_sample(dimension, count, dtype=tf.float64)
    shift = tf.random.stateless_uniform(
        [1, dimension], tf.constant(seed, tf.int32), dtype=tf.float64
    )
    values = tf.math.floormod(base + shift, 1.0)
    epsilon = tf.constant(1e-14, tf.float64)
    return tf.clip_by_value(values, epsilon, 1.0 - epsilon)


def _standard_normal_rows(tf, tfp, count: int, dimension: int, seed: tuple[int, int]):
    with tf.device("/CPU:0"):
        uniform = _rotated_sobol_uniform(tf, count, dimension, seed)
        rows = tfp.distributions.Normal(
            tf.constant(0.0, tf.float64), tf.constant(1.0, tf.float64)
        ).quantile(uniform)
        weights = tf.fill([count], tf.constant(1.0 / count, tf.float64))
    return rows, weights, float(count)


def _student_t_mixture_rows(
    tf,
    tfp,
    count: int,
    dimension: int,
    seed: tuple[int, int],
    nu: float,
):
    with tf.device("/CPU:0"):
        uniform = _rotated_sobol_uniform(tf, count, dimension + 1, seed)
        probabilities = uniform[:, :dimension]
        normal = tfp.distributions.Normal(
            tf.constant(0.0, tf.float64), tf.constant(1.0, tf.float64)
        )
        student = tfp.distributions.StudentT(
            df=tf.constant(nu, tf.float64),
            loc=tf.constant(0.0, tf.float64),
            scale=tf.constant(1.0, tf.float64),
        )
        normal_rows = normal.quantile(probabilities)
        student_rows = student.quantile(probabilities)
        choose_student = uniform[:, dimension] >= 0.5
        rows = tf.where(choose_student[:, None], student_rows, normal_rows)
        log_eta = tf.reduce_sum(normal.log_prob(rows), axis=1)
        log_student = tf.reduce_sum(student.log_prob(rows), axis=1)
        log_half = tf.constant(math.log(0.5), tf.float64)
        log_proposal = tf.reduce_logsumexp(
            tf.stack([log_half + log_eta, log_half + log_student], axis=0),
            axis=0,
        )
        weights = tf.exp(log_eta - log_proposal) / tf.cast(count, tf.float64)
        ess = tf.square(tf.reduce_sum(weights)) / tf.reduce_sum(tf.square(weights))
    return rows, weights, float(ess.numpy())


def _save_snapshot(tf, snapshot, root: Path, snapshot_api) -> dict:
    metadata, tensors = snapshot_api.gaussian_xla_frozen_snapshot_parts(snapshot)
    root.mkdir(parents=True)
    tensor_dir = root / "tensors"
    tensor_dir.mkdir()
    files = {}
    for name, tensor in tensors.items():
        path = tensor_dir / f"{name}.tensor"
        tf.io.write_file(str(path), tf.io.serialize_tensor(tensor))
        files[name] = {
            "path": str(path.relative_to(root.parent.parent)),
            "sha256": stage1_support._sha256_file(path),
        }
    metadata["snapshot_fingerprint"] = (
        snapshot_api.gaussian_xla_frozen_snapshot_fingerprint(snapshot)
    )
    metadata["tensor_files"] = files
    metadata_path = root / "metadata.json"
    stage1_support._write_json(metadata_path, metadata)
    return {
        "metadata_path": str(metadata_path.relative_to(root.parent.parent)),
        "metadata_sha256": stage1_support._sha256_file(metadata_path),
        "fingerprint": metadata["snapshot_fingerprint"],
        "tensor_count": len(tensors),
    }


def _one_evaluation(
    tf,
    snapshot_api,
    snapshot,
    adapter,
    rows,
    weights,
    *,
    row_seed: tuple[int, int],
    row_ess: float,
    pf_increment: float,
):
    evaluation = snapshot_api.evaluate_gaussian_xla_frozen_transition(
        snapshot, adapter, rows, weights, jit_compile=True
    )
    record = stage1_support._evaluation_record(tf, evaluation, weights)
    z_t = evaluation["z_t"]
    e_fit = tf.math.log(snapshot.z_h) - tf.math.log(z_t)
    state_increment = (
        snapshot.frozen_shift
        + tf.math.log(z_t)
        - tf.math.log(snapshot.z_complete_previous)
    )
    e_state = state_increment - tf.constant(pf_increment, tf.float64)
    observed = snapshot.corrected_increment - tf.constant(pf_increment, tf.float64)
    closure = observed - e_fit - e_state
    return {
        "row_seed": list(row_seed),
        "row_ess": row_ess,
        "weight_sum": float(tf.reduce_sum(weights).numpy()),
        "evaluation": record,
        "decomposition": {
            "log_z_t": float(tf.math.log(z_t).numpy()),
            "log_z_h_qmc": float(tf.math.log(evaluation["z_h_qmc"]).numpy()),
            "e_fit": float(e_fit.numpy()),
            "fit_under_row_measure": float(
                evaluation["fit_log_ratio_qmc"].numpy()
            ),
            "exact_gram_integration_gap": float(
                evaluation["gram_vs_qmc_log_gap"].numpy()
            ),
            "approximate_state_increment": float(state_increment.numpy()),
            "e_state": float(e_state.numpy()),
            "observed_error": float(observed.numpy()),
            "closure": float(closure.numpy()),
        },
    }


def _summarize_arm(records: Sequence[dict]) -> dict:
    keys = (
        "log_z_t",
        "log_z_h_qmc",
        "e_fit",
        "fit_under_row_measure",
        "exact_gram_integration_gap",
        "approximate_state_increment",
        "e_state",
        "observed_error",
        "closure",
    )
    return {
        key: _sample_stats([record["decomposition"][key] for record in records])
        for key in keys
    }


def _run_arm(
    tf,
    tfp,
    snapshot_api,
    snapshots,
    adapter,
    config,
    pf_means,
    *,
    arm: str,
    row_count: int,
    nu: float,
):
    records_by_step = {step: [] for step in CAPTURE_STEPS}
    for step in CAPTURE_STEPS:
        for scramble in range(SCRAMBLES):
            if arm == "christoffel":
                seed = (CONFIG_SEED, 920_000 + 100 * step + scramble)
                from bayesfilter.highdim.squared_tt_engine_gaussian_tf import (
                    _christoffel_rows,
                )

                rows, weights, ess = _christoffel_rows(
                    config, row_count, 2 * N, seed, DEGREE
                )
            elif arm == "standard_normal":
                seed = (CONFIG_SEED + 1, 930_000 + 100 * step + scramble)
                rows, weights, ess = _standard_normal_rows(
                    tf, tfp, row_count, 2 * N, seed
                )
            elif arm == "student_t_mixture":
                seed = (CONFIG_SEED + 2, 940_000 + 100 * step + scramble)
                rows, weights, ess = _student_t_mixture_rows(
                    tf, tfp, row_count, 2 * N, seed, nu
                )
            else:
                raise ValueError(f"unknown integration arm {arm!r}")
            if seed == snapshots[step].training_row_seed:
                raise ValueError("diagnostic and training row seeds overlap")
            records_by_step[step].append(
                _one_evaluation(
                    tf,
                    snapshot_api,
                    snapshots[step],
                    adapter,
                    rows,
                    weights,
                    row_seed=seed,
                    row_ess=ess,
                    pf_increment=pf_means[step],
                )
            )
    return {
        "arm": arm,
        "row_count": row_count,
        "scramble_count": SCRAMBLES,
        "records": {str(step): records_by_step[step] for step in CAPTURE_STEPS},
        "summaries": {
            str(step): _summarize_arm(records_by_step[step])
            for step in CAPTURE_STEPS
        },
    }


def _material(mean: float, half_width: float) -> bool:
    interval_excludes_zero = abs(mean) > half_width
    return interval_excludes_zero and abs(mean) > SCIENTIFIC_BAR


def _classify_steps(christoffel: Mapping, pf_reference: Mapping) -> dict:
    classifications = {}
    pf_se = pf_reference["per_step_se"]
    for step in CAPTURE_STEPS:
        summary = christoffel["summaries"][str(step)]
        qmc_half = summary["log_z_t"]["half_width_95"]
        qmc_precision_pass = qmc_half <= QMC_HALF_WIDTH_LIMIT
        fit = summary["e_fit"]
        state = summary["e_state"]
        pf_half = PF_T_CRITICAL_95 * float(pf_se[step])
        state_half = state["half_width_95"] + pf_half
        descriptive_fit_material = _material(fit["mean"], fit["half_width_95"])
        descriptive_state_material = _material(state["mean"], state_half)
        fit_material = qmc_precision_pass and descriptive_fit_material
        state_material = qmc_precision_pass and descriptive_state_material
        if not qmc_precision_pass:
            label = "unresolved_qmc_precision"
        elif fit_material and state_material:
            label = "fit_and_state_material"
        elif fit_material:
            label = "fit_material_state_small_or_unresolved"
        elif state_material:
            label = "state_material_fit_small_or_unresolved"
        else:
            label = "neither_component_material_or_evidence_unresolved"
        classifications[str(step)] = {
            "classification": label,
            "qmc_precision_pass": qmc_precision_pass,
            "qmc_log_z_t_half_width_95": qmc_half,
            "qmc_log_z_t_half_width_95_limit": QMC_HALF_WIDTH_LIMIT,
            "fit_material": fit_material,
            "state_material": state_material,
            "descriptive_fit_material": descriptive_fit_material,
            "descriptive_state_material": descriptive_state_material,
            "fit_mean": fit["mean"],
            "fit_half_width_95": fit["half_width_95"],
            "state_mean": state["mean"],
            "state_qmc_half_width_95": state["half_width_95"],
            "pf_half_width_95": pf_half,
            "state_conservative_half_width_95": state_half,
        }
    return classifications


def _arm_disagreements(arms: Mapping[str, Mapping]) -> list[dict]:
    disagreements = []
    baseline = arms["christoffel"]
    for step in CAPTURE_STEPS:
        base = baseline["summaries"][str(step)]["log_z_t"]
        for name in ("standard_normal", "student_t_mixture"):
            other = arms[name]["summaries"][str(step)]["log_z_t"]
            difference = other["mean"] - base["mean"]
            combined_half_width = (
                other["half_width_95"] + base["half_width_95"]
            )
            disagreements.append(
                {
                    "time_index": step,
                    "comparator": name,
                    "log_z_t_difference": difference,
                    "combined_half_width_95": combined_half_width,
                    "intervals_disjoint": abs(difference) > combined_half_width,
                }
            )
    return disagreements


def _result_markdown(result: Mapping) -> str:
    classification_rows = []
    for step in CAPTURE_STEPS:
        row = result["step_classifications"][str(step)]
        classification_rows.append(
            f"| {step} | {row['fit_mean']:+.6f} +/- {row['fit_half_width_95']:.3g} | "
            f"{row['state_mean']:+.6f} +/- {row['state_conservative_half_width_95']:.3g} | "
            f"{row['classification']} |"
        )
    arm_rows = []
    for step in CAPTURE_STEPS:
        for name, arm in result["arms"].items():
            summary = arm["summaries"][str(step)]
            arm_rows.append(
                f"| {step} | {name} | {summary['log_z_t']['mean']:+.6f} +/- "
                f"{summary['log_z_t']['half_width_95']:.3g} | "
                f"{summary['fit_under_row_measure']['mean']:+.6f} | "
                f"{summary['exact_gram_integration_gap']['mean']:+.6f} |"
            )
    decision = result["decision"]
    return f"""# C2 n=4 Frozen-State Stage 2 Result

**Date:** 2026-08-28  
**Status:** {decision['status']}  
**Decision:** {decision['headline']}

The unchanged attempt05 configuration was fitted once through t=4. Full fitted
TTs and frozen pre-update states were captured at t=2, t=3, and t=4. Each
target was evaluated without refitting on four independent scrambles per row
law. PF uncertainty comes from ten fresh 800,000-particle replicate prefixes.

## Primary Decomposition

| t | Exact fit term (95% CI half-width) | State term (conservative 95% half-width) | Classification |
|---:|---:|---:|---|
{chr(10).join(classification_rows)}

## Integration Adversaries

`fit under row measure` is `log Z_H,QMC - log Z_T`; `exact-Gram gap` is
`log Z_H - log Z_H,QMC`. Their sum is the primary exact fit term.

| t | Arm | log Z_T (95% CI half-width) | Fit under row measure | Exact-Gram gap |
|---:|---|---:|---:|---:|
{chr(10).join(arm_rows)}

## Evidence Decision

{decision['interpretation']}

- Hard vetoes: {', '.join(decision['hard_vetoes']) if decision['hard_vetoes'] else 'none'}
- Statistically supported candidate ranking: no candidate ranking was attempted.
- Descriptive-only quantities: shell residuals, Gram-QMC gaps, RMS, row ESS,
  condition numbers, and cross-arm tail behavior.
- Default readiness: not evaluated.
- Next justified action: {decision['next_action']}.

## Nonconclusions

This result does not establish a minimum rank, a repaired algorithm, a new
ridge/floor/tau policy, HMC readiness, or default readiness. A fitted component
being wrong at these frozen steps does not by itself reject the route or prove
that rank six is insufficient.

## Post-Run Red Team

The strongest alternative explanation is that an inherited state or GH9 chart
makes the frozen target abnormally difficult, so the fit failure may be
downstream rather than initiating. A frozen-target fitter study followed by
backward state localization is needed if both components remain material.
"""


def _formal(output_root: Path, args: argparse.Namespace) -> None:
    allowed_parent = (
        ROOT / "docs/benchmarks/artifacts/c2_n4_root_cause_20260828"
    ).resolve()
    if output_root.parent != allowed_parent:
        raise ValueError(f"output root must be a direct child of {allowed_parent}")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)
    campaign_started = time.perf_counter()
    workspace = stage1_support._workspace_state()
    stage0 = stage1_support._run_stage0_checks(output_root)
    pf_reference = _run_pf_subprocess(output_root)

    sys.path.insert(0, str(ROOT))
    import tensorflow as tf
    import tensorflow_probability as tfp

    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    if not args.jit_compile:
        raise ValueError("formal Stage 2 requires --jit-compile")
    logical_gpus = tuple(tf.config.list_logical_devices("GPU"))
    if not logical_gpus:
        raise RuntimeError("no logical TensorFlow GPU")
    with tf.device("/GPU:0"):
        placement_probe = tf.reduce_sum(tf.ones([32], tf.float64))
    if "GPU" not in placement_probe.device.upper():
        raise RuntimeError("TensorFlow placement probe did not execute on GPU")

    import sv_fixture_c2_20260826 as sv
    from bayesfilter.highdim.squared_tt_engine_gaussian_tf import (
        student_t_nu_criterion,
    )
    from bayesfilter.highdim.squared_tt_engine_v0_tf import EngineConfig
    import bayesfilter.highdim.squared_tt_engine_gaussian_xla_tf as snapshot_api

    historical_reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    if not historical_reference.get("valid"):
        raise ValueError("historical screened PF comparator is invalid")
    historical_means = historical_reference["arms"]["800000"]["per_step_mean"]
    pf_mean_difference = max(
        abs(float(pf_reference["per_step_mean"][step]) - float(historical_means[step]))
        for step in range(RUN_HORIZON)
    )
    if pf_mean_difference > 1e-12:
        raise ValueError(
            "fresh PF prefix does not reproduce the screened reference means"
        )
    pf_means = [float(value) for value in pf_reference["per_step_mean"]]
    nu = student_t_nu_criterion(ALPHA_MAX, NU_MARGIN_CAP)
    model = sv.sv_model(N, MODEL_SEED)
    observations_all = sv.sv_simulate(model, 20, OBS_SEED)
    observations = tf.constant(observations_all[:RUN_HORIZON], tf.float64)
    adapter = sv.sv_adapter(model)
    initial_raw, predictive_raw = sv.sv_gh_hint_factory(model, gh_points=9)
    alpha_values = []

    def _record_alpha(covariance):
        minimum = tf.reduce_min(tf.linalg.eigvalsh(covariance[:N, :N]))
        alpha_values.append(float((1.0 - minimum / sv.SIGMA**2).numpy()))

    def initial_hint(observation):
        mean, covariance = initial_raw(observation)
        _record_alpha(covariance)
        return mean, covariance

    def predictive_hint(step, observation):
        mean, covariance = predictive_raw(step, observation)
        _record_alpha(covariance)
        return mean, covariance

    config = EngineConfig(
        basis_degree=DEGREE,
        rank=RANK,
        row_count=BASE_ROWS,
        sweeps=SWEEPS,
        ridge=RIDGE,
        tau=CONFIGURED_TAU,
        coordinate_half_width=3.0,
        seed=CONFIG_SEED,
        row_design="sobol",
    )
    run_identity_payload = {
        "model": "zc24_sv_vector_extension_v1",
        "model_seed": MODEL_SEED,
        "observation_seed": OBS_SEED,
        "observation_prefix_sha256": hashlib.sha256(
            bytes(tf.io.serialize_tensor(observations).numpy())
        ).hexdigest(),
        "config_seed": CONFIG_SEED,
        "capture_steps": list(CAPTURE_STEPS),
        "degree": DEGREE,
        "rank": RANK,
        "rows": BASE_ROWS,
        "sweeps": SWEEPS,
        "ridge": RIDGE,
        "defensive_nu": nu,
    }
    run_identity = hashlib.sha256(
        json.dumps(run_identity_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    fit_started = time.perf_counter()
    value, diagnostics, snapshots = (
        snapshot_api.run_value_filter_branch_axis_gaussian_xla_diagnostic(
            adapter,
            observations,
            config,
            predictive_moment_hint=predictive_hint,
            initial_moment_hint=initial_hint,
            capture_steps=CAPTURE_STEPS,
            run_identity=run_identity,
            defensive_nu=nu,
        )
    )
    fit_wall = time.perf_counter() - fit_started
    snapshot_records = {
        str(step): _save_snapshot(
            tf,
            snapshots[step],
            output_root / "snapshots" / f"t{step:02d}",
            snapshot_api,
        )
        for step in CAPTURE_STEPS
    }
    training_evaluations = {
        str(step): stage1_support._evaluation_record(
            tf,
            snapshot_api.evaluate_gaussian_xla_frozen_transition(
                snapshots[step],
                adapter,
                snapshots[step].training_rows,
                snapshots[step].training_weights,
                jit_compile=True,
            ),
            snapshots[step].training_weights,
        )
        for step in CAPTURE_STEPS
    }

    christoffel_ladder = []
    selected_christoffel = None
    integration_started = time.perf_counter()
    for row_count in ROW_LADDER:
        arm = _run_arm(
            tf,
            tfp,
            snapshot_api,
            snapshots,
            adapter,
            config,
            pf_means,
            arm="christoffel",
            row_count=row_count,
            nu=nu,
        )
        christoffel_ladder.append(arm)
        maximum_half_width = max(
            arm["summaries"][str(step)]["log_z_t"]["half_width_95"]
            for step in CAPTURE_STEPS
        )
        if maximum_half_width <= QMC_HALF_WIDTH_LIMIT:
            selected_christoffel = arm
            break
    if selected_christoffel is None:
        selected_christoffel = christoffel_ladder[-1]
    selected_rows = selected_christoffel["row_count"]
    standard_normal = _run_arm(
        tf,
        tfp,
        snapshot_api,
        snapshots,
        adapter,
        config,
        pf_means,
        arm="standard_normal",
        row_count=selected_rows,
        nu=nu,
    )
    student_t_mixture = _run_arm(
        tf,
        tfp,
        snapshot_api,
        snapshots,
        adapter,
        config,
        pf_means,
        arm="student_t_mixture",
        row_count=selected_rows,
        nu=nu,
    )
    integration_wall = time.perf_counter() - integration_started
    arms = {
        "christoffel": selected_christoffel,
        "standard_normal": standard_normal,
        "student_t_mixture": student_t_mixture,
    }
    step_classifications = _classify_steps(selected_christoffel, pf_reference)
    disagreements = _arm_disagreements(arms)
    hard_vetoes = []
    maximum_log_z_t_half_width = max(
        selected_christoffel["summaries"][str(step)]["log_z_t"]["half_width_95"]
        for step in CAPTURE_STEPS
    )
    if maximum_log_z_t_half_width > QMC_HALF_WIDTH_LIMIT:
        hard_vetoes.append("christoffel_qmc_uncertainty")
    maximum_closure = max(
        abs(record["decomposition"]["closure"])
        for arm in arms.values()
        for records in arm["records"].values()
        for record in records
    )
    if maximum_closure > 5e-12:
        hard_vetoes.append("decomposition_closure")
    if any(item["intervals_disjoint"] for item in disagreements):
        hard_vetoes.append("integration_arm_log_z_t_disagreement")
    if any(
        record["evaluation"]["target_all_finite"] != 1.0
        for arm in arms.values()
        for records in arm["records"].values()
        for record in records
    ):
        hard_vetoes.append("non_finite_target")
    fit_steps = [
        int(step) for step, row in step_classifications.items() if row["fit_material"]
    ]
    state_steps = [
        int(step) for step, row in step_classifications.items() if row["state_material"]
    ]
    if hard_vetoes:
        decision = {
            "status": "STAGE_2_UNRESOLVED",
            "headline": "Stage 2 hit a continuation veto and cannot classify the cause.",
            "interpretation": (
                "The frozen-state algebra closed, but integration uncertainty or "
                "cross-arm disagreement prevents a formal component classification."
            ),
            "next_action": "repair or extend the integration evidence within budget",
            "hard_vetoes": hard_vetoes,
        }
    elif fit_steps and state_steps:
        decision = {
            "status": "STAGE_2_FIT_AND_STATE_LOCALIZED",
            "headline": "Both fit and inherited-state terms are material at measured steps.",
            "interpretation": (
                "The contemporaneous exact fitted-normalizer error is material, and at "
                "least one inherited-state term also exceeds the bar after PF/QMC "
                "uncertainty. The fit term must be studied first, then the state term "
                "re-evaluated with that repair held fixed."
            ),
            "next_action": "Stage 3 frozen-target fitter study, then Stage 4 if state error remains",
            "hard_vetoes": [],
        }
    elif fit_steps:
        decision = {
            "status": "STAGE_2_FIT_LOCALIZED",
            "headline": "Contemporaneous fitted-normalizer error is the localized component.",
            "interpretation": (
                "The exact fit term is material with adequate QMC precision while the "
                "state term is not formally material under the current uncertainty."
            ),
            "next_action": "Stage 3 frozen-target fitter study",
            "hard_vetoes": [],
        }
    elif state_steps:
        decision = {
            "status": "STAGE_2_STATE_LOCALIZED",
            "headline": "Inherited state, hint, or target error is the localized component.",
            "interpretation": (
                "The state term is material while the contemporaneous fit term is not."
            ),
            "next_action": "Stage 4 backward state and hint validation",
            "hard_vetoes": [],
        }
    else:
        decision = {
            "status": "STAGE_2_NO_COMPONENT_LOCALIZED",
            "headline": "Neither component is formally localized.",
            "interpretation": "The evidence is adequate but does not isolate a material component.",
            "next_action": "audit comparator/reporting convention",
            "hard_vetoes": [],
        }
    corrected_prefix_total = float(value.numpy()) - sum(
        math.log1p(row["tau_t"]) for row in diagnostics
    )
    result = {
        "schema_id": "c2_n4_frozen_state_stage2_result_v1",
        "run_identity": run_identity,
        "run_identity_payload": run_identity_payload,
        "scope": {
            "n": N,
            "degree": DEGREE,
            "rank": RANK,
            "base_fit_rows": BASE_ROWS,
            "selected_integration_rows": selected_rows,
            "sweeps": SWEEPS,
            "ridge": RIDGE,
            "configured_tau": CONFIGURED_TAU,
            "model_seed": MODEL_SEED,
            "observation_seed": OBS_SEED,
            "config_seed": CONFIG_SEED,
            "capture_steps": list(CAPTURE_STEPS),
            "defensive_nu": nu,
            "student_t_mixture_weight": 0.5,
            "student_t_mixture_nu": nu,
            "alpha_max_seen": max(alpha_values),
        },
        "prefix_corrected_total": corrected_prefix_total,
        "per_step_diagnostics": diagnostics,
        "training_evaluations": training_evaluations,
        "pf_reference": {
            "path": "pf_per_step_reference.json",
            "sha256": stage1_support._sha256_file(
                output_root / "pf_per_step_reference.json"
            ),
            "fresh_vs_historical_max_abs_mean_difference": pf_mean_difference,
            "per_step_mean": pf_reference["per_step_mean"],
            "per_step_se": pf_reference["per_step_se"],
            "per_step_mean_covariance": pf_reference["per_step_mean_covariance"],
        },
        "christoffel_row_ladder": christoffel_ladder,
        "arms": arms,
        "arm_disagreements": disagreements,
        "step_classifications": step_classifications,
        "decision": decision,
        "diagnostic_checks": {
            "maximum_decomposition_closure_abs": maximum_closure,
            "maximum_christoffel_log_z_t_half_width_95": maximum_log_z_t_half_width,
            "qmc_half_width_limit": QMC_HALF_WIDTH_LIMIT,
        },
        "snapshots": snapshot_records,
        "inference_status": {
            "hard_veto_screen": "pass" if not hard_vetoes else "fail",
            "statistically_supported_ranking": False,
            "descriptive_only_differences": (
                "shell residuals, Gram-QMC gaps, RMS, row ESS, conditions, tail behavior"
            ),
            "default_readiness": False,
            "next_evidence_needed": decision["next_action"],
        },
    }
    stage1_support._write_json(output_root / "stage2_result.json", result)
    (output_root / "stage2_result.md").write_text(
        _result_markdown(result), encoding="utf-8"
    )

    allocator = tf.config.experimental.get_memory_info("GPU:0")
    input_paths = (
        PLAN,
        REFERENCE,
        ENGINE_SOURCE,
        FIXTURE_SOURCE,
        STAGE0_TEST,
        Path(__file__).resolve(),
        Path(stage1_support.__file__).resolve(),
    )
    manifest = {
        "schema_id": "c2_n4_root_cause_stage2_manifest_v1",
        "plan": str(PLAN.relative_to(ROOT)),
        "command": (
            f"TF_FORCE_GPU_ALLOW_GROWTH={os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH')} "
            f"CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER', 'unset')} "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')} "
            + shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
        ),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "python_executable": sys.executable,
        "tensorflow_version": tf.__version__,
        "tensorflow_probability_version": tfp.__version__,
        "dtype": "float64",
        "jit_compile": True,
        "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        "physical_memory_policy": memory_policy,
        "logical_gpus": [str(device) for device in logical_gpus],
        "placement_probe_device": placement_probe.device,
        "gpu_allocator_bytes": {key: int(value) for key, value in allocator.items()},
        "logical_device_memory_limit": None,
        "trust_basis": TRUST_BASIS,
        "stage0_verification": stage0,
        "workspace": workspace,
        "wall_seconds": {
            "pf_reference_subprocess": pf_reference["subprocess_wall_seconds"],
            "fit_and_capture": fit_wall,
            "independent_integrations": integration_wall,
            "campaign_total": time.perf_counter() - campaign_started,
        },
        "input_sha256": {
            str(path.relative_to(ROOT)): stage1_support._sha256_file(path)
            for path in input_paths
        },
        "output_sha256": {
            name: stage1_support._sha256_file(output_root / name)
            for name in (
                "stage0_verification.json",
                "stage0_tests.log",
                "pf_per_step_reference.json",
                "pf_reference.log",
                "stage2_result.json",
                "stage2_result.md",
            )
        },
        "attempt_budget": {
            "maximum_fitted_targets": 30,
            "consumed_fitted_targets_total": 2,
            "maximum_gpu_hours": 6,
            "maximum_cpu_hours": 4,
        },
    }
    stage1_support._write_json(output_root / "run_manifest.json", manifest)
    print(json.dumps({"decision": decision, "step_classifications": step_classifications}, indent=2))


def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_root).resolve()
    if args.mode == "pf-reference":
        _pf_reference(output_root)
        return
    _formal(output_root, args)


if __name__ == "__main__":
    main()
