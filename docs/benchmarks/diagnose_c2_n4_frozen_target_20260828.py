"""C2 n=4 frozen-target root-cause diagnostics (2026-08-28 plan).

Stage ``heldout-t3`` runs the unchanged attempt05 candidate only through t=3,
captures the already fitted t=3 TT, and evaluates it on one disjoint
Christoffel scramble without refitting. The single-scramble result is a
nomination diagnostic, not a formal cause classification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Mapping

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "docs/plans/bayesfilter-n4-root-cause-diagnostic-plan-2026-08-28.md"
REFERENCE = (
    ROOT
    / "docs/benchmarks/artifacts/c2_completion_20260824/attempt05/reference_n4_s42.json"
)
ATTEMPT05_CELL = (
    ROOT
    / "docs/benchmarks/artifacts/c2_completion_20260824/attempt05/"
    "cell_n4_d6_r6_s42_w32.json"
)
STAGE0_TEST = ROOT / "tests/highdim/test_c2_gaussian_frozen_target_diagnostics.py"
ENGINE_SOURCE = ROOT / "bayesfilter/highdim/squared_tt_engine_gaussian_xla_tf.py"
FIXTURE_SOURCE = ROOT / "docs/benchmarks/sv_fixture_c2_20260826.py"

N = 4
DEGREE = 6
RANK = 6
ROWS = 8192
SWEEPS = 32
RIDGE = 1e-10
CONFIGURED_TAU = 1e-6
MODEL_SEED = 52
OBS_SEED = 42
CONFIG_SEED = 98000 + 100 * N + 10 * DEGREE + RANK
CAPTURE_STEP = 3
RUN_HORIZON = CAPTURE_STEP + 1
ALPHA_MAX = 0.8
NU_MARGIN_CAP = 12.0
HELDOUT_ROW_SEED = (CONFIG_SEED, 910_003)
SCIENTIFIC_BAR = 0.0025
TRUST_BASIS = "owner_designated_managed_session_visible_gpu_trusted"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--stage", choices=("heldout-t3",), default="heldout-t3")
    parser.add_argument("--jit-compile", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_json(path: Path, value: Mapping | list) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _workspace_state() -> dict:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    tracked_diff = subprocess.run(
        ["git", "diff", "--binary", "--no-ext-diff", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    digest = hashlib.sha256()
    digest.update(status)
    digest.update(tracked_diff)
    untracked = []
    for entry in status.split(b"\0"):
        if not entry.startswith(b"?? "):
            continue
        relative = entry[3:].decode("utf-8", errors="surrogateescape")
        path = ROOT / relative
        if path.is_file():
            file_hash = _sha256_file(path)
            digest.update(relative.encode("utf-8", errors="surrogateescape"))
            digest.update(file_hash.encode("ascii"))
            untracked.append({"path": relative, "sha256": file_hash})
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "git_commit": commit,
        "git_status_porcelain_sha256": _sha256_bytes(status),
        "tracked_diff_sha256": _sha256_bytes(tracked_diff),
        "workspace_state_sha256": digest.hexdigest(),
        "run_critical_untracked_files": untracked,
    }


def _run_stage0_checks(output_root: Path) -> dict:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        str(STAGE0_TEST.relative_to(ROOT)),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    environment["MPLCONFIGDIR"] = "/tmp/matplotlib-c2-stage0-artifact"
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
    (output_root / "stage0_tests.log").write_text(log, encoding="utf-8")
    record = {
        "classification": "cpu_only_mechanics_not_production_evidence",
        "cuda_visible_devices": "-1",
        "command": shlex.join(command),
        "exit_code": process.returncode,
        "wall_seconds": wall,
        "log_path": "stage0_tests.log",
        "log_sha256": _sha256_bytes(log.encode("utf-8")),
    }
    _write_json(output_root / "stage0_verification.json", record)
    if process.returncode != 0:
        raise RuntimeError("Stage 0 focused checks failed; see stage0_tests.log")
    return record


def _shell_table(tf, evaluation: Mapping[str, object], weights) -> list[dict]:
    u_max = evaluation["row_u_abs_max"]
    target = evaluation["row_target_energy"]
    residual = evaluation["row_residual_energy"]
    prediction = evaluation["row_prediction_energy"]
    total_target = tf.reduce_sum(weights * target)
    total_residual = tf.reduce_sum(weights * residual)
    total_prediction = tf.reduce_sum(weights * prediction)
    rows = []
    definitions = (
        ("[0,2]", u_max <= 2.0),
        ("(2,4]", tf.logical_and(u_max > 2.0, u_max <= 4.0)),
        ("(4,inf)", u_max > 4.0),
    )
    for label, mask in definitions:
        mask_f = tf.cast(mask, tf.float64)
        target_mass = tf.reduce_sum(weights * target * mask_f)
        residual_mass = tf.reduce_sum(weights * residual * mask_f)
        prediction_mass = tf.reduce_sum(weights * prediction * mask_f)
        rows.append(
            {
                "shell": label,
                "row_count": int(tf.reduce_sum(tf.cast(mask, tf.int64)).numpy()),
                "weight_mass": float(tf.reduce_sum(weights * mask_f).numpy()),
                "target_energy": float(target_mass.numpy()),
                "target_energy_fraction": float((target_mass / total_target).numpy()),
                "residual_energy": float(residual_mass.numpy()),
                "residual_energy_fraction": float(
                    (residual_mass / total_residual).numpy()
                ),
                "prediction_energy": float(prediction_mass.numpy()),
                "prediction_energy_fraction": float(
                    (prediction_mass / total_prediction).numpy()
                ),
            }
        )
    return rows


def _evaluation_record(tf, evaluation: Mapping[str, object], weights) -> dict:
    scalar_names = (
        "z_t",
        "z_h_direct",
        "z_h_factored",
        "z_h_qmc",
        "counting_residual",
        "emitted_rms",
        "rho_h_qmc",
        "rho_h_exact_denominator",
        "reverse_triangle_bound_valid",
        "reverse_triangle_log_lower",
        "reverse_triangle_log_upper",
        "fit_log_ratio_exact",
        "fit_log_ratio_qmc",
        "gram_vs_qmc_log_gap",
        "recomputed_shift_delta",
        "expanded_weight_sum",
        "target_all_finite",
        "target_log_g_min",
        "target_log_g_max",
        "target_log_f_min",
        "target_log_f_max",
        "target_sqrt_target_min",
        "target_sqrt_target_max",
        "target_branch_closure_relative_max",
        "target_floor_dominance_fraction",
    )
    record = {name: float(evaluation[name].numpy()) for name in scalar_names}
    record["shells"] = _shell_table(tf, evaluation, weights)
    return record


def _save_snapshot(tf, snapshot, output_root: Path, snapshot_api) -> dict:
    metadata, tensors = snapshot_api.gaussian_xla_frozen_snapshot_parts(snapshot)
    snapshot_dir = output_root / "snapshot_tensors"
    snapshot_dir.mkdir()
    tensor_files = {}
    for name, tensor in tensors.items():
        relative = Path("snapshot_tensors") / f"{name}.tensor"
        tf.io.write_file(
            str(output_root / relative), tf.io.serialize_tensor(tensor)
        )
        tensor_files[name] = {
            "path": str(relative),
            "sha256": _sha256_file(output_root / relative),
        }
    metadata["snapshot_fingerprint"] = (
        snapshot_api.gaussian_xla_frozen_snapshot_fingerprint(snapshot)
    )
    metadata["tensor_files"] = tensor_files
    _write_json(output_root / "snapshot_metadata.json", metadata)
    return {
        "metadata_path": "snapshot_metadata.json",
        "metadata_sha256": _sha256_file(output_root / "snapshot_metadata.json"),
        "fingerprint": metadata["snapshot_fingerprint"],
        "tensor_count": len(tensors),
    }


def _result_markdown(result: Mapping[str, object]) -> str:
    decomposition = result["decomposition"]
    heldout = result["heldout_evaluation"]
    decision = result["decision"]
    shell_lines = "\n".join(
        "| {shell} | {row_count} | {weight_mass:.6g} | {target_energy_fraction:.6g} | "
        "{residual_energy_fraction:.6g} |".format(**row)
        for row in heldout["shells"]
    )
    return f"""# C2 n=4 Frozen-Target Stage 1 Result

**Date:** 2026-08-28  
**Status:** {decision['status']}  
**Plan:** `docs/plans/bayesfilter-n4-root-cause-diagnostic-plan-2026-08-28.md`

## Result

The t=3 fitted target was evaluated on one independent 8,192-row
Christoffel scramble with the training shift frozen. This is a gross-failure
nomination test, not a formal cause classification.

| Quantity | Value (nat unless stated) |
|---|---:|
| Corrected engine increment | {decomposition['engine_corrected_increment']:+.12g} |
| Screened PF mean increment | {decomposition['pf_increment_mean']:+.12g} |
| Observed engine-minus-PF error | {decomposition['observed_error']:+.12g} |
| Fit term, log Z_H - log Z_T | {decomposition['e_fit']:+.12g} |
| Approximate-state increment | {decomposition['approximate_state_increment']:+.12g} |
| State term versus PF | {decomposition['e_state']:+.12g} |
| Algebraic closure residual | {decomposition['closure']:+.3e} |
| Held-out counting RMS | {heldout['emitted_rms']:.12g} |
| Held-out rho_H under held-out measure | {heldout['rho_h_qmc']:.12g} |
| Exact-Gram versus held-out H-energy gap | {heldout['gram_vs_qmc_log_gap']:+.12g} |

Decision: **{decision['headline']}**

{decision['interpretation']}

## Conditional Shell Check

| Infinity-norm shell | Rows | Weight mass | Target-energy fraction | Residual-energy fraction |
|---|---:|---:|---:|---:|
{shell_lines}

These shell values are explanatory only. They are not tuning targets or
promotion criteria.

## Evidence Status

| Question | Status |
|---|---|
| Stage 0 shared target/capture checks | PASS |
| Hard validity veto | {decision['hard_veto_status']} |
| Statistically supported root-cause classification | No; one scramble and no PF per-step SE |
| Descriptive nomination | {decision['nomination']} |
| Default readiness | Not evaluated |
| Next evidence | {decision['next_evidence']} |

## Nonconclusions

This result does not establish a minimum rank, a route-wide n=4 failure, a
repair, a numerical-policy change, HMC readiness, or default readiness. It
does not distinguish a poor inherited state from a poor hint inside the state
term.

## Post-Run Red Team

The strongest alternative explanation is that a poor inherited state makes
the contemporaneous target unusually difficult, so a large fit term and a
large state term can coexist. Four independent scrambles at t=2, t=3, and t=4,
with PF per-step replicate uncertainty, are required to measure that split.
"""


def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_root).resolve()
    allowed_root = (
        ROOT / "docs/benchmarks/artifacts/c2_n4_root_cause_20260828"
    ).resolve()
    if output_root.parent != allowed_root:
        raise ValueError(f"output root must be a direct child of {allowed_root}")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output root: {output_root}")
    output_root.mkdir(parents=True)

    campaign_started = time.perf_counter()
    workspace = _workspace_state()
    stage0 = _run_stage0_checks(output_root)

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "docs/benchmarks"))
    import tensorflow as tf
    import tensorflow_probability as tfp

    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    if not args.jit_compile:
        raise ValueError("serious Stage 1 execution requires --jit-compile")
    logical_gpus = tuple(tf.config.list_logical_devices("GPU"))
    if not logical_gpus:
        raise RuntimeError("no logical TensorFlow GPU after memory-policy setup")
    with tf.device("/GPU:0"):
        placement_probe = tf.reduce_sum(tf.ones([32], tf.float64))
    if "GPU" not in placement_probe.device.upper():
        raise RuntimeError(f"TensorFlow placement probe was not on GPU: {placement_probe.device}")

    import sv_fixture_c2_20260826 as sv
    from bayesfilter.highdim.squared_tt_engine_gaussian_tf import (
        _christoffel_rows,
        student_t_nu_criterion,
    )
    from bayesfilter.highdim.squared_tt_engine_v0_tf import EngineConfig
    import bayesfilter.highdim.squared_tt_engine_gaussian_xla_tf as snapshot_api

    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    if not reference.get("valid"):
        raise ValueError("attempt05 PF comparator is not valid")
    pf_increment = float(
        reference["arms"]["800000"]["per_step_mean"][CAPTURE_STEP]
    )
    nu = student_t_nu_criterion(ALPHA_MAX, NU_MARGIN_CAP)
    model = sv.sv_model(N, MODEL_SEED)
    observations_all = sv.sv_simulate(model, 20, OBS_SEED)
    observations = tf.constant(observations_all[:RUN_HORIZON], tf.float64)
    adapter = sv.sv_adapter(model)
    initial_hint_raw, predictive_hint_raw = sv.sv_gh_hint_factory(model, gh_points=9)
    alpha_values = []

    def _record_alpha(covariance):
        minimum = tf.reduce_min(tf.linalg.eigvalsh(covariance[:N, :N]))
        alpha_values.append(float((1.0 - minimum / sv.SIGMA**2).numpy()))

    def initial_hint(observation):
        mean, covariance = initial_hint_raw(observation)
        _record_alpha(covariance)
        return mean, covariance

    def predictive_hint(time_index, observation):
        mean, covariance = predictive_hint_raw(time_index, observation)
        _record_alpha(covariance)
        return mean, covariance

    config = EngineConfig(
        basis_degree=DEGREE,
        rank=RANK,
        row_count=ROWS,
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
        "observation_prefix_sha256": _sha256_bytes(
            bytes(tf.io.serialize_tensor(observations).numpy())
        ),
        "config": {
            "n": N,
            "degree": DEGREE,
            "rank": RANK,
            "rows": ROWS,
            "sweeps": SWEEPS,
            "ridge": RIDGE,
            "config_seed": CONFIG_SEED,
            "defensive_nu": nu,
        },
    }
    run_identity = _sha256_bytes(
        json.dumps(run_identity_payload, sort_keys=True).encode("utf-8")
    )

    fit_started = time.perf_counter()
    value, diagnostics, snapshots = (
        snapshot_api.run_value_filter_branch_axis_gaussian_xla_diagnostic(
            adapter,
            observations,
            config,
            predictive_moment_hint=predictive_hint,
            initial_moment_hint=initial_hint,
            capture_steps=(CAPTURE_STEP,),
            run_identity=run_identity,
            defensive_nu=nu,
        )
    )
    fit_wall = time.perf_counter() - fit_started
    snapshot = snapshots[CAPTURE_STEP]
    snapshot_artifact = _save_snapshot(tf, snapshot, output_root, snapshot_api)

    training_started = time.perf_counter()
    training_evaluation = snapshot_api.evaluate_gaussian_xla_frozen_transition(
        snapshot,
        adapter,
        snapshot.training_rows,
        snapshot.training_weights,
        jit_compile=True,
    )
    training_eval_wall = time.perf_counter() - training_started
    heldout_rows, heldout_weights, heldout_ess = _christoffel_rows(
        config,
        ROWS,
        2 * N,
        HELDOUT_ROW_SEED,
        DEGREE,
    )
    if HELDOUT_ROW_SEED == snapshot.training_row_seed:
        raise ValueError("held-out and training row seeds must differ")
    if bool(tf.reduce_all(heldout_rows == snapshot.training_rows).numpy()):
        raise ValueError("held-out and training rows overlap exactly")
    heldout_started = time.perf_counter()
    heldout_evaluation = snapshot_api.evaluate_gaussian_xla_frozen_transition(
        snapshot,
        adapter,
        heldout_rows,
        heldout_weights,
        jit_compile=True,
    )
    heldout_eval_wall = time.perf_counter() - heldout_started

    training_record = _evaluation_record(
        tf, training_evaluation, snapshot.training_weights
    )
    heldout_record = _evaluation_record(tf, heldout_evaluation, heldout_weights)
    z_t = heldout_evaluation["z_t"]
    e_fit = tf.math.log(snapshot.z_h) - tf.math.log(z_t)
    approximate_state_increment = (
        snapshot.frozen_shift
        + tf.math.log(z_t)
        - tf.math.log(snapshot.z_complete_previous)
    )
    e_state = approximate_state_increment - tf.constant(pf_increment, tf.float64)
    observed_error = snapshot.corrected_increment - tf.constant(
        pf_increment, tf.float64
    )
    closure = observed_error - e_fit - e_state
    decomposition = {
        "time_index": CAPTURE_STEP,
        "engine_raw_increment": float(snapshot.raw_increment.numpy()),
        "engine_corrected_increment": float(snapshot.corrected_increment.numpy()),
        "pf_increment_mean": pf_increment,
        "pf_per_step_se": None,
        "observed_error": float(observed_error.numpy()),
        "z_h": float(snapshot.z_h.numpy()),
        "z_t": float(z_t.numpy()),
        "e_fit": float(e_fit.numpy()),
        "approximate_state_increment": float(approximate_state_increment.numpy()),
        "e_state": float(e_state.numpy()),
        "closure": float(closure.numpy()),
    }
    gross_fit = abs(decomposition["e_fit"]) >= 0.1
    gross_state = abs(decomposition["e_state"]) >= 0.1
    if gross_fit and gross_state:
        nomination = "both_fit_and_state_terms_are_gross_on_this_scramble"
        headline = "Stage 1 nominates a recursive combination, not a single cause."
        interpretation = (
            "Both decomposition terms are descriptively material on the one held-out "
            "scramble. The formal multi-scramble decomposition must measure t=2, t=3, "
            "and t=4 before either mechanism is assigned causal status."
        )
    elif gross_fit:
        nomination = "contemporaneous_fit_generalization"
        headline = "Stage 1 nominates contemporaneous fit/generalization error."
        interpretation = (
            "The held-out fitted-normalizer discrepancy is gross on this scramble. "
            "Stage 2 is required before classifying it as the root cause."
        )
    elif gross_state:
        nomination = "inherited_state_hint_or_target"
        headline = "Stage 1 nominates inherited state/hint error."
        interpretation = (
            "The fitted-normalizer term is small on this scramble while the state term "
            "is gross. Stage 2 must move backward to t=2 and quantify uncertainty."
        )
    else:
        nomination = "identity_or_comparator_audit_required"
        headline = "Stage 1 did not reproduce a gross component."
        interpretation = (
            "A large observed discrepancy with two small components would veto the "
            "diagnostic identity; otherwise the one-scramble evidence is unresolved."
        )
    hard_vetoes = []
    if abs(decomposition["closure"]) > 5e-12:
        hard_vetoes.append("decomposition_closure")
    if heldout_record["target_all_finite"] != 1.0:
        hard_vetoes.append("non_finite_heldout_target")
    if heldout_record["target_branch_closure_relative_max"] > 5e-12:
        hard_vetoes.append("branch_target_closure")
    if abs(training_record["emitted_rms"] - float(snapshot.weighted_fit_rms.numpy())) > (
        5e-12 * max(1.0, abs(float(snapshot.weighted_fit_rms.numpy())))
    ):
        hard_vetoes.append("training_evaluator_rms_parity")
    if hard_vetoes:
        nomination = "none_due_to_hard_veto"
        headline = "Stage 1 is invalid because a hard diagnostic veto fired."
        interpretation = "Return to Stage 0 before running another research diagnostic."
    decision = {
        "status": "BLOCKED" if hard_vetoes else "STAGE_1_COMPLETE_STAGE_2_TRIGGERED",
        "headline": headline,
        "interpretation": interpretation,
        "hard_vetoes": hard_vetoes,
        "hard_veto_status": "FAIL: " + ", ".join(hard_vetoes) if hard_vetoes else "PASS",
        "nomination": nomination,
        "next_evidence": (
            "repair Stage 0"
            if hard_vetoes
            else "four-scramble t=2/t=3/t=4 decomposition with PF per-step uncertainty"
        ),
        "statistically_supported_ranking": False,
        "default_readiness": False,
    }
    corrected_prefix_total = float(value.numpy()) - sum(
        math.log1p(row["tau_t"]) for row in diagnostics
    )
    result = {
        "schema_id": "c2_n4_frozen_target_stage1_result_v1",
        "stage": args.stage,
        "run_identity": run_identity,
        "run_identity_payload": run_identity_payload,
        "scope": {
            "n": N,
            "degree": DEGREE,
            "rank": RANK,
            "rows": ROWS,
            "sweeps": SWEEPS,
            "ridge": RIDGE,
            "configured_tau": CONFIGURED_TAU,
            "model_seed": MODEL_SEED,
            "observation_seed": OBS_SEED,
            "config_seed": CONFIG_SEED,
            "capture_step": CAPTURE_STEP,
            "run_horizon": RUN_HORIZON,
            "defensive_nu": nu,
            "training_row_seed": list(snapshot.training_row_seed),
            "heldout_row_seed": list(HELDOUT_ROW_SEED),
            "heldout_row_ess": heldout_ess,
            "alpha_max_seen": max(alpha_values),
        },
        "prefix_corrected_total": corrected_prefix_total,
        "per_step_diagnostics": diagnostics,
        "training_evaluation": training_record,
        "heldout_evaluation": heldout_record,
        "decomposition": decomposition,
        "decision": decision,
        "evidence_limits": {
            "scramble_count": 1,
            "pf_per_step_se_available": False,
            "formal_classification_allowed": False,
            "scientific_bar_nat_per_step": SCIENTIFIC_BAR,
        },
        "snapshot": snapshot_artifact,
    }
    _write_json(output_root / "stage1_result.json", result)
    (output_root / "stage1_result.md").write_text(
        _result_markdown(result), encoding="utf-8"
    )

    allocator = tf.config.experimental.get_memory_info("GPU:0")
    input_hashes = {
        str(path.relative_to(ROOT)): _sha256_file(path)
        for path in (PLAN, REFERENCE, ATTEMPT05_CELL, ENGINE_SOURCE, FIXTURE_SOURCE, STAGE0_TEST)
    }
    command = (
        f"TF_FORCE_GPU_ALLOW_GROWTH={os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH')} "
        f"CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER', 'unset')} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')} "
        + shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    )
    manifest = {
        "schema_id": "c2_n4_root_cause_run_manifest_v1",
        "plan": str(PLAN.relative_to(ROOT)),
        "result_json": "stage1_result.json",
        "result_markdown": "stage1_result.md",
        "command": command,
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
        "seeds": result["scope"],
        "wall_seconds": {
            "fit_and_capture": fit_wall,
            "training_evaluation": training_eval_wall,
            "heldout_evaluation": heldout_eval_wall,
            "campaign_total": time.perf_counter() - campaign_started,
        },
        "input_sha256": input_hashes,
        "output_sha256": {
            "stage0_verification.json": _sha256_file(output_root / "stage0_verification.json"),
            "stage0_tests.log": _sha256_file(output_root / "stage0_tests.log"),
            "snapshot_metadata.json": _sha256_file(output_root / "snapshot_metadata.json"),
            "stage1_result.json": _sha256_file(output_root / "stage1_result.json"),
            "stage1_result.md": _sha256_file(output_root / "stage1_result.md"),
        },
        "attempt_budget": {
            "maximum_fitted_targets": 30,
            "consumed_fitted_targets": 1,
            "maximum_gpu_hours": 6,
            "maximum_cpu_hours": 4,
        },
    }
    _write_json(output_root / "run_manifest.json", manifest)
    print(json.dumps({"decision": decision, "decomposition": decomposition}, indent=2))


if __name__ == "__main__":
    main()
