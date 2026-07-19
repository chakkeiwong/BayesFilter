#!/usr/bin/env python3
"""Fresh smaller-step repair for the SSL-LSTM A4 HMC acquisition."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_HARNESS_PATH = Path(
    "docs/benchmarks/run_ssl_lstm_a4_hmc_acquisition_2026_07_14.py"
)
REPAIR_HARNESS_PATH = Path(
    "docs/benchmarks/run_ssl_lstm_a4_hmc_repair_2026_07_14.py"
)
REPAIR_TEST_PATH = Path("tests/test_ssl_lstm_a4_hmc_repair.py")
REPAIR_PLAN_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-completion-phase-a4-hmc-acquisition-"
    "repair-plan-2026-07-14.md"
)
REPAIR_RESULT_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-completion-phase-a4-hmc-acquisition-"
    "repair-result-2026-07-14.md"
)
REPAIR_ROOT = Path(
    "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/"
    "hmc-acquisition/repair-01"
)
ARCHIVE_DIR = REPAIR_ROOT / "private"
TUNING_OUTPUT = REPAIR_ROOT / "tune.json"
REPAIR_NAME = "repair_01"
TUNING_LABEL = "repair_01_tune_smaller_step"
TUNING_SEED = (20260714, 1521)
SEGMENT_SEED_BASE = 1530
STEP_SIZE = 0.19625
LEAPFROG_STEPS = 8
TRAJECTORY_LENGTH = 1.57

PRIOR_RECEIPTS = (
    (
        Path(
            "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/"
            "hmc-acquisition/gpu-canary.json"
        ),
        "d5aa099cc4835d427b570a7a22430a7b79498760dc99d8ed280c9bf39692c048",
    ),
    (
        Path(
            "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/"
            "hmc-acquisition/gpu-canary-repair-01.json"
        ),
        "b30098f573fb2a7a22f8a1a71b910d2b931fac7c169f049ac9e9efe6af87ab2d",
    ),
    (
        Path(
            "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/"
            "hmc-acquisition/tune-0.json"
        ),
        "9e70e8dbd04de09c0bc3946d100d24d67ce520c18f63e58c8b5d3502762fa76f",
    ),
    (
        Path(
            "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/"
            "hmc-acquisition/segment-0.json"
        ),
        "d12e7aeb1c9760b9d4bba9f9827c027e371d227a3cf5b84d7775f3a922021892",
    ),
)


def _load_base() -> ModuleType:
    path = ROOT / BASE_HARNESS_PATH
    spec = importlib.util.spec_from_file_location("ssl_lstm_a4_hmc_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load base acquisition harness: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.PLAN_PATH = REPAIR_PLAN_PATH
    module.RESULT_PATH = REPAIR_RESULT_PATH
    return module


base = _load_base()
RepairError = base.AcquisitionError


def _sha256(path: Path) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _file_row(path: Path, role: str) -> dict[str, Any]:
    absolute = ROOT / path
    return {
        "path": path.as_posix(),
        "role": role,
        "bytes": absolute.stat().st_size,
        "sha256": _sha256(path),
    }


def _source_bindings() -> list[dict[str, Any]]:
    return [
        _file_row(REPAIR_PLAN_PATH, "prospective_repair_evidence_contract"),
        _file_row(REPAIR_HARNESS_PATH, "repair_harness"),
        _file_row(REPAIR_TEST_PATH, "focused_repair_tests"),
        _file_row(BASE_HARNESS_PATH, "reviewed_acquisition_authority"),
        _file_row(base.A0_LOCK_PATH, "locked_sampler_geometry"),
        _file_row(
            Path("bayesfilter/nonlinear/ssl_lstm_posterior_tf.py"),
            "locked_a1_target",
        ),
        _file_row(
            Path("bayesfilter/inference/hmc.py"),
            "retained_hmc_archive_runtime",
        ),
        _file_row(
            Path("bayesfilter/inference/hmc_posterior_diagnostics.py"),
            "rank_normalized_admission_diagnostics",
        ),
    ]


def _assert_source_bindings(payload: Mapping[str, Any], path: Path) -> None:
    if payload.get("source_files") != _source_bindings():
        raise RepairError(f"repair source binding drift since artifact: {path}")


def _assert_trusted_repair_manifest(payload: Mapping[str, Any], path: Path) -> None:
    manifest = payload.get("run_manifest", {})
    if (
        manifest.get("trust_basis")
        != "owner_designated_managed_session_visible_gpu_trusted"
        or manifest.get("cpu_gpu_status") != "trusted_gpu_xla"
        or manifest.get("data_version") != base.TARGET_SEMANTIC_SHA256
        or manifest.get("plan_path") != REPAIR_PLAN_PATH.as_posix()
        or manifest.get("result_path") != REPAIR_RESULT_PATH.as_posix()
    ):
        raise RepairError(f"invalid trusted repair run manifest: {path}")
    wall_time = float(manifest.get("wall_time_seconds", math.nan))
    if not math.isfinite(wall_time) or wall_time < 0.0:
        raise RepairError(f"invalid repair GPU wall time: {path}")


def _prior_paths() -> tuple[Path, ...]:
    return tuple(path for path, _digest in PRIOR_RECEIPTS)


def validate_prior_receipts() -> float:
    total = 0.0
    for path, expected_digest in PRIOR_RECEIPTS:
        if not (ROOT / path).is_file():
            raise RepairError(f"missing prior GPU receipt: {path}")
        if _sha256(path) != expected_digest:
            raise RepairError(f"prior GPU receipt SHA-256 drift: {path}")
        payload = base._strict_load(path)
        manifest = payload.get("run_manifest", {})
        if (
            manifest.get("trust_basis")
            != "owner_designated_managed_session_visible_gpu_trusted"
            or manifest.get("cpu_gpu_status") != "trusted_gpu_xla"
        ):
            raise RepairError(f"prior receipt is not trusted GPU/XLA evidence: {path}")
        if manifest.get("data_version") != base.TARGET_SEMANTIC_SHA256:
            raise RepairError(f"prior receipt target identity drift: {path}")
        wall_time = float(manifest.get("wall_time_seconds", math.nan))
        if not math.isfinite(wall_time) or wall_time < 0.0:
            raise RepairError(f"invalid prior GPU wall time: {path}")
        total += wall_time
    failed = base._strict_load(PRIOR_RECEIPTS[-1][0])
    if (
        failed.get("status") != "HARD_VETO"
        or "unmoved_chain"
        not in failed.get("admission_diagnostics", {}).get("hard_vetoes", ())
    ):
        raise RepairError("repair trigger receipt no longer records the unmoved-chain veto")
    return total


def _segment_output(index: int) -> Path:
    return REPAIR_ROOT / f"segment-{index}.json"


def _segment_label(index: int) -> str:
    return f"repair_01_segment_{index}"


def _private_members(label: str) -> tuple[Path, ...]:
    return tuple(
        ARCHIVE_DIR / f"{label}_{suffix}"
        for suffix in (
            "retained_samples.tftensor",
            "final_state.tftensor",
            "final_target_log_prob.tftensor",
            "private_manifest.json",
        )
    )


def _require_fresh(output: Path, label: str) -> None:
    collisions = [path for path in (output, *_private_members(label)) if (ROOT / path).exists()]
    if collisions:
        raise RepairError(
            "repair artifact collision; refusing overwrite: "
            + ", ".join(path.as_posix() for path in collisions)
        )


def _movement(samples: Any) -> Any:
    import tensorflow as tf

    return tf.reduce_any(tf.not_equal(samples[1:], samples[:-1]), axis=(0, 2))


def _run_manifest(
    *, started: str, completed: str, wall_time: float, output: Path, seed: Any
) -> dict[str, Any]:
    return base._environment_manifest(
        started=started,
        completed=completed,
        wall_time=wall_time,
        output_paths=(output, ARCHIVE_DIR),
        random_seeds=(base.ROOT_SEED, seed),
    )


def run_tuning() -> dict[str, Any]:
    import tensorflow as tf

    base._require_gpu()
    _require_fresh(TUNING_OUTPUT, TUNING_LABEL)
    prior_seconds = validate_prior_receipts()
    projected_seconds = 900.0
    if prior_seconds + projected_seconds > base.GPU_BUDGET_SECONDS:
        raise RepairError("repair tuning projection exceeds remaining shared GPU budget")

    started_at = base._now()
    started = time.perf_counter()
    adapter = base.A4CalibrationHMCAdapter()
    initial = tf.constant(base.INITIAL_STATES, tf.float64)
    (samples, _state, manifest, call_s), diagnostics, metadata, _ = base._run_archive(
        adapter=adapter,
        archive_dir=ARCHIVE_DIR,
        label=TUNING_LABEL,
        current_state=initial,
        num_results=64,
        num_burnin_steps=32,
        step_size=STEP_SIZE,
        leapfrog_steps=LEAPFROG_STEPS,
        seed=TUNING_SEED,
        role="trusted_gpu_xla_a4_hmc_smaller_step_repair_tuning",
    )
    private = manifest["diagnostics_private_metadata"]
    health = private["sampler_health_diagnostics"]
    acceptance = [float(value) for value in health["acceptance_rate_by_chain"]]
    moved = _movement(samples)
    hard_vetoes = []
    if not bool(tf.reduce_all(tf.math.is_finite(samples)).numpy()):
        hard_vetoes.append("nonfinite_tuning_samples")
    if not bool(tf.reduce_all(moved).numpy()):
        hard_vetoes.append("unmoved_tuning_chain")
    if int(health["log_accept_ratio"]["nonfinite_count"]):
        hard_vetoes.append("nonfinite_tuning_log_accept_ratio")
    if int(health["target_log_prob"]["nonfinite_count"]):
        hard_vetoes.append("nonfinite_tuning_target_log_prob")
    divergence = private.get("divergence_count")
    if divergence is not None and int(divergence) > 0:
        hard_vetoes.append("positive_native_divergence_count")
    acceptance_passed = all(
        base.ACCEPTANCE_MIN <= value <= base.ACCEPTANCE_MAX for value in acceptance
    )
    selected = not hard_vetoes and acceptance_passed
    completed_at = base._now()
    wall_time = time.perf_counter() - started
    payload = {
        "schema_version": "bayesfilter.ssl_lstm.a4_hmc_repair_tuning.v1",
        "status": "SELECTED" if selected else ("HARD_VETO" if hard_vetoes else "NOT_SELECTED"),
        "repair_attempt": REPAIR_NAME,
        "candidate": {
            "name": "smaller_step",
            "step_size": STEP_SIZE,
            "num_leapfrog_steps": LEAPFROG_STEPS,
            "trajectory_length": TRAJECTORY_LENGTH,
            "seed": TUNING_SEED,
        },
        "initial_state_policy": "original_fixed_four_dispersed_starts",
        "acceptance_rate_by_chain": acceptance,
        "acceptance_bounds": [base.ACCEPTANCE_MIN, base.ACCEPTANCE_MAX],
        "acceptance_passed": acceptance_passed,
        "chain_moved": base._json_safe(moved),
        "hard_vetoes": hard_vetoes,
        "runner_diagnostics": diagnostics,
        "runner_metadata": metadata,
        "private_manifest_sha256": _sha256(
            ARCHIVE_DIR / f"{TUNING_LABEL}_private_manifest.json"
        ),
        "budget_lineage_artifacts": [path.as_posix() for path in _prior_paths()],
        "source_files": _source_bindings(),
        "run_manifest": _run_manifest(
            started=started_at,
            completed=completed_at,
            wall_time=wall_time,
            output=TUNING_OUTPUT,
            seed=TUNING_SEED,
        ),
        "gpu_budget": {
            "cap_seconds": base.GPU_BUDGET_SECONDS,
            "prior_consumed_seconds": prior_seconds,
            "this_run_seconds": wall_time,
            "hmc_call_seconds": call_s,
            "remaining_seconds": base.GPU_BUDGET_SECONDS - prior_seconds - wall_time,
        },
        "native_divergence_interpretation": "not_exposed_by_kernel_is_not_zero_divergences",
        "nonclaims": [
            "repair tuning screen only",
            "passing does not establish convergence, posterior correctness, or superiority",
        ],
    }
    base._write_json(TUNING_OUTPUT, payload)
    if hard_vetoes:
        raise RepairError(f"repair tuning hard vetoes: {hard_vetoes}")
    return payload


def _load_segment_inputs(index: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tuning = base._strict_load(TUNING_OUTPUT)
    if tuning.get("schema_version") != "bayesfilter.ssl_lstm.a4_hmc_repair_tuning.v1":
        raise RepairError("unexpected repair tuning schema")
    if tuning.get("status") != "SELECTED":
        raise RepairError("repair retained rung requires selected fresh tuning")
    _assert_source_bindings(tuning, TUNING_OUTPUT)
    _assert_trusted_repair_manifest(tuning, TUNING_OUTPUT)
    if tuning.get("budget_lineage_artifacts") != [
        path.as_posix() for path in _prior_paths()
    ]:
        raise RepairError("repair tuning prior-receipt ancestry drift")
    if tuning.get("initial_state_policy") != "original_fixed_four_dispersed_starts":
        raise RepairError("repair tuning initial-state policy drift")
    tuning_manifest = ARCHIVE_DIR / f"{TUNING_LABEL}_private_manifest.json"
    if (
        not (ROOT / tuning_manifest).is_file()
        or tuning.get("private_manifest_sha256") != _sha256(tuning_manifest)
    ):
        raise RepairError("repair tuning private-manifest binding drift")
    candidate = tuning.get("candidate", {})
    if (
        float(candidate.get("step_size", math.nan)) != STEP_SIZE
        or int(candidate.get("num_leapfrog_steps", -1)) != LEAPFROG_STEPS
        or list(candidate.get("seed", ())) != list(TUNING_SEED)
    ):
        raise RepairError("repair tuning candidate or seed drift")
    previous = []
    for prior_index in range(index):
        path = _segment_output(prior_index)
        payload = base._strict_load(path)
        if (
            payload.get("schema_version")
            != "bayesfilter.ssl_lstm.a4_hmc_repair_segment.v1"
            or payload.get("status") != "EXTEND"
            or int(payload.get("segment", {}).get("index", -1)) != prior_index
        ):
            raise RepairError("repair segments must be consecutive EXTEND artifacts")
        _assert_source_bindings(payload, path)
        _assert_trusted_repair_manifest(payload, path)
        expected_label = _segment_label(prior_index)
        expected_seed = [20260714, SEGMENT_SEED_BASE + prior_index]
        expected_lineage = [*_prior_paths(), TUNING_OUTPUT]
        expected_lineage.extend(
            _segment_output(earlier) for earlier in range(prior_index)
        )
        if payload.get("budget_lineage_artifacts") != [
            item.as_posix() for item in expected_lineage
        ]:
            raise RepairError("repair segment ancestry drift")
        if (
            payload.get("segment", {}).get("label") != expected_label
            or payload.get("segment", {}).get("seed") != expected_seed
        ):
            raise RepairError("repair segment label or seed drift")
        prior_manifest = ARCHIVE_DIR / f"{expected_label}_private_manifest.json"
        if (
            not (ROOT / prior_manifest).is_file()
            or payload.get("private_manifest_sha256") != _sha256(prior_manifest)
        ):
            raise RepairError("repair segment private-manifest binding drift")
        admission = payload.get("admission_diagnostics", {})
        if (
            admission.get("decision") != "PROMOTION_VETO_EXTEND_IF_BUDGET_ALLOWS"
            or admission.get("hard_vetoes")
            or admission.get("chain_moved") != [True, True, True, True]
        ):
            raise RepairError("prior repair segment is not eligible for extension")
        previous.append(payload)
    return tuning, previous


def _consumed_seconds(tuning: Mapping[str, Any], previous: Sequence[Mapping[str, Any]]) -> float:
    return (
        validate_prior_receipts()
        + float(tuning["run_manifest"]["wall_time_seconds"])
        + sum(float(item["run_manifest"]["wall_time_seconds"]) for item in previous)
    )


def _projected_segment_seconds(index: int, previous: Sequence[Mapping[str, Any]]) -> float:
    transitions = base.SEGMENT_DRAWS[index] + (250 if index == 0 else 0)
    if not previous:
        return 7200.0
    rates = []
    for item in previous:
        segment = item["segment"]
        prior_transitions = int(segment["draw_count"]) + int(segment["burnin_count"])
        rates.append(float(item["run_manifest"]["wall_time_seconds"]) / prior_transitions)
    return max(900.0, 1.5 * max(rates) * transitions)


def run_segment(index: int) -> dict[str, Any]:
    import tensorflow as tf

    base._require_gpu()
    if index < 0 or index >= len(base.SEGMENT_DRAWS):
        raise RepairError("invalid repair segment index")
    output = _segment_output(index)
    label = _segment_label(index)
    _require_fresh(output, label)
    tuning, previous = _load_segment_inputs(index)
    prior_seconds = _consumed_seconds(tuning, previous)
    projected_seconds = _projected_segment_seconds(index, previous)
    if prior_seconds + projected_seconds > base.GPU_BUDGET_SECONDS:
        raise RepairError("repair segment projection exceeds remaining shared GPU budget")

    adapter = base.A4CalibrationHMCAdapter()
    if index == 0:
        current_state = tf.constant(base.INITIAL_STATES, tf.float64)
        burnin = 250
    else:
        _prior_samples, current_state, _manifest = base._read_archive(
            ARCHIVE_DIR, _segment_label(index - 1)
        )
        burnin = 0
    draws = base.SEGMENT_DRAWS[index]
    seed = (20260714, SEGMENT_SEED_BASE + index)
    started_at = base._now()
    started = time.perf_counter()
    (samples, _state, manifest, call_s), diagnostics, metadata, _ = base._run_archive(
        adapter=adapter,
        archive_dir=ARCHIVE_DIR,
        label=label,
        current_state=current_state,
        num_results=draws,
        num_burnin_steps=burnin,
        step_size=STEP_SIZE,
        leapfrog_steps=LEAPFROG_STEPS,
        seed=seed,
        role="trusted_gpu_xla_a4_hmc_smaller_step_repair_retained_segment",
    )
    cumulative = []
    manifests = []
    for prior_index in range(index):
        prior_samples, _prior_state, prior_manifest = base._read_archive(
            ARCHIVE_DIR, _segment_label(prior_index)
        )
        cumulative.append(prior_samples)
        manifests.append(prior_manifest)
    cumulative.append(samples)
    manifests.append(manifest)
    latent = tf.concat(cumulative, axis=0)
    admission = base._admission_diagnostics(
        latent_draw_major=latent,
        adapter=adapter,
        segment_manifests=manifests,
    )
    completed_at = base._now()
    wall_time = time.perf_counter() - started
    hard_vetoes = list(admission["hard_vetoes"])
    status = "ADMITTED" if admission["admitted"] else (
        "HARD_VETO" if hard_vetoes else "EXTEND"
    )
    lineage = [*_prior_paths(), TUNING_OUTPUT]
    lineage.extend(_segment_output(prior_index) for prior_index in range(index))
    payload = {
        "schema_version": "bayesfilter.ssl_lstm.a4_hmc_repair_segment.v1",
        "status": status,
        "repair_attempt": REPAIR_NAME,
        "segment": {
            "index": index,
            "draw_count": draws,
            "burnin_count": burnin,
            "cumulative_draw_count": int(latent.shape[0]),
            "seed": seed,
            "label": label,
        },
        "selected_kernel": tuning["candidate"],
        "initial_state_policy": (
            "original_fixed_four_dispersed_starts"
            if index == 0
            else "exact_prior_repair_segment_final_state"
        ),
        "admission_diagnostics": admission,
        "runner_diagnostics": diagnostics,
        "runner_metadata": metadata,
        "private_manifest_sha256": _sha256(
            ARCHIVE_DIR / f"{label}_private_manifest.json"
        ),
        "cumulative_private_sample_sha256": hashlib.sha256(
            bytes(tf.io.serialize_tensor(latent).numpy())
        ).hexdigest(),
        "budget_lineage_artifacts": [path.as_posix() for path in lineage],
        "source_files": _source_bindings(),
        "run_manifest": _run_manifest(
            started=started_at,
            completed=completed_at,
            wall_time=wall_time,
            output=output,
            seed=seed,
        ),
        "gpu_budget": {
            "cap_seconds": base.GPU_BUDGET_SECONDS,
            "prior_consumed_seconds": prior_seconds,
            "projected_seconds_before_run": projected_seconds,
            "this_run_seconds": wall_time,
            "hmc_call_seconds": call_s,
            "remaining_seconds": base.GPU_BUDGET_SECONDS - prior_seconds - wall_time,
        },
        "nonclaims": admission["nonclaims"],
    }
    base._write_json(output, payload)
    if hard_vetoes:
        raise RepairError(f"repair acquisition hard vetoes: {hard_vetoes}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate-lineage")
    subparsers.add_parser("tune")
    segment_parser = subparsers.add_parser("segment")
    segment_parser.add_argument("--segment-index", type=int, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.command == "validate-lineage":
        seconds = validate_prior_receipts()
        payload = {
            "status": "PASSED",
            "prior_gpu_seconds": seconds,
            "remaining_gpu_seconds": base.GPU_BUDGET_SECONDS - seconds,
        }
    elif args.command == "tune":
        payload = run_tuning()
    else:
        payload = run_segment(args.segment_index)
    print(
        base._canonical_bytes(
            {
                "command": args.command,
                "status": payload["status"],
                "output": (
                    None
                    if args.command == "validate-lineage"
                    else (
                        TUNING_OUTPUT.as_posix()
                        if args.command == "tune"
                        else _segment_output(args.segment_index).as_posix()
                    )
                ),
            }
        ).decode("ascii")
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RepairError as exc:
        print(f"A4_HMC_REPAIR_ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
