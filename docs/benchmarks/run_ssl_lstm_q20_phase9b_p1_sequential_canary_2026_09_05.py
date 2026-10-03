#!/usr/bin/env python3
"""Run the bounded q=20 Phase 9B P1 cold-scope sequential canary."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-phase9b-p1-sequential-canary-plan-2026-09-05.md"
P0_MANIFEST = ROOT / "docs/plans/artifacts/ssl-lstm-q20-phase9b-sequential-validation-2026-09-05/p0-source-preflight/run_manifest.json"
P1_AUDIT_MANIFEST = ROOT / "docs/plans/artifacts/ssl-lstm-q20-phase9b-p1-sequential-canary-2026-09-05/p1-plan-audit-20260907-r14/run_manifest.json"
PHASE0_ARTIFACT_ROOT = ROOT / "docs/plans/artifacts/ssl-lstm-q20-phase9b-executable-readiness-2026-09-06"
CAMPAIGN_LEDGER_PATH = PHASE0_ARTIFACT_ROOT / "campaign_budget_ledger.json"
FACTOR_ROOT = ROOT / "docs/plans/artifacts/ssl-lstm-q20-factor-route-fresh-tuning-2026-09-04/source-sync-20260905T203000Z"
PHASE9A_RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_phase9a_fresh_tuning_preflight_2026_08_31.py"
TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
FACTOR_BACKEND = "tensorflow_eigh_strict_factor_cached"
STRICT_BACKEND = "tensorflow_eigh_strict"
POLICY_ID = "bayesfilter_neutra_sequential_hmc_v1"
SCOPE_INDEX = 2
CHART_INDEX = 0
BETA = 1.0
COMPONENT_ID = "phase9a-chart-0"
CHAIN_COUNT = 4
DIMENSION = 4
MATERIAL_CAP_SECONDS = 86_400.0
HISTORICAL_MATERIAL_CAP_SECONDS = 5_200.0
ARM_CAP_SECONDS = 28_800.0
REQUIRED_SEQUENTIAL_CHUNKS_PER_ARM = 6
OUTPUT_SCHEMA = "bayesfilter.ssl_lstm_q20.phase9b_p1_sequential_canary.v1"
EXPECTED_FACTOR_MANIFEST_SHA256 = "e0225382192ceb9da1e075cb9c7a91ed424e2c5c67adcfec7a0f34c05003a4e1"
EXPECTED_FACTOR_AUDIT_SHA256 = "9848779a1fe1611f3b3acfb666bec8b08bef6a42296fe30e84d3762d892c5933"
HISTORICAL_CONSUMED_SECONDS = 1_832.61
CAMPAIGN_ID = "ssl-lstm-q20-phase9b-p1"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.runtime.campaign_budget_ledger import (
    CampaignBudgetLedger,
    CampaignBudgetLedgerError,
)
from bayesfilter.runtime.display_gpu_policy import (
    GPUPlacementError,
    POLICY_ID as GPU_PLACEMENT_POLICY_ID,
    check_runtime_headroom,
    select_and_pin_gpu,
)

_ACTIVE_CONTEXT: dict[str, Any] = {}


class P1CanaryError(RuntimeError):
    """Raised when the P1 canary contract cannot be preserved."""


def _select_phase9b_gpu() -> Mapping[str, Any]:
    try:
        return select_and_pin_gpu()
    except GPUPlacementError as exc:
        raise P1CanaryError(str(exc)) from exc


def _campaign_source_hash() -> str:
    return _sha256(SCRIPT)


def _campaign_plan_hash() -> str:
    return _sha256(PLAN)


def _open_campaign_ledger(
    output: Path, *, minimum_remaining_seconds: float = 0.0
) -> CampaignBudgetLedger:
    minimum = float(minimum_remaining_seconds)
    if not math.isfinite(minimum) or minimum < 0.0:
        raise P1CanaryError("minimum campaign allocation must be finite and nonnegative")
    source_hash = _campaign_source_hash()
    plan_hash = _campaign_plan_hash()
    ledger = CampaignBudgetLedger(CAMPAIGN_LEDGER_PATH)
    if not CAMPAIGN_LEDGER_PATH.exists():
        ledger = CampaignBudgetLedger.create(
            CAMPAIGN_LEDGER_PATH,
            campaign_id=CAMPAIGN_ID,
            total_budget_seconds=HISTORICAL_MATERIAL_CAP_SECONDS,
            source_hash=source_hash,
            plan_hash=plan_hash,
            claim_boundary="phase9b_m4_p0_executable_readiness_only",
            initial_consumed_seconds=HISTORICAL_CONSUMED_SECONDS,
            initial_attempts=(
                {
                    "attempt_id": "p1-attempt-20260905T120000Z",
                    "status": "historical_failed",
                    "failure_class": "harness_chart_selection",
                    "measured_seconds": "timestamp_estimate_in_2026-09-06_result",
                    "output_root": str(
                        ROOT
                        / "docs/plans/artifacts/ssl-lstm-q20-phase9b-p1-sequential-canary-2026-09-05/p1-attempt-20260905T120000Z"
                    ),
                },
                {
                    "attempt_id": "p1-attempt-20260906T104500Z",
                    "status": "historical_failed",
                    "failure_class": "harness_chart_selection",
                    "measured_seconds": "timestamp_estimate_in_2026-09-06_result",
                    "output_root": str(
                        ROOT
                        / "docs/plans/artifacts/ssl-lstm-q20-phase9b-p1-sequential-canary-2026-09-05/p1-attempt-20260906T104500Z"
                    ),
                },
                {
                    "attempt_id": "p1-attempt-20260906T105000Z",
                    "status": "historical_failed",
                    "failure_class": "harness_callback_contract",
                    "measured_seconds": "timestamp_estimate_in_2026-09-06_result",
                    "output_root": str(
                        ROOT
                        / "docs/plans/artifacts/ssl-lstm-q20-phase9b-p1-sequential-canary-2026-09-05/p1-attempt-20260906T105000Z"
                    ),
                },
            ),
        )
    else:
        ledger.read()
    if minimum > ledger.remaining_seconds():
        raise P1CanaryError("complete attempt allocation exceeds the remaining campaign budget")
    try:
        ledger.start_attempt(
            attempt_id=output.name,
            output_root=output,
            seed_namespace={
                arm: _seed_map(output, arm) for arm in ("factor", "strict")
            },
            source_hash=source_hash,
            plan_hash=plan_hash,
        )
    except CampaignBudgetLedgerError as exc:
        raise P1CanaryError(str(exc)) from exc
    return ledger


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return {"__nonfinite__": repr(value)}
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if hasattr(value, "numpy"):
        return _json_ready(value.numpy())
    if hasattr(value, "tolist"):
        return _json_ready(value.tolist())
    if hasattr(value, "item"):
        return _json_ready(value.item())
    return str(value)


def _settle_unaccounted_work(
    ledger: CampaignBudgetLedger, output: Path, started: float, *, status: str
) -> None:
    attempt = next(
        item for item in ledger.read()["attempts"] if item["attempt_id"] == output.name
    )
    unaccounted = max(
        0.0, time.monotonic() - started - float(attempt["consumed_seconds"])
    )
    ledger.settle_arm(
        attempt_id=output.name,
        arm="unaccounted_setup_cleanup_or_failed_tuning",
        measured_seconds=unaccounted,
        status=status,
    )


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _json_ready(value),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_namespace(output: Path) -> int:
    digest = hashlib.sha256(str(output).encode("utf-8")).hexdigest()
    return 1 + int(digest[:8], 16) % 10_000


def _seed_map(output: Path, arm_name: str) -> Mapping[str, tuple[int, int]]:
    if arm_name not in {"factor", "strict"}:
        raise P1CanaryError(f"unsupported seed namespace arm: {arm_name}")
    arm_base = 100_000 if arm_name == "factor" else 200_000
    sequential_base = 10_000_000 if arm_name == "factor" else 20_000_000
    base = arm_base + _seed_namespace(output) * 1_000
    sequential = sequential_base + _seed_namespace(output) * 1_000
    return {
        "initialization_0": (20260906, base + 1),
        "initialization_1": (20260906, base + 2),
        "preflight_0": (20260906, base + 101),
        "preflight_1": (20260906, base + 102),
        "training_0": (20260906, base + 201),
        "training_1": (20260906, base + 202),
        "transition": (20260906, base + 401),
        "reliability": (20260906, base + 501),
        "warmup": (20260906, sequential + 601),
        "retained": (20260906, sequential + 602),
    }


def _select_chart(chart_map: Mapping[float, Any], beta: float) -> Any:
    try:
        return chart_map[beta]
    except KeyError as exc:
        raise P1CanaryError(f"chart map lacks beta={beta:g}") from exc


def _validate_budget_forecast(forecast_seconds: float) -> None:
    forecast = float(forecast_seconds)
    if not math.isfinite(forecast) or forecast <= 0.0:
        raise P1CanaryError("sequential chunk forecast must be positive and finite")
    arm_forecast = REQUIRED_SEQUENTIAL_CHUNKS_PER_ARM * forecast
    campaign_forecast = 2 * arm_forecast
    if arm_forecast > ARM_CAP_SECONDS or campaign_forecast > MATERIAL_CAP_SECONDS:
        raise P1CanaryError(
            "declared minimum P1 schedule does not fit the available budget: "
            f"{forecast:g}s forecast x {REQUIRED_SEQUENTIAL_CHUNKS_PER_ARM} "
            f"chunks/arm = {arm_forecast:g}s/arm, cap={ARM_CAP_SECONDS:g}s"
        )


def _readiness_source_hashes() -> Mapping[str, str]:
    paths = {
        "p1_runner": SCRIPT,
        "p1_plan": PLAN,
        "controller": ROOT / "bayesfilter/inference/neutra_hmc.py",
        "budget_ledger": ROOT / "bayesfilter/runtime/campaign_budget_ledger.py",
        "gpu_placement": ROOT / "bayesfilter/runtime/display_gpu_policy.py",
        "continuation_plan": ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-phase9b-multigpu-continuation-2026-09-07.md",
        "diagnostic": ROOT / "docs/benchmarks/diagnose_ssl_lstm_q20_phase9b_executable_readiness_2026_09_06.py",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _verify_phase0_closeout(path: Path, forecast_seconds: float) -> Mapping[str, Any]:
    closeout_path = path.expanduser().resolve()
    try:
        closeout = json.loads(closeout_path.read_text(encoding="utf-8"))
        if not isinstance(closeout, Mapping):
            raise ValueError("closeout must be a JSON object")
        if closeout.get("status") != "P1_LAUNCHABLE_PENDING_FRESH_ATTEMPT":
            raise ValueError("executable-readiness closeout is not passing")
        expected_sources = _readiness_source_hashes()
        runtime_reference = closeout["runtime_diagnostic"]
        runtime_path = Path(runtime_reference["path"]).expanduser()
        if not runtime_path.is_absolute():
            runtime_path = ROOT / runtime_path
        if _sha256(runtime_path) != runtime_reference["sha256"]:
            raise ValueError("runtime diagnostic checksum mismatch")
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        if runtime.get("status") != "PASS_PHASE9B_M4_P0_RUNTIME_DIAGNOSTIC":
            raise ValueError("runtime diagnostic is not passing")
        for label, payload in (("closeout", closeout), ("runtime diagnostic", runtime)):
            if payload.get("target_signature") != TARGET_SIGNATURE:
                raise ValueError(f"{label} target mismatch")
            if payload.get("policy_id") != POLICY_ID:
                raise ValueError(f"{label} controller policy mismatch")
            if payload.get("claim_boundary") != "phase9b_m4_p0_executable_readiness_only":
                raise ValueError(f"{label} claim boundary mismatch")
            for name, expected in expected_sources.items():
                if payload.get("source_hashes", {}).get(name) != expected:
                    raise ValueError(f"{label} has stale {name} source")
        budget = closeout["budget"]
        if budget["material_cap_seconds"] != MATERIAL_CAP_SECONDS:
            raise ValueError("closeout campaign cap differs from the P1 plan")
        if budget["arm_cap_seconds"] != ARM_CAP_SECONDS:
            raise ValueError("closeout arm cap differs from the P1 plan")
        timing = runtime["timing"]
        if timing["p1_required_chunks_per_arm"] != REQUIRED_SEQUENTIAL_CHUNKS_PER_ARM:
            raise ValueError("runtime diagnostic did not forecast the complete P1 schedule")
        measured_chunk = float(timing["sequential_chunk_forecast_seconds"])
        if not math.isfinite(measured_chunk) or measured_chunk <= 0.0:
            raise ValueError("runtime chunk forecast must be positive and finite")
        if float(forecast_seconds) != measured_chunk:
            raise ValueError("requested chunk forecast differs from the measured forecast")
        arm_forecasts = {
            arm: float(timing["p1_forecast_seconds_per_arm"][arm])
            for arm in ("factor", "strict")
        }
        if any(
            not math.isfinite(value) or value <= 0.0 or value > ARM_CAP_SECONDS
            for value in arm_forecasts.values()
        ):
            raise ValueError("complete arm forecast does not fit the P1 arm cap")
        total = float(timing["p1_complete_schedule_forecast_seconds"])
        if not math.isfinite(total) or total < sum(arm_forecasts.values()):
            raise ValueError("complete schedule forecast omits arm costs")
        if total > MATERIAL_CAP_SECONDS:
            raise ValueError("complete schedule forecast exceeds the campaign cap")
        gpu = runtime["gpu"]
        if gpu["placement_policy_id"] != GPU_PLACEMENT_POLICY_ID:
            raise ValueError("runtime GPU placement policy mismatch")
        measured_gpu = gpu["selection"]["selected"]
        capacity = float(measured_gpu["memory_total_mib"])
        if (
            not math.isfinite(capacity) or capacity <= 0.0
            or any(
                not isinstance(measured_gpu.get(key), str) or not measured_gpu[key].strip()
                for key in ("uuid", "pci_bus_id", "name")
            )
            or not measured_gpu["uuid"].startswith("GPU-")
        ):
            raise ValueError("runtime physical GPU identity missing")
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        raise P1CanaryError(f"Phase 0 closeout required: {exc}") from exc
    return {
        "path": str(closeout_path),
        "sha256": _sha256(closeout_path),
        "status": closeout["status"],
        "runtime_diagnostic": dict(runtime_reference),
        "complete_schedule_forecast_seconds": total,
        "measured_gpu": measured_gpu,
    }


def _verify_forecast_hardware(selection: Mapping[str, Any], measured_gpu: Mapping[str, Any]) -> None:
    selected = selection["selected"]
    required_identity = ("uuid", "pci_bus_id", "name", "memory_total_mib")
    if any(
        selected.get(key) is None or measured_gpu.get(key) is None
        or selected[key] != measured_gpu[key]
        for key in required_identity
    ):
        raise P1CanaryError("selected GPU hardware differs from the readiness forecast; remeasure")


class _BudgetGuard:
    def __init__(
        self,
        *,
        campaign_started: float,
        arm_started: float,
        forecast_seconds: float,
        ledger: CampaignBudgetLedger,
        attempt_id: str,
        arm: str,
        output_root: Path,
        seed_namespace: Mapping[str, Any],
    ) -> None:
        self.campaign_started = campaign_started
        self.arm_started = arm_started
        self.forecast_seconds = float(forecast_seconds)
        self.ledger = ledger
        self.attempt_id = str(attempt_id)
        self.arm = str(arm)
        self.output_root = output_root
        self.seed_namespace = seed_namespace
        self.reservation_ids: list[str] = []

    def before_chunk(self, _work_units: int) -> bool:
        now = time.monotonic()
        if now - self.arm_started + self.forecast_seconds > ARM_CAP_SECONDS:
            raise P1CanaryError(
                f"{self.arm} exceeded its {ARM_CAP_SECONDS:g}s arm cap before the next chunk"
            )
        if now - self.campaign_started + self.forecast_seconds > MATERIAL_CAP_SECONDS:
            raise P1CanaryError(
                f"campaign exceeded its {MATERIAL_CAP_SECONDS:g}s cap before the next chunk"
            )
        chunk_index = len(self.reservation_ids)
        try:
            reservation_id = self.ledger.reserve_chunk(
                attempt_id=self.attempt_id,
                arm=self.arm,
                chunk_index=chunk_index,
                reserve_seconds=self.forecast_seconds,
                output_root=self.output_root,
                seed_namespace={
                    "arm": self.arm,
                    "chunk_index": chunk_index,
                    "root": self.seed_namespace,
                },
                source_hash=_campaign_source_hash(),
                plan_hash=_campaign_plan_hash(),
            )
        except CampaignBudgetLedgerError as exc:
            raise P1CanaryError(str(exc)) from exc
        self.reservation_ids.append(reservation_id)
        return True

    def settle(self, measured_seconds: float, *, status: str, failure_class: str | None = None) -> Mapping[str, Any]:
        try:
            return self.ledger.settle_arm(
                attempt_id=self.attempt_id,
                arm=self.arm,
                measured_seconds=measured_seconds,
                status=status,
                failure_class=failure_class,
                repair="phase9b_m4_p0_persistent_budget_ledger",
            )
        except CampaignBudgetLedgerError as exc:
            raise P1CanaryError(str(exc)) from exc


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise P1CanaryError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value))


def _git(command: Sequence[str]) -> str:
    try:
        return subprocess.check_output(
            tuple(command), cwd=ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable:{type(exc).__name__}"


def _git_payload() -> Mapping[str, Any]:
    status = _git(("git", "status", "--porcelain"))
    return {
        "commit": _git(("git", "rev-parse", "HEAD")),
        "worktree_dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _load_phase9a_runner() -> Any:
    spec = importlib.util.spec_from_file_location("q20_phase9a_runner", PHASE9A_RUNNER)
    if spec is None or spec.loader is None:
        raise P1CanaryError("unable to load source-owned Phase 9A chart/tuning runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _arm_profile(source_module: Any, backend: str, arm_name: str) -> Any:
    base = source_module._FACTOR_TUNING_R2_PROFILE
    if backend not in {FACTOR_BACKEND, STRICT_BACKEND}:
        raise P1CanaryError(f"unsupported P1 backend: {backend}")
    seeds = _seed_map(Path(_ACTIVE_CONTEXT["output"]), arm_name)
    seed_root = seeds["initialization_0"][1] - 1
    return replace(
        base,
        profile_id=f"phase9b_p1_{arm_name}_scope2_{_seed_namespace(Path(_ACTIVE_CONTEXT['output'])):04d}_v2",
        plan_path=PLAN,
        initialization_roots=(seeds["initialization_0"], seeds["initialization_1"]),
        preflight_roots=(seeds["preflight_0"], seeds["preflight_1"]),
        training_roots=(seeds["training_0"], seeds["training_1"]),
        tuning_roots=tuple(
            (20260906, seed_root + 301 + index) for index in range(6)
        ),
        transition_root=seeds["transition"],
        reliability_root=seeds["reliability"],
        scope_start=SCOPE_INDEX,
        scope_limit=1,
        principal_sqrt_backend=backend,
        material_cap_seconds=ARM_CAP_SECONDS,
    )


def _verify_source_inputs() -> Mapping[str, Any]:
    failures: list[str] = []
    for path in (PLAN, P0_MANIFEST, P1_AUDIT_MANIFEST, PHASE9A_RUNNER):
        if not path.is_file():
            failures.append(f"missing:{path.relative_to(ROOT)}")
    if not P0_MANIFEST.is_file():
        raise P1CanaryError("P0 manifest is missing")
    if failures:
        raise P1CanaryError("; ".join(failures))
    p0 = json.loads(P0_MANIFEST.read_text(encoding="utf-8"))
    p1_audit = json.loads(P1_AUDIT_MANIFEST.read_text(encoding="utf-8"))
    if p0.get("status") != "PASS_PHASE9B_P0_SOURCE_PREFLIGHT":
        failures.append("p0_status_mismatch")
    if p1_audit.get("status") != "PASS_PHASE9B_P1_PLAN_AUDIT":
        failures.append("p1_audit_status_mismatch")
    if p1_audit.get("checks", {}).get("plan", {}).get("sha256") != _sha256(PLAN):
        failures.append("p1_audit_plan_hash_mismatch")
    if p1_audit.get("checks", {}).get("runner", {}).get("sha256") != _sha256(SCRIPT):
        failures.append("p1_audit_runner_hash_mismatch")
    if not FACTOR_ROOT.joinpath("run_manifest.json").is_file():
        failures.append("factor_manifest_missing")
    if not FACTOR_ROOT.joinpath("source-sync-audit.json").is_file():
        failures.append("factor_audit_missing")
    if failures:
        raise P1CanaryError("; ".join(failures))
    factor_manifest = FACTOR_ROOT / "run_manifest.json"
    factor_audit = FACTOR_ROOT / "source-sync-audit.json"
    manifest = json.loads(factor_manifest.read_text(encoding="utf-8"))
    audit = json.loads(factor_audit.read_text(encoding="utf-8"))
    if _sha256(factor_manifest) != EXPECTED_FACTOR_MANIFEST_SHA256:
        raise P1CanaryError("factor source manifest hash mismatch")
    if _sha256(factor_audit) != EXPECTED_FACTOR_AUDIT_SHA256:
        raise P1CanaryError("factor source audit hash mismatch")
    if manifest.get("target_signature") != TARGET_SIGNATURE:
        raise P1CanaryError("factor target signature mismatch")
    if manifest.get("profile", {}).get("principal_sqrt_backend") != FACTOR_BACKEND:
        raise P1CanaryError("factor backend identity mismatch")
    if audit.get("status") != "PASS_FACTOR_SOURCE-SYNC_AUDIT":
        raise P1CanaryError("factor source audit status is not passing")
    return {
        "p0_status": p0.get("status"),
        "p1_audit_status": p1_audit.get("status"),
        "p1_audit_manifest_sha256": _sha256(P1_AUDIT_MANIFEST),
        "factor_manifest_sha256": _sha256(factor_manifest),
        "factor_audit_sha256": _sha256(factor_audit),
        "target_signature": TARGET_SIGNATURE,
        "factor_backend": FACTOR_BACKEND,
        "p1_plan_sha256": _sha256(PLAN),
        "p1_runner_sha256": _sha256(SCRIPT),
    }


def _archive_callback(output: Path, tf: Any, arm: str):
    def archive(
        *,
        stage: str,
        chunk_index: int | None,
        latent_samples: Any,
        model_samples: Any,
        seed: Any,
        cumulative: bool,
    ) -> Mapping[str, Any]:
        label = "cumulative" if cumulative else f"chunk-{int(chunk_index or 0):03d}"
        path = output / "sequential" / arm / "archive" / stage / f"{label}.json"
        payload = {
            "schema": "bayesfilter.ssl_lstm_q20.phase9b_p1_sample_archive.v1",
            "arm": arm,
            "stage": stage,
            "chunk_index": chunk_index,
            "seed": seed,
            "cumulative": bool(cumulative),
            "latent_samples": tf.convert_to_tensor(latent_samples, tf.float64),
            "model_samples": tf.convert_to_tensor(model_samples, tf.float64),
            "warmup_excluded_from_posterior": stage == "warmup",
        }
        _write_json(path, payload)
        return {
            "path": str(path.relative_to(ROOT)),
            "sha256": _sha256(path),
            "stage": stage,
            "chunk_index": chunk_index,
            "cumulative": bool(cumulative),
        }

    return archive


def _diagnostic_summary(tf: Any, samples: Any) -> Mapping[str, Any]:
    from bayesfilter.inference.hmc_posterior_diagnostics import (
        posterior_mean_diagnostics,
        rank_normalized_bulk_tail_ess,
        rank_normalized_split_rhat,
    )

    tensor = tf.convert_to_tensor(samples, tf.float64)
    if tensor.shape.rank != 3 or any(dim is None for dim in tensor.shape):
        return {"status": "NOT_COMPUTABLE", "reason": "missing_static_sample_shape"}
    if int(tensor.shape[0]) < 4 or int(tensor.shape[0]) % 2:
        return {"status": "NOT_COMPUTABLE", "reason": "insufficient_even_draws"}
    chain_major = tf.transpose(tensor, perm=(1, 0, 2))
    rhat = rank_normalized_split_rhat(chain_major)
    ess = rank_normalized_bulk_tail_ess(chain_major)
    mean = posterior_mean_diagnostics(chain_major)
    finite = bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy())
    return {
        "status": "COMPUTED",
        "input_layout": "[draw, chain, parameter]",
        "diagnostic_layout": "[chain, draw, parameter]",
        "input_shape": tuple(int(item) for item in tensor.shape),
        "diagnostic_shape": tuple(int(item) for item in chain_major.shape),
        "samples_all_finite": finite,
        "rhat": rhat,
        "ess": ess,
        "mcse": {
            "status": "COMPUTED",
            "mean_mcse": mean["mean_mcse"],
            "mcse_sd_ratio": mean["mcse_sd_ratio"],
        },
    }


def _run_arm(
    *,
    tf: Any,
    source_module: Any,
    bridge: Any,
    profile: Any,
    arm_name: str,
    arm_root: Path,
) -> Mapping[str, Any]:
    started = time.monotonic()
    chart_map, checkpoints, preflight = source_module._build_fresh_chart(
        tf,
        bridge,
        CHART_INDEX,
        COMPONENT_ID,
        profile,
        arm_root,
    )
    chart = _select_chart(chart_map, BETA)
    tuning_root = arm_root / "tuning" / f"chart-{CHART_INDEX}" / f"beta-{BETA:g}"
    row = source_module._tune_scope(
        tf,
        bridge,
        chart,
        chart_index=CHART_INDEX,
        beta=BETA,
        output_dir=tuning_root,
        profile=profile,
        scope_index=SCOPE_INDEX,
    )
    handoff = row.get("_live_handoff")
    if handoff is None:
        raise P1CanaryError(f"{arm_name} did not produce a live verified handoff")
    if time.monotonic() - started > ARM_CAP_SECONDS:
        raise P1CanaryError(f"{arm_name} exceeded its {ARM_CAP_SECONDS:g}s arm cap")

    from bayesfilter.inference.neutra_hmc import (
        SequentialNeuTraHMCConfig,
        run_sequential_neutra_hmc,
    )

    adapter = handoff.transformed_adapter
    initial_state = tf.convert_to_tensor(profile.initial_state_bank, tf.float64)

    def model_transform(samples: Any) -> Any:
        values = tf.convert_to_tensor(samples, tf.float64)
        shape = values.shape
        flat = tf.reshape(values, (-1, DIMENSION))
        mapped = chart.forward_batch(flat)
        return tf.reshape(mapped, shape)

    sequential_root = arm_root / "sequential" / arm_name
    sequential_root.mkdir(parents=True, exist_ok=True)
    budget_guard = _BudgetGuard(
        campaign_started=float(_ACTIVE_CONTEXT["started"]),
        arm_started=started,
        forecast_seconds=float(_ACTIVE_CONTEXT["chunk_forecast_seconds"]),
        ledger=_ACTIVE_CONTEXT["budget_ledger"],
        attempt_id=str(_ACTIVE_CONTEXT["campaign_attempt_id"]),
        arm=arm_name,
        output_root=Path(_ACTIVE_CONTEXT["output"]),
        seed_namespace=_seed_map(Path(_ACTIVE_CONTEXT["output"]), arm_name),
    )
    try:
        sequential = run_sequential_neutra_hmc(
            adapter=adapter,
            initial_state=initial_state,
            model_transform=model_transform,
            parameter_names=tuple(f"theta.{index}" for index in range(DIMENSION)),
            config=SequentialNeuTraHMCConfig(
                step_size=float(handoff.step_size),
                num_leapfrog_steps=int(handoff.num_leapfrog_steps),
                warmup_seed=_seed_map(Path(_ACTIVE_CONTEXT["output"]), arm_name)["warmup"],
                retained_seed=_seed_map(Path(_ACTIVE_CONTEXT["output"]), arm_name)["retained"],
                warmup_chunk_results=500,
                warmup_min_results=2000,
                warmup_check_window_results=1000,
                warmup_max_results=2000,
                warmup_rhat_max=1.05,
                retained_chunk_results=500,
                retained_min_results=1000,
                retained_max_results=1000,
                retained_rhat_max=1.01,
                minimum_chain_count=4,
                jit_compile=True,
            ),
            archive_callback=_archive_callback(arm_root, tf, arm_name),
            budget_check=budget_guard.before_chunk,
        )
    except Exception as exc:
        budget_guard.settle(
            time.monotonic() - started,
            status="failed",
            failure_class=type(exc).__name__,
        )
        raise
    budget_settlement = budget_guard.settle(
        time.monotonic() - started,
        status="completed" if sequential.get("passed") is True else "incomplete",
        failure_class=None if sequential.get("passed") is True else "sequential_veto",
    )
    retained = sequential.get("private_retained_raw")
    warmup = sequential.get("private_warmup_raw")
    summary = {
        "schema": "bayesfilter.ssl_lstm_q20.phase9b_p1_arm_result.v1",
        "arm": arm_name,
        "backend": profile.principal_sqrt_backend,
        "scope": {"scope_index": SCOPE_INDEX, "chart_index": CHART_INDEX, "beta": BETA},
        "target_signature": TARGET_SIGNATURE,
        "tuning_artifact": row.get("tuning_artifact"),
        "handoff": row.get("handoff"),
        "checkpoint_count": len(checkpoints),
        "preflight_count": len(preflight),
        "sequential": {
            key: value
            for key, value in sequential.items()
            if not str(key).startswith("private_")
        },
        "warmup_diagnostics": _diagnostic_summary(tf, warmup),
        "retained_diagnostics": _diagnostic_summary(tf, retained),
        "elapsed_seconds": time.monotonic() - started,
        "budget_settlement": budget_settlement,
        "seed_namespace": _seed_map(Path(_ACTIVE_CONTEXT["output"]), arm_name),
        "claim_boundary": "phase9b_p1_cold_scope_sequential_canary_only",
        "nonclaims": [
            "no posterior correctness",
            "no ensemble or replica-exchange claim",
            "no whitening",
            "no sampler superiority",
            "no default readiness",
        ],
    }
    _write_json(sequential_root / "arm-summary.json", summary)
    return summary


def _failure_payload(
    output: Path, exc: BaseException, *, traceback_text: str | None = None
) -> Mapping[str, Any]:
    context = dict(_ACTIVE_CONTEXT)
    output = context.get("output", output)
    started = context.get("started")
    return {
        "schema": OUTPUT_SCHEMA,
        "status": "FAIL_PHASE9B_P1_CANARY",
        "error_type": type(exc).__name__,
        "error": str(exc)[:12_000],
        "traceback": None if traceback_text is None else traceback_text[:20_000],
        "claim_boundary": "phase9b_p1_cold_scope_sequential_canary_only",
        "output_dir": str(output),
        "command": list(sys.argv),
        "git": _git_payload(),
        "started_at_utc": context.get("started_at_utc"),
        "elapsed_seconds": None if started is None else time.monotonic() - float(started),
        "run_start": str((Path(output) / "run_start.json").relative_to(ROOT))
        if Path(output).is_relative_to(ROOT)
        else str(Path(output)),
        "seed_namespaces": context.get("seed_namespaces"),
        "gpu_selection": context.get("gpu_selection", {"status": "not_collected"}),
        "memory_policy": context.get("memory_policy", {"status": "not_collected"}),
        "allocator_telemetry": {"status": "not_collected_on_failure"},
        "budget_ledger": context.get("budget_ledger_receipt", {"status": "not_initialized"}),
        "nonclaims": [
            "no posterior correctness",
            "no convergence promotion",
            "no ensemble or replica-exchange claim",
            "no sampler superiority",
            "no repository default change",
        ],
    }


def _handle_signal(signum: int, _frame: Any) -> None:
    output = _ACTIVE_CONTEXT.get("output")
    if isinstance(output, Path):
        try:
            _write_json(
                output / "failure.json",
                _failure_payload(
                    output,
                    P1CanaryError(f"interrupted by signal {signum}"),
                    traceback_text=None,
                ),
            )
        except Exception:
            pass
    raise P1CanaryError(f"interrupted by signal {signum}")


def _recovery_coordinator():
    path = ROOT / "docs/benchmarks/run_ssl_lstm_q20_phase9b_recovery_runtime_2026_09_09.py"
    spec = importlib.util.spec_from_file_location("phase9b_p1_recovery_coordinator", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.coordinator


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--phase0-closeout", type=Path)
    parser.add_argument("--sequential-chunk-forecast-seconds", type=float)
    args = parser.parse_args(argv)
    legacy_values = (args.output_dir, args.phase0_closeout, args.sequential_chunk_forecast_seconds)
    if args.campaign_root is not None:
        if any(value is not None for value in legacy_values):
            parser.error("--campaign-root cannot be combined with legacy closeout/forecast arguments")
        return _recovery_coordinator()(args.campaign_root, resume=args.resume, through_p1=True,
                                       initialize_only=args.initialize_only)
    if args.resume or args.initialize_only or any(value is None for value in legacy_values):
        parser.error("use --campaign-root for recoverable parallel P1, or supply all three legacy arguments")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        print(json.dumps({"status": "FAIL_PHASE9B_P1_CANARY", "error": "output directory already exists"}), file=sys.stderr)
        return 2
    output.mkdir(parents=True)
    _ACTIVE_CONTEXT.clear()
    _ACTIVE_CONTEXT["output"] = output
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    started = time.monotonic()
    started_at_utc = datetime.now(timezone.utc).isoformat()
    _ACTIVE_CONTEXT.update(
        {
            "started": started,
            "started_at_utc": started_at_utc,
            "chunk_forecast_seconds": args.sequential_chunk_forecast_seconds,
            "seed_namespaces": {
                arm: _seed_map(output, arm) for arm in ("factor", "strict")
            },
            "memory_policy": {"status": "not_initialized"},
        }
    )
    try:
        _validate_budget_forecast(args.sequential_chunk_forecast_seconds)
        phase0_closeout = _verify_phase0_closeout(
            args.phase0_closeout, args.sequential_chunk_forecast_seconds
        )
        source_inputs = _verify_source_inputs()
        gpu_selection = _select_phase9b_gpu()
        _ACTIVE_CONTEXT["gpu_selection"] = gpu_selection
        _write_json(output / "gpu-selection.json", gpu_selection)
        _verify_forecast_hardware(gpu_selection, phase0_closeout["measured_gpu"])
        budget_ledger = _open_campaign_ledger(
            output,
            minimum_remaining_seconds=phase0_closeout["complete_schedule_forecast_seconds"],
        )
    except Exception as exc:
        payload = _failure_payload(output, exc, traceback_text=traceback.format_exc())
        _write_json(output / "failure.json", payload)
        print(json.dumps(payload, sort_keys=True), file=sys.stderr)
        return 2
    _ACTIVE_CONTEXT.update(
        {
            "budget_ledger": budget_ledger,
            "phase0_closeout": phase0_closeout,
            "campaign_attempt_id": output.name,
            "budget_ledger_receipt": {
                "path": str(CAMPAIGN_LEDGER_PATH.relative_to(ROOT)),
                "checksum": budget_ledger.checksum(),
                "remaining_seconds_before_attempt": budget_ledger.remaining_seconds(),
            },
        }
    )
    _write_json(
        output / "run_start.json",
        {
            "schema": "bayesfilter.ssl_lstm_q20.phase9b_p1_run_start.v1",
            "status": "RUN_STARTED",
            "started_at_utc": started_at_utc,
            "output_dir": str(output),
            "command": list(sys.argv),
            "target_signature": TARGET_SIGNATURE,
            "policy_id": POLICY_ID,
            "plan": str(PLAN.relative_to(ROOT)),
            "plan_sha256": _sha256(PLAN) if PLAN.is_file() else None,
            "p0_manifest": str(P0_MANIFEST.relative_to(ROOT)),
            "phase0_closeout": phase0_closeout,
            "p1_audit_manifest": str(P1_AUDIT_MANIFEST.relative_to(ROOT)),
            "gpu_selection": gpu_selection,
            "seed_namespaces": _ACTIVE_CONTEXT["seed_namespaces"],
            "budget": {
                "material_cap_seconds": MATERIAL_CAP_SECONDS,
                "arm_cap_seconds": ARM_CAP_SECONDS,
                "sequential_chunk_forecast_seconds": args.sequential_chunk_forecast_seconds,
                "required_sequential_chunks_per_arm": REQUIRED_SEQUENTIAL_CHUNKS_PER_ARM,
            },
            "git": _git_payload(),
            "memory_policy": {"status": "not_initialized"},
            "allocator_telemetry": {"status": "not_collected"},
            "budget_ledger": _ACTIVE_CONTEXT["budget_ledger_receipt"],
        },
    )
    try:
        if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").strip().lower() != "true":
            raise P1CanaryError("TF_FORCE_GPU_ALLOW_GROWTH=true is required before TensorFlow import")
        if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != gpu_selection["selected"]["uuid"]:
            raise P1CanaryError("P1 GPU environment changed after placement selection")
        source_module = _load_phase9a_runner()
        import tensorflow as tf

        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

        memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        _ACTIVE_CONTEXT["memory_policy"] = memory_policy
        tf.config.experimental.enable_tensor_float_32_execution(True)
        tf.config.set_soft_device_placement(False)
        logical_gpus = tuple(tf.config.list_logical_devices("GPU"))
        if len(logical_gpus) != 1:
            raise P1CanaryError(f"expected one visible logical GPU, got {len(logical_gpus)}")
        from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge

        arms: dict[str, Mapping[str, Any]] = {}
        for arm_name, backend in (("factor", FACTOR_BACKEND), ("strict", STRICT_BACKEND)):
            if time.monotonic() - started > MATERIAL_CAP_SECONDS:
                raise P1CanaryError("P1 material cap exhausted before the next arm")
            arm_root = output / arm_name
            profile = _arm_profile(source_module, backend, arm_name)
            bridge = make_q20_tempered_bridge(
                20, jit_compile=True, principal_sqrt_backend=backend
            )
            if str(bridge.target_signature) != TARGET_SIGNATURE:
                raise P1CanaryError(f"{arm_name} target signature changed")
            values, scores, status = bridge.value_score_status(
                tf.stack(
                    (
                        tf.convert_to_tensor(bridge.prior_center, tf.float64),
                        tf.convert_to_tensor(bridge.prior_center, tf.float64)
                        + tf.constant([0.1, -0.1, 0.1, -0.1], tf.float64),
                    )
                ),
                tf.constant(BETA, tf.float64),
            )
            if not bool(tf.reduce_all(tf.math.is_finite(values)).numpy()):
                raise P1CanaryError(f"{arm_name} bridge values are nonfinite")
            if not bool(tf.reduce_all(tf.math.is_finite(scores)).numpy()):
                raise P1CanaryError(f"{arm_name} bridge scores are nonfinite")
            if not bool(tf.reduce_all(status["bridge_valid"]).numpy()):
                raise P1CanaryError(f"{arm_name} bridge status is invalid")
            arms[arm_name] = _run_arm(
                tf=tf,
                source_module=source_module,
                bridge=bridge,
                profile=profile,
                arm_name=arm_name,
                arm_root=arm_root,
            )
            headroom = check_runtime_headroom(
                gpu_selection, int(tf.config.experimental.get_memory_info("GPU:0")["peak"])
            )
            _write_json(arm_root / "gpu-headroom.json", headroom)
        elapsed = time.monotonic() - started
        result = {
            "schema": OUTPUT_SCHEMA,
            "status": "PASS_PHASE9B_P1_CANARY" if all(
                bool(row["sequential"].get("passed")) for row in arms.values()
            ) else "P1_CANARY_COMPLETED_WITH_ARM_FAILURE",
            "role": "cold_scope_sequential_canary",
            "policy_id": POLICY_ID,
            "scope": {"scope_index": SCOPE_INDEX, "chart_index": CHART_INDEX, "beta": BETA},
            "target_signature": TARGET_SIGNATURE,
            "source_inputs": source_inputs,
            "phase0_closeout": phase0_closeout,
            "memory_policy": memory_policy,
            "gpu_placement_policy_id": GPU_PLACEMENT_POLICY_ID,
            "gpu_selection": gpu_selection,
            "logical_gpus": [str(device.name) for device in logical_gpus],
            "tf32_execution_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "gpu_environment": {
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", ""),
            },
            "arms": arms,
            "budget": {
                "material_cap_seconds": MATERIAL_CAP_SECONDS,
                "elapsed_seconds": elapsed,
                "arm_cap_seconds": ARM_CAP_SECONDS,
                "sequential_chunk_forecast_seconds": args.sequential_chunk_forecast_seconds,
                "required_sequential_chunks_per_arm": REQUIRED_SEQUENTIAL_CHUNKS_PER_ARM,
            },
            "run_start": str((output / "run_start.json").relative_to(ROOT)),
            "git": _git_payload(),
            "python": sys.executable,
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "platform": platform.platform(),
            "command": list(sys.argv),
            "claim_boundary": "phase9b_p1_cold_scope_sequential_canary_only",
            "budget_ledger": {
                "path": str(CAMPAIGN_LEDGER_PATH.relative_to(ROOT)),
                "checksum": budget_ledger.checksum(),
                "remaining_seconds_after_attempt": budget_ledger.remaining_seconds(),
            },
            "nonclaims": [
                "no posterior correctness",
                "no ensemble or replica-exchange claim",
                "no whitening",
                "no sampler superiority",
                "no default readiness",
            ],
        }
        _settle_unaccounted_work(
            budget_ledger, output, started,
            status="completed" if result["status"] == "PASS_PHASE9B_P1_CANARY" else "failed",
        )
        budget_ledger.finish_attempt(
            attempt_id=output.name,
            status="completed" if result["status"] == "PASS_PHASE9B_P1_CANARY" else "failed",
            failure_class=None if result["status"] == "PASS_PHASE9B_P1_CANARY" else "p1_arm_failure",
        )
        result["budget_ledger"] = {
            "path": str(CAMPAIGN_LEDGER_PATH.relative_to(ROOT)),
            "checksum": budget_ledger.checksum(),
            "remaining_seconds_after_attempt": budget_ledger.remaining_seconds(),
        }
        result_bytes = _canonical_bytes(result)
        _write_json(output / "result.json", result)
        _write_json(
            output / "run_manifest.json",
            {**result, "manifest_hash": hashlib.sha256(result_bytes).hexdigest()},
        )
        print(json.dumps({"status": result["status"], "elapsed_seconds": elapsed, "output_dir": str(output)}, sort_keys=True))
        return 0 if result["status"] == "PASS_PHASE9B_P1_CANARY" else 2
    except Exception as exc:
        active_ledger = _ACTIVE_CONTEXT.get("budget_ledger")
        if isinstance(active_ledger, CampaignBudgetLedger):
            try:
                _settle_unaccounted_work(active_ledger, output, started, status="failed")
                active_ledger.finish_attempt(
                    attempt_id=output.name,
                    status="failed",
                    failure_class=type(exc).__name__,
                )
            except Exception:
                pass
        payload = _failure_payload(output, exc, traceback_text=traceback.format_exc())
        if not (output / "failure.json").exists():
            _write_json(output / "failure.json", payload)
        print(json.dumps(payload, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
