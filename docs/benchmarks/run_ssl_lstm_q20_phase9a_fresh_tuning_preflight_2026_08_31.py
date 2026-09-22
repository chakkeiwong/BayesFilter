#!/usr/bin/env python3
"""Run the bounded q=20 Phase 9A fresh-map/tuning preflight.

This launch is deliberately below the Phase 9 confirmation boundary.  It
rebuilds two compact-high charts, tunes one fixed kernel per (beta, chart),
and exercises one proper replica-exchange chunk.  The output is mechanics and
provenance evidence only; it is never a posterior or whitening result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-phase9a-fresh-tuning-preflight-subplan-2026-08-31.md"
REPAIR_PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-phase9a-chart1-beta0-program-repair-subplan-2026-09-01.md"
FULL_REPLAY_PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-phase9a-full-replay-performance-subplan-2026-09-01.md"
C5_MANIFEST = ROOT / "docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c5-freeze/attempt-02/freeze_manifest.json"
EXPECTED_TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
EXPECTED_BACKEND = "tensorflow_eigh_strict"
SCHEMA = "bayesfilter.ssl_lstm_q20.tempered_rkl_phase9a_fresh_tuning_preflight.v2"
BETAS = (0.0, 0.5, 1.0)
COMPONENT_IDS = ("phase9a-chart-0", "phase9a-chart-1")
DIMENSION = 4
BATCH_SIZE = 32
TRAIN_UPDATES_PER_LEVEL = 2
CHAIN_COUNT = 4
ALLOCATOR_CAP_BYTES = 4 * 1024**3
MATERIAL_CAP_SECONDS = 1800.0
SCOPE_COUNT = len(BETAS) * len(COMPONENT_IDS)


@dataclass(frozen=True)
class _Phase9AProfile:
    """Immutable, source-owned controls for one bounded Phase 9A launch."""

    profile_id: str
    plan_path: Path
    initialization_roots: tuple[tuple[int, int], ...]
    preflight_roots: tuple[tuple[int, int], ...]
    training_roots: tuple[tuple[int, int], ...]
    tuning_roots: tuple[tuple[int, int], ...]
    transition_root: tuple[int, int]
    reliability_root: tuple[int, int]
    max_step_size: float
    step_size_candidates: tuple[float, ...]
    leapfrog_grid: tuple[int, ...]
    budget_schedule: tuple[int, ...]
    initial_step_size: float
    initial_state_bank: tuple[tuple[float, ...], ...]
    acceptance_band: tuple[float, float]
    repair_band: tuple[float, float]
    screen_num_results: int
    screen_num_burnin_steps: int
    target_accept_prob: float
    selection_replications: int
    selection_num_results: int
    selection_num_burnin_steps: int
    verification_num_results: int
    verification_num_burnin_steps: int
    scope_start: int | None = None
    scope_limit: int | None = None
    material_cap_seconds: float = MATERIAL_CAP_SECONDS

    def payload(self) -> Mapping[str, Any]:
        return {
            "profile_id": self.profile_id,
            "plan_path": str(self.plan_path.relative_to(ROOT)),
            "step_size_candidates": self.step_size_candidates,
            "leapfrog_grid": self.leapfrog_grid,
            "declared_joint_candidate_count": len(self.step_size_candidates)
            * len(self.leapfrog_grid),
            "max_step_size": self.max_step_size,
            "budget_schedule": self.budget_schedule,
            "initial_step_size": self.initial_step_size,
            "initial_state_bank": self.initial_state_bank,
            "acceptance_band": self.acceptance_band,
            "repair_band": self.repair_band,
            "screen_num_results": self.screen_num_results,
            "screen_num_burnin_steps": self.screen_num_burnin_steps,
            "target_accept_prob": self.target_accept_prob,
            "selection_replications": self.selection_replications,
            "selection_num_results": self.selection_num_results,
            "selection_num_burnin_steps": self.selection_num_burnin_steps,
            "verification_num_results": self.verification_num_results,
            "verification_num_burnin_steps": self.verification_num_burnin_steps,
            "seed_namespace": {
                "initialization_roots": self.initialization_roots,
                "preflight_roots": self.preflight_roots,
                "training_roots": self.training_roots,
                "tuning_roots": self.tuning_roots,
                "transition_root": self.transition_root,
                "reliability_root": self.reliability_root,
            },
            "scope_start": self.scope_start,
            "scope_limit": self.scope_limit,
            "material_cap_seconds": self.material_cap_seconds,
        }


_HISTORICAL_PROFILE = _Phase9AProfile(
    profile_id="phase9a_measured_preflight_v1_historical",
    plan_path=PLAN,
    initialization_roots=((20260831, 73001), (20260831, 73002)),
    preflight_roots=((20260831, 73101), (20260831, 73102)),
    training_roots=((20260831, 73201), (20260831, 73202)),
    tuning_roots=(
        (20260831, 73301),
        (20260831, 73302),
        (20260831, 73303),
        (20260831, 73304),
        (20260831, 73305),
        (20260831, 73306),
    ),
    transition_root=(20260831, 73401),
    reliability_root=(20260831, 73501),
    max_step_size=2.0,
    step_size_candidates=(0.4, 0.8, 1.2, 1.6),
    leapfrog_grid=(3, 5, 10),
    budget_schedule=(4, 4, 4),
    initial_step_size=0.01,
    initial_state_bank=(
        (0.0, 0.0, 0.0, 0.0),
        (0.10, 0.0, 0.0, 0.0),
        (-0.10, 0.0, 0.0, 0.0),
        (0.0, 0.10, 0.0, 0.0),
    ),
    acceptance_band=(0.45, 0.90),
    repair_band=(0.30, 0.95),
    screen_num_results=8,
    screen_num_burnin_steps=2,
    target_accept_prob=0.70,
    selection_replications=2,
    selection_num_results=64,
    selection_num_burnin_steps=32,
    verification_num_results=8,
    verification_num_burnin_steps=2,
)

_CHART1_BETA0_REPAIR_PROFILE = _Phase9AProfile(
    profile_id="chart1_beta0_repair_v1",
    plan_path=REPAIR_PLAN,
    initialization_roots=((20260901, 74001), (20260901, 74002)),
    preflight_roots=((20260901, 74101), (20260901, 74102)),
    training_roots=((20260901, 74201), (20260901, 74202)),
    tuning_roots=(
        (20260901, 74301),
        (20260901, 74302),
        (20260901, 74303),
        (20260901, 74304),
        (20260901, 74305),
        (20260901, 74306),
    ),
    transition_root=(20260901, 74401),
    reliability_root=(20260901, 74501),
    max_step_size=2.0,
    step_size_candidates=(0.25, 0.40, 0.55, 0.70, 0.85, 1.00, 1.20, 1.40),
    leapfrog_grid=(3, 5, 8, 12),
    budget_schedule=(4, 4, 4),
    initial_step_size=0.10,
    initial_state_bank=(
        (0.0, 0.0, 0.0, 0.0),
        (0.25, 0.0, 0.0, 0.0),
        (-0.25, 0.0, 0.0, 0.0),
        (0.0, 0.25, 0.0, 0.0),
    ),
    acceptance_band=(0.45, 0.90),
    repair_band=(0.30, 0.95),
    screen_num_results=16,
    screen_num_burnin_steps=8,
    target_accept_prob=0.70,
    selection_replications=2,
    selection_num_results=64,
    selection_num_burnin_steps=16,
    verification_num_results=64,
    verification_num_burnin_steps=32,
    scope_start=3,
    scope_limit=1,
)

# The first repair profile was scientifically well specified but exceeded the
# bounded wall budget because every measured pair receives a full-chain screen
# and replicated selection.  This second, fresh profile keeps the measured
# joint-grid policy while reducing only the localization grid and draw counts;
# it is still mechanics evidence and cannot issue a confirmation handoff.
_CHART1_BETA0_BOUNDED_PROFILE = _Phase9AProfile(
    profile_id="chart1_beta0_repair_v2_bounded",
    plan_path=REPAIR_PLAN,
    initialization_roots=((20260901, 75001), (20260901, 75002)),
    preflight_roots=((20260901, 75101), (20260901, 75102)),
    training_roots=((20260901, 75201), (20260901, 75202)),
    tuning_roots=(
        (20260901, 75301),
        (20260901, 75302),
        (20260901, 75303),
        (20260901, 75304),
        (20260901, 75305),
        (20260901, 75306),
    ),
    transition_root=(20260901, 75401),
    reliability_root=(20260901, 75501),
    max_step_size=2.0,
    step_size_candidates=(0.25, 0.55, 0.85, 1.20),
    leapfrog_grid=(3, 8),
    budget_schedule=(2, 2, 2),
    initial_step_size=0.10,
    initial_state_bank=(
        (0.0, 0.0, 0.0, 0.0),
        (0.25, 0.0, 0.0, 0.0),
        (-0.25, 0.0, 0.0, 0.0),
        (0.0, 0.25, 0.0, 0.0),
    ),
    acceptance_band=(0.45, 0.90),
    repair_band=(0.30, 0.95),
    screen_num_results=4,
    screen_num_burnin_steps=2,
    target_accept_prob=0.70,
    selection_replications=2,
    selection_num_results=16,
    selection_num_burnin_steps=4,
    verification_num_results=16,
    verification_num_burnin_steps=4,
    scope_start=3,
    scope_limit=1,
)

_CHART1_BETA0_MINIMAL_PROFILE = _Phase9AProfile(
    profile_id="chart1_beta0_repair_v3_minimal",
    plan_path=REPAIR_PLAN,
    initialization_roots=((20260901, 76001), (20260901, 76002)),
    preflight_roots=((20260901, 76101), (20260901, 76102)),
    training_roots=((20260901, 76201), (20260901, 76202)),
    tuning_roots=(
        (20260901, 76301),
        (20260901, 76302),
        (20260901, 76303),
        (20260901, 76304),
        (20260901, 76305),
        (20260901, 76306),
    ),
    transition_root=(20260901, 76401),
    reliability_root=(20260901, 76501),
    max_step_size=2.0,
    step_size_candidates=(0.55, 1.20),
    leapfrog_grid=(3, 8),
    budget_schedule=(1, 1, 1),
    initial_step_size=0.10,
    initial_state_bank=(
        (0.0, 0.0, 0.0, 0.0),
        (0.25, 0.0, 0.0, 0.0),
        (-0.25, 0.0, 0.0, 0.0),
        (0.0, 0.25, 0.0, 0.0),
    ),
    acceptance_band=(0.45, 0.90),
    repair_band=(0.30, 0.95),
    screen_num_results=1,
    screen_num_burnin_steps=1,
    target_accept_prob=0.70,
    selection_replications=2,
    selection_num_results=4,
    selection_num_burnin_steps=1,
    verification_num_results=4,
    verification_num_burnin_steps=1,
    scope_start=3,
    scope_limit=1,
)

_CHART1_BETA0_FRESH_PROFILE = replace(
    _CHART1_BETA0_MINIMAL_PROFILE,
    profile_id="chart1_beta0_repair_v4_fresh",
    initialization_roots=((20260901, 77001), (20260901, 77002)),
    preflight_roots=((20260901, 77101), (20260901, 77102)),
    training_roots=((20260901, 77201), (20260901, 77202)),
    tuning_roots=(
        (20260901, 77301),
        (20260901, 77302),
        (20260901, 77303),
        (20260901, 77304),
        (20260901, 77305),
        (20260901, 77306),
    ),
    transition_root=(20260901, 77401),
    reliability_root=(20260901, 77501),
)

# Canary and full replay use disjoint source-owned seed namespaces.  The
# canary is calibration only; it cannot silently become replay evidence.
_FULL_REPLAY_CANARY_PROFILE = replace(
    _CHART1_BETA0_BOUNDED_PROFILE,
    profile_id="phase9a_full_replay_canary_v1",
    plan_path=FULL_REPLAY_PLAN,
    initialization_roots=((20260902, 78001), (20260902, 78002)),
    preflight_roots=((20260902, 78101), (20260902, 78102)),
    training_roots=((20260902, 78201), (20260902, 78202)),
    tuning_roots=(
        (20260902, 78301),
        (20260902, 78302),
        (20260902, 78303),
        (20260902, 78304),
        (20260902, 78305),
        (20260902, 78306),
    ),
    transition_root=(20260902, 78401),
    reliability_root=(20260902, 78501),
    scope_start=3,
    scope_limit=1,
    material_cap_seconds=1800.0,
)

_FULL_REPLAY_PROFILE = replace(
    _FULL_REPLAY_CANARY_PROFILE,
    profile_id="phase9a_full_replay_v1",
    initialization_roots=((20260902, 79001), (20260902, 79002)),
    preflight_roots=((20260902, 79101), (20260902, 79102)),
    training_roots=((20260902, 79201), (20260902, 79202)),
    tuning_roots=(
        (20260902, 79301),
        (20260902, 79302),
        (20260902, 79303),
        (20260902, 79304),
        (20260902, 79305),
        (20260902, 79306),
    ),
    transition_root=(20260902, 79401),
    reliability_root=(20260902, 79501),
    scope_start=0,
    scope_limit=SCOPE_COUNT,
    material_cap_seconds=7800.0,
)

_PROFILES = {
    _HISTORICAL_PROFILE.profile_id: _HISTORICAL_PROFILE,
    _CHART1_BETA0_REPAIR_PROFILE.profile_id: _CHART1_BETA0_REPAIR_PROFILE,
    _CHART1_BETA0_BOUNDED_PROFILE.profile_id: _CHART1_BETA0_BOUNDED_PROFILE,
    _CHART1_BETA0_MINIMAL_PROFILE.profile_id: _CHART1_BETA0_MINIMAL_PROFILE,
    _CHART1_BETA0_FRESH_PROFILE.profile_id: _CHART1_BETA0_FRESH_PROFILE,
    _FULL_REPLAY_CANARY_PROFILE.profile_id: _FULL_REPLAY_CANARY_PROFILE,
    _FULL_REPLAY_PROFILE.profile_id: _FULL_REPLAY_PROFILE,
}


def _scope_pairs() -> tuple[tuple[int, float], ...]:
    """Return the repository-owned, chart-major scope ordering."""

    return tuple(
        (chart_index, float(beta))
        for chart_index in range(len(COMPONENT_IDS))
        for beta in BETAS
    )


def _resolve_profile(
    profile_id: str,
    *,
    scope_start: int | None = None,
    scope_limit: int | None = None,
) -> _Phase9AProfile:
    """Resolve and validate one immutable launch profile.

    Scope overrides are allowed for the historical diagnostic profile and the
    full replay profile so bounded slices remain reproducible.  Localized
    repair profiles and the replay canary are intentionally pinned to
    chart-1/beta-0 and cannot be widened by a caller.
    """

    try:
        base = _PROFILES[str(profile_id)]
    except KeyError as exc:
        raise Phase9AError(
            f"unknown Phase 9A profile {profile_id!r}; choose one of {tuple(_PROFILES)}"
        ) from exc
    start = base.scope_start if scope_start is None else int(scope_start)
    limit = base.scope_limit if scope_limit is None else int(scope_limit)
    if start is None:
        start = 0
    if limit is None:
        limit = SCOPE_COUNT
    if start < 0 or start >= SCOPE_COUNT:
        raise Phase9AError(f"scope-start must lie in [0,{SCOPE_COUNT - 1}]")
    if limit < 1 or limit > SCOPE_COUNT:
        raise Phase9AError(f"scope-limit must lie in [1,{SCOPE_COUNT}]")
    if start + limit > SCOPE_COUNT:
        raise Phase9AError(
            f"scope interval [{start},{start + limit}) exceeds {SCOPE_COUNT} scopes"
        )
    if base.profile_id in {
        _CHART1_BETA0_REPAIR_PROFILE.profile_id,
        _CHART1_BETA0_BOUNDED_PROFILE.profile_id,
        _CHART1_BETA0_MINIMAL_PROFILE.profile_id,
        _CHART1_BETA0_FRESH_PROFILE.profile_id,
    }:
        expected = (3, 1)
        if (start, limit) != expected:
            raise Phase9AError(
                "chart1 beta-0 repair profiles are pinned to --scope-start 3 --scope-limit 1"
            )
    if base.profile_id == _FULL_REPLAY_CANARY_PROFILE.profile_id:
        if (start, limit) != (3, 1):
            raise Phase9AError(
                "Phase 9A full-replay canary is pinned to --scope-start 3 --scope-limit 1"
            )
    if not (0.0 < float(base.material_cap_seconds) <= 7800.0):
        raise Phase9AError("profile material cap must lie in (0,7800] seconds")
    return replace(base, scope_start=start, scope_limit=limit)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class Phase9AError(RuntimeError):
    """Raised when a Phase 9A hard screen fails."""


_ACTIVE_RUN_CONTEXT: dict[str, Any] = {}


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        _json_ready(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git(command: Sequence[str]) -> str:
    try:
        return subprocess.check_output(
            tuple(command), cwd=ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable:{type(exc).__name__}"


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "numpy"):
        return _json_ready(value.numpy())
    if hasattr(value, "tolist"):
        return _json_ready(value.tolist())
    if hasattr(value, "item"):
        return _json_ready(value.item())
    return str(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise Phase9AError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(value), sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _nvidia_snapshot() -> Mapping[str, Any]:
    command = (
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
        return {"command": list(command), "rows": output.strip().splitlines()}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"command": list(command), "error": type(exc).__name__}


def _memory_info(tf: Any, device_name: str) -> Mapping[str, Any]:
    try:
        return _json_ready(tf.config.experimental.get_memory_info(device_name))
    except (AttributeError, RuntimeError, ValueError) as exc:
        return {"unavailable": type(exc).__name__}


def _route_scan() -> Mapping[str, Any]:
    # Keep the scan focused on the executable route.  The forbidden strings
    # themselves are intentionally constructed here rather than searched in
    # this benchmark source.
    paths = (
        ROOT / "bayesfilter/inference/tempered_target_tf.py",
        ROOT / "bayesfilter/inference/tempered_transport_ensemble_tf.py",
        ROOT / "bayesfilter/inference/tempered_transitions_tf.py",
        ROOT / "bayesfilter/inference/fixed_transport_hmc_tuning_tf.py",
    )
    tokens = ("tf." + "map_fn", "tf." + "vectorized_map", "p" + "for")
    hits = {token: [] for token in tokens}
    for path in paths:
        source = path.read_text(encoding="utf-8")
        for token in tokens:
            if token in source:
                hits[token].append(str(path.relative_to(ROOT)))
    return {
        "paths": [str(path.relative_to(ROOT)) for path in paths],
        "forbidden_tokens": list(tokens),
        "hits": hits,
        "passed": not any(hits.values()),
    }


def _seed(tf: Any, root: tuple[int, int], *folds: int) -> tuple[int, int]:
    seed = tf.constant(root, tf.int32)
    for fold in folds:
        seed = tf.random.experimental.stateless_fold_in(seed, int(fold))
    return tuple(int(item) for item in seed.numpy().tolist())


def _scope(
    component_id: str,
    beta: float,
    chart_index: int,
    profile: _Phase9AProfile | None = None,
) -> Mapping[str, Any]:
    profile = _HISTORICAL_PROFILE if profile is None else profile
    return {
        "data_identity": f"ssl-lstm-q20:{EXPECTED_TARGET_SIGNATURE}",
        "target_signature": EXPECTED_TARGET_SIGNATURE,
        "bridge_backend": EXPECTED_BACKEND,
        "dtype": "float64",
        "backend": "tensorflow_tfp_gpu",
        "jit_compile": True,
        "tf32_execution_enabled": True,
        "component_id": component_id,
        "chart_index": int(chart_index),
        "beta": float(beta),
        "ladder": list(BETAS),
        "component_count": len(COMPONENT_IDS),
        "lineage_policy": "pure_continuation",
        "gamma_policy": {
            "policy_id": "fixed_state_independent_chart_mixture_v1",
            "values": [0.5, 0.5],
        },
        "training_updates_per_level": TRAIN_UPDATES_PER_LEVEL,
        "profile_id": profile.profile_id,
        "tuning_policy": "measured_joint_grid_v1",
        "phase_role": "phase9a_mechanics_preflight_only",
    }


def _check_prerequisites(profile: _Phase9AProfile) -> Mapping[str, Any]:
    if not profile.plan_path.is_file() or not C5_MANIFEST.is_file():
        raise Phase9AError("Phase 9A plan or C5 freeze manifest is missing")
    c5 = json.loads(C5_MANIFEST.read_text(encoding="utf-8"))
    if c5.get("status") != "PASS_PHASE8_C5_FREEZE":
        raise Phase9AError("C5 freeze status is not passing")
    if c5.get("target_signature") != EXPECTED_TARGET_SIGNATURE:
        raise Phase9AError("C5 target signature mismatch")
    selected = c5.get("candidates", {}).get("k2", {})
    if selected.get("candidate_id") != "phase8-k2-compact-high-l3-pure":
        raise Phase9AError("C5 did not freeze the compact-high pure L3 protocol")
    if selected.get("status") != "FROZEN_FOR_PHASE9_TUNING_ONLY":
        raise Phase9AError("C5 candidate is not marked tuning-only")
    return {
        "plan": {
            "path": str(profile.plan_path.relative_to(ROOT)),
            "sha256": _sha256(profile.plan_path),
            "profile_id": profile.profile_id,
        },
        "c5_freeze": {
            "path": str(C5_MANIFEST.relative_to(ROOT)),
            "sha256": _sha256(C5_MANIFEST),
            "status": c5["status"],
            "candidate_id": selected["candidate_id"],
        },
    }


def _finite_bool(tf: Any, value: Any) -> bool:
    return bool(tf.reduce_all(tf.math.is_finite(tf.convert_to_tensor(value))).numpy())


def _checkpoint_scope(
    component_id: str,
    beta: float,
    chart_index: int,
    profile: _Phase9AProfile,
) -> Mapping[str, Any]:
    return {
        "data_identity": f"ssl-lstm-q20:{EXPECTED_TARGET_SIGNATURE}",
        "dtype": "float64",
        "backend": "tensorflow_tfp_gpu",
        "jit_compile": True,
        "training_seed_derivation": {
            "initialization_root": list(profile.initialization_roots[chart_index]),
            "preflight_root": list(profile.preflight_roots[chart_index]),
            "training_root": list(profile.training_roots[chart_index]),
            "beta": float(beta),
        },
        "validation_bank_ids": [
            f"phase9a-{component_id}-beta-{beta:g}-preflight"
        ],
        "profile_id": profile.profile_id,
    }


def _build_fresh_chart(
    tf: Any,
    bridge: Any,
    chart_index: int,
    component_id: str,
    profile: _Phase9AProfile,
    artifact_root: Path | None = None,
) -> tuple[dict[float, Any], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    from bayesfilter.inference.neutra_weighted_training import (
        WeightedDenseIAFTransport,
        WeightedNeuTraConfig,
    )
    from bayesfilter.inference.tempered_transport_ensemble_tf import (
        IndependentTemperedReverseKLTrainer,
        capture_trainable_transport_checkpoint,
        prepare_transport_initialization,
        restore_trainable_transport_checkpoint,
        transport_preflight_state_hash,
    )

    config = WeightedNeuTraConfig(
        dimension=DIMENSION,
        hidden_layers=(16, 16),
        stages=2,
        activation="tanh",
        initialization_scale=0.02,
        initialization_seed=profile.initialization_roots[chart_index],
        learning_rate=1.0e-3,
        jit_compile=True,
    )
    raw = WeightedDenseIAFTransport(config)
    center = tf.convert_to_tensor(bridge.prior_center, tf.float64)
    prior_scale = tf.fill([DIMENSION], tf.sqrt(tf.constant(float(bridge.prior_variance), tf.float64)))
    checkpoints: list[Mapping[str, Any]] = []
    receipts: list[Mapping[str, Any]] = []
    level_charts: dict[float, Any] = {}
    current = raw
    parent_hash: str | None = None
    for beta_index, beta in enumerate(BETAS):
        prepared = prepare_transport_initialization(
            current,
            bridge,
            component_id=component_id,
            seed=_seed(tf, profile.preflight_roots[chart_index], beta_index),
            batch_size=BATCH_SIZE,
            repair_scales=(1.0,),
            beta=beta,
            reference_center=center if beta_index == 0 else None,
            reference_scale=prior_scale if beta_index == 0 else None,
        )
        if not prepared.receipt.valid or not prepared.receipt.optimizer_state_absent:
            raise Phase9AError(f"fresh chart preflight failed: {component_id}, beta={beta}")
        current = prepared.transport
        receipts.append(prepared.receipt.payload())
        if artifact_root is not None:
            receipt_dir = artifact_root / f"chart-{chart_index}" / f"beta-{beta:g}"
            _write_json(
                receipt_dir / "chart_preflight_receipt.json",
                {
                    "schema": "bayesfilter.ssl_lstm_q20.phase9a.chart_preflight.v1",
                    "status": "PASS_CHART_PREFLIGHT",
                    "profile_id": profile.profile_id,
                    "component_id": component_id,
                    "chart_index": chart_index,
                    "beta": beta,
                    "receipt": prepared.receipt.payload(),
                },
            )
        trainer = IndependentTemperedReverseKLTrainer(
            config,
            bridge,
            beta=beta,
            component_id=component_id,
            batch_size=BATCH_SIZE,
            prepared_initialization=prepared,
        )
        updates = []
        for update_index in range(TRAIN_UPDATES_PER_LEVEL):
            update = trainer.train_step(
                _seed(tf, profile.training_roots[chart_index], beta_index, update_index)
            )
            if not bool(update.valid.numpy()):
                raise Phase9AError(f"fresh chart update invalid: {component_id}, beta={beta}")
            updates.append(
                {
                    "update": update_index + 1,
                    "loss": update.loss,
                    "gradient_norm": update.gradient_norm,
                    "step": update.step,
                    "target_call_count": update.target_call_count,
                    "valid": update.valid,
                }
            )
        training_trace_count = int(
            trainer._compiled_train_step.experimental_get_tracing_count()
        )
        if training_trace_count != 1:
            raise Phase9AError(
                f"fresh chart training graph retraced: {component_id}, beta={beta}, "
                f"count={training_trace_count}"
            )
        checkpoint = capture_trainable_transport_checkpoint(
            current,
            component_id=component_id,
            beta=beta,
            bridge_signature=str(bridge.signature),
            target_signature=EXPECTED_TARGET_SIGNATURE,
            parent_checkpoint_hash=parent_hash,
            update_count=TRAIN_UPDATES_PER_LEVEL,
            checkpoint_scope=_checkpoint_scope(component_id, beta, chart_index, profile),
        )
        parent_hash = str(checkpoint["checkpoint_hash"])
        if artifact_root is not None:
            checkpoint_dir = artifact_root / f"chart-{chart_index}" / f"beta-{beta:g}"
            _write_json(
                checkpoint_dir / "chart_checkpoint.json",
                {
                    "schema": "bayesfilter.ssl_lstm_q20.phase9a.chart_checkpoint.v1",
                    "status": "PASS_FRESH_CHART_CHECKPOINT",
                    "profile_id": profile.profile_id,
                    "component_id": component_id,
                    "chart_index": chart_index,
                    "beta": beta,
                    "checkpoint": checkpoint,
                },
            )
        # Snapshot the exact frozen map for this temperature.  Restoration
        # verifies the tensors, but deliberately does not invent an HMC
        # identity; bind that identity from the repository-issued checkpoint
        # before the adapter is constructed.
        restored = restore_trainable_transport_checkpoint(checkpoint)
        restored_binder = getattr(restored, "bind_frozen_identity", None)
        if not callable(restored_binder):
            raise Phase9AError(
                f"restored fresh chart lacks frozen-identity binding: {component_id}, beta={beta}"
            )
        restored_state_hash = transport_preflight_state_hash(restored)
        if restored_state_hash != checkpoint["transport_state_hash"]:
            raise Phase9AError(
                f"restored fresh chart state hash changed: {component_id}, beta={beta}"
            )
        restored_binder(
            {
                "checkpoint_sha256": checkpoint["checkpoint_hash"],
                "training_state_hash": checkpoint["transport_state_hash"],
                "transport_tensor_hash": restored_state_hash,
            }
        )
        level_charts[float(beta)] = restored
        checkpoints.append(
            {
                "checkpoint": checkpoint,
                "updates": updates,
                "compiled_training_trace_count": training_trace_count,
                "state_hash": transport_preflight_state_hash(current),
            }
        )
    return level_charts, checkpoints, receipts


def _reliability(
    tf: Any,
    bridge: Any,
    charts: Sequence[Any],
    *,
    beta: float,
    profile: _Phase9AProfile,
) -> Mapping[str, Any]:
    from bayesfilter.inference.tempered_transport_ensemble_tf import (
        pullback_gaussianization_diagnostic,
        transport_preflight_state_hash,
    )
    from bayesfilter.inference.tempered_transitions_tf import screen_transport_reliability

    center = tf.convert_to_tensor(bridge.prior_center, tf.float64)
    latent = tf.random.stateless_normal(
        [len(charts), 64, DIMENSION],
        tf.constant(_seed(tf, profile.reliability_root, 0), tf.int32),
        dtype=tf.float64,
    )
    physical = tf.stack(
        [charts[index].forward_batch(latent[index]) for index in range(len(charts))], axis=0
    )
    adapter = bridge.fixed_beta_adapter(beta)
    score_fn = lambda values: adapter.log_prob_and_grad(values)[1]
    reference = tf.concat(
        (center[tf.newaxis, :], center[tf.newaxis, :] + 4.0 * tf.eye(DIMENSION, dtype=tf.float64)), axis=0
    )
    receipt = screen_transport_reliability(
        charts,
        component_ids=COMPONENT_IDS,
        self_latent_bank=latent,
        cross_physical_bank=physical,
        reference_points=reference,
        declared_points=reference,
        physical_score_fn=score_fn,
        maximum_condition_number=1.0e8,
        tolerance=1.0e-8,
    )
    if not receipt.passed:
        raise Phase9AError(f"fresh chart reliability failed: {receipt.payload()}")
    diagnostics = []
    for index, chart in enumerate(charts):
        rows = tf.random.stateless_normal(
            [64, DIMENSION],
            tf.constant(_seed(tf, profile.reliability_root, 1, index), tf.int32),
            dtype=tf.float64,
        )
        value = pullback_gaussianization_diagnostic(
            chart, bridge, beta=beta, latent=rows
        )
        if not bool(value.finite.numpy()):
            raise Phase9AError(f"fresh chart pullback diagnostic nonfinite: {COMPONENT_IDS[index]}")
        diagnostics.append(
            {
                "component_id": COMPONENT_IDS[index],
                "centered_log_density_rms": value.centered_log_density_rms,
                "pullback_score_rms_per_coordinate": value.pullback_score_rms_per_coordinate,
                "pullback_score_maximum_row_norm": value.pullback_score_maximum_row_norm,
                "state_hash": transport_preflight_state_hash(chart),
            }
        )
    return {"reliability": receipt.payload(), "pullback": diagnostics}


def _tune_scope(
    tf: Any,
    bridge: Any,
    chart: Any,
    *,
    chart_index: int,
    beta: float,
    output_dir: Path,
    profile: _Phase9AProfile,
    scope_index: int,
) -> Mapping[str, Any]:
    from bayesfilter.inference.fixed_transport_hmc_tuning_tf import (
        FixedTransportHMCKernelTuningConfig,
        build_verified_fixed_transport_hmc_handoff_from_tuning_result,
        tune_fixed_transport_hmc_kernel,
    )
    from bayesfilter.inference.fixed_transport_hmc_mechanics_tf import (
        FixedTransportReusableRunnerPool,
    )

    adapter = bridge.fixed_beta_adapter(beta)
    z_bank = profile.initial_state_bank
    scope = _scope(COMPONENT_IDS[chart_index], beta, chart_index, profile)
    tuning_root = profile.tuning_roots[scope_index]
    call_root = output_dir / "full_chain_calls"
    runner_pool = FixedTransportReusableRunnerPool()
    call_counter = 0

    def timed_run(adapter_value: Any, initial_state: Any, chain_config: Any) -> Any:
        nonlocal call_counter
        call_index = call_counter
        call_counter += 1
        config_payload = (
            chain_config.signature_payload()
            if callable(getattr(chain_config, "signature_payload", None))
            else {
                "num_results": getattr(chain_config, "num_results", None),
                "num_burnin_steps": getattr(chain_config, "num_burnin_steps", None),
                "step_size": getattr(chain_config, "step_size", None),
                "num_leapfrog_steps": getattr(chain_config, "num_leapfrog_steps", None),
                "seed": getattr(chain_config, "seed", None),
            }
        )
        _write_json(
            call_root / f"call-{call_index:03d}-start.json",
            {
                "schema": "bayesfilter.ssl_lstm_q20.phase9a.full_chain_call_start.v1",
                "status": "RUNNING_FULL_CHAIN_CALL",
                "profile_id": profile.profile_id,
                "scope_index": scope_index,
                "chart_index": chart_index,
                "beta": beta,
                "call_index": call_index,
                "config": config_payload,
                "started_at_unix": time.time(),
            },
        )
        pool_before = runner_pool.evidence()
        started_call = time.monotonic()
        try:
            result_value = runner_pool(adapter_value, initial_state, chain_config)
        except Exception as exc:
            _write_json(
                call_root / f"call-{call_index:03d}-failure.json",
                {
                    "schema": "bayesfilter.ssl_lstm_q20.phase9a.full_chain_call_failure.v1",
                    "status": "FAIL_FULL_CHAIN_CALL",
                    "profile_id": profile.profile_id,
                    "scope_index": scope_index,
                    "call_index": call_index,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "elapsed_seconds": time.monotonic() - started_call,
                },
            )
            raise
        diagnostics = getattr(result_value, "diagnostics", {})
        pool_after = runner_pool.evidence()
        metadata_value = getattr(result_value, "metadata", {})
        metadata = metadata_value if isinstance(metadata_value, Mapping) else {}
        before_count = int(pool_before.get("runner_count", 0))
        after_count = int(pool_after.get("runner_count", 0))
        _write_json(
            call_root / f"call-{call_index:03d}-complete.json",
            {
                "schema": "bayesfilter.ssl_lstm_q20.phase9a.full_chain_call_complete.v1",
                "status": "PASS_FULL_CHAIN_CALL",
                "profile_id": profile.profile_id,
                "scope_index": scope_index,
                "call_index": call_index,
                "elapsed_seconds": time.monotonic() - started_call,
                "config": config_payload,
                "diagnostics": diagnostics,
                "runner_pool": pool_after,
                "runner_created_on_call": after_count > before_count,
                "sample_chain_call_seconds": metadata.get("sample_chain_call_s"),
                "timing_role": (
                    "first_trace_or_compile_included"
                    if after_count > before_count
                    else "steady_state_reused_runner"
                ),
            },
        )
        return result_value

    setattr(timed_run, "evidence", runner_pool.evidence)
    config = FixedTransportHMCKernelTuningConfig(
        initial_step_size=profile.initial_step_size,
        maximum_candidate_step_size=profile.max_step_size,
        step_size_candidates=profile.step_size_candidates,
        leapfrog_grid=profile.leapfrog_grid,
        chain_count=CHAIN_COUNT,
        initial_state_bank=z_bank,
        target_accept_prob=profile.target_accept_prob,
        acceptance_band=profile.acceptance_band,
        repair_band=profile.repair_band,
        budget_schedule=profile.budget_schedule,
        tune_num_results=profile.budget_schedule[0],
        screen_num_results=profile.screen_num_results,
        screen_num_burnin_steps=profile.screen_num_burnin_steps,
        selection_policy="replicated_min_bulk_ess_per_gradient",
        selection_replications=profile.selection_replications,
        selection_num_results=profile.selection_num_results,
        selection_num_burnin_steps=profile.selection_num_burnin_steps,
        verification_num_results=profile.verification_num_results,
        verification_num_burnin_steps=profile.verification_num_burnin_steps,
        require_modern_rank_normalized_verification=False,
        report_modern_rank_normalized_verification=False,
        tune_seed_base=tuning_root,
        screen_seed_base=(tuning_root[0], tuning_root[1] + 100),
        selection_seed_base=(tuning_root[0], tuning_root[1] + 150),
        verification_seed_base=(tuning_root[0], tuning_root[1] + 200),
        chain_execution_mode="tf_function",
        use_xla=True,
        target_scope=f"{adapter.target_scope}:chart={COMPONENT_IDS[chart_index]}:phase9a",
        target_status_trace_policy="per_chain_step",
        output_filename="fixed_transport_hmc_tuning_result.json",
        tuning_policy="measured_joint_grid_v1",
    )
    result = tune_fixed_transport_hmc_kernel(
        base_adapter=adapter,
        fixed_transport=chart,
        initial_position=tf.zeros([DIMENSION], tf.float64),
        config=config,
        output_dir=output_dir,
        run_full_chain=timed_run,
    )
    if not result.passed:
        raise Phase9AError(
            f"scope tuner did not produce a handoff: {COMPONENT_IDS[chart_index]}, beta={beta}, "
            f"status={result.final_status}, vetoes={result.hard_vetoes}"
        )
    handoff = build_verified_fixed_transport_hmc_handoff_from_tuning_result(
        tuning_result=result,
        base_adapter=adapter,
        fixed_transport=chart,
    )
    return {
        "component_id": COMPONENT_IDS[chart_index],
        "chart_index": chart_index,
        "scope_index": scope_index,
        "beta": beta,
        "scope": scope,
        "tuning_result": result.payload(),
        "tuning_artifact": str(output_dir / config.output_filename),
        "handoff": handoff.payload(),
        "handoff_hash": handoff.handoff_hash,
        "step_size": handoff.step_size,
        "num_leapfrog_steps": handoff.num_leapfrog_steps,
        "_live_handoff": handoff,
        "_live_tuning_result": result,
    }


def _run_transition(
    tf: Any,
    bridge: Any,
    charts_by_beta: Mapping[float, Sequence[Any]],
    tuned: Sequence[Mapping[str, Any]],
    device_name: str,
    profile: _Phase9AProfile,
) -> Mapping[str, Any]:
    from bayesfilter.inference.tempered_transitions_tf import (
        BoundWithinTemperatureKernel,
        FixedChartKernelMixture,
        ProperBridgeReplicaExchange,
        ProperReplicaExchangeTransitionProgram,
        build_tuned_fixed_transport_hmc_kernel,
    )
    from bayesfilter.inference.neutra_hmc import (
        SequentialExactTransitionConfig,
        run_sequential_exact_transition,
    )

    by_beta: dict[float, list[Mapping[str, Any]]] = {beta: [] for beta in BETAS}
    for row in tuned:
        by_beta[float(row["beta"])].append(row)
    bindings = []
    for beta in BETAS:
        kernels = []
        for chart_index, chart in enumerate(charts_by_beta[beta]):
            row = next(item for item in by_beta[beta] if int(item["chart_index"]) == chart_index)
            # Rebuild the typed handoff from the durable tuner result payload is
            # intentionally deferred to the full Phase 9B loader.  The object
            # used here is created in the same process and has already passed
            # the exact handoff constructor above.
            handoff = row["_live_handoff"]
            kernels.append(build_tuned_fixed_transport_hmc_kernel(handoff, state_shape=(CHAIN_COUNT, DIMENSION)))
        mixture = FixedChartKernelMixture(
            tuple(kernel for kernel in kernels),
            gamma=(0.5, 0.5),
            chart_ids=COMPONENT_IDS,
        )
        bindings.append(
            BoundWithinTemperatureKernel(
                beta=beta,
                bridge_signature=str(bridge.signature),
                kernel_signature=mixture.selection.signature,
                kernel=mixture.transition,
                mechanics_role="phase9a_verified_scope_specific_tuner_handoff",
            )
        )
    exchange = ProperBridgeReplicaExchange(bridge, BETAS)
    program = ProperReplicaExchangeTransitionProgram(exchange, tuple(bindings), jit_compile=True)
    center = tf.convert_to_tensor(bridge.prior_center, tf.float64)
    initial = tf.broadcast_to(center, [len(BETAS), CHAIN_COUNT, DIMENSION])
    perturb = tf.constant(
        [
            [[0.00, 0.00, 0.00, 0.00], [0.10, 0.00, 0.00, 0.00], [-0.10, 0.00, 0.00, 0.00], [0.00, 0.10, 0.00, 0.00]],
            [[0.05, 0.00, 0.00, 0.00], [0.00, -0.05, 0.00, 0.00], [0.00, 0.00, 0.05, 0.00], [0.00, 0.00, 0.00, -0.05]],
            [[0.15, 0.00, 0.00, 0.00], [0.00, -0.15, 0.00, 0.00], [0.00, 0.00, 0.15, 0.00], [0.00, 0.00, 0.00, -0.15]],
        ],
        tf.float64,
    )
    initial = initial + perturb
    # Exercise the same sequential controller used by claim-bearing routes.
    # Four draws are the smallest legal four-chain mechanics schedule.  The
    # deliberately permissive R-hat threshold and finite-only retained
    # diagnostic are explanatory preflight settings, never convergence gates.
    controller = run_sequential_exact_transition(
        transition_program=program,
        initial_transition_state=program.initial_state(initial),
        posterior_state_fn=program.posterior_state,
        parameter_names=tuple(f"theta.{index}" for index in range(DIMENSION)),
        config=SequentialExactTransitionConfig(
            transition_signature=program.transition_signature,
            warmup_seed=_seed(tf, profile.transition_root, 1),
            retained_seed=_seed(tf, profile.transition_root, 2),
            warmup_chunk_results=4,
            warmup_min_results=4,
            warmup_check_window_results=4,
            warmup_max_results=4,
            warmup_rhat_max=100.0,
            retained_chunk_results=4,
            retained_min_results=4,
            retained_max_results=4,
            retained_rhat_max=100.0,
        ),
        retained_diagnostic_fn=lambda samples: {
            "passed": _finite_bool(tf, samples),
            "hard_vetoes": ()
            if _finite_bool(tf, samples)
            else ("mechanics_preflight_nonfinite_retained_samples",),
            "role": "phase9a_mechanics_preflight_no_convergence",
        },
    )
    if controller["hard_vetoes"] or not controller["passed"]:
        raise Phase9AError(
            "sequential controller mechanics preflight failed: "
            f"passed={controller['passed']}, vetoes={controller['hard_vetoes']}"
        )
    return {
        "transition_signature": program.transition_signature,
        "controller_passed": bool(controller["passed"]),
        "controller_policy_id": controller["policy_id"],
        "controller_config": controller["config"],
        "warmup_results_per_chain": controller["warmup_results_per_chain"],
        "retained_results_per_chain": controller["retained_results_per_chain"],
        "warmup_checks": controller["warmup_checks"],
        "retained_checks": controller["retained_checks"],
        "hard_vetoes": controller["hard_vetoes"],
        "posterior_sample_shape": tuple(
            int(value) for value in controller["private_retained_beta_one"].shape
        ),
        "posterior_stream_only": controller["posterior_stream_only"],
        "posterior_temperature": controller["posterior_temperature"],
        "posterior_replica_identities": controller[
            "private_retained_replica_identities"
        ],
        "device": device_name,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--profile",
        choices=tuple(_PROFILES),
        default=_HISTORICAL_PROFILE.profile_id,
        help="source-owned launch profile; repair profiles are scope-pinned",
    )
    parser.add_argument(
        "--scope-start",
        type=int,
        default=None,
        help="zero-based chart-major scope index (profile default when omitted)",
    )
    parser.add_argument(
        "--scope-limit",
        type=int,
        default=None,
        help="number of contiguous scopes (profile default when omitted)",
    )
    return parser.parse_args()


def _failure_classification(exc: BaseException) -> str:
    """Classify a failed launch without changing its hard-veto semantics."""

    message = f"{type(exc).__name__}: {exc}".lower()
    if any(token in message for token in ("target", "bridge", "signature", "proper")):
        return "target_or_bridge"
    if any(token in message for token in ("chart", "inverse", "logdet", "transport")):
        return "chart_or_numerical"
    if any(token in message for token in ("tuning", "handoff", "candidate", "acceptance", "movement")):
        return "tuning_or_evidence"
    if any(token in message for token in ("gpu", "xla", "allocator", "memory", "wall cap", "resource", "signal", "timeout")):
        return "resource_or_execution"
    if any(token in message for token in ("artifact", "manifest", "output directory", "overwrite")):
        return "artifact_or_provenance"
    return "infrastructure_or_unknown"


def _git_payload() -> Mapping[str, Any]:
    status_output = _git(("git", "status", "--porcelain"))
    return {
        "commit": _git(("git", "rev-parse", "HEAD")),
        "worktree_dirty": bool(status_output),
        "status_sha256": hashlib.sha256(status_output.encode("utf-8")).hexdigest(),
    }


def _write_run_start(
    output_dir: Path, args: argparse.Namespace, profile: _Phase9AProfile
) -> float:
    started_at = time.time()
    _write_json(
        output_dir / "run_start.json",
        {
            "schema": "bayesfilter.ssl_lstm_q20.phase9a_run_start.v1",
            "status": "RUNNING_PHASE9A_SCOPE_PREFLIGHT",
            "started_at_unix": started_at,
            "output_dir": str(output_dir),
            "command": list(sys.argv),
            "python": sys.executable,
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "platform": platform.platform(),
            "profile": profile.payload(),
            "scope_pairs": [
                {"scope_index": index, "chart_index": chart, "beta": beta}
                for index, (chart, beta) in enumerate(_scope_pairs())
            ],
            "selected_scope_indices": list(
                range(profile.scope_start or 0, (profile.scope_start or 0) + (profile.scope_limit or SCOPE_COUNT))
            ),
            "target_signature": EXPECTED_TARGET_SIGNATURE,
            "principal_sqrt_backend": EXPECTED_BACKEND,
            "gpu_environment": {
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", ""),
            },
            "gpu_snapshot_before_tensorflow": _nvidia_snapshot(),
            "git": _git_payload(),
        },
    )
    return started_at


def _write_failure_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    profile: _Phase9AProfile | None,
    exc: BaseException,
) -> None:
    if not output_dir.is_dir() or (output_dir / "failure.json").exists():
        return
    start_path = output_dir / "run_start.json"
    start_payload: Mapping[str, Any] = {}
    if start_path.is_file():
        try:
            loaded = json.loads(start_path.read_text(encoding="utf-8"))
            if isinstance(loaded, Mapping):
                start_payload = loaded
        except (OSError, json.JSONDecodeError):
            start_payload = {}
    started_at = start_payload.get("started_at_unix")
    elapsed = None
    if isinstance(started_at, (int, float)):
        elapsed = max(0.0, time.time() - float(started_at))
    payload = {
            "schema": "bayesfilter.ssl_lstm_q20.phase9a_failure.v2",
            "status": "FAIL_PHASE9A_SCOPE_PREFLIGHT",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failure_classification": _failure_classification(exc),
            "command": list(sys.argv),
            "output_dir": str(output_dir),
            "profile": profile.payload() if profile is not None else None,
            "scope_start": profile.scope_start if profile is not None else None,
            "scope_limit": profile.scope_limit if profile is not None else None,
            "target_signature": EXPECTED_TARGET_SIGNATURE,
            "git": _git_payload(),
            "elapsed_seconds": elapsed,
            "run_start_path": str(start_path) if start_path.is_file() else None,
    }
    # Keep a conventional run manifest as well as the easy-to-find failure
    # alias.  Neither path is ever overwritten, so a partial launch remains
    # auditable even when it fails before TensorFlow initialization.
    _write_json(output_dir / "failure.json", payload)
    if not (output_dir / "run_manifest.json").exists():
        _write_json(output_dir / "run_manifest.json", payload)


def _handle_run_signal(signum: int, _frame: Any) -> None:
    """Turn a wall-time/interrupt signal into the normal durable failure path."""

    context = _ACTIVE_RUN_CONTEXT
    exc = Phase9AError(f"Phase 9A interrupted by signal {int(signum)}")
    if context:
        _write_failure_manifest(
            context["output_dir"],
            context["args"],
            context["profile"],
            exc,
        )
    raise exc


def _run(args: argparse.Namespace) -> int:
    profile = _resolve_profile(
        args.profile, scope_start=args.scope_start, scope_limit=args.scope_limit
    )
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise Phase9AError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    started_at = _write_run_start(output_dir, args, profile)
    started = time.monotonic()
    _ACTIVE_RUN_CONTEXT.clear()
    _ACTIVE_RUN_CONTEXT.update(
        {"output_dir": output_dir, "args": args, "profile": profile}
    )
    signal.signal(signal.SIGTERM, _handle_run_signal)
    signal.signal(signal.SIGINT, _handle_run_signal)
    if not _truthy(os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH")):
        raise Phase9AError("Phase 9A GPU launch requires TF_FORCE_GPU_ALLOW_GROWTH=true before import")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() in {"", "-1"}:
        raise Phase9AError("Phase 9A requires one explicitly visible GPU")
    prerequisites = _check_prerequisites(profile)
    route_scan = _route_scan()
    if not route_scan["passed"]:
        raise Phase9AError(f"active route scan failed: {route_scan}")

    import tensorflow as tf

    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    tf.config.set_soft_device_placement(False)
    logical_gpus = tuple(tf.config.list_logical_devices("GPU"))
    if len(logical_gpus) != 1:
        raise Phase9AError(f"Phase 9A requires exactly one visible logical GPU, got {len(logical_gpus)}")
    device_name = str(logical_gpus[0].name)

    from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge

    bridge = make_q20_tempered_bridge(
        20, jit_compile=True, principal_sqrt_backend=EXPECTED_BACKEND
    )
    if str(bridge.target_signature) != EXPECTED_TARGET_SIGNATURE:
        raise Phase9AError("q=20 target signature changed")
    for beta in BETAS:
        center = tf.convert_to_tensor(bridge.prior_center, tf.float64)
        values, scores, status = bridge.value_score_status(
            tf.stack((center, center + tf.constant([0.1, -0.1, 0.1, -0.1], tf.float64))),
            tf.constant(beta, tf.float64),
        )
        if not _finite_bool(tf, values) or not _finite_bool(tf, scores) or not bool(tf.reduce_all(status["bridge_valid"]).numpy()):
            raise Phase9AError(f"bridge status preflight failed at beta={beta}")

    charts_by_beta: dict[float, list[Any]] = {beta: [] for beta in BETAS}
    chart_records = []
    for chart_index, component_id in enumerate(COMPONENT_IDS):
        chart_started = time.monotonic()
        level_charts, checkpoints, receipts = _build_fresh_chart(
            tf, bridge, chart_index, component_id, profile, output_dir
        )
        for beta in BETAS:
            charts_by_beta[beta].append(level_charts[beta])
        chart_records.append(
            {
                "component_id": component_id,
                "checkpoints": checkpoints,
                "preflight": receipts,
                "build_elapsed_seconds": time.monotonic() - chart_started,
            }
        )
    reliability = _reliability(
        tf, bridge, charts_by_beta[1.0], beta=1.0, profile=profile
    )

    scope_records = []
    live_tuned_rows: list[Mapping[str, Any]] = []
    scope_start = int(profile.scope_start or 0)
    scope_limit = int(profile.scope_limit or SCOPE_COUNT)
    selected_scope_indices = tuple(range(scope_start, scope_start + scope_limit))
    for scope_index in selected_scope_indices:
        chart_index, beta = _scope_pairs()[scope_index]
        chart = charts_by_beta[beta][chart_index]
        scope_dir = output_dir / f"chart-{chart_index}" / f"beta-{beta:g}"
        scope_started_at = time.time()
        _write_json(
            scope_dir / "scope_start.json",
            {
                "schema": "bayesfilter.ssl_lstm_q20.phase9a.scope_start.v1",
                "status": "RUNNING_SCOPE_TUNING",
                "profile_id": profile.profile_id,
                "scope_index": scope_index,
                "chart_index": chart_index,
                "beta": beta,
                "target_signature": EXPECTED_TARGET_SIGNATURE,
                "tuning_policy": "measured_joint_grid_v1",
                "declared_pair_count": len(profile.step_size_candidates)
                * len(profile.leapfrog_grid),
                "started_at_unix": scope_started_at,
                "seed_root": list(profile.tuning_roots[scope_index]),
            },
        )
        row = _tune_scope(
            tf,
            bridge,
            chart,
            chart_index=chart_index,
            beta=beta,
            output_dir=scope_dir,
            profile=profile,
            scope_index=scope_index,
        )
        live_tuned_rows.append(row)
        public_row = {key: value for key, value in row.items() if not key.startswith("_")}
        scope_records.append(public_row)
        _write_json(
            scope_dir / "scope_complete.json",
            {
                "schema": "bayesfilter.ssl_lstm_q20.phase9a.scope_complete.v1",
                "status": "PASS_SCOPE_TUNING",
                "profile_id": profile.profile_id,
                "scope_index": scope_index,
                "chart_index": chart_index,
                "beta": beta,
                "tuning_policy": "measured_joint_grid_v1",
                "completed_at_unix": time.time(),
                "elapsed_seconds": max(0.0, time.time() - scope_started_at),
                "tuning_artifact": public_row.get("tuning_artifact"),
                "handoff_hash": public_row.get("handoff_hash"),
                "step_size": public_row.get("step_size"),
                "num_leapfrog_steps": public_row.get("num_leapfrog_steps"),
            },
        )

    # The first implementation pass intentionally localizes scope binding.  A
    # complete run stores live handoffs alongside the public records and then
    # exercises the shared transition program.
    transition = None
    if scope_start == 0 and scope_limit == SCOPE_COUNT:
        transition = _run_transition(
            tf, bridge, charts_by_beta, live_tuned_rows, device_name, profile
        )

    elapsed = time.monotonic() - started
    if elapsed > profile.material_cap_seconds:
        raise Phase9AError(
            "Phase 9A material wall cap exceeded: "
            f"{elapsed:.3f}s > {profile.material_cap_seconds:.3f}s"
        )
    allocator = _memory_info(tf, device_name)
    peak_allocator = allocator.get("peak") if isinstance(allocator, Mapping) else None
    if not isinstance(peak_allocator, int):
        raise Phase9AError(
            "allocator peak telemetry is unavailable; cannot certify the 4-GiB preflight cap"
        )
    if peak_allocator > ALLOCATOR_CAP_BYTES:
        raise Phase9AError(
            f"allocator peak cap exceeded: {peak_allocator} > {ALLOCATOR_CAP_BYTES}"
        )
    status_output = _git(("git", "status", "--porcelain"))
    manifest = {
        "schema": SCHEMA,
        "status": "PASS_PHASE9A_SCOPE_PREFLIGHT" if transition is not None else "PASS_PHASE9A_SCOPE_PREFLIGHT_PARTIAL",
        "role": "fresh_chart_and_scope_tuning_localization_only",
        "profile_id": profile.profile_id,
        "profile": profile.payload(),
        "scope_start": scope_start,
        "scope_limit": scope_limit,
        "scope_count": SCOPE_COUNT,
        "selected_scope_indices": list(selected_scope_indices),
        "scope_pairs": [
            {"scope_index": index, "chart_index": chart, "beta": beta}
            for index, (chart, beta) in enumerate(_scope_pairs())
        ],
        "target_signature": EXPECTED_TARGET_SIGNATURE,
        "run_start_unix": started_at,
        "git_commit": _git(("git", "rev-parse", "HEAD")),
        "git_worktree_dirty": bool(status_output),
        "git_status_sha256": hashlib.sha256(status_output.encode("utf-8")).hexdigest(),
        "command": list(sys.argv),
        "python": sys.executable,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "platform": platform.platform(),
        "tensorflow": str(tf.__version__),
        "bridge_signature": str(bridge.signature),
        "properness_receipt": bridge.properness_receipt.payload(),
        "principal_sqrt_backend": EXPECTED_BACKEND,
        "protocol": {"betas": list(BETAS), "component_ids": list(COMPONENT_IDS), "gamma": [0.5, 0.5]},
        "prerequisites": prerequisites,
        "route_scan": route_scan,
        "memory_policy": memory_policy,
        "logical_gpus": [str(item.name) for item in logical_gpus],
        "tf32_execution_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        "gpu_environment": {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""), "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "")},
        "gpu_snapshot_before": _nvidia_snapshot(),
        "gpu_snapshot_after": _nvidia_snapshot(),
        "allocator": allocator,
        "allocator_cap_bytes": ALLOCATOR_CAP_BYTES,
        "budget": {
            "material_cap_seconds": profile.material_cap_seconds,
            "wall_time_seconds": elapsed,
        },
        "seed_ledger": {
            "initialization_roots": [list(item) for item in profile.initialization_roots],
            "preflight_roots": [list(item) for item in profile.preflight_roots],
            "training_roots": [list(item) for item in profile.training_roots],
            "tuning_roots": [list(item) for item in profile.tuning_roots],
            "transition_root": list(profile.transition_root),
            "reliability_root": list(profile.reliability_root),
        },
        "result_note_path": (
            "docs/plans/bayesfilter-ssl-lstm-q20-phase9a-chart1-beta0-program-repair-result-2026-09-01.md"
            if profile.plan_path == REPAIR_PLAN
            else (
                "docs/plans/bayesfilter-ssl-lstm-q20-phase9a-full-replay-performance-result-2026-09-02.md"
                if profile.plan_path == FULL_REPLAY_PLAN
                else "docs/plans/bayesfilter-ssl-lstm-q20-phase9a-fresh-tuning-preflight-result-2026-08-31.md"
            )
        ),
        "charts": chart_records,
        "reliability": reliability,
        "scope_records": scope_records,
        "transition": transition,
        "wall_time_seconds": elapsed,
        "nonclaims": ["no whitening", "no mode discovery", "no posterior", "no convergence", "no HMC readiness", "no ranking", "no scaling"],
    }
    manifest["manifest_hash"] = _stable_hash(manifest)
    _write_json(output_dir / "run_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output_dir": str(output_dir),
                "profile_id": profile.profile_id,
                "scope_start": scope_start,
                "scope_limit": scope_limit,
                "wall_time_seconds": elapsed,
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    args = _parse_args()
    profile = None
    try:
        profile = _resolve_profile(
            args.profile, scope_start=args.scope_start, scope_limit=args.scope_limit
        )
    except Exception:
        # _run emits the authoritative validation error; this fallback keeps
        # invalid CLI invocations from pretending that a run started.
        profile = None
    try:
        return _run(args)
    except Exception as exc:  # noqa: BLE001 - preserve a structured failure.
        if isinstance(args.output_dir, Path):
            path = args.output_dir.expanduser().resolve()
            _write_failure_manifest(path, args, profile, exc)
        payload = {
            "status": "FAIL_PHASE9A_SCOPE_PREFLIGHT",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failure_classification": _failure_classification(exc),
        }
        if profile is not None:
            payload["profile_id"] = profile.profile_id
            payload["scope_start"] = profile.scope_start
            payload["scope_limit"] = profile.scope_limit
        print(json.dumps(payload, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
