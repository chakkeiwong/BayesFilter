"""CPU-only contract checks for the bounded Phase 9A repair launcher."""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "docs/benchmarks/run_ssl_lstm_q20_phase9a_fresh_tuning_preflight_2026_08_31.py"
)


def _runner_module():
    spec = importlib.util.spec_from_file_location("phase9a_repair_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("unable to load Phase 9A runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_repair_profile_is_pinned_and_measures_declared_joint_grid() -> None:
    module = _runner_module()
    profile = module._resolve_profile("chart1_beta0_repair_v1")

    assert (profile.scope_start, profile.scope_limit) == (3, 1)
    assert module._scope_pairs()[3] == (1, 0.0)
    assert len(profile.step_size_candidates) == 8
    assert len(profile.leapfrog_grid) == 4
    assert len(profile.step_size_candidates) * len(profile.leapfrog_grid) == 32
    assert profile.initial_state_bank[1][0] == pytest.approx(0.25)
    assert profile.plan_path == module.REPAIR_PLAN
    assert profile.initialization_roots[0][0] == 20260901


def test_repair_profile_rejects_scope_widening() -> None:
    module = _runner_module()
    with pytest.raises(module.Phase9AError, match="pinned"):
        module._resolve_profile(
            "chart1_beta0_repair_v1", scope_start=0, scope_limit=6
        )
    with pytest.raises(module.Phase9AError, match="scope interval"):
        module._resolve_profile(
            "phase9a_measured_preflight_v1_historical", scope_start=5, scope_limit=2
        )


def test_bounded_retry_profile_has_fresh_namespace_and_eight_measured_pairs() -> None:
    module = _runner_module()
    profile = module._resolve_profile("chart1_beta0_repair_v2_bounded")
    assert (profile.scope_start, profile.scope_limit) == (3, 1)
    assert len(profile.step_size_candidates) * len(profile.leapfrog_grid) == 8
    assert profile.selection_num_results == 16
    assert profile.verification_num_results == 16
    assert profile.tuning_roots[0][1] == 75301
    assert profile.tuning_roots != module._CHART1_BETA0_REPAIR_PROFILE.tuning_roots


def test_minimal_retry_profile_is_the_smallest_joint_grid_with_fresh_seeds() -> None:
    module = _runner_module()
    profile = module._resolve_profile("chart1_beta0_repair_v3_minimal")
    assert len(profile.step_size_candidates) == 2
    assert len(profile.leapfrog_grid) == 2
    assert len(profile.step_size_candidates) * len(profile.leapfrog_grid) == 4
    assert profile.screen_num_results == 1
    assert profile.selection_num_results == 4
    assert profile.verification_num_results == 4
    assert profile.tuning_roots[0][1] == 76301


def test_final_retry_profile_preserves_v3_contract_with_a_new_seed_namespace() -> None:
    module = _runner_module()
    v3 = module._resolve_profile("chart1_beta0_repair_v3_minimal")
    v4 = module._resolve_profile("chart1_beta0_repair_v4_fresh")
    assert (v4.scope_start, v4.scope_limit) == (3, 1)
    assert v4.step_size_candidates == v3.step_size_candidates
    assert v4.leapfrog_grid == v3.leapfrog_grid
    assert v4.screen_num_results == v3.screen_num_results
    assert v4.selection_num_results == v3.selection_num_results
    assert v4.verification_num_results == v3.verification_num_results
    assert v4.initialization_roots != v3.initialization_roots
    assert v4.preflight_roots != v3.preflight_roots
    assert v4.training_roots != v3.training_roots
    assert v4.tuning_roots != v3.tuning_roots
    assert v4.tuning_roots[0][1] == 77301


def test_full_replay_canary_and_full_profiles_are_disjoint_and_cap_bound() -> None:
    module = _runner_module()
    canary = module._resolve_profile("phase9a_full_replay_canary_v1")
    full = module._resolve_profile("phase9a_full_replay_v1")
    assert (canary.scope_start, canary.scope_limit) == (3, 1)
    assert (full.scope_start, full.scope_limit) == (0, module.SCOPE_COUNT)
    assert canary.material_cap_seconds == pytest.approx(1800.0)
    assert full.material_cap_seconds == pytest.approx(7800.0)
    assert canary.plan_path == module.FULL_REPLAY_PLAN
    assert full.plan_path == module.FULL_REPLAY_PLAN
    assert canary.initialization_roots != full.initialization_roots
    assert canary.preflight_roots != full.preflight_roots
    assert canary.training_roots != full.training_roots
    assert canary.tuning_roots != full.tuning_roots
    assert canary.transition_root != full.transition_root
    assert canary.reliability_root != full.reliability_root
    with pytest.raises(module.Phase9AError, match="canary"):
        module._resolve_profile(
            "phase9a_full_replay_canary_v1", scope_start=0, scope_limit=6
        )


def test_gpu10000_continuation_profiles_are_fresh_and_cap_bound() -> None:
    module = _runner_module()
    canary = module._resolve_profile("phase9a_full_replay_canary_gpu10000_v1")
    full = module._resolve_profile("phase9a_full_replay_gpu10000_v1")
    assert (canary.scope_start, canary.scope_limit) == (3, 1)
    assert (full.scope_start, full.scope_limit) == (0, module.SCOPE_COUNT)
    assert canary.material_cap_seconds == pytest.approx(1800.0)
    assert full.material_cap_seconds == pytest.approx(10000.0)
    assert canary.plan_path == module.GPU10000_REPLAY_PLAN
    assert full.plan_path == module.GPU10000_REPLAY_PLAN
    assert canary.initialization_roots[0][0] == 20260904
    assert full.initialization_roots[0][0] == 20260904
    assert canary.initialization_roots != full.initialization_roots
    assert canary.tuning_roots != full.tuning_roots
    assert full.payload()["material_cap_seconds"] == pytest.approx(10000.0)
    with pytest.raises(module.Phase9AError, match="canary"):
        module._resolve_profile(
            "phase9a_full_replay_canary_gpu10000_v1", scope_start=0, scope_limit=6
        )


def test_factor_profiles_are_backend_bound_fresh_disjoint_and_scope_pinned() -> None:
    module = _runner_module()
    canary = module._resolve_profile("phase9a_factor_tuning_canary_v1")
    full = module._resolve_profile("phase9a_factor_tuning_full_v1")
    r2 = module._resolve_profile("phase9a_factor_tuning_full_r2_v1")
    source_sync = module._resolve_profile(
        "phase9a_factor_tuning_full_source_sync_v1"
    )

    assert canary.principal_sqrt_backend == module.FACTOR_BACKEND
    assert full.principal_sqrt_backend == module.FACTOR_BACKEND
    assert canary.payload()["principal_sqrt_backend"] == module.FACTOR_BACKEND
    assert canary.plan_path == module.FACTOR_TUNING_PLAN
    assert full.plan_path == module.FACTOR_TUNING_PLAN
    assert (canary.scope_start, canary.scope_limit) == (3, 1)
    assert (full.scope_start, full.scope_limit) == (0, module.SCOPE_COUNT)
    assert canary.material_cap_seconds == pytest.approx(1800.0)
    assert full.material_cap_seconds == pytest.approx(10000.0)
    assert r2.material_cap_seconds == pytest.approx(8400.0)
    assert r2.step_size_candidates == (0.055, 0.060, 0.070, 0.075)
    assert r2.selection_num_results == 8
    assert r2.selection_num_burnin_steps == 2
    assert r2.verification_num_results == 8
    assert r2.verification_num_burnin_steps == 2
    assert r2.principal_sqrt_backend == module.FACTOR_BACKEND
    assert source_sync.principal_sqrt_backend == module.FACTOR_BACKEND
    assert source_sync.material_cap_seconds == pytest.approx(4000.0)
    assert source_sync.step_size_candidates == r2.step_size_candidates
    assert source_sync.initialization_roots != r2.initialization_roots
    assert source_sync.tuning_roots != r2.tuning_roots
    assert r2.initialization_roots != full.initialization_roots
    assert r2.tuning_roots != full.tuning_roots
    assert canary.initialization_roots != full.initialization_roots
    assert canary.preflight_roots != full.preflight_roots
    assert canary.training_roots != full.training_roots
    assert canary.tuning_roots != full.tuning_roots
    assert canary.transition_root != full.transition_root
    assert canary.reliability_root != full.reliability_root
    assert module._scope("phase9a-chart-1", 0.0, 1, canary)[
        "bridge_backend"
    ] == module.FACTOR_BACKEND
    assert module._checkpoint_scope("phase9a-chart-1", 0.0, 1, canary)[
        "principal_sqrt_backend"
    ] == module.FACTOR_BACKEND
    with pytest.raises(module.Phase9AError, match="canary"):
        module._resolve_profile(
            canary.profile_id, scope_start=0, scope_limit=module.SCOPE_COUNT
        )
    with pytest.raises(module.Phase9AError, match="all six"):
        module._resolve_profile(full.profile_id, scope_start=0, scope_limit=1)
    with pytest.raises(module.Phase9AError, match="all six"):
        module._resolve_profile(r2.profile_id, scope_start=3, scope_limit=1)


def test_factor_profile_rejects_source_backend_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _runner_module()
    profile_id = module._FACTOR_TUNING_PROFILE.profile_id
    monkeypatch.setitem(
        module._PROFILES,
        profile_id,
        replace(
            module._FACTOR_TUNING_PROFILE,
            principal_sqrt_backend=module.STRICT_BACKEND,
        ),
    )
    with pytest.raises(module.Phase9AError, match="ownership mismatch"):
        module._resolve_profile(profile_id)


def test_all_nonfactor_profiles_remain_strict() -> None:
    module = _runner_module()
    factor_ids = {
        module._FACTOR_TUNING_CANARY_PROFILE.profile_id,
        module._FACTOR_TUNING_PROFILE.profile_id,
        module._FACTOR_TUNING_R2_PROFILE.profile_id,
        module._FACTOR_TUNING_SOURCE_SYNC_PROFILE.profile_id,
    }
    for profile_id in module._PROFILES:
        profile = module._resolve_profile(profile_id)
        if profile_id not in factor_ids:
            assert profile.principal_sqrt_backend == module.STRICT_BACKEND


def test_repair_profiles_bind_repair_result_note_and_manifest_profile_id() -> None:
    module = _runner_module()
    for profile_id in (
        "chart1_beta0_repair_v1",
        "chart1_beta0_repair_v2_bounded",
        "chart1_beta0_repair_v3_minimal",
        "chart1_beta0_repair_v4_fresh",
    ):
        profile = module._resolve_profile(profile_id)
        assert profile.plan_path == module.REPAIR_PLAN
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert '"profile_id": profile.profile_id' in source
    assert "profile.plan_path == REPAIR_PLAN" in source


def test_profile_payload_exposes_material_cap_and_full_replay_result_note() -> None:
    module = _runner_module()
    profile = module._resolve_profile("phase9a_full_replay_v1")
    payload = profile.payload()
    assert payload["material_cap_seconds"] == pytest.approx(7800.0)
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "phase9a-full-replay-performance-result-2026-09-02.md" in source
    assert "runner_created_on_call" in source


def test_historical_profile_can_reproduce_full_scope_order() -> None:
    module = _runner_module()
    profile = module._resolve_profile("phase9a_measured_preflight_v1_historical")
    assert (profile.scope_start, profile.scope_limit) == (0, module.SCOPE_COUNT)
    assert module._scope_pairs() == (
        (0, 0.0),
        (0, 0.5),
        (0, 1.0),
        (1, 0.0),
        (1, 0.5),
        (1, 1.0),
    )


def test_failure_manifest_is_durable_and_classified(tmp_path: Path) -> None:
    module = _runner_module()
    output_dir = tmp_path / "attempt"
    output_dir.mkdir()
    profile = module._resolve_profile("chart1_beta0_repair_v1")
    module._write_json(
        output_dir / "run_start.json",
        {
            "schema": "test",
            "started_at_unix": 1.0,
            "profile": profile.payload(),
        },
    )
    args = argparse.Namespace(
        output_dir=output_dir,
        profile=profile.profile_id,
        scope_start=3,
        scope_limit=1,
    )
    module._write_failure_manifest(
        output_dir, args, profile, RuntimeError("scope tuner movement failed")
    )
    payload = json.loads((output_dir / "failure.json").read_text(encoding="utf-8"))
    assert payload["schema"].endswith("failure.v2")
    assert payload["failure_classification"] == "tuning_or_evidence"
    assert payload["scope_start"] == 3
    assert payload["scope_limit"] == 1
    assert (output_dir / "run_manifest.json").is_file()
    assert json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))["status"].startswith("FAIL_")
    module._write_failure_manifest(output_dir, args, profile, RuntimeError("second failure"))
    preserved = json.loads((output_dir / "failure.json").read_text(encoding="utf-8"))
    assert preserved["error"] == "scope tuner movement failed"


def test_artifact_serializer_preserves_nonfinite_diagnostics_explicitly() -> None:
    module = _runner_module()
    payload = module._artifact_json_ready(
        {"nan": float("nan"), "positive": float("inf"), "negative": -float("inf")}
    )
    assert payload == {
        "nan": {"__nonfinite__": "nan"},
        "positive": {"__nonfinite__": "inf"},
        "negative": {"__nonfinite__": "-inf"},
    }
    encoded = json.dumps(payload, allow_nan=False)
    assert "NaN" not in encoded


def test_manifest_hash_normalizes_nonfinite_diagnostics() -> None:
    module = _runner_module()
    normalized = module._json_ready(
        {"nan": float("nan"), "positive": float("inf"), "negative": -float("inf")}
    )
    assert normalized == {
        "nan": {"__nonfinite__": "nan"},
        "positive": {"__nonfinite__": "inf"},
        "negative": {"__nonfinite__": "-inf"},
    }
    digest = module._stable_hash({"diagnostics": {"value": float("nan")}})
    assert isinstance(digest, str) and len(digest) == 64


def test_repair_wrapper_binds_gpu_growth_and_exact_scope() -> None:
    wrapper = (
        ROOT / "scripts/run_ssl_lstm_q20_phase9a_chart1_beta0_repair_gpu.sh"
    ).read_text(encoding="utf-8")
    assert "TF_FORCE_GPU_ALLOW_GROWTH=true" in wrapper
    assert "--profile chart1_beta0_repair_v4_fresh" in wrapper
    assert "--scope-start 3" in wrapper
    assert "--scope-limit 1" in wrapper
    assert "--output-dir" in wrapper
    assert "--kill-after=120s 1800s" in wrapper
    assert 'cp "${output_dir}/failure.json" "${output_dir}/run_manifest.json"' in wrapper


def test_full_replay_wrapper_uses_exact_source_owned_root() -> None:
    wrapper = (
        ROOT / "scripts/run_ssl_lstm_q20_phase9a_full_replay_gpu.sh"
    ).read_text(encoding="utf-8")
    assert "phase9a-full-replay}" not in wrapper
    assert (
        'output_root="${repo_root}/docs/plans/artifacts/'
        'ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-09-02/'
        'phase9a-full-replay"'
        in wrapper
    )
    assert "TF_FORCE_GPU_ALLOW_GROWTH=true" in wrapper
    assert "phase9a_full_replay_canary_v1" in wrapper
    assert "phase9a_full_replay_v1" in wrapper


def test_gpu10000_full_replay_wrapper_binds_new_cap_and_root() -> None:
    wrapper = (
        ROOT / "scripts/run_ssl_lstm_q20_phase9a_full_replay_gpu10000.sh"
    ).read_text(encoding="utf-8")
    assert "phase9a_full_replay_canary_gpu10000_v1" in wrapper
    assert "phase9a_full_replay_gpu10000_v1" in wrapper
    assert "phase9a-full-replay-gpu10000" in wrapper
    assert "--kill-after=120s" in wrapper
    assert '"${material_cap_seconds}s"' in wrapper
    assert "material_cap_seconds=10000" in wrapper
    assert "TF_FORCE_GPU_ALLOW_GROWTH=true" in wrapper


def test_factor_tuning_wrapper_binds_backend_profiles_gpu_and_caps() -> None:
    wrapper = (
        ROOT / "scripts/run_ssl_lstm_q20_phase9a_factor_tuning_admission.sh"
    ).read_text(encoding="utf-8")
    assert "phase9a_factor_tuning_canary_v1" in wrapper
    assert "phase9a_factor_tuning_full_v1" in wrapper
    assert "phase9a_factor_tuning_full_r2_v1" in wrapper
    assert "phase9a_factor_tuning_full_source_sync_v1" in wrapper
    assert "ssl-lstm-q20-factor-route-fresh-tuning-2026-09-04" in wrapper
    assert "tensorflow_eigh_strict_factor_cached" in wrapper
    assert "CUDA_VISIBLE_DEVICES=0" in wrapper
    assert "TF_FORCE_GPU_ALLOW_GROWTH=true" in wrapper
    assert "material_cap_seconds=1800" in wrapper
    assert "material_cap_seconds=10000" in wrapper
    assert "material_cap_seconds=8400" in wrapper
    assert "expected canary, full, r2, or source-sync" in wrapper
    assert "material_cap_seconds=4000" in wrapper
    assert "--kill-after=120s" in wrapper


def test_fail_closed_environment_still_leaves_start_and_failure_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _runner_module()
    output_dir = tmp_path / "invalid-launch"
    args = argparse.Namespace(
        output_dir=output_dir,
        profile="chart1_beta0_repair_v2_bounded",
        scope_start=3,
        scope_limit=1,
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    monkeypatch.setattr(module, "_nvidia_snapshot", lambda: {"rows": []})
    monkeypatch.setattr(module, "_parse_args", lambda: args)
    assert module.main() == 2
    assert (output_dir / "run_start.json").is_file()
    failure = json.loads((output_dir / "failure.json").read_text(encoding="utf-8"))
    assert failure["failure_classification"] == "resource_or_execution"
    assert (output_dir / "run_manifest.json").is_file()
