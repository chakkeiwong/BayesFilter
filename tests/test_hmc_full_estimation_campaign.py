from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from bayesfilter.runtime import stable_config_hash
from bayesfilter.testing import deterministic_lgssm_hmc_phase7_tf as phase7
from docs.benchmarks import (
    run_multidim_lgssm_serious_hmc_tuning_2026_07_09 as driver,
)
from scripts import build_hmc_full_estimation_phase7_config as builder
from scripts import run_hmc_full_estimation_campaign as launcher


def _sha(char: str = "1") -> str:
    return "sha256:" + char * 64


def _v3_payload(root: str = "docs/benchmarks/artifacts/fresh") -> dict:
    sources = {
        name: {
            "path": f"{root}/{name}.json",
            "schema": phase7._PHASE7_V3_SOURCE_SCHEMAS[name],
            "file_sha256": str(index + 1) * 64,
            "byte_count": index + 1,
            "artifact_hash": (
                _sha(str(index + 1))
                if name
                not in {"source_tuning_config", "source_contract"}
                else None
            ),
        }
        for index, name in enumerate(phase7._PHASE7_V3_SOURCE_KEYS)
    }
    sources["source_tuning_config"]["path"] = (
        "docs/benchmarks/configs/fresh.json"
    )
    return {
        "schema": phase7.PHASE7_CONFIG_SCHEMA_V3,
        "config_id": "fresh",
        "plan_path": "docs/plans/fresh.md",
        "source_tuning_config_path": "docs/benchmarks/configs/fresh.json",
        "source_tuning_config_hash": _sha("a"),
        "artifact_root": root,
        "execution": {
            "worker_count": 2,
            "chains_per_worker": 2,
            "root_seed": [20260713, 701],
            "cuda_visible_devices": "-1",
            "jit_compile": True,
            "use_xla": True,
            "chain_execution_mode": "tf_function",
            "compile_workers_sequentially": True,
            "wall_time_cap_seconds": 28800,
            "thread_environment": {
                "TF_NUM_INTRAOP_THREADS": "8",
                "TF_NUM_INTEROP_THREADS": "1",
                "OMP_NUM_THREADS": "8",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
        },
        "burnin": {
            "initial_results_per_chain": 2000,
            "extension_results_per_chain": 1000,
            "check_window_results_per_chain": 1000,
            "max_results_per_chain": 16000,
        },
        "retained": {
            "initial_results_per_chain": 4000,
            "extension_results_per_chain": 2000,
            "check_interval_results_per_chain": 2000,
            "max_results_per_chain": 40000,
        },
        "diagnostics": {
            "rhat_max": 1.01,
            "bulk_ess_min": 1000.0,
            "tail_ess_min": 400.0,
            "all_parameters_required": True,
            "coordinate_system": "raw_lgssm_parameters_after_two_mass_transforms",
            "rhat_definition": phase7._MODERN_RHAT_DEFINITION,
        },
        "artifacts": {
            "public_result": f"{root}/phase7_campaign/result.json",
            "public_progress": f"{root}/phase7_campaign/progress.json",
            "private_replay": sources["private_replay"]["path"],
            "private_retained_samples": (
                f"{root}/phase7_campaign/private/retained_samples.npz"
            ),
        },
        "governed_source_references": sources,
        "expected_identities": {
            name: _sha(chr(ord("a") + index))
            for index, name in enumerate(phase7._PHASE7_V3_IDENTITY_KEYS)
        },
        "fresh_run_policy": {
            "historical_identity_inputs_allowed": False,
            "migration_certificates_required": False,
            "approval_manifests_required": False,
            "no_overwrite": True,
            "tuning_root_seed": [20260709, 501],
            "serious_root_seed": [20260713, 701],
        },
        "nonclaims": ["test only"],
    }


def test_v3_config_accepts_sorted_json_and_rejects_historical_or_seed_drift(
    tmp_path: Path,
) -> None:
    payload = _v3_payload()
    path = tmp_path / "phase7.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    loaded = phase7.DeterministicLGSSMPhase7Config.load(path)

    assert loaded.payload["schema"] == phase7.PHASE7_CONFIG_SCHEMA_V3
    assert loaded.runtime_authority is True
    assert "baseline_adoption" not in loaded.payload
    assert "historical_legacy_hashes" not in loaded.payload
    assert loaded.payload["fresh_run_policy"][
        "migration_certificates_required"
    ] is False
    assert all(
        "certificate" not in name
        for name in loaded.payload["governed_source_references"]
    )

    mutated = copy.deepcopy(payload)
    mutated["execution"]["root_seed"] = [20260711, 701]
    path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(ValueError, match="root seed"):
        phase7.DeterministicLGSSMPhase7Config.load(path)


def test_v3_worker_request_uses_only_declared_fresh_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _v3_payload()
    config = phase7.DeterministicLGSSMPhase7Config(
        payload=payload,
        path=Path("/tmp/fresh-phase7.json"),
    )
    paths = {
        name: Path("/tmp/fresh") / f"{name}.json"
        for name in phase7._PHASE7_V3_SOURCE_KEYS
    }
    monkeypatch.setattr(phase7, "phase7_governed_source_paths", lambda _cfg: paths)

    request = phase7._worker_request(
        config,
        worker_index=0,
        action="initialize",
        count=0,
        seed=(1, 2),
        state=None,
        worker_env={},
        smoke=True,
        target_scope="fresh_target",
    )

    assert request["v3_config_payload"] == payload
    assert request["expected_transition_identity_hash"] == payload[
        "expected_identities"
    ]["transition_identity_hash"]
    assert request["historical_v1_config_path"] is None
    assert "v2_config_payload" not in request
    assert "adopted_identities" not in request
    assert request["secure_source_verification"] is False


def test_v3_preflight_exposes_private_replay_hash_for_retained_archive() -> None:
    config_path = Path(__file__).resolve().parents[1] / (
        "docs/benchmarks/configs/"
        "multidim_lgssm_full_estimation_phase7_2026_07_13.json"
    )
    config = phase7.DeterministicLGSSMPhase7Config.load(config_path)
    replay_path = phase7.phase7_governed_source_paths(config)["private_replay"]
    replay = json.loads(replay_path.read_text(encoding="utf-8"))

    preflight = phase7.validate_phase7_v3_inputs(
        config, require_fresh_outputs=False
    )

    assert preflight["private_replay_artifact_hash"] == replay["artifact_hash"]


def test_builder_emits_direct_v3_payload_and_refuses_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder.tuning_driver, "ROOT", tmp_path)
    artifact_root = tmp_path / "artifacts" / "fresh"
    artifact_root.mkdir(parents=True)
    contract = tmp_path / "contract.json"
    _write_json(contract, {"schema": phase7._PHASE7_V3_SOURCE_SCHEMAS["source_contract"]})
    config_path = tmp_path / "configs" / "fresh.json"
    payload = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/benchmarks/configs/"
            "multidim_lgssm_full_estimation_rerun_2026_07_13.json"
        ).read_text(encoding="utf-8")
    )
    payload["source_contract"]["path"] = "contract.json"
    for key, filename in {
        "root": "artifacts/fresh",
        "fixture": "artifacts/fresh/fixture.json",
        "xla_compile": "artifacts/fresh/xla.json",
        "geometry": "artifacts/fresh/geometry.json",
        "mass": "artifacts/fresh/mass.json",
        "kernel_tuning": "artifacts/fresh/kernel.json",
        "burnin_sampling": "artifacts/fresh/phase7_campaign/result.json",
        "final_result": "artifacts/fresh/recovery.json",
        "log": "artifacts/fresh/run.log",
    }.items():
        payload["artifact_paths"][key] = filename
    payload["truth_and_data"]["artifact_path"] = payload["artifact_paths"][
        "fixture"
    ]
    _write_json(config_path, payload)
    artifact_payloads = {
        "fixture.json": {"schema": phase7._PHASE7_V3_SOURCE_SCHEMAS["fixture"], "artifact_hash": _sha("1")},
        "xla.json": {"schema": phase7._PHASE7_V3_SOURCE_SCHEMAS["xla_compile"], "artifact_hash": _sha("2")},
        "geometry.json": {"schema": phase7._PHASE7_V3_SOURCE_SCHEMAS["geometry"], "artifact_hash": _sha("3")},
        "mass.json": {"schema": phase7._PHASE7_V3_SOURCE_SCHEMAS["mass"], "artifact_hash": _sha("4")},
        "kernel.json": {
            "schema": phase7._PHASE7_V3_SOURCE_SCHEMAS["kernel"],
            "artifact_hash": _sha("5"),
            "passed": True,
            "final_status": "passed",
        },
    }
    for filename, value in artifact_payloads.items():
        _write_json(artifact_root / filename, value)
    replay = {
        "schema": phase7._PHASE7_V3_SOURCE_SCHEMAS["private_replay"],
        "artifact_hash": _sha("6"),
        "tuning_payload": {
            "tune_verify_repair_loop": {"passed": True}
        },
    }
    replay_path = artifact_root / "private_diagnostics" / "kernel_tuning_replay.json"
    _write_json(replay_path, replay)
    live = SimpleNamespace(
        transition=SimpleNamespace(identity_hash=_sha("a")),
        serious_execution=SimpleNamespace(identity_hash=_sha("b")),
        smoke_execution=SimpleNamespace(identity_hash=_sha("c")),
        provenance=SimpleNamespace(identity_hash=_sha("d")),
    )
    monkeypatch.setattr(builder, "build_phase7_live_identity_bundle", lambda _cfg: live)
    output = tmp_path / "configs" / "phase7.json"

    observed = builder.build_phase7_v3_payload(
        tuning_config_path=config_path,
        output_path=output,
    )

    assert observed["schema"] == phase7.PHASE7_CONFIG_SCHEMA_V3
    assert observed["expected_identities"]["transition_identity_hash"] == _sha("a")
    assert observed["fresh_run_policy"]["historical_identity_inputs_allowed"] is False
    assert "baseline_adoption" not in observed
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        builder.build_phase7_v3_payload(
            tuning_config_path=config_path,
            output_path=output,
        )


@pytest.mark.parametrize("relative_config_path", (False, True))
def test_final_recovery_verifies_archive_and_recomputed_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_config_path: bool,
) -> None:
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    parameter_names = tuple(f"p{index}" for index in range(18))
    config_payload = {
        "model": {"parameter_names": parameter_names},
        "prior": {
            "scale_by_block": {
                "diagonal_raw": 0.5,
                "lower_raw": 0.6,
                "log_process_std": 0.35,
                "log_observation_std": 0.35,
            }
        },
        "final_recovery_gate": {
            "r_hat_threshold": 1.01,
            "bulk_ess_min_per_parameter": 1000,
            "tail_ess_min_per_parameter": 400,
            "truth_distance_max_abs_z": 3.0,
        },
        "artifact_paths": {
            "root": "artifacts",
            "fixture": "artifacts/fixture.json",
            "final_result": "artifacts/recovery.json",
        },
    }
    config_path = tmp_path / "config.json"
    _write_json(config_path, config_payload)
    stored_config_path = (
        Path(os.path.relpath(config_path, Path.cwd()))
        if relative_config_path
        else config_path
    )
    config = driver.DeterministicLGSSMHMCConfig(
        config_payload, stored_config_path
    )
    config_hash = config.hash
    root = tmp_path / "artifacts"
    campaign = root / "phase7_campaign"
    private = campaign / "private" / "retained_samples.npz"
    private.parent.mkdir(parents=True)
    rng = np.random.default_rng(20260713)
    retained = rng.normal(0.0, 0.1, size=(1000, 4, 18))
    np.savez_compressed(
        private,
        retained_raw_samples=retained,
        final_worker_states=np.zeros((2, 2, 18)),
        config_hash=np.asarray(_sha("9")),
        private_replay_hash=np.asarray(_sha("8")),
    )
    private_hash = hashlib.sha256(private.read_bytes()).hexdigest()
    diagnostics = _passing_diagnostics(parameter_names)
    terminal_path = campaign / "result.json"
    _write_json(
        terminal_path,
        {
            "passed": True,
            "smoke": False,
            "config_hash": _sha("9"),
            "retained_results_per_chain": 1000,
            "artifact_hash": _sha("7"),
            "final_diagnostics": diagnostics,
            "private_retained_sample_reference": {
                "file_sha256": private_hash,
                "byte_count": private.stat().st_size,
                "shape_verified": True,
                "finite_verified": True,
                "provenance_verified": True,
            },
        },
    )
    _write_json(
        root / "fixture.json",
        {"raw_truth": [0.0] * 18},
    )
    phase7_config_path = tmp_path / "phase7.json"
    phase7_config_path.write_text("{}", encoding="utf-8")
    fake_phase7 = SimpleNamespace(
        payload={
            "schema": phase7.PHASE7_CONFIG_SCHEMA_V3,
            "source_tuning_config_hash": config_hash,
            "governed_source_references": {
                "private_replay": {"artifact_hash": _sha("8")}
            },
        },
        path=phase7_config_path,
        hash=_sha("9"),
        artifact_root=root,
        artifact_path=lambda name: {
            "public_result": terminal_path,
            "private_retained_samples": private,
        }[name],
    )
    monkeypatch.setattr(
        phase7.DeterministicLGSSMPhase7Config,
        "load",
        classmethod(lambda _cls, _path: fake_phase7),
    )
    monkeypatch.setattr(
        phase7,
        "validate_phase7_v3_inputs",
        lambda _cfg, require_fresh_outputs: {"artifact_hash": _sha("6")},
    )
    from bayesfilter.inference import hmc_convergence

    monkeypatch.setattr(
        hmc_convergence,
        "rank_normalized_hmc_diagnostics",
        lambda *_args, **_kwargs: diagnostics,
    )

    result = driver.build_final_recovery(
        config,
        phase7_config_path=phase7_config_path,
    )

    assert result["passed"] is True
    assert result["config_path"] == "config.json"
    assert result["terminal_diagnostics_agree"] is True
    assert result["private_retained_samples"]["file_sha256"] == private_hash
    assert len(result["parameter_recovery"]) == 18
    assert all(row["recovery_passed"] for row in result["parameter_recovery"])


def test_campaign_refuses_existing_root_before_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "phase7_campaign"
    root.mkdir()
    fake = SimpleNamespace(
        payload={"schema": phase7.PHASE7_CONFIG_SCHEMA_V3},
        artifact_path=lambda name: root / "result.json",
    )
    monkeypatch.setattr(
        launcher.DeterministicLGSSMPhase7Config,
        "load",
        classmethod(lambda _cls, _path: fake),
    )
    with pytest.raises(FileExistsError, match="already exists"):
        launcher.run_campaign(
            config_path=tmp_path / "phase7.json",
            campaign_root=root,
            command=("python", "launcher"),
        )


def _passing_diagnostics(parameter_names: tuple[str, ...]) -> dict:
    rows = tuple(
        {
            "parameter": name,
            "rank_normalized_split_rhat": 1.001,
            "folded_rank_normalized_split_rhat": 1.002,
            "rhat": 1.002,
            "bulk_ess": 1500.0,
            "tail_ess": 700.0,
            "lower_tail_ess": 700.0,
            "upper_tail_ess": 750.0,
            "passed": True,
        }
        for name in parameter_names
    )
    return {
        "schema": "bayesfilter.rank_normalized_hmc_diagnostics.v1",
        "passed": True,
        "input_all_finite": True,
        "diagnostics_all_finite": True,
        "draw_count_per_chain": 1000,
        "chain_count": 4,
        "parameter_count": 18,
        "split_draw_count_per_chain": 500,
        "split_chain_count": 8,
        "thresholds": {
            "rhat_max": 1.01,
            "bulk_ess_min": 1000.0,
            "tail_ess_min": 400.0,
        },
        "definitions": {"rhat": phase7._MODERN_RHAT_DEFINITION},
        "max_rhat": 1.002,
        "min_bulk_ess": 1500.0,
        "min_tail_ess": 700.0,
        "parameter_diagnostics": rows,
        "hard_vetoes": (),
        "nonclaims": (),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
