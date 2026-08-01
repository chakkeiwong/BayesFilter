from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import stat
import subprocess
import sys
from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from bayesfilter.inference import hmc_smoke_authority as smoke_authority_module
from bayesfilter.inference.hmc_identity import canonical_artifact_payload_hash
from bayesfilter.inference.hmc_identity_adoption import (
    build_phase5_artifact_reference,
)
from bayesfilter.inference.hmc_smoke_authority import (
    AUTHORITY_NONCLAIMS,
    HMC_PHASE6_SMOKE_LAUNCH_CLAIM_SCHEMA_V1,
    HMC_PHASE6_SMOKE_FAILURE_SCHEMA_V1,
    HMC_PHASE6_SMOKE_INFRASTRUCTURE_FAILURE_SCHEMA_V1,
    HMC_PHASE6_SMOKE_INFRASTRUCTURE_MANIFEST_SCHEMA_V1,
    HMC_PHASE6_SMOKE_PROGRESS_SCHEMA_V1,
    HMC_PHASE6_SMOKE_RESULT_SCHEMA_V1,
    PROPOSAL_NONCLAIMS,
    Phase6SmokeLaunchContext,
    PinnedSmokeOutputDirectories,
    SecureSmokeOutputSession,
    SMOKE_AUTHORITY_DECISION,
    SMOKE_BLOCK_DECISION,
    SMOKE_EXECUTION_IDENTITY_HASH,
    SMOKE_FAILURE_NONCLAIMS,
    SMOKE_NONCLAIMS,
    SMOKE_PASS_DECISION,
    TRANSITION_IDENTITY_HASH,
    build_launch_claim,
    build_smoke_infrastructure_manifest,
    build_smoke_output_manifest,
    build_smoke_authority,
    build_smoke_authority_proposal,
    build_smoke_authority_proposal_manifest,
    build_file_reference,
    create_durable_launch_claim,
    create_durable_launch_claim_with_consumed_evidence,
    default_smoke_paths,
    default_smoke_runtime,
    default_implementation_paths,
    expected_smoke_approval_statement,
    parse_launch_claim,
    parse_smoke_failure,
    parse_smoke_infrastructure_failure,
    parse_smoke_infrastructure_manifest,
    parse_smoke_output_manifest,
    parse_smoke_progress,
    parse_smoke_result,
    parse_smoke_authority,
    parse_smoke_authority_proposal,
    parse_smoke_authority_proposal_manifest,
    verify_smoke_infrastructure_manifest,
    verify_file_reference,
    verify_smoke_authority_proposal,
    verify_smoke_authority_proposal_manifest,
    verify_smoke_output_manifest,
    write_phase6_json,
    write_smoke_infrastructure_terminal,
)
from bayesfilter.runtime import atomic_write_json
from bayesfilter.testing import deterministic_lgssm_hmc_phase7_tf as controller
from scripts import run_hmc_phase6_typed_identity_smoke as smoke_launcher
from scripts import build_hmc_phase6_smoke_authority as authority_builder
from scripts import build_hmc_phase6_smoke_authority_proposal as proposal_builder


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = ROOT / (
    "docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11"
)
V2_PATH = ROOT / (
    "docs/benchmarks/configs/"
    "multidim_lgssm_phase7_typed_identity_baseline_2026_07_11.json"
)
ADOPTION_PATH = PUBLIC_ROOT / "typed_identity_baseline_adoption_record.json"
PREFLIGHT_PATH = PUBLIC_ROOT / "typed_identity_baseline_preflight.json"
PHASE5_MANIFEST_PATH = PUBLIC_ROOT / "phase5_output_integrity_manifest.json"
SUBPLAN_PATH = ROOT / (
    "docs/plans/bayesfilter-hmc-semantic-identity-migration-"
    "phase6-smoke-subplan-2026-07-11.md"
)


def _rehash(payload: dict) -> dict:
    restored = copy.deepcopy(payload)
    restored.pop("artifact_hash", None)
    restored["artifact_hash"] = canonical_artifact_payload_hash(restored)
    return restored


class _TestConsumedEvidenceSession:
    def __init__(self, *, fail_on_calls: set[int] | None = None) -> None:
        self.closed = False
        self.verify_calls = 0
        self.fail_on_calls = set() if fail_on_calls is None else set(fail_on_calls)

    def verify(self) -> dict:
        if self.closed:
            raise RuntimeError("test consumed evidence session is closed")
        self.verify_calls += 1
        if self.verify_calls in self.fail_on_calls:
            raise smoke_authority_module.ConsumedAttempt1EvidenceDriftError(
                f"test evidence drift at call {self.verify_calls}"
            )
        return {}

    def close(self) -> None:
        self.closed = True


def _live_implementation_references() -> dict[str, dict]:
    return {
        name: dict(build_file_reference(path))
        for name, path in default_implementation_paths(sys.executable).items()
    }


def _live_implementation_source_bundle() -> dict[str, bytes]:
    references = _live_implementation_references()
    return dict(
        smoke_authority_module.build_verified_implementation_source_bundle(
            references,
            python_executable=sys.executable,
        )
    )


def _proposal(tmp_path: Path) -> tuple[dict, dict, Path, Path]:
    command = (
        str(Path(sys.executable).resolve()),
        "scripts/run_hmc_phase6_typed_identity_smoke.py",
        "--stage",
        "burnin_sampling",
        "--phase7-smoke",
        "--phase7-smoke-authority",
        "docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/"
        "phase6_smoke_attempt2_authority.json",
    )
    proposal = build_smoke_authority_proposal(
        phase6_subplan_path=SUBPLAN_PATH,
        v2_config_path=V2_PATH,
        adoption_record_path=ADOPTION_PATH,
        preflight_path=PREFLIGHT_PATH,
        phase5_manifest_path=PHASE5_MANIFEST_PATH,
        runtime=default_smoke_runtime(),
        paths=default_smoke_paths(),
        command=command,
        implementation_references={
            name: build_file_reference(path)
            for name, path in default_implementation_paths(sys.executable).items()
        },
    )
    proposal_path = tmp_path / "proposal.json"
    atomic_write_json(proposal_path, proposal)
    manifest = build_smoke_authority_proposal_manifest(proposal_path=proposal_path)
    manifest_path = tmp_path / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    return proposal, manifest, proposal_path, manifest_path


def _authority(tmp_path: Path) -> tuple[dict, dict, dict, tuple[str, ...]]:
    proposal, manifest, _proposal_path, manifest_path = _proposal(tmp_path)
    approval = expected_smoke_approval_statement(manifest["artifact_hash"])
    authority = build_smoke_authority(
        proposal_manifest_path=manifest_path,
        human_approval_statement=approval,
        human_approval_date="2026-07-11",
    )
    command = tuple(proposal["command"])
    claim = build_launch_claim(
        authority=authority,
        proposal_manifest=manifest,
        command=command,
        paths=proposal["paths"],
        pid=os.getpid(),
        started_at_utc="2026-07-11T00:00:00+00:00",
    )
    return proposal, manifest, authority, claim


def test_pending_proposal_round_trip_and_live_references(tmp_path: Path) -> None:
    proposal, manifest, proposal_path, _manifest_path = _proposal(tmp_path)

    assert parse_smoke_authority_proposal(proposal) == proposal
    assert proposal["status"] == "pending_human_smoke_approval"
    assert proposal["decision"] == SMOKE_AUTHORITY_DECISION
    assert proposal["transition_identity_hash"] == TRANSITION_IDENTITY_HASH
    assert proposal["smoke_execution_identity_hash"] == SMOKE_EXECUTION_IDENTITY_HASH
    assert proposal["serious_runtime_authority"] is False
    assert proposal["nonclaims"] == PROPOSAL_NONCLAIMS
    assert parse_smoke_authority_proposal_manifest(manifest) == manifest
    assert verify_smoke_authority_proposal_manifest(
        manifest, proposal_path=proposal_path
    ) == manifest


def test_proposal_live_verifier_rejects_implementation_drift(tmp_path: Path) -> None:
    proposal, _manifest, _proposal_path, _manifest_path = _proposal(tmp_path)
    verify_smoke_authority_proposal(
        proposal,
        phase6_subplan_path=SUBPLAN_PATH,
        artifact_paths={
            "v2_config_reference": V2_PATH,
            "adoption_record_reference": ADOPTION_PATH,
            "preflight_reference": PREFLIGHT_PATH,
            "phase5_manifest_reference": PHASE5_MANIFEST_PATH,
        },
        implementation_paths=default_implementation_paths(sys.executable),
    )
    drifted = dict(default_implementation_paths(sys.executable))
    implementation = tmp_path / "implementation.py"
    implementation.write_text("VALUE = 2\n", encoding="utf-8")
    authority_role = "repository_file:bayesfilter/inference/hmc_smoke_authority.py"
    drifted[authority_role] = implementation
    with pytest.raises(ValueError, match="current bytes"):
        verify_smoke_authority_proposal(
            proposal,
            phase6_subplan_path=SUBPLAN_PATH,
            artifact_paths={
                "v2_config_reference": V2_PATH,
                "adoption_record_reference": ADOPTION_PATH,
                "preflight_reference": PREFLIGHT_PATH,
                "phase5_manifest_reference": PHASE5_MANIFEST_PATH,
            },
            implementation_paths=drifted,
        )


def test_child_source_bundle_is_bound_to_exact_approved_bytes() -> None:
    references = _live_implementation_references()
    bundle = _live_implementation_source_bundle()
    assert smoke_authority_module.verify_implementation_source_bundle(
        references,
        bundle,
        python_executable=sys.executable,
    ) == bundle
    role = (
        "repository_file:bayesfilter/testing/"
        "deterministic_lgssm_hmc_phase7_tf.py"
    )
    tampered = dict(bundle)
    tampered[role] += b"\n# restored-path race payload\n"
    with pytest.raises(ValueError, match="source bytes mismatch"):
        smoke_authority_module.verify_implementation_source_bundle(
            references,
            tampered,
            python_executable=sys.executable,
        )


def test_child_source_loader_bootstrap_rejects_import_restore_race() -> None:
    references = _live_implementation_references()
    bundle = _live_implementation_source_bundle()
    role = (
        "repository_file:bayesfilter/testing/"
        "deterministic_lgssm_hmc_phase7_tf.py"
    )
    tampered = dict(bundle)
    tampered[role] += b"\n# imported before restored path\n"
    with pytest.raises(ValueError, match="source bytes mismatch"):
        smoke_authority_module.child_source_loader_initializer(
            references=references,
            source_bundle=tampered,
            worker_environment=controller._worker_environment(
                controller.DeterministicLGSSMPhase7Config.load(
                    controller.DEFAULT_CONFIG_PATH
                )
            ),
            python_executable=sys.executable,
        )


def test_child_source_loader_owns_benchmark_namespace_parents(
    tmp_path: Path,
) -> None:
    approved_root = tmp_path / "approved"
    benchmark_path = approved_root / "docs/benchmarks/retained_driver.py"
    benchmark_path.parent.mkdir(parents=True)
    benchmark_path.write_text(
        "RETAINED_VALUE = 17\nRETAINED_FILE = __file__\n", encoding="utf-8"
    )

    poison_root = tmp_path / "poison"
    poison_benchmarks = poison_root / "docs/benchmarks"
    poison_benchmarks.mkdir(parents=True)
    marker_path = tmp_path / "poison_parent_executed"
    poison_source = (
        "from pathlib import Path\n"
        f"Path({str(marker_path)!r}).write_text('executed', encoding='utf-8')\n"
    )
    (poison_root / "docs/__init__.py").write_text(
        poison_source, encoding="utf-8"
    )
    (poison_benchmarks / "__init__.py").write_text(
        poison_source, encoding="utf-8"
    )

    script = r'''
import hashlib
import importlib
from pathlib import Path
import sys
import types

from bayesfilter.inference.hmc_smoke_authority import (
    CHILD_SOURCE_LOADER_BOOTSTRAP,
    implementation_source_bundle_hash,
)

approved_root = Path(sys.argv[1])
poison_root = Path(sys.argv[2])
marker_path = Path(sys.argv[3])
benchmark_path = approved_root / "docs/benchmarks/retained_driver.py"
source = benchmark_path.read_bytes()
role = "repository_file:docs/benchmarks/retained_driver.py"
references = {
    role: {
        "file_sha256": hashlib.sha256(source).hexdigest(),
        "byte_count": len(source),
        "resolved_path_sha256": hashlib.sha256(
            str(benchmark_path.resolve()).encode("utf-8")
        ).hexdigest(),
    }
}
source_bundle = {role: source}
bundle_hash = implementation_source_bundle_hash(source_bundle)

sys.path.insert(0, str(poison_root))
sys.modules["docs"] = types.ModuleType("docs")
sys.modules["docs.benchmarks"] = types.ModuleType("docs.benchmarks")
bootstrap_globals = {
    "__name__": "_phase6_namespace_test",
    "_worker_environment": {},
    "_implementation_references": references,
    "_implementation_source_bundle": source_bundle,
    "_implementation_source_bundle_hash": bundle_hash,
    "_approved_repository_root": str(approved_root),
    "_benchmark_driver_relative_path": "docs/benchmarks/retained_driver.py",
    "_benchmark_driver_module": "docs.benchmarks.retained_driver",
}
exec(CHILD_SOURCE_LOADER_BOOTSTRAP, bootstrap_globals)
module = importlib.import_module("docs.benchmarks.retained_driver")
assert module.RETAINED_VALUE == 17
assert module.RETAINED_FILE == str(benchmark_path)
assert module.__file__ == str(benchmark_path)
assert not marker_path.exists()
for name in ("docs", "docs.benchmarks"):
    parent = sys.modules[name]
    assert parent.__phase6_synthetic_namespace__ == name
    assert parent.__phase6_source_bundle_hash__ == bundle_hash
try:
    importlib.import_module("docs.unapproved")
except ImportError as error:
    assert "unapproved docs module" in str(error)
else:
    raise AssertionError("unapproved docs import unexpectedly succeeded")
'''
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(approved_root),
            str(poison_root),
            str(marker_path),
        ],
        cwd=ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not marker_path.exists()


def test_loaded_module_verifier_rejects_imported_then_restored_source() -> None:
    references = _live_implementation_references()
    controller_role = (
        "repository_file:bayesfilter/testing/"
        "deterministic_lgssm_hmc_phase7_tf.py"
    )
    loaded_controller = SimpleNamespace(
        __phase6_source_role__=controller_role,
        __phase6_source_sha256__="f" * 64,
        __phase6_source_bundle_hash__="sha256:" + "a" * 64,
    )
    with pytest.raises(
        controller.DeterministicLGSSMPhase7Error,
        match="imported unverified",
    ):
        controller._verify_loaded_child_modules(
            references,
            expected_bundle_hash="sha256:" + "a" * 64,
            loaded_modules={
                "bayesfilter.testing.deterministic_lgssm_hmc_phase7_tf": (
                    loaded_controller
                )
            },
            require_runtime_imports=False,
        )


def test_loaded_module_verifier_accepts_required_benchmark_source() -> None:
    references = _live_implementation_references()
    bundle_hash = "sha256:" + "a" * 64
    controller_role = (
        "repository_file:bayesfilter/testing/"
        "deterministic_lgssm_hmc_phase7_tf.py"
    )
    benchmark_role = (
        "repository_file:docs/benchmarks/"
        "run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py"
    )

    def loaded(role: str) -> SimpleNamespace:
        return SimpleNamespace(
            __phase6_source_role__=role,
            __phase6_source_sha256__=references[role]["file_sha256"],
            __phase6_source_bundle_hash__=bundle_hash,
        )

    assert controller._verify_loaded_child_modules(
        references,
        expected_bundle_hash=bundle_hash,
        loaded_modules={
            "docs": SimpleNamespace(
                __phase6_synthetic_namespace__="docs",
                __phase6_source_bundle_hash__=bundle_hash,
            ),
            "docs.benchmarks": SimpleNamespace(
                __phase6_synthetic_namespace__="docs.benchmarks",
                __phase6_source_bundle_hash__=bundle_hash,
            ),
            "bayesfilter.testing.deterministic_lgssm_hmc_phase7_tf": loaded(
                controller_role
            ),
            "docs.benchmarks."
            "run_multidim_lgssm_serious_hmc_tuning_2026_07_09": loaded(
                benchmark_role
            ),
        },
        require_runtime_imports=True,
    ) is None


def test_implementation_inventory_is_runtime_scoped_and_role_drift_is_rejected(
    tmp_path: Path,
) -> None:
    inventory = default_implementation_paths(sys.executable)
    required = {
        "repository_file:bayesfilter/inference/__init__.py",
        "repository_file:bayesfilter/inference/hmc.py",
        "repository_file:bayesfilter/inference/hmc_kernel_tuning.py",
        "repository_file:bayesfilter/inference/hmc_convergence.py",
        "repository_file:docs/benchmarks/"
        "run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py",
        "repository_file:scripts/run_hmc_phase6_typed_identity_smoke.py",
        "repository_file:tests/test_hmc_smoke_authority.py",
        "python_executable",
    }
    assert required <= set(inventory)
    assert {
        role
        for role in inventory
        if role.startswith("repository_file:tests/")
    } == {
        "repository_file:tests/test_hmc_identity.py",
        "repository_file:tests/test_hmc_identity_adoption.py",
        "repository_file:tests/test_hmc_identity_integration.py",
        "repository_file:tests/test_hmc_identity_migration_certificate.py",
        "repository_file:tests/test_hmc_convergence.py",
        "repository_file:tests/test_deterministic_lgssm_hmc_phase7_tf.py",
        "repository_file:tests/test_deterministic_lgssm_hmc_tuning_driver.py",
        "repository_file:tests/test_hmc_smoke_authority.py",
    }
    assert "repository_file:bayesfilter/testing/multidim_triangular_lgssm_tf.py" in inventory
    assert "repository_file:bayesfilter/linear/kalman_svd_derivatives_tf.py" in inventory
    assert not any("complete_highdim_leaderboard" in role for role in inventory)

    proposal, _manifest, _proposal_path, _manifest_path = _proposal(tmp_path)
    missing = dict(inventory)
    missing.pop("repository_file:bayesfilter/inference/hmc.py")
    with pytest.raises(ValueError, match="implementation roles mismatch"):
        verify_smoke_authority_proposal(
            proposal,
            phase6_subplan_path=SUBPLAN_PATH,
            artifact_paths={
                "v2_config_reference": V2_PATH,
                "adoption_record_reference": ADOPTION_PATH,
                "preflight_reference": PREFLIGHT_PATH,
                "phase5_manifest_reference": PHASE5_MANIFEST_PATH,
            },
            implementation_paths=missing,
        )


def test_runtime_scoped_inventory_covers_clean_process_import_closure() -> None:
    script = r'''
import importlib
import json
from pathlib import Path
import sys

root = Path.cwd().resolve()
importlib.import_module("bayesfilter.testing.deterministic_lgssm_hmc_phase7_tf")
importlib.import_module(
    "docs.benchmarks.run_multidim_lgssm_serious_hmc_tuning_2026_07_09"
)
loaded = []
for module in tuple(sys.modules.values()):
    source = getattr(module, "__file__", None)
    if source is None:
        continue
    try:
        relative = Path(source).resolve().relative_to(root).as_posix()
    except (OSError, ValueError):
        continue
    if relative.startswith("bayesfilter/") and relative.endswith(".py"):
        loaded.append(relative)
print(json.dumps(sorted(set(loaded))))
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "-1",
            "TF_NUM_INTRAOP_THREADS": "8",
            "TF_NUM_INTEROP_THREADS": "1",
            "OMP_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "MPLCONFIGDIR": "/tmp/matplotlib-bayesfilter-phase6-import-test",
            "TF_CPP_MIN_LOG_LEVEL": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    observed = set(json.loads(result.stdout.strip().splitlines()[-1]))
    inventory = {
        role.removeprefix("repository_file:")
        for role in default_implementation_paths(sys.executable)
        if role.startswith("repository_file:bayesfilter/")
    }
    assert observed <= inventory


def test_live_proposal_verifier_rejects_inventory_addition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposal, _manifest, _proposal_path, _manifest_path = _proposal(tmp_path)
    original = default_implementation_paths(sys.executable)
    added = tmp_path / "added.py"
    added.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        smoke_authority_module,
        "default_implementation_paths",
        lambda _python: {**original, "repository_file:added.py": added},
    )
    with pytest.raises(ValueError, match="implementation roles mismatch"):
        verify_smoke_authority_proposal(
            proposal,
            phase6_subplan_path=SUBPLAN_PATH,
            artifact_paths={
                "v2_config_reference": V2_PATH,
                "adoption_record_reference": ADOPTION_PATH,
                "preflight_reference": PREFLIGHT_PATH,
                "phase5_manifest_reference": PHASE5_MANIFEST_PATH,
            },
            implementation_paths=None,
        )


def test_consumed_v2_output_manifest_remains_archivally_verifiable() -> None:
    proposal_path = PUBLIC_ROOT / "phase6_smoke_authority_proposal_v2.json"
    proposal_manifest_path = (
        PUBLIC_ROOT / "phase6_smoke_authority_proposal_manifest_v2.json"
    )
    authority_path = PUBLIC_ROOT / "phase6_smoke_authority.json"
    claim_path = PUBLIC_ROOT / "phase6_smoke_launch_claim.json"
    progress_path = PUBLIC_ROOT / "phase6_smoke_progress.json"
    result_path = PUBLIC_ROOT / "phase6_smoke_result.json"
    output_manifest_path = PUBLIC_ROOT / "phase6_smoke_output_manifest.json"
    log_path = ROOT / (
        "docs/plans/logs/hmc-semantic-identity-migration-2026-07-11/"
        "phase6_smoke.log"
    )
    private_samples_path = ROOT / (
        "docs/benchmarks/artifacts/"
        "multidim_lgssm_serious_hmc_tuning_2026_07_09/private_diagnostics/"
        "phase6_typed_identity_smoke_retained_samples.npz"
    )
    infrastructure_failure_path = PUBLIC_ROOT / "phase6_smoke_infrastructure_failure.json"
    infrastructure_manifest_path = PUBLIC_ROOT / "phase6_smoke_infrastructure_manifest.json"
    payload = json.loads(output_manifest_path.read_text(encoding="utf-8"))

    assert verify_smoke_output_manifest(
        payload,
        proposal_path=proposal_path,
        proposal_manifest_path=proposal_manifest_path,
        authority_path=authority_path,
        claim_path=claim_path,
        progress_path=progress_path,
        result_path=result_path,
        log_path=log_path,
        private_samples_path=private_samples_path,
        infrastructure_failure_path=infrastructure_failure_path,
        infrastructure_manifest_path=infrastructure_manifest_path,
    ) == payload

    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="reviewed paths|reviewed command"):
        verify_smoke_authority_proposal(
            proposal,
            phase6_subplan_path=SUBPLAN_PATH,
            artifact_paths={
                "v2_config_reference": V2_PATH,
                "adoption_record_reference": ADOPTION_PATH,
                "preflight_reference": PREFLIGHT_PATH,
                "phase5_manifest_reference": PHASE5_MANIFEST_PATH,
            },
            implementation_paths=None,
        )


def test_consumed_attempt1_integrity_gate_verifies_exact_terminal_evidence() -> None:
    report = smoke_authority_module.verify_consumed_attempt1_evidence()
    assert set(report) == {
        "original_proposal",
        "original_proposal_manifest",
        "attempt1_proposal",
        "attempt1_proposal_manifest",
        "attempt1_authority",
        "attempt1_claim",
        "attempt1_result",
        "attempt1_progress",
        "attempt1_output_manifest",
        "attempt1_infrastructure_failure_reservation",
        "attempt1_infrastructure_manifest_reservation",
        "attempt1_private_sample_reservation",
        "attempt1_log",
    }
    assert report["attempt1_claim"]["file_mode"] == "0400"
    assert report["attempt1_private_sample_reservation"]["byte_count"] == 0
    assert report["attempt1_log"]["file_sha256"] == (
        "6dee7ec170811c18c87fc1ee3fa0397213325363a5c1e4e2c294874cc5e7bf80"
    )


def test_consumed_attempt1_integrity_gate_rejects_exact_byte_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed_log = tmp_path / "phase6_smoke.log"
    changed_log.write_bytes(
        smoke_authority_module.SUPERSEDED_LOG_PATH.read_bytes() + b"tamper"
    )
    changed_log.chmod(0o400)
    monkeypatch.setattr(smoke_authority_module, "SUPERSEDED_LOG_PATH", changed_log)
    with pytest.raises(
        smoke_authority_module.ConsumedAttempt1EvidenceDriftError,
        match="byte count mismatch|exact bytes mismatch",
    ):
        smoke_authority_module.verify_consumed_attempt1_evidence()


def test_consumed_attempt1_integrity_gate_rejects_mode_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed_log = tmp_path / "phase6_smoke.log"
    changed_log.write_bytes(smoke_authority_module.SUPERSEDED_LOG_PATH.read_bytes())
    changed_log.chmod(0o600)
    monkeypatch.setattr(smoke_authority_module, "SUPERSEDED_LOG_PATH", changed_log)
    with pytest.raises(
        smoke_authority_module.ConsumedAttempt1EvidenceDriftError,
        match="file mode mismatch",
    ):
        smoke_authority_module.verify_consumed_attempt1_evidence()


def test_consumed_attempt1_integrity_gate_rejects_same_size_byte_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed_log = tmp_path / "phase6_smoke.log"
    data = bytearray(smoke_authority_module.SUPERSEDED_LOG_PATH.read_bytes())
    data[0] ^= 1
    changed_log.write_bytes(data)
    changed_log.chmod(0o400)
    monkeypatch.setattr(smoke_authority_module, "SUPERSEDED_LOG_PATH", changed_log)
    with pytest.raises(
        smoke_authority_module.ConsumedAttempt1EvidenceDriftError,
        match="exact bytes mismatch",
    ):
        smoke_authority_module.verify_consumed_attempt1_evidence()


def test_consumed_attempt1_integrity_gate_rejects_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.log"
    target.write_bytes(smoke_authority_module.SUPERSEDED_LOG_PATH.read_bytes())
    target.chmod(0o400)
    linked = tmp_path / "phase6_smoke.log"
    linked.symlink_to(target)
    monkeypatch.setattr(smoke_authority_module, "SUPERSEDED_LOG_PATH", linked)
    with pytest.raises(ValueError, match="contains a symlink"):
        smoke_authority_module.verify_consumed_attempt1_evidence()


def test_consumed_attempt1_integrity_gate_rejects_hard_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.log"
    target.write_bytes(smoke_authority_module.SUPERSEDED_LOG_PATH.read_bytes())
    target.chmod(0o400)
    linked = tmp_path / "phase6_smoke.log"
    os.link(target, linked)
    monkeypatch.setattr(smoke_authority_module, "SUPERSEDED_LOG_PATH", linked)
    with pytest.raises(RuntimeError, match="link count"):
        smoke_authority_module.verify_consumed_attempt1_evidence()


def test_consumed_attempt1_integrity_gate_rejects_exclusive_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed_log = tmp_path / "phase6_smoke.log"
    changed_log.write_bytes(smoke_authority_module.SUPERSEDED_LOG_PATH.read_bytes())
    writer = os.open(changed_log, os.O_RDWR)
    changed_log.chmod(0o400)
    monkeypatch.setattr(smoke_authority_module, "SUPERSEDED_LOG_PATH", changed_log)
    try:
        fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="locked for mutation"):
            smoke_authority_module.verify_consumed_attempt1_evidence()
    finally:
        os.close(writer)


def test_consumed_attempt1_integrity_gate_detects_concurrent_same_size_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed_log = tmp_path / "phase6_smoke.log"
    original = smoke_authority_module.SUPERSEDED_LOG_PATH.read_bytes()
    changed_log.write_bytes(original)
    writer = os.open(changed_log, os.O_RDWR)
    changed_log.chmod(0o400)
    target_inode = os.fstat(writer).st_ino
    real_pread = os.pread
    mutated = bytes([original[0] ^ 1]) + original[1:]
    wrote = False

    def race(fd: int, count: int, offset: int) -> bytes:
        nonlocal wrote
        data = real_pread(fd, count, offset)
        if not wrote and os.fstat(fd).st_ino == target_inode:
            wrote = True
            os.pwrite(writer, mutated, 0)
            os.fsync(writer)
        return data

    monkeypatch.setattr(smoke_authority_module, "SUPERSEDED_LOG_PATH", changed_log)
    monkeypatch.setattr(os, "pread", race)
    try:
        with pytest.raises(RuntimeError, match="changed during verification"):
            smoke_authority_module.verify_consumed_attempt1_evidence()
        assert wrote is True
    finally:
        os.close(writer)


def test_consumed_attempt1_session_rejects_restored_bytes_after_capture(
    tmp_path: Path,
) -> None:
    changed_log = tmp_path / "phase6_smoke.log"
    original = smoke_authority_module.SUPERSEDED_LOG_PATH.read_bytes()
    changed_log.write_bytes(original)
    writer = os.open(changed_log, os.O_RDWR)
    changed_log.chmod(0o400)
    parent_fd = os.open(tmp_path, smoke_authority_module._directory_open_flags())
    evidence_fd = os.open(
        changed_log.name,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_fd,
    )
    fcntl.flock(evidence_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    parent_info = os.fstat(parent_fd)
    session = smoke_authority_module.ConsumedAttempt1EvidenceSession(
        entries={
            "attempt1_log": {
                "path": changed_log,
                "parent_fd": parent_fd,
                "fd": evidence_fd,
                "expected_size": len(original),
                "expected_sha256": hashlib.sha256(original).hexdigest(),
                "expected_mode": 0o400,
                "parent_identity": (parent_info.st_dev, parent_info.st_ino),
            }
        }
    )
    try:
        with session:
            session.verify(capture=True)
            captured = session.entries["attempt1_log"]["capture_signature"]
            mutated = bytes([original[0] ^ 1]) + original[1:]
            os.pwrite(writer, mutated, 0)
            os.fsync(writer)
            os.pwrite(writer, original, 0)
            os.fsync(writer)
            # Some filesystems coalesce rapid write timestamps. Move mtime to a
            # deterministic later value while preserving exact bytes and mode.
            current = os.fstat(writer)
            os.utime(
                writer,
                ns=(current.st_atime_ns, captured[7] + 1_000_000_000),
            )
            assert changed_log.read_bytes() == original
            assert stat.S_IMODE(changed_log.stat().st_mode) == 0o400
            assert smoke_authority_module._consumed_evidence_signature(
                os.fstat(writer)
            ) != captured
            with pytest.raises(
                smoke_authority_module.ConsumedAttempt1EvidenceDriftError,
                match="capture-time signature",
            ):
                session.verify()
    finally:
        os.close(writer)


def test_consumed_attempt1_snapshot_rejects_proposal_semantic_drift() -> None:
    with smoke_authority_module.ConsumedAttempt1EvidenceSession.open() as session:
        snapshots = dict(session.snapshots)
    proposal = json.loads(snapshots["attempt1_proposal"])
    proposal["status"] = "approved"
    snapshots["attempt1_proposal"] = json.dumps(proposal).encode("utf-8")
    with pytest.raises(ValueError, match="must remain pending"):
        smoke_authority_module._verify_consumed_attempt1_snapshot_semantics(
            snapshots
        )


def test_consumed_attempt1_snapshot_rejects_terminal_classification_drift() -> None:
    with smoke_authority_module.ConsumedAttempt1EvidenceSession.open() as session:
        snapshots = dict(session.snapshots)
    result = json.loads(snapshots["attempt1_result"])
    result["reason"] = "runtime_error:DifferentFailure"
    snapshots["attempt1_result"] = json.dumps(result).encode("utf-8")
    with pytest.raises(ValueError, match="terminal classification mismatch"):
        smoke_authority_module._verify_consumed_attempt1_snapshot_semantics(
            snapshots
        )


def test_live_candidate_checks_consumed_attempt1_before_current_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposal, _manifest, _proposal_path, _manifest_path = _proposal(tmp_path)

    def reject_consumed_evidence():
        raise RuntimeError("attempt1 evidence drift")

    monkeypatch.setattr(
        smoke_authority_module,
        "verify_consumed_attempt1_evidence",
        reject_consumed_evidence,
    )
    with pytest.raises(RuntimeError, match="attempt1 evidence drift"):
        smoke_authority_module.verify_default_smoke_authority_proposal_candidate(
            proposal,
            python_executable=sys.executable,
        )


def test_smoke_parent_environment_is_observed_not_assumed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "99")
    with pytest.raises(ValueError, match="thread environment mismatch"):
        default_smoke_runtime()


def test_preclaim_protected_inventory_covers_governed_and_serious_paths() -> None:
    config = controller.DeterministicLGSSMPhase7Config.load(controller.DEFAULT_CONFIG_PATH)
    protected = smoke_authority_module._protected_smoke_paths(config)
    expected = set(controller.phase7_governed_source_paths(config).values())
    expected.update(
        ROOT / value for value in config.payload["artifacts"].values()
    )
    source_config = json.loads(
        (ROOT / config.payload["source_tuning_config_path"]).read_text(encoding="utf-8")
    )
    expected.update(
        ROOT / value for value in source_config["artifact_paths"].values()
    )
    expected.update(default_implementation_paths(sys.executable).values())
    assert {path.resolve() for path in expected} <= protected
    assert (ROOT / source_config["artifact_paths"]["log"]).resolve() in protected


def test_preclaim_collision_fails_without_claim_output_or_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = controller.DeterministicLGSSMPhase7Config.load(controller.DEFAULT_CONFIG_PATH)
    paths = _secure_output_paths(tmp_path)
    paths["public_result_path"].write_bytes(b"existing")
    proposal = {"paths": {name: str(path) for name, path in paths.items()}}
    worker_created = False

    def forbidden(*_args, **_kwargs):
        nonlocal worker_created
        worker_created = True

    monkeypatch.setattr(
        controller.concurrent.futures,
        "ProcessPoolExecutor",
        forbidden,
    )
    monkeypatch.setattr(
        smoke_authority_module,
        "_resolve_reviewed_paths",
        lambda _paths: paths,
    )
    monkeypatch.setattr(
        smoke_authority_module,
        "_protected_smoke_paths",
        lambda _config: set(),
    )
    with pytest.raises(FileExistsError, match="public_result_path"):
        smoke_authority_module._verify_no_path_alias_or_existing_output(
            proposal=proposal,
            config=config,
        )
    assert not paths["claim_path"].exists()
    assert not paths["log_path"].exists()
    assert not paths["public_progress_path"].exists()
    assert not paths["private_samples_path"].exists()
    assert worker_created is False


def test_preclaim_serious_alias_fails_without_claim_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = controller.DeterministicLGSSMPhase7Config.load(controller.DEFAULT_CONFIG_PATH)
    paths = _secure_output_paths(tmp_path)
    serious_log = ROOT / (
        "docs/benchmarks/artifacts/"
        "multidim_lgssm_serious_hmc_tuning_2026_07_09/run.log"
    )
    paths["log_path"] = serious_log
    proposal = {"paths": {name: str(path) for name, path in paths.items()}}
    monkeypatch.setattr(
        smoke_authority_module,
        "_resolve_reviewed_paths",
        lambda _paths: paths,
    )
    with pytest.raises(ValueError, match="aliases a governed or serious"):
        smoke_authority_module._verify_no_path_alias_or_existing_output(
            proposal=proposal,
            config=config,
        )
    assert not paths["claim_path"].exists()
    assert not paths["public_result_path"].exists()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(status="approved"),
        lambda payload: payload.update(serious_runtime_authority=True),
        lambda payload: payload["runtime"].update(mode="serious"),
        lambda payload: payload["runtime"].update(worker_count=1),
        lambda payload: payload["paths"].update(public_result_path="/tmp/result.json"),
        lambda payload: payload.update(unexpected=True),
    ],
)
def test_proposal_rejects_rehashed_scope_or_schema_tamper(
    tmp_path: Path, mutation
) -> None:
    proposal, _manifest, _proposal_path, _manifest_path = _proposal(tmp_path)
    mutation(proposal)
    with pytest.raises((TypeError, ValueError)):
        parse_smoke_authority_proposal(_rehash(proposal))


def test_authority_requires_exact_manifest_bound_human_statement(tmp_path: Path) -> None:
    _proposal_payload, manifest, _proposal_path, manifest_path = _proposal(tmp_path)
    expected = expected_smoke_approval_statement(manifest["artifact_hash"])
    with pytest.raises(ValueError, match="approval statement"):
        build_smoke_authority(
            proposal_manifest_path=manifest_path,
            human_approval_statement="approved",
            human_approval_date="2026-07-11",
        )
    authority = build_smoke_authority(
        proposal_manifest_path=manifest_path,
        human_approval_statement=expected,
        human_approval_date="2026-07-11",
    )
    assert parse_smoke_authority(authority) == authority
    assert authority["launches_authorized"] == 1
    assert authority["mode"] == "smoke"
    assert authority["nonclaims"] == AUTHORITY_NONCLAIMS


@pytest.mark.parametrize(
    "approval_date",
    ["2026-7-11", "2026-07-11T00:00:00Z", "not-a-date", " 2026-07-11"],
)
def test_authority_rejects_noncanonical_approval_date(
    tmp_path: Path,
    approval_date: str,
) -> None:
    _proposal_payload, manifest, _proposal_path, manifest_path = _proposal(tmp_path)
    with pytest.raises(ValueError, match="YYYY-MM-DD|trimmed"):
        build_smoke_authority(
            proposal_manifest_path=manifest_path,
            human_approval_statement=expected_smoke_approval_statement(
                manifest["artifact_hash"]
            ),
            human_approval_date=approval_date,
        )


def test_phase6_json_writer_is_durable_and_exactly_restartable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _rehash({"schema": "test.phase6.v1", "value": 1})
    path = tmp_path / "artifact.json"
    calls: list[int] = []
    real_fsync = os.fsync

    def parser(candidate):
        assert candidate == payload
        return candidate

    def observed(fd: int) -> None:
        calls.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", observed)
    assert write_phase6_json(path, payload, parser=parser) == payload
    assert len(calls) == 2
    assert write_phase6_json(path, payload, parser=parser) == payload
    assert len(calls) == 4


@pytest.mark.parametrize("failure_index", [1, 2])
def test_phase6_json_writer_recovers_exact_bytes_after_sync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_index: int,
) -> None:
    payload = _rehash({"schema": "test.phase6.v1", "value": 1})
    path = tmp_path / "artifact.json"
    real_fsync = os.fsync
    calls = 0

    def parser(candidate):
        assert candidate == payload
        return candidate

    def injected(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == failure_index:
            raise OSError(f"injected Phase 6 fsync failure {failure_index}")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", injected)
    with pytest.raises(OSError, match="injected Phase 6 fsync failure"):
        write_phase6_json(path, payload, parser=parser)
    assert path.exists()
    assert write_phase6_json(path, payload, parser=parser) == payload


@pytest.mark.parametrize("existing", [b"{", b"different\n"])
def test_phase6_json_writer_rejects_partial_or_different_existing_bytes(
    tmp_path: Path,
    existing: bytes,
) -> None:
    payload = _rehash({"schema": "test.phase6.v1", "value": 1})
    path = tmp_path / "artifact.json"
    path.write_bytes(existing)
    with pytest.raises(FileExistsError, match="different or partial"):
        write_phase6_json(path, payload, parser=lambda candidate: candidate)
    assert path.read_bytes() == existing


def test_phase6_json_writer_rejects_existing_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    payload = _rehash({"schema": "test.phase6.v1", "value": 1})
    path = tmp_path / "artifact.json"
    protected = tmp_path / "protected.json"
    protected.write_bytes(b"protected")
    path.symlink_to(protected)
    with pytest.raises(OSError):
        write_phase6_json(path, payload, parser=lambda candidate: candidate)
    assert protected.read_bytes() == b"protected"


def test_authority_builder_requires_exact_statement_and_writes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposal, manifest, _proposal_path, manifest_path = _proposal(tmp_path)
    authority_path = tmp_path / "authority.json"
    monkeypatch.setattr(authority_builder, "PROPOSAL_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(authority_builder, "AUTHORITY_PATH", authority_path)
    monkeypatch.setattr(
        authority_builder,
        "verify_default_smoke_authority_proposal_bundle",
        lambda **_kwargs: (proposal, manifest, object(), {}),
    )
    expected = expected_smoke_approval_statement(manifest["artifact_hash"])
    with pytest.raises(ValueError, match="approval statement mismatch"):
        authority_builder.main(
            ["--approval-statement", "approved", "--approval-date", "2026-07-11"]
        )
    assert not authority_path.exists()
    assert authority_builder.main(
        ["--approval-statement", expected, "--approval-date", "2026-07-11"]
    ) == 0
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    assert parse_smoke_authority(authority) == authority
    assert authority["human_approval_statement"] == expected
    assert authority_builder.main(
        ["--approval-statement", expected, "--approval-date", "2026-07-11"]
    ) == 0


def test_proposal_builder_verifies_live_bundle_before_terminal_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposal_path = tmp_path / "proposal.json"
    manifest_path = tmp_path / "manifest.json"
    proposal, _manifest, _source, _terminal = _proposal(tmp_path / "source")
    events: list[str] = []
    monkeypatch.setattr(proposal_builder, "PROPOSAL_PATH", proposal_path)
    monkeypatch.setattr(proposal_builder, "PROPOSAL_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(
        proposal_builder,
        "build_default_smoke_authority_proposal",
        lambda **_kwargs: proposal,
    )
    def fail_live_bundle(_proposal, **kwargs):
        assert kwargs["python_executable"] == Path(sys.executable).resolve()
        events.append("live_bundle_failed")
        raise RuntimeError("injected live bundle mismatch")

    monkeypatch.setattr(
        proposal_builder,
        "verify_default_smoke_authority_proposal_candidate",
        fail_live_bundle,
    )
    with pytest.raises(RuntimeError, match="live bundle mismatch"):
        proposal_builder.main()
    assert events == ["live_bundle_failed"]
    assert not proposal_path.exists()
    assert not manifest_path.exists()


def test_durable_claim_uses_exclusive_create_and_is_permanent(
    tmp_path: Path,
) -> None:
    _proposal_payload, _manifest, _authority_payload, claim = _authority(tmp_path)
    claim_path = tmp_path / "claim.json"

    create_durable_launch_claim(claim_path, claim)

    assert canonical_artifact_payload_hash(
        json.loads(claim_path.read_text(encoding="utf-8"))
    ) == canonical_artifact_payload_hash(claim)
    assert parse_launch_claim(claim) == claim
    assert claim["schema"] == HMC_PHASE6_SMOKE_LAUNCH_CLAIM_SCHEMA_V1
    assert claim["permanent_authority_consumption"] is True
    assert claim["file_mode"] == "0400"
    assert stat.S_IMODE(claim_path.stat().st_mode) == stat.S_IRUSR
    with pytest.raises(FileExistsError):
        create_durable_launch_claim(claim_path, claim)


def test_durable_claim_fsyncs_file_then_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _proposal_payload, _manifest, _authority_payload, claim = _authority(tmp_path)
    claim_path = tmp_path / "claim.json"
    calls: list[int] = []
    real_fsync = os.fsync

    def observed(fd: int) -> None:
        calls.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", observed)
    create_durable_launch_claim(claim_path, claim)
    assert len(calls) == 2


@pytest.mark.parametrize("failure_index", [1, 2])
def test_durable_claim_sync_failure_permanently_consumes_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_index: int,
) -> None:
    _proposal_payload, _manifest, _authority_payload, claim = _authority(tmp_path)
    claim_path = tmp_path / "claim.json"
    real_fsync = os.fsync
    calls = 0

    def injected(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == failure_index:
            raise OSError(f"injected fsync failure {failure_index}")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", injected)
    with pytest.raises(OSError, match="injected fsync failure"):
        create_durable_launch_claim(claim_path, claim)
    assert os.path.lexists(claim_path)
    with pytest.raises(FileExistsError):
        create_durable_launch_claim(claim_path, claim)
    assert os.path.lexists(claim_path)


def test_partial_or_malformed_existing_claim_still_consumes_authority(
    tmp_path: Path,
) -> None:
    _proposal_payload, _manifest, _authority_payload, claim = _authority(tmp_path)
    claim_path = tmp_path / "claim.json"
    claim_path.write_text("{", encoding="utf-8")
    with pytest.raises(FileExistsError):
        create_durable_launch_claim(claim_path, claim)


def test_consumed_evidence_failure_before_claim_creates_nothing(
    tmp_path: Path,
) -> None:
    _proposal_payload, _manifest, _authority_payload, claim = _authority(tmp_path)
    claim_path = tmp_path / "claim.json"
    directories = PinnedSmokeOutputDirectories.open(
        {"claim_path": claim_path},
        repo_root=tmp_path,
    )
    evidence = _TestConsumedEvidenceSession(fail_on_calls={1})
    try:
        with pytest.raises(RuntimeError, match="evidence drift"):
            create_durable_launch_claim_with_consumed_evidence(
                claim_path,
                claim,
                pinned_directories=directories,
                consumed_evidence_session=evidence,
            )
        assert not claim_path.exists()
    finally:
        directories.close()


def test_consumed_evidence_failure_after_claim_leaves_only_permanent_claim(
    tmp_path: Path,
) -> None:
    _proposal_payload, _manifest, _authority_payload, claim = _authority(tmp_path)
    claim_path = tmp_path / "claim.json"
    directories = PinnedSmokeOutputDirectories.open(
        {"claim_path": claim_path},
        repo_root=tmp_path,
    )
    evidence = _TestConsumedEvidenceSession(fail_on_calls={2})
    try:
        with pytest.raises(RuntimeError, match="evidence drift"):
            create_durable_launch_claim_with_consumed_evidence(
                claim_path,
                claim,
                pinned_directories=directories,
                consumed_evidence_session=evidence,
            )
        assert claim_path.is_file()
        assert stat.S_IMODE(claim_path.stat().st_mode) == 0o400
        with pytest.raises(FileExistsError):
            create_durable_launch_claim(claim_path, claim)
    finally:
        directories.close()


def test_consumed_evidence_failure_before_reservation_creates_no_outputs(
    tmp_path: Path,
) -> None:
    _proposal_payload, _manifest, _authority_payload, claim = _authority(tmp_path)
    paths = _secure_output_paths(tmp_path)
    directories = PinnedSmokeOutputDirectories.open(paths, repo_root=tmp_path)
    claim_fd = create_durable_launch_claim(
        paths["claim_path"],
        claim,
        pinned_directories=directories,
        keep_open=True,
    )
    assert claim_fd is not None
    evidence = _TestConsumedEvidenceSession(fail_on_calls={1})
    try:
        with pytest.raises(RuntimeError, match="evidence drift"):
            SecureSmokeOutputSession.reserve(
                directories=directories,
                claim_fd=claim_fd,
                consumed_evidence_session=evidence,
            )
        assert paths["claim_path"].is_file()
        for role, path in paths.items():
            if role != "claim_path":
                assert not path.exists()
    finally:
        os.close(claim_fd)
        directories.close()


def _secure_output_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        name: tmp_path / filename
        for name, filename in {
            "claim_path": "claim.json",
            "log_path": "smoke.log",
            "public_result_path": "result.json",
            "public_progress_path": "progress.json",
            "output_manifest_path": "smoke_output_manifest.json",
            "infrastructure_failure_path": "infrastructure_failure.json",
            "infrastructure_manifest_path": "infrastructure_manifest.json",
            "private_samples_path": "samples.npz",
        }.items()
    }


def _secure_session(
    tmp_path: Path,
) -> tuple[SecureSmokeOutputSession, dict[str, Path], dict]:
    _proposal_payload, _manifest, _authority_payload, claim = _authority(tmp_path)
    paths = _secure_output_paths(tmp_path)
    directories = PinnedSmokeOutputDirectories.open(paths, repo_root=tmp_path)
    claim_fd = create_durable_launch_claim(
        paths["claim_path"],
        claim,
        pinned_directories=directories,
        keep_open=True,
    )
    assert claim_fd is not None
    session = SecureSmokeOutputSession.reserve(
        directories=directories,
        claim_fd=claim_fd,
    )
    return session, paths, claim


def test_secure_session_retains_permanent_claim_readonly(tmp_path: Path) -> None:
    session, paths, claim = _secure_session(tmp_path)
    original = paths["claim_path"].read_bytes()
    try:
        with pytest.raises(RuntimeError, match="claim is read-only"):
            session.write_bytes("claim_path", b"replacement")
        with pytest.raises(RuntimeError, match="claim is read-only"):
            session.begin_binary_write("claim_path")
        with pytest.raises(OSError):
            os.write(session.fd("claim_path"), b"replacement")
        assert paths["claim_path"].read_bytes() == original
        with pytest.raises(FileExistsError):
            create_durable_launch_claim(paths["claim_path"], claim)
    finally:
        session.close()


def _launcher_context(
    tmp_path: Path,
    *,
    consumed_evidence_session: _TestConsumedEvidenceSession | None = None,
) -> tuple[Phase6SmokeLaunchContext, dict[str, Path], dict, dict, dict]:
    proposal, manifest, _proposal_path, manifest_path = _proposal(tmp_path)
    authority = build_smoke_authority(
        proposal_manifest_path=manifest_path,
        human_approval_statement=expected_smoke_approval_statement(
            manifest["artifact_hash"]
        ),
        human_approval_date="2026-07-11",
    )
    authority_path = tmp_path / "authority.json"
    atomic_write_json(authority_path, authority)
    claim = build_launch_claim(
        authority=authority,
        proposal_manifest=manifest,
        command=proposal["command"],
        paths=proposal["paths"],
        pid=1234,
    )
    paths = _secure_output_paths(tmp_path)
    directories = PinnedSmokeOutputDirectories.open(paths, repo_root=tmp_path)
    claim_fd = create_durable_launch_claim(
        paths["claim_path"],
        claim,
        pinned_directories=directories,
        keep_open=True,
    )
    assert claim_fd is not None
    context = Phase6SmokeLaunchContext(
        config=controller.DeterministicLGSSMPhase7Config.load(
            controller.DEFAULT_CONFIG_PATH
        ),
        preflight=json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8")),
        proposal=proposal,
        proposal_manifest=manifest,
        authority=authority,
        authority_reference=build_phase5_artifact_reference(
            authority_path, embedded_hash_rule="canonical_without_hash"
        ),
        claim=claim,
        paths=paths,
        command=tuple(proposal["command"]),
        implementation_source_bundle=_live_implementation_source_bundle(),
        output_directories=directories,
        claim_fd=claim_fd,
        consumed_evidence_session=(
            _TestConsumedEvidenceSession()
            if consumed_evidence_session is None
            else consumed_evidence_session
        ),
        output_session=None,
    )
    context = _issue_test_prepared_context(context)
    return context, paths, authority, manifest, claim


def _issue_test_prepared_context(
    context: Phase6SmokeLaunchContext,
) -> Phase6SmokeLaunchContext:
    token = object()
    snapshot_hash = smoke_authority_module._prepared_context_snapshot_hash(
        config=context.config,
        authority_reference=context.authority_reference,
        proposal=context.proposal,
        proposal_manifest=context.proposal_manifest,
        authority=context.authority,
        claim=context.claim,
        preflight=context.preflight,
        command=context.command,
        paths=context.paths,
        implementation_source_bundle=context.implementation_source_bundle,
    )
    issued = replace(
        context,
        prepared_snapshot_hash=snapshot_hash,
        _prepared_token=token,
    )
    smoke_authority_module._PREPARED_CONTEXT_TOKENS[id(issued)] = token
    smoke_authority_module._PREPARED_CONTEXT_EVIDENCE_SESSIONS[id(issued)] = (
        issued.consumed_evidence_session
    )
    return issued


def _write_valid_primary(
    *,
    context: Phase6SmokeLaunchContext,
    authority: dict,
    manifest: dict,
    claim: dict,
) -> dict:
    session = context.output_session
    assert session is not None
    private_bytes = b"private smoke samples"
    session.write_bytes("private_samples_path", private_bytes)
    primary = _smoke_result(authority, manifest, claim)
    primary["config_hash"] = context.config.hash
    source_bundle_hash = smoke_authority_module.implementation_source_bundle_hash(
        context.implementation_source_bundle
    )
    for metadata in primary["worker_metadata"]:
        metadata["child_implementation_source_bundle_hash"] = source_bundle_hash
    primary["private_retained_sample_reference"]["file_sha256"] = hashlib.sha256(
        private_bytes
    ).hexdigest()
    primary["private_retained_sample_reference"]["byte_count"] = len(private_bytes)
    primary = _rehash(primary)
    session.write_json("public_result_path", primary, parser=parse_smoke_result)
    session.write_json(
        "public_progress_path",
        _smoke_progress(primary, authority, manifest, claim),
        parser=parse_smoke_progress,
    )
    return primary


@pytest.mark.parametrize(
    "role",
    [
        "log_path",
        "public_progress_path",
        "public_result_path",
        "output_manifest_path",
        "private_samples_path",
    ],
)
def test_secure_output_creation_rejects_symlink_race_without_touching_target(
    tmp_path: Path,
    role: str,
) -> None:
    _proposal_payload, _manifest, _authority_payload, claim = _authority(tmp_path)
    paths = _secure_output_paths(tmp_path)
    directories = PinnedSmokeOutputDirectories.open(paths, repo_root=tmp_path)
    claim_fd = create_durable_launch_claim(
        paths["claim_path"],
        claim,
        pinned_directories=directories,
        keep_open=True,
    )
    assert claim_fd is not None
    protected = tmp_path / "protected"
    protected.write_bytes(b"untouched")
    paths[role].symlink_to(protected)
    try:
        with pytest.raises(Exception):
            SecureSmokeOutputSession.reserve(
                directories=directories,
                claim_fd=claim_fd,
            )
        assert protected.read_bytes() == b"untouched"
        assert paths[role].is_symlink()
        with pytest.raises(FileExistsError):
            create_durable_launch_claim(paths["claim_path"], claim)
    finally:
        directories.close()


@pytest.mark.parametrize(
    "role",
    [
        "log_path",
        "public_progress_path",
        "public_result_path",
        "output_manifest_path",
        "private_samples_path",
    ],
)
def test_secure_output_update_rejects_path_replacement_without_touching_target(
    tmp_path: Path,
    role: str,
) -> None:
    session, paths, claim = _secure_session(tmp_path)
    protected = tmp_path / "protected"
    protected.write_bytes(b"untouched")
    try:
        session.write_bytes(role, b"first")
        paths[role].unlink()
        paths[role].symlink_to(protected)
        with pytest.raises(RuntimeError, match="pinned file"):
            session.write_bytes(role, b"second")
        assert protected.read_bytes() == b"untouched"
        with pytest.raises(FileExistsError):
            create_durable_launch_claim(paths["claim_path"], claim)
    finally:
        session.close()


def test_pinned_output_parent_replacement_is_a_hard_veto(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    paths = {
        name: output_dir / path.name
        for name, path in _secure_output_paths(tmp_path).items()
    }
    directories = PinnedSmokeOutputDirectories.open(paths, repo_root=tmp_path)
    moved = tmp_path / "moved"
    output_dir.rename(moved)
    output_dir.mkdir()
    try:
        with pytest.raises(RuntimeError, match="parent identity changed"):
            directories.open_exclusive("claim_path")
        assert not paths["claim_path"].exists()
    finally:
        directories.close()


@pytest.mark.parametrize(
    "fault_stage",
    [
        "controller_runtime",
        "log_sealing",
        "output_manifest_construction",
        "output_manifest_write",
    ],
)
def test_post_claim_stage_failure_preserves_primary_and_seals_infrastructure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault_stage: str,
) -> None:
    context, paths, authority, manifest, claim = _launcher_context(tmp_path)
    primary: dict | None = None

    def mocked_run(_config, *, smoke, smoke_launch_context):
        nonlocal primary
        assert smoke is True
        primary = _write_valid_primary(
            context=smoke_launch_context,
            authority=authority,
            manifest=manifest,
            claim=claim,
        )
        if fault_stage == "controller_runtime":
            raise RuntimeError("controller boom")
        return primary

    monkeypatch.setattr(smoke_launcher, "run_phase7", mocked_run)
    monkeypatch.setattr(smoke_launcher, "_redirect_after_claim", lambda _session: None)
    if fault_stage == "log_sealing":
        monkeypatch.setattr(
            smoke_launcher,
            "_seal_log_before_manifest",
            lambda _session: (_ for _ in ()).throw(RuntimeError("log boom")),
        )
    else:
        monkeypatch.setattr(
            smoke_launcher, "_seal_log_before_manifest", lambda _session: None
        )
    if fault_stage == "output_manifest_construction":
        monkeypatch.setattr(
            smoke_launcher,
            "build_smoke_output_manifest",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("manifest boom")),
        )
    elif fault_stage == "output_manifest_write":
        original_write_json = SecureSmokeOutputSession.write_json

        def write_then_fail(self, role, payload, *, parser=None):
            restored = original_write_json(self, role, payload, parser=parser)
            if role == "output_manifest_path":
                raise OSError("output manifest fsync aftermath")
            return restored

        monkeypatch.setattr(SecureSmokeOutputSession, "write_json", write_then_fail)

    assert smoke_launcher._supervise_after_claim(context) == 2
    assert primary is not None
    restored_primary = json.loads(paths["public_result_path"].read_text(encoding="utf-8"))
    assert restored_primary == primary
    failure = json.loads(
        paths["infrastructure_failure_path"].read_text(encoding="utf-8")
    )
    infrastructure_manifest = json.loads(
        paths["infrastructure_manifest_path"].read_text(encoding="utf-8")
    )
    assert parse_smoke_infrastructure_failure(failure) == failure
    assert parse_smoke_infrastructure_manifest(infrastructure_manifest) == (
        infrastructure_manifest
    )
    assert failure["primary_result_preserved"] is True
    assert failure["primary_result_artifact_hash"] == primary["artifact_hash"]
    assert failure["stage"] == fault_stage
    assert infrastructure_manifest["public_result_nonempty"] is True
    assert infrastructure_manifest["public_result_reference"]["file_sha256"] == (
        hashlib.sha256(paths["public_result_path"].read_bytes()).hexdigest()
    )
    assert infrastructure_manifest["output_manifest_nonempty"] is (
        fault_stage == "output_manifest_write"
    )


def test_post_claim_redirection_failure_is_strictly_classified_before_controller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, paths, _authority, _manifest, _claim = _launcher_context(tmp_path)
    monkeypatch.setattr(
        smoke_launcher,
        "_redirect_after_claim",
        lambda _session: (_ for _ in ()).throw(OSError("redirection boom")),
    )
    monkeypatch.setattr(
        smoke_launcher,
        "run_phase7",
        lambda *_args, **_kwargs: pytest.fail("controller must not run"),
    )

    assert smoke_launcher._supervise_after_claim(context) == 2
    failure = json.loads(paths["infrastructure_failure_path"].read_text())
    manifest = json.loads(paths["infrastructure_manifest_path"].read_text())
    assert parse_smoke_infrastructure_failure(failure) == failure
    assert parse_smoke_infrastructure_manifest(manifest) == manifest
    assert failure["stage"] == "log_redirection"
    assert failure["primary_result_preserved"] is False
    assert manifest["public_result_nonempty"] is False


def test_post_claim_failure_flushes_redirected_log_before_manifest_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, paths, _authority, _manifest, _claim = _launcher_context(tmp_path)
    original_redirect = smoke_launcher._redirect_after_claim

    def redirect_and_buffer(session):
        original_redirect(session)
        print("buffered stdout evidence", end="")
        sys.stderr.write("buffered stderr evidence")

    monkeypatch.setattr(smoke_launcher, "_redirect_after_claim", redirect_and_buffer)
    monkeypatch.setattr(
        smoke_launcher,
        "run_phase7",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("runtime boom")),
    )

    saved_stdout = os.dup(sys.stdout.fileno())
    saved_stderr = os.dup(sys.stderr.fileno())
    try:
        assert smoke_launcher._supervise_after_claim(context) == 2
        sys.stdout.write("post-manifest stdout")
        sys.stdout.flush()
        sys.stderr.write("post-manifest stderr")
        sys.stderr.flush()
    finally:
        os.dup2(saved_stdout, sys.stdout.fileno())
        os.dup2(saved_stderr, sys.stderr.fileno())
        os.close(saved_stdout)
        os.close(saved_stderr)
    log_bytes = paths["log_path"].read_bytes()
    manifest = json.loads(paths["infrastructure_manifest_path"].read_text())
    assert b"buffered stdout evidence" in log_bytes
    assert b"buffered stderr evidence" in log_bytes
    assert b"post-manifest" not in log_bytes
    assert manifest["log_reference"]["file_sha256"] == hashlib.sha256(
        log_bytes
    ).hexdigest()


def test_log_stabilization_ignores_unredirected_python_streams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _paths, _claim = _secure_session(tmp_path)
    stdout = StringIO()
    stderr = StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    try:
        smoke_launcher._stabilize_redirected_log_best_effort(session)
        assert session.file_reference("log_path")["byte_count"] == 0
    finally:
        session.close()


def test_post_claim_reservation_failure_keeps_partial_session_for_classification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, paths, _authority, _manifest, _claim = _launcher_context(tmp_path)
    original = context.output_directories.open_exclusive

    def injected(role: str) -> int:
        if role == "log_path":
            raise OSError("reservation boom")
        return original(role)

    monkeypatch.setattr(context.output_directories, "open_exclusive", injected)
    monkeypatch.setattr(
        smoke_launcher,
        "run_phase7",
        lambda *_args, **_kwargs: pytest.fail("controller must not run"),
    )

    assert smoke_launcher._supervise_after_claim(context) == 2
    failure = json.loads(paths["infrastructure_failure_path"].read_text())
    manifest = json.loads(paths["infrastructure_manifest_path"].read_text())
    assert parse_smoke_infrastructure_failure(failure) == failure
    assert parse_smoke_infrastructure_manifest(manifest) == manifest
    assert failure["stage"] == "secure_output_reservation:log_path"
    assert failure["reason"] == "infrastructure_error:OSError"
    assert manifest["log_reserved"] is False
    assert manifest["public_result_reserved"] is True


def test_final_reservation_evidence_drift_writes_no_bytes_and_closes_fds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _TestConsumedEvidenceSession(fail_on_calls={9})
    context, paths, _authority, _manifest, _claim = _launcher_context(
        tmp_path,
        consumed_evidence_session=evidence,
    )
    opened_fds: list[int] = []
    parent_fds = list(context.output_directories._owned_fds)
    original = context.output_directories.open_exclusive

    def tracked_open(role: str) -> int:
        fd = original(role)
        opened_fds.append(fd)
        return fd

    monkeypatch.setattr(context.output_directories, "open_exclusive", tracked_open)
    monkeypatch.setattr(
        smoke_launcher,
        "run_phase7",
        lambda *_args, **_kwargs: pytest.fail("controller must not run"),
    )
    monkeypatch.setattr(
        smoke_launcher,
        "_seal_infrastructure_best_effort",
        lambda **_kwargs: pytest.fail("evidence drift must not be sealed"),
    )

    assert smoke_launcher._supervise_after_claim(context) == 2
    assert evidence.verify_calls == 9
    assert evidence.closed is True
    assert paths["claim_path"].stat().st_size > 0
    for role, path in paths.items():
        if role != "claim_path":
            assert path.is_file()
            assert path.stat().st_size == 0
    for fd in [context.claim_fd, *opened_fds, *parent_fds]:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_controller_evidence_drift_bypasses_all_terminal_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _TestConsumedEvidenceSession(fail_on_calls={12})
    context, paths, _authority, _manifest, _claim = _launcher_context(
        tmp_path,
        consumed_evidence_session=evidence,
    )
    worker_created = False

    def forbidden_worker(*_args, **_kwargs):
        nonlocal worker_created
        worker_created = True
        raise AssertionError("worker creation must remain unreachable")

    monkeypatch.setattr(smoke_launcher, "_redirect_after_claim", lambda _session: None)
    # The production path validator is covered separately; this temporary fixture
    # cannot satisfy its repository-relative path binding.
    monkeypatch.setattr(
        controller,
        "_validate_smoke_launch_context",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        smoke_launcher,
        "_seal_infrastructure_best_effort",
        lambda **_kwargs: pytest.fail("evidence drift must not be sealed"),
    )
    monkeypatch.setattr(
        controller.concurrent.futures,
        "ProcessPoolExecutor",
        forbidden_worker,
    )

    assert smoke_launcher._supervise_after_claim(context) == 2
    assert evidence.verify_calls == 12
    assert evidence.closed is True
    assert worker_created is False
    assert paths["claim_path"].stat().st_size > 0
    for role, path in paths.items():
        if role != "claim_path":
            assert path.is_file()
            assert path.stat().st_size == 0


def test_post_progress_evidence_drift_preserves_only_prior_progress_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _TestConsumedEvidenceSession()
    context, paths, _authority, _manifest, _claim = _launcher_context(
        tmp_path,
        consumed_evidence_session=evidence,
    )
    worker_created = False
    progress_write_completed = False
    original_write_runtime_json = controller._write_runtime_json

    def arm_drift_after_progress(*args, **kwargs) -> None:
        nonlocal progress_write_completed
        original_write_runtime_json(*args, **kwargs)
        if not progress_write_completed:
            progress_write_completed = True
            evidence.fail_on_calls.add(evidence.verify_calls + 1)

    def forbidden_worker(*_args, **_kwargs):
        nonlocal worker_created
        worker_created = True
        raise AssertionError("worker creation must remain unreachable")

    monkeypatch.setattr(smoke_launcher, "_redirect_after_claim", lambda _session: None)
    # The production path validator is covered separately; this temporary fixture
    # cannot satisfy its repository-relative path binding.
    monkeypatch.setattr(
        controller,
        "_validate_smoke_launch_context",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(controller, "_write_runtime_json", arm_drift_after_progress)
    monkeypatch.setattr(
        smoke_launcher,
        "_seal_infrastructure_best_effort",
        lambda **_kwargs: pytest.fail("evidence drift must not be sealed"),
    )
    monkeypatch.setattr(
        controller.concurrent.futures,
        "ProcessPoolExecutor",
        forbidden_worker,
    )

    assert smoke_launcher._supervise_after_claim(context) == 2
    assert progress_write_completed is True
    assert evidence.closed is True
    assert worker_created is False
    assert paths["claim_path"].stat().st_size > 0
    progress = json.loads(paths["public_progress_path"].read_text(encoding="utf-8"))
    assert progress["status"] == "preflight_passed"
    assert progress["completed"] is False
    for role, path in paths.items():
        if role not in {"claim_path", "public_progress_path"}:
            assert path.is_file()
            assert path.stat().st_size == 0


def test_post_claim_reservation_keyboard_interrupt_is_sealed_then_reraised(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, paths, _authority, _manifest, _claim = _launcher_context(tmp_path)
    original = context.output_directories.open_exclusive

    def injected(role: str) -> int:
        if role == "log_path":
            raise KeyboardInterrupt("reservation interrupted")
        return original(role)

    monkeypatch.setattr(context.output_directories, "open_exclusive", injected)
    with pytest.raises(KeyboardInterrupt, match="reservation interrupted"):
        smoke_launcher._supervise_after_claim(context)
    failure = json.loads(paths["infrastructure_failure_path"].read_text())
    manifest = json.loads(paths["infrastructure_manifest_path"].read_text())
    assert parse_smoke_infrastructure_failure(failure) == failure
    assert parse_smoke_infrastructure_manifest(manifest) == manifest
    assert failure["stage"] == "secure_output_reservation:log_path"
    assert failure["reason"] == "infrastructure_error:KeyboardInterrupt"


def test_later_reservation_control_flow_supersedes_earlier_ordinary_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, paths, _authority, _manifest, _claim = _launcher_context(tmp_path)
    original = context.output_directories.open_exclusive

    def injected(role: str) -> int:
        if role == "output_manifest_path":
            raise OSError("earlier ordinary failure")
        if role == "log_path":
            raise KeyboardInterrupt("later control-flow failure")
        return original(role)

    monkeypatch.setattr(context.output_directories, "open_exclusive", injected)
    with pytest.raises(KeyboardInterrupt, match="later control-flow failure"):
        smoke_launcher._supervise_after_claim(context)
    failure = json.loads(paths["infrastructure_failure_path"].read_text())
    manifest = json.loads(paths["infrastructure_manifest_path"].read_text())
    assert parse_smoke_infrastructure_failure(failure) == failure
    assert parse_smoke_infrastructure_manifest(manifest) == manifest
    assert failure["stage"] == "secure_output_reservation:log_path"
    assert failure["reason"] == "infrastructure_error:KeyboardInterrupt"
    assert manifest["output_manifest_reserved"] is False
    assert manifest["log_reserved"] is False


@pytest.mark.parametrize(
    "fault_role",
    ["infrastructure_failure_path", "infrastructure_manifest_path"],
)
def test_infrastructure_terminal_retries_complete_bytes_after_one_fsync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault_role: str,
) -> None:
    context, paths, _authority, _manifest, _claim = _launcher_context(tmp_path)
    session = SecureSmokeOutputSession.reserve(
        directories=context.output_directories,
        claim_fd=context.claim_fd,
        consumed_evidence_session=context.consumed_evidence_session,
    )
    target_fd = session.fds[fault_role]
    real_fsync = os.fsync
    injected_count = 0

    def injected(fd: int) -> None:
        nonlocal injected_count
        if fd == target_fd and injected_count == 0:
            injected_count += 1
            raise OSError("one-shot emergency fsync failure")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", injected)
    try:
        first = write_smoke_infrastructure_terminal(
            context=context,
            session=session,
            stage="controller_runtime",
            error=RuntimeError("primary boom"),
        )
        second = write_smoke_infrastructure_terminal(
            context=context,
            session=session,
            stage="controller_runtime",
            error=RuntimeError("primary boom"),
        )
        failure = session.read_json("infrastructure_failure_path")
        manifest = session.read_json("infrastructure_manifest_path")
        assert injected_count == 1
        assert parse_smoke_infrastructure_failure(failure) == failure
        assert parse_smoke_infrastructure_manifest(manifest) == manifest
        assert first == second == manifest
    finally:
        session.close()
    assert paths[fault_role].stat().st_size > 0


def test_infrastructure_terminal_retries_one_shot_manifest_construction_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, _paths, _authority, _manifest, _claim = _launcher_context(tmp_path)
    session = SecureSmokeOutputSession.reserve(
        directories=context.output_directories,
        claim_fd=context.claim_fd,
    )
    calls = 0

    def injected(*, context, session):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("one-shot manifest construction failure")
        return build_smoke_infrastructure_manifest(context=context, session=session)

    monkeypatch.setattr(
        "bayesfilter.inference.hmc_smoke_authority."
        "build_smoke_infrastructure_manifest",
        injected,
    )
    try:
        manifest = write_smoke_infrastructure_terminal(
            context=context,
            session=session,
            stage="controller_runtime",
            error=RuntimeError("primary boom"),
        )
        assert calls == 2
        assert parse_smoke_infrastructure_manifest(manifest) == manifest
    finally:
        session.close()


def test_infrastructure_terminal_retries_one_shot_failure_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, _paths, _authority, _manifest, _claim = _launcher_context(tmp_path)
    session = SecureSmokeOutputSession.reserve(
        directories=context.output_directories,
        claim_fd=context.claim_fd,
    )
    calls = 0
    original = smoke_authority_module.build_smoke_infrastructure_failure

    def injected(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("one-shot failure construction failure")
        return original(**kwargs)

    monkeypatch.setattr(
        smoke_authority_module,
        "build_smoke_infrastructure_failure",
        injected,
    )
    try:
        manifest = write_smoke_infrastructure_terminal(
            context=context,
            session=session,
            stage="controller_runtime",
            error=RuntimeError("primary boom"),
        )
        assert calls == 2
        assert parse_smoke_infrastructure_failure(
            session.read_json("infrastructure_failure_path")
        )
        assert parse_smoke_infrastructure_manifest(manifest) == manifest
    finally:
        session.close()


def test_infrastructure_terminal_retries_one_shot_failure_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, _paths, _authority, _manifest, _claim = _launcher_context(tmp_path)
    session = SecureSmokeOutputSession.reserve(
        directories=context.output_directories,
        claim_fd=context.claim_fd,
    )
    calls = 0
    original = SecureSmokeOutputSession.write_json

    def injected(self, role, payload, *, parser=None):
        nonlocal calls
        if role == "infrastructure_failure_path" and calls == 0:
            calls += 1
            raise OSError("one-shot failure write failure")
        return original(self, role, payload, parser=parser)

    monkeypatch.setattr(SecureSmokeOutputSession, "write_json", injected)
    try:
        manifest = write_smoke_infrastructure_terminal(
            context=context,
            session=session,
            stage="controller_runtime",
            error=RuntimeError("primary boom"),
        )
        assert calls == 1
        assert parse_smoke_infrastructure_failure(
            session.read_json("infrastructure_failure_path")
        )
        assert parse_smoke_infrastructure_manifest(manifest) == manifest
    finally:
        session.close()


@pytest.mark.parametrize(
    ("role", "prefix"),
    [("claim_path", "claim"), ("log_path", "log")],
)
def test_infrastructure_manifest_preserves_held_bytes_after_path_replacement(
    tmp_path: Path,
    role: str,
    prefix: str,
) -> None:
    context, paths, _authority, _proposal_manifest, _claim = _launcher_context(
        tmp_path
    )
    session = SecureSmokeOutputSession.reserve(
        directories=context.output_directories,
        claim_fd=context.claim_fd,
    )
    if role == "log_path":
        session.write_bytes(role, b"held log evidence")
    held_reference = (
        session.artifact_reference(role, require_path_match=False)
        if role == "claim_path"
        else session.file_reference(role, require_path_match=False)
    )
    protected = tmp_path / f"protected-{prefix}"
    protected.write_bytes(b"untouched replacement target")
    paths[role].unlink()
    paths[role].symlink_to(protected)
    try:
        manifest = write_smoke_infrastructure_terminal(
            context=context,
            session=session,
            stage="path_identity_check",
            error=RuntimeError("reviewed pathname replaced"),
        )
        assert manifest[f"{prefix}_path_intact"] is False
        assert manifest[f"{prefix}_reference"] == held_reference
        assert protected.read_bytes() == b"untouched replacement target"
        assert verify_smoke_infrastructure_manifest(
            manifest,
            authority_path=tmp_path / "authority.json",
            claim_path=paths["claim_path"],
            infrastructure_failure_path=paths["infrastructure_failure_path"],
            public_result_path=paths["public_result_path"],
            public_progress_path=paths["public_progress_path"],
            output_manifest_path=paths["output_manifest_path"],
            log_path=paths["log_path"],
            private_samples_path=paths["private_samples_path"],
        ) == manifest
    finally:
        session.close()


def test_secondary_infrastructure_sealing_failure_never_masks_control_flow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, _paths, _authority, _manifest, _claim = _launcher_context(tmp_path)
    monkeypatch.setattr(smoke_launcher, "_redirect_after_claim", lambda _session: None)
    monkeypatch.setattr(
        smoke_launcher,
        "run_phase7",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            KeyboardInterrupt("original controller interrupt")
        ),
    )
    monkeypatch.setattr(
        smoke_launcher,
        "write_smoke_infrastructure_terminal",
        lambda **_kwargs: (_ for _ in ()).throw(SystemExit(99)),
    )

    with pytest.raises(KeyboardInterrupt, match="original controller interrupt"):
        smoke_launcher._supervise_after_claim(context)


def test_control_flow_during_sealing_supersedes_ordinary_launcher_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, _paths, _authority, _manifest, _claim = _launcher_context(tmp_path)
    monkeypatch.setattr(
        smoke_launcher,
        "_redirect_after_claim",
        lambda _session: (_ for _ in ()).throw(RuntimeError("ordinary failure")),
    )
    monkeypatch.setattr(
        smoke_launcher,
        "write_smoke_infrastructure_terminal",
        lambda **_kwargs: (_ for _ in ()).throw(
            KeyboardInterrupt("interrupt during sealing")
        ),
    )

    with pytest.raises(KeyboardInterrupt, match="interrupt during sealing"):
        smoke_launcher._supervise_after_claim(context)


def test_log_stabilization_failure_never_masks_original_control_flow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, _paths, _authority, _manifest, _claim = _launcher_context(tmp_path)
    monkeypatch.setattr(smoke_launcher, "_redirect_after_claim", lambda _session: None)
    monkeypatch.setattr(
        smoke_launcher,
        "run_phase7",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            KeyboardInterrupt("original controller interrupt")
        ),
    )
    monkeypatch.setattr(
        smoke_launcher,
        "_stabilize_redirected_log_best_effort",
        lambda _session: (_ for _ in ()).throw(SystemExit(88)),
    )

    with pytest.raises(KeyboardInterrupt, match="original controller interrupt"):
        smoke_launcher._supervise_after_claim(context)


def test_infrastructure_failure_does_not_call_invalid_bytes_a_preserved_primary(
    tmp_path: Path,
) -> None:
    proposal, manifest, _proposal_path, manifest_path = _proposal(tmp_path)
    approval = expected_smoke_approval_statement(manifest["artifact_hash"])
    authority = build_smoke_authority(
        proposal_manifest_path=manifest_path,
        human_approval_statement=approval,
        human_approval_date="2026-07-11",
    )
    authority_path = tmp_path / "authority.json"
    atomic_write_json(authority_path, authority)
    claim = build_launch_claim(
        authority=authority,
        proposal_manifest=manifest,
        command=proposal["command"],
        paths=proposal["paths"],
        pid=1234,
    )
    paths = _secure_output_paths(tmp_path)
    directories = PinnedSmokeOutputDirectories.open(paths, repo_root=tmp_path)
    claim_fd = create_durable_launch_claim(
        paths["claim_path"], claim, pinned_directories=directories, keep_open=True
    )
    assert claim_fd is not None
    session = SecureSmokeOutputSession.reserve(
        directories=directories, claim_fd=claim_fd
    )
    context = Phase6SmokeLaunchContext(
        config=controller.DeterministicLGSSMPhase7Config.load(
            controller.DEFAULT_CONFIG_PATH
        ),
        preflight=json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8")),
        proposal=proposal,
        proposal_manifest=manifest,
        authority=authority,
        authority_reference=build_phase5_artifact_reference(
            authority_path, embedded_hash_rule="canonical_without_hash"
        ),
        claim=claim,
        paths=paths,
        command=tuple(proposal["command"]),
        implementation_source_bundle=_live_implementation_source_bundle(),
        output_directories=directories,
        claim_fd=claim_fd,
        consumed_evidence_session=_TestConsumedEvidenceSession(),
        output_session=session,
    )
    context = _issue_test_prepared_context(context)
    try:
        session.write_bytes("public_result_path", b"not json")
        write_smoke_infrastructure_terminal(
            context=context,
            session=session,
            stage="controller_runtime",
            error=RuntimeError("boom"),
        )
        failure = session.read_json("infrastructure_failure_path")
        infrastructure_manifest = session.read_json("infrastructure_manifest_path")
        assert failure["primary_result_preserved"] is False
        assert failure["primary_result_artifact_hash"] is None
        assert infrastructure_manifest["public_result_nonempty"] is True
        assert infrastructure_manifest["public_result_reference"]["byte_count"] == 8
    finally:
        session.close()


def test_file_reference_detects_compatible_copy(tmp_path: Path) -> None:
    original = tmp_path / "original.py"
    copied = tmp_path / "copied.py"
    original.write_text("VALUE = 1\n", encoding="utf-8")
    copied.write_bytes(original.read_bytes())
    reference = build_file_reference(original)
    verify_file_reference(reference, path=original)
    with pytest.raises(ValueError, match="current bytes"):
        verify_file_reference(reference, path=copied)


def _fake_launch_context(tmp_path: Path) -> SimpleNamespace:
    proposal, manifest, authority, claim = _authority(tmp_path)
    config = controller.DeterministicLGSSMPhase7Config.load(controller.DEFAULT_CONFIG_PATH)
    claim_path = tmp_path / "claim.json"
    atomic_write_json(claim_path, claim)
    paths = {
        "claim_path": claim_path,
        "log_path": tmp_path / "smoke.log",
        "public_result_path": tmp_path / "result.json",
        "public_progress_path": tmp_path / "progress.json",
        "private_samples_path": tmp_path / "samples.npz",
    }
    proposal = copy.deepcopy(proposal)
    proposal["paths"] = {name: str(path.relative_to(ROOT)) if ROOT in path.parents else str(path) for name, path in paths.items()}
    # The context validator is tested independently from repository path policy.
    return SimpleNamespace(
        config=config,
        preflight=json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8")),
        proposal=proposal,
        proposal_manifest=manifest,
        authority=authority,
        claim=claim,
        paths=paths,
        command=tuple(proposal["command"]),
    )


def test_smoke_specific_terminal_semantics_do_not_reuse_serious_claims() -> None:
    cfg = controller.DeterministicLGSSMPhase7Config.load(controller.DEFAULT_CONFIG_PATH)
    context = SimpleNamespace(
        authority={"artifact_hash": "sha256:" + "a" * 64},
        claim={"artifact_hash": "sha256:" + "b" * 64},
        proposal_manifest={"artifact_hash": "sha256:" + "c" * 64},
    )
    assert controller._pass_decision(context) == SMOKE_PASS_DECISION
    assert controller._block_decision(context) == SMOKE_BLOCK_DECISION
    assert controller._result_nonclaims(cfg, context) == SMOKE_NONCLAIMS
    assert controller._failure_nonclaims(cfg, context) == SMOKE_FAILURE_NONCLAIMS
    assert "PASS_PHASE7_TO_PHASE8_APPROVAL_BOUNDARY" != SMOKE_PASS_DECISION
    assert all("not Phase 7 smoke or serious execution" not in item for item in SMOKE_NONCLAIMS)


def test_v2_context_rejects_serious_mode_before_any_controller_output(
    tmp_path: Path,
) -> None:
    config = controller.DeterministicLGSSMPhase7Config.load(controller.DEFAULT_CONFIG_PATH)
    with pytest.raises(controller.DeterministicLGSSMPhase7Error, match="serious mode"):
        controller._validate_smoke_launch_context(
            config,
            smoke=False,
            context=SimpleNamespace(),
            output_override=None,
            progress_override=None,
            private_samples_override=None,
        )
    assert not tuple(tmp_path.iterdir())


def test_run_phase7_rejects_forged_smoke_context_before_output_or_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = controller.DeterministicLGSSMPhase7Config.load(controller.DEFAULT_CONFIG_PATH)
    worker_created = False

    def forbidden(*_args, **_kwargs):
        nonlocal worker_created
        worker_created = True
        raise AssertionError("worker creation must remain unreachable")

    monkeypatch.setattr(
        controller.concurrent.futures,
        "ProcessPoolExecutor",
        forbidden,
    )
    forged = SimpleNamespace(
        paths={
            "public_result_path": tmp_path / "result.json",
            "public_progress_path": tmp_path / "progress.json",
            "private_samples_path": tmp_path / "samples.npz",
        }
    )
    with pytest.raises(
        controller.DeterministicLGSSMPhase7Error,
        match="not prepared by the verified launcher",
    ):
        controller.run_phase7(
            config,
            smoke=True,
            smoke_launch_context=forged,
        )
    assert worker_created is False
    assert not tuple(tmp_path.iterdir())


def test_prepared_context_capability_is_consumed_on_session_attachment(
    tmp_path: Path,
) -> None:
    context, _paths, _authority, _manifest, _claim = _launcher_context(tmp_path)
    session = SecureSmokeOutputSession.reserve(
        directories=context.output_directories,
        claim_fd=context.claim_fd,
        consumed_evidence_session=context.consumed_evidence_session,
    )
    try:
        attached = smoke_authority_module.attach_prepared_output_session(
            context,
            session,
        )
        with pytest.raises(ValueError, match="not issued"):
            smoke_authority_module.verify_prepared_smoke_launch_context(context)
        assert smoke_authority_module.verify_prepared_smoke_launch_context(
            attached,
            consume=True,
        ) is attached
        with pytest.raises(ValueError, match="not issued"):
            smoke_authority_module.verify_prepared_smoke_launch_context(attached)
    finally:
        session.close()


def test_child_tamper_rejects_before_transition_can_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = controller.DeterministicLGSSMPhase7Config.load(controller.DEFAULT_CONFIG_PATH)
    request = controller._worker_request(
        config,
        worker_index=0,
        action="initialize",
        count=0,
        seed=(1, 2),
        state=None,
        worker_env=controller._worker_environment(config),
        smoke=True,
        target_scope="bayesfilter_multidim_lower_triangular_lgssm_t120_hmc_2026_07_09",
        implementation_references=_live_implementation_references(),
        implementation_source_bundle=_live_implementation_source_bundle(),
    )
    live = controller.build_phase7_live_replay(config)
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(
        "bayesfilter.inference.hmc_smoke_authority."
        "verify_artifact_reference_snapshot",
        forbidden,
    )
    request["expected_transition_identity_hash"] = "sha256:" + "f" * 64
    with pytest.raises(controller.DeterministicLGSSMPhase7Error, match="transition"):
        controller._verify_child_live_identity(request, live)
    assert called is False


def test_child_governed_source_tamper_rejects_before_runner_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = controller.DeterministicLGSSMPhase7Config.load(controller.DEFAULT_CONFIG_PATH)
    request = controller._worker_request(
        config,
        worker_index=0,
        action="initialize",
        count=0,
        seed=(1, 2),
        state=None,
        worker_env=controller._worker_environment(config),
        smoke=True,
        target_scope="bayesfilter_multidim_lower_triangular_lgssm_t120_hmc_2026_07_09",
        implementation_references=_live_implementation_references(),
        implementation_source_bundle=_live_implementation_source_bundle(),
    )
    live = controller.build_phase7_live_replay(config)
    tampered_snapshots = dict(live.governed_source_snapshots)
    fixture = dict(tampered_snapshots["fixture"])
    fixture["file_sha256"] = "f" * 64
    tampered_snapshots["fixture"] = fixture
    tampered_live = replace(live, governed_source_snapshots=tampered_snapshots)
    runner_constructed = False

    def forbidden(*_args, **_kwargs):
        nonlocal runner_constructed
        runner_constructed = True

    monkeypatch.setattr(
        "bayesfilter.inference.build_fixed_size_hmc_chunk_runner",
        forbidden,
    )
    with pytest.raises(ValueError, match="snapshot reference mismatch"):
        controller._verify_child_live_identity(request, tampered_live)
    assert runner_constructed is False


def test_child_implementation_tamper_rejects_before_runner_construction() -> None:
    config = controller.DeterministicLGSSMPhase7Config.load(controller.DEFAULT_CONFIG_PATH)
    references = _live_implementation_references()
    role = "repository_file:bayesfilter/inference/hmc.py"
    references[role] = dict(references[role])
    references[role]["file_sha256"] = "f" * 64
    request = controller._worker_request(
        config,
        worker_index=0,
        action="initialize",
        count=0,
        seed=(1, 2),
        state=None,
        worker_env=controller._worker_environment(config),
        smoke=True,
        target_scope="bayesfilter_multidim_lower_triangular_lgssm_t120_hmc_2026_07_09",
        implementation_references=references,
        implementation_source_bundle=_live_implementation_source_bundle(),
    )
    with pytest.raises(ValueError, match="current bytes"):
        controller._verify_child_implementation_identity(request)


def test_later_smoke_worker_request_uses_initialized_identity_cache() -> None:
    config = controller.DeterministicLGSSMPhase7Config.load(controller.DEFAULT_CONFIG_PATH)
    request = controller._worker_request(
        config,
        worker_index=0,
        action="burnin",
        count=4,
        seed=(1, 2),
        state=None,
        worker_env=controller._worker_environment(config),
        smoke=True,
        target_scope="bayesfilter_multidim_lower_triangular_lgssm_t120_hmc_2026_07_09",
    )
    assert "implementation_references" not in request
    original_cache = dict(controller._WORKER_CACHE)
    try:
        controller._WORKER_CACHE.clear()
        with pytest.raises(
            controller.DeterministicLGSSMPhase7Error,
            match="cached implementation evidence",
        ):
            controller._verify_child_implementation_identity(request)
        cached = {
            "child_implementation_references_verified": True,
            "child_loaded_source_bytes_verified": True,
            "child_implementation_source_bundle_hash": "sha256:" + "a" * 64,
            "child_transition_identity_hash": TRANSITION_IDENTITY_HASH,
        }
        controller._WORKER_CACHE["child_identity"] = cached
        seal_payload = {
            "schema": controller.SECURE_WORKER_CACHE_SEAL_SCHEMA,
            "config_hash": config.hash,
            "authority_kind": "phase6_smoke",
            "authority_artifact_hash": "sha256:" + "1" * 64,
            "claim_artifact_hash": "sha256:" + "2" * 64,
            "proposal_manifest_artifact_hash": "sha256:" + "3" * 64,
            "worker_index": 0,
            "smoke": True,
            "target_scope": request["target_scope"],
            "transition_identity_hash": TRANSITION_IDENTITY_HASH,
            "implementation_source_bundle_hash": "sha256:" + "a" * 64,
            "chains_per_worker": 2,
            "total_chain_count": 4,
        }
        seal = {
            **seal_payload,
            "artifact_hash": canonical_artifact_payload_hash(seal_payload),
        }
        request["worker_cache_seal"] = seal
        controller._WORKER_CACHE["worker_cache_seal"] = seal
        assert controller._verify_secure_worker_cache_seal(request) == seal
        assert controller._verify_child_implementation_identity(request) == cached
    finally:
        controller._WORKER_CACHE.clear()
        controller._WORKER_CACHE.update(original_cache)


def test_run_phase7_rejects_smoke_context_overrides_before_writes(
    tmp_path: Path,
) -> None:
    config = controller.DeterministicLGSSMPhase7Config.load(controller.DEFAULT_CONFIG_PATH)
    with pytest.raises(
        controller.DeterministicLGSSMPhase7Error,
        match="output overrides",
    ):
        controller.run_phase7(
            config,
            smoke=True,
            output_override=tmp_path / "result.json",
            smoke_launch_context=SimpleNamespace(),
        )
    assert not tuple(tmp_path.iterdir())


def test_worker_request_binds_all_v2_sources_and_transition() -> None:
    config = controller.DeterministicLGSSMPhase7Config.load(controller.DEFAULT_CONFIG_PATH)
    request = controller._worker_request(
        config,
        worker_index=0,
        action="initialize",
        count=0,
        seed=(1, 2),
        state=None,
        worker_env=controller._worker_environment(config),
        smoke=True,
        target_scope="bayesfilter_multidim_lower_triangular_lgssm_t120_hmc_2026_07_09",
        implementation_references=_live_implementation_references(),
        implementation_source_bundle=_live_implementation_source_bundle(),
    )
    assert set(request["governed_source_paths"]) == {
        "fixture",
        "xla_compile",
        "geometry",
        "mass",
        "kernel",
        "private_replay",
        "source_tuning_config",
        "historical_v1_config",
        "source_contract",
    }
    assert request["expected_transition_identity_hash"] == TRANSITION_IDENTITY_HASH


def test_child_live_identity_passes_without_taking_transition() -> None:
    config = controller.DeterministicLGSSMPhase7Config.load(controller.DEFAULT_CONFIG_PATH)
    request = controller._worker_request(
        config,
        worker_index=0,
        action="initialize",
        count=0,
        seed=(1, 2),
        state=None,
        worker_env=controller._worker_environment(config),
        smoke=True,
        target_scope="bayesfilter_multidim_lower_triangular_lgssm_t120_hmc_2026_07_09",
        implementation_references=_live_implementation_references(),
        implementation_source_bundle=_live_implementation_source_bundle(),
    )
    live = controller.build_phase7_live_replay(config)
    result = controller._verify_child_live_identity(request, live)
    assert set(live.governed_source_snapshots) == set(
        request["governed_source_paths"]
    )
    assert result == {
        "child_source_references_verified": True,
        "child_transition_identity_verified": True,
        "child_transition_identity_hash": TRANSITION_IDENTITY_HASH,
    }


def _smoke_diagnostics() -> dict:
    rows = [
        {
            "parameter": f"p{index}",
            "rank_normalized_split_rhat": 1.0,
            "folded_rank_normalized_split_rhat": 1.0,
            "rhat": 1.0,
            "bulk_ess": 8.0,
            "tail_ess": 8.0,
            "lower_tail_ess": 8.0,
            "upper_tail_ess": 8.0,
            "passed": True,
        }
        for index in range(18)
    ]
    return {
        "schema": "bayesfilter.rank_normalized_hmc_diagnostics.v1",
        "passed": True,
        "input_all_finite": True,
        "diagnostics_all_finite": True,
        "draw_count_per_chain": 8,
        "chain_count": 4,
        "parameter_count": 18,
        "split_draw_count_per_chain": 4,
        "split_chain_count": 8,
        "thresholds": {
            "rhat_max": 1.01,
            "bulk_ess_min": 1000.0,
            "tail_ess_min": 400.0,
        },
        "definitions": {
            "rank_transform": "Blom average-rank normal score",
            "rhat": "max rank and folded",
            "bulk_ess": "bulk",
            "tail_ess": "tail",
            "autocorrelation_truncation": "positive pairs",
            "quantile_interpolation": "linear",
        },
        "max_rhat": 1.0,
        "min_bulk_ess": 8.0,
        "min_tail_ess": 8.0,
        "parameter_diagnostics": rows,
        "hard_vetoes": [],
        "nonclaims": [
            "finite-only smoke engineering diagnostic screen",
            "R-hat and ESS values are explanatory only",
            "no posterior recovery or HMC convergence claim",
            "no sampler superiority, production, or default readiness claim",
        ],
        "smoke_gate": "finite_diagnostics_only_non_promoting",
    }


def _smoke_result(authority: dict, manifest: dict, claim: dict) -> dict:
    preflight = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    metadata = [
        {
            "jit_compile": True,
            "use_xla": True,
            "compile_trace_count": 1,
            "first_call_s": 1.0,
            "warm_call_s": 0.5,
            "tensorflow_version": "2.19.1",
            "tfp_version": "0.25.0",
            "python_version": "3.11.14",
            "cuda_visible_devices": "-1",
            "thread_environment": {
                "TF_NUM_INTRAOP_THREADS": "8",
                "TF_NUM_INTEROP_THREADS": "1",
                "OMP_NUM_THREADS": "8",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "CUDA_VISIBLE_DEVICES": "-1",
                "TF_CPP_MIN_LOG_LEVEL": "1",
                "MPLCONFIGDIR": "/tmp/matplotlib-bayesfilter-phase7-worker",
            },
            "child_source_references_verified": True,
            "child_implementation_references_verified": True,
            "child_loaded_source_bytes_verified": True,
            "child_implementation_source_bundle_hash": "sha256:" + "9" * 64,
            "child_transition_identity_verified": True,
            "child_transition_identity_hash": TRANSITION_IDENTITY_HASH,
        }
        for _ in range(2)
    ]
    payload = {
        "schema": HMC_PHASE6_SMOKE_RESULT_SCHEMA_V1,
        "passed": True,
        "decision": SMOKE_PASS_DECISION,
        "smoke": True,
        "smoke_authority_artifact_hash": authority["artifact_hash"],
        "smoke_launch_claim_artifact_hash": claim["artifact_hash"],
        "smoke_proposal_manifest_artifact_hash": manifest["artifact_hash"],
        "preflight_before_runtime_artifact_hash": preflight["artifact_hash"],
        "config_hash": "sha256:" + "1" * 64,
        "preflight_before_runtime": preflight,
        "burnin_results_per_chain": 4,
        "retained_results_per_chain": 8,
        "final_diagnostics": _smoke_diagnostics(),
        "worker_count": 2,
        "chains_per_worker": 2,
        "chain_count": 4,
        "worker_pids": [1001, 1002],
        "worker_metadata": metadata,
        "private_retained_sample_reference": {
            "file_sha256": "2" * 64,
            "byte_count": 10,
            "shape_verified": True,
            "finite_verified": True,
            "provenance_verified": True,
            "path_publicized": False,
            "raw_samples_publicized": False,
        },
        "jit_compile": True,
        "jit_compile_false_runtime_executed": False,
        "cuda_visible_devices": "-1",
        "elapsed_seconds": 1.5,
        "serious_runtime_executed": False,
        "neutra_executed": False,
        "phase8_executed": False,
        "nonclaims": list(SMOKE_NONCLAIMS),
    }
    return _rehash(payload)


def _smoke_progress(result: dict, authority: dict, manifest: dict, claim: dict) -> dict:
    preflight = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    check = lambda stage, count: {
        "stage": stage,
        "completed_results_per_chain": count,
        "passed": True,
        "max_rhat": 1.0,
        "min_bulk_ess": 8.0,
        "min_tail_ess": 8.0,
        "input_all_finite": True,
        "diagnostics_all_finite": True,
        "hard_vetoes": [],
    }
    return _rehash(
        {
            "schema": HMC_PHASE6_SMOKE_PROGRESS_SCHEMA_V1,
            "status": "result_written",
            "config_hash": result["config_hash"],
            "smoke": True,
            "smoke_authority_artifact_hash": authority["artifact_hash"],
            "smoke_launch_claim_artifact_hash": claim["artifact_hash"],
            "smoke_proposal_manifest_artifact_hash": manifest["artifact_hash"],
            "preflight_before_runtime_artifact_hash": preflight["artifact_hash"],
            "burnin_checks": [check("burnin", 4)],
            "retained_checks": [check("retained", 8)],
            "completed": True,
            "passed": True,
            "result_artifact_hash": result["artifact_hash"],
        }
    )


def test_smoke_result_and_progress_strict_round_trip(tmp_path: Path) -> None:
    _proposal_payload, manifest, authority, claim = _authority(tmp_path)
    result = _smoke_result(authority, manifest, claim)
    progress = _smoke_progress(result, authority, manifest, claim)
    assert parse_smoke_result(result) == result
    assert parse_smoke_progress(progress) == progress
    result["worker_metadata"][0]["adapter_signature"] = "private"
    with pytest.raises(ValueError, match="fields mismatch"):
        parse_smoke_result(_rehash(result))


def test_smoke_failure_uses_distinct_schema_and_nonclaims(tmp_path: Path) -> None:
    _proposal_payload, manifest, authority, claim = _authority(tmp_path)
    preflight = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    failure = _rehash(
        {
            "schema": HMC_PHASE6_SMOKE_FAILURE_SCHEMA_V1,
            "passed": False,
            "decision": SMOKE_BLOCK_DECISION,
            "smoke": True,
            "smoke_authority_artifact_hash": authority["artifact_hash"],
            "smoke_launch_claim_artifact_hash": claim["artifact_hash"],
            "smoke_proposal_manifest_artifact_hash": manifest["artifact_hash"],
            "preflight_before_runtime_artifact_hash": preflight["artifact_hash"],
            "stage": "worker_initialize",
            "reason": "runtime_error:Example",
            "config_hash": "sha256:" + "1" * 64,
            "preflight_before_runtime": preflight,
            "worker_pids": [],
            "final_diagnostics": None,
            "jit_compile_false_runtime_executed": False,
            "cuda_visible_devices": "-1",
            "elapsed_seconds": 1.0,
            "serious_runtime_executed": False,
            "neutra_executed": False,
            "phase8_executed": False,
            "nonclaims": list(SMOKE_FAILURE_NONCLAIMS),
        }
    )
    assert parse_smoke_failure(failure) == failure


def test_terminal_output_manifest_binds_exact_bytes(tmp_path: Path) -> None:
    proposal, manifest, authority, claim = _authority(tmp_path)
    proposal_path = tmp_path / "proposal.json"
    manifest_path = tmp_path / "manifest.json"
    authority_path = tmp_path / "authority.json"
    claim_path = tmp_path / "claim.json"
    result_path = tmp_path / "result.json"
    progress_path = tmp_path / "progress.json"
    log_path = tmp_path / "smoke.log"
    private_path = tmp_path / "samples.npz"
    infrastructure_failure_path = tmp_path / "infrastructure_failure.json"
    infrastructure_manifest_path = tmp_path / "infrastructure_manifest.json"
    for path, payload in (
        (proposal_path, proposal),
        (manifest_path, manifest),
        (authority_path, authority),
        (claim_path, claim),
    ):
        atomic_write_json(path, payload)
    result = _smoke_result(authority, manifest, claim)
    progress = _smoke_progress(result, authority, manifest, claim)
    atomic_write_json(result_path, result)
    atomic_write_json(progress_path, progress)
    log_path.write_text("bounded log\n", encoding="utf-8")
    private_path.write_bytes(b"private")
    infrastructure_failure_path.touch()
    infrastructure_manifest_path.touch()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["private_retained_sample_reference"]["file_sha256"] = (
        __import__("hashlib").sha256(private_path.read_bytes()).hexdigest()
    )
    result["private_retained_sample_reference"]["byte_count"] = private_path.stat().st_size
    atomic_write_json(result_path, _rehash(result))
    progress["result_artifact_hash"] = _rehash(result)["artifact_hash"]
    atomic_write_json(progress_path, _rehash(progress))
    output = build_smoke_output_manifest(
        proposal_path=proposal_path,
        proposal_manifest_path=manifest_path,
        authority_path=authority_path,
        claim_path=claim_path,
        progress_path=progress_path,
        result_path=result_path,
        log_path=log_path,
        private_samples_path=private_path,
        infrastructure_failure_path=infrastructure_failure_path,
        infrastructure_manifest_path=infrastructure_manifest_path,
    )
    assert parse_smoke_output_manifest(output) == output
    assert verify_smoke_output_manifest(
        output,
        proposal_path=proposal_path,
        proposal_manifest_path=manifest_path,
        authority_path=authority_path,
        claim_path=claim_path,
        progress_path=progress_path,
        result_path=result_path,
        log_path=log_path,
        private_samples_path=private_path,
        infrastructure_failure_path=infrastructure_failure_path,
        infrastructure_manifest_path=infrastructure_manifest_path,
    ) == output
    log_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="current bytes"):
        verify_smoke_output_manifest(
            output,
            proposal_path=proposal_path,
            proposal_manifest_path=manifest_path,
            authority_path=authority_path,
            claim_path=claim_path,
            progress_path=progress_path,
            result_path=result_path,
            log_path=log_path,
            private_samples_path=private_path,
            infrastructure_failure_path=infrastructure_failure_path,
            infrastructure_manifest_path=infrastructure_manifest_path,
        )


def test_failure_output_manifest_treats_empty_private_reservation_as_unavailable(
    tmp_path: Path,
) -> None:
    proposal, manifest, authority, claim = _authority(tmp_path)
    proposal_path = tmp_path / "proposal.json"
    manifest_path = tmp_path / "manifest.json"
    authority_path = tmp_path / "authority.json"
    claim_path = tmp_path / "claim.json"
    result_path = tmp_path / "result.json"
    progress_path = tmp_path / "progress.json"
    log_path = tmp_path / "smoke.log"
    private_path = tmp_path / "samples.npz"
    infrastructure_failure_path = tmp_path / "infrastructure_failure.json"
    infrastructure_manifest_path = tmp_path / "infrastructure_manifest.json"
    for path, payload in (
        (proposal_path, proposal),
        (manifest_path, manifest),
        (authority_path, authority),
        (claim_path, claim),
    ):
        atomic_write_json(path, payload)
    preflight = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    failure = _rehash(
        {
            "schema": HMC_PHASE6_SMOKE_FAILURE_SCHEMA_V1,
            "passed": False,
            "decision": SMOKE_BLOCK_DECISION,
            "smoke": True,
            "smoke_authority_artifact_hash": authority["artifact_hash"],
            "smoke_launch_claim_artifact_hash": claim["artifact_hash"],
            "smoke_proposal_manifest_artifact_hash": manifest["artifact_hash"],
            "preflight_before_runtime_artifact_hash": preflight["artifact_hash"],
            "stage": "preflight_passed",
            "reason": "runtime_error:BrokenProcessPool",
            "config_hash": "sha256:" + "1" * 64,
            "preflight_before_runtime": preflight,
            "worker_pids": [],
            "final_diagnostics": None,
            "jit_compile_false_runtime_executed": False,
            "cuda_visible_devices": "-1",
            "elapsed_seconds": 1.0,
            "serious_runtime_executed": False,
            "neutra_executed": False,
            "phase8_executed": False,
            "nonclaims": list(SMOKE_FAILURE_NONCLAIMS),
        }
    )
    progress = _rehash(
        {
            "schema": HMC_PHASE6_SMOKE_PROGRESS_SCHEMA_V1,
            "status": "blocked_result_written",
            "config_hash": failure["config_hash"],
            "smoke": True,
            "smoke_authority_artifact_hash": authority["artifact_hash"],
            "smoke_launch_claim_artifact_hash": claim["artifact_hash"],
            "smoke_proposal_manifest_artifact_hash": manifest["artifact_hash"],
            "preflight_before_runtime_artifact_hash": preflight["artifact_hash"],
            "burnin_checks": [],
            "retained_checks": [],
            "completed": True,
            "passed": False,
            "result_artifact_hash": failure["artifact_hash"],
        }
    )
    atomic_write_json(result_path, failure)
    atomic_write_json(progress_path, progress)
    log_path.write_text("bounded failure log\n", encoding="utf-8")
    private_path.touch()
    infrastructure_failure_path.touch()
    infrastructure_manifest_path.touch()
    output = build_smoke_output_manifest(
        proposal_path=proposal_path,
        proposal_manifest_path=manifest_path,
        authority_path=authority_path,
        claim_path=claim_path,
        progress_path=progress_path,
        result_path=result_path,
        log_path=log_path,
        private_samples_path=private_path,
        infrastructure_failure_path=infrastructure_failure_path,
        infrastructure_manifest_path=infrastructure_manifest_path,
    )
    assert output["private_samples_available"] is False
    assert output["private_samples_reference"] is None
    assert verify_smoke_output_manifest(
        output,
        proposal_path=proposal_path,
        proposal_manifest_path=manifest_path,
        authority_path=authority_path,
        claim_path=claim_path,
        progress_path=progress_path,
        result_path=result_path,
        log_path=log_path,
        private_samples_path=private_path,
        infrastructure_failure_path=infrastructure_failure_path,
        infrastructure_manifest_path=infrastructure_manifest_path,
    ) == output
