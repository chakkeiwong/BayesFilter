from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from bayesfilter.inference.hmc_identity import canonical_artifact_payload_hash
from bayesfilter.inference.hmc_identity_adoption import (
    ADOPTION_STATUS,
    APPROVED_CERTIFICATE_ARTIFACT_HASH,
    GOVERNED_SOURCE_KEYS,
    HMC_PHASE5_ADOPTION_RECORD_SCHEMA_V1,
    HMC_PHASE5_OUTPUT_MANIFEST_SCHEMA_V1,
    HMC_PHASE5_PREFLIGHT_REPORT_SCHEMA_V1,
    HUMAN_APPROVAL_STATEMENT,
    LEGACY_GATE_STATUS,
    PHASE7_CONFIG_SCHEMA_V2,
    PREFLIGHT_DECISION,
    build_phase5_artifact_reference,
    build_phase5_output_manifest,
    parse_phase5_adoption_record,
    parse_phase5_artifact_reference,
    parse_phase5_output_manifest,
    parse_phase5_preflight_report,
    parse_phase7_v2_config,
    verify_phase5_adoption_record,
    verify_phase5_artifact_reference,
    verify_phase5_output_manifest,
    verify_phase7_v2_sources,
)
from bayesfilter.runtime import atomic_write_json
from bayesfilter.testing.deterministic_lgssm_hmc_phase7_tf import (
    DEFAULT_CONFIG_PATH,
    HISTORICAL_V1_CONFIG_PATH,
    PHASE5_ADOPTION_RECORD_PATH,
    DeterministicLGSSMPhase7Config,
    DeterministicLGSSMPhase7Error,
    build_phase7_live_identity_bundle,
    phase7_governed_source_paths,
    run_phase7,
    validate_phase7_inputs,
    validate_phase7_v1_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / (
    "docs/benchmarks/artifacts/"
    "multidim_lgssm_serious_hmc_tuning_2026_07_09"
)
PUBLIC_ROOT = ROOT / (
    "docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11"
)
CERTIFICATE_PATH = ARTIFACT_ROOT / (
    "private_diagnostics/hmc_semantic_identity_migration_certificate.json"
)
PUBLIC_PROPOSAL_PATH = PUBLIC_ROOT / "migration_certificate_proposal.json"
PHASE4_MANIFEST_PATH = PUBLIC_ROOT / "migration_certificate_output_manifest.json"


def _read(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _rehash(payload: dict) -> dict:
    result = copy.deepcopy(payload)
    result.pop("artifact_hash", None)
    result["artifact_hash"] = canonical_artifact_payload_hash(result)
    return result


@pytest.fixture(scope="module")
def v2_config() -> DeterministicLGSSMPhase7Config:
    return DeterministicLGSSMPhase7Config.load(DEFAULT_CONFIG_PATH)


@pytest.fixture(scope="module")
def v2_payload() -> dict:
    return _read(DEFAULT_CONFIG_PATH)


@pytest.fixture(scope="module")
def adoption_record() -> dict:
    return _read(PHASE5_ADOPTION_RECORD_PATH)


def test_v2_config_and_terminal_adoption_record_round_trip(
    v2_payload: dict,
    adoption_record: dict,
) -> None:
    assert parse_phase7_v2_config(v2_payload) == v2_payload
    assert parse_phase5_adoption_record(adoption_record) == adoption_record
    assert v2_payload["schema"] == PHASE7_CONFIG_SCHEMA_V2
    assert v2_payload["runtime_authority"] is False
    assert v2_payload["baseline_adoption"]["human_approval_statement"] == (
        HUMAN_APPROVAL_STATEMENT
    )
    assert adoption_record["schema"] == HMC_PHASE5_ADOPTION_RECORD_SCHEMA_V1
    assert adoption_record["status"] == ADOPTION_STATUS
    assert adoption_record["runtime_authority"] is False
    assert adoption_record["certificate_reference"]["embedded_artifact_hash"] == (
        APPROVED_CERTIFICATE_ARTIFACT_HASH
    )
    assert "adoption_record" not in json.dumps(v2_payload, sort_keys=True)


def test_real_v2_preflight_reconstructs_all_identities_without_runtime(
    monkeypatch: pytest.MonkeyPatch,
    v2_config: DeterministicLGSSMPhase7Config,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    report = validate_phase7_inputs(v2_config)

    assert parse_phase5_preflight_report(report) == report
    assert report["schema"] == HMC_PHASE5_PREFLIGHT_REPORT_SCHEMA_V1
    assert report["decision"] == PREFLIGHT_DECISION
    assert all(report["identity_checks"].values())
    assert all(report["integrity_checks"].values())
    assert report["legacy_audit"] == {
        "status": LEGACY_GATE_STATUS,
        "public_final_kernel": "different",
        "private_loop_final_kernel": "different",
        "selected_trajectory": "different",
    }
    assert report["runtime_authority"] is False
    assert report["runtime_executed"] is False


def test_historical_v1_remains_immutable_and_exactly_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    config = DeterministicLGSSMPhase7Config.load(HISTORICAL_V1_CONFIG_PATH)
    assert build_phase5_artifact_reference(
        HISTORICAL_V1_CONFIG_PATH,
        embedded_hash_rule="none",
    )["file_sha256"] == (
        "746b001d3facb771b3b57b032a212683743187deb944e6fae8eb577af073c0b8"
    )
    with pytest.raises(
        DeterministicLGSSMPhase7Error,
        match="^public final kernel hash mismatch$",
    ):
        validate_phase7_v1_inputs(config)


def test_v2_run_refuses_before_output_or_worker_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    v2_config: DeterministicLGSSMPhase7Config,
) -> None:
    output = tmp_path / "result.json"
    progress = tmp_path / "progress.json"
    private = tmp_path / "samples.npz"

    def forbidden(*_args, **_kwargs):
        raise AssertionError("worker creation must not be reached")

    monkeypatch.setattr(
        "bayesfilter.testing.deterministic_lgssm_hmc_phase7_tf."
        "concurrent.futures.ProcessPoolExecutor",
        forbidden,
    )
    with pytest.raises(
        DeterministicLGSSMPhase7Error,
        match="runtime is not authorized",
    ):
        run_phase7(
            v2_config,
            output_override=output,
            progress_override=progress,
            private_samples_override=private,
        )
    assert not output.exists()
    assert not progress.exists()
    assert not private.exists()


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda value: value.update(runtime_authority=True),
            "cannot authorize runtime",
        ),
        (
            lambda value: value["baseline_adoption"].update(
                human_approval_statement="approved"
            ),
            "human approval mismatch",
        ),
        (
            lambda value: value["adopted_identities"].update(
                transition_identity_hash="sha256:" + "f" * 64
            ),
            None,
        ),
        (
            lambda value: value.update(unexpected=True),
            "fields mismatch",
        ),
    ],
)
def test_v2_parser_rejects_rehashed_boundary_or_schema_tamper(
    v2_payload: dict,
    mutation,
    match: str | None,
) -> None:
    tampered = copy.deepcopy(v2_payload)
    mutation(tampered)
    tampered = _rehash(tampered)
    if match is None:
        assert parse_phase7_v2_config(tampered) == tampered
        return
    with pytest.raises(ValueError, match=match):
        parse_phase7_v2_config(tampered)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["execution"].update(unknown_policy=True),
        lambda value: value["execution"]["thread_environment"].update(
            UNKNOWN_THREADS="1"
        ),
        lambda value: value["diagnostics"].update(proxy_promotion=True),
        lambda value: value["artifacts"].update(unreviewed_output="/tmp/output"),
    ],
)
def test_v2_parser_rejects_rehashed_unknown_nested_fields(
    v2_payload: dict,
    mutation,
) -> None:
    tampered = copy.deepcopy(v2_payload)
    mutation(tampered)
    with pytest.raises(ValueError, match="fields mismatch"):
        parse_phase7_v2_config(_rehash(tampered))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["execution"].update(worker_count="2"),
        lambda value: value["execution"].update(root_seed=[20260711, 702]),
        lambda value: value["execution"].update(jit_compile=False),
        lambda value: value["diagnostics"].update(rhat_max="1.01"),
        lambda value: value["burnin"].update(initial_results_per_chain=2001),
        lambda value: value["artifacts"].update(public_result="/tmp/result.json"),
    ],
)
def test_v2_parser_rejects_rehashed_type_or_reviewed_value_drift(
    v2_payload: dict,
    mutation,
) -> None:
    tampered = copy.deepcopy(v2_payload)
    mutation(tampered)
    with pytest.raises(ValueError, match="contract mismatch"):
        parse_phase7_v2_config(_rehash(tampered))


def test_live_preflight_rejects_rehashed_transition_baseline_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    v2_payload: dict,
    adoption_record: dict,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    payload = copy.deepcopy(v2_payload)
    payload["adopted_identities"]["transition_identity_hash"] = "sha256:" + "f" * 64
    payload = _rehash(payload)
    config_path = tmp_path / "v2.json"
    atomic_write_json(config_path, payload)
    config = DeterministicLGSSMPhase7Config(payload=payload, path=config_path)
    record = copy.deepcopy(adoption_record)
    record["v2_config_reference"] = build_phase5_artifact_reference(
        config_path,
        embedded_hash_rule="canonical_without_hash",
    )
    record["adopted_identities"] = dict(payload["adopted_identities"])
    record = _rehash(record)
    record_path = tmp_path / "adoption.json"
    atomic_write_json(record_path, record)

    with pytest.raises(
        ValueError,
        match="adopted identities do not match certificate",
    ):
        from bayesfilter.testing.deterministic_lgssm_hmc_phase7_tf import (
            validate_phase7_v2_inputs,
        )

        validate_phase7_v2_inputs(config, adoption_record_path=record_path)


def test_live_identity_ownership_domains_remain_separate(
    monkeypatch: pytest.MonkeyPatch,
    v2_config: DeterministicLGSSMPhase7Config,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    bundle = build_phase7_live_identity_bundle(v2_config)
    changed_transition = replace(
        bundle.transition,
        num_leapfrog_steps=bundle.transition.num_leapfrog_steps + 1,
    )
    changed_execution = replace(
        bundle.serious_execution,
        root_seed=(20260711, 702),
    )
    changed_provenance = replace(
        bundle.provenance,
        tuning_config_hash="sha256:" + "f" * 64,
    )

    assert changed_transition.identity_hash != bundle.transition.identity_hash
    assert changed_execution.identity_hash != bundle.serious_execution.identity_hash
    assert changed_provenance.identity_hash != bundle.provenance.identity_hash
    assert bundle.transition.identity_hash == v2_config.payload["adopted_identities"][
        "transition_identity_hash"
    ]


def test_source_reference_rejects_byte_tamper_and_compatible_copy(
    tmp_path: Path,
    v2_payload: dict,
) -> None:
    kernel_path = ARTIFACT_ROOT / "kernel_tuning.json"
    reference = v2_payload["governed_source_references"]["kernel"]
    assert verify_phase5_artifact_reference(reference, path=kernel_path) == reference

    tampered = tmp_path / "kernel.json"
    tampered.write_bytes(kernel_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="current bytes"):
        verify_phase5_artifact_reference(reference, path=tampered)

    copied_config = tmp_path / "v1.json"
    copied_config.write_bytes(HISTORICAL_V1_CONFIG_PATH.read_bytes())
    paths = dict(
        phase7_governed_source_paths(
            DeterministicLGSSMPhase7Config.load(DEFAULT_CONFIG_PATH)
        )
    )
    paths["historical_v1_config"] = copied_config
    with pytest.raises(ValueError, match="current bytes"):
        verify_phase7_v2_sources(v2_payload, source_paths=paths)


def test_adoption_record_rejects_reference_and_approval_tamper(
    adoption_record: dict,
) -> None:
    tampered = copy.deepcopy(adoption_record)
    tampered["human_approval_statement"] = "approved"
    with pytest.raises(ValueError, match="decision or approval mismatch"):
        parse_phase5_adoption_record(_rehash(tampered))

    reference = copy.deepcopy(adoption_record)
    reference["certificate_reference"]["embedded_artifact_hash"] = (
        "sha256:" + "f" * 64
    )
    with pytest.raises(ValueError, match="certificate mismatch"):
        parse_phase5_adoption_record(_rehash(reference))


def test_adoption_record_live_verification_and_acyclicity(
    adoption_record: dict,
    v2_payload: dict,
) -> None:
    assert verify_phase5_adoption_record(
        adoption_record,
        v2_config_path=DEFAULT_CONFIG_PATH,
        historical_v1_config_path=HISTORICAL_V1_CONFIG_PATH,
        certificate_path=CERTIFICATE_PATH,
        public_proposal_path=PUBLIC_PROPOSAL_PATH,
        phase4_output_manifest_path=PHASE4_MANIFEST_PATH,
    ) == adoption_record
    assert "adoption_record" not in json.dumps(v2_payload, sort_keys=True)
    assert "adoption_record_reference" not in json.dumps(v2_payload, sort_keys=True)


def test_phase5_terminal_manifest_detects_exact_byte_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    v2_config: DeterministicLGSSMPhase7Config,
    adoption_record: dict,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    config_path = tmp_path / "v2.json"
    adoption_path = tmp_path / "adoption.json"
    preflight_path = tmp_path / "preflight.json"
    atomic_write_json(config_path, v2_config.payload)

    record = copy.deepcopy(adoption_record)
    record["v2_config_reference"] = build_phase5_artifact_reference(
        config_path,
        embedded_hash_rule="canonical_without_hash",
    )
    atomic_write_json(adoption_path, _rehash(record))
    from bayesfilter.testing.deterministic_lgssm_hmc_phase7_tf import (
        validate_phase7_v2_inputs,
    )

    config = DeterministicLGSSMPhase7Config(
        payload=v2_config.payload,
        path=config_path,
    )
    report = validate_phase7_v2_inputs(
        config,
        adoption_record_path=adoption_path,
    )
    atomic_write_json(preflight_path, report)
    manifest = build_phase5_output_manifest(
        v2_config_path=config_path,
        adoption_record_path=adoption_path,
        preflight_report_path=preflight_path,
    )
    assert parse_phase5_output_manifest(manifest) == manifest
    assert manifest["schema"] == HMC_PHASE5_OUTPUT_MANIFEST_SCHEMA_V1
    assert verify_phase5_output_manifest(
        manifest,
        v2_config_path=config_path,
        adoption_record_path=adoption_path,
        preflight_report_path=preflight_path,
    ) == manifest
    preflight_path.write_bytes(preflight_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="current bytes"):
        verify_phase5_output_manifest(
            manifest,
            v2_config_path=config_path,
            adoption_record_path=adoption_path,
            preflight_report_path=preflight_path,
        )


def test_public_phase5_outputs_do_not_expose_private_mechanics(
    v2_payload: dict,
    adoption_record: dict,
) -> None:
    serialized = json.dumps(
        {"config": v2_payload, "adoption": adoption_record},
        sort_keys=True,
    )
    for key in (
        "observations",
        "transforms",
        "step_size",
        "num_leapfrog_steps",
        "stage_lineage",
        "reconstruction_links",
    ):
        assert f'"{key}":' not in serialized


def test_artifact_reference_parser_rejects_unknown_rule(v2_payload: dict) -> None:
    reference = copy.deepcopy(
        v2_payload["governed_source_references"]["historical_v1_config"]
    )
    reference["embedded_hash_rule"] = "ignore_fields"
    with pytest.raises(ValueError, match="unsupported embedded hash rule"):
        parse_phase5_artifact_reference(reference)
