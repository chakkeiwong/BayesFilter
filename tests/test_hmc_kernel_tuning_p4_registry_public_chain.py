from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import bayesfilter.inference.hmc_kernel_tuning as hmc_kernel_tuning
import bayesfilter.inference.hmc_warmup as hmc_warmup
from bayesfilter.inference import HMCKernelTuningConfig, tune_hmc_kernel
from tests.test_hmc_kernel_tuning_fixed_mass_step import (
    _ToyGaussianAdapter,
    _bootstrap,
    _geometry,
)


def test_public_p4_call_chain_uses_one_private_registry_without_transitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public endpoint carries one registry through every P4 boundary."""

    geometry = _geometry()
    bootstrap = _bootstrap()
    seen: dict[str, Any] = {}
    original_loop = hmc_kernel_tuning.run_hmc_tune_verify_repair_loop

    def bootstrap_runner(**kwargs: Any) -> Any:
        seen["bootstrap"] = kwargs.get("_g2_seed_use_registry")
        return bootstrap

    def windowed_runner(**kwargs: Any) -> Any:
        seen["windowed"] = kwargs.get("_g2_seed_use_registry")
        raise RuntimeError("zero-transition windowed boundary sentinel")

    def loop_runner(**kwargs: Any) -> Any:
        seen["loop"] = kwargs.get("_g2_seed_use_registry")
        assert isinstance(seen["loop"], hmc_warmup.G2PreboundarySeedUseRegistry)
        # Run the real Phase 7 dispatcher with its transition-producing stage
        # replaced by a sentinel. This proves the public-to-windowed hop while
        # keeping the test strictly zero-transition.
        return original_loop(
            **kwargs,
            _windowed_stage_runner=windowed_runner,
        )

    monkeypatch.setattr(hmc_kernel_tuning, "initialize_hmc_kernel_geometry", lambda **_: geometry)
    monkeypatch.setattr(hmc_kernel_tuning, "run_hmc_bootstrap_screen", bootstrap_runner)
    monkeypatch.setattr(hmc_kernel_tuning, "run_hmc_tune_verify_repair_loop", loop_runner)

    result = tune_hmc_kernel(
        adapter=_ToyGaussianAdapter(),
        initial_position=[0.0, 0.0],
        config=HMCKernelTuningConfig.smoke(
            target_scope="kernel_fixed_mass_step_toy_gaussian",
            engineering_probe_covariance_multiplier=2.0,
        ),
    )

    assert result.final_kernel_payload is None
    assert result.final_status in {"hard_veto", "budget_exhausted"}
    assert isinstance(seen["bootstrap"], hmc_warmup.G2PreboundarySeedUseRegistry)
    assert seen["bootstrap"] is seen["loop"]
    assert seen["loop"] is seen["windowed"]


def test_public_p4_retry_reuses_one_registry_without_hmc_runner_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Phase 7 retry must carry the original private registry object."""

    geometry = _geometry()
    bootstrap = _bootstrap()
    original_loop = hmc_kernel_tuning.run_hmc_tune_verify_repair_loop
    seen_registries: list[Any] = []
    hmc_runner_calls: list[tuple[Any, ...]] = []

    retry_state = hmc_kernel_tuning._HMCPhaseAttemptState(
        mass_artifact_payload={"schema": "test.phase4.mass.v1"},
        mass_artifact_signature="a" * 64,
        canonical_theta_state=[0.0, 0.0],
        private_start_bank_theta=[[0.0, 0.0]] * 4,
        private_start_bank_signature="b" * 64,
        handoff_stage="phase4",
    )

    class _RetryableWindowedStage:
        def __init__(self, config: Any, attempt_index: int) -> None:
            self.config = config
            self.final_status = "passed_no_metric_update"
            self.diagnostic_role = "metric_adaptation_not_observed"
            self.hard_vetoes = ()
            self.repair_triggers = ("windowed_mass_no_operational_metric_update",)
            self.attempt_index = int(attempt_index)
            self.passed = False
            self.artifact_hash = f"test-windowed-stage-{attempt_index}"

        def payload(self) -> dict[str, Any]:
            return {
                "schema": "test.windowed_stage.v1",
                "attempt_index": self.attempt_index,
                "final_status": self.final_status,
                "diagnostic_role": self.diagnostic_role,
            }

    def bootstrap_runner(**kwargs: Any) -> Any:
        assert isinstance(
            kwargs.get("_g2_seed_use_registry"),
            hmc_warmup.G2PreboundarySeedUseRegistry,
        )
        return bootstrap

    def windowed_runner(**kwargs: Any) -> Any:
        registry = kwargs.get("_g2_seed_use_registry")
        assert isinstance(registry, hmc_warmup.G2PreboundarySeedUseRegistry)
        seen_registries.append(registry)
        return _RetryableWindowedStage(kwargs["config"], len(seen_registries) - 1)

    def forbidden_hmc_runner(*args: Any, **kwargs: Any) -> Any:
        hmc_runner_calls.append(args)
        raise AssertionError("retry identity test must not invoke an HMC runner")

    def loop_runner(**kwargs: Any) -> Any:
        registry = kwargs.get("_g2_seed_use_registry")
        assert isinstance(registry, hmc_warmup.G2PreboundarySeedUseRegistry)
        kwargs["_windowed_stage_runner"] = windowed_runner
        kwargs["_fixed_mass_step_stage_runner"] = (
            lambda **_: (_ for _ in ()).throw(
                AssertionError("Phase 5 must not run for the Phase 4 retry fixture")
            )
        )
        return original_loop(**kwargs)

    monkeypatch.setattr(
        hmc_kernel_tuning,
        "initialize_hmc_kernel_geometry",
        lambda **_: geometry,
    )
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "run_hmc_bootstrap_screen",
        bootstrap_runner,
    )
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "run_hmc_tune_verify_repair_loop",
        loop_runner,
    )
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "run_full_chain_tfp_hmc",
        forbidden_hmc_runner,
    )
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "_phase7_attempt_state_from_stages",
        lambda **_: retry_state,
    )

    result = tune_hmc_kernel(
        adapter=_ToyGaussianAdapter(),
        initial_position=[0.0, 0.0],
        config=HMCKernelTuningConfig.standard(
            target_scope="kernel_fixed_mass_step_toy_gaussian",
            engineering_probe_covariance_multiplier=2.0,
            max_attempts=2,
            metric_update_requirement="require_operational_update",
            chain_execution_mode="eager",
            public_timeout_budget_s=None,
        ),
    )

    assert result.final_kernel_payload is None
    assert result.final_status == "budget_exhausted"
    assert len(seen_registries) == 2
    assert seen_registries[0] is seen_registries[1]
    assert hmc_runner_calls == []


def test_public_p4_source_coverage_is_complete_deterministic_and_hash_bound() -> None:
    first = hmc_kernel_tuning._public_p4_seed_source_coverage_payload()
    second = hmc_kernel_tuning._public_p4_seed_source_coverage_payload()

    assert first == second
    assert first["schema"] == "bayesfilter.hmc_g2_public_p4_runtime_source_coverage.v1"
    contracts = first["source_site_contracts"]
    semantics = hmc_warmup._G2_SEED_GATE_SEMANTIC_CONTRACTS
    expected_ids = {
        str(site_id)
        for semantic in semantics.values()
        for site_id in (
            semantic["derivation_site_id"],
            *semantic["interface_hop_site_ids"],
        )
    } | {str(site_id) for site_id in semantics}
    assert set(contracts) == expected_ids
    assert len(contracts) == 61
    assert sum(
        contract["site_kind"] == "terminal_consumption_gate"
        for contract in contracts.values()
    ) == 7

    for source in first["source_files"]:
        path = Path(str(source["path"]))
        assert source["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    encoded = json.dumps(
        first,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    registry = hmc_kernel_tuning._build_public_p4_seed_use_registry()
    assert registry.source_coverage_artifact_sha256 == hashlib.sha256(encoded).hexdigest()
    assert registry.source_coverage_artifact_sha256 == (
        hmc_kernel_tuning._build_public_p4_seed_use_registry().source_coverage_artifact_sha256
    )


def test_p4_phase7_loop_requires_typed_registry_and_non_p4_does_not_create_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = _geometry()
    bootstrap = _bootstrap()
    p4_config = hmc_kernel_tuning.HMCTuneVerifyRepairLoopConfig(
        engineering_probe_covariance_multiplier=2.0,
        max_attempts=1,
        chain_execution_mode="eager",
        target_scope="kernel_fixed_mass_step_toy_gaussian",
    )
    with pytest.raises(TypeError, match="caller-owned G2 seed-use registry"):
        hmc_kernel_tuning.run_hmc_tune_verify_repair_loop(
            adapter=_ToyGaussianAdapter(),
            geometry=geometry,
            bootstrap=bootstrap,
            config=p4_config,
        )

    def forbidden_builder() -> Any:
        raise AssertionError("non-P4 route constructed a registry")

    monkeypatch.setattr(
        hmc_kernel_tuning,
        "_build_public_p4_seed_use_registry",
        forbidden_builder,
    )
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "initialize_hmc_kernel_geometry",
        lambda **_: geometry,
    )
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "run_hmc_bootstrap_screen",
        lambda **_: bootstrap,
    )
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "run_hmc_tune_verify_repair_loop",
        lambda **kwargs: kwargs.get("_g2_seed_use_registry") is None
        and hmc_kernel_tuning.HMCTuneVerifyRepairLoopResult(
            config=kwargs["config"],
            geometry_artifact_hash=geometry.artifact_hash,
            bootstrap_artifact_hash=bootstrap.artifact_hash,
            adapter_signature=geometry.adapter_signature,
            target_dimension=geometry.target_dimension,
            attempts=(),
            final_status="budget_exhausted",
            diagnostic_role="budget_exhausted_non_promoting",
            hard_vetoes=(),
            repair_triggers=(),
            final_kernel_payload=None,
            final_kernel_hash=None,
            seed_report={"seed_owner": "BayesFilter"},
            diagnostic_roles={},
        ),
    )
    result = tune_hmc_kernel(
        adapter=_ToyGaussianAdapter(),
        initial_position=[0.0, 0.0],
        config=HMCKernelTuningConfig.smoke(
            target_scope="kernel_fixed_mass_step_toy_gaussian",
        ),
    )
    assert result.final_kernel_payload is None


def test_registry_freezes_after_p4_and_rejects_post_boundary_calls() -> None:
    registry = hmc_kernel_tuning._build_public_p4_seed_use_registry()
    config = hmc_warmup.Phase7EngineeringProbeBankConfig(
        chain_count=4,
        covariance_multiplier=2.0,
        root_seed=(20260829, 1300),
    )
    registry.consume(
        derivation_site_id=hmc_warmup._G2_P4_SEED_DERIVATION_SITE_ID,
        terminal_gate_site_id=hmc_warmup._G2_P4_SEED_GATE_SITE_ID,
        key="p4/engineering_probe",
        owner_file="hmc_warmup.py",
        owner_qualname="build_phase7_engineering_probe_bank",
        terminal_consumer="tensorflow_stateless_rng",
        derivation={
            "kind": "p4_domain_hash",
            "base_key": "engineering_probe_config.root_seed",
            "domain_label": hmc_warmup._PHASE7_ENGINEERING_PROBE_SEED_DOMAIN,
        },
        indices=(),
        seed=config.derived_seed,
        interface_hop_site_ids=hmc_warmup._G2_P4_SEED_INTERFACE_HOPS,
        is_p4=True,
    )
    with pytest.raises(hmc_warmup._G2SeedRegistryError) as raised:
        registry.consume(
            derivation_site_id=hmc_warmup._G2_P4_SEED_DERIVATION_SITE_ID,
            terminal_gate_site_id=hmc_warmup._G2_P4_SEED_GATE_SITE_ID,
            key="p4/engineering_probe/retry",
            owner_file="hmc_warmup.py",
            owner_qualname="build_phase7_engineering_probe_bank",
            terminal_consumer="tensorflow_stateless_rng",
            derivation={
                "kind": "p4_domain_hash",
                "base_key": "engineering_probe_config.root_seed",
                "domain_label": hmc_warmup._PHASE7_ENGINEERING_PROBE_SEED_DOMAIN,
            },
            indices=(),
            seed=(config.derived_seed[0] + 1, config.derived_seed[1]),
            interface_hop_site_ids=hmc_warmup._G2_P4_SEED_INTERFACE_HOPS,
            is_p4=True,
        )
    assert raised.value.failure_code == "seed_registry_postboundary_call"
    assert registry.post_boundary_registry_call_count == 1
