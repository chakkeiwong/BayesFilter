from __future__ import annotations

from pathlib import Path

from bayesfilter.inference import (
    HMCControllerConfig,
    HMCCandidateSetScope,
    HMCTypedCandidateSetRun,
    issue_hmc_candidate_set_adapter,
    tune_fixed_transport_hmc_kernel,
    tune_hmc_kernel,
)


def _scope(name: str, *, kind: str) -> HMCCandidateSetScope:
    coordinate = "fixed_transport_latent_z" if kind == "fixed_transport" else "ordinary"
    return HMCCandidateSetScope(
        scope_id=name,
        search_id="search-1",
        target_signature=f"target-{kind}-v1",
        mass_signature=f"mass-{kind}-v1",
        coordinate_system=coordinate,
        start_bank_signature="bank-v1",
        warmup_protocol="warmup-v1",
        adapter_signature=f"adapter-{kind}-v1",
        target_preparation_identity=f"{kind}-target-preparation-v1",
        transition_identity=f"{kind}-fixed-transition-v1",
    )


def _config() -> HMCControllerConfig:
    return HMCControllerConfig(
        primary_l_grid=(3, 5),
        epsilon_by_l=((3, (0.2, 0.3)), (5, (0.15,))),
        total_budget_units=30,
        repair_reserve_units=6,
    )


def _adapter(scope: HMCCandidateSetScope, kind: str):
    def observe(_work, _candidate):
        return {"decision": "passed", "acceptance": 0.70}

    return issue_hmc_candidate_set_adapter(
        scope=scope,
        adapter_kind=kind,
        observe=observe,
        source_dependency_closure={"module": f"{kind}-fixture", "version": 1},
        target_preparation_identity=f"{kind}-target-preparation-v1",
        transition_identity=f"{kind}-fixed-transition-v1",
    )


def test_ordinary_dispatch_uses_typed_shared_candidate_controller(tmp_path: Path):
    adapter = _adapter(_scope("ordinary-scope", kind="ordinary"), "ordinary")
    run = tune_hmc_kernel(
        adapter=object(),
        initial_position=(0.0,),
        config=_config(),
        output_dir=tmp_path / "ordinary",
        candidate_set_adapter=adapter,
    )

    assert isinstance(run, HMCTypedCandidateSetRun)
    assert run.result.final_status == "complete"
    assert len(run.result.verified_candidate_ids) == 3
    assert run.artifact_receipt is not None
    assert (tmp_path / "ordinary" / "candidate_set_result.json").is_file()


def test_fixed_transport_compatibility_wrapper_uses_the_same_controller():
    adapter = _adapter(_scope("transport-scope", kind="fixed_transport"), "fixed_transport")
    run = tune_fixed_transport_hmc_kernel(
        base_adapter=object(),
        fixed_transport=object(),
        initial_position=(0.0,),
        config=_config(),
        candidate_set_adapter=adapter,
    )

    assert isinstance(run, HMCTypedCandidateSetRun)
    assert run.adapter.adapter_kind == "fixed_transport"
    assert run.result.verified_candidate_ids == tuple(
        candidate.candidate_id for candidate in run.result.candidates
    )


def test_equivalent_entry_points_have_the_same_ordered_work_contract():
    ordinary = _adapter(_scope("ordinary-scope", kind="ordinary"), "ordinary")
    transport = _adapter(_scope("transport-scope", kind="fixed_transport"), "fixed_transport")
    ordinary_run = tune_hmc_kernel(
        adapter=object(), initial_position=(0.0,), config=_config(), candidate_set_adapter=ordinary
    )
    transport_run = tune_fixed_transport_hmc_kernel(
        base_adapter=object(), fixed_transport=object(), initial_position=(0.0,),
        config=_config(), candidate_set_adapter=transport
    )

    ordinary_trace = [
        (item.stage, item.ordinal, item.priority, item.candidate_id.split(":candidate:")[-1])
        for item in ordinary_run.result.work_items
    ]
    transport_trace = [
        (item.stage, item.ordinal, item.priority, item.candidate_id.split(":candidate:")[-1])
        for item in transport_run.result.work_items
    ]
    assert ordinary_trace == transport_trace


def test_adapter_cannot_change_scope_identity_inside_an_observation():
    scope = _scope("ordinary-scope", kind="ordinary")

    def observe(_work, _candidate):
        return {
            "decision": "passed",
            "adapter_signature": "stale-adapter",
        }

    adapter = issue_hmc_candidate_set_adapter(
        scope=scope,
        adapter_kind="ordinary",
        observe=observe,
        source_dependency_closure={"module": "fixture", "version": 1},
        target_preparation_identity="ordinary-target-preparation-v1",
        transition_identity="ordinary-fixed-transition-v1",
    )
    run = tune_hmc_kernel(
        adapter=object(), initial_position=(0.0,), config=_config(), candidate_set_adapter=adapter
    )
    assert run.result.final_status == "shared_invalidity"
    assert run.result.verified_candidate_ids == ()
