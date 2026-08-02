from __future__ import annotations

from pathlib import Path

import pytest


class _Adapter:
    parameter_dim = 3
    target_scope = "test:operational_broad_fixed_identity_grid"
    transport_manifest_hash = "transport-manifest"

    def adapter_signature(self):
        return "adapter-signature"


def _evidence(policy, *, label: str):
    from bayesfilter.inference.hmc_operational_broad_grid import (
        classify_operational_pair_evidence,
    )

    return classify_operational_pair_evidence(
        chain_run_means=(0.70,) * policy.evidence_unit_count,
        evidence_signature=label,
        policy=policy,
    )


class _Callbacks:
    def __init__(self, *, adapter, policy, handoff, config):
        self.adapter = adapter
        self.policy = policy
        self.handoff = handoff
        self.config = config
        self.events = []

    def primary(self, request):
        from bayesfilter.inference.hmc_operational_broad_grid import (
            OperationalPrimaryCandidate,
        )

        epsilon = 0.1 + 0.01 * request.num_leapfrog_steps
        self.events.append(
            {
                "role": "independent_epsilon_tune",
                "num_leapfrog_steps": request.num_leapfrog_steps,
                "tuned_step_size": epsilon,
                "all_draws_discarded": True,
            }
        )
        return OperationalPrimaryCandidate(
            request=request,
            tuned_step_size=epsilon,
            evidence=_evidence(self.policy, label=f"primary-{request.num_leapfrog_steps}"),
            metric_signature=self.handoff.frozen_metric_signature,
            coordinate_signature=self.handoff.coordinate_signature,
            lineage_signature=self.handoff.lineage_signature,
            tune_evidence_signature=f"tune-{request.num_leapfrog_steps}",
        )

    def guard(self, request):
        from bayesfilter.inference.hmc_operational_broad_grid import (
            SameEpsilonNeighborGuard,
        )

        self.events.append(
            {
                "role": "same_epsilon_neighbor_coverage_screen",
                "num_leapfrog_steps": request.num_leapfrog_steps,
                "inherited_step_size": request.inherited_step_size,
                "all_draws_discarded": True,
            }
        )
        return SameEpsilonNeighborGuard(
            request=request,
            evidence=_evidence(
                self.policy,
                label=(
                    f"coverage-{request.num_leapfrog_steps}-"
                    f"{request.inherited_step_size.hex()}"
                ),
            ),
        )


def test_config_preserves_reviewed_execution_floor() -> None:
    from bayesfilter.inference.neutra_broad_grid import (
        NeuTraBroadGridTuningConfig,
    )

    config = NeuTraBroadGridTuningConfig(
        initial_step_size=0.2,
        root_seed=(20260730, 1),
    )
    assert config.payload()["primary_l_grid"] == (3, 5, 9, 13, 18, 25)
    with pytest.raises(ValueError, match="exceed"):
        NeuTraBroadGridTuningConfig(
            initial_step_size=0.2,
            root_seed=(20260730, 1),
            screen_results=64,
        )


def test_fixed_identity_handoff_binds_target_transport_and_metric() -> None:
    from bayesfilter.inference.neutra_broad_grid import (
        build_fixed_identity_broad_grid_handoff,
    )

    handoff = build_fixed_identity_broad_grid_handoff(
        adapter=_Adapter(),
        target_signature="target-signature",
        evidence_path="plan.md",
    )
    assert handoff.grid_ready
    assert handoff.update_disposition == "fixed_identity"
    assert handoff.target_signature == "target-signature"
    assert handoff.frozen_metric_signature == handoff.prior_metric_signature


def test_generic_runner_preserves_complete_unranked_union_and_redacts_epsilon(
    tmp_path: Path,
) -> None:
    from bayesfilter.inference.neutra_broad_grid import (
        NeuTraBroadGridTuningConfig,
        run_neutra_operational_broad_grid_tuning,
    )

    payload = run_neutra_operational_broad_grid_tuning(
        adapter=_Adapter(),
        target_signature="target-signature",
        config=NeuTraBroadGridTuningConfig(
            initial_step_size=0.2,
            root_seed=(20260730, 2),
        ),
        output_dir=tmp_path / "broad",
        callbacks_factory=_Callbacks,
    )
    private = payload["private"]
    public = payload["public"]
    primary_events = tuple(
        row for row in private["events"] if row["role"] == "independent_epsilon_tune"
    )
    assert tuple(row["num_leapfrog_steps"] for row in primary_events) == (
        3,
        5,
        9,
        13,
        18,
        25,
    )
    assert len({row["tuned_step_size"] for row in primary_events}) == 6
    assert private["primary_barrier"]["complete"] is True
    assert private["coverage_barrier"]["complete"] is True
    assert private["representative"] is None
    assert private["retained_sampling_authorized"] is False
    assert public["stochastic_ranking_performed"] is False
    assert public["retained_sampling_authorized"] is False
    assert public["epsilon_values_exposed"] is False
    assert "initial_step_size" not in public["execution_config"]
    assert Path(payload["private_result_path"]).is_file()
    assert Path(payload["public_result_path"]).is_file()


def test_coverage_uses_exact_parent_epsilon_without_retuning(tmp_path: Path) -> None:
    from bayesfilter.inference.neutra_broad_grid import (
        NeuTraBroadGridTuningConfig,
        run_neutra_operational_broad_grid_tuning,
    )

    payload = run_neutra_operational_broad_grid_tuning(
        adapter=_Adapter(),
        target_signature="target-signature",
        config=NeuTraBroadGridTuningConfig(
            initial_step_size=0.2,
            root_seed=(20260730, 3),
        ),
        output_dir=tmp_path / "coverage",
        callbacks_factory=_Callbacks,
    )["private"]
    primaries = {
        row["request"]["num_leapfrog_steps"]: row["tuned_step_size"]
        for row in payload["primary_candidates"]
    }
    for row in payload["coverage_candidates"]:
        request = row["request"]
        assert request["epsilon_retuned"] is False
        assert request["recursive_expansion_allowed"] is False
        assert request["inherited_step_size"] in {
            primaries[parent] for parent in request["parent_l_values"]
        }
        assert row["parent_promotion_veto"] is False
