"""Timing must remain auditable without changing candidate random streams."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

from tests.test_hmc_bootstrap_initialization import inputs
from tests.test_hmc_candidate_set_execution import make_binding, execution_config


def geometry_with_clocks(clock):
    _, geometry, _ = inputs()
    startup = {"original_geometry_hash": geometry.artifact_hash,
               "rounds": [{"wall_seconds": clock, "epsilon": .2, "seed": [3,4],
                            "mean_acceptance_probability": .8}],
               "first_invalid_proposal": {"wall_seconds": clock, "epsilon": .4,
                                          "proposal_valid": False}}
    return replace(geometry, formula_report={**geometry.formula_report, "bootstrap_initialization": startup})


def test_numerical_geometry_ignores_only_probe_clocks_and_keeps_full_artifact():
    left, right = geometry_with_clocks(.1), geometry_with_clocks(20.)
    assert left.artifact_hash != right.artifact_hash
    assert left.numerical_hash == right.numerical_hash
    assert left.formula_report["bootstrap_initialization"]["rounds"][0]["wall_seconds"] == .1
    for key, value in (("epsilon", .3), ("seed", [4,5]), ("mean_acceptance_probability", .7)):
        report = deepcopy(left.formula_report)
        report["bootstrap_initialization"]["rounds"][0][key] = value
        changed = replace(left, formula_report=report)
        assert changed.numerical_hash != left.numerical_hash
    report = deepcopy(left.formula_report)
    report["bootstrap_initialization"]["first_invalid_proposal"]["proposal_valid"] = True
    assert replace(left, formula_report=report).numerical_hash != left.numerical_hash
    _, geometry, _ = inputs()
    assert geometry.numerical_hash == geometry.artifact_hash


def test_candidate_stream_identity_is_clock_invariant_but_numerically_sensitive():
    from bayesfilter.inference.hmc_candidate_set_execution import _issue_binding
    base = make_binding()
    def bind(geometry):
        return _issue_binding(adapter=base._base_adapter, layers=base._spec["layers"],
            initial_active_state=base.initial_active_state, target_scope=base._spec["target_scope"],
            target_lineage=base._spec["target_lineage"],
            preparation={"geometry_hash": geometry.artifact_hash,
                         "numerical_geometry_hash": geometry.numerical_hash},
            config=execution_config(), source_paths=[__file__], scope_id="test", search_id="stream",
            epsilon_domain=(.01,1.95), repair_factor=1.1, max_repairs_per_family=0)
    left, right = bind(geometry_with_clocks(.1)), bind(geometry_with_clocks(20.))
    assert left.binding_hash != right.binding_hash  # Audit still binds the timing report.
    assert left.scope == right.scope
    work = SimpleNamespace(work_item_id="work-1", stage="measurement", candidate_record_hash="pair")
    assert left.work_seed(work) == right.work_seed(work)
    different = bind(replace(geometry_with_clocks(.1), initial_step_size=.123))
    assert left.work_seed(work) != different.work_seed(work)
