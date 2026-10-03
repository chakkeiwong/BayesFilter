"""Explicit graph-reuse policy, isolation and checkpoint mechanics."""
from dataclasses import replace

import pytest
import tensorflow as tf

from bayesfilter.inference import HMCCandidateExecutionConfig, HMCControllerConfig, tune_hmc_kernel, resume_hmc_candidate_set_tuning
from bayesfilter.inference.hmc_candidate_set_tuning import HMCTuningCandidateSetController
from tests.test_hmc_candidate_set_execution import GaussianTarget, execution_config, make_binding, mass_for


def search():
    return HMCControllerConfig(primary_l_grid=(2, 3), epsilon_by_l=((2, (1.3,)), (3, (1.3,))),
        total_budget_units=10, repair_reserve_units=1, evidence_rungs=(1,))


def test_reuse_is_explicit_serialized_and_does_not_change_same_source_streams():
    plain = make_binding()
    shared = make_binding(config=execution_config(reuse_leapfrog_graphs=True))
    assert "reuse_leapfrog_graphs" not in plain.config.payload()
    assert HMCCandidateExecutionConfig.from_payload(plain.config.payload()) == plain.config
    assert HMCCandidateExecutionConfig.from_payload(shared.config.payload()) == shared.config
    assert shared.config.payload()["reuse_leapfrog_graphs"] is True
    assert plain.binding_hash != shared.binding_hash
    assert plain.scope == shared.scope
    with pytest.raises(TypeError, match="reuse_leapfrog_graphs"):
        replace(plain.config, reuse_leapfrog_graphs="false")


def test_public_search_resume_reuses_graphs_without_reusing_evidence(tmp_path):
    plain, shared = make_binding(), make_binding(config=execution_config(reuse_leapfrog_graphs=True))
    full = tune_hmc_kernel(adapter=plain._base_adapter, initial_position=plain.initial_active_state,
        config=search(), candidate_set_adapter=plain.typed_adapter).result
    tune_hmc_kernel(adapter=shared._base_adapter, initial_position=shared.initial_active_state,
        config=search(), candidate_set_adapter=shared.typed_adapter, output_dir=tmp_path, max_work_items=1)
    resumed = resume_hmc_candidate_set_tuning(tmp_path / "tuning_checkpoint.json", adapter=shared._base_adapter)
    assert full.verified_candidate_ids and full.verified_candidate_ids == resumed.result.verified_candidate_ids
    assert full.candidates == resumed.result.candidates
    assert all(row["stage"] in {"measurement", "verification"} for row in resumed.result.payload()["observations"])
    import json
    checkpoint = json.loads((tmp_path / "tuning_checkpoint.json").read_text())
    assert checkpoint["binding_hash"] != plain.binding_hash
    rows = [json.loads(p.read_text()) for p in (tmp_path / "numerical_evidence").glob("*.json")]
    baseline = {e["work"]["work_item_id"]: e for e in plain._evidence.values()}
    assert len(rows) == len(baseline)
    for row in rows:
        expected = baseline[row["work"]["work_item_id"]]
        for key in ("candidate", "seed", "samples", "trace", "analysis"):
            assert row[key] == expected[key], key


def test_cache_lifetime_respects_target_geometry_and_count():
    target = GaussianTarget()
    bindings = [make_binding(config=execution_config(reuse_leapfrog_graphs=True)),
        make_binding(config=execution_config(reuse_leapfrog_graphs=True),
                     mass_artifact=mass_for(target, factor=[[2., 0.], [0., .5]])),
        make_binding(target=GaussianTarget(scale=2.), config=execution_config(reuse_leapfrog_graphs=True))]
    outputs = []
    for binding in bindings:
        controller = HMCTuningCandidateSetController(binding.scope, search())
        candidates = controller.result().candidates
        assert len(candidates) == 2
        for count in (8, 13):
            for candidate in candidates:
                actual = binding._run(candidate, binding.initial_active_state, count, (8, 3))
                outputs.append(actual.samples)
        assert set(binding._runners) == {(None, 8), (None, 13)}
        assert all(r._runner.experimental_get_tracing_count() == 1
                   for runner in binding._runners.values() for r in runner._runners)
    assert bindings[0]._runners[(None, 8)] is not bindings[1]._runners[(None, 8)]
    assert not bool(tf.reduce_all(tf.equal(outputs[0], outputs[4])))
    assert not bool(tf.reduce_all(tf.equal(outputs[0], outputs[8])))


def test_position_field_rejects_exact_score_reuse_option_before_preparation(tmp_path):
    from tests.test_hmc_tuning_dispatch import _Adapter, _binding, _config
    with pytest.raises(ValueError, match="exact-score TFP"):
        tune_hmc_kernel(adapter=_Adapter(), initial_position=tf.zeros([4, 2], tf.float64),
            parameter_scales=tf.ones([2], tf.float64), config=_config(), runner_binding=_binding(),
            execution_config=execution_config(reuse_leapfrog_graphs=True), output_dir=tmp_path)
