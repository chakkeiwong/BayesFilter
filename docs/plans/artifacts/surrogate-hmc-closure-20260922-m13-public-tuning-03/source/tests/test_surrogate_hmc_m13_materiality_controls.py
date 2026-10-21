"""Diagnostic authority and actual proposal-kernel wiring checks."""
import copy
import json
import math

import pytest
import tensorflow as tf

from bayesfilter.inference import surrogate_hmc_materiality_selection as selection
from bayesfilter.inference.surrogate_hmc_program import file_hash, write_json, CPU_CAP
from bayesfilter.testing.surrogate_hmc_m13_materiality_controls import probe_graph
from bayesfilter.testing.surrogate_hmc_m13_four_chain_cost import (
    bind_components, native_force, batched_endpoint, STARTS,
)


def arm(multiplier, alpha=.8):
    return dict(multiplier=multiplier, acceptance_probabilities=[alpha]*16,
                nominal_healthy=True, proposal_healthy=True,
                stress_failures_flagged=True, stress_healthy_rows=[True, False, True, False])


def curves():
    return {p: dict(arms=[arm(m, a) for m,a in zip(selection.MULTIPLIERS, (.5,.7,.8,.805))],
                    probe_steps=[.05,.2], probe_leapfrogs=2) for p in selection.PARTITIONS}


def test_materiality_compares_probability_units_and_requires_both_partitions():
    data = curves()
    index, decisions = selection.choose(data)
    assert index == 2
    assert decisions['calibration'][2]['maximum_acceptance_probability_difference'] == pytest.approx(.005)
    data['validation']['arms'][3]['acceptance_probabilities'][0] = .85
    assert selection.choose(data)[0] is None


@pytest.mark.parametrize('field,value', [('nominal_healthy',False), ('proposal_healthy',False),
    ('stress_failures_flagged',False), ('stress_healthy_rows',[False,False,True,False])])
def test_health_cannot_be_overruled_by_equal_acceptance(field, value):
    strong, weak = arm(.01), arm(.001)
    strong[field] = value
    strong['non_harm_passed'] = True
    assert not selection.pair_decision(strong, weak)['non_harm_passed']


def test_bad_probability_or_probe_design_is_not_admissible():
    bad = arm(.01)
    bad['acceptance_probabilities'][0] = float('nan')
    with pytest.raises(ValueError, match='sixteen finite'):
        selection.pair_decision(bad, arm(.001))
    data = curves(); data['validation']['probe_steps'] = [.002,.002]
    with pytest.raises(ValueError, match='prospective'):
        selection.choose(data)


def test_selection_reload_checks_evidence_scope_and_recomputes_decision(tmp_path):
    scope, hashes, paths, data_paths = {'model':'M13', 'particle_count':4096}, {}, {}, {}
    for partition, curve in curves().items():
        seed, design = selection.PARTITIONS[partition]
        data_paths[partition] = tmp_path/(partition+'-data.json')
        write_json(data_paths[partition], dict(scope=scope, partition=partition,
                   data_seed=seed, design_partition=design))
        paths[partition] = tmp_path/(partition+'-curve.json')
        write_json(paths[partition], dict(curve, scope=scope, source_hashes=hashes,
                   data_sha256=file_hash(data_paths[partition])))
    path = tmp_path/'selection.json'
    selection.issue_selection(path, scope=scope, source_hashes=hashes, curves=paths, data=data_paths)
    good = selection.load_selection(path, expected_scope=scope, expected_source_hashes=hashes)
    assert good['multiplier'] == .01 and good['posterior_qualified'] is False
    with pytest.raises(ValueError, match='stale'):
        selection.load_selection(path, expected_scope={'model':'wrong'}, expected_source_hashes=hashes)
    altered = copy.deepcopy(good); altered['multiplier'] = .1
    write_json(path, altered)
    with pytest.raises(ValueError, match='measured'):
        selection.load_selection(path, expected_scope=scope, expected_source_hashes=hashes)
    write_json(path, good); data_paths['validation'].write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        selection.load_selection(path, expected_scope=scope, expected_source_hashes=hashes)


class DiagnosticTarget:
    def compile(self, count, **kwargs):
        assert count == 4 and kwargs == dict(filter_moments=False, all_correction_diagnostics=False)
        def evaluate(x):
            return -tf.reduce_sum(x*x, -1), -2*x, dict(program_valid=tf.ones([4], tf.bool))
        return evaluate


def test_probe_calls_actual_endpoint_corrected_kernel_with_common_seed_and_dynamic_step():
    def evaluator(row):
        return -tf.reduce_sum(row*row), tf.constant(True), tf.constant(0., tf.float64)
    scope = 'test-prospective-materiality'
    binding = bind_components(native_force(DiagnosticTarget()), batched_endpoint(evaluator), scope)
    graph = probe_graph(binding, scope)
    points = tf.constant(STARTS, tf.float64)
    seed = tf.constant([2501, 101], tf.int32)
    small = graph(points, seed, tf.constant(.05, tf.float64))
    replay = graph(points, seed, tf.constant(.05, tf.float64))
    large = graph(points, seed, tf.constant(.2, tf.float64))
    assert small.states.shape == (1,4,2)
    tf.debugging.assert_equal(small.states, replay.states)
    tf.debugging.assert_equal(small.log_accept_ratio, replay.log_accept_ratio)
    assert bool(tf.reduce_any(small.states != large.states))
    assert not bool(tf.reduce_any(small.force_fallback | large.force_fallback))
    assert graph.experimental_get_tracing_count() == 1


def test_approved_cpu_expansion_and_materiality_job_reservation():
    from bayesfilter.inference.surrogate_hmc_closure_program import jobs, PASS_CAPS
    assert CPU_CAP == 198000
    selected = jobs('m13-materiality-controls')
    assert [j['name'] for j in selected] == ['review','m13-materiality-controls']
    cap = PASS_CAPS['m13-materiality-controls']
    assert sum(j['cpu'] for j in selected) < cap['cpu_worker_seconds']
    assert sum(j['timeout'] for j in selected if j['device']=='gpu') <= cap['gpu_seconds']


def test_repair_uses_new_family_and_fresh_data_without_relaxing_tolerance(tmp_path):
    scope = {"model": "M13", "materiality_protocol": selection.REPAIR_PROTOCOL}
    multipliers, partitions, attempt = selection.protocol(scope)
    assert multipliers == (.001, .0001, .00001, .000001)
    assert partitions == {"calibration": (1706, 15), "validation": (2706, 16)}
    assert attempt == 6 and selection.ALLOWANCE == .01
    paths, data = {}, {}
    for partition, (seed, design) in partitions.items():
        data[partition] = tmp_path / (partition+"-data.json")
        write_json(data[partition], dict(scope=scope, partition=partition,
            data_seed=seed, design_partition=design))
        paths[partition] = tmp_path / (partition+"-curve.json")
        write_json(paths[partition], dict(scope=scope, source_hashes={},
            data_sha256=file_hash(data[partition]), probe_steps=[.05,.2], probe_leapfrogs=2,
            arms=[arm(m, .8) for m in multipliers]))
    path = tmp_path / "selected.json"
    selection.issue_selection(path, scope=scope, source_hashes={}, curves=paths, data=data)
    result = selection.load_selection(path, expected_scope=scope, expected_source_hashes={})
    assert result['multiplier'] == .001
    assert result['controls'] == selection.original_controls(.001)
    row = json.loads(data['validation'].read_text()); row['data_seed'] = 2705
    write_json(data['validation'], row)
    curve = json.loads(paths['validation'].read_text())
    curve['data_sha256'] = file_hash(data['validation']); write_json(paths['validation'], curve)
    with pytest.raises(ValueError, match='data/scope'):
        selection.issue_selection(tmp_path/'bad.json', scope=scope, source_hashes={}, curves=paths, data=data)
    with pytest.raises(ValueError, match='protocol'):
        selection.protocol({'materiality_protocol': 'invented'})
