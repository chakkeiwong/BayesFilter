"""CPU-hidden independent reference and corruption tests, not training evidence."""
import importlib.util
import math
from pathlib import Path

import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.testing.neutra_warm_start_targets_tf import WarmStartTarget, F64
from bayesfilter.testing.neutra_rare_regions_tf import (
    Proposal, ImportanceProgram, GlobalProgram, UmbrellaProgram, EmusProgram,
    ConstrainedProgram, emus_stationary, reflected_coordinate, stratified_bank, known_masses)
from bayesfilter.testing.neutra_rare_region_phases import probability_screen, load_flow, StratifiedTrainer
from bayesfilter.inference.neutra_warm_start_tf import independence_step
from bayesfilter.inference.neutra_post_training import PostTrainingProbe
from bayesfilter.testing.neutra_warm_start_campaign import make_transport, write_json


def test_emus_stationary_matches_exact_discrete_overlap():
    # pi=(.2,.3,.5), windows {0,1}/{1,2}; their masses are .5/.8.
    matrix = tf.constant([[.7, .3], [.1875, .8125]], F64)
    z, residual = emus_stationary(matrix)
    tf.debugging.assert_near(z, tf.constant([.5/1.3, .8/1.3], F64), atol=1e-13)
    assert float(residual) < 1e-13
    # Indicator of bin0: window averages of g/S and 1/S.
    got = tf.reduce_sum(z*tf.constant([.4, 0.], F64))/tf.reduce_sum(z*tf.constant([.7, .8125], F64))
    assert abs(float(got)-.2) < 1e-13
    wrong_equal_weights = .4/(.7+.8125)
    assert abs(wrong_equal_weights-.2) > .05


def test_disconnected_overlap_cannot_supply_positive_unique_normalizers():
    z, residual = emus_stationary(tf.eye(3, dtype=F64))
    assert not bool(tf.reduce_all(tf.math.is_finite(z)))


def test_emus_gaussian_windows_independent_quantile_quadrature():
    # pi=N(0,1) x N(0,1), psi_k Gaussian around c_k, sigma=1.
    centers = [-2., 0., 2.]
    u = (tf.cast(tf.range(4096), F64)+.5)/4096
    quantile = tfp.distributions.Normal(tf.constant(0., F64), tf.constant(1., F64)).quantile(u)
    rows = tf.stack([tf.stack((c/2+quantile/math.sqrt(2), tf.zeros_like(quantile)), 1) for c in centers])
    x, lw, z, matrix, residual = EmusProgram(centers, [1., 1., 1.]).run(rows)
    expected = tf.nn.softmax(tf.constant([-1., 0., -1.], F64))
    tf.debugging.assert_near(z, expected, atol=2e-4)
    assert abs(float(tf.reduce_sum(tf.exp(lw)*x[:, 0]**2))-1.) < .005
    assert float(residual) < 1e-12


@pytest.mark.parametrize('name', ['mixture', 'warped_mixture'])
def test_snis_proposal_support_and_offset_cancellation(name):
    target = WarmStartTarget(name)
    proposal = Proposal(target, bridge_weight=.5)
    rows, lw, estimate, se = ImportanceProgram(target, proposal, 8192).run(tf.constant([1, 9]))
    tf.debugging.assert_near(lw, tf.nn.log_softmax(target.log_prob(rows)+17.-proposal.log_prob(rows)), atol=1e-12)
    assert bool(tf.reduce_all(tf.math.is_finite(lw)))
    assert float(estimate[1]) > 0.
    truth = tf.constant(known_masses(), F64)
    assert bool(tf.reduce_all(tf.abs(estimate[:3]-truth[:3]) < 8*se[:3]+.1*truth[:3]))


def test_mh_flux_correction_with_unequal_proposal_weights():
    pi = tf.constant([.1, .9], F64); q = tf.constant([.8, .2], F64)
    xy = tf.constant([[0.], [1.]], F64); yx = tf.reverse(xy, [0])
    lp = lambda x: tf.math.log(tf.gather(pi, tf.cast(x[:, 0], tf.int32)))
    lq = lambda x: tf.math.log(tf.gather(q, tf.cast(x[:, 0], tf.int32)))
    _, _, ratio, valid = independence_step(lp, lq, xy, yx, tf.constant([.5, .5], F64))
    flux = pi*tf.reverse(q, [0])*tf.exp(tf.minimum(ratio, 0.))
    tf.debugging.assert_near(flux[0], flux[1], atol=1e-14)
    assert bool(tf.reduce_all(valid))
    wrong = pi*tf.reverse(q, [0])*tf.exp(tf.minimum(lp(yx)-lp(xy), 0.))
    assert abs(float(wrong[0]-wrong[1])) > .01


def test_reflection_matches_symmetric_image_kernel():
    x = tf.constant([-17., -1.2, -.2, .7, 9.], F64)
    y = reflected_coordinate(x, tf.constant(.1, F64))
    assert bool(tf.reduce_all(tf.abs(y) <= .1+1e-14))
    # Independent method-of-images transition sum on [-b,b].
    b, sigma, a, c = .3, .11, -.17, .23
    def density(v):
        return math.exp(-.5*(v/sigma)**2)/(math.sqrt(2*math.pi)*sigma)
    def kernel(y, x):
        return sum(density(y-x+4*k*b)+density(y+x+2*b+4*k*b) for k in range(-8, 9))
    assert abs(kernel(c, a)-kernel(a, c)) < 1e-14


def test_split_shell_weights_telescope_without_oracle_masses():
    mass = 1.; total = 0.
    for survivors, n in ((6, 10), (3, 10), (7, 10)):
        total += (n-survivors)*mass/n
        mass *= survivors/n
    assert abs(total+mass-1.) < 1e-14
    target = WarmStartTarget('warped_mixture')
    x = tf.zeros([32, 2], F64)
    moved, acc = ConstrainedProgram(target, 32).run(x, tf.constant(.1, F64), tf.constant([7, 13]))
    assert bool(tf.reduce_all(tf.abs(moved[:, 0]) <= .1+1e-14))
    assert 0 < float(acc) <= 1
    assert float(tf.reduce_max(tf.abs(moved))) > 0.


def test_stratified_measure_preserves_tiny_mass():
    x = tf.constant([[0., 0.], [1., 0.], [-5., 0.], [5., 0.]], F64)
    w = tf.constant([1e-7, .001, .3, .6989999], F64)
    ids, logs, masses = stratified_bank(x, tf.math.log(w))
    tf.debugging.assert_near(tf.exp(masses), w, atol=1e-14)
    # Equal allocation of one deterministic point per stratum keeps weights w.
    selected = tf.gather(x, ids[:, 0])
    estimate = tf.reduce_sum(tf.exp(masses)*selected[:, 0])
    tf.debugging.assert_near(estimate, tf.reduce_sum(w*x[:, 0]), atol=1e-14)
    assert abs(float(tf.reduce_mean(selected[:, 0])-estimate)) > .1


def test_rare_screen_rejects_zero_and_relative_bias():
    true = known_masses()
    missing = [true.copy() for _ in range(8)]
    for row in missing:
        row[1] = 0.
    assert not probability_screen(missing)['passed']
    wrong = [true.copy() for _ in range(8)]
    for row in wrong:
        row[0] *= .1
    assert not probability_screen(wrong)['passed']
    assert probability_screen([true]*8)['passed']


def test_local_flow_roundtrip_payload(tmp_path):
    target = WarmStartTarget('mixture')
    flow = make_transport(target, 16, (11, 13), variance_scale=.2)
    path = tmp_path/'flow.json'
    write_json(path, flow.frozen_payload(target_signature=target.signature))
    loaded = load_flow(path, target)
    x = tf.constant([[-5., .2], [5., -.2]], F64)
    tf.debugging.assert_near(flow.log_prob(x), loaded.log_prob(x), atol=1e-12)
    payload = flow.frozen_payload(target_signature=target.signature)
    payload['parameters'][0]['biases'][0][0] += 1.
    write_json(path, payload)
    with pytest.raises(ValueError):
        load_flow(path, target)


def test_stratified_training_two_update_cpu_smoke_only():
    # Reviewed tiny CPU-hidden mechanics exception; no learned-quality evidence.
    target = WarmStartTarget('mixture')
    flow = make_transport(target, 4, (7, 17), variance_scale=.2)
    rows = tf.constant([[0., 0.], [1., 0.], [-5., 0.], [5., 0.]], F64)
    weights = tf.constant([1e-7, .001, .3, .6989999], F64)
    trainer = StratifiedTrainer(flow, rows, tf.math.log(weights), .001, batch=8)
    x, lw = trainer.draw(tf.constant([3, 9]))
    tf.debugging.assert_near(tf.reduce_sum(tf.exp(lw)), tf.constant(1., F64), atol=1e-14)
    assert int(tf.reduce_sum(tf.cast(tf.abs(x[:, 0]) < .1, tf.int32))) == 2
    result = trainer.run(tf.constant([5, 11]), tf.constant(2))
    assert int(result[0]) == 2 and bool(result[-1])
    # beta is a configuration scalar; passing a Tensor breaks the traced API.
    probe = PostTrainingProbe(flow, target, 1., rows=20)(seed=(5, 13))
    assert probe['complete'] and probe['finite'] and probe['valid_rows'] == 20


def test_hot_samplers_compile_and_are_batched():
    target = WarmStartTarget('mixture'); proposal = Proposal(target)
    x = tf.constant([[-5., 0.], [5., 0.]], F64)
    program = GlobalProgram(target, proposal, 8)
    last, trace, ga, la, bad = program.run(x, tf.constant(.01, F64), tf.constant([9, 11]))
    assert tuple(trace.shape) == (8, 2, 2) and int(bad) == 0
    umbrella = UmbrellaProgram(target, [-.1, .1], [.05, .05], 2, 8)
    last, rows, acc, bad = umbrella.run(tf.repeat(x*0., 2, 0), tf.constant(.0005, F64), tf.constant([9, 11]))
    assert tuple(rows.shape) == (2, 16, 2) and int(tf.reduce_sum(bad)) == 0


def test_controller_budget_and_completed_resume(tmp_path):
    path = Path(__file__).resolve().parents[1]/'scripts/run_neutra_rare_region_master.py'
    spec = importlib.util.spec_from_file_location('rare_master_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    control = module.Controller.__new__(module.Controller)
    control.cfg = {'gpu_process_seconds': 10., 'cpu_core_seconds': 20.}
    control.state = {'jobs': {'one': [{'status': 'complete', 'output': str(tmp_path),
        'gpu_process_seconds': 3., 'cpu_core_seconds': 4.}]}}
    assert control.remaining() == {'gpu_process_seconds': 7., 'cpu_core_seconds': 16.}
    assert control.done('one') == tmp_path
    assert control.done('missing') is None


def test_wrapper_rejects_arbitrary_subcommands():
    import subprocess
    wrapper = Path(__file__).resolve().parents[1]/'scripts/run_neutra_rare_region_campaign.sh'
    result = subprocess.run(['bash', str(wrapper), 'arbitrary-shell'], capture_output=True)
    assert result.returncode == 2
