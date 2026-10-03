"""Independent derivative/source checks; tiny CPU/XLA engineering fixtures."""
import ast
import dataclasses
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import neutra_transport_core as core
from bayesfilter.inference.neutra_transport import (
    NeuTraTransport, NeuTraTransportConfig, NeuTraTransportTrainer, NeuTraOptimizerConfig,
)
from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact, stable_frozen_neutra_artifact_signature


def config(kind="naf_dsf", stages=2):
    return NeuTraTransportConfig(3, kind, (6, 6), stages, "elu", (247, 19), 2.,
                                mixture_components=3, final_weight_scale=.15)


def rows():
    return tf.constant([[.4, -.7, 1.1], [-1.3, .8, -.5], [2., 1.3, -.4], [.1, -.3, .6]], tf.float64)


def gaussian(x):
    return -.5*tf.reduce_sum(x*x, axis=-1), -x, tf.ones(tf.shape(x)[0], tf.bool)


def finite_jacobian(fn, x, h=1.e-5):
    return np.stack([((fn(x+tf.one_hot(j, 3, dtype=tf.float64)*h)
                      -fn(x-tf.one_hot(j, 3, dtype=tf.float64)*h))/(2*h)).numpy()
                     for j in range(3)], axis=-1)


@pytest.mark.parametrize("kind", ["iaf", "naf_dsf"])
def test_conditional_map_jacobian_score_inverse_and_logdet(kind):
    transport = NeuTraTransport(config(kind))
    if kind == "naf_dsf":
        # The author's hidden directions start at std=.001. Exercise a
        # nontrivial conditioner rather than letting near-identity pass parity.
        for stage in transport.stages:
            for weight in stage.weights[:-1]:
                weight.assign(weight*100.)
    x = rows()
    y, ld = transport.forward_and_logdet(x)
    recovered, inverse_ld = transport.inverse_and_forward_logdet(y)
    np.testing.assert_allclose(recovered, x, atol=3.e-10, rtol=3.e-10)
    np.testing.assert_allclose(ld, inverse_ld, atol=3.e-10)
    jac = finite_jacobian(transport.forward_batch, x)
    np.testing.assert_allclose(ld, np.linalg.slogdet(jac)[1], atol=2.e-9)
    pullback = transport.pullback_score_batch(x, y)
    np.testing.assert_allclose(pullback, np.einsum("bij,bi->bj", jac, y), atol=2.e-9)
    with tf.GradientTape() as tape:
        tape.watch(x)
        density = transport.forward_and_logdet(x)[1]
    np.testing.assert_allclose(transport.log_abs_det_jacobian_score_batch(x), tape.gradient(density, x), atol=1.e-12)
    for layer in transport.stages:
        _, _, j, _ = core.value_jacobian_logdet_score(layer, x)
        np.testing.assert_array_equal(np.triu(j.numpy(), 1), 0.)
        assert np.max(np.abs(np.tril(j.numpy(), -1))) > 1.e-5


def test_author_free_bias_is_outside_cap_and_has_unrestricted_gradient():
    transport = NeuTraTransport(config("iaf", stages=1))
    stage = transport.stages[0]
    bias = stage.biases[-1]
    bias.assign([5., -6., 7., 0., 0., 0.])
    x = rows()
    with tf.GradientTape() as tape:
        scale = core.iaf_parameters(stage, x)[0]
        objective = tf.reduce_sum(scale)
    gradient = tape.gradient(objective, bias)
    np.testing.assert_allclose(gradient[:3], 4., atol=1.e-12)
    raw, _ = core.masked_network(stage, x)
    expected = bias[:3]+2.*tf.tanh((raw[:, :3]-bias[:3])/2.)
    np.testing.assert_allclose(scale, expected, atol=1.e-12)
    assert float(tf.reduce_max(tf.abs(scale))) > stage.s_max


def test_hoffman_profile_masks_match_author_tfp_blocks():
    # Diagnostic-only TFP reference generator uses NumPy; runtime masks do not.
    from tensorflow_probability.python.bijectors.masked_autoregressive import _gen_mask
    cfg = NeuTraTransportConfig.hoffman_author_iaf(3, conditional_scale_cap=2., seed=(3, 7))
    stage = NeuTraTransport(cfg).stages[0]
    expected_first = _gen_mask(3, 3, 3, mask_type="exclusive").T
    expected_later = _gen_mask(3, 3, 3, mask_type="inclusive").T
    np.testing.assert_array_equal(stage.masks[0], expected_first)
    np.testing.assert_array_equal(stage.masks[1], expected_later)
    np.testing.assert_array_equal(stage.masks[2], np.concatenate([expected_later, expected_later], axis=-1))


@pytest.mark.parametrize("convention", ["bounded_tanh", "dsge_bounded_tanh", "identity"])
def test_frozen_and_training_legacy_numerics_share_authority(convention):
    from bayesfilter.inference.neutra_weighted_training import WeightedNeuTraConfig, WeightedDenseIAFTransport
    from bayesfilter.inference.neutra_artifacts import _DenseAutoregressiveIAFComponent
    stage = WeightedDenseIAFTransport(WeightedNeuTraConfig(dimension=3, hidden_layers=(6,), stages=1)).stages[0]
    stage.weights[-1].assign(tf.random.stateless_normal(stage.weights[-1].shape, [3, 4], dtype=tf.float64)*.12)
    stage.scale_transform = convention
    frozen = _DenseAutoregressiveIAFComponent(dim=3, hidden_layers=(6,), activation="elu", s_max=2.,
        scale_transform=convention, weights=tuple(tf.identity(w) for w in stage.weights),
        biases=tuple(tf.identity(b) for b in stage.biases))
    x = rows()
    a, al = stage.forward_and_logdet(x)
    b, bl = frozen.forward_and_logdet(x)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(al, bl)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        output, ld = frozen.forward_and_logdet(x)
    np.testing.assert_allclose(frozen.pullback_score(x, x), tape.gradient(output, x, output_gradients=x), atol=1.e-12)
    np.testing.assert_allclose(frozen.logdet_score(x), tape.gradient(ld, x), atol=1.e-12)


def test_naf_paper_equation_interior_and_unbounded_tails():
    transport = NeuTraTransport(config(stages=1))
    stage = transport.stages[0]
    x = rows()
    logs, offsets, weights = stage.pseudo_parameters(x)
    s = tf.reduce_sum(tf.nn.softmax(weights)*tf.sigmoid(tf.exp(logs)*x[..., None]+offsets), axis=-1)
    expected = tf.math.log(s)-tf.math.log1p(-s)
    actual, _ = stage.forward_and_logdet(x)
    np.testing.assert_allclose(actual, expected, atol=1.e-12)
    tail = tf.constant([[-100., -75., -50.], [100., 75., 50.]], tf.float64)
    mapped, ld = stage.forward_and_logdet(tail)
    assert bool(tf.reduce_all(tf.math.is_finite(ld)))
    assert float(mapped[0, 0]) < -50. and float(mapped[1, 0]) > 50.
    np.testing.assert_allclose(stage.inverse(mapped), tail, atol=2.e-8)
    first_slopes = tf.exp(stage.pseudo_parameters(x)[0])[0, 0]
    assert float(tf.reduce_max(first_slopes)-tf.reduce_min(first_slopes)) > 0.


def test_naf_inverse_input_and_parameter_derivatives_are_implicit():
    # Inverse residual error divided by the finite-difference perturbation must
    # be below the derivative tolerance; production inverse tolerance is too
    # coarse for a 1e-5 finite-difference oracle.
    stage = NeuTraTransport(dataclasses.replace(config(stages=1),
        inverse_atol=1.e-14, inverse_rtol=1.e-14)).stages[0]
    y = rows()
    with tf.GradientTape() as tape:
        tape.watch(y)
        x = stage.inverse(y)
        objective = tf.reduce_sum(x*x)
    analytic = tape.gradient(objective, y)
    finite = finite_jacobian(stage.inverse, y)
    np.testing.assert_allclose(analytic, np.einsum("bij,bi->bj", finite, 2*x), atol=2.e-8)
    var = stage.biases[-1]
    original = tf.identity(var)
    direction = tf.random.stateless_normal(var.shape, [5, 9], dtype=tf.float64)
    with tf.GradientTape() as tape:
        x = stage.inverse(y)
        objective = tf.reduce_sum(x*x)
    predicted = tf.reduce_sum(tape.gradient(objective, var)*direction)
    h = 1.e-4
    var.assign(original+h*direction)
    plus = tf.reduce_sum(tf.square(stage.inverse(y)))
    var.assign(original-h*direction)
    minus = tf.reduce_sum(tf.square(stage.inverse(y)))
    var.assign(original)
    np.testing.assert_allclose(predicted, (plus-minus)/(2*h), atol=2.e-6, rtol=2.e-6)


def test_naf_author_conditioner_matches_independent_source_operations():
    stage = NeuTraTransport(config(stages=1)).stages[0]
    x = rows().numpy()
    hidden = x
    # NumPy is an independent diagnostic oracle here, never a runtime backend.
    for i, (direction, bias, mask) in enumerate(zip(stage.weights, stage.biases, stage.masks)):
        weight = direction.numpy()
        if i == len(stage.weights)-1:
            weight /= np.linalg.norm(weight, axis=0, keepdims=True)
        scale = stage.context_scale_weights[i].numpy()+stage.context_scale_biases[i].numpy()
        hidden = scale*(hidden@(weight*mask.numpy()))+bias.numpy()+stage.context_bias_weights[i].numpy()
        if i+1 < len(stage.weights):
            hidden = np.where(hidden > 0., hidden, np.expm1(hidden))
    features = hidden.reshape(4, stage.features, 3).transpose(0, 2, 1)
    pseudo = (features@stage.projection[0].numpy()+stage.projection[1].numpy()).reshape(4, 3, 3, 3)
    slopes = np.logaddexp(0., pseudo[:, :, 0])+stage.config.slope_floor
    np.testing.assert_allclose(tf.exp(stage.pseudo_parameters(rows())[0]), slopes, atol=1.e-12)
    np.testing.assert_allclose(stage.pseudo_parameters(rows())[1], pseudo[:, :, 1], atol=1.e-12)
    np.testing.assert_allclose(stage.pseudo_parameters(rows())[2], pseudo[:, :, 2], atol=1.e-12)


def test_failed_inverse_is_nonfinite_instead_of_silent_approximation():
    transport = NeuTraTransport(dataclasses.replace(config(stages=1), inverse_max_iterations=1))
    assert not bool(tf.reduce_all(tf.math.is_finite(transport.inverse_theta_to_z_batch(rows()))))


@pytest.mark.parametrize("kind", ["iaf", "naf_dsf"])
def test_layerwise_vaitl_score_matches_density_score(kind):
    transport = NeuTraTransport(config(kind))
    x = rows()
    y, ld, score = core.forward_proposal_score(transport, x)
    with tf.GradientTape() as tape:
        tape.watch(y)
        logq = transport.log_prob(y)
    expected = tape.gradient(logq, y)
    np.testing.assert_allclose(score, expected, atol=5.e-10, rtol=5.e-10)
    np.testing.assert_allclose(ld, transport.forward_and_logdet(x)[1], atol=1.e-12)


def test_exact_gaussian_map_has_zero_path_gradient_without_target_hessian():
    from bayesfilter.inference.neutra_training_mechanisms import FreeDiagonalAffine
    map_ = FreeDiagonalAffine([.4, -.2, .7], [.2, -.4, .1])
    mean, scale = tf.identity(map_.shift), tf.exp(map_.log_scale)
    def exact_target(y):
        z = (y-mean)/scale
        return tf.stop_gradient(-.5*tf.reduce_sum(z*z, axis=-1)), tf.stop_gradient(-z/scale), tf.ones([4], tf.bool)
    result = core.reverse_kl_evaluate(map_, exact_target, rows(), estimator="path")
    assert bool(result["valid"])
    for grad in result["gradients"]:
        np.testing.assert_allclose(grad, 0., atol=1.e-12)


@pytest.mark.parametrize("kind", ["iaf", "naf_dsf"])
def test_frozen_loader_roundtrip_and_tamper_rejection(kind):
    transport = NeuTraTransport(config(kind))
    payload = json.loads(json.dumps(transport.frozen_payload(target_signature="a"*64)))
    loaded = load_frozen_neutra_artifact(payload, expected_target_signature="a"*64)
    assert loaded.artifact_signature == stable_frozen_neutra_artifact_signature(loaded)
    assert not loaded.transport.trainable_variables
    np.testing.assert_array_equal(transport.forward_batch(rows()), loaded.transport.forward_batch(rows()))
    np.testing.assert_allclose(transport.pullback_score_batch(rows(), rows()),
                               loaded.transport.pullback_score_batch(rows(), rows()), atol=1.e-12)
    payload["parameters"][0]["biases"][0][0] += .01
    with pytest.raises(ValueError, match="hash"):
        load_frozen_neutra_artifact(payload, expected_target_signature="a"*64)


@pytest.mark.parametrize("kind,estimator", [("iaf", "standard"), ("iaf", "path"), ("naf_dsf", "path")])
def test_xla_adam_resume_rejection_and_graph_contract(kind, estimator):
    cfg = dataclasses.replace(config(kind, stages=1), hidden_layers=(4,))
    opt = NeuTraOptimizerConfig(4, estimator, .001, .9, .999, 1.e-7, None)
    trainer = NeuTraTransportTrainer(NeuTraTransport(cfg), gaussian, opt, target_signature="a"*64)
    first = trainer.train_step(rows())
    assert bool(first["valid"]) and int(first["iteration"]) == 1
    saved = json.loads(json.dumps(trainer.checkpoint()))
    resumed = NeuTraTransportTrainer(NeuTraTransport(cfg), gaussian, opt, target_signature="a"*64)
    resumed.restore(saved)
    trainer.train_step(rows()*.9)
    resumed.train_step(rows()*.9)
    assert trainer.checkpoint() == resumed.checkpoint()
    before = trainer.checkpoint()
    bad = trainer.train_step(tf.fill([4, 3], tf.constant(float("nan"), tf.float64)))
    assert not bool(bad["valid"])
    assert before == trainer.checkpoint()
    concrete = trainer.train_step.get_concrete_function()
    assert concrete.function_def.attr["_XlaMustCompile"].b
    assert trainer.train_step.experimental_get_tracing_count() == 1
    graph = concrete.graph.as_graph_def()
    nodes = [*graph.node, *(n for f in graph.library.function for n in f.node_def)]
    assert not any(n.op in ("PyFunc", "EagerPyFunc") for n in nodes)
    assert not any("pfor" in n.name.lower() for n in nodes)


def test_new_naf_uses_standard_1000_point_post_training_procedure():
    class Bridge:
        parameter_dim = 3
        def value_score_status(self, x, beta):
            value, score, valid = gaussian(x)
            return value, score, {"bridge_valid": valid}
    opt = NeuTraOptimizerConfig(4, "path", .001, .9, .999, 1.e-7, None)
    trainer = NeuTraTransportTrainer(NeuTraTransport(config(stages=1)), gaussian, opt, target_signature="a"*64)
    handoff = trainer.finalize(Bridge(), diagnostic_seed=(53, 81))
    report = handoff["post_training"]
    assert report["rows"] == report["valid_rows"] == 1000
    assert report["finite"] and report["complete"]
    assert report["scale_diagnostics"][0]["fraction_slope_below_point1"] is None
    assert not handoff["training_quality_established"]
    assert handoff["frozen_transport"]["training_state_hash"] == handoff["checkpoint"]["checkpoint_hash"]


def test_adam_rolls_back_finite_but_overflowing_parameter_proposal():
    def target(x):
        delta = x-10.
        return -.5*tf.reduce_sum(delta*delta, axis=-1), -delta, tf.ones([4], tf.bool)
    opt = NeuTraOptimizerConfig(4, "standard", 1000., .9, .999, 1.e-7, None)
    trainer = NeuTraTransportTrainer(NeuTraTransport(config("iaf", stages=1)), target, opt, target_signature="a"*64)
    before = trainer.checkpoint()
    result = trainer.train_step(rows())
    assert not bool(result["valid"])
    assert trainer.checkpoint() == before


def test_naf_inverse_density_gradient_compiles_for_weighted_training():
    transport = NeuTraTransport(dataclasses.replace(config(stages=1), hidden_layers=(4,)))
    @tf.function(input_signature=[tf.TensorSpec([4, 3], tf.float64)], jit_compile=True)
    def gradients(y):
        with tf.GradientTape() as tape:
            loss = -tf.reduce_mean(transport.log_prob(y))
        return loss, tape.gradient(loss, transport.trainable_variables)
    loss, grad = gradients(rows())
    assert bool(tf.math.is_finite(loss))
    assert all(g is not None and bool(tf.reduce_all(tf.math.is_finite(g))) for g in grad)


def test_actual_weighted_trainer_accepts_configured_naf():
    from bayesfilter.inference.neutra_weighted_training import WeightedForwardKLNeuTraTrainer, WeightedNeuTraConfig
    transport = NeuTraTransport(dataclasses.replace(config(stages=1), hidden_layers=(4,)))
    trainer = WeightedForwardKLNeuTraTrainer(WeightedNeuTraConfig(dimension=3, hidden_layers=(4,), stages=1), transport=transport)
    before = [tf.identity(v) for v in transport.trainable_variables]
    result = trainer.train_step(rows(), tf.zeros([4], tf.float64))
    assert int(result.step) == 1
    assert any(bool(tf.reduce_any(v != b)) for v, b in zip(transport.trainable_variables, before))


def test_tempered_checkpoint_reconstructs_configured_naf_and_affine_wrapper():
    from bayesfilter.inference.tempered_transport_ensemble_tf import (
        ReferenceAffineTransport, _trainable_transport_structure, _restore_trainable_transport_structure,
        transport_preflight_state_hash,
    )
    transport = ReferenceAffineTransport(NeuTraTransport(config()), center=[.2, -.3, .1],
                                         scale=[.5, 1., 2.], component_id="source-naf")
    structure = json.loads(json.dumps(_trainable_transport_structure(transport)))
    restored = _restore_trainable_transport_structure(structure)
    for old, new in zip(transport.trainable_variables, restored.trainable_variables, strict=True):
        new.assign(old)
    assert transport_preflight_state_hash(transport) == transport_preflight_state_hash(restored)
    np.testing.assert_array_equal(transport.forward_batch(rows()), restored.forward_batch(rows()))
    _, _, score = core.forward_proposal_score(transport, rows())
    y = transport.forward_batch(rows())
    with tf.GradientTape() as tape:
        tape.watch(y)
        q = transport.log_prob(y)
    np.testing.assert_allclose(score, tape.gradient(q, y), atol=1.e-9)


def test_path_score_reuses_each_layer_forward_once():
    transport = NeuTraTransport(config())
    counts = [0]*len(transport.components)
    # Permutations may be shared objects. Count only distinct trainable layers.
    for i, stage in enumerate(transport.stages):
        original = stage.forward_and_logdet
        def record(values, original=original, i=i):
            counts[i] += 1
            return original(values)
        stage.forward_and_logdet = record
    core.forward_proposal_score(transport, rows())
    assert counts[:len(transport.stages)] == [1]*len(transport.stages)


def test_frozen_naf_is_wired_through_actual_hmc_geometry_builder():
    from bayesfilter.inference.hmc_candidate_set_execution import _rebuild_geometry
    from bayesfilter.inference.posterior_adapter import ValueScoreCapability
    from bayesfilter.inference.hmc import stable_adapter_signature
    class Target:
        parameter_dim = 3
        def adapter_signature(self):
            return "a"*64
        def value_score_capability(self):
            return ValueScoreCapability(value_score_authority="graph_native", xla_hmc_ready=True,
                full_chain_xla_diagnostic_ready=True, target_scope="neutra-core-fixture",
                runtime_backend="tensorflow", evidence_path=__file__, nonclaims=("engineering only",))
        def log_prob_and_grad(self, x):
            return gaussian(x)[:2]
    target = Target()
    transport = NeuTraTransport(config())
    payload = transport.frozen_payload(target_signature=stable_adapter_signature(target))
    adapter, transforms = _rebuild_geometry(target, [{"kind": "frozen_transport", "artifact": payload}], "neutra-core-fixture")
    assert len(transforms) == 1
    with tf.GradientTape() as tape:
        x = rows()
        tape.watch(x)
        y, ld = transport.forward_and_logdet(x)
        expected = gaussian(y)[0]+ld
    expected_score = tape.gradient(expected, x)
    compiled = tf.function(adapter.log_prob_and_grad, input_signature=[tf.TensorSpec([4, 3], tf.float64)], jit_compile=True)
    actual, score = compiled(x)
    np.testing.assert_allclose(actual, expected, atol=1.e-11)
    np.testing.assert_allclose(score, expected_score, atol=1.e-11)


def test_existing_iaf_facades_cannot_contain_numerical_forks():
    from bayesfilter.inference import neutra_artifacts, neutra_training, neutra_training_legacy, neutra_weighted_training
    routes = [(neutra_training._TrainableDenseIAF, ("forward_and_logdet", "_network_with_diagnostics")),
        (neutra_training_legacy.TrainableDenseAutoregressiveIAF, ("forward_and_logdet",)),
        (neutra_weighted_training._DenseAutoregressiveStage, ("forward_and_logdet", "_network", "pullback_score", "logdet_score")),
        (neutra_artifacts._DenseAutoregressiveIAFComponent, ("forward_and_logdet", "_network", "pullback_score", "logdet_score"))]
    for cls, methods in routes:
        for name in methods:
            source = inspect.getsource(getattr(cls, name))
            assert "_transport_core." in source
            assert "tf.matmul" not in source and "tf.math.tanh" not in source
    # Repository discovery catches a new copied IAF/DSF numerical class unless
    # it is explicitly wired to the single authority or a diagnostic fixture.
    root = Path(neutra_training.__file__).parents[1]
    for path in root.rglob("*.py"):
        if "testing" in path.parts or path.name == "neutra_transport_core.py":
            continue
        text = path.read_text()
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and ("IAF" in node.name or "Autoregressive" in node.name or "SigmoidMixture" in node.name):
                numerical = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in ("_network", "scalar_value_logdet_score")]
                for method in numerical:
                    snippet = ast.get_source_segment(text, method)
                    assert "_transport_core" in snippet or "_network_with_diagnostics" in snippet, str(path)
