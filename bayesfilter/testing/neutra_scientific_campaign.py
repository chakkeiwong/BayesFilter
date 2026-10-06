"""Exploratory mixture training study using the shared canonical NeuTra code.

Evaluator truth is excluded from native learner interfaces. Teacher uncertainty
is estimated across independent complete populations, never by treating
resampled particles or Markov time steps as iid. All screens are exploratory.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from pathlib import Path

import tensorflow as tf

from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
from bayesfilter.inference.neutra_flow_smc_tf import FlowTransportSMC
from bayesfilter.inference.neutra_post_training import PostTrainingProbe
from bayesfilter.inference.neutra_transport import (
    NeuTraTransport, NeuTraTransportConfig, NeuTraTransportTrainer, NeuTraOptimizerConfig,
)
from bayesfilter.inference.neutra_warm_start_tf import (
    AnnealedSMC, GabrieProgram, SMCConfig, CandidateFailure, valid_log_weights,
)
from bayesfilter.inference.neutra_weighted_training import WeightedForwardKLNeuTraTrainer, WeightedNeuTraConfig
from bayesfilter.testing.neutra_generic_targets import ExactTargetEvaluator
from bayesfilter.testing.neutra_scientific_design import (
    METHODS, FIT_SEEDS, CALIBRATION_TARGETS, FINAL_TARGETS, TEACHER_REPLICATIONS,
    REFERENCE_ROWS, target_catalog, profile_candidates, student_candidates,
    DEFAULT_STUDENT_KIND,
)

F64 = tf.float64


def serializable(value):
    if tf.is_tensor(value) or isinstance(value, tf.Variable):
        return serializable(value.numpy().tolist())
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable(value), indent=2, allow_nan=False) + '\n')


def save_tensor(path, value):
    data = tf.io.serialize_tensor(value)
    tf.io.write_file(str(path), data)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flow_config(dimension, seed, width=64, *, kind=DEFAULT_STUDENT_KIND):
    if kind == 'naf_dsf':
        return NeuTraTransportConfig.huang_dsf(dimension, hidden_layers=(width,width),
            stages=3, mixture_components=4, seed=(int(seed),731))
    if kind != 'iaf':
        raise ValueError('unsupported scientific student family')
    width = max(dimension, math.ceil(width / dimension) * dimension)
    return dataclasses.replace(NeuTraTransportConfig.hoffman_author_iaf(
        dimension, conditional_scale_cap=2.0, seed=(int(seed), 731)), hidden_layers=(width, width))


class IsotropicProposal:
    def __init__(self, dimension, scale):
        self.dimension = int(dimension)
        self.scale = tf.constant(float(scale), F64)

    def log_prob(self, x):
        return -0.5 * tf.reduce_sum(tf.square(x / self.scale), 1) - self.dimension * (
            0.5 * math.log(2.0 * math.pi) + tf.math.log(self.scale))

    def sample(self, count, seed):
        return self.scale * tf.random.stateless_normal([int(count), self.dimension], seed, dtype=F64)


def mean_variance(rows):
    """Unbiased variance estimate for the mean of independent rows."""
    n = tf.cast(tf.shape(rows)[0], F64)
    mean = tf.reduce_mean(rows, 0)
    return mean, tf.reduce_sum(tf.square(rows - mean), 0) / (n * (n - 1.0))


def teacher_screen(evaluator, banks, reference, *, threshold=5.0):
    """banks contains independent (particles, log_weights) populations.

    Independence is between populations. Particle counts and weight ESS are
    NEVER used as the number of independent observations for the feature SE.
    """
    if len(banks) < 2:
        raise ValueError('teacher uncertainty requires independent populations')
    summaries, fractions = [], []
    finite = True
    for x, lw in banks:
        features = evaluator.feature_program(x)
        w = tf.nn.softmax(lw)
        summaries.append(tf.reduce_sum(w[:, None] * features, 0))
        fractions.append(1.0 / (tf.reduce_sum(w*w) * tf.cast(tf.size(w), F64)))
        finite &= bool(tf.reduce_all(tf.math.is_finite(features)).numpy()) and bool(valid_log_weights(lw).numpy())
    summary, teacher_variance = mean_variance(tf.stack(summaries))
    ref, reference_variance = mean_variance(evaluator.feature_program(reference))
    se = tf.sqrt(teacher_variance + reference_variance)
    difference = tf.abs(summary-ref)
    z = tf.where(se > 0, tf.math.divide_no_nan(difference, se),
                 tf.where(difference == 0, tf.zeros_like(se), tf.fill(tf.shape(se), tf.constant(math.inf, F64))))
    k = len(evaluator.specification.get('weights', []))
    mass_error = float(tf.reduce_max(difference[:k]).numpy()) if k else 0.0
    ess_fraction = float(tf.reduce_min(fractions).numpy())
    finite &= bool(tf.reduce_all(tf.math.is_finite(summary)).numpy())
    return serializable({
        'passed': finite and ess_fraction >= 0.20 and mass_error <= 0.15 and bool(tf.reduce_all(z <= threshold).numpy()),
        'finite': finite, 'replications': len(banks), 'rows_per_replication': [int(x.shape[0]) for x, _ in banks],
        'summary': summary, 'reference_summary': ref, 'population_summaries': tf.stack(summaries),
        'teacher_mean_variance': teacher_variance, 'reference_mean_variance': reference_variance,
        'combined_standard_error': se, 'summary_z_max': tf.reduce_max(z), 'summary_z': z,
        'maximum_responsibility_discrepancy': mass_error, 'ess_fraction': ess_fraction,
        'population_ess_fractions': fractions,
        'screen_thresholds': {'ess_fraction': 0.20, 'responsibility_discrepancy': 0.15, 'summary_z_max': threshold},
        'uncertainty_unit': 'independent_complete_populations',
        'inference_status': 'exploratory_screen_four_population_variance_is_noisy_no_equilibrium_claim',
    })


def native_banks(method, target, profile, seed, *, repetitions=TEACHER_REPLICATIONS, jit_compile=True):
    """Learner sees only DensityTarget; sampler instances reuse compiled kernels."""
    if method == 'fab':
        return [], {'status': 'prerequisite_failed', 'native_training': 'not_run',
                    'reason': 'nonlinear alpha-two tail eligibility unresolved; FAB not tested'}
    proposal = IsotropicProposal(target.parameter_dim, profile['proposal_scale'])
    config = flow_config(target.parameter_dim, seed)
    if method in ('ais', 'smc'):
        cfg = SMCConfig(profile['particles'], profile['mutation_steps'], profile['stages'],
                        0.8, 0.5, profile['step_size'], jit_compile=jit_compile,
                        temperature_schedule=tuple(i/profile['stages'] for i in range(profile['stages']+1)),
                        use_resampling=method == 'smc')
        sampler = AnnealedSMC(target, proposal, cfg)
        sample = sampler.run
        route = 'AnnealedSMC.fixed_program'
    elif method in ('aft', 'craft'):
        sampler = FlowTransportSMC(target, proposal, config, particles=profile['particles'],
            stages=profile['stages'], inner_updates=profile['inner_updates'], passes=profile['passes'],
            learning_rate=profile['learning_rate'], gradient_clip=1000.0,
            mutation_steps=profile['mutation_steps'], step_size=profile['step_size'],
            resampling_fraction=0.5, jit_compile=jit_compile)
        # Both public controllers reset all map/Adam states to sampler.initial
        # before each new complete population; no fitted state crosses seeds.
        sample = sampler.run_aft if method == 'aft' else sampler.run_craft
        route = 'FlowTransportSMC.' + ('run_aft' if method == 'aft' else 'run_craft')
    elif method == 'gabrie':
        flow = NeuTraTransport(config)
        sampler = GabrieProgram(flow, target, walkers=profile['walkers'],
                                steps=profile['walker_steps'], jit_compile=jit_compile)
        def sample(key):
            x = proposal.sample(profile['walkers'], key)
            banks, details = [], []
            for iteration in range(max(4, profile['walker_steps'])):
                x, bank, ga, la, invalid = sampler.run(x, tf.constant(profile['step_size'], F64),
                    tf.random.experimental.stateless_fold_in(key, iteration+1))
                banks.append(bank)
                details.append({'global_acceptance': ga, 'local_acceptance': la, 'invalid': invalid})
            bank = tf.concat(banks, 0)
            return {'particles': bank, 'log_weights': tf.zeros([int(bank.shape[0])], F64),
                    'complete': True, 'steps': details}
        route = 'GabrieProgram.fixed_map_global_MH_then_MALA_control'
    else:
        raise ValueError('unknown native teacher method')
    banks, details = [], []
    for replicate in range(repetitions):
        key = tf.constant([int(seed), 4001+replicate], tf.int32)
        result = sample(key)
        if not result.get('complete'):
            raise CandidateFailure('native teacher did not reach endpoint')
        banks.append((result['particles'], result['log_weights']))
        detail = {k: v for k, v in result.items() if k not in ('particles', 'log_weights', 'roots')}
        if 'roots' in result:
            detail['unique_roots'] = tf.size(tf.unique(result['roots']).y)
        details.append({'seed': [int(seed), 4001+replicate], 'result': detail})
    return banks, serializable({'status': 'native_complete', 'source_route': route,
        'replications': details, 'full_controller_equivalence': 'not_established' if method in ('gabrie','aft','craft') else 'local_fixed_schedule_implementation'})


def _train_block(trainer, pool, log_weights, updates, seed, *, kind='forward', batch=64, jit_compile=True):
    dimension = trainer.transport.parameter_dim
    def run(rows, weights, key, count, offset):
        def body(index, total_loss, first_loss, last_loss, norm_max, clipped, valid):
            subkey = tf.random.experimental.stateless_fold_in(key, index+offset)
            if kind == 'forward':
                draw = tf.random.stateless_categorical(weights[None, :], batch, subkey)[0]
                step = trainer._train_step_impl(tf.stop_gradient(tf.gather(rows, draw)), tf.zeros([batch], F64))
                loss, norm, clipnorm, okay = step[0], step[4], step[5], step[-1]
            else:
                z = tf.random.stateless_normal([batch, dimension], subkey, dtype=F64)
                step = trainer._train_step(z)
                loss, norm, clipnorm, okay = step['loss'], step['gradient_norm'], step['clipped_gradient_norm'], step['valid']
            variables = (*trainer.variables, *trainer.optimizer.variables)
            okay &= tf.reduce_all(tf.stack([tf.reduce_all(tf.math.is_finite(v))
                                            for v in variables if tf.as_dtype(v.dtype).is_floating]))
            return (index+1, total_loss+loss, tf.where(index == 0, loss, first_loss), loss,
                    tf.maximum(norm_max, norm), clipped+tf.cast(norm > clipnorm*(1+1e-12), tf.int32), valid & okay)
        return tf.while_loop(lambda index, *rest: (index < count) & rest[-1], body,
            (tf.constant(0), tf.constant(0., F64), tf.constant(0., F64), tf.constant(0., F64),
             tf.constant(0., F64), tf.constant(0), tf.constant(True)))
    cache = getattr(trainer, '_campaign_programs', {})
    cache_key = (kind, tuple(pool.shape), tuple(log_weights.shape), batch, jit_compile)
    if cache_key not in cache:
        cache[cache_key] = tf.function(run, input_signature=[tf.TensorSpec(pool.shape, F64),
            tf.TensorSpec(log_weights.shape, F64), tf.TensorSpec([2], tf.int32),
            tf.TensorSpec([],tf.int32), tf.TensorSpec([],tf.int32)],jit_compile=jit_compile,autograph=False)
        trainer._campaign_programs = cache
    compiled = cache[cache_key]
    offset = int(trainer.optimizer.iterations.numpy())
    key = [int(seed), 7001 if kind == 'forward' else 7002]
    value = compiled(pool,log_weights,tf.constant(key,tf.int32),tf.constant(updates),tf.constant(offset))
    report = serializable({'updates': value[0], 'requested_updates': updates, 'mean_loss': value[1]/tf.cast(value[0], F64),
        'first_batch_loss': value[2], 'last_batch_loss': value[3], 'max_gradient_norm': value[4],
        'clipped_updates': value[5], 'finite': value[6], 'batch_size': batch,
        'jit_compile': jit_compile, 'training_device': value[1].device, 'traces': compiled.experimental_get_tracing_count(),
        'samplewise_loop': False, 'numpy_training_path': False, 'stateless_seed': key, 'next_rng_counter': value[0]+offset})
    report['clipping_repair_trigger'] = report['clipped_updates'] > report['updates']/2
    return report


def evaluate_student(evaluator, flow, reference, *, seed, sample_rows=2048, jit_compile=True):
    with tf.device('/CPU:0'):
        z = tf.random.stateless_normal([sample_rows, evaluator.dimension], [int(seed), 9011], dtype=F64)
    @tf.function(input_signature=[tf.TensorSpec(reference.shape, F64), tf.TensorSpec(z.shape, F64)],
                 jit_compile=jit_compile, autograph=False)
    def evaluate(ref, latent):
        qref = flow.log_prob(ref)
        baseline = -0.5*(tf.reduce_sum(ref*ref, 1)+evaluator.dimension*math.log(2*math.pi))
        physical, _ = flow.forward_and_logdet(latent)
        qlog, plog = flow.log_prob(physical), evaluator.target.log_prob_kernel(physical)
        qmean, qvar = mean_variance(evaluator._features(physical))
        rmean, rvar = mean_variance(evaluator._features(ref))
        delta = -qref+baseline
        kl = evaluator.target.log_prob_kernel(ref)-qref
        return qref, qlog, plog, qmean, qvar, rmean, rvar, delta, kl
    qref, qlog, plog, qm, qv, rm, rv, delta, kl = evaluate(reference, z)
    se = tf.sqrt(qv+rv)
    diff = tf.abs(qm-rm)
    score = tf.where(se > 0, tf.math.divide_no_nan(diff, se),
                     tf.where(diff == 0, tf.zeros_like(se), tf.fill(tf.shape(se), tf.constant(math.inf, F64))))
    k = len(evaluator.specification.get('weights', []))
    mass_error = float(tf.reduce_max(diff[:k]).numpy()) if k else 0.0
    finite = all(bool(tf.reduce_all(tf.math.is_finite(v)).numpy()) for v in (qref,qlog,plog,qm,qv,rm,rv,delta,kl))
    return serializable({'heldout_cross_entropy': -tf.reduce_mean(qref),
        'identity_cross_entropy': tf.reduce_mean(-qref-delta), 'cross_entropy_delta': tf.reduce_mean(delta),
        'cross_entropy_delta_standard_error': tf.sqrt(mean_variance(delta)[1]),
        'forward_kl_estimate': tf.reduce_mean(kl), 'reverse_kl_estimate': tf.reduce_mean(qlog-plog),
        'q_summary': qm, 'reference_summary': rm, 'q_mean_variance': qv, 'reference_mean_variance': rv,
        'combined_standard_error': se, 'summary_z_max': tf.reduce_max(score), 'summary_z': score,
        'maximum_responsibility_discrepancy': mass_error, 'finite': finite,
        'reference_rows': int(reference.shape[0]), 'map_rows': sample_rows,
        'uncertainty_unit': 'iid_map_draws_conditional_on_frozen_map_and_independent_reference',
        'screen_thresholds': {'cross_entropy_delta': 0.25, 'summary_z_max': 5., 'responsibility_discrepancy': 0.15}})


def freeze_and_assess(output, label, trainer, evaluator, reference, seed, *, jit_compile=True):
    checkpoint = trainer.state_payload() if hasattr(trainer, 'state_payload') else trainer.checkpoint()
    write(output / (label+'-checkpoint.json'), checkpoint)
    flow = trainer.transport
    payload = flow.frozen_payload(target_signature=evaluator.target.signature,
        training_state_hash=checkpoint.get('state_hash', checkpoint.get('checkpoint_hash')))
    path = output / (label+'-frozen.json')
    write(path, payload)
    loaded = load_frozen_neutra_artifact(json.loads(path.read_text()), expected_target_signature=evaluator.target.signature)
    heldout = evaluate_student(evaluator, loaded.transport, reference, seed=seed, jit_compile=jit_compile)
    probe = PostTrainingProbe(loaded.transport, evaluator.target, 1.0, jit_compile=jit_compile)((int(seed),8001))
    write(output / (label+'-heldout.json'), heldout)
    write(output / (label+'-post-training-1000.json'), probe)
    passed = (heldout['finite'] and heldout['cross_entropy_delta'] <= 0.25
        and heldout['summary_z_max'] is not None and heldout['summary_z_max'] <= 5.
        and heldout['maximum_responsibility_discrepancy'] <= 0.15
        and probe['complete'] and probe['finite'] and probe['valid_rows'] == 1000)
    return {'passed': bool(passed), 'heldout': heldout, 'transport_hash': payload['transport_hash'],
            'checkpoint_reloaded': True, 'probe_path': str(output / (label+'-post-training-1000.json'))}


def run_trial(method, target_label, specification, profile, seed, output, *, role, exact_teacher=False, jit_compile=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with tf.device('/CPU:0'):
        evaluator = ExactTargetEvaluator(specification, jit_compile=jit_compile)
        reference = evaluator.sample(REFERENCE_ROWS, tf.constant([int(seed),1001], tf.int32))
        heldout_reference = evaluator.sample(REFERENCE_ROWS, tf.constant([int(seed),1002], tf.int32))
    write(output/'target.json', specification)
    references = {'teacher': save_tensor(output/'teacher-reference.tensor', reference),
                  'map': save_tensor(output/'map-reference.tensor', heldout_reference)}
    base = {'method': method, 'target': target_label, 'target_signature': evaluator.target.signature,
            'role': role, 'seed': seed, 'scientific_promotion': False, 'teacher_admitted': False,
            'sampling_quality': 'not_established', 'downstream_hmc': 'not_run',
            'reference_sha256': references, 'reference_seeds': [[seed,1001],[seed,1002]],
            'profile': profile, 'dtype': 'float64_benchmark_reference_exception'}
    if exact_teacher:
        with tf.device('/CPU:0'):
            banks = [(evaluator.sample(1024, tf.constant([int(seed),2001+r],tf.int32)), tf.zeros([1024],F64))
                     for r in range(TEACHER_REPLICATIONS)]
        native = {'status': 'exact_iid_teacher', 'source_route': 'ExactTargetEvaluator.sample',
                  'seeds': [[seed,2001+r] for r in range(TEACHER_REPLICATIONS)]}
    else:
        try:
            banks, native = native_banks(method, evaluator.target, profile, seed, jit_compile=jit_compile)
        except CandidateFailure as error:
            return {**base, 'status': 'native_failed', 'reason': str(error)}
    write(output/'native-summary.json', native)
    if not banks:
        return {**base, 'status': 'prerequisite_failed', 'native': native, 'reason': native['reason']}
    hashes = []
    for i,(x,lw) in enumerate(banks):
        hashes.append({'particles': save_tensor(output/f'teacher-{i}.tensor', x),
                       'weights': save_tensor(output/f'teacher-weights-{i}.tensor', lw)})
    teacher = teacher_screen(evaluator, banks, reference)
    write(output/'teacher-summary.json', teacher)
    base.update(native=native, teacher=teacher, teacher_bank_sha256=hashes)
    if not teacher['passed']:
        return {**base, 'status': 'teacher_failed'}
    student = profile.get('student', profile)
    pool = tf.concat([x for x,_ in banks],0)
    lw = tf.concat([tf.nn.log_softmax(w)-math.log(len(banks)) for _,w in banks],0)
    flow = NeuTraTransport(flow_config(evaluator.dimension, seed, student['width']))
    trainer = WeightedForwardKLNeuTraTrainer(WeightedNeuTraConfig(dimension=evaluator.dimension,
        hidden_layers=flow.config.hidden_layers, stages=3, learning_rate=student['learning_rate'],
        beta1=.9, beta2=.999, epsilon=1e-8, gradient_clip_norm=1000., jit_compile=jit_compile), transport=flow)
    history = []
    completed = 0
    requested = int(student['updates'])
    rungs = sorted(set([n for n in (512,1024,2048,4096,8192) if n < requested]+[requested]))
    for rung in rungs:
        timing = time.monotonic()
        training = _train_block(trainer,pool,lw,rung-completed,seed,jit_compile=jit_compile)
        elapsed = time.monotonic()-timing
        completed += training['updates']
        timing = time.monotonic()
        assessment = evaluate_student(evaluator,flow,heldout_reference,seed=seed,jit_compile=jit_compile)
        history.append({'cumulative_updates':completed,'training':training,'heldout':assessment,
                        'training_wall_seconds':elapsed,'assessment_wall_seconds':time.monotonic()-timing})
        write(output/'forward-learning-history.json',history)
        if not training['finite'] or training['clipping_repair_trigger']:
            return {**base,'teacher_admitted':True,'status':'fit_failed',
                    'reason':'forward_numerical_or_majority_clipping_trigger','learning_history':history}
    training = {**training,'updates':completed,'requested_updates':requested,
                'clipped_updates':sum(r['training']['clipped_updates'] for r in history),
                'max_gradient_norm':max(r['training']['max_gradient_norm'] for r in history)}
    write(output/'forward-training.json',training)
    base.update(teacher_admitted=True,sampling_quality='exploratory_teacher_screen_passed',
                forward_training=training,learning_history=history)
    forward = freeze_and_assess(output,'forward',trainer,evaluator,heldout_reference,seed,jit_compile=jit_compile)
    rkl = NeuTraTransportTrainer(flow, evaluator.target.value_score, NeuTraOptimizerConfig(
        64,'standard',student['rkl_learning_rate'],.9,.999,1e-8,1000.,jit_compile),target_signature=evaluator.target.signature)
    rkl_training = _train_block(rkl,pool,lw,student['rkl_updates'],seed,kind='rkl',jit_compile=jit_compile)
    write(output/'rkl-training.json', rkl_training)
    base.update(forward=forward, rkl_training=rkl_training)
    if not rkl_training['finite'] or rkl_training['clipping_repair_trigger']:
        return {**base, 'status':'fit_failed','reason':'reverse_numerical_or_majority_clipping_trigger'}
    final = freeze_and_assess(output,'student',rkl,evaluator,heldout_reference,seed,jit_compile=jit_compile)
    return {**base, 'status':'passed' if final['passed'] else 'map_failed', 'final':final,
            'heldout':final['heldout'], 'wall_seconds':time.monotonic()-started}
