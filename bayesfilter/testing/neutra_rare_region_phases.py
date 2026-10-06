"""Phases of the bounded rare-region research campaign (TF/TFP only)."""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.testing.neutra_warm_start_targets_tf import WarmStartTarget, F64
from bayesfilter.testing.neutra_warm_start_campaign import (
    write_json, save_tensor, read_tensor, make_transport, TrainingBlock)
from bayesfilter.testing.neutra_rare_regions_tf import (
    Proposal, ImportanceProgram, GlobalProgram, UmbrellaProgram, EmusProgram,
    ConstrainedProgram, event_features, known_masses, seed_fold, stratified_bank)
from bayesfilter.inference.neutra_warm_start_tf import systematic_indices
from bayesfilter.inference.neutra_weighted_training import WeightedForwardKLNeuTraTrainer, WeightedNeuTraConfig
from bayesfilter.inference.neutra_post_training import PostTrainingProbe


def key(seed, role):
    return tf.constant([int(seed), int(role)], tf.int32)


def load_flow(path, target):
    from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
    payload = json.loads(Path(path).read_text())
    return load_frozen_neutra_artifact(payload, expected_target_signature=target.signature).transport


def bank_report(target, rows, lw):
    weights = tf.nn.softmax(lw)
    values = event_features(target, rows)
    return {'estimate': tf.reduce_sum(weights[:, None]*values, 0),
        'rows': int(tf.shape(rows)[0]), 'event_rows': tf.reduce_sum(tf.cast(values[:, :3] != 0., tf.int32), 0),
        'weight_ess': 1/tf.reduce_sum(weights**2), 'maximum_weight': tf.reduce_max(weights),
        'finite': bool(tf.reduce_all(tf.math.is_finite(rows)).numpy()) and
                  bool(tf.reduce_all(tf.math.is_finite(weights)).numpy()),
        'weight_ess_is_not_event_precision': True}


def save_bank(output, rows, lw):
    save_tensor(Path(output)/'rows.tensor', rows)
    save_tensor(Path(output)/'log-weights.tensor', tf.nn.log_softmax(lw))


def probability_screen(estimates):
    estimates = tf.constant(estimates, F64)
    n = tf.cast(tf.shape(estimates)[0], F64)
    mean = tf.reduce_mean(estimates, 0)
    se = tf.math.reduce_std(estimates, 0)/tf.sqrt(n-1.)
    truth = tf.constant(known_masses(), F64)
    rse = se[:3]/truth[:3]
    tol = tf.maximum(4*se[:3], .1*truth[:3])
    seen = tf.reduce_all(estimates[:, :3] > 0., 0)
    passed = seen & (rse <= .2) & (tf.abs(mean[:3]-truth[:3]) <= tol)
    continuous_sd = tf.constant([math.sqrt(26.-25/9), 1., math.sqrt(102.), math.sqrt(2.)], F64)
    moment_passes = tf.abs(mean[3:]-truth[3:]) <= 4*se[3:]+.03*continuous_sd
    return {'mean': mean, 'replicate_se': se, 'exact': truth, 'relative_se': rse,
        'event_seen_in_every_replication': seen, 'probability_passes': passed,
        'moment_passes': moment_passes,
        'passed': bool(tf.reduce_all(passed).numpy()) and bool(tf.reduce_all(moment_passes).numpy()), 'replications': int(n),
        'statistically_supported_method_ranking': False,
        'uncertainty_scope': 'independent-run standard errors; working screen, no exact coverage guarantee'}


def prepare(target_name, output, cfg):
    target = WarmStartTarget(target_name)
    out = Path(output)
    sampler = tf.function(lambda seed: target.reference_sample(32768, seed),
        input_signature=[tf.TensorSpec([2], tf.int32)], jit_compile=True, autograph=False)
    save_tensor(out/'validation.tensor', sampler(key(71001, 1))[:8192])
    save_tensor(out/'confirmation.tensor', sampler(key(71001, 2)))
    # Reserved data never enters local-bank or proposal construction.
    proposal = Proposal(target)
    pilot = GlobalProgram(target, proposal, 128)
    starts = tf.repeat(target.known_representatives(), 32, axis=0)
    trials = []
    selected_dt = None
    for dt in (.002, .01, .05):
        last, trace, ga, la, bad = pilot.run(starts, tf.constant(dt, F64), key(73001, int(dt*10000)))
        trials.append({'dt': dt, 'global_acceptance': ga, 'local_acceptance': la, 'invalid': bad})
        if int(bad) == 0 and float(la) >= .5:
            selected_dt = dt
    if selected_dt is None:
        raise ValueError('no valid physical MALA pilot')
    for side in (-1, 1):
        start = tf.repeat(target.known_representatives()[0 if side < 0 else 1][None, :], 32, 0)
        burn = GlobalProgram(target, proposal, 512, halfspace=side)
        keep = GlobalProgram(target, proposal, 256, halfspace=side)
        last, _, _, _, bad = burn.run(start, tf.constant(selected_dt, F64), key(75001, side+2))
        _, trace, _, acc, bad2 = keep.run(last, tf.constant(selected_dt, F64), key(76001, side+2))
        if int(bad+bad2):
            raise ValueError('invalid local training bank')
        save_tensor(out/f'local-{side}.tensor', tf.reshape(trace, [-1, 2]))
    baseline = ImportanceProgram(target, proposal, cfg['particles'])
    baseline_rows = []
    iid_rows = []
    for seed in cfg['replication_seeds']:
        _, _, estimate, _ = baseline.run(key(seed, 78001))
        baseline_rows.append(estimate.numpy().tolist())
        iid_rows.append(tf.reduce_mean(event_features(target, sampler(key(seed, 79001))[:cfg['particles']]), 0).numpy().tolist())
    report = {'target': target.specification, 'signature': target.signature,
        'mala_dt': selected_dt, 'pilot': trials, 'reference_role': 'validation/final-only, never teacher or proposal fitting',
        'iid_baseline': probability_screen(iid_rows), 'laplace_baseline': probability_screen(baseline_rows),
        'known_mode_locations_supplied': True, 'known_mode_weights_supplied': False}
    write_json(out/'preparation.json', report)
    return report


class StratifiedTrainer:
    """Fixed bank, conditional sampling, exactly normalized stratum weights."""
    def __init__(self, flow, rows, lw, lr, *, batch=256):
        indices, logs, logmass = stratified_bank(rows, lw)
        k = int(indices.shape[0])
        counts = [batch//k + (i < batch % k) for i in range(k)]
        def draw(seed):
            xs, weights = [], []
            for i, count in enumerate(counts):
                picked = tf.random.stateless_categorical(logs[i:i+1], count, seed_fold(seed, i))[0]
                xs.append(tf.gather(rows, tf.gather(indices[i], picked)))
                weights.append(tf.fill([count], logmass[i]-tf.math.log(tf.cast(count, F64))))
            return tf.concat(xs, 0), tf.concat(weights, 0)
        self.draw = tf.function(draw, input_signature=[tf.TensorSpec([2], tf.int32)], jit_compile=True, autograph=False)
        @tf.function(input_signature=[tf.TensorSpec([2], tf.int32)], jit_compile=True, autograph=False)
        def gradient(seed):
            x, logw = draw(seed)
            with tf.GradientTape() as tape:
                loss = -tf.reduce_sum(tf.exp(logw)*flow.log_prob(x))
            return tf.linalg.global_norm(tape.gradient(loss, flow.trainable_variables))
        norms = tf.stack([gradient(key(91001, i)) for i in range(8)])
        clip = max(float(tf.reduce_max(norms))*5, 1e-8)
        self.calibration = {'norms': norms, 'clip': clip, 'provenance': 'five times maximum of eight weighted stratified gradient pilots',
            'strata_masses': tf.exp(logmass), 'batch_allocations': counts, 'batch': batch}
        trainer = WeightedForwardKLNeuTraTrainer(WeightedNeuTraConfig(
            dimension=2, hidden_layers=flow.config.hidden_layers, stages=3, learning_rate=lr,
            gradient_clip_norm=clip, jit_compile=True), transport=flow)
        self.trainer = trainer
        def run(seed, count):
            def body(i, loss, norm, clips, valid):
                x, weights = draw(seed_fold(seed, i))
                r = trainer._train_step_impl(x, weights)
                finite = r[-1] & tf.reduce_all(tf.stack([tf.reduce_all(tf.math.is_finite(v)) for v in flow.trainable_variables]))
                return i+1, loss+r[0], norm+r[4], clips+tf.cast(r[6], F64), valid & finite
            return tf.while_loop(lambda i, _l, _n, _c, valid: (i < count) & valid, body,
                (tf.constant(0), tf.constant(0., F64), tf.constant(0., F64), tf.constant(0., F64), tf.constant(True)))
        self.run = tf.function(run, input_signature=[tf.TensorSpec([2], tf.int32), tf.TensorSpec([], tf.int32)],
                               jit_compile=True, autograph=False)


def fit_bank(target, rows, lw, validation, output, seed, cfg, *, local=False):
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    best, best_loss, best_path = None, math.inf, None
    trials = []
    for width in cfg['widths']:
        for lr in cfg['learning_rates']:
            flow = make_transport(target, width, (seed, width), variance_scale=.2)
            # A fitted mean is an initialization, not an alternative map layer.
            mean = tf.reduce_sum(tf.nn.softmax(lw)[:, None]*rows, 0)
            bias = flow.stages[-1].biases[-1]
            bias.assign(tf.concat((bias[:2], mean), 0))
            program = StratifiedTrainer(flow, rows, lw, lr)
            candidate = out/f'w{width}-lr{lr}'
            candidate.mkdir(exist_ok=False)
            write_json(candidate/'gradient-calibration.json', program.calibration)
            history = []
            evaluate = tf.function(lambda x: -tf.reduce_mean(flow.log_prob(x)),
                input_signature=[tf.TensorSpec([None, 2], F64)], jit_compile=True, autograph=False)
            result = program.run(key(seed, 93001), tf.constant(cfg['pilot_updates']))
            valid = bool(result[-1]) and int(result[0]) == cfg['pilot_updates']
            loss = float(evaluate(validation)) if valid else math.inf
            history.append({'updates': int(result[0]), 'loss': loss, 'clipped_fraction': float(result[3])/max(1, int(result[0]))})
            write_json(candidate/'pilot.json', {'history': history, 'valid': valid})
            if valid and math.isfinite(loss) and loss < best_loss:
                best, best_loss, best_path = (flow, program, lr), loss, candidate
            trials.append({'path': str(candidate), 'width': width, 'lr': lr, 'validation_cross_entropy': loss, 'valid': valid})
    if best is None:
        raise ValueError('all target-specific training pilots invalid')
    flow, program, lr = best
    result = program.run(key(seed, 94001), tf.constant(cfg['forward_updates']-cfg['pilot_updates']))
    if not bool(result[-1]):
        raise ValueError('nonfinite selected training continuation')
    report = {'pilots': trials, 'selected': str(best_path), 'selection': 'minimum heldout FKL for nomination only',
        'architecture': flow.config.payload(), 'forward_updates': cfg['forward_updates'],
        'forward_clip_fraction': result[3]/tf.cast(tf.maximum(result[0], 1), F64),
        'batch': cfg['batch_size'], 'batch_native': True, 'jit_compile': True,
        'variance_scale': .2, 'variance_scale_status': 'prior same-target repair warm-start hypothesis',
        'canonical_core': 'bayesfilter_neutra_iaf_author_v1', 'local': local}
    write_json(out/'forward-frozen.json', flow.frozen_payload(target_signature=target.signature))
    if local:
        probe = PostTrainingProbe(flow, target, 1.)(seed=(seed, 95001))
        probe['interpretation'] = 'local map against full-target score; no global mode coverage claim'
        write_json(out/'local-probe.json', probe)
        report['finite_probe'] = probe['finite']
    else:
        checkpoints = []
        def inspect(name):
            frozen = flow.frozen_payload(target_signature=target.signature)
            write_json(out/f'{name}-frozen.json', frozen)
            probe = PostTrainingProbe(flow, target, 1.)(seed=(seed, 95001))
            probe['transport_hash'] = frozen['transport_hash']
            write_json(out/f'{name}-probe.json', probe)
            directed = directed_geometry(flow, target)
            write_json(out/f'{name}-directed.json', directed)
            checkpoints.append({'name': name, 'frozen': str(out/f'{name}-frozen.json'),
                'finite': probe['finite'] and directed['finite'], 'probe': str(out/f'{name}-probe.json')})
        inspect('forward')
        rkl = TrainingBlock(flow, target, batch=cfg['batch_size'], learning_rate=lr, clip=None,
            kind='rkl', walkers=4, walk_steps=1)
        total = 0
        for rung in cfg['reverse_rungs']:
            result = rkl.run(key(seed, 96001+total), tf.constant(rung-total), rows, lw,
                tf.zeros([4, 2], F64), tf.constant(.01, F64))
            if not bool(result[-1]):
                report['rkl_failure'] = 'nonfinite candidate; earlier checkpoints retained'; break
            total = rung
            inspect(f'rkl-{rung}')
        report['checkpoints'] = checkpoints
        # Fixed latest-first shortlist, never chosen from final reference.
        report['qualified_checkpoint_order'] = list(reversed([c['name'] for c in checkpoints if c['finite']]))
        candidates = []
        for c in reversed(checkpoints):
            frozen = json.loads(Path(c['frozen']).read_text())
            candidates.append({'stage': c['name'], 'filename': c['name']+'-frozen.json',
                'eligible': c['finite'], 'probe_file': c['name']+'-probe.json',
                'transport_hash': frozen['transport_hash']})
        write_json(out/'checkpoint-candidates.json', {'schema': 'neutra.checkpoint_candidates.v1',
            'selection_order': 'latest RKL, earlier RKL, forward; downstream screens, one final holdout',
            'candidates': candidates})
    write_json(out/'fit.json', report)
    return report


def directed_geometry(flow, target, *, offsets=(0.,)):
    cache=getattr(flow,'_directed_geometry_programs',{})
    key=(target.signature,tuple(offsets))
    @tf.function(input_signature=[], jit_compile=True, autograph=False)
    def run():
        axis = tf.linspace(tf.constant(-2., F64), tf.constant(2., F64), 1001)
        y = .1*(axis*axis-26.) if target.name == 'warped_mixture' else tf.zeros_like(axis)
        x = tf.concat([tf.stack((axis, y+offset), 1) for offset in offsets],0)
        z, _ = flow.inverse_and_forward_logdet(x)
        _, score, valid = target.value_score(x)
        residual = flow.pullback_score_batch(z, score)+flow.log_abs_det_jacobian_score_batch(z)+z
        back, _ = flow.forward_and_logdet(z)
        return x, z, residual, tf.reduce_max(tf.abs(x-back)), tf.reduce_all(valid)
    if key not in cache:
        cache[key]=run
        flow._directed_geometry_programs=cache
    x, z, r, error, valid = cache[key]()
    finite = bool(valid) and bool(tf.reduce_all(tf.math.is_finite(r))) and float(error) < 1e-8
    return {'physical': x, 'latent': z, 'residual': r, 'roundtrip_error': error,
        'finite': finite, 'maximum_norm': tf.reduce_max(tf.linalg.norm(r, axis=1)),
        'conditional_offsets':list(offsets),
        'role': 'directed explanatory valley slices, not posterior draws or uniform bounds'}


def local_fit(target_name, prepared, output, cfg):
    target = WarmStartTarget(target_name)
    results = []
    for side in (-1, 1):
        rows = read_tensor(Path(prepared)/f'local-{side}.tensor')
        # Last temporal block is validation; both remain correlated local data.
        result = fit_bank(target, rows[:6144], tf.zeros([6144], F64), rows[6144:],
            Path(output)/f'component-{side}', 11001+side, cfg, local=True)
        results.append(result)
    return {'components': results, 'data_role': 'halfspace-local MCMC, no exact reference rows',
            'no_local_sampler_convergence_proof': True}


def run_method(target_name, method, prepared, local, output, cfg, repair=0):
    target = WarmStartTarget(target_name); out = Path(output)
    dt = json.loads((Path(prepared)/'preparation.json').read_text())['mala_dt']
    n = cfg['particles']*(2 if repair else 1)
    estimates, diagnostics = [], []
    proposal = Proposal(target)
    extra = {}
    if method == 'importance':
        trials = []
        selected, risk = None, math.inf
        for b in (.2, .5):
            p = Proposal(target, bridge_weight=b)
            prog = ImportanceProgram(target, p, n)
            _, _, estimate, se = prog.run(key(12001+repair, 1))
            # Nomination uses pilot estimates, never the exact event masses.
            relative = float(tf.reduce_max(se[:2]/estimate[:2])) if bool(tf.reduce_all(estimate[:2] > 0.)) else math.inf
            trials.append({'bridge_weight': b, 'estimated_relative_se': relative})
            if relative < risk:
                selected, risk = prog, relative
        extra['pilot'] = trials
    elif method == 'umbrella':
        spacing = .2 if repair else .4
        centers = [-8+i*spacing for i in range(round(16/spacing)+1)] + [0.]
        scales = [.2]*(len(centers)-1)+[.05]
        initial = tf.repeat(tf.constant(centers, F64), 4)
        xstart = tf.stack((initial, .1*(initial**2-26.) if target_name == 'warped_mixture' else tf.zeros_like(initial)), 1)
        burn = UmbrellaProgram(target, centers, scales, 4, 2048 if repair else 512)
        keep = UmbrellaProgram(target, centers, scales, 4, 4096 if repair else 1024)
        emus = EmusProgram(centers, scales)
        pilot = UmbrellaProgram(target, centers, scales, 4, 128)
        choices = []
        umbrella_dt = None
        # Narrow window has precision400, so add a derived 1/400 step hypothesis.
        for step in (.0005, .002, .005):
            last, _, acc, bad = pilot.run(xstart, tf.constant(step, F64), key(13001+repair, int(step*100000)))
            choices.append({'dt': step, 'minimum_acceptance': tf.reduce_min(acc), 'invalid': tf.reduce_sum(bad)})
            if int(tf.reduce_sum(bad)) == 0 and float(tf.reduce_min(acc)) >= .5:
                umbrella_dt = step
        if umbrella_dt is None:
            raise ValueError('no eligible umbrella pilot')
        extra = {'centers': centers, 'scales': scales, 'dt': umbrella_dt, 'pilot': choices}
    elif method == 'splitting':
        init = GlobalProgram(target, proposal, 64 if repair else 32)
        mutation = ConstrainedProgram(target, 128 if repair else 32)
    elif method == 'local_maps':
        flows = [load_flow(Path(local)/f'component-{side}/forward-frozen.json', target) for side in (-1, 1)]
        proposal = Proposal(target, flows=flows)
        burn = GlobalProgram(target, proposal, 2048 if repair else 512, adapt=True)
        keep = GlobalProgram(target, proposal, n//4)
    else:
        raise ValueError('unknown method')
    for rep, seed in enumerate(cfg['replication_seeds']):
        s = key(seed+100000*repair, 14001)
        diag = {}
        if method == 'importance':
            rows, lw, _, se = selected.run(s)
            diag['within_run_delta_se'] = se
        elif method == 'umbrella':
            last, _, acc0, bad0 = burn.run(xstart, tf.constant(umbrella_dt, F64), s)
            last, window_rows, acc, bad = keep.run(last, tf.constant(umbrella_dt, F64), seed_fold(s, 1))
            rows, lw, z, matrix, residual = emus.run(window_rows)
            if int(tf.reduce_sum(bad+bad0)) or not bool(tf.reduce_all(z > 0.)) or float(residual) > 1e-8:
                raise ValueError('EMUS numerical/overlap invariant failure')
            diag = {'normalizers': z, 'overlap': matrix, 'stationary_residual': residual,
                'minimum_acceptance': tf.reduce_min(acc), 'burn_minimum_acceptance': tf.reduce_min(acc0)}
        elif method == 'splitting':
            x = proposal.sample(n, s)
            x, _, ga, la, bad = init.run(x, tf.constant(dt, F64), seed_fold(s, 1))
            if int(bad):
                raise ValueError('split initialization invalid')
            mass = tf.constant(1., F64)
            shells, weights, stages = [], [], []
            ancestry = tf.range(n)
            for j, bound in enumerate((4., 3., 2., 1., .5, .2, .1)):
                accepted = tf.abs(x[:, 0]) < bound
                count = int(tf.reduce_sum(tf.cast(accepted, tf.int32)))
                outside = tf.boolean_mask(x, ~accepted)
                shells.append(outside)
                weights.append(tf.fill([tf.shape(outside)[0]], tf.math.log(mass)-tf.math.log(tf.cast(n, F64))))
                if count == 0:
                    raise ValueError('split extinction; candidate repair required')
                fraction = tf.cast(count, F64)/n
                mass *= fraction
                indices = tf.cast(tf.where(accepted)[:, 0], tf.int32)
                choices = tf.random.stateless_uniform([n], seed_fold(s, 100+j), 0, count, dtype=tf.int32)
                indices = tf.gather(indices, choices)
                x = tf.gather(x, indices); ancestry = tf.gather(ancestry, indices)
                x, acc = mutation.run(x, tf.constant(bound, F64), seed_fold(s, 200+j))
                stages.append({'bound': bound, 'survival': fraction, 'mass': mass,
                    'mutation_acceptance': acc, 'distinct_ancestors': tf.size(tf.unique(ancestry).y)})
            shells.append(x)
            weights.append(tf.fill([n], tf.math.log(mass)-tf.math.log(tf.cast(n, F64))))
            rows, lw = tf.concat(shells, 0), tf.concat(weights, 0)
            diag = {'stages': stages, 'shell_mass_sum': tf.reduce_sum(tf.exp(lw)),
                    'initial_global_acceptance': ga, 'initial_local_acceptance': la,
                    'ideal_independent_conditional_assumption': False}
            if abs(float(tf.reduce_sum(tf.exp(lw)))-1.) > 1e-10:
                raise ValueError('split shell conservation failed')
        else:
            proposal.logits.assign(tf.zeros([2], F64))
            x = tf.repeat(target.known_representatives(), 2, 0)
            last, _, ga0, la0, bad0 = burn.run(x, tf.constant(dt, F64), s)
            frozen = proposal.logits.read_value()
            _, trace, ga, la, bad = keep.run(last, tf.constant(dt, F64), seed_fold(s, 1))
            if int(bad+bad0) or not bool(tf.reduce_all(frozen == proposal.logits)):
                raise ValueError('invalid or adapting retained mixture kernel')
            rows = tf.reshape(trace, [-1, 2]); lw = tf.zeros([tf.shape(rows)[0]], F64)
            diag = {'global_acceptance': ga, 'local_acceptance': la, 'weights': proposal.weights(),
                    'weights_frozen_for_retained_sampling': True, 'warmup_steps': 2048 if repair else 512}
        report = bank_report(target, rows, lw)
        if not report['finite']:
            raise ValueError('nonfinite weighted bank')
        estimates.append(report['estimate'].numpy().tolist())
        diagnostics.append({'seed': int(s[0]), **diag, **report})
        write_json(out/f'replication-{rep}.json', diagnostics[-1])
        if rep == 0:
            save_bank(out, rows, lw)
    screen = probability_screen(estimates)
    result = {'target': target_name, 'method': method, 'repair': repair, 'screen': screen,
              'settings': extra, 'diagnostics': diagnostics, 'training_bank_replication': 0,
              'all_replications_saved': True, 'jit_compile': True, 'cpu_sample_generation': True}
    write_json(out/'method.json', result)
    return result


def global_fit(target_name, teacher, prepared, output, seed, cfg):
    target = WarmStartTarget(target_name)
    rows = read_tensor(Path(teacher)/'rows.tensor'); lw = read_tensor(Path(teacher)/'log-weights.tensor')
    reference = read_tensor(Path(prepared)/'validation.tensor')
    return fit_bank(target, rows, lw, reference, output, seed, cfg)
