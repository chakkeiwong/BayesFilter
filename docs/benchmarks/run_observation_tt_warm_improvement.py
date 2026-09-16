#!/usr/bin/env python3
"""A09 bounded d<=4 regression/filter diagnostic; not a production TT route.

Dense Gaussian projection, reference setup and statistics are diagnostic
exceptions. All numerical operations use TensorFlow; core fits/sampling use XLA.
"""
import argparse
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from docs.benchmarks import run_observation_tt_independent_filtering as prior

PLAN = ROOT/'docs/plans/observation-aware-tt-master-amendment-09-standalone-warm-tt-20260916.md'
write, plain, sha = prior.write, prior.plain, prior.sha
HEURISTICS = ('transition', 'stationary_prior', 'sgqf_gaussian', 'sgqf_joint')


def config(degree, rows, l1, center='zero'):
    return dict(degree=degree, rows=rows, l1=l1, center=center, rank=3,
                sweeps=4, proximal_steps=128, defensive_mass=1e-5)


def key(cfg):
    return f"p{cfg['degree']}-n{cfg['rows']}-l{cfg['l1']:g}-{cfg['center']}"


def panel(model, observation, current, condition, retained, count, seed, scale=None, *, physical_guide=None):
    d = model.dimension
    guide_args = {} if physical_guide is None else dict(guide_current=physical_guide[0], guide_condition=physical_guide[1])
    coordinates, logw, info = lib.joint_sgqf_row_sampler(model, current, condition, count, seed, **guide_args)
    rows = tf.stack([coordinates[:, :d], coordinates[:, d:]], axis=-1)
    x, z = current.forward(rows[:, :, 0]), condition.forward(rows[:, :, 1])
    logtarget = (model.observation_log_prob(x, observation)+model.transition_log_prob(x, z)
                 +retained.physical_log_density(z)+current.logdet+condition.logdet-logrho(coordinates))
    if scale is None:
        scale = tf.reduce_logsumexp(logtarget+logw)-tf.math.log(tf.cast(count, D))
    target, weights = tf.exp(.5*(logtarget-scale)), tf.exp(logw)
    lib.finite(target, 'warm target')
    target_weights = weights*target**2
    target_weights /= tf.reduce_sum(target_weights)
    info.update(target_weight_ess=1/tf.reduce_sum(target_weights**2),
                maximum_target_weight=tf.reduce_max(target_weights), rows=count, seed=seed)
    return dict(rows=rows, target=target, weights=weights, scale=scale, info=info)


def discrepancy(cores, data):
    h = pair.evaluate_pair_cores(cores, data['rows'])
    w, b = data['weights'], data['target']
    rms = tf.sqrt(tf.reduce_sum(w*(h-b)**2)/tf.reduce_sum(w*b*b))
    Z = pair.pair_total_mass(cores)
    tf.debugging.assert_positive(Z)
    amplitude = tf.sqrt((1-1e-5)*h*h/Z+1e-5)
    h2 = 1-tf.reduce_sum(w*amplitude*b)/tf.sqrt(tf.reduce_sum(w*amplitude**2)*tf.reduce_sum(w*b*b))
    lib.finite(h2, 'warm heldout discrepancy')
    tf.debugging.assert_greater_equal(h2, tf.constant(-1e-12, D))
    return dict(amplitude_rms=rms, defended_h2=tf.maximum(tf.constant(0., D), h2))


def fit_step(model, observation, guide, retained, t, seed, cfg, budget, *, audit=False, charts=None):
    budget(); started = time.monotonic()
    physical_guide = (guide[t][1], guide[t-1][1])
    current, condition = physical_guide if charts is None else (charts[t], charts[t-1])
    m, C = projection.paired_gaussian(model, current, condition,
        guide_current=physical_guide[0], guide_condition=physical_guide[1])
    coefficients, _ = projection.coefficient_kernel(2*model.dimension, cfg['degree'])(m, C)
    with tf.device('/CPU:0'):
        initial, compression, _ = projection.pair_svd(coefficients, cfg['rank'])
    conversion = dict(truncation_squared_l2=1-tf.reduce_sum(coefficients**2),
                      compression_squared_l2=compression)
    train = panel(model, observation, current, condition, retained, cfg['rows'], seed+t, physical_guide=physical_guide)
    validation = panel(model, observation, current, condition, retained, 4096, seed+t+100000, train['scale'], physical_guide=physical_guide)
    values = pair.evaluate_pair_cores(initial, train['rows'])
    scalar = tf.reduce_sum(train['weights']*values*train['target'])/tf.reduce_sum(train['weights']*values**2)
    tf.debugging.assert_positive(scalar, 'nonpositive initial amplitude scale')
    initial = (initial[0]*scalar, *initial[1:])
    features = pair._pair_features(train['rows'], cfg['degree'])
    cores, diag = pair.fit_pair_features(features, train['target'], train['weights'],
        degree=cfg['degree'], rank=cfg['rank'], sweeps=cfg['sweeps'],
        proximal_steps=cfg['proximal_steps'], penalty=cfg['l1'], initial=initial,
        jit_compile=True, regularization_center=cfg['center'])
    Z = pair.pair_total_mass(cores)
    tau = Z*cfg['defensive_mass']/(1-cfg['defensive_mass'])
    retained_next = lib.PairRetainedProposal(cores, current, Z, tau, t)
    step = lib.PairTTStep(cores, current, condition, tau, diag, t, retained_next)
    record = dict(time=t, config=cfg, initial_scale=scalar, target_scale=train['scale'],
        conversion=conversion, train=discrepancy(cores, train),
        validation=dict(initial=discrepancy(initial, validation), fitted=discrepancy(cores, validation)),
        row_diagnostics=dict(train=train['info'], validation=validation['info']), fit_diagnostics=diag,
        cores=cores, tau=tau, mass=Z, current_mean=current.mean, current_factor=current.factor,
        condition_mean=condition.mean, condition_factor=condition.factor)
    if audit:
        heldout = panel(model, observation, current, condition, retained, 8192, seed+t+200000, train['scale'], physical_guide=physical_guide)
        record['audit'] = dict(initial=discrepancy(initial, heldout), fitted=discrepancy(cores, heldout))
        record['row_diagnostics']['audit'] = heldout['info']
    record['wall_seconds'] = time.monotonic()-started
    return step, record


def warm_path(model, observations, guide, seed, cfg, dest, budget):
    first = joint.make_sgqf_joint_step(model, guide[0][1], None, 0)
    path = [first]
    write(dest/'initial.json', dict(type='fixed_initial_sgqf', mean=first.current_chart.mean,
                                  factor=first.current_chart.factor, config=cfg))
    for t, observation in enumerate(tf.unstack(observations)[1:], 1):
        step, record = fit_step(model, observation, guide, path[-1].retained_proposal,
                                t, seed, cfg, budget)
        if not isinstance(step, lib.PairTTStep):
            raise TypeError('Every post-initial warm step must be TT')
        path.append(step)
        write(dest/f'fit-t{t}.json', record)
    return path


def add_arm(context, name, cfg, budget):
    dest = context['dest']/name
    dest.mkdir()
    started = time.monotonic()
    entry = context['entry']
    try:
        model, observations, guide = context['model'], context['observations'], context['guide']
        if guide is None and name not in ('transition', 'stationary_prior'):
            raise ValueError('Shared SGQF guide failed; no fallback')
        path = None
        consumer = name
        if cfg is not None:
            path = warm_path(model, observations, guide, context['seed']+2000000, cfg, dest, budget)
            consumer = 'tt_sgqf_initialized'
        elif name == 'sgqf_joint':
            path = [joint.make_sgqf_joint_step(model, g[1], None if t == 0 else guide[t-1][1], t)
                    for t, g in enumerate(guide)]
        elif name == 'tt_pair_block':
            path = lib.build_pair_tt_path(model, observations, guide, seed=context['seed']+2000000)
        build = time.monotonic()-started
        runs = []
        for r in range(4):
            budget()
            value, _ = base.particle_filter(model, observations, guide, path, consumer, 512,
                                            context['seed']+3000000+1000*r, True)
            runs.append(value); write(dest/f'particles-r{r}.json', value)
        metrics = prior.sequence_metrics(model, observations, {name: runs}, context['refmeans'], context['refz'], 512)[name]
        refse = float(entry['reference'].get('log_evidence_mcse', 0.))
        allowance = .15+3.182446*(metrics['log_evidence_mcse']**2+refse**2)**.5
        metrics['log_evidence_screen'] = dict(allowance=allowance,
            passed=abs(metrics['log_evidence_bias']) <= allowance)
        metrics['cdf_bracket_failures'] = sum(int(not bool(s['cdf_bracket_valid']))
            for run in runs for s in run['steps'] if 'cdf_bracket_valid' in s)
        metrics['maximum_cdf_residual'] = max([float(s['cdf_residual'])
            for run in runs for s in run['steps'] if 'cdf_residual' in s] or [0.])
        metrics['consumer_invalid_steps'] = sum(int(not bool(s['finite']))
            for run in runs for s in run['steps'] if 'finite' in s)
        entry['metrics'][name] = metrics
        entry['times'][name] = dict(build_seconds=build, particle_seconds=time.monotonic()-started-build,
                                   total_seconds=time.monotonic()-started, particle_replicates=4)
        if path is not None:
            context['paths'][name] = path
    except (ValueError, tf.errors.OpError) as exc:
        entry['failures'][name] = repr(exc)
        write(dest/'failure.json', dict(error=repr(exc), traceback=traceback.format_exc(), fallback=False))
    write(context['dest']/'summary.json', entry)


def choose(contexts, configs):
    valid = [c for c in contexts if c['guide'] is not None]
    if len(valid) < 2 or any(not c['entry']['reference_pass'] for c in contexts):
        raise ValueError('Calibration requires >=2 guide-valid sequences and all references valid')
    table = []
    for cfg in configs:
        name = key(cfg)
        available = all(name in c['entry']['metrics'] for c in valid)
        mse = sum(c['entry']['metrics'][name]['regimes']['all']['mse'] for c in valid)/len(valid) if available else None
        table.append(dict(config=cfg, mean_mse=mse, eligible=available, sequences=len(valid)))
    eligible = [row for row in table if row['eligible']]
    if not eligible:
        raise ValueError('No eligible calibration candidate')
    return min(eligible, key=lambda row: row['mean_mse'])['config'], table


def sequence_seed(partition, dimension, sequence):
    """Reserve ten million scalar seeds per independent data/algorithm unit."""
    if partition not in range(7) or dimension not in (1, 4) or sequence not in range(12):
        raise ValueError('Seed partition/dimension/sequence exceeds reserved blocks')
    return 100000000 + 10000000*(24*partition + 12*(dimension == 4) + sequence)


def infer(entries):
    """Simultaneous intervals for predeclared paired sequence contrasts only."""
    records, distributions = [], []
    for d in (1, 4):
        dimension_entries = [e for e in entries if e['dimension'] == d]
        indices = tf.random.stateless_uniform([9999, len(dimension_entries)], [927777, d],
                                              minval=0, maxval=len(dimension_entries), dtype=tf.int32)
        for candidate in ('capacity', 'preservation'):
            valid = [e for e in dimension_entries if e['reference_pass'] and
                     all(k in e['metrics'] for k in ('baseline', candidate))]
            differences = [e['metrics'][candidate]['regimes']['all']['mse']-
                           e['metrics']['baseline']['regimes']['all']['mse'] for e in valid]
            center = sum(differences)/len(differences) if differences else None
            complete = len(valid) == len(dimension_entries) == 12
            se = (sum((x-center)**2 for x in differences)/(len(differences)*(len(differences)-1)))**.5 if len(differences)>1 else 0.
            eligible = complete and se > 0
            if eligible:
                x = tf.constant(differences, D)
                boot = tf.reduce_mean(tf.gather(x, indices), axis=1)
                distributions.append(tf.abs(boot-center)/se)
            records.append(dict(dimension=d, candidate=candidate, sequences=len(valid),
                complete=complete, mean_delta=center, se=se, interval_eligible=eligible,
                exactly_identical=bool(differences) and all(x == 0 for x in differences)))
    critical = float(tf.sort(tf.reduce_max(tf.stack(distributions), axis=0))[9498]) if distributions else None
    for r in records:
        width = critical*r['se'] if r['interval_eligible'] else None
        r.update(lower=r['mean_delta']-width if width is not None else None,
                 upper=r['mean_delta']+width if width is not None else None)
        r['mse_improvement_supported'] = r['upper'] is not None and r['upper'] < 0
    heuristics = []
    for d, candidate, heuristic, regime in itertools.product((1,4), ('baseline','capacity','preservation'), HEURISTICS, ('all','near_zero','ordinary','large')):
        available = [e for e in entries if e['dimension']==d and e['reference_pass'] and
                     all(k in e['metrics'] for k in (candidate,heuristic))]
        deltas = [e['metrics'][candidate]['regimes'][regime]['mse']-e['metrics'][heuristic]['regimes'][regime]['mse']
                  for e in available if e['metrics'][candidate]['regimes'][regime]['count']]
        mean = sum(deltas)/len(deltas) if deltas else None
        heuristics.append(dict(dimension=d,candidate=candidate,heuristic=heuristic,regime=regime,
                               sequences=len(deltas),mean_delta=mean,observed_loss=mean is not None and mean>0))
    return dict(primary_contrasts=records, simultaneous_critical=critical,
                heuristic_contrasts=heuristics,
                heuristic_dominance_verdict='PROMOTION_VETO' if any(x['observed_loss'] for x in heuristics) else 'NO_OBSERVED_LOSS',
                default_ready=False)


def main():
    global tf, D, lib, pair, joint, projection, base, logrho
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('smoke','calibration','confirmation'), required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--calibration-root')
    parser.add_argument('--wall-budget-seconds', type=float, default=5400)
    parser.add_argument('--seed-partition', type=int)
    args = parser.parse_args(); started = time.monotonic()
    out = Path(args.output_root).resolve(); out.mkdir(parents=True, exist_ok=False)
    # Preserve full compiler/driver output even when launched without a shell wrapper.
    command_log = open(out/'command.log', 'a', buffering=1)
    os.dup2(command_log.fileno(), 1)
    os.dup2(command_log.fileno(), 2)
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
    os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
    os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    import tensorflow as tf
    devices = tf.config.list_physical_devices('GPU')
    if not devices: raise RuntimeError('Trusted GPU required')
    growth = []
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
        if not tf.config.experimental.get_memory_growth(device): raise RuntimeError('Memory growth required')
        growth.append(dict(name=device.name,growth=True,details=tf.config.experimental.get_device_details(device)))
    from bayesfilter.highdim import observation_guided_tt_tf as lib
    from bayesfilter.highdim import pair_block_tt_tf as pair
    from bayesfilter.highdim import sgqf_joint_consumer_tf as joint
    from bayesfilter.highdim.c2_gaussian_hermite_proposal_tf import _log_standard_normal as logrho
    from docs.benchmarks import observation_tt_sgqf_projection_diagnostic as projection
    from docs.benchmarks import run_observation_aware_tt_complete as base
    D = tf.float64
    prior.tf=tf; prior.D=D; prior.base=base
    base.tf=tf; base.D=D; base.lib=lib
    dependencies = [Path(__file__),PLAN,Path(prior.__file__),Path(base.__file__),Path(lib.__file__),
                    Path(pair.__file__),Path(joint.__file__),Path(projection.__file__),prior.FIXTURE]
    frozen = None
    if args.stage == 'confirmation':
        if not args.calibration_root: raise ValueError('Frozen calibration required')
        frozen_path = Path(args.calibration_root)/'selected-controls.json'
        calibration_result = json.loads((Path(args.calibration_root)/'result.json').read_text())
        if calibration_result['status'] != 'COMPLETE':
            raise ValueError('Completed calibration required')
        if calibration_result['selected_controls_sha256'] != sha(frozen_path):
            raise ValueError('Frozen controls do not match completed calibration')
        frozen = json.loads(frozen_path.read_text())
        if frozen['status'] != 'FROZEN': raise ValueError('Unfrozen controls')
        dependencies.append(frozen_path)
        write(out/'selected-controls.json', frozen)
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(),status='RUNNING',stage=args.stage,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True),
        command=sys.argv, environment=sys.prefix,python=sys.version,tensorflow=tf.__version__,
        cpu_only=False,gpu_intentionally_hidden=False,jit_compile=True,dtype='float64',
        tf32_enabled=tf.config.experimental.tensor_float_32_execution_enabled(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        gpu_memory_policy=dict(schema='a09_growth_v1',mode='memory_growth',devices=growth),
        trust_basis='escalated_gpu_access',classification='bounded_regression_filter_diagnostic',
        source_classification='extension_or_invention',plan=str(PLAN.relative_to(ROOT)),
        result=str(out/'result.json'),wall_budget_seconds=args.wall_budget_seconds,
        source_hashes={str(p.resolve()):sha(p) for p in dependencies},
        seeds=dict(data='100000000+10000000*(24*partition+12*(d==4)+sequence)',fit='data+2000000',
                   particle='data+3000000+1000*replicate',reference='data+4000000+1000000*level+1000*replicate',
                   partition=args.seed_partition if args.seed_partition is not None else (2 if args.stage=='confirmation' else 0),
                   bootstrap=927777,revision=2),
        data_version='A09 new stateless SV sequences, old fixture parameters only',
        reference_exceptions='Dense Gaussian projection/CPU TT-SVD, guide/row/feature/target setup, independent references and post-run statistics are diagnostics; only the numerical kernels claim XLA, not end-to-end orchestration')
    write(out/'run_manifest.json',manifest)
    result=dict(status='RUNNING',stage=args.stage,sequences=[],default_ready=False)
    contexts=[]
    def budget():
        if time.monotonic()-started > args.wall_budget_seconds: raise TimeoutError('A09 attempt budget exhausted')
    try:
        if args.stage == 'smoke':
            parity=[]
            points=tf.random.stateless_normal([64,2,2],[929999,7],dtype=D)
            features=pair._pair_features(points,4)
            target=tf.exp(.05*points[:,0,0]-.03*points[:,1,1])
            initial=pair.initial_pair_cores(2,4,3)
            for center in ('zero','initial'):
                kwargs=dict(degree=4,rank=3,sweeps=1,proximal_steps=8,
                            penalty=1e-3,initial=initial,regularization_center=center)
                with tf.device('/CPU:0'):
                    reference,_=pair.fit_pair_features(features,target,tf.ones([64],D),jit_compile=False,**kwargs)
                actual,_=pair.fit_pair_features(features,target,tf.ones([64],D),jit_compile=True,**kwargs)
                error=max(float(tf.reduce_max(tf.abs(a-b))) for a,b in zip(actual,reference))
                for a,b in zip(actual,reference): tf.debugging.assert_near(a,b,atol=2e-10,rtol=2e-10)
                parity.append(dict(center=center,maximum_absolute_error=error,passed=True))
            write(out/'gpu-cpu-fit-parity.json',parity)
        fixture=json.loads(prior.FIXTURE.read_text())
        grid=[config(p,n,l1) for p,n,l1 in itertools.product((3,4),(1024,4096),(0.,1e-5,1e-3))]
        write(out/'grid.json',grid)
        count=1 if args.stage=='smoke' else (3 if args.stage=='calibration' else 12)
        for d in (1,4):
            data=fixture['dimensions'][str(d)]
            model=lib.SVModel(tf.constant(data['A'],D),tf.constant(data['P0'],D),fixture['beta'],fixture['sigma'])
            generator=prior.data_generator(model,3 if args.stage=='smoke' else 20)
            for sequence in range(count):
                budget(); dest=out/f'd{d}-s{sequence:02d}';dest.mkdir()
                seed=929999+d if args.stage=='smoke' else sequence_seed(manifest['seeds']['partition'],d,sequence)
                states,observations=generator(tf.constant(seed,tf.int32))
                write(dest/'data.json',dict(seed=seed,states=states,observations=observations,
                    A=model.transition,P0=model.covariance0,beta=model.beta,sigma=model.sigma))
                failures={}; guide=None; tic=time.monotonic()
                try:
                    guide,records=lib.build_guide_path(model,observations)
                    write(dest/'guide.json',records)
                except (ValueError,tf.errors.OpError) as exc:
                    failures['guide']=repr(exc)
                    write(dest/'guide-failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
                guide_seconds=time.monotonic()-tic
                refmeans,refz,refpass,refinfo=prior.references(model,observations,dest,seed+4000000,budget,replicate_seed_stride=1000)
                entry=dict(dimension=d,sequence=sequence,data_seed=seed,reference_pass=refpass,reference=refinfo,
                           metrics={},failures=failures,times={},guide_seconds=guide_seconds)
                ctx=dict(model=model,observations=observations,guide=guide,seed=seed,dest=dest,entry=entry,
                         refmeans=refmeans,refz=refz,paths={})
                contexts.append(ctx);result['sequences'].append(entry)
                if args.stage=='calibration':
                    arms=[(key(cfg),cfg) for cfg in grid]
                elif args.stage=='smoke':
                    arms=[('baseline',config(3,1024,1e-5)),('capacity',config(4,4096,1e-5)),
                          ('preservation',config(4,4096,1e-3,'initial'))]
                else:
                    arms=list(frozen['dimensions'][str(d)]['selected'].items())
                seen={}
                for name,cfg in arms:
                    identity=key(cfg)
                    if identity in seen and seen[identity] in entry['metrics']:
                        original=seen[identity]
                        entry['metrics'][name]=entry['metrics'][original]
                        entry['times'][name]=dict(**entry['times'][original],alias_of=original)
                        continue
                    add_arm(ctx,name,cfg,budget);seen[identity]=name
                if args.stage!='calibration':
                    for name in (*HEURISTICS,'tt_pair_block'):
                        add_arm(ctx,name,None,budget)
                write(dest/'summary.json',entry);write(out/'result.json',result)
                print(f"{args.stage} d={d} sequence={sequence}: {len(entry['metrics'])} methods, failures={list(entry['failures'])}",flush=True)
        if args.stage=='calibration':
            selected=dict(status='FROZEN',dimensions={},selection_uses_audit=False,
                          criterion='calibration corrected-filter normalized mean MSE')
            for d in (1,4):
                subset=[c for c in contexts if c['model'].dimension==d]
                baseline,btable=choose(subset,[c for c in grid if c['degree']==3 and c['rows']==1024])
                capacity,ctable=choose(subset,grid)
                anchors=[config(capacity['degree'],capacity['rows'],l1,'initial') for l1 in (1e-5,1e-3)]
                for ctx in subset:
                    for cfg in anchors: add_arm(ctx,key(cfg),cfg,budget)
                shared=config(capacity['degree'],capacity['rows'],0.)
                preservation,ptable=choose(subset,[shared,*anchors])
                selected['dimensions'][str(d)]=dict(selected=dict(baseline=baseline,capacity=capacity,preservation=preservation),
                    baseline_grid=btable,capacity_grid=ctable,preservation_grid=ptable)
            write(out/'selected-controls.json',selected)
            # Freeze before any untouched same-target audit rows are generated.
            selected_hash=sha(out/'selected-controls.json')
            for ctx in contexts:
                if ctx['guide'] is None: continue
                chosen=selected['dimensions'][str(ctx['model'].dimension)]['selected']
                path=ctx['paths'][key(chosen['baseline'])]
                same=ctx['dest']/'same-target-audit';same.mkdir()
                for t in (1,10,19):
                    for name,cfg in chosen.items():
                        _,record=fit_step(ctx['model'],ctx['observations'][t],ctx['guide'],path[t-1].retained_proposal,
                            t,ctx['seed']+2000000,cfg,budget,audit=True)
                        record.update(target_source=key(chosen['baseline']),controls_sha256=selected_hash)
                        write(same/f'{name}-t{t}.json',record)
            result['selected_controls_sha256']=selected_hash
        elif args.stage=='confirmation':
            result['inference']=infer(result['sequences'])
        result['status']='COMPLETE';manifest['status']='COMPLETE'
    except BaseException as exc:
        result.update(status='FAILED',error=repr(exc));manifest.update(status='FAILED',error=repr(exc))
        traceback.print_exc();raise
    finally:
        manifest.update(wall_seconds=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat(),
                        gpu_allocator=tf.config.experimental.get_memory_info('GPU:0'))
        manifest['source_unchanged']=all(sha(Path(p))==digest for p,digest in manifest['source_hashes'].items())
        write(out/'result.json',result);write(out/'run_manifest.json',manifest)


if __name__=='__main__':
    main()
