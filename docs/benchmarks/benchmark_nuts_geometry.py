"""Bounded standalone geometry and execution-overhead diagnostic, fully XLA kernels."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time

import benchmark_nuts_xla as base
import compare_nuts_fixed_hmc as compare
from nuts_benchmark_targets import make_target, NAMES

PLAN = 'docs/plans/bayesfilter-nuts-geometry-overhead-2026-10-27.md'


def config(output, dimension=2, chains=4, method='nuts', length=3, accept=.8, warmup=256, draws=256):
    return base.parser().parse_args(['--output', str(output), '--device', 'cpu',
        '--dimension', str(dimension), '--chains', str(chains), '--method', method,
        '--leapfrog-steps', str(length), '--target-accept', str(accept),
        '--warmup', str(warmup), '--draws', str(draws), '--max-tree-depth', '8'])


def clean_json(path, value):
    missing = []
    value = compare.serialize_missing(value, missing)
    if isinstance(value, dict):
        value['unavailable_numeric_fields'] = missing
    base.write_json(path, value)


def compile_record(tf, function, inputs, folder, name):
    result, seconds = base.timed(function, *inputs)
    compare.health(tf, result)
    graph = base.verify_graph(function)
    hlo = function.experimental_get_compiler_ir(*inputs)(stage='optimized_hlo')
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f'{name}.hlo.txt').write_text(hlo)
    assert function.experimental_get_tracing_count() == 1
    assert all('/device:CPU:0' in x.device for x in result)
    return dict(graph, first_call_seconds=seconds, traces=1,
                hlo_sha256=hashlib.sha256(hlo.encode()).hexdigest())


def assess(tf, target, result, seconds):
    from bayesfilter.inference.hmc_ess import stan_cross_chain_ess
    from bayesfilter.inference.hmc_posterior_diagnostics import (
        rank_normalized_split_rhat, rank_normalized_bulk_tail_ess)
    compare.health(tf, result)
    samples = result[2]
    z = tf.reshape(target.to_reference(tf.reshape(samples, [-1, target.dimension])), samples.shape)
    observables = tf.concat((z, tf.square(z)), axis=-1)
    ess = stan_cross_chain_ess(observables)
    rhat = rank_normalized_split_rhat(tf.transpose(
        tf.concat((samples, z, tf.square(z)), axis=-1), [1, 0, 2]))
    tails = rank_normalized_bulk_tail_ess(tf.transpose(z, [1, 0, 2]))
    stats = result[3]
    movements = tf.reduce_sum(tf.cast(tf.reduce_any(samples[1:] != samples[:-1], axis=-1), tf.int32), axis=0)
    raw = base.json_tensor(tf, dict(
        seconds=seconds, mean_ess=ess, minimum_reference_ess=tf.reduce_min(ess),
        minimum_reference_ess_per_second=tf.reduce_min(ess) / seconds,
        max_rhat=tf.reduce_max(rhat['maximum']), reference_ess=tails,
        reference_mean=tf.reduce_mean(z, axis=[0, 1]),
        reference_second_moment=tf.reduce_mean(tf.square(z), axis=[0, 1]),
        movements_per_chain=movements, acceptance=tf.reduce_mean(stats[..., 0]),
        reported_leapfrogs=tf.reduce_mean(stats[..., 2]),
        max_chain_leapfrogs=tf.reduce_mean(tf.reduce_max(stats[..., 2], axis=1)),
        depth_cap_fraction=tf.reduce_mean(stats[..., 3]),
        native_flag_fraction=tf.reduce_mean(stats[..., 4]),
        finite_cache=tf.reduce_all(stats[..., 5] == 1.),
        valid_acceptance=tf.reduce_all(stats[..., 6] == 1.)))
    missing = []
    report = compare.serialize_missing(raw, missing)
    report['unavailable_diagnostics'] = missing
    report['rhat_movement_screen_pass'] = (report['max_rhat'] is not None and
        report['max_rhat'] <= 1.01 and all(x > 0 for x in report['movements_per_chain']) and not missing)
    report['posterior_qualification'] = False
    return report


def reference_initial(tf, target, chains, seed):
    return target.from_reference(tf.random.stateless_normal(
        [chains, target.dimension], [20261027, seed], dtype=tf.float64))


def overhead(tf, tfp, args, manifest):
    folder = args.output / 'overhead'
    folder.mkdir()
    target = make_target(tf, 'banana2')
    reports = []
    # Different batch widths change RNG streams: these are workload probes.
    for method, count in [('hmc',4),('nuts',1),('nuts',4),('nuts',16)]:
        conf = config(args.output, chains=count, method=method, draws=1024)
        conf.max_tree_depth = 6
        epsilon = tf.constant(.6089081806811285 if method == 'hmc' else .7586082110008886, tf.float64)
        q = reference_initial(tf, target, count, 71)
        seed = tf.constant([20261027, 710], tf.int32)
        inputs = (q, epsilon, seed)
        function = base.make_chain(tf, tfp, conf, adapting=False, log_prob_fn=target.log_prob)
        name = f'{method}-c{count}'
        manifest['compiled'][name] = compile_record(tf, function, inputs, folder, name)
        times = []
        for _ in range(7):
            result, seconds = base.timed(function, *inputs)
            times.append(seconds)
        # A resource counter is present only in this untimed replay.
        calls = tf.Variable(0, dtype=tf.int64, trainable=False)
        def counted(q):
            with tf.control_dependencies([calls.assign_add(1)]):
                return target.log_prob(q)
        instrumented = base.make_chain(tf, tfp, conf, adapting=False, log_prob_fn=counted)
        replay, _ = base.timed(instrumented, *inputs)
        exact = all(bool(tf.reduce_all(x == y).numpy().item()) for x,y in zip(result,replay))
        if not exact:
            raise RuntimeError('Counted replay changes numerical output')
        target_calls = int(calls.numpy().item())
        reports.append(dict(method=method, chains=count, timings=times,
            median_seconds=statistics.median(times), exact_counted_replay=exact,
            target_calls_including_bootstrap=target_calls,
            dense_gradient_calls_per_transition=(target_calls - 1) / conf.draws,
            reported_leapfrogs=float(tf.reduce_mean(result[3][...,2]).numpy().item()),
            max_chain_leapfrogs=float(tf.reduce_mean(tf.reduce_max(result[3][...,2],axis=1)).numpy().item())))
        compare.save_tensors(tf, folder, name, result)
        clean_json(folder/'counts.json',reports)
        print(json.dumps(reports[-1]),flush=True)
    # Serial vs threaded scalar chains: identical q, epsilon and seed per chain.
    conf = config(args.output, chains=1, draws=1024)
    conf.max_tree_depth = 6
    epsilon=tf.constant(.7586082110008886,tf.float64)
    initial=reference_initial(tf,target,4,72)
    function=base.make_chain(tf,tfp,conf,adapting=False,log_prob_fn=target.log_prob)
    inputs=[(initial[i:i+1],epsilon,tf.constant([20261027,720+i],tf.int32)) for i in range(4)]
    manifest['compiled']['scalar-scheduling']=compile_record(tf,function,inputs[0],folder,'scalar-scheduling')
    pairs=[]
    order_rng=random.Random(20261027)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for replica in range(7):
            order=['serial','threads'];order_rng.shuffle(order);results={};times={}
            for layout in order:
                start=time.perf_counter()
                if layout=='serial':
                    outputs=[base.timed(function,*item)[0] for item in inputs]
                else:
                    futures=[pool.submit(base.timed,function,*item) for item in inputs]
                    outputs=[future.result()[0] for future in futures]
                times[layout]=time.perf_counter()-start;results[layout]=outputs
            assert all(bool(tf.reduce_all(a==b).numpy().item())
                for ra,rb in zip(results['serial'],results['threads']) for a,b in zip(ra,rb))
            pairs.append(dict(replica=replica,order=order,seconds=times,exact_outputs=True))
    clean_json(folder/'scalar-scheduling.json',pairs)
    return dict(counted_work=reports,scalar_scheduling=pairs,
                qualification='Engineering diagnostic only; no posterior ranking')


def geometry(tf, tfp, args, manifest):
    targets=args.targets.split(',') if args.targets else list(NAMES)
    reports=[]
    for name in targets:
        target=make_target(tf,name);folder=args.output/name;folder.mkdir()
        target_start=time.perf_counter()
        configurations=[('hmc',length,accept) for length in args.lengths for accept in (.8,.9)]
        configurations += [('nuts',3,accept) for accept in (.8,.9)]
        cells=[]
        for method,length,accept in configurations:
            conf=config(args.output,target.dimension,method=method,length=length,accept=accept)
            q=reference_initial(tf,target,4,81+args.seed_offset)
            epsilon=tf.constant(.1,tf.float64);seed=tf.constant([20261027,810+args.seed_offset],tf.int32)
            warm=base.make_chain(tf,tfp,conf,adapting=True,log_prob_fn=target.log_prob)
            sample=base.make_chain(tf,tfp,conf,adapting=False,log_prob_fn=target.log_prob)
            label=f'{name}-{method}-L{length}-a{accept}'
            manifest['compiled'][label+'-warm']=compile_record(tf,warm,(q,epsilon,seed),folder,label+'-warm')
            warmed,ws=base.timed(warm,q,epsilon,seed)
            inputs=(warmed[0],warmed[1],tf.constant([20261027,811+args.seed_offset],tf.int32))
            manifest['compiled'][label+'-sample']=compile_record(tf,sample,inputs,folder,label+'-sample')
            result,seconds=base.timed(sample,*inputs)
            record=assess(tf,target,result,seconds)
            record.update(method=method,length=length,acceptance_target=accept,
                          epsilon=float(warmed[1].numpy().item()),warmup_seconds=ws)
            cells.append(record)
            compare.save_tensors(tf,folder,label+'-tune',result)
            clean_json(folder/'tuning.json',cells)
        selected={}
        for method in ('hmc','nuts'):
            eligible=[r for r in cells if r['method']==method and
                      r['minimum_reference_ess_per_second'] is not None and
                      r['minimum_reference_ess_per_second']>0]
            if eligible:
                selected[method]=max(eligible,key=lambda r:r['minimum_reference_ess_per_second'])
        clean_json(folder/'frozen-selection.json',selected)
        functions={}
        for method,choice in selected.items():
            conf=config(args.output,target.dimension,method=method,length=choice['length'],draws=1024)
            warmconf=copy.copy(conf);warmconf.draws=512
            q=reference_initial(tf,target,4,90+args.seed_offset)
            ep=tf.constant(choice['epsilon'],tf.float64);seed=tf.constant([20261027,900+args.seed_offset],tf.int32)
            warm=base.make_chain(tf,tfp,warmconf,adapting=False,log_prob_fn=target.log_prob)
            sample=base.make_chain(tf,tfp,conf,adapting=False,log_prob_fn=target.log_prob)
            for suffix,fun in [('warm',warm),('sample',sample)]:
                manifest['compiled'][name+'-'+method+'-heldout-'+suffix]=compile_record(tf,fun,(q,ep,seed),folder,method+'-heldout-'+suffix)
            functions[method]=(warm,sample,ep)
        heldout=[]
        order_rng=random.Random(20261028)
        for replica in range(3):
            q=reference_initial(tf,target,4,2000+replica+args.seed_offset)
            methods=list(functions);order_rng.shuffle(methods)
            for method in methods:
                warm,sample,ep=functions[method]
                warmed,ws=base.timed(warm,q,ep,tf.constant([20261028,2000+replica+args.seed_offset],tf.int32))
                result,seconds=base.timed(sample,warmed[0],ep,tf.constant([20261029,2000+replica+args.seed_offset],tf.int32))
                report=assess(tf,target,result,seconds)
                report.update(method=method,replica=replica,warmup_seconds=ws,epsilon=float(ep.numpy().item()))
                heldout.append(report)
                compare.save_tensors(tf,folder,f'{method}-heldout-{replica}',result)
                clean_json(folder/'heldout.json',heldout)
        summary=dict(target=name,description=target.description,selected={m:{k:r[k] for k in ('length','acceptance_target','epsilon')} for m,r in selected.items()},methods={},seconds=time.perf_counter()-target_start)
        for method in selected:
            rows=[r for r in heldout if r['method']==method]
            rates=[r['minimum_reference_ess_per_second'] for r in rows]
            summary['methods'][method]=dict(median_seconds=statistics.median(r['seconds'] for r in rows),
                median_ess_per_second=statistics.median(rates) if all(r is not None for r in rates) else None,
                max_rhat=max(r['max_rhat'] for r in rows if r['max_rhat'] is not None),
                all_rhat_movement_screens_pass=all(r['rhat_movement_screen_pass'] for r in rows),
                mean_leapfrogs=statistics.mean(r['reported_leapfrogs'] for r in rows),
                mean_batch_max_leapfrogs=statistics.mean(r['max_chain_leapfrogs'] for r in rows),
                max_depth_cap_fraction=max(r['depth_cap_fraction'] for r in rows))
        reports.append(summary);clean_json(args.output/'summary.json',reports)
        clean_json(args.output/'manifest.json',manifest)
        print(json.dumps(summary),flush=True)
    return reports


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('overhead','geometry'),required=True)
    parser.add_argument('--targets',default='')
    parser.add_argument('--lengths',type=lambda s:tuple(int(x) for x in s.split(',')),default=(3,8,16))
    parser.add_argument('--seed-offset',type=int,default=0)
    args=parser.parse_args()
    if not args.lengths or any(length < 1 for length in args.lengths):
        parser.error('--lengths must contain positive integers')
    if args.seed_offset < 0:
        parser.error('--seed-offset must be nonnegative')
    if args.targets and any(name not in NAMES for name in args.targets.split(',')):
        parser.error('--targets must contain names from '+','.join(NAMES))
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    manifest=dict(schema='bayesfilter.nuts_geometry_diagnostic.v1',status='running',
        command=[sys.executable,*sys.argv],git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        plan=PLAN,device='CPU',gpu_intentionally_hidden=True,jit_compile=True,dtype='float64',tf32=False,
        target_data='synthetic_analytic_reference',seed=20261027,seed_offset=args.seed_offset,
        hmc_lengths=args.lengths,compiled={},sources={},
        posterior_promotion=False,statistically_supported_ranking=False)
    source=args.output/'source';source.mkdir()
    for path in (Path(__file__),Path(base.__file__),Path(compare.__file__),Path(__file__).with_name('nuts_benchmark_targets.py')):
        shutil.copy2(path,source/(path.name+'.txt'));manifest['sources'][str(path)]=base.source_hash(path)
    clean_json(args.output/'manifest.json',manifest)
    try:
        tf,tfp,memory=base.setup(config(args.output))
        manifest.update(tensorflow=tf.__version__,tfp=tfp.__version__,gpu_memory_policy=memory)
        with tf.device('/CPU:0'):
            result=overhead(tf,tfp,args,manifest) if args.phase=='overhead' else geometry(tf,tfp,args,manifest)
        clean_json(args.output/'summary.json',result);manifest['status']='completed'
    except Exception as exc:
        manifest['status']='failed';manifest['error']=repr(exc);raise
    finally:
        manifest['seconds']=time.perf_counter()-started
        clean_json(args.output/'manifest.json',manifest)


if __name__=='__main__':
    main()
