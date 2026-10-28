"""Native TFP sample_chain versus pinned CmdStan; bounded diagnostic study.

All retained transitions use stan_nuts_tfp, not the old manual sampling loop.
Stan-style warmup shares the already audited native-TFP adaptation. Run with
CPU/GPU visibility configured by setup before importing the adapter.
"""
import argparse
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback

import benchmark_nuts_xla as base
import benchmark_stan_transplant as common
import benchmark_full_stan as direct
from nuts_benchmark_targets import make_target, NAMES
from stan_nuts_warmup import Config

ROOT = Path(__file__).resolve().parents[2]
PLAN = 'docs/plans/bayesfilter-stan-tfp-native-2026-10-28.md'
RESULT = 'docs/plans/bayesfilter-stan-tfp-native-result-2026-10-28.md'


def native_functions(tf, tfp, target, args, folder):
    from stan_nuts_tfp import make_functions
    cfg = Config(target.dimension, chains=1, warmup=args.warmup, draws=args.draws,
                 metric='dense' if target.name == 'rotated_gaussian32' else 'diag',
                 depth=args.depth)
    warm, sample = make_functions(tf, tfp, target.log_prob, cfg)
    q, _ = direct.inputs(tf, target, 20260901)
    seed = tf.constant([20260901, 13], tf.int32)
    warmed, wr = common.compile_fn(tf, warm, (q[:1], seed), folder, 'warmup', 'cpu')
    if not bool(tf.reduce_all(warmed.valid)): raise RuntimeError('Invalid warmup compilation')
    sampled, sr = common.compile_fn(tf, sample, warmed[:4], folder, 'sample', 'cpu')
    if not bool(tf.reduce_all(sampled.valid)): raise RuntimeError('Invalid retained compilation')
    return (warm, sample), dict(config=asdict(cfg), warmup=wr, sample=sr,
        implementation='stan_nuts_tfp.make_functions / tfp.mcmc.sample_chain / stock NoUTurnSampler')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--targets', default=','.join((*NAMES, 'regression64')))
    p.add_argument('--seeds', default='20261028,20261029,20261030')
    p.add_argument('--warmup', type=int, default=1000)
    p.add_argument('--draws', type=int, default=2000)
    p.add_argument('--depth', type=int, default=10)
    p.add_argument('--stan', type=Path, default=ROOT/'docs/benchmarks/stan_targets')
    p.add_argument('--direct-controls', default='gaussian32,banana8')
    a = p.parse_args()
    a.output = a.output or Path(os.environ['BAYESFILTER_NATIVE_TFP_OUTPUT'])/'comparison'
    a.output = a.output.resolve();a.output.mkdir(parents=True, exist_ok=False)
    a.device, a.layout = 'cpu', 'scalar'
    begin = time.perf_counter()
    setup = base.parser().parse_args(['--output', str(a.output), '--device', 'cpu', '--threads', '1'])
    tf, tfp, memory = base.setup(setup)
    import stan_nuts_tfp as native
    manifest = dict(schema='bayesfilter.native_tfp_stan.v1', command=sys.argv, plan=PLAN, result_file=RESULT,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        environment=sys.executable, python=platform.python_version(), tensorflow=tf.__version__, tfp=tfp.__version__,
        stan_version='2.37.0', device='cpu', gpu_intentionally_hidden=True,
        device_trust_basis='explicit_cpu_reference_using_previous_scalar_calibration',
        memory_policy=memory, jit_compile=True, tf32=False, layout='four_parallel_scalar_chains',
        intra_op_threads=1, data_version='analytic target source; regression stateless seed20261027,61',
        seeds=a.seeds, config=vars(a)|{'output':str(a.output),'stan':str(a.stan)},
        framework_startup_seconds=time.perf_counter()-begin, compiled={}, density_checks={}, results=[], source_hashes={})
    sources = [Path(__file__), Path(native.__file__), Path(common.__file__), Path(direct.__file__),
        Path(base.__file__), ROOT/'docs/benchmarks/stan_nuts_warmup.py', ROOT/'docs/benchmarks/stan_nuts.py',
        ROOT/'docs/benchmarks/nuts_benchmark_targets.py', ROOT/'docs/benchmarks/stan_targets.stan', a.stan,
        ROOT/'bayesfilter/inference/hmc_ess.py', ROOT/'bayesfilter/inference/hmc_posterior_diagnostics.py', ROOT/PLAN]
    archive=a.output/'source';archive.mkdir()
    for source in sources:
        manifest['source_hashes'][str(source.relative_to(ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
        if source.suffix:shutil.copyfile(source,archive/(source.name+'.txt'))
    tfp_root=Path(tfp.__file__).parent/'python'
    manifest['tfp_source_hashes']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
        [tfp_root/'mcmc/nuts.py',tfp_root/'mcmc/sample.py',tfp_root/'mcmc/transformed_kernel.py']}
    common.write(a.output/'manifest.json',manifest)
    try:
        with tf.device('/CPU:0'):
            for name in a.targets.split(','):
                target=make_target(tf,name);folder=a.output/name;folder.mkdir()
                compile_dir=folder/'native-tfp-compile';compile_dir.mkdir()
                functions, compiled=native_functions(tf,tfp,target,a,compile_dir)
                manifest['compiled'][name]={'native_tfp':compiled}
                controls=name in a.direct_controls.split(',')
                if controls:
                    compile_direct=folder/'direct-compile';compile_direct.mkdir()
                    direct_f, dr=direct.direct_functions(tf,target,a,compile_direct)
                    manifest['compiled'][name]['direct']=dr
                q,data=direct.inputs(tf,target,20261020)
                manifest['density_checks'][name]=common.density_check(tf,target,q,data,a,folder)
                common.write(a.output/'manifest.json',manifest)
                for index,seed in enumerate(map(int,a.seeds.split(','))):
                    starts,data=direct.inputs(tf,target,seed)
                    engines=['native_tfp','stan']+(['direct'] if controls else [])
                    order=engines[index%len(engines):]+engines[:index%len(engines)]
                    record=dict(target=name,seed=seed,order=order)
                    for engine in order:
                        out=folder/f'{engine}-{seed}';out.mkdir()
                        if engine=='native_tfp':
                            draws,stats,report=common.tf_run(tf,*functions,starts,seed,a,out)
                            report['implementation']='native_tfp_v2; stock TFP tree, sample_chain, fixed coordinate bijector'
                        elif engine=='stan':draws,stats,report=common.stan_run(tf,starts,data,seed,a,out)
                        else:draws,stats,report=direct.direct_run(tf,direct_f,starts,seed,a,out)
                        report['diagnostics']=common.assess(tf,target,draws,stats)
                        if engine=='stan':
                            report['diagnostics']['finite_cache']=None
                            report['diagnostics']['cache_note']='Internal cache not exported; successful native exit and finite retained draws checked'
                        report['workflow_seconds_excluding_compile_and_diagnostics']=(
                            report['tuned_total_seconds']+report.get('serialization_seconds',0.))
                        record[engine]=report;common.write(out/'result.json',report)
                        print(__import__('json').dumps(dict(target=name,seed=seed,engine=engine,
                            seconds=report['tuned_total_seconds'],passed=report['diagnostics']['posterior_screen_pass'])),flush=True)
                    manifest['results'].append(record);common.write(a.output/'manifest.json',manifest)
                for fn in functions:
                    if fn.experimental_get_tracing_count()!=1:raise RuntimeError('Unexpected measured retrace')
        manifest['status']='complete'
    except Exception as exc:
        manifest['status']='failed';manifest['error']=str(exc);manifest['traceback']=traceback.format_exc()
        raise
    finally:
        manifest['wall_seconds']=time.perf_counter()-begin
        manifest['host_max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        common.write(a.output/'manifest.json',manifest)


if __name__=='__main__':main()
