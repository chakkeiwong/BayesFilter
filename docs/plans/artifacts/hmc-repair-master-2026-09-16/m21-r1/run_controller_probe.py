"""Bounded actual-controller diagnostic; see the M21 design."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

CASES={
    'iid':(0.,'stationary'), 'correlated':(.8,'stationary'),
    'slow_stationary':(.98,'stationary'), 'slow_shifted':(.98,'shifted'),
    'dispersed':(.995,'dispersed'), 'common_shift':(.9999,'shifted'),
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',choices=CASES,required=True)
    p.add_argument('--replications',type=int,required=True)
    p.add_argument('--start-rep',type=int,default=0)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--seconds',type=float,required=True)
    p.add_argument('--device',choices=('cpu_reference','gpu'),required=True)
    a=p.parse_args()
    assert a.replications>0 and a.seconds>0 and a.start_rep>=0
    assert os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH')=='true'
    gpu=a.device=='gpu'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') not in (None,'','-1') if gpu else os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    a.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();deadline=started+a.seconds
    sys.path.insert(0,str(a.source.resolve()))
    import tensorflow as tf
    if gpu:
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        memory=configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    else:
        memory={'gpu_intentionally_hidden':True}
    from bayesfilter.inference.neutra_hmc import SequentialExactTransitionConfig,run_sequential_exact_transition
    from bayesfilter.inference.hmc_precision import HMCPrecisionPolicy,HMCPrecisionTarget
    from bayesfilter.inference.hmc_posterior_assessment import HMCPosteriorAssessmentPolicy
    from bayesfilter.testing.inference_validation.engines.controller_stopping import GaussianAR1Transition,fixed_mean_law,mean_interval
    from bayesfilter.testing.inference_validation.designs import seed_for
    from bayesfilter.testing.inference_validation.storage import write_json,write_tensor
    policy=HMCPosteriorAssessmentPolicy(precision=HMCPrecisionPolicy(
        (HMCPrecisionTarget('x',mcse_absolute_max=.05),),method='lugsail',jit_compile=gpu))
    rho,regime=CASES[a.case]
    transition=GaussianAR1Transition(rho,jit_compile=gpu)
    manifest={'command':sys.argv,'python':sys.executable,'tensorflow':tf.__version__,
        'source':str(a.source.resolve()),'source_manifest':str(a.source.parent/'source-manifest-r1.json'),
        'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'plan':'docs/plans/bayesfilter-hmc-repair-m21-design-2026-09-22.md',
        'result':str(a.output/'summary.json'),'device':a.device,'memory_policy':memory,
        'jit_compile':gpu,'cpu_exception':'explicit diagnostic/reference controller validation' if not gpu else None,
        'case':a.case,'rho':rho,'regime':regime,'root_seed':a.seed,
        'planned_replications':a.replications,'start_rep':a.start_rep,'maximum_seconds':a.seconds,
        'environment':{k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','TF_FORCE_GPU_ALLOW_GROWTH','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS')},
        'data_version':'N/A: exact synthetic Gaussian transition','actual_controller_stopping_test':True,
        'not_an_hmc_tuning_test':True,'statistical_unit':'independent four-chain complete controller replication'}
    write_json(a.output/'manifest.json',manifest)
    rows=[]
    for rep in range(a.start_rep,a.start_rep+a.replications):
        if time.monotonic()>=deadline:break
        root=a.output/f'rep-{rep:04d}';root.mkdir()
        rep_started=time.monotonic()
        streams={part:seed_for(a.seed,a.case,rep,part) for part in ('initial_stopped','initial_fixed','warmup','retained','fixed_warmup','fixed_retained')}
        assert len(set(streams.values()))==len(streams)
        def initial(arm):
            if regime=='stationary':return tf.random.stateless_normal((4,1),streams['initial_'+arm],dtype=tf.float64)
            return tf.constant([[-8.],[-4.],[4.],[8.]] if regime=='dispersed' else [[8.]]*4,tf.float64)
        def archive(**data):
            stem=f"{data['stage']}-"+('cumulative' if data['cumulative'] else str(data['chunk_index']))
            path=write_tensor(root/(stem+'.tensor'),data['posterior_samples'])
            return {'path':str(path),'seed':data['seed']}
        try:
            config=SequentialExactTransitionConfig(transition.signature,streams['warmup'],streams['retained'],
                warmup_chunk_results=500,warmup_min_results=2000,warmup_check_window_results=1000,
                warmup_max_results=10000,retained_chunk_results=500,retained_min_results=1000,
                retained_max_results=10000,assessment_policy=policy)
            begin=initial('stopped');write_tensor(root/'initial-stopped.tensor',begin)
            result=run_sequential_exact_transition(transition_program=transition,initial_transition_state=begin,
                posterior_state_fn=lambda x:x,parameter_names=('x',),config=config,
                archive_callback=archive,budget_check=lambda count:time.monotonic()<deadline)
            stopped=mean_interval(result['private_retained_beta_one'],jit_compile=gpu)
            write_json(root/'controller.json',{k:v for k,v in result.items() if not k.startswith('private_')})
            row={'rep':rep,'status':'complete','streams':streams,'stopped':stopped,
                'passed':result['passed'],'warmup_passed':result['warmup_passed'],
                'warmup_count':result['warmup_results_per_chain'],'retained_count':result['retained_results_per_chain'],
                'warmup_cap':result['warmup_cap_hit'],'retained_cap':result['retained_cap_hit'],
                'hard_vetoes':result['hard_vetoes'],
                'initial_mean_at_warmup_end':0. if regime=='stationary' else float(tf.reduce_mean(begin))*rho**result['warmup_results_per_chain'],
                'initial_max_abs_mean_at_warmup_end':0. if regime=='stationary' else float(tf.reduce_max(tf.abs(begin)))*rho**result['warmup_results_per_chain']}
            if time.monotonic()<deadline:
                fixed_initial=initial('fixed');write_tensor(root/'initial-fixed.tensor',fixed_initial)
                fixed_warmup=transition(fixed_initial,num_results=10000,seed=tf.constant(streams['fixed_warmup']),stage='fixed')['posterior_samples']
                fixed=transition(fixed_warmup[-1],num_results=10000,seed=tf.constant(streams['fixed_retained']),stage='fixed')['posterior_samples']
                write_tensor(root/'fixed-warmup.tensor',fixed_warmup);write_tensor(root/'fixed-retained.tensor',fixed)
                law=fixed_mean_law(rho,[float(x) for x in tf.unstack(fixed_initial[:,0])],warmup=10000,draws=10000,stationary_start=regime=='stationary')
                estimate=float(tf.reduce_mean(fixed))
                row.update(fixed=mean_interval(fixed,jit_compile=gpu),fixed_exact_law=law,
                    fixed_oracle={'available':True,'covered':abs(estimate)<=1.959963984540054*law['mcse'],
                        'covered_transient_expectation':abs(estimate-law['mean'])<=1.959963984540054*law['mcse'],
                        'z_against_exact_expectation':(estimate-law['mean'])/law['mcse']})
            else:
                row.update(status='fixed_unavailable_at_budget',fixed={'available':False,'covered':False})
        except Exception as exc:
            row={'rep':rep,'status':'implementation_failure','error':repr(exc),'traceback':traceback.format_exc(),'streams':streams}
        row['elapsed_seconds']=time.monotonic()-rep_started
        rows.append(row);write_json(root/'result.json',row)
        print(json.dumps({k:row.get(k) for k in ('rep','status','passed','warmup_count','retained_count','elapsed_seconds','error')}),flush=True)
        if row['status']=='implementation_failure':break
    summary={'manifest':manifest,'rows':rows,'completed_replications':sum(r['status']=='complete' for r in rows),
        'recorded_replications':len(rows),'planned_replications':a.replications,'elapsed_seconds':time.monotonic()-started,
        'complete':len(rows)==a.replications and all(r['status']=='complete' for r in rows),
        'ranking_supported':False,'default_promoted':False,'anytime_coverage_claim':False}
    write_json(a.output/'summary.json',summary)
    return 0 if summary['complete'] else 1

if __name__=='__main__':raise SystemExit(main())
