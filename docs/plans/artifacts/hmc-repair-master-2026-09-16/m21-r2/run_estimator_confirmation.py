"""Fresh fixed-count validation under the merged-source continuation plan."""
from pathlib import Path
import hashlib,json,math,os,statistics,sys,time
ROOT=Path(__file__).resolve().parent
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
assert os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH')=='true'
sys.path.insert(0,str(ROOT/'source-r1'))
started=time.monotonic()
import tensorflow as tf
import tensorflow_probability as tfp
from bayesfilter.testing.inference_validation.engines.controller_stopping import GaussianAR1Transition,fixed_mean_law
from bayesfilter.testing.inference_validation.engines.statistics import binomial_interval
from bayesfilter.testing.inference_validation.storage import write_tensor
from bayesfilter.inference.hmc_precision import mean_precision
OUT=ROOT/'estimator-confirmation-cpu-r1';OUT.mkdir(exist_ok=False)
CASES=(('slow_stationary',.98,(0.,)*4,True),('dispersed',.995,(-8.,-4.,4.,8.),False))
ARMS=('lugsail_100','lugsail_500','autocorrelation')
manifest={'plan_file':'docs/plans/bayesfilter-hmc-merged-source-continuation-2026-09-22.md','source_manifest':str(ROOT/'source-manifest-r1.json'),'source_identity':json.loads((ROOT/'source-manifest-r1.json').read_text())['source_identity'],'git_commit':json.loads((ROOT/'source-manifest-r1.json').read_text())['git_commit'],'command':[sys.executable,*sys.argv],'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'seed':2026092244,'replications_per_case':400,'planned':800,'warmup':10000,'retained':10000,'chains':4,'cases':CASES,'arms':ARMS,'cpu_worker_limit_seconds':900,'device':'cpu_reference','gpu_intentionally_hidden':True,'jit_compile':False,'tensorflow':tf.__version__,'tfp':tfp.__version__,'environment':{k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','TF_FORCE_GPU_ALLOW_GROWTH','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')},'data_version':'synthetic exact Gaussian AR1 laws declared in this manifest'}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
results={}
for case_index,(name,rho,starts,stationary) in enumerate(CASES):
    rows=[];directory=OUT/name;directory.mkdir()
    transition=GaussianAR1Transition(rho,jit_compile=False)
    law=fixed_mean_law(rho,starts,warmup=10000,draws=10000,stationary_start=stationary)
    for rep in range(400):
        if time.monotonic()-started>870:
            raise TimeoutError('fixed inventory stopped at declared budget; preserve all partial rows')
        counter=10000*case_index+3*rep
        state=(tf.random.stateless_normal((4,1),(2026092244,counter),dtype=tf.float64) if stationary else tf.constant(starts,tf.float64)[:,None])
        values=transition(state,num_results=20000,seed=(2026092244,counter+1),stage='fixed')['posterior_samples'][10000:]
        assert bool(tf.reduce_all(tf.math.is_finite(values)))
        tensor_path=directory/f'rep-{rep:04d}.tensor'
        write_tensor(tensor_path,values)
        estimate=float(tf.reduce_mean(values))
        row={'rep':rep,'seed_counter':counter,'array_sha256':hashlib.sha256(tensor_path.read_bytes()).hexdigest(),'oracle_covered':abs(estimate-law['mean'])<=1.959963984540054*law['mcse'],'arms':{}}
        for arm in ARMS:
            report=mean_precision(values,method='autocorrelation' if arm=='autocorrelation' else 'lugsail',batch_size=None if arm=='autocorrelation' else int(arm.rsplit('_',1)[1]),jit_compile=False)
            valid=bool(report['valid'][0]);se=float(report['mcse'][0]) if valid else None
            row['arms'][arm]={'available':valid,'covered':valid and abs(estimate)<=1.959963984540054*se,'estimate':estimate,'mcse':se,'mcse_over_exact':se/law['mcse'] if valid else None}
        rows.append(row)
        (directory/f'rep-{rep:04d}.json').write_text(json.dumps(row,indent=2)+'\n')
    summary={'planned':400,'recorded':len(rows),'exact_law':law,'arms':{},'rows':rows}
    for arm in ARMS:
        available=sum(r['arms'][arm]['available'] for r in rows)
        covered=sum(r['arms'][arm]['covered'] for r in rows)
        interval=binomial_interval(covered,400)
        summary['arms'][arm]={'available':available,'covered':covered,'denominator':400,'coverage_interval':interval,'median_mcse_over_exact':statistics.median(r['arms'][arm]['mcse_over_exact'] for r in rows if r['arms'][arm]['available']),'eligible_under_declared_screen':available==400 and interval[0]>=.90}
    count=sum(r['oracle_covered'] for r in rows)
    interval=binomial_interval(count,400,alpha=.05/2)
    summary['oracle']={'covered':count,'interval':interval,'alpha':.05/2,'screen_passed':interval[0]<=.95<=interval[1]}
    results[name]=summary
    (OUT/'result.json').write_text(json.dumps({'manifest':manifest,'cases':results,'complete':len(results)==2,'ranking_supported':False,'default_promoted':False,'anytime_coverage_established':False,'elapsed_seconds':time.monotonic()-started},indent=2)+'\n')
    print(name,json.dumps({k:v for k,v in summary.items() if k!='rows'}),flush=True)
