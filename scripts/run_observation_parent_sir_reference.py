"""CPU independent-reference reweighting of saved Zhao-Cui smoothing proposals."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_NUM_INTRAOP_THREADS']='2'
os.environ['TF_NUM_INTEROP_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import sys, json, math, time, hashlib, subprocess, argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'docs/benchmarks'))
import numpy as np
from scipy.special import logsumexp
import tensorflow as tf
from run_zhao_cui_quadratic_score_reference import load_proposal
from bayesfilter.testing.sir_path_score_reference_tf import make_path_value_score
from bayesfilter.testing.zhao_cui_path_likelihood_reference import make_path_log_joint
from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec
CASES={'nominal':[0.,0.,0.], 'infection09':[math.log(.9),0.,0.], 'infection11':[math.log(1.1),0.,0.],
       'recovery09':[0.,math.log(.9),0.], 'recovery11':[0.,math.log(1.1),0.],
       'noise08':[0.,0.,math.log(.8)], 'noise125':[0.,0.,math.log(1.25)]}
ZROOT=Path('/home/chakwong/BayesFilter-SQMC/docs/plans/artifacts/ledh-zhao-horizons-20261006-01')
FITS=['037-zhao-sir_d18-rank20-fit17-runtime-repair','036-zhao-sir_d18-rank40-fit2-runtime-repair']
PLAN='docs/plans/observation-parent-default-and-sir-accuracy-20261028.md'

def save(p,v):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');temp.replace(p)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--limit',type=float,default=1500);ap.add_argument('--local-only',action='store_true');a=ap.parse_args()
    cases={k:[math.log(v) if j==i else 0. for j in range(3)] for i,name in [(0,'infection'),(1,'recovery')] for k,v in [(name+'0999',.999),(name+'1001',1.001)]} if a.local_only else CASES
    a.out.mkdir(parents=True,exist_ok=False);start=time.monotonic();rows=[]
    manifest=dict(command=sys.argv,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        environment=sys.executable,device='CPU reference; GPU intentionally hidden',jit_compile=True,plan=PLAN,
        data_seed=26100611,paths_per_fit=100000,source_anchors=['Zhao-Cui 2024 eq26 Algorithm4 p18','third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:139-206'],
        method='fixed_hmc_adaptation saved target-adapted smoothing proposals; exact path reweighting at evaluation theta; not a new TT fit',
        caveat='rank and fit seed both change; conditional path MCSE omits TT support bias',rows_file=str(a.out/'rows.json'), result_file='docs/plans/observation-parent-sir-accuracy-result-20261028.md',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'bayesfilter/testing/sir_path_score_reference_tf.py',ROOT/'bayesfilter/testing/zhao_cui_path_likelihood_reference.py']})
    save(a.out/'manifest.json',manifest)
    for horizon in (10,20,50):
        kernel=make_path_value_score(horizon,1000);independent=make_path_log_joint('sir_d18',horizon)
        for fit in FITS:
            paths,raw,history,obs,meta=load_proposal(ZROOT/fit,NonlinearSQMCSpec('sir_d18'),horizon)
            observations=tf.constant(obs,tf.float64)
            # Full canonical-callback parity and analytical derivative on a fixed chunk.
            sample=tf.constant(paths[:1000],tf.float64);theta=tf.constant([.03,-.02,.04],tf.float64)
            v,s=kernel(theta,sample,observations);iv=independent(theta,sample,observations)
            parity=float(tf.reduce_max(tf.abs(v-iv)));assert parity<1e-7,parity
            fd=[]
            for k in range(3):
                delta=tf.one_hot(k,3,dtype=tf.float64)*1e-5
                difference=(independent(theta+delta,sample,observations)-independent(theta-delta,sample,observations))/(2e-5)
                err=float(tf.reduce_max(tf.abs(difference-s[:,k])/(1+tf.abs(s[:,k]))));fd.append(err)
            assert max(fd)<1e-4,fd
            baseline=[]
            for j in range(0,len(paths),1000): baseline.append(kernel(tf.zeros([3],tf.float64),tf.constant(paths[j:j+1000],tf.float64),observations)[0].numpy())
            baseline=np.concatenate(baseline)
            for case,q in cases.items():
                if time.monotonic()-start>a.limit: raise TimeoutError('reference stage budget')
                vals=[];scores=[]
                for j in range(0,len(paths),1000):
                    v,s=kernel(tf.constant(q,tf.float64),tf.constant(paths[j:j+1000],tf.float64),observations);vals.append(v.numpy());scores.append(s.numpy())
                values=np.concatenate(vals);score=np.concatenate(scores);lw=raw+values-baseline
                norm=logsumexp(lw);weights=np.exp(lw-norm);value=float(norm-math.log(len(lw)));est=np.sum(weights[:,None]*score,axis=0)
                ess=float(1/np.sum(weights**2));jk=[]
                for group in range(20):
                    keep=np.arange(len(lw))//(len(lw)//20)!=group
                    z=logsumexp(lw[keep]);w=np.exp(lw[keep]-z)
                    jk.append([z-math.log(int(np.sum(keep))),*np.sum(w[:,None]*score[keep],axis=0)])
                jk=np.asarray(jk);se=np.sqrt(19/20*np.sum((jk-jk.mean(axis=0))**2,axis=0))
                assert np.all(np.isfinite(lw)) and np.all(np.isfinite(score)), 'nonfinite reference'
                row=dict(horizon=horizon,case=case,theta=q,fit=fit,value=value,score=est.tolist(),conditional_jackknife_se=se.tolist(),importance_ess=ess,
                    samples=len(lw),reference_overlap_pass=ess>=1000,metadata=meta,callback_value_parity=parity,complete_data_score_fd_scaled_error=fd,
                    wall_seconds=time.monotonic()-start)
                rows.append(row);save(a.out/'rows.json',rows);print(json.dumps({k:row[k] for k in ('horizon','case','fit','value','score','importance_ess','wall_seconds')}),flush=True)
    manifest.update(wall_seconds=time.monotonic()-start,status='complete',row_count=len(rows));save(a.out/'manifest.json',manifest)
if __name__=='__main__': main()
