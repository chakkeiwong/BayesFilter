"""GPU diagnostic: skipped operations must not falsely flag healthy batch rows."""
import json,sys,time
from pathlib import Path
import validate_lorenz_first_failure_tf as baseline

tf,D=baseline.tf,baseline.D


def main():
    baseline.main()
    output=Path(sys.argv[sys.argv.index('--output')+1])
    first=json.loads((output/'summary.json').read_text())
    (output/'single_replica_summary.json').write_text(json.dumps(first,indent=2)+'\n')
    started=time.monotonic()
    result=dict(status='FAIL',scope='FIRST_FAILURE_ROW_ATTRIBUTION_DIAGNOSTIC',
        gpu_memory_policy=baseline.memory,device='GPU',jit_compile=True,tf32=False,dtype='float64',
        checks=first['checks'],candidate_admitted=False,replication='INCOMPLETE',
        guard_full_scope_parity='FAILED_132_133_NOT_RETESTED',
        heuristic_dominance_verdict='NOT_ASSESSED_CONTROL_FLOW_ONLY')
    def check(name,condition,**extra):
        result['checks'].append(dict(name=name,passed=bool(condition),**extra))
        if not condition:raise ValueError(name)
    try:
        h,n=4,128
        controller=baseline.LorenzRefinement(forcing=2.5,first_failure_guards=True,
            observations=h,particles=n,replicas=2)
        e=controller.engine;y=tf.zeros([h,2,6],D)
        normals=tf.random.stateless_normal([h,2,n,8],[20261002,137],dtype=D)
        uniform=tf.fill([h,2],tf.constant(.3,D));policy=controller.initial_policy(y)
        healthy=e.forward(*policy,y,normals,uniform)
        check('two_rows_healthy',healthy['valid'].numpy())
        def rejected(name,actual,codes):
            check(name,not actual['valid'].numpy() and actual['failure_codes'].numpy().tolist()==codes
                and int(actual['first_failure_time'].numpy())==0
                and bool(tf.reduce_all(actual['ancestors']==-1).numpy()),
                actual_codes=actual['failure_codes'].numpy().tolist())
        a=tf.tensor_scatter_nd_update(policy[0],[[0,0,0,0]],[-100.])
        rejected('only_invalid_gaussian_row_flagged',e.forward(a,*policy[1:],y,normals,uniform),[2,0])
        badnormal=tf.tensor_scatter_nd_update(normals,[[0,1,0,0]],[float('nan')])
        rejected('only_invalid_particle_row_flagged',e.forward(*policy,y,badnormal,uniform),[0,3])
        baduniform=tf.tensor_scatter_nd_update(uniform,[[0,1]],[1.])
        rejected('only_invalid_uniform_row_flagged',e.forward(*policy,y,normals,baduniform),[0,7])
        # Even another invalid input at a later stage is unexecuted once the
        # Gaussian stage has rejected the batch; report the first operation.
        rejected('later_stage_does_not_override_first',e.forward(a,*policy[1:],y,badnormal,baduniform),[2,0])
        fit=e.fit(*policy,healthy['x'],healthy['log_incremental_weight'])
        check('two_rows_healthy_fit',fit['valid'].numpy())
        weights=tf.tensor_scatter_nd_update(healthy['log_incremental_weight'],[[h-1,0,0]],[float('nan')])
        bad=e.fit(*policy,healthy['x'],weights)
        check('skipped_qr_not_a_rank_failure',not bad['valid'].numpy()
            and int(bad['failure_code'].numpy())==9
            and bad['failure_rows'].numpy().tolist()==[True,False]
            and int(bad['first_failure_time'].numpy())==h-1,
            actual_rows=bad['failure_rows'].numpy().tolist())
        check('two_rows_stable_traces',e.forward.experimental_get_tracing_count()==1
            and e.fit.experimental_get_tracing_count()==1)
        result['status']='PASS'
    except Exception as exc:
        result['error']=repr(exc)
        import traceback;traceback.print_exc()
    result['wall_seconds']=first['wall_seconds']+time.monotonic()-started
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],checks=len(result['checks']),error=result.get('error'))))
    if result['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
