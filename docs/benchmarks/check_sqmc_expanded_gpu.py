"""Bounded FP64 GPU mechanics/parity and cost check, never accuracy evidence."""
from pathlib import Path
import argparse
import json
import time
import traceback

import run_sqmc_expanded_comparison as campaign


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    meta = campaign.manifest()
    records = []
    result = dict(program='fp64_gpu_xla_preflight', tuning_status='UNTUNED mechanics only',
                  nonclaims='No accuracy, ranking, production or default evidence', records=records)
    campaign.dump(out/'manifest.json', meta)
    tf = None
    try:
        # Trusted NVIDIA inventory precedes the first framework initialization.
        inventory = campaign.subprocess.check_output(
            ['nvidia-smi', '--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu',
             '--format=csv,noheader'], text=True)
        (out/'nvidia-before.txt').write_text(inventory)
        apps=campaign.subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
        (out/'nvidia-processes-before.txt').write_text(apps)
        if any(campaign.GPU_UUID in line for line in apps.splitlines()):
            raise RuntimeError('requested GPU already has a compute process; resource continuation veto')
        tf, gpu_meta = campaign.configure_gpu()
        meta.update(gpu_meta)
        campaign.dump(out/'manifest.json', meta)
        from bayesfilter.highdim.sqmc_lgssm_tf import LGSSMSpec
        from bayesfilter.highdim.sqmc_full_lgssm_tf import FullLGSSMSpec
        from bayesfilter.highdim.sqmc_campaign_tf import value_and_score, random_inputs
        fixtures = [(LGSSMSpec('p44',3),12,[2]),
                    (FullLGSSMSpec('full_matrix',3),12,[2]),
                    (FullLGSSMSpec('full_matrix',10),40,[2,8])]
        for spec,n,grids in fixtures:
            theta = spec.default_theta(tf.float64)
            observations = spec.simulate(theta,2,179001,jit_compile=True)
            for route in campaign.ROUTES:
                inputs = random_inputs(route,179002,n,spec.dimension,2,tf.float64)
                for steps in grids:
                    controls = dict(campaign.CONTROLS[0],flow_substeps=steps)
                    outputs, diagnostics, times = [], [], []
                    for jit in (False,True):
                        tick=time.perf_counter()
                        details={}
                        with tf.device('/GPU:0'):
                            value,score,valid=value_and_score(spec,route,controls,theta,observations,
                                179002,n,jit_compile=jit,inputs=inputs,diagnostics=details)
                        outputs.append((float(value.numpy()),score.numpy().tolist(),bool(valid.numpy())))
                        diagnostics.append(details)
                        times.append(time.perf_counter()-tick)
                        if 'GPU' not in value.device:
                            raise RuntimeError('candidate kernel was not placed on GPU')
                    dv=abs(outputs[0][0]-outputs[1][0])
                    ds=max(abs(a-b) for a,b in zip(outputs[0][1],outputs[1][1]))
                    scale=max(1.,*(abs(v) for v in outputs[0][1]))
                    passed=all(o[2] for o in outputs) and dv<=1e-8*(1+abs(outputs[0][0])) and ds<=1e-8*scale
                    row=dict(stage='parity',target=spec.target_id,route=route,n=n,parameters=spec.parameter_count,
                        flow_substeps=steps,value_abs_difference=dv,score_max_abs_difference=ds,
                        atol_rtol_policy='1e-8 * max(1, reference scale)',passed=passed,
                        graph_value=outputs[0][0],xla_value=outputs[1][0],
                        graph_score=outputs[0][1],xla_score=outputs[1][1],
                        graph_seconds=times[0],xla_seconds=times[1],validity=diagnostics)
                    records.append(row)
                    campaign.dump(out/'result.json',result)
                    print(json.dumps({k:v for k,v in row.items() if k not in ('graph_score','xla_score','validity')}),flush=True)
                    if not passed:
                        raise RuntimeError('GPU graph/XLA parity fixture failed')
        spec=FullLGSSMSpec('full_matrix',10)
        theta=spec.default_theta(tf.float64)
        observations=spec.simulate(theta,2,179001,jit_compile=True)
        for route in ('iid_dual_cap','repaired_permutation'):
            inputs=random_inputs(route,179002,1020,10,2,tf.float64)
            for controls in campaign.CONTROLS:
                timings=[]
                for phase in ('cold','warm'):
                    tick=time.perf_counter()
                    details={}
                    with tf.device('/GPU:0'):
                        value,score,valid=value_and_score(spec,route,controls,theta,observations,
                            179002,1020,jit_compile=True,inputs=inputs,diagnostics=details)
                    score.numpy()
                    elapsed=time.perf_counter()-tick
                    timings.append(elapsed)
                    row=dict(stage='cost',phase=phase,route=route,n=1020,parameters=157,horizon=2,
                        flow_substeps=controls['flow_substeps'],wall_seconds=elapsed,
                        valid=bool(valid.numpy()),value=float(value.numpy()),
                        score=score.numpy().tolist(),validity=details)
                    records.append(row)
                    campaign.dump(out/'result.json',result)
                    print(json.dumps({k:v for k,v in row.items() if k not in ('score','validity')}),flush=True)
                    if not row['valid']:
                        raise RuntimeError('cost fixture invalid; no affordable-cost inference')
        costs=[r for r in records if r['stage']=='cost']
        warm=max(r['wall_seconds'] for r in costs if r['phase']=='warm')
        cold=max(r['wall_seconds'] for r in costs if r['phase']=='cold')
        units=[]
        for scope,family,d,t,n in campaign.SCOPES:
            parameters=4 if family=='p44' else d*d+d*(d+1)//2+2
            estimated=4*(9*warm*(t/2)*(parameters/157)*(n/1020)**2+2*cold+15)
            units.append(dict(scope=scope,estimated_seconds=estimated))
        result.update(status='passed',projection=dict(scope_estimates=units,
            base_seconds=sum(r['estimated_seconds'] for r in units),
            conservative_seconds=1.5*sum(r['estimated_seconds'] for r in units),
            assumptions='Linear T and direction cost; N squared; full-d10 per-direction cost for smaller d; two cold compilations and 15 s startup per unit; 50% margin. This is a feasibility estimate, not a hard performance guarantee.'))
        meta['status']='passed'
        if campaign.source_hashes()!=meta['source_sha256']:
            raise RuntimeError('source changed during preflight')
    except BaseException as error:
        meta.update(status='check_failed',error=repr(error))
        result.update(status='check_failed',error=repr(error))
        traceback.print_exc()
        raise
    finally:
        meta.update(wall_seconds=time.perf_counter()-started,finished_utc=campaign.now())
        if tf is not None:
            meta['gpu_allocator_bytes']=tf.config.experimental.get_memory_info('GPU:0')
        campaign.dump(out/'result.json',result)
        campaign.dump(out/'manifest.json',meta)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    main(Path(parser.parse_args().output).resolve())
