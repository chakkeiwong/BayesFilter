"""Independent post-run diagnostic of archived fixed-HMC coordinates/caches.

Reads the numerical archive directly; does not resume, retune, choose a member
or alter the posterior screens. Affine composition and fresh target values
test whether the benchmark assessed the intended coordinates.
"""
import argparse
import json
from pathlib import Path
import time

import benchmark_nuts_xla as base
import benchmark_stan_transplant as common
from nuts_benchmark_targets import make_target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit',type=Path,required=True)
    parser.add_argument('--target',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    cfg=base.parser().parse_args(['--device','cpu','--output',str(args.output),'--threads','1'])
    tf,_,_=base.setup(cfg)
    from bayesfilter.inference.hmc_candidate_set_execution import _tensor_from_payload,_trace_from_payload
    spec=json.loads((args.fit/'tuning/execution_spec.json').read_text())['execution']
    target=make_target(tf,args.target);d=target.dimension
    matrix=tf.eye(d,dtype=tf.float64);center=tf.zeros([d],tf.float64)
    logdet=tf.constant(0.,tf.float64)
    for layer in reversed(spec['layers']):
        assert layer['kind'] in ('bootstrap_mass','fixed_mass'),layer['kind']
        artifact=layer['artifact']
        factor=tf.constant(artifact['factor'],tf.float64)
        offset=tf.constant(artifact['position'],tf.float64)
        center=tf.linalg.matvec(factor,center)+offset
        matrix=factor@matrix
        logdet+=tf.linalg.slogdet(factor)[1]
    @tf.function(input_signature=[tf.TensorSpec([None,d],tf.float64)],autograph=False,jit_compile=True)
    def values(q):return target.log_prob(q)
    result=dict(target=args.target,fit=str(args.fit),interpretation='engineering diagnostic only',members=[])
    began=time.perf_counter()
    for path in sorted(args.fit.glob('retained-*/retained_archive.json')):
        archive=json.loads(path.read_text());trace=_trace_from_payload(archive['trace'])
        active=_tensor_from_payload(archive['active_samples'])
        physical=_tensor_from_payload(archive['position_samples'])
        fresh=tf.reshape(values(tf.reshape(physical,[-1,d])),physical.shape[:-1])
        mapped=tf.linalg.matvec(matrix,active)+center
        offset=trace['target_log_prob']-fresh
        row=dict(archive=str(path),candidate=archive['candidate'],
            max_coordinate_residual=float(tf.reduce_max(tf.abs(mapped-physical))),
            max_cached_log_prob_residual=float(tf.reduce_max(tf.abs(offset-offset[0,0]))),
            constant_target_offset=float(offset[0,0]),expected_jacobian_offset=float(logdet),
            all_retained_score_finite=bool(tf.reduce_all(trace['accepted_target_score_finite'])))
        if args.target=='gaussian32':
            initial=_tensor_from_payload(archive['initial_active_state'])
            previous=tf.concat((initial[None],active[:-1]),axis=0)
            previous_q=tf.linalg.matvec(matrix,previous)+center
            proposed_q=tf.linalg.matvec(matrix,trace['proposed_state'])+center
            energy_ratio=(-.5*tf.reduce_sum(tf.square(proposed_q),-1)
                +.5*tf.reduce_sum(tf.square(previous_q),-1)
                +.5*tf.reduce_sum(tf.square(trace['initial_momentum']),-1)
                -.5*tf.reduce_sum(tf.square(trace['final_momentum']),-1))
            selected=tf.where(trace['is_accepted'][...,None],trace['proposed_state'],previous)
            row['max_mh_energy_residual']=float(tf.reduce_max(tf.abs(energy_ratio-trace['log_accept_ratio'])))
            row['max_mh_selection_residual']=float(tf.reduce_max(tf.abs(selected-active)))
            # In whitened coordinates the Gaussian Hessian is A^T A. A leapfrog
            # mode rotates by 2 asin(epsilon * frequency / 2) per step.
            frequency=tf.linalg.svd(matrix,compute_uv=False)
            phase=2.*archive['candidate']['leapfrog_steps']*tf.asin(
                .5*archive['candidate']['epsilon']*frequency)
            row['smallest_abs_sin_leapfrog_mode_phase']=float(tf.reduce_min(tf.abs(tf.sin(phase))))
        result['members'].append(row)
    result['wall_seconds']=time.perf_counter()-began
    result['passed']=bool(result['members']) and all(
        x['max_coordinate_residual']<1e-10 and x['max_cached_log_prob_residual']<1e-10
        and x.get('max_mh_energy_residual',0.)<1e-10
        and x.get('max_mh_selection_residual',0.)==0.
        and x['all_retained_score_finite'] for x in result['members'])
    common.write(args.output,result)
    print(json.dumps(dict(passed=result['passed'],members=len(result['members']))))
    if not result['passed']:raise RuntimeError('Archived HMC target/coordinate mismatch')


if __name__=='__main__':main()
