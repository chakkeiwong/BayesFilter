"""Saved-map score/geometry diagnosis. No training or posterior promotion."""
from pathlib import Path

import tensorflow as tf

from bayesfilter.testing.neutra_warm_start_campaign import read_tensor,save_tensor,write_json
from bayesfilter.testing.neutra_warm_start_closure import load_flow,read_json,key,phase_report,valid_terminal_probe
from bayesfilter.testing.neutra_warm_start_targets_tf import WarmStartTarget,F64
from bayesfilter.inference.neutra_post_training import PostTrainingProbe


class GeometryProgram:
    """Independent AD Jacobian and scalar-density differences; no pfor."""
    def __init__(self,flow,target):
        d=target.parameter_dim
        def density(z):
            x,ld=flow.forward_and_logdet(z)
            return target.log_prob_kernel(x)+ld

        def evaluate(z):
            with tf.GradientTape(persistent=True,watch_accessed_variables=False) as tape:
                tape.watch(z)
                x,ld=flow.forward_and_logdet(z)
                value=target.log_prob_kernel(x)+ld
            full_score=tape.gradient(value,z)
            ld_score=tape.gradient(ld,z)
            rows=tf.TensorArray(F64,size=d,element_shape=[None,d])
            def body(i,rows):
                row=tape.gradient(x,z,output_gradients=tf.broadcast_to(tf.one_hot(i,d,dtype=F64),tf.shape(x)))
                return i+1,rows.write(i,row)
            _,rows=tf.while_loop(lambda i,*_:i<d,body,(0,rows))
            jac=tf.transpose(rows.stack(),[1,0,2])
            _,score,valid=target.value_score(x)
            pullback=flow.pullback_score_batch(z,score)
            manual_ld=flow.log_abs_det_jacobian_score_batch(z)
            manual=pullback+manual_ld
            singular=tf.linalg.svd(jac,compute_uv=False)
            return {'latent':z,'physical':x,'log_density':value,
                'log_ratio_up_to_constant':value+.5*tf.reduce_sum(z*z,1),
                'target_score':score,'target_pullback':pullback,'logdet_score':manual_ld,
                'autodiff_logdet_score':ld_score,'autodiff_score':full_score,
                'score_residual':manual+z,'jacobian':jac,'singular_values':singular,
                'condition_number':singular[:,0]/singular[:,-1],
                'manual_ad_scaled_error':tf.reduce_max(tf.abs(manual-full_score)/(1+tf.abs(full_score)),1),
                'logdet_ad_scaled_error':tf.reduce_max(tf.abs(manual_ld-ld_score)/(1+tf.abs(ld_score)),1),
                'roundtrip_error':tf.reduce_max(tf.abs(flow.inverse_theta_to_z_batch(x)-z),1),
                'valid':valid}

        def differences(z):
            steps=tf.constant([1e-4,1e-5,1e-6,1e-7],F64)
            # Static two-coordinate traversal is not a sample-wise loop.
            columns=[]
            for j in range(d):
                offset=steps[:,None,None]*tf.one_hot(j,d,dtype=F64)[None,None,:]
                plus=tf.reshape(z[None,:,:]+offset,[-1,d])
                minus=tf.reshape(z[None,:,:]-offset,[-1,d])
                columns.append((tf.reshape(density(plus),[4,-1])-tf.reshape(density(minus),[4,-1]))/(2*steps[:,None]))
            return tf.stack(columns,axis=-1)
        signature=[tf.TensorSpec([None,d],F64)]
        self.evaluate=tf.function(evaluate,input_signature=signature,jit_compile=True,autograph=False)
        self.differences=tf.function(differences,input_signature=signature,jit_compile=True,autograph=False)
        self.inverse=tf.function(flow.inverse_theta_to_z_batch,input_signature=signature,jit_compile=True,autograph=False)


def diagnose_geometry(target_name,prepared,output,cfg,seed):
    output=Path(output);target=WarmStartTarget(target_name)
    map_path=Path(cfg['closure']['geometry_map'])
    flow=load_flow(map_path,target);payload=read_json(map_path)
    write_json(output/'frozen.json',payload)
    assessment_path=cfg['closure'].get('geometry_assessment')
    if assessment_path:write_json(output/'assessment.json',read_json(assessment_path))
    probe=PostTrainingProbe(flow,target,1.,rows=1000,batch_size=1000,jit_compile=True)
    program=GeometryProgram(flow,target)
    reference=read_tensor(Path(prepared)/'validation.tensor')
    if int(reference.shape[0])<2000:raise ValueError('need two disjoint development reference subsets')
    banks=[]
    for replication in range(2):
        seed_pair=key(seed,36100+replication)
        # Small diagnostic bank, not a training or external dataset generator.
        latent=probe.latent_bank(seed_pair)
        report=probe.summarize([probe.batch(latent)],seed_pair.numpy().tolist())
        report['transport_hash']=payload['transport_hash']
        write_json(output/f'post-training-1000-{replication}.json',report)
        if replication==0:write_json(output/'post-training-1000.json',report)
        banks.append((f'base-{replication}',latent,valid_terminal_probe(report)))
        x=reference[1000*replication:1000*(replication+1)]
        posterior_latent=program.inverse(x)
        banks.append((f'posterior-{replication}',posterior_latent,True))
    summaries=[]
    for name,z,probe_valid in banks:
        rows=program.evaluate(z)
        for field,value in rows.items():
            save_tensor(output/f'{name}-{field}.tensor',value)
        norm=tf.linalg.norm(rows['score_residual'],axis=1)
        worst=tf.argsort(norm,direction='DESCENDING')[:8]
        ordinary=tf.argsort(norm)[492:500]
        indices=tf.concat((worst,ordinary),0)
        chosen=tf.gather(z,indices);fd=program.differences(chosen)
        expected=tf.gather(rows['autodiff_score'],indices)
        errors=tf.reduce_max(tf.abs(fd-expected[None,:,:])/(1+tf.abs(expected[None,:,:])),2)
        save_tensor(output/f'{name}-fd.tensor',fd)
        write_json(output/f'{name}-derivatives.json',{'row_indices':indices,
            'steps':[1e-4,1e-5,1e-6,1e-7],'autodiff':expected,'scaled_errors_by_step':errors,
            'definition':'max_coordinate_absolute_error/(1+absolute_autodiff_coordinate)',
            'best_step_scaled_error_per_row':tf.reduce_min(errors,0)})
        finite=all(bool(tf.reduce_all(tf.math.is_finite(v)).numpy()) for v in rows.values() if v.dtype!=tf.bool)
        ad=float(tf.reduce_max(rows['manual_ad_scaled_error']).numpy())
        ld_ad=float(tf.reduce_max(rows['logdet_ad_scaled_error']).numpy())
        roundtrip=float(tf.reduce_max(rows['roundtrip_error']).numpy())
        best_fd=float(tf.reduce_max(tf.reduce_min(errors,0)).numpy())
        ordered=tf.sort(norm)
        summary={'bank':name,'measure':'standard_normal_base' if name.startswith('base') else 'development_posterior_reference_inverse',
            'reference_scope':'existing development reference, never final holdout',
            'median':ordered[499],'p95':ordered[949],'p99':ordered[989],'maximum':ordered[-1],
            'max_manual_ad_scaled_error':ad,'max_logdet_ad_scaled_error':ld_ad,
            'max_roundtrip_error':roundtrip,'max_best_fd_scaled_error':best_fd,
            'max_jacobian_singular_value':tf.reduce_max(rows['singular_values'][:,0]),
            'max_condition_number':tf.reduce_max(rows['condition_number']),
            'worst_rows':indices[:8],'worst_physical':tf.gather(rows['physical'],worst),
            'worst_target_pullback_norm':tf.linalg.norm(tf.gather(rows['target_pullback'],worst),axis=1),
            'worst_logdet_score_norm':tf.linalg.norm(tf.gather(rows['logdet_score'],worst),axis=1),
            'worst_singular_values':tf.gather(rows['singular_values'],worst),
            'finite':finite,'passed':probe_valid and finite and bool(tf.reduce_all(rows['valid']).numpy())
                and ad<=1e-9 and ld_ad<=1e-9 and roundtrip<=1e-8 and best_fd<=1e-5}
        summaries.append(summary)
        write_json(output/'geometry.json',{'transport_hash':payload['transport_hash'],
            'map_source':str(map_path),'seed':seed,'banks':summaries,
            'interpretation':'geometry descriptive; derivative and roundtrip checks numerical',
            'common_bank_seeds':[[seed,36100],[seed,36101]]})
    return phase_report(output,phase='geometry',passed=all(s['passed'] for s in summaries),
        reason='score_geometry_checked',transport_hash=payload['transport_hash'],
        continuation_veto=not all(s['passed'] for s in summaries),
        failure_class='implementation_or_numerical' if not all(s['passed'] for s in summaries) else None)
