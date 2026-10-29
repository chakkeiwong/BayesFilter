"""Frozen Gabrié measure and forward-gradient diagnostics, not training."""
from pathlib import Path

import tensorflow as tf

from bayesfilter.inference.neutra_warm_start_tf import GabrieProgram
from bayesfilter.testing.neutra_warm_start_campaign import read_tensor,save_tensor,write_json
from bayesfilter.testing.neutra_warm_start_closure import (
    load_flow,read_json,key,cloud_assessment,gabrie_measure_summary,phase_report)
from bayesfilter.testing.neutra_warm_start_policy import select_mala_candidate
from bayesfilter.testing.neutra_warm_start_targets_tf import WarmStartTarget,F64


def forward_directional_check(flow, rows, seed):
    """Check the actual inverse-density objective at one fixed parameter direction."""
    variables=tuple(flow.trainable_variables)
    directions=[tf.random.stateless_normal(v.shape,key(seed,35100+i),dtype=v.dtype)
                for i,v in enumerate(variables)]
    norm=tf.linalg.global_norm(directions)
    directions=[v/norm for v in directions]

    @tf.function(input_signature=[tf.TensorSpec(rows.shape,F64)],jit_compile=True,autograph=False)
    def loss_and_gradient(x):
        with tf.GradientTape() as tape:
            loss=-tf.reduce_mean(flow.log_prob(x))
        gradient=tape.gradient(loss,variables)
        derivative=tf.add_n([tf.reduce_sum(g*d) for g,d in zip(gradient,directions)])
        return loss,derivative,tf.linalg.global_norm(gradient)

    original=[tf.identity(v) for v in variables]
    loss,analytic,gradient_norm=loss_and_gradient(rows)
    step=tf.constant(1e-5,F64)
    try:
        for variable,value,direction in zip(variables,original,directions):
            variable.assign(value+step*direction)
        plus=loss_and_gradient(rows)[0]
        for variable,value,direction in zip(variables,original,directions):
            variable.assign(value-step*direction)
        minus=loss_and_gradient(rows)[0]
    finally:
        for variable,value in zip(variables,original):variable.assign(value)
    finite_difference=(plus-minus)/(2*step)
    error=tf.abs(finite_difference-analytic)
    passed=bool((tf.math.is_finite(error)&(error<1e-6)&tf.math.is_finite(gradient_norm)).numpy())
    return {'passed':passed,'loss':loss,'analytic':analytic,'finite_difference':finite_difference,
        'absolute_error':error,'gradient_norm':gradient_norm,'step':step,'absolute_tolerance':1e-6,
        'direction':'one unit Gaussian parameter direction, seeds [seed,35100+variable_index]',
        'role':'implementation check; not population stationarity or optimizer convergence'}


def diagnose(target_name,prepared,parent,output,cfg,seed):
    target=WarmStartTarget(target_name);prepared=Path(prepared);parent=Path(parent);output=Path(output)
    total=max(int(p.name.split('-')[1]) for p in parent.glob('warm-*-checkpoint.json'))
    flow=load_flow(parent/f'warm-{total}-frozen.json',target,trainable=True)
    calibration=read_json(parent.parent/'mutation-calibration.json')
    dt=select_mala_candidate(calibration)['step_size']
    reference=read_tensor(prepared/'validation.tensor');reps=read_tensor(prepared/'discovered_modes.tensor')
    settings=cfg.get('closure',{});walkers=cfg['walkers']
    burn=settings.get('diagnostic_burn',256);steps=settings.get('diagnostic_steps',1024)
    warm_program=GabrieProgram(flow,target,walkers=walkers,steps=burn)
    program=GabrieProgram(flow,target,walkers=walkers,steps=steps)
    groups={'saved':read_tensor(parent/f'walkers-{total}.tensor')}
    for i in range(int(reps.shape[0])):
        groups[f'mode{i}']=tf.repeat(reps[i:i+1],walkers,axis=0)+.2*tf.random.stateless_normal(
            [walkers,target.parameter_dim],key(seed,35200+i),dtype=F64)
    reports=[];derivative=None
    for i,(name,current) in enumerate(groups.items()):
        save_tensor(output/f'{name}-initial.tensor',current)
        current,warm,_,_,bad0=warm_program.run(current,tf.constant(dt,F64),key(seed,35300+i))
        _,retained,ga,la,bad=program.run(current,tf.constant(dt,F64),key(seed,35400+i))
        trace=tf.reshape(retained,[steps,walkers,target.parameter_dim])
        save_tensor(output/f'{name}-warmup.tensor',tf.reshape(warm,[burn,walkers,target.parameter_dim]))
        save_tensor(output/f'{name}-retained.tensor',trace)
        clouds=tf.unstack(tf.transpose(trace,[1,0,2]),axis=0)
        assessment=cloud_assessment(target,clouds,[tf.zeros([steps],F64)]*walkers,reference)
        row={'group':name,'assessment':assessment,'measure':gabrie_measure_summary(target,retained),
            'global_acceptance':ga,'local_acceptance':la,'invalid_proposals':bad+bad0,
            'passed':assessment['passed'] and int((bad+bad0).numpy())==0}
        reports.append(row);write_json(output/'measure.json',reports)
        if derivative is None:
            derivative=forward_directional_check(flow,retained[:cfg['batch_size']],seed)
            write_json(output/'forward-derivative.json',derivative)
    invalid=any(int(r['invalid_proposals'].numpy()) for r in reports) or not derivative['passed']
    return phase_report(output,phase='diagnose',passed=not invalid and all(r['passed'] for r in reports),
        reason='frozen_measure_and_gradient_assessed',continuation_veto=invalid,
        failure_class='implementation_or_numerical' if invalid else 'finite_measure_screen',
        warmup_steps=burn,retained_steps=steps,walkers=walkers,mala_step_size=dt,
        parent=str(parent),derivative_passed=derivative['passed'],
        exact_mixture_right_mass='2/3 to Gaussian tail correction',
        exact_mixture_valley_probability='Phi(7)-Phi(3)',
        repair='longer_frozen_measure_check' if not all(r['passed'] for r in reports) else None)
