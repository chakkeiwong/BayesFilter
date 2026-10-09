"""Stage-separated research diagnostics using the canonical TF NeuTra authority.

All thresholds are explicit hypotheses in the September 30 repair plan.
This module does not implement an alternative flow or HMC sampler.
"""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.testing.neutra_warm_start_targets_tf import WarmStartTarget, BroadStudentProposal, F64, wiggle_quadrature
from bayesfilter.testing.neutra_warm_start_campaign import (
    read_tensor, save_tensor, write_json, make_transport, gradient_calibration,
    TrainingBlock, gpu_preflight)
from bayesfilter.testing.neutra_warm_start_policy import (
    MALACalibrationFailure, select_mala_candidate, fit_repair_decision, teacher_repair_settings)
from bayesfilter.inference.neutra_warm_start_tf import (
    discover_modes, AnnealedSMC, SMCConfig, MALAProgram, GabrieProgram)
from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
from bayesfilter.inference.neutra_post_training import PostTrainingProbe


def key(seed, role):
    return tf.constant([int(seed), int(role)], tf.int32)


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_continuation_teacher(parent, teacher):
    """Require the exact weighted empirical distribution used by the parent."""
    inputs=read_json(Path(parent).parent/'manifest.json')['input_sha256']
    checked={}
    for name in ('teacher-particles.tensor','teacher-log-weights.tensor'):
        previous=[digest for path,digest in inputs.items() if Path(path).name==name]
        if len(previous)!=1:
            raise ValueError(f'Continuation teacher identity missing or ambiguous: {name}')
        path=Path(teacher)/name
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=previous[0]:raise ValueError(f'Continuation teacher hash mismatch: {name}')
        checked[name]=actual
    return checked


def scaled_displacement(displacement, variance):
    """Explanatory movement in squared coordinate SDs; no sampler decision."""
    displacement=tf.convert_to_tensor(displacement,F64)
    variance=tf.convert_to_tensor(variance,F64)
    valid=tf.math.is_finite(variance)&(variance>0)
    scaled=tf.where(valid,tf.math.divide_no_nan(displacement,variance),
                    tf.fill(tf.shape(variance),tf.constant(math.nan,F64)))
    return {'coordinate_variance':variance,'scale_valid':valid,
            'variance_scaled_displacement':scaled,'role':'explanatory_only',
            'zero_variance_interpretation':'undefined; never evidence of mobility'}


def valid_terminal_probe(report):
    return (report.get('complete') is True and report.get('finite') is True
            and report.get('rows')==1000 and report.get('valid_rows')==1000)


def phase_report(output, **values):
    values.update(posterior_correctness_established=False, default_promotion=False)
    write_json(Path(output)/'phase.json', values)
    return values


def temporal_calibration(output,seed=11,replications=2048):
    """CPU reference calibration of the optional block diagnostic, no HMC claim."""
    from bayesfilter.inference.hmc_verification import temporal_block_conflicts
    rows=[]
    for count in (64,256):
        @tf.function(input_signature=[tf.TensorSpec([],F64),tf.TensorSpec([2],tf.int32)],
                     jit_compile=True,autograph=False)
        def reference(rho,seed_pair):
            initial=tf.cast(tf.random.stateless_uniform([replications,4],seed_pair,dtype=F64)<.7,F64)
            trace=tf.TensorArray(F64,size=count,element_shape=[replications,4])
            def body(t,state,trace):
                k=tf.random.experimental.stateless_fold_in(seed_pair,t+1)
                fresh=tf.cast(tf.random.stateless_uniform([replications,4],k,dtype=F64)<.7,F64)
                keep=tf.random.stateless_uniform([replications,4],tf.random.experimental.stateless_fold_in(k,1),dtype=F64)<rho
                state=tf.where(keep,state,fresh)
                return t+1,state,trace.write(t,state)
            _,_,trace=tf.while_loop(lambda t,*_:t<count,body,(0,initial,trace))
            return tf.transpose(tf.reduce_mean(tf.reshape(trace.stack(),[4,count//4,replications,4]),1),[1,2,0])
        for index,rho in enumerate((0.,.5,.9)):
            blocks=reference(tf.constant(rho,F64),key(seed,count+index))
            flags=temporal_block_conflicts(blocks,practical_width=.1)
            rate=float(tf.reduce_mean(tf.cast(flags,F64)).numpy())
            legacy=tf.reduce_any((tf.reduce_min(blocks,2)<.65)&(tf.reduce_max(blocks,2)>.75),1)
            # Wilson 95% interval, descriptive finite-replication uncertainty.
            n=replications;z=1.959963984540054;den=1+z*z/n
            center=(rate+z*z/(2*n))/den
            margin=z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/den
            rows.append({'kind':'stationary_binary_refresh','decisions_per_chain':count,'persistence':rho,
                'replications':n,'temporal_flag_rate':rate,'wilson95':[center-margin,center+margin],
                'legacy_flag_rate':float(tf.reduce_mean(tf.cast(legacy,F64)).numpy()),
                'passed':center+margin<=.05})
        # Strong coherent drift of acceptance probabilities with bounded noise.
        noise=tf.random.stateless_uniform([replications,4,4],key(seed,count+41),minval=-.02,maxval=.02,dtype=F64)
        blocks=tf.constant([.2,.4,.8,.95],F64)[None,None,:]+noise
        rate=float(tf.reduce_mean(tf.cast(temporal_block_conflicts(blocks,practical_width=.1),F64)).numpy())
        rows.append({'kind':'strong_common_shift','decisions_per_chain':count,'replications':replications,
            'block_expectations':[.2,.4,.8,.95],'bounded_noise':.02,'detection_rate':rate,'passed':rate>=.8})
    write_json(Path(output)/'temporal-calibration.json',{'rows':rows,'seed':seed,
        'scope':'reference diagnostic calibration; not actual HMC error-rate proof',
        'cpu_only_gpu_intentionally_hidden':True,'paired_contrast_familywise_working_level':.01})
    return phase_report(output,phase='temporal',passed=all(row['passed'] for row in rows),
        reason='temporal_calibration_passed' if all(row['passed'] for row in rows) else 'temporal_calibration_failed')


class LaplaceMixtureProposal:
    """Normalized local Laplace components plus declared full-support Student t."""
    def __init__(self, target, centers, defensive_weight=0.1):
        if not 0 < defensive_weight < 1:
            raise ValueError('defensive mixture weight must be strictly between zero and one')
        if int(tf.shape(centers)[0]) == 0:
            raise ValueError('no discovered modes for the local mixture')
        h = -target.hessian(centers)
        h = (h+tf.linalg.matrix_transpose(h))/2
        eigen = tf.linalg.eigvalsh(h)
        if not bool(tf.reduce_all(tf.math.is_finite(eigen) & (eigen > 0)).numpy()):
            raise ValueError('Laplace component requires finite positive curvature')
        covariance = tf.linalg.inv(h)
        chol = tf.linalg.cholesky(covariance)
        logits = target.log_prob(centers)-.5*tf.linalg.logdet(h)
        weights = tf.nn.softmax(logits)
        self.weights = tf.concat(((1-defensive_weight)*weights, [tf.constant(defensive_weight, F64)]), 0)
        components = [tfp.distributions.MultivariateNormalTriL(centers[i], chol[i])
                      for i in range(int(centers.shape[0]))]
        components.append(BroadStudentProposal(target.parameter_dim).distribution)
        self.distribution = tfp.distributions.Mixture(
            tfp.distributions.Categorical(probs=self.weights), components)
        self.centers, self.covariance = centers, covariance
        self.log_prob = tf.function(self.distribution.log_prob,
            input_signature=[tf.TensorSpec([None, target.parameter_dim], F64)], jit_compile=True, autograph=False)

    def sample(self, count, seed):
        return self.distribution.sample(count, seed=seed)


class FrozenMapTarget:
    """Exact change of coordinates; no learned approximation to the target."""
    def __init__(self,target,flow):
        self.parameter_dim=target.parameter_dim
        self.name=target.name
        self.flow=flow
        def log_prob(z):
            x,ld=flow.forward_and_logdet(z)
            return target.log_prob(x)+ld
        self.log_prob=tf.function(log_prob,input_signature=[tf.TensorSpec([None,self.parameter_dim],F64)],
                                  jit_compile=True,autograph=False)


class LatentDefensiveProposal:
    """Normalized 90% Gaussian / 10% inherited broad t5 proposal in z."""
    def __init__(self,dimension):
        self.distribution=tfp.distributions.Mixture(
            tfp.distributions.Categorical(probs=tf.constant([.9,.1],F64)),
            [tfp.distributions.MultivariateNormalDiag(tf.zeros([dimension],F64),tf.ones([dimension],F64)),
             BroadStudentProposal(dimension).distribution])
        self.covariance=tf.eye(dimension,batch_shape=[1],dtype=F64)
        self.log_prob=tf.function(self.distribution.log_prob,
            input_signature=[tf.TensorSpec([None,dimension],F64)],jit_compile=True,autograph=False)

    def sample(self,count,seed):return self.distribution.sample(count,seed=seed)


def reference_samples(target, count, seed, cfg):
    if target.name != 'wiggle':
        return target.reference_sample(count, seed)
    grid, lw, _ = wiggle_quadrature(target, extent=32., resolution=768)
    indices = tf.random.stateless_categorical(lw[None], count, seed)[0]
    return tf.gather(grid, indices)


def prepare(target_name, output, cfg, seed, repair=False):
    target = WarmStartTarget(target_name, jit_compile=False)
    output = Path(output)
    starts = BroadStudentProposal(target.parameter_dim).sample(256 if repair else 64, key(seed, 30001))
    modes = discover_modes(target, starts, max_iterations=400 if repair else 200,
        score_tolerance=1e-8, merge_distance=1e-6, curvature_tolerance=1e-7)
    expected = {'gaussian': 1, 'mixture': 2, 'warped_mixture': 2, 'wiggle': 3}.get(target_name)
    mode_ok = modes['mode_count'] > 0 and (expected is None or modes['mode_count'] == expected)
    write_json(output/'discovery.json', modes)
    save_tensor(output/'discovered_modes.tensor', modes['representatives'])
    if not mode_ok:
        return phase_report(output, phase='prepare', passed=False, reason='incomplete_discovery', repair='independent_larger_search')
    proposal = LaplaceMixtureProposal(target, modes['representatives'])
    write_json(output/'proposal.json', {'centers': proposal.centers, 'covariance': proposal.covariance,
        'weights': proposal.weights, 'defensive_weight': .1,
        'weight_provenance': 'local Laplace masses, corrected by importance weights; not posterior masses'})
    normalizer = 0.
    if target_name == 'wiggle':
        estimates = [wiggle_quadrature(target, extent=e, resolution=r)[2] for e, r in ((24.,512),(32.,768))]
        if not bool((tf.abs(estimates[1]-estimates[0]) < .001).numpy()):
            return phase_report(output, phase='prepare', passed=False, reason='quadrature_reference_unstable', continuation_veto=True)
        normalizer = float(estimates[-1].numpy())
    for role, salt in (('training',30011), ('validation',30012)):
        save_tensor(output/(role+'.tensor'), reference_samples(target,8192,key(seed,salt),cfg))
    # This reference is development-only. No final confirmation is created here.
    write_json(output/'preparation.json', {'target':target_name,'target_signature':target.signature,
        'log_normalizer':normalizer, 'reference_role':'development_only', 'seed':seed,
        'mode_count_matches_reference':mode_ok,'proposal':'normalized_laplace_defensive_mixture'})
    return phase_report(output, phase='prepare', passed=True, reason='discovery_and_normalized_proposal_ready')


def observables(target, x):
    """Physical moments plus fixed CDF bins, including the separated-mode valley."""
    parts = [x, x*x, target.region_features(x)]
    if target.name in ('mixture','warped_mixture'):
        first = x[:, :1]
        cuts = tf.constant([[-7.,-5.,-3.,-2.,0.,2.,3.,5.,7.]], F64)
        parts += [tf.cast(first < cuts,F64), tf.cast(tf.abs(first)<2.,F64)]
        second = x[:,1:2] if target.name=='mixture' else x[:,1:2]-.1*(first*first-26.)
        parts += [tf.cast(second<tf.constant([[-2.,0.,2.]],F64),F64)]
    elif target.name=='funnel':
        parts += [tf.cast(x[:,:1]<tf.constant([[-2.,-1.,0.,1.,2.]],F64),F64)]
        standardized=x[:,1:]*tf.exp(-x[:,:1])
        parts += [tf.cast(tf.abs(standardized)<1.,F64), tf.cast(tf.abs(standardized)>2.,F64)]
    elif target.name=='gaussian':
        parts += [tf.cast(x[:, :1]<tf.constant([[-4.,1.,6.]],F64),F64),
                  tf.cast(x[:,1:2]<tf.constant([[-6.,-1.,4.]],F64),F64)]
    else:
        parts += [tf.cast(x[:, :1]<tf.constant([[-2.,0.,2.,4.,6.]],F64),F64),
                  tf.cast(x[:,1:2]<tf.constant([[-6.,-3.,0.,3.,6.]],F64),F64)]
    return tf.concat(parts, axis=1)


def cloud_assessment(target, clouds, log_weights, reference, *, iid=False):
    """Between-population uncertainty for teachers; iid uncertainty only for flow draws."""
    ref=observables(target,reference)
    expected=tf.reduce_mean(ref,axis=0)
    variance=tf.math.reduce_variance(ref,axis=0)
    ref_se=tf.sqrt(variance/tf.cast(tf.shape(ref)[0],F64))
    means=[];row_values=[]
    for x,lw in zip(clouds,log_weights,strict=True):
        values=observables(target,x);weights=tf.nn.softmax(lw)
        means.append(tf.reduce_sum(weights[:,None]*values,axis=0));row_values.append(values)
    per_population=tf.stack(means)
    estimate=tf.reduce_mean(per_population,axis=0)
    if iid:
        values=tf.concat(row_values,axis=0)
        se=tf.sqrt(tf.math.reduce_variance(values,axis=0)/tf.cast(tf.shape(values)[0],F64)+ref_se**2)
    else:
        if len(clouds)<2:raise ValueError('teacher assessment requires independent populations')
        se=tf.sqrt(tf.math.reduce_variance(per_population,axis=0)/tf.cast(len(clouds)-1,F64)+ref_se**2)
    d=target.parameter_dim
    tolerance=tf.concat((.1*tf.sqrt(tf.maximum(variance[:2*d],tf.constant(1e-20,F64))),
                         tf.fill(tf.shape(variance[2*d:]),tf.constant(.05,F64))),0)
    if target.name in ('mixture','warped_mixture'):
        # Regions(3), first-coordinate CDF(9), then valley.
        tolerance=tf.tensor_scatter_nd_update(tolerance,[[2*d+12]],[tf.constant(.01,F64)])
    error=tf.abs(estimate-expected)
    precision=se <= tolerance/2
    agreement=error <= tolerance+3*se
    finite=tf.reduce_all(tf.math.is_finite(per_population))
    return {'passed':bool((finite & tf.reduce_all(precision & agreement)).numpy()),
        'finite':bool(finite.numpy()),'estimate':estimate,'reference':expected,'error':error,
        'standard_error':se,'coarse_tolerance':tolerance,'precision_passed':bool(tf.reduce_all(precision).numpy()),
        'agreement_passed':bool(tf.reduce_all(agreement).numpy()),'per_population':per_population,
        'uncertainty_method':'iid_generated_draws' if iid else 'between_independent_population_means',
        'interpretation':'coarse warm-start screen; 3 SE rule is not simultaneous confidence coverage'}


def mutation_calibration(target,proposal,output,seed,steps=8,*,per_bridge=False):
    sampler=MALAProgram(lambda x,b:(1-b)*proposal.log_prob(x)+b*target.log_prob(x),target.parameter_dim,steps=steps)
    pilot=proposal.sample(256,key(seed,30101));ratio=target.log_prob(pilot)-proposal.log_prob(pilot)
    candidates=[]
    def assess(dt,provenance):
        rows=[]
        for index,beta in enumerate((0.,.5,1.)):
            indices=tf.random.stateless_categorical((beta*ratio)[None],256,key(seed,30110+index))[0]
            x=tf.gather(pilot,indices)
            y,_,acc,bad=sampler.run(x,tf.constant(beta,F64),tf.constant(dt,F64),key(seed,30120+index))
            rows.append({'beta':beta,'acceptance':float(acc.numpy()),'invalid_proposals':int(bad.numpy()),
                         'squared_displacement':float(tf.reduce_mean(tf.reduce_sum((y-x)**2,1)).numpy()),
                         'importance_ess':float((1./tf.reduce_sum(tf.nn.softmax(beta*ratio)**2)).numpy()),
                         'unique_pilot_rows':int(tf.size(tf.unique(indices).y).numpy()),
                         'mean_squared_displacement_by_coordinate':tf.reduce_mean((y-x)**2,0),
                         'population_scaled_mobility':scaled_displacement(
                             tf.reduce_mean((y-x)**2,0),tf.math.reduce_variance(x,axis=0))})
        candidates.append({'step_size':dt,'provenance':provenance,'acceptance':min(r['acceptance'] for r in rows),
            'invalid_proposals':sum(r['invalid_proposals'] for r in rows),
            'squared_displacement':min(r['squared_displacement'] for r in rows),'bridge_pilots':rows})
        write_json(Path(output)/'mutation-calibration.json',candidates)

    for dt in (.0001,.001,.01,.1):
        assess(dt,'inherited_initial_grid')
    try:
        selected=select_mala_candidate(candidates)
    except MALACalibrationFailure:
        # For x + dt*score + sqrt(2*dt)*noise, local Gaussian stiffness is
        # dt / variance. This nominates a search scale, never proves mixing.
        variance=float(tf.reduce_min(tf.linalg.eigvalsh(proposal.covariance)).numpy())
        if not math.isfinite(variance) or variance<=0:
            raise ValueError('Invalid covariance in proposal curvature calibration')
        for factor in (1.,.1,.01):
            dt=factor*variance
            if 0<dt<.0001:
                assess(dt,{'kind':'local_covariance_scaled_repair',
                           'minimum_component_variance':variance,'dimensionless_ratio':factor})
        selected=select_mala_candidate(candidates)
    if per_bridge:
        schedule=[]
        for index,beta in enumerate((0.,.5,1.)):
            choice=select_mala_candidate([{**row['bridge_pilots'][index],'step_size':row['step_size']} for row in candidates])
            schedule.append((beta,choice['step_size']))
        write_json(Path(output)/'mutation-schedule.json',{'schedule':schedule,
            'source':'independent_pilot_frozen_before_teacher','global_pilot_step':selected['step_size'],
            'coverage_established':False})
        return tuple(schedule)
    return selected['step_size']


def gabrie_measure_summary(target, rows):
    """Record the realized finite-time training measure; explanatory only."""
    rows = tf.convert_to_tensor(rows, F64)
    regions = tf.reduce_mean(target.region_features(rows), axis=0)
    result = {
        'rows': tf.shape(rows)[0],
        'finite': tf.reduce_all(tf.math.is_finite(rows)),
        'region_frequency': regions,
        'target_log_prob_mean': tf.reduce_mean(target.log_prob(rows)),
        'role': 'explanatory_finite_markov_measure_not_equilibrium_proof',
    }
    if target.name in ('mixture', 'warped_mixture'):
        result.update(
            hard_mode_right=tf.reduce_mean(tf.cast(rows[:, 0] > 0., F64)),
            valley_frequency=tf.reduce_mean(tf.cast(tf.abs(rows[:, 0]) < 2., F64)),
            declared_mixture_mass=tf.constant([1/3, 2/3], F64),
        )
    return result


def teacher(target_name,prepared,output,cfg,seed,repair=0):
    target=WarmStartTarget(target_name);output=Path(output);prepared=Path(prepared)
    gpu_preflight(target,output)
    flow_path=cfg.get('closure',{}).get('teacher_transport')
    flow=load_flow(flow_path,target) if flow_path else None
    sampling_target=FrozenMapTarget(target,flow) if flow else target
    proposal=LatentDefensiveProposal(target.parameter_dim) if flow else LaplaceMixtureProposal(target,read_tensor(prepared/'discovered_modes.tensor'))
    try:
        schedule=mutation_calibration(sampling_target,proposal,output,seed,per_bridge=True)
        dt=schedule[0][1]
    except MALACalibrationFailure as exc:
        return phase_report(output,phase='teacher',passed=False,reason='mutation_calibration_failed',
            failure_class='kernel_calibration',detail=str(exc),repair='teacher_calibration')
    settings=teacher_repair_settings(repair)
    particles=settings['particles']
    sampler=AnnealedSMC(sampling_target,proposal,SMCConfig(particles,settings['mutation_steps'],128,.8,.5,dt,False,True,schedule))
    write_json(output/'effective-teacher.json',{**settings,'step_size_schedule':schedule,
        'coordinates':'frozen_map_latent' if flow else 'physical','transport':flow_path,
        'transport_hash':read_json(flow_path)['transport_hash'] if flow_path else None,
        'proposal':'latent_defensive_gaussian_student' if flow else 'laplace_defensive_student'})
    clouds=[];weights=[];reports=[]
    for replication in range(3):
        result=sampler.run(key(seed,30200+replication),callback=lambda row:write_json(output/'progress.json',row))
        reports.append({k:v for k,v in result.items() if k not in ('particles','log_weights','roots')})
        for row in reports[-1]['stages']:
            row['population_scaled_mobility']=scaled_displacement(
                row['mutation_mean_squared_displacement'],row['mutation_population_variance'])
        write_json(output/f'teacher-{replication}.json',reports[-1])
        if not result['complete'] or any(r['invalid_proposals'] for r in result['stages']):
            return phase_report(output,phase='teacher',passed=False,reason='invalid_or_incomplete_teacher',repair='teacher_calibration')
        if flow:
            save_tensor(output/f'teacher-{replication}-latent.tensor',result['particles'])
            result['particles']=flow.forward_batch(result['particles'])
        clouds.append(result['particles']);weights.append(result['log_weights'])
        for name in ('particles','log_weights','roots'):
            save_tensor(output/f'teacher-{replication}-{name}.tensor',tf.cast(result[name],F64))
    reference=read_tensor(prepared/'validation.tensor')
    assessment=cloud_assessment(target,clouds,weights,reference)
    write_json(output/'teacher-assessment.json',assessment)
    # Each independent population contributes equal total mass.
    save_tensor(output/'teacher-particles.tensor',tf.concat(clouds,0))
    save_tensor(output/'teacher-log-weights.tensor',tf.concat([tf.nn.log_softmax(w)-math.log(3.) for w in weights],0))
    return phase_report(output,phase='teacher',passed=assessment['passed'],reason='teacher_screen_passed' if assessment['passed'] else 'teacher_accuracy_or_precision',
        particles_per_population=particles,replications=3,mala_step_size=dt,
        mutation_steps=settings['mutation_steps'],repair_axis=settings['repair_axis'],
        repair='increase_mutation' if repair<2 else 'increase_particles',
        coordinates='frozen_map_latent' if flow else 'physical')


def load_flow(path,target,*,trainable=False):
    payload=read_json(path)
    frozen=load_frozen_neutra_artifact(payload,expected_target_signature=target.signature).transport
    if not trainable:return frozen
    from bayesfilter.inference.neutra_transport import NeuTraTransport
    flow=NeuTraTransport(frozen.config)
    flow.restore_parameters(payload['parameters'])
    return flow


def flow_assessment(flow,target,reference,training,normalizer,seed):
    densities=getattr(flow,'_closure_densities',None)
    if densities is None:
        densities=tf.function(lambda x:(target.log_prob_kernel(x)-normalizer,flow.log_prob(x)),
            input_signature=[tf.TensorSpec([None,target.parameter_dim],F64)],jit_compile=True,autograph=False)
        flow._closure_densities=densities
    p,q=densities(reference)
    mean=tf.reduce_mean(training,0);centered=training-mean
    covariance=tf.matmul(centered,centered,transpose_a=True)/tf.cast(tf.shape(training)[0],F64)
    gaussian=tfp.distributions.MultivariateNormalTriL(mean,tf.linalg.cholesky(covariance))
    baseline=gaussian.log_prob(reference)
    gain=q-baseline;gain_mean=tf.reduce_mean(gain)
    gain_se=tf.math.reduce_std(gain)/tf.sqrt(tf.cast(tf.shape(gain)[0],F64))
    gaussian_fkl=tf.reduce_mean(p-baseline)
    if target.name=='gaussian':
        learned=bool((gain_mean >= -.02-3*gain_se).numpy())
    else:
        learned=bool((gain_mean > tf.maximum(.05*gaussian_fkl,3*gain_se)).numpy())
    z=tf.random.stateless_normal([8192,target.parameter_dim],key(seed,30301),dtype=F64)
    samples,_=flow.forward_and_logdet(z)
    screen=cloud_assessment(target,[samples],[tf.zeros([8192],F64)],reference,iid=True)
    finite=bool(tf.reduce_all(tf.math.is_finite(q)).numpy()) and screen['finite']
    return {'passed':finite and learned and screen['passed'],'finite':finite,'shape':screen,
        'nonlinear_learning_passed':learned,'heldout_forward_kl':tf.reduce_mean(p-q),
        'gaussian_forward_kl':gaussian_fkl,'gain_over_gaussian':gain_mean,'gain_standard_error':gain_se,
        'data_role':'development; never final confirmation'}


def fit(target_name,prepared,teacher_path,output,cfg,seed,arm='smc',repair=0):
    target=WarmStartTarget(target_name);output=Path(output);prepared=Path(prepared)
    gpu_preflight(target,output)
    normalizer=read_json(prepared/'preparation.json')['log_normalizer']
    reference=read_tensor(prepared/'validation.tensor');baseline=read_tensor(prepared/'training.tensor')
    reps=read_tensor(prepared/'discovered_modes.tensor')
    teacher_path=Path(teacher_path)
    parent=Path(cfg['warm_parent']) if cfg.get('warm_parent') else None
    teacher_identity=validate_continuation_teacher(parent,teacher_path) if parent and arm!='gabrie' else None
    if arm=='gabrie':
        proposal=LaplaceMixtureProposal(target,reps)
        try:
            dt=mutation_calibration(target,proposal,output,seed)
        except MALACalibrationFailure as exc:
            return phase_report(output,phase='fit',passed=False,reason='mutation_calibration_failed',
                failure_class='kernel_calibration',detail=str(exc),repair='teacher_calibration',arm=arm)
        pool=tf.repeat(reps[:1],cfg['walkers'],axis=0);lw=tf.zeros([cfg['walkers']],F64)
    else:
        pool=read_tensor(teacher_path/'teacher-particles.tensor');lw=read_tensor(teacher_path/'teacher-log-weights.tensor')
        dt=read_json(teacher_path/'phase.json')['mala_step_size']
    parent_total=None
    if parent:
        parent_total=max(int(p.name.split('-')[1]) for p in parent.glob('warm-*-checkpoint.json'))
        parent_checkpoint=read_json(parent/f'warm-{parent_total}-checkpoint.json')
        parent_lr=parent_checkpoint['config']['learning_rate']
        parent_width=parent_checkpoint['transport_config']['hidden_layers'][0]
    widths=[parent_width] if parent else cfg['widths_funnel'] if target_name=='funnel' else cfg['widths_2d']
    learning_rates=[parent_lr] if parent else cfg['learning_rates']
    records=[];viable=[];continuations=[];stopped=[]
    for width in widths:
        for lr in learning_rates:
            candidate=output/f'w{width}-lr{lr:g}';candidate.mkdir(exist_ok=False)
            flow=load_flow(parent/f'warm-{parent_total}-frozen.json',target,trainable=True) if parent else make_transport(
                target,width,(seed+repair*1000,91),
                variance_scale=cfg.get('closure',{}).get('iaf_variance_scale'))
            centers=tf.gather(reps,tf.range(cfg['walkers'])%tf.shape(reps)[0])
            current=centers+.2*tf.random.stateless_normal(tf.shape(centers),key(seed,30310),dtype=F64)
            initial_pool=current if arm=='gabrie' else pool
            initial_weights=tf.zeros([cfg['walkers']],F64) if arm=='gabrie' else lw
            if parent:
                current=read_tensor(parent/f'walkers-{parent_total}.tensor')
                clip=parent_checkpoint['config']['gradient_clip_norm']
                measurement=read_json(parent/'gradient-calibration.json')
            else:
                clip,measurement=gradient_calibration(flow,initial_pool,cfg,seed+31000,log_weights=initial_weights,learning_rate=lr)
            write_json(candidate/'gradient-calibration.json',measurement)
            escape=int(cfg.get('closure',{}).get('escape_rkl_updates',0))
            if escape:
                escape_clip,escape_measure=gradient_calibration(flow,baseline,cfg,seed+35000,
                    target=target,kind='rkl',learning_rate=lr)
                escape_trainer=TrainingBlock(flow,target,batch=cfg['batch_size'],learning_rate=lr,
                    clip=escape_clip,kind='rkl',walkers=cfg['walkers'],walk_steps=cfg['walker_steps'])
                escape_result=escape_trainer.run(key(seed,35001),tf.constant(escape),baseline,
                    tf.zeros([tf.shape(baseline)[0]],F64),current,tf.constant(dt,F64))
                escape_finite=bool(escape_result[-1].numpy())
                write_json(candidate/'escape.json',{'updates':int(escape_result[0].numpy()),
                    'finite':escape_finite,'calibration':escape_measure,
                    'gradient_norm_total':escape_result[3],'clipped_updates':escape_result[4],
                    'optimizer_reset_reason':'distinct RKL escape then new forward objective',
                    'checkpoint':escape_trainer.checkpoint() if escape_finite else None})
                if not escape_finite:
                    stopped.append({'candidate':str(candidate),'decision':'numerical_failure'});continue
            trainer=TrainingBlock(flow,target,batch=cfg['batch_size'],learning_rate=lr,clip=clip,
                kind='gabrie' if arm=='gabrie' else 'forward',walkers=cfg['walkers'],walk_steps=cfg['walker_steps'])
            reset=bool(escape or cfg.get('closure',{}).get('reset_forward_optimizer',False))
            if parent and not reset:trainer.restore(parent_checkpoint)
            total=parent_total or 0;history=[];previous=None;flat_count=0
            if parent:
                write_json(candidate/'restoration.json',{'parent':str(parent),
                    'parent_state_hash':parent_checkpoint['state_hash'],'lifetime_forward_updates':total,
                    'optimizer_step':int(trainer.trainer.step.numpy()),'optimizer_restored':not reset,
                    'optimizer_reset_reason':'RKL escape then forward objective' if escape else
                        'matched reset-only control' if reset else None,
                    'teacher_input_sha256':teacher_identity})
            previous_q=tf.stop_gradient(flow.log_prob(reference))
            rungs=cfg.get('closure',{}).get('fit_rungs',(256,1024,2048,4096,8192))
            if not rungs or not any(rung>total for rung in rungs):
                raise ValueError('Fit rungs contain no new updates beyond the restored lifetime')
            for rung in rungs:
                if rung<=total:continue
                result=trainer.run(key(seed,31000+total),tf.constant(rung-total),pool,lw,current,tf.constant(dt,F64))
                current=result[1];total+=int(result[0].numpy())
                if not bool(result[-1].numpy()) or int(result[7].numpy()):
                    history.append({'updates':total,'numerical_veto':True,
                        'gradient_norm_total':result[3],'clipped_updates':result[4],
                        'block_updates':result[0],'invalid_proposals':int(result[7].numpy()),
                        'finite':bool(result[-1].numpy())});break
                metrics=flow_assessment(flow,target,reference,baseline,normalizer,seed)
                current_q=tf.stop_gradient(flow.log_prob(reference));difference=current_q-previous_q
                paired={'mean_log_density_gain':tf.reduce_mean(difference),
                    'standard_error':tf.math.reduce_std(difference)/tf.sqrt(tf.cast(tf.size(difference),F64)),
                    'previous_updates':total-int(result[0].numpy()),'current_updates':total}
                previous_q=current_q
                checkpoint=trainer.checkpoint();write_json(candidate/f'warm-{total}-checkpoint.json',checkpoint)
                save_tensor(candidate/f'walkers-{total}.tensor',current)
                frozen=flow.frozen_payload(target_signature=target.signature,training_state_hash=checkpoint['state_hash'])
                write_json(candidate/f'warm-{total}-frozen.json',frozen)
                write_json(candidate/f'warm-{total}-assessment.json',metrics)
                row={'updates':total,'assessment':metrics,'invalid_proposals':int(result[7].numpy()),
                     'optimizer_step':int(trainer.trainer.step.numpy()),
                     'paired_progress':{k:float(v.numpy()) if hasattr(v,'numpy') else v for k,v in paired.items()},
                     'gradient_norm_total':result[3],'clipped_updates':result[4],
                     'block_updates':result[0],'mean_gradient_norm':result[3]/tf.cast(result[0],F64),
                     'clipped_fraction':result[4]/tf.cast(result[0],F64),
                     'global_acceptance':result[5]/max(1,int(result[0].numpy())),
                     'local_acceptance':result[6]/max(1,int(result[0].numpy()))}
                history.append(row);write_json(candidate/'history.json',history)
                if arm=='gabrie':
                    write_json(candidate/f'training-measure-{total}.json',trainer.measure_summary())
                score=float(metrics['heldout_forward_kl'].numpy())
                if metrics['passed']:
                    viable.append((score,candidate/f'warm-{total}-frozen.json',candidate/f'walkers-{total}.tensor',width,lr,metrics))
                    break
                # Diagnose a non-learning plateau before spending every update cap.
                gain=float(metrics['gain_over_gaussian'].numpy())
                uncertainty=float(metrics['gain_standard_error'].numpy())
                flat_count=flat_count+1 if previous is not None and abs(gain-previous)<=max(.01,3*uncertainty) else 0
                previous=gain
                if total>=cfg.get('closure',{}).get('plateau_min_updates',2048) and flat_count>=2 and not metrics['nonlinear_learning_passed']:
                    row['repair_trigger']='nonlinear_learning_plateau';break
            write_json(candidate/'history.json',history)
            decision=fit_repair_decision(history)
            stopped.append({'candidate':str(candidate),'decision':decision,'updates':total})
            if decision!='eligible':
                finite_terminal=bool(history and history[-1].get('assessment',{}).get('finite'))
                diagnostic={'updates':total,'decision':decision,'role':'explanatory_only',
                            'status':'executed' if finite_terminal else 'not_run_numerical_or_missing_evidence'}
                if finite_terminal:
                    terminal=load_flow(candidate/f'warm-{total}-frozen.json',target)
                    probe=PostTrainingProbe(terminal,target,1.,rows=1000,batch_size=1000,jit_compile=True)
                    report=probe(key(seed,31901))
                    write_json(candidate/'post-training-1000.json',report)
                    if not valid_terminal_probe(report):
                        decision='numerical_failure';stopped[-1]['decision']=decision
                        diagnostic.update(decision=decision,status='probe_numerical_veto')
                    diagnostic['frozen_checkpoint']=str(candidate/f'warm-{total}-frozen.json')
                write_json(candidate/'terminal-diagnostic.json',diagnostic)
            if decision=='continue_checkpoint':continuations.append(str(candidate))
            records.append({'width':width,'learning_rate':lr,'history':history})
            write_json(output/'calibration-progress.json',records)
    if not viable:
        all_numerical=bool(stopped) and all(row['decision']=='numerical_failure' for row in stopped)
        return phase_report(output,phase='fit',passed=False,
            reason='improving_fit_at_cap' if continuations else 'numerical_failure' if all_numerical else 'no_useful_warm_fit',
            repair='continue_checkpoint' if continuations else 'numerical_diagnosis' if all_numerical else
                'nonlinear_plateau_repair' if any(r['decision']=='nonlinear_plateau_repair' for r in stopped) else 'shape_repair',
            failure_class='candidate_numerical' if all_numerical else 'fit_screen',
            continuation_veto=all_numerical,arm=arm,
            continuation_candidates=continuations,candidate_decisions=stopped)
    score,path,walkers,width,lr,metrics=min(viable,key=lambda item:item[0])
    # Descriptive choice for further checking, not statistical method superiority.
    frozen=read_json(path);write_json(output/'selected-frozen.json',frozen)
    save_tensor(output/'selected-walkers.tensor',read_tensor(walkers))
    write_json(output/'selected-assessment.json',metrics)
    flow=load_flow(output/'selected-frozen.json',target)
    probe=PostTrainingProbe(flow,target,1.,rows=1000,batch_size=1000,jit_compile=True)
    report=probe(key(seed,31901));write_json(output/'post-training-1000.json',report)
    if not valid_terminal_probe(report):
        return phase_report(output,phase='fit',passed=False,reason='numerical_failure',
            failure_class='candidate_numerical',repair='numerical_diagnosis',arm=arm,
            continuation_veto=True,continuation_candidates=[],
            detail='selected map failed post-training probe numerical validity')
    return phase_report(output,phase='fit',passed=True,reason='warm_fit_screen_passed',arm=arm,
        selected_source=str(path),width=width,learning_rate=lr,mala_step_size=dt,
        selection_role='lowest development loss among screened candidates; not method ranking')


def refine(target_name,prepared,training,output,cfg,seed):
    target=WarmStartTarget(target_name);output=Path(output);prepared=Path(prepared);training=Path(training)
    flow=load_flow(training/'selected-frozen.json',target,trainable=True)
    reference=read_tensor(prepared/'validation.tensor');baseline=read_tensor(prepared/'training.tensor')
    normalizer=read_json(prepared/'preparation.json')['log_normalizer']
    lr=read_json(training/'phase.json')['learning_rate']
    clip,measurement=gradient_calibration(flow,baseline,cfg,seed+32000,target=target,kind='rkl',learning_rate=lr)
    write_json(output/'gradient-calibration.json',measurement)
    trainer=TrainingBlock(flow,target,batch=cfg['batch_size'],learning_rate=lr,clip=clip,kind='rkl',
        walkers=cfg['walkers'],walk_steps=cfg['walker_steps'])
    initial=read_tensor(training/'selected-walkers.tensor')
    incumbent=read_json(training/'selected-frozen.json');incumbent_metrics=read_json(training/'selected-assessment.json')
    write_json(output/'warm-frozen.json',incumbent)
    write_json(output/'warm-assessment.json',incumbent_metrics)
    warm_probe=PostTrainingProbe(flow,target,1.,rows=1000,batch_size=1000,jit_compile=True)(key(seed,32901))
    warm_probe['transport_hash']=incumbent['transport_hash']
    write_json(output/'warm-post-training-1000.json',warm_probe)
    candidates=[{'stage':'warm','filename':'warm-frozen.json',
        'transport_hash':incumbent['transport_hash'],'probe_file':'warm-post-training-1000.json',
        'assessment_file':'warm-assessment.json','eligible':bool(
            incumbent_metrics.get('passed',True) and valid_terminal_probe(warm_probe))}]
    selected='warm';history=[];total=0
    for rung in (256,1024,2048):
        result=trainer.run(key(seed,32000+total),tf.constant(rung-total),baseline,
                           tf.zeros([tf.shape(baseline)[0]],F64),initial,tf.constant(.01,F64))
        total+=int(result[0].numpy())
        if not bool(result[-1].numpy()):
            history.append({'updates':total,'reason':'nonfinite_rkl'});break
        metrics=flow_assessment(flow,target,reference,baseline,normalizer,seed)
        checkpoint=trainer.checkpoint();write_json(output/f'rkl-{total}-checkpoint.json',checkpoint)
        frozen=flow.frozen_payload(target_signature=target.signature,training_state_hash=checkpoint.get('state_hash'))
        write_json(output/f'rkl-{total}-frozen.json',frozen)
        write_json(output/f'rkl-{total}-assessment.json',metrics)
        report=PostTrainingProbe(flow,target,1.,rows=1000,batch_size=1000,jit_compile=True)(key(seed,32901))
        report['transport_hash']=frozen['transport_hash']
        write_json(output/f'rkl-{total}-post-training-1000.json',report)
        candidates.append({'stage':f'rkl-{total}','filename':f'rkl-{total}-frozen.json',
            'transport_hash':frozen['transport_hash'],'probe_file':f'rkl-{total}-post-training-1000.json',
            'assessment_file':f'rkl-{total}-assessment.json',
            'eligible':bool(metrics['passed'] and valid_terminal_probe(report))})
        history.append({'updates':total,'assessment':metrics,'block_updates':result[0],
            'gradient_norm_total':result[3],'clipped_updates':result[4],
            'clipped_fraction':result[4]/tf.cast(result[0],F64)})
        if not candidates[-1]['eligible']:
            history[-1]['decision']='reject_refinement_preserve_warm';break
        if float(metrics['heldout_forward_kl'].numpy())<float(incumbent_metrics['heldout_forward_kl']):
            incumbent,incumbent_metrics,selected=frozen,metrics,f'rkl-{total}'
    write_json(output/'history.json',history)
    write_json(output/'checkpoint-candidates.json',{'schema':'neutra.checkpoint_candidates.v1',
        'selection_order':'latest_eligible_RKL_then_earlier_RKL_then_warm',
        'candidates':list(reversed(candidates)),
        'legacy_selected_role':'forward_KL_nominee_only; actual consumer uses shortlist'})
    write_json(output/'selected-frozen.json',incumbent)
    write_json(output/'selected-assessment.json',incumbent_metrics)
    save_tensor(output/'selected-walkers.tensor',initial)
    selected_flow=load_flow(output/'selected-frozen.json',target)
    probe=PostTrainingProbe(selected_flow,target,1.,rows=1000,batch_size=1000,jit_compile=True)
    report=probe(key(seed,32901));write_json(output/'post-training-1000.json',report)
    if not valid_terminal_probe(report):
        return phase_report(output,phase='refine',passed=False,reason='numerical_failure',
            failure_class='candidate_numerical',repair='numerical_diagnosis',
            continuation_veto=True,detail='refined map failed post-training probe numerical validity')
    return phase_report(output,phase='refine',passed=True,reason='eligible_checkpoint_preserved',selected_stage=selected,
        parent=str(training),learning_rate=lr,mala_step_size=read_json(training/'phase.json')['mala_step_size'])


def sampler_check(target_name,prepared,training,output,cfg,seed,repair=0):
    target=WarmStartTarget(target_name);prepared=Path(prepared);training=Path(training);output=Path(output)
    flow=load_flow(training/'selected-frozen.json',target)
    reps=read_tensor(prepared/'discovered_modes.tensor');n_modes=int(tf.shape(reps)[0]);walkers=cfg['walkers']
    steps=512*(repair+1);program=GabrieProgram(flow,target,walkers=walkers,steps=steps)
    dt=read_json(training/'phase.json')['mala_step_size']
    reference=read_tensor(prepared/'validation.tensor');groups=[];report=[]
    for group in range(n_modes):
        current=tf.repeat(reps[group:group+1],walkers,axis=0)+.2*tf.random.stateless_normal([walkers,target.parameter_dim],key(seed,33000+group),dtype=F64)
        current,warm,_,_,bad0=program.run(current,tf.constant(dt,F64),key(seed,33100+group))
        current,retained,ga,la,bad=program.run(current,tf.constant(dt,F64),key(seed,33200+group))
        save_tensor(output/f'warmup-group-{group}.tensor',tf.reshape(warm,[steps,walkers,target.parameter_dim]))
        trace=tf.reshape(retained,[steps,walkers,target.parameter_dim]);save_tensor(output/f'retained-group-{group}.tensor',trace)
        save_tensor(output/f'last-walkers-group-{group}.tensor',current)
        # Frozen chains are independent conditional on the fixed map; preserve each mean.
        clouds=tf.unstack(tf.transpose(trace,[1,0,2]),axis=0)
        assessment=cloud_assessment(target,clouds,[tf.zeros([steps],F64)]*walkers,reference)
        features=tf.stack([observables(target,trace[i]) for i in range(0,steps,max(1,steps//64))])
        rhat=tfp.mcmc.potential_scale_reduction(features,independent_chain_ndims=1,split_chains=True)
        constant=tf.math.reduce_variance(features,axis=(0,1))<tf.constant(1e-20,F64)
        finite_rhat=tf.where(constant,tf.ones_like(rhat),tf.where(tf.math.is_finite(rhat),rhat,
            tf.fill(tf.shape(rhat),tf.constant(math.inf,F64))))
        # Raw split R-hat is only a coarse training-sampler screen; final HMC uses modern diagnostics.
        passed=assessment['passed'] and int(bad.numpy())==0 and int(bad0.numpy())==0 and bool(tf.reduce_all(finite_rhat<1.1).numpy())
        row={'initial_mode_group':group,'assessment':assessment,'split_rhat':rhat,
             'passed':passed,'global_acceptance':ga,'local_acceptance':la,'invalid_proposals':bad+bad0}
        report.append(row);write_json(output/'sampler-assessments.json',report)
    return phase_report(output,phase='sampler',passed=all(r['passed'] for r in report),reason='frozen_gabrie_sampler_assessed',
        repair='extend_frozen_sampling_only',warmup_steps=steps,retained_steps=steps,walkers_per_group=walkers)


def final_reference(target_name,prepared,output,cfg,seed):
    target=WarmStartTarget(target_name,jit_compile=False);output=Path(output)
    save_tensor(output/'confirmation.tensor',reference_samples(target,32768,key(seed,34001),cfg))
    save_tensor(output/'discovered_modes.tensor',read_tensor(Path(prepared)/'discovered_modes.tensor'))
    write_json(output/'preparation.json',{'target_signature':target.signature,'role':'fresh_final_after_selection',
        'seed':[seed,34001],'generated_after_candidate_freeze':True})
    return phase_report(output,phase='reference',passed=True,reason='fresh_final_reference_created')


def qualify_selected(target_name,prepared,training,output,cfg,seed):
    from bayesfilter.testing.neutra_warm_start_qualification import qualify,qualify_shortlist
    if not read_json(Path(training)/'phase.json').get('passed'):
        return phase_report(output,phase='qualify',passed=False,reason='upstream_candidate_invalid',
            candidate=str(training),reference=str(prepared),heldout_consumed=False)
    if (Path(training)/'checkpoint-candidates.json').is_file():
        result=qualify_shortlist(target_name,seed,prepared,training,output,cfg)
    else:
        result=qualify(target_name,seed,prepared,training,output,cfg,
            frozen_filename='selected-frozen.json',discovered_starts=True)
    return phase_report(output,phase='qualify',passed=result['qualified'],reason=result.get('reason'),
        candidate=str(training),reference=str(prepared),heldout_consumed=result.get('heldout_consumed',False))
