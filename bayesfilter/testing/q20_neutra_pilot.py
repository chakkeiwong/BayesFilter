"""Bounded q20 training diagnostics; no posterior or default promotion.

The target is the shared four-parameter UKF approximate posterior. Independent
importance populations diagnose whether a cheap approximate teacher suffices.
TensorFlow owns numerical work; Python owns scheduling and evidence files.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import time

import tensorflow as tf

F64 = tf.float64


def native(value):
    if isinstance(value, dict):
        return {k: native(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [native(v) for v in value]
    if tf.is_tensor(value):
        return value.numpy().tolist()
    return value


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(native(value), indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def save_tensor(path, value):
    data = tf.io.serialize_tensor(value).numpy()
    Path(path).write_bytes(data)
    return hashlib.sha256(data).hexdigest()


class Q20Target:
    def __init__(self):
        from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge
        self.bridge = make_q20_tempered_bridge(20, jit_compile=True,
            principal_sqrt_backend='tensorflow_eigh_strict_factor_cached')
        self.parameter_dim = self.bridge.parameter_dim
        self.signature = self.bridge.target_signature
        if self.parameter_dim != 4:
            raise ValueError('q20 parameter dimension changed')

    def value_score(self, x):
        v, s, status = self.bridge.value_score_status(x, tf.constant(1., F64))
        return v, s, status['bridge_valid']

    def log_prob(self, x):
        # MALA needs the checked analytic score, not differentiation through UKF.
        @tf.custom_gradient
        def evaluated(y):
            value, score, valid = self.value_score(y)
            value = tf.where(valid, value, tf.constant(float('nan'), F64))
            return value, lambda adjoint: adjoint[:, None]*score
        return evaluated(x)


def summary(points, log_weights):
    weights = tf.nn.softmax(log_weights)
    mean = tf.reduce_sum(weights[:, None]*points, axis=0)
    centered = points-mean
    covariance = tf.einsum('n,ni,nj->ij', weights, centered, centered)
    return native(dict(ess=1./tf.reduce_sum(weights**2),
        max_weight=tf.reduce_max(weights), mean=mean, covariance=covariance,
        negative_mass=tf.reduce_sum(weights*tf.cast(points[:, 2] < 0., F64)),
        log_normalizer=tf.reduce_logsumexp(log_weights)-tf.math.log(tf.cast(tf.size(weights), F64))))


def teacher_screen(populations):
    """Replication-based exploratory screen; no iid ESS substitution for SE."""
    if len(populations) < 4 or len(populations) % 2:
        raise ValueError('equal training/validation replication groups required')
    mass = tf.constant([p['negative_mass'] for p in populations], F64)
    means = tf.constant([p['mean'] for p in populations], F64)/4.
    half = len(populations)//2
    def disagreement(x):
        a, b = x[:half], x[half:]
        ma, mb = tf.reduce_mean(a, 0), tf.reduce_mean(b, 0)
        variance = (tf.reduce_sum((a-ma)**2, 0)+tf.reduce_sum((b-mb)**2, 0))/(half*(half-1))
        return tf.abs(ma-mb), tf.sqrt(variance)
    dm, sm = disagreement(mass)
    dx, sx = disagreement(means)
    adequate = all(p['ess'] >= 64 and p['max_weight'] <= .1 and
                   0. < p['negative_mass'] < 1. for p in populations)
    stable = bool((dm <= 4*sm+.05).numpy()) and bool(tf.reduce_all(dx <= 4*sx+.25).numpy())
    return native(dict(passed=adequate and stable, weight_screen=adequate,
        replication_screen=stable, mass_difference=dm, mass_standard_error=sm,
        prior_scaled_mean_difference=dx, prior_scaled_mean_standard_error=sx,
        role='approximate_teacher_pilot_screen_not_global_posterior_validation'))


def price(output):
    target = Q20Target()
    bridge = target.bridge
    write(output/'target.json', bridge.signature_payload())
    batch = 32
    # Status key is defined by the component target, not inferred from finiteness.
    def checked(x):
        value, score, status = bridge.value_score_status(x, tf.constant(1., F64))
        delta = x-bridge.prior_center
        log_prior = -.5*tf.reduce_sum(delta**2/bridge.prior_variance, axis=1)
        log_prior -= .5*4*tf.math.log(tf.constant(2*math.pi*bridge.prior_variance, F64))
        return value, score, value-log_prior, status['bridge_valid']
    # The bridge already owns a stable compiled batch kernel. Keep this cheap
    # reporting subtraction on the host boundary rather than enclosing it in
    # another XLA wrapper before pricing the target itself.
    evaluate = checked
    def generate(index):
        with tf.device('/CPU:0'):
            return bridge.prior_center+4*tf.random.stateless_normal([512, 4], [61006, 100+index], dtype=F64)
    with ThreadPoolExecutor(max_workers=2) as pool:
        banks = list(pool.map(generate, range(8)))
    manifest = dict(target_signature=target.signature, batch_size=batch,
        source='batch_native_q20_T30_UKF_analytic_score', cpu_sample_workers=2,
        seed=61006, population_seeds=[[61006,100+i] for i in range(8)],
        teacher_kind='prior_importance', populations=[], artifact_sha256={},
        timings=[], scientific_promotion=False)
    for i, points in enumerate(banks):
        weights = []
        for j in range(0, 512, batch):
            start = time.monotonic()
            write(output/'current-call.json',dict(population=i,chunk=j,started_unix=time.time()))
            value, score, lw, valid = evaluate(points[j:j+batch])
            okay = bool(tf.reduce_all(valid).numpy()) and bool(tf.reduce_all(tf.math.is_finite(score)).numpy())
            if not okay or not bool(tf.reduce_all(tf.math.is_finite(value)).numpy()):
                write(output/'target-invalid.json', dict(population=i, chunk=j,
                    points=points[j:j+batch], valid=valid))
                raise ValueError('invalid prior point: cannot silently discard or redefine target')
            weights.append(lw)
            manifest['timings'].append(time.monotonic()-start)
            if j == 0:
                print(json.dumps(dict(first_chunk=i,seconds=manifest['timings'][-1])),flush=True)
        lw = tf.concat(weights, 0)
        label = ('training' if i < 4 else 'validation')+f'-{i%4}'
        for name, tensor in ((label+'-points.tensor', points), (label+'-logweights.tensor', lw)):
            manifest['artifact_sha256'][name] = save_tensor(output/name, tensor)
        manifest['populations'].append(summary(points, lw))
        write(output/'progress.json', manifest)
        print(json.dumps(dict(population=i,ess=manifest['populations'][-1]['ess'],
            negative_mass=manifest['populations'][-1]['negative_mass'])), flush=True)
    # Four-coordinate directional derivative checks at eight nearby points.
    with tf.device('/CPU:0'):
        base = bridge.prior_center+.1*tf.random.stateless_normal([8,4],[61006,991],dtype=F64)
        x = tf.repeat(base,4,axis=0)
        directions = tf.tile(tf.eye(4,dtype=F64),[8,1])
    _, score, _, _ = evaluate(x)
    errors = []
    for step in (1e-4, 5e-5):
        plus, _, _, pv = evaluate(x+step*directions)
        minus, _, _, mv = evaluate(x-step*directions)
        finite = bool(tf.reduce_all(pv & mv).numpy())
        analytic = tf.reduce_sum(score*directions,axis=1)
        error = float(tf.reduce_max(tf.abs((plus-minus)/(2*step)-analytic)/(1+tf.abs(analytic))).numpy())
        errors.append(dict(step=step, scaled_max_error=error, valid=finite))
    manifest.update(derivative_check=errors, teacher_screen=teacher_screen(manifest['populations']),
        compile_and_first_seconds=manifest['timings'][0],
        steady_batch_seconds=sum(manifest['timings'][1:])/len(manifest['timings'][1:]),
        dtype='float64', jit_compile=True, samplewise_loop=False,
        traces=bridge._compiled[batch].experimental_get_tracing_count())
    manifest['derivative_passed'] = all(e['valid'] and e['scaled_max_error'] <= 1e-5 for e in errors)
    manifest['status'] = ('teacher_available' if manifest['teacher_screen']['passed']
                          else 'teacher_repair_required') if manifest['derivative_passed'] else 'derivative_failure'
    write(output/'result.json', manifest)
    return manifest


def load_teacher(path, signature):
    path = Path(path)
    report = json.loads((path/'result.json').read_text())
    if report['target_signature'] != signature or not report['teacher_screen']['passed']:
        raise ValueError('teacher failed or belongs to another target')
    for name, digest in report['artifact_sha256'].items():
        if hashlib.sha256((path/name).read_bytes()).hexdigest() != digest:
            raise ValueError('teacher data hash changed: '+name)
    banks = {}
    for label in ('training','validation'):
        parts = []
        for i in range(4):
            with tf.device('/CPU:0'):
                points = tf.io.parse_tensor(tf.io.read_file(str(path/f'{label}-{i}-points.tensor')),out_type=F64)
                weights = tf.io.parse_tensor(tf.io.read_file(str(path/f'{label}-{i}-logweights.tensor')),out_type=F64)
                parts.append((points,tf.nn.log_softmax(weights)-math.log(4)))
        banks[label] = tuple(tf.concat([p[i] for p in parts],0) for i in range(2))
    return banks, report


def legacy_affine_chart(geometry):
    """Reconstruct the August SMC chart, including its density Jacobian.

    Source: physical_annealed_smc_canary_2026_08_10.py:116--132.
    The precision matrices belong to the two original mode representatives.
    """
    labels = ('plus', 'minus')
    means = tf.constant([geometry['representatives'][s]['position'] for s in labels], F64)
    precision = tf.constant([geometry['source_curvature'][s]['records'][-1]['precision']
                             for s in labels], F64)
    center = tf.reduce_mean(means, 0)
    delta = means-center
    covariance = tf.reduce_mean(tf.linalg.inv(precision), 0)+tf.einsum('ni,nj->ij',delta,delta)/2.
    eigenvalues, eigenvectors = tf.linalg.eigh(covariance)
    tf.debugging.assert_positive(eigenvalues)
    factor = tf.matmul(eigenvectors*tf.sqrt(eigenvalues)[None], eigenvectors, transpose_b=True)
    return center, factor, tf.reduce_sum(tf.math.log(eigenvalues))/2.


def reuse_smc(output, material_path, data_root):
    """Replay saved SMC values before using their particles as a warm start."""
    material_path, data_root = Path(material_path), Path(data_root)
    material = json.loads((material_path/'material.json').read_text())
    if material['status'] != 'SMC_WEIGHT_EVIDENCE_PASSED':
        raise ValueError('saved SMC material did not pass its original checks')
    geometry_path = data_root/'docs/plans/artifacts/ssl-lstm-q20-seed-b-neutra-mode-failure-root-cause-2026-08-10/r1/geometry.json'
    geometry_bytes = geometry_path.read_bytes()
    geometry_hash = hashlib.sha256(geometry_bytes).hexdigest()
    if geometry_hash != 'dc3dd7b84566867bc49c11ad16f50778d21457adbb398a17c2a75f3c3b461eeb':
        raise ValueError('original SMC proposal geometry changed')
    center, factor, log_jacobian = legacy_affine_chart(json.loads(geometry_bytes))
    target = Q20Target()
    if target.signature != '9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278':
        raise ValueError('original SMC target signature changed')
    write(output/'target.json',target.bridge.signature_payload())
    report = dict(target_signature=target.signature,teacher_kind='replayed_approximate_annealed_smc',
        source_material=str(material_path),source_material_sha256=hashlib.sha256((material_path/'material.json').read_bytes()).hexdigest(),
        geometry_sha256=geometry_hash,chart_log_jacobian=float(log_jacobian.numpy()),
        source_scope='two_known_proposal_supported_sign_regions_not_exhaustive_discovery',
        source_execution='historical_CPU_XLA; current_replay_GPU_XLA',
        partitions={'training':[0,1,2,3],'validation':[4,5,6,7]},
        populations=[],replay=[],artifact_sha256={},source_receipts={},scientific_promotion=False)
    children = {c['index']:c for c in material['child_runs'] if c['family']=='central'}
    for i in range(8):
        root = material_path/f'central-{i:02d}'
        canary_bytes = (root/'canary.json').read_bytes()
        if hashlib.sha256(canary_bytes).hexdigest() != children[i]['terminal_sha256']:
            raise ValueError('saved SMC child report changed')
        canary = json.loads(canary_bytes)
        if canary['status'] != 'SMC_CANARY_PASSED':
            raise ValueError('incomplete saved SMC child')
        stage_path = root/f"stage-{canary['stage_count']-1:02d}.json"
        stage = json.loads(stage_path.read_text())
        if stage['beta']!=1. or not stage['terminal_pre_resampling'] or stage['resampled']:
            raise ValueError('teacher requires beta-one weights before terminal resampling')
        values = {}
        for key,receipt in stage['receipts'].items():
            data = (data_root/receipt['path']).read_bytes()
            if hashlib.sha256(data).hexdigest()!=receipt['sha256']:
                raise ValueError('saved SMC tensor changed: '+receipt['path'])
            values[key] = tf.io.parse_tensor(data,out_type=tf.dtypes.as_dtype(receipt['dtype']))
            tf.ensure_shape(values[key],receipt['shape'])
            report['source_receipts'][receipt['path']] = receipt['sha256']
        points, weights = values['theta'], values['log_weights']
        tf.debugging.assert_near(tf.nn.softmax(weights),values['normalized_weights'],atol=1e-12,rtol=1e-12)
        chart_error = tf.reduce_max(tf.abs(center+tf.matmul(values['z'],factor,transpose_b=True)-points)/(1+tf.abs(points)))
        if float(chart_error.numpy())>1e-9:
            raise ValueError('saved theta and chart coordinates disagree')
        batches = []
        started = time.monotonic()
        for start in range(0,int(points.shape[0]),32):
            part = points[start:start+32]
            n = int(part.shape[0])
            if n<32:
                part = tf.concat([part,tf.repeat(part[:1],32-n,0)],0)
            value, score, valid = target.value_score(part)
            if not bool(tf.reduce_all(valid & tf.math.is_finite(value) & tf.reduce_all(tf.math.is_finite(score),1)).numpy()):
                raise ValueError('saved teacher has invalid current target values or scores')
            batches.append((value[:n],score[:n]))
        current = tf.concat([v[0] for v in batches],0)
        scores = tf.concat([v[1] for v in batches],0)
        expected = values['target_log_prob']-log_jacobian
        error = float(tf.reduce_max(tf.abs(current-expected)/(1+tf.abs(expected))).numpy())
        replay = dict(population=i,rows=int(points.shape[0]),scaled_value_error=error,
            chart_error=float(chart_error.numpy()),wall_seconds=time.monotonic()-started,
            roots=stage['unique_root_count'],source_hmc_sign_changes=canary['total_hmc_sign_changes'])
        report['replay'].append(replay)
        if error>1e-8:
            write(output/'replay-failure.json',report)
            raise ValueError('current target differs from saved SMC target after exact chart correction')
        label = ('training' if i<4 else 'validation')+f'-{i%4}'
        for suffix,tensor in (('points',points),('logweights',weights),('scores',scores)):
            name = label+'-'+suffix+'.tensor'
            report['artifact_sha256'][name] = save_tensor(output/name,tensor)
        report['populations'].append(summary(points,weights))
        write(output/'progress.json',report)
        print(json.dumps(replay),flush=True)
    report['teacher_screen'] = teacher_screen(report['populations'])
    report['status'] = 'teacher_available' if report['teacher_screen']['passed'] else 'teacher_repair_required'
    write(output/'result.json',report)
    return report


class AffineControl:
    """Constructed Gaussian diagnostic control; never trained or promoted."""
    parameter_dim = 4

    def __init__(self, center, covariance):
        self.center = tf.convert_to_tensor(center,F64)
        self.factor = tf.linalg.cholesky(tf.convert_to_tensor(covariance,F64))
        if not bool(tf.reduce_all(tf.math.is_finite(self.factor)).numpy()):
            raise ValueError('invalid empirical covariance for Gaussian control')

    def forward_and_logdet(self,z):
        x = self.center+tf.linalg.matmul(z,self.factor,transpose_b=True)
        return x,tf.fill([z.shape[0]],tf.reduce_sum(tf.math.log(tf.linalg.diag_part(self.factor))))

    def inverse_and_forward_logdet(self,x):
        z = tf.transpose(tf.linalg.triangular_solve(self.factor,tf.transpose(x-self.center)))
        return z,tf.fill([x.shape[0]],tf.reduce_sum(tf.math.log(tf.linalg.diag_part(self.factor))))

    def pullback_score_batch(self,z,score):
        return tf.linalg.matmul(score,self.factor)

    def log_abs_det_jacobian_score_batch(self,z):
        return tf.zeros_like(z)


class GeometryProbe:
    """Fixed 32-row graph; padding repeats only rows already counted once."""
    def __init__(self, flow, target):
        self.flow = flow
        def evaluate(z):
            x, ld = flow.forward_and_logdet(z)
            value, score, valid = target.value_score(x)
            residual = flow.pullback_score_batch(z,score)+flow.log_abs_det_jacobian_score_batch(z)+z
            ratio = value+ld+.5*tf.reduce_sum(z*z,axis=1)
            valid &= tf.reduce_all(tf.math.is_finite(residual),axis=1) & tf.math.is_finite(ratio)
            return x,residual,ratio,valid
        self.evaluate = tf.function(evaluate,input_signature=[tf.TensorSpec([32,4],F64)],
            jit_compile=True,autograph=False)

    def __call__(self,z,output,label):
        started = time.monotonic()
        pieces = []
        count = int(z.shape[0])
        for i in range(0,count,32):
            part = z[i:i+32]
            size = int(part.shape[0])
            if size<32:
                part = tf.concat([part,tf.repeat(part[:1],32-size,axis=0)],0)
            values = self.evaluate(part)
            pieces.append(tuple(v[:size] for v in values))
        physical,residual,ratio,valid = [tf.concat([p[i] for p in pieces],0) for i in range(4)]
        okay = bool(tf.reduce_all(valid).numpy())
        report = dict(rows=count,complete=True,finite=okay,valid_rows=int(tf.reduce_sum(tf.cast(valid,tf.int32)).numpy()),
            distribution='standard_normal_base_draws_not_posterior_draws',
            verification_scope='standard_1000_point' if count==1000 else 'short_diagnostic_only',
            batch_size=32,padded_rows=(32-count%32)%32,jit_compile=True,
            traces=self.evaluate.experimental_get_tracing_count(),
            geometry_role='explanatory_only_no_posterior_promotion',wall_seconds=time.monotonic()-started)
        tensors = dict(latent=z,physical=physical,residual=residual,log_ratio=ratio,valid=valid)
        report['artifact_sha256'] = {f'{label}-{k}.tensor':save_tensor(output/f'{label}-{k}.tensor',v)
            for k,v in tensors.items()}
        if okay:
            norm = tf.linalg.norm(residual,axis=1)
            order = tf.sort(norm)
            report.update(native(dict(score_residual_norm=dict(median=order[(count-1)//2],
                p95=order[int(.95*(count-1))],p99=order[int(.99*(count-1))],max=order[-1],
                mean=tf.reduce_mean(norm),rms=tf.sqrt(tf.reduce_mean(norm**2))),
                centered_log_density_rms=tf.math.reduce_std(ratio),
                negative_mass=tf.reduce_mean(tf.cast(physical[:,2]<0,F64)),
                importance_ess=1./tf.reduce_sum(tf.nn.softmax(ratio)**2),
                residual_exceeds_one_fraction=tf.reduce_mean(tf.cast(norm>1.,F64)),
                residual_exceeds_base_score_fraction=tf.reduce_mean(tf.cast(norm>tf.linalg.norm(z,axis=1),F64)))))
            conditional = {}
            for key,mask in (('negative_region',physical[:,2]<0),('positive_region',physical[:,2]>=0),
                             ('base_center',tf.linalg.norm(z,axis=1)<=3),('base_tail',tf.linalg.norm(z,axis=1)>3)):
                selected = tf.boolean_mask(norm,mask)
                n = int(tf.size(selected).numpy())
                conditional[key] = native(dict(rows=n,mean=tf.reduce_mean(selected),max=tf.reduce_max(selected))) if n else dict(rows=0,mean=None,max=None)
            report['conditional'] = conditional
        write(output/f'{label}-probe.json',report)
        return report


class TeacherGeometry:
    """Inverse-map independent teacher points without repeating UKF evaluation."""
    def __init__(self, flow, target, bank, teacher_path):
        self.points, self.weights = bank
        paths = [Path(teacher_path)/f'validation-{i}-scores.tensor' for i in range(4)]
        if all(p.exists() for p in paths):
            self.scores = tf.concat([tf.io.parse_tensor(p.read_bytes(),out_type=F64) for p in paths],0)
        else:
            chunks = []
            for start in range(0,int(self.points.shape[0]),32):
                x = self.points[start:start+32]
                n = int(x.shape[0])
                x = tf.concat([x,tf.repeat(x[:1],32-n,0)],0)
                _, score, valid = target.value_score(x)
                if not bool(tf.reduce_all(valid).numpy()):
                    raise ValueError('invalid validation teacher target')
                chunks.append(score[:n])
            self.scores = tf.concat(chunks,0)
        def evaluate(x,score):
            z, _ = flow.inverse_and_forward_logdet(x)
            residual = flow.pullback_score_batch(z,score)+flow.log_abs_det_jacobian_score_batch(z)+z
            return z, residual
        self.evaluate = tf.function(evaluate,input_signature=[tf.TensorSpec([32,4],F64)]*2,
                                    jit_compile=True,autograph=False)

    def __call__(self,output,label):
        pieces = []
        for start in range(0,int(self.points.shape[0]),32):
            x, score = self.points[start:start+32],self.scores[start:start+32]
            n = int(x.shape[0])
            x, score = [tf.concat([v,tf.repeat(v[:1],32-n,0)],0) for v in (x,score)]
            pieces.append(tuple(v[:n] for v in self.evaluate(x,score)))
        z, residual = [tf.concat([p[i] for p in pieces],0) for i in range(2)]
        if not bool(tf.reduce_all(tf.math.is_finite(z) & tf.math.is_finite(residual)).numpy()):
            raise ValueError('nonfinite inverse-mapped validation-teacher geometry')
        weights, norms = tf.nn.softmax(self.weights),tf.linalg.norm(residual,axis=1)
        mean = tf.reduce_sum(weights[:,None]*z,0)
        covariance = tf.einsum('n,ni,nj->ij',weights,z-mean,z-mean)
        report = native(dict(rows=int(z.shape[0]),finite=True,latent_mean=mean,latent_covariance=covariance,
            weighted_residual_mean=tf.reduce_sum(weights*norms),
            weighted_residual_rms=tf.sqrt(tf.reduce_sum(weights*norms**2)),max_residual=tf.reduce_max(norms),
            scope='approximate_independent_validation_teacher_not_exact_posterior',conditional={}))
        for name,mask in (('negative',self.points[:,2]<0),('positive',self.points[:,2]>=0)):
            w = weights*tf.cast(mask,F64)
            report['conditional'][name] = native(dict(mass=tf.reduce_sum(w),
                weighted_residual_mean=tf.reduce_sum(w*norms)/tf.reduce_sum(w)))
        report['artifact_sha256'] = {f'{label}-teacher-{k}.tensor':save_tensor(output/f'{label}-teacher-{k}.tensor',v)
            for k,v in (('latent',z),('residual',residual),('weights',weights))}
        write(output/f'{label}-teacher-geometry.json',report)
        return report


def fit(output, teacher_path, seed, *, extended=False, reverse_from=None):
    from bayesfilter.inference.neutra_transport import NeuTraTransport, NeuTraTransportConfig, NeuTraOptimizerConfig
    from bayesfilter.inference.neutra_joint_training import JointNeuTraTrainer
    from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
    from bayesfilter.testing.neutra_forward_reverse import train_block, teacher_loss
    target = Q20Target()
    banks, teacher = load_teacher(teacher_path,target.signature)
    training = summary(*banks['training'])
    validation = summary(*banks['validation'])
    with tf.device('/CPU:0'):
        latent = tf.random.stateless_normal([1000,4],[61006,2001],dtype=F64)
        short = latent[:128]
    cfg = NeuTraTransportConfig.huang_dsf(4,hidden_layers=(64,64),stages=3,
        mixture_components=4,seed=(seed,731))
    flow = NeuTraTransport(cfg)
    parent = None
    if reverse_from is not None:
        reverse_from = Path(reverse_from)
        parent = json.loads(reverse_from.read_text())
        parent_manifest = json.loads((reverse_from.parent/'manifest.json').read_text())
        digest = hashlib.sha256(reverse_from.read_bytes()).hexdigest()
        if parent_manifest.get('artifact_sha256',{}).get(reverse_from.name)!=digest:
            raise ValueError('reverse parent is not the preserved checked checkpoint')
        if parent.get('forward_weight')!=0. or parent.get('reverse_weight')!=1.:
            raise ValueError('reverse continuation requires a reverse objective parent')
    opt = (NeuTraOptimizerConfig(**parent['base']['optimizer_config']) if parent else
           NeuTraOptimizerConfig(64,'standard',.001,.9,.999,1e-8,None,True))
    trainer = JointNeuTraTrainer(flow,target.value_score,opt,target_signature=target.signature,
        teacher_id=hashlib.sha256((Path(teacher_path)/'result.json').read_bytes()).hexdigest(),
        forward_weight=0. if parent else 1.,reverse_weight=1. if parent else 0.)
    if parent:
        trainer.restore(parent)
    report = dict(status='running',target_signature=target.signature,seed=seed,
        teacher=str(teacher_path),teacher_kind=teacher['teacher_kind'],configuration=cfg.payload(),
        scientific_promotion=False,batch_size=64,dtype='float64_diagnostic_exception',
        jit_compile=True,samplewise_loop=False,checkpoints=[],controls={},
        forward_budget=0 if parent else (2048 if extended else 512),
        reverse_budget=128 if (extended or parent) else 32)
    controls = (('prior',target.bridge.prior_center,16*tf.eye(4,dtype=F64)),
        ('diagonal',training['mean'],tf.linalg.diag(tf.linalg.diag_part(tf.constant(training['covariance'],F64)))),
        ('full_covariance',training['mean'],training['covariance']))
    for name,center,cov in (() if parent else controls):
        control = AffineControl(center,cov)
        report['controls'][name] = GeometryProbe(control,target)(short,output,name)
        write(output/'progress.json',report)
    probe = GeometryProbe(flow,target)
    teacher_probe = TeacherGeometry(flow,target,banks['validation'],teacher_path)
    def assess(label,rows):
        cp = trainer.checkpoint()
        write(output/f'{label}-checkpoint.json',cp)
        frozen = flow.frozen_payload(target_signature=target.signature)
        write(output/f'{label}-frozen.json',frozen)
        restored = load_frozen_neutra_artifact(json.loads((output/f'{label}-frozen.json').read_text()),
            expected_target_signature=target.signature).transport
        @tf.function(input_signature=[tf.TensorSpec([32,4],F64)],jit_compile=True,autograph=False)
        def reconstruction(z):
            x,ld = restored.forward_and_logdet(z)
            back,ild = restored.inverse_and_forward_logdet(x)
            original,_ = flow.forward_and_logdet(z)
            return tf.reduce_max(tf.abs(back-z)/(1+tf.abs(z))),tf.reduce_max(tf.abs(ild-ld)),tf.reduce_max(tf.abs(x-original))
        inv,ld,reload = [float(x.numpy()) for x in reconstruction(short[:32])]
        if inv>1e-7 or ld>1e-7 or reload>1e-12:
            raise ValueError('frozen-map reconstruction failed')
        # Same parameters in the reusable graph; reload identity checked above.
        diagnostic = probe(rows,output,label)
        inverse_teacher = teacher_probe(output,label)
        ce = float(teacher_loss(flow,banks['validation']))
        mass = diagnostic.get('negative_mass')
        vm = validation['negative_mass']
        coverage = mass is not None and abs(mass-vm)<=.15 and mass>=.5*vm and (1-mass)>=.5*(1-vm)
        record = dict(label=label,probe=diagnostic,validation_cross_entropy=ce,
            validation_teacher_geometry=inverse_teacher,
            coverage_passed=coverage,teacher_negative_mass=vm,checkpoint_reloaded=True,
            inverse_error=inv,logdet_error=ld,forward_reload_error=reload)
        report['checkpoints'].append(record)
        write(output/'progress.json',report)
        print(json.dumps(dict(label=label,ce=ce,coverage=coverage,score=diagnostic.get('score_residual_norm'))),flush=True)
        return record
    if parent:
        counter = int(trainer.optimizer.iterations.numpy())
        report.update(parent_checkpoint=str(reverse_from),parent_sha256=digest,
                      starting_reverse_updates=counter,training_blocks=[])
        for offset in range(counter,counter+128,32):
            report['training_blocks'].append(train_block(trainer,banks['training'],32,[seed,4001],offset))
            write(output/f'reverse{offset+32}-checkpoint.json',trainer.checkpoint())
            write(output/'progress.json',report)
            print(json.dumps(dict(reverse_updates=offset+32,block=report['training_blocks'][-1])),flush=True)
        final = assess(f'reverse{counter+128}',latent)
        report['status'] = 'reverse_continuation_viable' if final['probe']['finite'] and final['coverage_passed'] else 'reverse_repair_required'
        write(output/'result.json',report)
        return report
    assess('initial',short)
    report['training_blocks'] = []
    rungs = ((0,128,'forward128'),(128,384,'forward512'))
    if extended:
        rungs += ((512,1536,'forward2048'),)
    for offset,count,label in rungs:
        block = train_block(trainer,banks['training'],count,[seed,3001],offset)
        report['training_blocks'].append(block)
        forward = assess(label,latent if label==rungs[-1][2] else short)
    if not forward['probe']['finite'] or not forward['coverage_passed']:
        report.update(status='forward_repair_required',reason='preserved forward endpoint not suitable for objective switch')
        write(output/'result.json',report)
        return report
    trainer = JointNeuTraTrainer(flow,target.value_score,
        NeuTraOptimizerConfig(64,'standard',.0001,.9,.999,1e-8,None,True),
        target_signature=target.signature,teacher_id=trainer.teacher_id,forward_weight=0.,reverse_weight=1.)
    report['training_blocks'].append(train_block(trainer,banks['training'],32,[seed,4001]))
    final = assess('reverse32',short if extended else latent)
    if extended and final['probe']['finite'] and final['coverage_passed']:
        report['training_blocks'].append(train_block(trainer,banks['training'],96,[seed,4001],32))
        final = assess('reverse128',latent)
    elif extended:
        # Preserve a complete 1000-point endpoint even if the early reverse
        # coverage guard vetoes more updates of this candidate.
        final = assess('reverse32_endpoint',latent)
    report['status'] = 'short_pair_viable' if final['probe']['finite'] and final['coverage_passed'] else 'reverse_repair_required'
    report['heuristic_dominance'] = 'diagnostic_controls_recorded_no_statistical_ranking'
    write(output/'result.json',report)
    return report
