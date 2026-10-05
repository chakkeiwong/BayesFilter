"""Diagnostic experiment support; never a tuning/admission authority.

Independent references may use AD; the candidate filter always uses its
analytical recursion. No reference gradient is supplied to the filter.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from dataclasses import replace

import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_batch_fused_tf import canonical_batch_fused_value_score
from bayesfilter.highdim.ledh_linear_parameter_adapter_tf import linear_gaussian_parameter_model
from bayesfilter.highdim.ledh_range_bearing_target_tf import make_range_bearing_model
from bayesfilter.highdim.sv_mixture_cut4 import ksc_1998_log_chi_square_mixture
from bayesfilter.testing.surrogate_hmc_ksc_consumer import ksc_model
from bayesfilter.testing.surrogate_hmc_lg1_tf import warm_start_controls

PHASES = ("lg1-t2", "lg3-t2", "lg3-t10", "ksc-t10", "ksc-t20", "m13-t1", "m13-t20")
PARTITIONS = {"calibration": 1, "validation": 2, "claim": 3, "engineering": 4}


def range_fixture(horizon, seed):
    from experiments.dpf_implementation.tf_tfp.fixtures.range_bearing_tf import (
        build_range_bearing_fixture_tf, _simulate_range_bearing, stable_digest)
    base = build_range_bearing_fixture_tf()
    states, observed = _simulate_range_bearing(A=base.A, Q=base.Q, R=base.R,
        m0=base.m0, P0=base.P0, horizon=horizon, seed=seed)
    return replace(base, states=states, observations=observed, fixture_generation_seed=seed,
        model_checksum=stable_digest(dict(base=base.model_checksum, horizon=horizon, seed=seed)),
        observation_checksum=stable_digest(serial(observed)))


def serial(value):
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    return value.numpy().tolist() if tf.is_tensor(value) else value


def json_safe(value):
    """Preserve invalid diagnostics as explicit strings, never as JSON NaN."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(serial(value)), indent=2, allow_nan=False))
    temporary.replace(path)


def specification(name, dtype=tf.float32):
    kind, horizon = name.split("-t")
    t = int(horizon)
    if kind.startswith("lg"):
        d = int(kind[2:])
        a = tf.eye(d, dtype=dtype) if d == 1 else tf.constant(
            [[.6, .1, 0.], [.1, .6, .1], [0., .1, .6]], dtype)
        q, h, r = tf.eye(d, dtype=dtype), tf.eye(d, dtype=dtype), .5*tf.eye(d, dtype=dtype)
        points = [[.4], [.8], [.9]] if d == 1 else [[1., 0.], [.8, -.5], [1.2, .3]]
        model = linear_gaussian_parameter_model(a, q, h, r, infer_process_scale=d > 1)
        return dict(name=name, kind=kind, t=t, d=d, m=d, p=len(points[0]),
                    points=points, truth=points[1 if d == 1 else 0], model=model, a=a, q=q, h=h, r=r)
    if kind == "ksc":
        return dict(name=name, kind=kind, t=t, d=1, m=1, p=2,
                    points=[[.5, .5], [.8, .3], [.3, .8]], truth=[.5, .5], model=ksc_model(dtype))
    if kind == "m13":
        fixture = range_fixture(t, 1704)
        cast = lambda x: tf.cast(x, dtype)
        return dict(name=name, kind=kind, t=t, d=4, m=2, p=2,
                    points=[[0., 0.], [-.73272085, -.01465508]], truth=[0., 0.], fixture=fixture,
                    model=make_range_bearing_model(cast(fixture.A), cast(fixture.Q), cast(fixture.R), dtype=dtype))
    raise ValueError(name)


def controls(index):
    settings = warm_start_controls()
    if index == 1:
        # Distinct complete warm-start hypothesis, from the inspected M13 scope.
        settings.update(reset_ridge=1e-8, correction_lm_damping=1e-6,
                        correction_lm_scale_floor=1e-9, correction_trust_radius=100.,
                        pairwise_rms_cap=2000., coordinate_cap=16000.)
    elif index != 0:
        raise ValueError(index)
    return settings


def dataset(spec, partition):
    """Small CPU-only fixed-input generation exception; no transport training."""
    role = PARTITIONS[partition]
    with tf.device("/CPU:0"):
        if spec["kind"] == "m13":
            seed = 1704 if partition == "claim" else 20261005 + role
            return tf.cast(range_fixture(spec["t"], seed).observations, tf.float64)
        truth = tf.constant(spec["truth"], tf.float64)
        model = specification(spec["name"], tf.float64)["model"]
        mean, covariance = model.initial_distribution_fn(truth)
        random = tf.random.stateless_normal([spec["t"]+1, spec["d"]+spec["m"]+1],
                                            [20261005, role*1000+spec["d"]*100+spec["t"]], dtype=tf.float64)
        x = mean + tf.linalg.matvec(tf.linalg.cholesky(covariance), random[0, :spec["d"]])
        values = []
        mixture = ksc_1998_log_chi_square_mixture() if spec["kind"] == "ksc" else None
        for i in range(spec["t"]):
            q = model.process_covariance if model.process_covariance_fn is None else model.process_covariance_fn(truth)
            x = model.transition_mean_fn(truth[None], x[None])[0] + tf.linalg.matvec(
                tf.linalg.cholesky(q), random[i+1, :spec["d"]])
            if mixture is not None:
                component = tf.random.stateless_categorical(tf.math.log(mixture.weights)[None], 1,
                    [20261005+role, i+100*spec["t"]])[0, 0]
                y = (x + 2*tf.math.log(truth[1]) + tf.gather(mixture.means, component)
                     + tf.sqrt(tf.gather(mixture.variances, component))*random[i+1, spec["d"]])
            else:
                y = tf.linalg.matvec(tf.cast(spec["h"], tf.float64), x) + tf.linalg.matvec(
                    tf.linalg.cholesky(tf.cast(spec["r"], tf.float64)), random[i+1, spec["d"]:2*spec["d"]])
            values.append(y)
        return tf.stack(values)


def designs(spec, n, partition, replicate, dtype=tf.float32):
    role = PARTITIONS[partition]
    seed = [20261005+role, 100000*spec["d"]+1000*spec["t"]+replicate]
    with tf.device("/CPU:0"):
        z = tf.random.stateless_normal([spec["t"]+2, n, spec["d"]], seed, dtype=tf.float64)
        base, noise, reset = z[0], z[1:-1], z[-1]
        reset -= tf.reduce_mean(reset, axis=0)
        reset = tf.transpose(tf.linalg.triangular_solve(
            tf.linalg.cholesky(tf.transpose(reset)@reset/n), tf.transpose(reset)))
        if spec["kind"] == "m13":
            fixture = spec["fixture"]
            base = tf.cast(fixture.m0, tf.float64) + base@tf.transpose(tf.linalg.cholesky(tf.cast(fixture.P0, tf.float64)))
        # All precision comparisons consume identical FP32-rounded designs.
        return tuple(tf.cast(tf.cast(v, tf.float32), dtype) for v in (base, noise, reset))


def compile_filter(spec, n, setting, policy, dtype=tf.float32):
    d, p, t, m = (spec[k] for k in ("d", "p", "t", "m"))
    covariance = (tf.cast(spec["fixture"].P0, dtype) if spec["kind"] == "m13" else tf.eye(d, dtype=dtype))
    initial_cov = tf.broadcast_to(covariance, [n, d, d])
    signature = [tf.TensorSpec([p], dtype), tf.TensorSpec([n, d], dtype),
                 tf.TensorSpec([t, n, d], dtype), tf.TensorSpec([n, d], dtype), tf.TensorSpec([t, m], dtype)]

    @tf.function(input_signature=signature, jit_compile=True, autograph=False)
    def evaluate(theta, base, noise, reset, observed):
        options = dict(setting, reset_design=reset, return_filter_moments=True)
        if policy is not None:  # None exercises the unchanged public default.
            options["importance_weight_policy"] = policy
        value, score, status = canonical_batch_fused_value_score(
            spec["model"], theta[None], tf.eye(p, dtype=dtype)[None], base, initial_cov,
            noise, observed, **options)
        return dict(value=value[0], score=score[0], valid=status["program_valid"][0],
                    value_valid=status["value_valid"][0], force_valid=status["force_valid"][0],
                    ess=status["filtering_ess"][0])
    return evaluate


def kalman_reference(spec, observed):
    """Independent covariance Kalman likelihood; AD is reference-only."""
    dtype = tf.float64
    d, p = spec["d"], spec["p"]
    a, base_q, h, r = (tf.cast(spec[k], dtype) for k in ("a", "q", "h", "r"))
    eye = tf.eye(d*d, dtype=dtype)

    @tf.function(input_signature=[tf.TensorSpec([p], dtype)], jit_compile=True, autograph=False)
    def evaluate(theta):
        with tf.GradientTape() as tape:
            tape.watch(theta)
            f, q = theta[0]*a, base_q*(tf.exp(2*theta[1]) if p == 2 else 1.)
            kron = tf.reshape(tf.einsum("ij,kl->ikjl", f, f), [d*d, d*d])
            covariance = tf.reshape(tf.linalg.solve(eye-kron, tf.reshape(q, [-1, 1])), [d, d])
            mean, total = tf.zeros([d], dtype), tf.zeros([], dtype)
            for i in range(spec["t"]):
                mean = tf.linalg.matvec(f, mean)
                covariance = f@covariance@tf.transpose(f)+q
                s = h@covariance@tf.transpose(h)+r
                residual = observed[i]-tf.linalg.matvec(h, mean)
                chol = tf.linalg.cholesky(s)
                solved = tf.linalg.cholesky_solve(chol, residual[:, None])[:, 0]
                total += -.5*(d*tf.constant(math.log(2*math.pi), dtype)
                               + 2*tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
                               + tf.reduce_sum(residual*solved))
                gain = tf.transpose(tf.linalg.cholesky_solve(chol, h@covariance))
                mean += tf.linalg.matvec(gain, residual)
                covariance -= gain@s@tf.transpose(gain)
        return dict(value=total, score=tape.gradient(total, theta))
    return evaluate


def ksc_grid_reference(spec, observed, cells=512, extent=18.):
    """Independent 1D forward integration and analytical density derivatives."""
    dtype = tf.float64
    mixture = ksc_1998_log_chi_square_mixture()
    dx = tf.constant(2*extent/cells, dtype)
    x = -tf.constant(extent, dtype) + (tf.cast(tf.range(cells), dtype)+.5)*dx
    log2pi = tf.constant(math.log(2*math.pi), dtype)

    @tf.function(input_signature=[tf.TensorSpec([2], dtype)], jit_compile=True, autograph=False)
    def evaluate(theta):
        gamma, beta = theta[0], theta[1]
        variance = 1/(1-gamma*gamma)
        initial = tf.exp(-.5*x*x/variance)/tf.sqrt(2*tf.constant(math.pi, dtype)*variance)*dx
        dvariance = 2*gamma*variance*variance
        dinitial = initial*.5*(x*x/variance**2-1/variance)*dvariance
        initial_d = tf.stack([dinitial, tf.zeros_like(initial)], axis=1)
        residual = x[:, None]-gamma*x[None, :]
        transition = tf.exp(-.5*residual*residual-.5*log2pi)*dx
        dtransition = transition*residual*x[None, :]

        def body(i, weights, dweights, total, dtotal):
            pred = tf.cond(i == 0, lambda: weights, lambda: tf.linalg.matvec(transition, weights))
            dpred = tf.cond(i == 0, lambda: dweights, lambda: transition@dweights + tf.stack(
                [tf.linalg.matvec(dtransition, weights), tf.zeros_like(weights)], axis=1))
            r = observed[i, 0]-x[:, None]-2*tf.math.log(beta)-mixture.means[None]
            logs = tf.math.log(mixture.weights)[None]-.5*(log2pi+tf.math.log(mixture.variances)[None]+r*r/mixture.variances[None])
            logg = tf.reduce_logsumexp(logs, axis=1)
            g = tf.exp(logg)
            dg = g*tf.reduce_sum(tf.nn.softmax(logs, axis=1)*r/mixture.variances[None], axis=1)*2/beta
            product = pred*g
            dproduct = dpred*g[:, None]+tf.stack([tf.zeros_like(pred), pred*dg], axis=1)
            normalizer = tf.reduce_sum(product)
            dnormalizer = tf.reduce_sum(dproduct, axis=0)
            normalized = product/normalizer
            return (i+1, normalized, (dproduct-normalized[:, None]*dnormalizer)/normalizer,
                    total+tf.math.log(normalizer), dtotal+dnormalizer/normalizer)
        _, _, _, value, score = tf.while_loop(lambda i, *_: i < spec["t"], body,
            (tf.constant(0), initial, initial_d, tf.zeros([], dtype), tf.zeros([2], dtype)))
        return dict(value=value, score=score)
    return evaluate


def bootstrap_reference(spec, observed, n):
    """Cheap no-flow particle-filter adversary; value only, no score claim."""
    model = specification(spec["name"], tf.float64)["model"]
    dtype = tf.float64
    d, p, t = spec["d"], spec["p"], spec["t"]

    @tf.function(input_signature=[tf.TensorSpec([p], dtype), tf.TensorSpec([2], tf.int32)], jit_compile=True, autograph=False)
    def evaluate(theta, seed):
        if spec["kind"] == "m13":
            mean, covariance = tf.cast(spec["fixture"].m0, dtype), tf.cast(spec["fixture"].P0, dtype)
        else:
            mean, covariance = model.initial_distribution_fn(theta)
        x = mean+tf.random.stateless_normal([n, d], seed, dtype=dtype)@tf.transpose(tf.linalg.cholesky(covariance))
        rows = tf.broadcast_to(theta, [n, p])
        q = model.process_covariance if model.process_covariance_fn is None else model.process_covariance_fn(theta)
        cholq = tf.linalg.cholesky(q)
        def body(i, x, value):
            noise = tf.random.stateless_normal([n, d], seed+[0, 3*i+1], dtype=dtype)
            x = model.transition_mean_fn(rows, x)+noise@tf.transpose(cholq)
            if model.observation_log_density_fn is None:
                residual = observed[i]-model.observation_fn(x)
                rchol = tf.linalg.cholesky(model.observation_covariance)
                z = tf.transpose(tf.linalg.triangular_solve(rchol, tf.transpose(residual)))
                logg = -.5*tf.reduce_sum(z*z, axis=1)-tf.reduce_sum(tf.math.log(tf.linalg.diag_part(rchol)))-.5*spec["m"]*tf.constant(math.log(2*math.pi), dtype)
            else:
                logg = model.observation_log_density_fn(rows, x, observed[i])
            value += tf.reduce_logsumexp(logg)-tf.math.log(tf.cast(n, dtype))
            cdf = tf.cumsum(tf.nn.softmax(logg))
            cdf /= cdf[-1]
            u = (tf.cast(tf.range(n), dtype)+tf.random.stateless_uniform([], seed+[0, 3*i+2], dtype=dtype))/n
            indices = tf.searchsorted(cdf, u, side="right")
            return i+1, tf.gather(x, indices), value
        return tf.while_loop(lambda i, *_: i < t, body, (tf.constant(0), x, tf.zeros([], dtype)))[2]
    return evaluate


def m13_fisher_reference(spec, observed, n=32768, replicates=16):
    """Independent genealogical Fisher score; MC variance and particle bias remain."""
    dtype = tf.float64
    a,q0,r0,m0,p0 = (tf.cast(getattr(spec["fixture"],k),dtype) for k in ("A","Q","R","m0","P0"))
    pi = tf.constant(math.pi,dtype)
    @tf.function(input_signature=[tf.TensorSpec([2],dtype),tf.TensorSpec([2],tf.int32)],jit_compile=True,autograph=False)
    def evaluate(theta,seed):
        x = m0+tf.random.stateless_normal([replicates,n,4],seed,dtype=dtype)@tf.transpose(tf.linalg.cholesky(p0))
        qchol = tf.linalg.cholesky(q0)*tf.exp(theta[0])
        sigma = tf.sqrt(tf.linalg.diag_part(r0))*tf.exp(theta[1])
        def body(i,x,complete,total,score):
            z = tf.random.stateless_normal([replicates,n,4],seed+[0,3*i+1],dtype=dtype)
            x = x@tf.transpose(a)+z@tf.transpose(qchol)
            radial = tf.sqrt(tf.reduce_sum(x[...,:2]**2,axis=-1)+1e-12)
            angle = tf.atan2(x[...,1],x[...,0])
            delta = tf.math.floormod(observed[i,1]-angle+pi,2*pi)-pi
            angular = (delta[...,None]+tf.cast(tf.range(-2,3),dtype)*2*pi)/sigma[1]
            radial_z = (observed[i,0]-radial)/sigma[0]
            angular_logs = -.5*angular**2
            logg = -.5*radial_z**2+tf.reduce_logsumexp(angular_logs,axis=-1)-tf.math.log(2*pi)-tf.reduce_sum(tf.math.log(sigma))
            complete += tf.stack([tf.reduce_sum(z*z,axis=-1)-4,
                radial_z**2+tf.reduce_sum(tf.nn.softmax(angular_logs)*angular**2,axis=-1)-2],axis=-1)
            weights = tf.nn.softmax(logg,axis=1)
            score = tf.reduce_sum(weights[...,None]*complete,axis=1)
            total += tf.reduce_logsumexp(logg,axis=1)-tf.math.log(tf.cast(n,dtype))
            cdf = tf.cumsum(weights,axis=1);cdf /= cdf[:,-1:]
            shifts = tf.random.stateless_uniform([replicates,1],seed+[0,3*i+2],dtype=dtype)
            indices = tf.searchsorted(cdf,(tf.cast(tf.range(n),dtype)[None]+shifts)/n,side="right")
            return i+1,tf.gather(x,indices,batch_dims=1),tf.gather(complete,indices,batch_dims=1),total,score
        _,_,_,value,score = tf.while_loop(lambda i,*_:i<spec["t"],body,
            (tf.constant(0),x,tf.zeros([replicates,n,2],dtype),tf.zeros([replicates],dtype),tf.zeros([replicates,2],dtype)))
        scaled = tf.exp(value-tf.reduce_max(value));total=tf.reduce_sum(scaled)
        mean_value = tf.reduce_logsumexp(value)-tf.math.log(tf.cast(replicates,dtype))
        mean_score = tf.reduce_sum(scaled[:,None]*score,axis=0)/total
        leave_value = tf.math.log((total-scaled)/(replicates-1))+tf.reduce_max(value)
        leave_score = (tf.reduce_sum(scaled[:,None]*score,axis=0)-scaled[:,None]*score)/(total-scaled)[:,None]
        jack = tf.concat([leave_value[:,None],leave_score],axis=1)
        mcse = tf.sqrt((replicates-1)*tf.reduce_mean((jack-tf.reduce_mean(jack,axis=0))**2,axis=0))
        return dict(value=mean_value,score=mean_score,mcse=mcse,replicate_values=value,replicate_scores=score)
    return evaluate


def references(spec, observed, *, claim=False):
    points = [tf.constant(x,tf.float64) for x in spec["points"]]
    kind = spec["kind"]
    if kind.startswith("lg"):
        function=kalman_reference(spec,observed)
        return [serial(function(p)) for p in points],dict(kind="independent_Kalman_AD_reference_only",valid=True)
    if kind=="ksc":
        coarse=ksc_grid_reference(spec,observed,512,18.)
        values=[serial(coarse(p)) for p in points]
        if not claim:return values,dict(kind="1D_grid_calibration",valid=True)
        refinements=[]
        for cells,extent in ((1024,18.),(1024,24.)):
            function=ksc_grid_reference(spec,observed,cells,extent)
            refinements.append(dict(cells=cells,extent=extent,rows=[serial(function(p)) for p in points]))
        discrepancy=max(abs(values[i]["value"]-r["rows"][i]["value"]) for i in range(len(points)) for r in refinements)
        score_discrepancy=max(.1*abs(values[i]["score"][j]-r["rows"][i]["score"][j]) for i in range(len(points)) for j in range(2) for r in refinements)
        fd_errors=[]
        for point,row in zip(points,values):
            for j in range(2):
                e=tf.one_hot(j,2,dtype=tf.float64)*1e-5
                fd=float((coarse(point+e)["value"]-coarse(point-e)["value"])/2e-5)
                fd_errors.append(abs(fd-row["score"][j]))
        return refinements[-1]["rows"],dict(kind="refined_1D_forward_grid_analytical_score",
            valid=discrepancy<.001 and score_discrepancy<.001 and max(fd_errors)<1e-5,
            coarse=values,refinements=refinements,value_discrepancy=discrepancy,
            scaled_score_discrepancy=score_discrepancy,max_score_fd_error=max(fd_errors))
    if spec["t"]==1:
        from bayesfilter.testing.surrogate_hmc_m13_reference import quadrature
        fixture=replace(spec["fixture"],observations=observed);all_rows=[]
        for size in ((256,512) if claim else (256,)):
            function=quadrature(fixture,size);rows=[]
            for point in points:
                row=serial(function(point))
                row["score"]=[float((function(point+tf.one_hot(j,2,dtype=tf.float64)*1e-4)["value"]-
                    function(point-tf.one_hot(j,2,dtype=tf.float64)*1e-4)["value"])/2e-4) for j in range(2)]
                rows.append(row)
            all_rows.append(dict(intervals=size,rows=rows))
        error=0. if not claim else max(abs(all_rows[0]["rows"][i]["value"]-all_rows[1]["rows"][i]["value"]) for i in range(len(points)))
        score_error=0. if not claim else max(.1*abs(all_rows[0]["rows"][i]["score"][j]-all_rows[1]["rows"][i]["score"][j]) for i in range(len(points)) for j in range(2))
        return all_rows[-1]["rows"],dict(kind="independent_T1_position_quadrature",valid=error<.001 and score_error<.001,
            refinements=all_rows,max_value_refinement=error,max_scaled_score_refinement=score_error)
    all_rows=[]
    for count in ((32768,65536) if claim else (8192,)):
        function=m13_fisher_reference(spec,observed,count,16 if claim else 8)
        all_rows.append(dict(particles=count,rows=[serial(function(p,tf.constant([20261005,count+i],tf.int32))) for i,p in enumerate(points)]))
    return all_rows[-1]["rows"],dict(kind="replicated_bootstrap_Fisher_score_reference",valid=True,exact=False,
        refinements=all_rows,qualification="particle bias remains; interpret differences with MCSE and refinement")


def errors(result,reference):
    return dict(value=abs(result["value"]-reference["value"]),
        score=math.sqrt(sum((.1*(x-y))**2 for x,y in zip(result["score"],reference["score"]))/len(reference["score"])))


def paired_summary(rows,reference):
    """Post-run diagnostic statistics, never runtime/admission logic."""
    from scipy.stats import t as student_t
    alpha=.05/(15*2*3*2)
    valid=all(row[method]["valid"] for row in rows for method in ("ancestor","marginal_mixture"))
    summary=dict(n=len(rows),valid=valid,metrics={})
    if not valid:
        summary["verdict"]="INVALID_CANDIDATE_OUTPUT";return summary
    for metric in ("value","score"):
        baseline=[errors(row["ancestor"],reference)[metric] for row in rows]
        candidate=[errors(row["marginal_mixture"],reference)[metric] for row in rows]
        differences=tf.constant([a-b for a,b in zip(candidate,baseline)],tf.float64)
        count=len(rows);mean=float(tf.reduce_mean(differences))
        sd=float(tf.sqrt(tf.reduce_sum((differences-mean)**2)/(count-1)));se=sd/math.sqrt(count)
        critical=float(student_t.ppf(1-alpha,count-1));lower,upper=mean-critical*se,mean+critical*se
        indices=tf.random.stateless_uniform([50000,count],[20261005,count+(metric=="score")],maxval=count,dtype=tf.int32)
        bootstrap=tf.sort(tf.reduce_mean(tf.gather(differences,indices),axis=1))
        bl,bu=float(bootstrap[int(alpha*50000)]),float(bootstrap[math.ceil((1-alpha)*50000)-1])
        extreme=sd>0 and float(tf.reduce_max(tf.abs(differences-mean)))/sd>5
        ni=upper<.05 and bu<.05 and not extreme
        material_harm=lower>.05 and bl>.05
        status="NONINFERIOR" if ni else ("MATERIAL_DETERIORATION" if material_harm else "INCONCLUSIVE")
        summary["metrics"][metric]=dict(baseline_mean=sum(baseline)/count,candidate_mean=sum(candidate)/count,
            difference_mean=mean,mcse=se,simultaneous_interval=[lower,upper],bootstrap_interval=[bl,bu],
            outlier_sensitivity=extreme,margin=.05,verdict=status,significant_deterioration=lower>0 and bl>0,
            significant_improvement=upper<0 and bu<0)
    summary["verdict"]="NONINFERIOR" if all(m["verdict"]=="NONINFERIOR" for m in summary["metrics"].values()) else "NOT_ESTABLISHED"
    return summary
