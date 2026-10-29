"""Frozen CPU/XLA reference statistics for controlled-SMC paired qualification.

TensorFlow performs every numerical operation. This is an independent
reference-statistics exception to the default GPU execution target.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
import tensorflow as tf

D = tf.float64
B = 20000
Z95 = 1.6448536269514722

@tf.function(input_signature=[tf.TensorSpec([None, 2], D), tf.TensorSpec([2], tf.int32)],
             jit_compile=True, autograph=False)
def statistics_kernel(log_values, seed):
    n = tf.shape(log_values)[0]
    nf = tf.cast(n, D)
    likelihoods = tf.exp(log_values - tf.reduce_max(log_values))
    means = tf.reduce_mean(likelihoods, axis=0)
    centered = log_values - tf.reduce_mean(log_values, axis=0)
    squares = centered ** 2
    moments = tf.reduce_mean(squares, axis=0)
    estimate = tf.stack([tf.math.log(means[1] / means[0]),
                         tf.math.log(moments[1] / moments[0])])
    influence = tf.stack([likelihoods[:, 1]/means[1]-likelihoods[:, 0]/means[0],
                          squares[:, 1]/moments[1]-squares[:, 0]/moments[0]], axis=1)
    centered_if = influence - tf.reduce_mean(influence, axis=0)
    if_squared = centered_if ** 2
    se = tf.sqrt(tf.reduce_sum(if_squared, axis=0) / (nf-1.) / nf)
    if_share = tf.math.divide_no_nan(tf.reduce_max(if_squared, axis=0),
                                     tf.reduce_sum(if_squared, axis=0))
    effective = tf.reduce_sum(squares, axis=0)**2 / tf.reduce_sum(squares**2, axis=0)
    maximum_likelihood_share = tf.reduce_max(likelihoods, axis=0) / tf.reduce_sum(likelihoods, axis=0)
    output = tf.TensorArray(D, size=B//200, element_shape=[200, 2])

    def body(i, output, valid):
        indices = tf.random.stateless_uniform([200, n], seed+tf.stack([0, i]),
                                               minval=0, maxval=n, dtype=tf.int32)
        samples = tf.gather(log_values, indices)
        sampled_z = tf.gather(likelihoods, indices)
        mean_z = tf.reduce_mean(sampled_z, axis=1)
        variance = tf.math.reduce_variance(samples, axis=1)
        value = tf.stack([tf.math.log(mean_z[:, 1]/mean_z[:, 0]),
                          tf.math.log(variance[:, 1]/variance[:, 0])], axis=1)
        valid = valid & tf.reduce_all(tf.math.is_finite(value)) & tf.reduce_all(variance > 0.)
        return i+1, output.write(i, value), valid
    _, output, bootstrap_valid = tf.while_loop(
        lambda i, *_: i<B//200, body, (0, output, tf.constant(True)), parallel_iterations=1)
    ordered = tf.sort(tf.reshape(output.stack(), [B, 2]), axis=0)
    def quantile(p):
        location = p*(B-1)
        low, high = int(location), int(location)+1
        return ordered[low] + tf.constant(location-low, D)*(ordered[high]-ordered[low])
    boot_lo, boot_hi = quantile(.05), quantile(.95)
    normal_lo, normal_hi = estimate-Z95*se, estimate+Z95*se
    lower, upper = tf.minimum(boot_lo, normal_lo), tf.maximum(boot_hi, normal_hi)
    return dict(estimate=estimate, lower=lower, upper=upper,
                bootstrap_lower=boot_lo, bootstrap_upper=boot_hi,
                influence_lower=normal_lo, influence_upper=normal_hi, influence_se=se,
                source_port_sample_variance=moments*nf/(nf-1),
                source_port_mean_logZ=tf.reduce_mean(log_values, axis=0),
                residual_effective_count=effective, influence_maximum_share=if_share,
                maximum_likelihood_share=maximum_likelihood_share,
                bootstrap_valid=bootstrap_valid, source_port_positive_variance=moments>0.)

def infer(source, port, cell):
    if len(source) != len(port) or len(source) < 2:
        return dict(status="INSUFFICIENT_PAIRS", pairs=len(source))
    values = tf.constant(list(zip(source, port)), D)
    raw = statistics_kernel(values, tf.constant([261109, cell], tf.int32))
    fields = {k:v.numpy().tolist() for k,v in raw.items()}
    finite = bool(tf.reduce_all(tf.math.is_finite(raw["lower"])).numpy()) and bool(tf.reduce_all(tf.math.is_finite(raw["upper"])).numpy())
    common = len(source)>=300 and all(fields["source_port_positive_variance"]) and fields["bootstrap_valid"] and finite
    mean_adequate = common and max(fields["maximum_likelihood_share"]) <= .05 and fields["influence_maximum_share"][0] <= .1
    variance_adequate = common and min(fields["residual_effective_count"]) >= 30 and fields["influence_maximum_share"][1] <= .1
    margin = float(tf.math.log(tf.constant(1.01, D)).numpy())
    variance_limit = float(tf.math.log(tf.constant(1.05, D)).numpy())
    fields.update(pairs=len(source), likelihood_adequate=mean_adequate, variance_adequate=variance_adequate,
                  likelihood_pass=mean_adequate and fields["lower"][0]>=-margin and fields["upper"][0]<=margin,
                  variance_pass=variance_adequate and fields["upper"][1]<=variance_limit,
                  supported_mean_discrepancy=mean_adequate and (fields["lower"][0]>margin or fields["upper"][0]<-margin),
                  supported_variance_degradation=variance_adequate and fields["lower"][1]>variance_limit,
                  status="POINTWISE_CRITERIA_REPORTED", cpu_only=True, gpu_intentionally_hidden=True,
                  tensorflow_version=tf.__version__, jit_compile=True, numpy_used=False,
                  trace_count=statistics_kernel.experimental_get_tracing_count(), bootstrap_resamples=B)
    # Nonfinite fields remain explicit unresolved quantities in JSON.
    def plain(value):
        if isinstance(value, dict):
            return {k:plain(v) for k,v in value.items()}
        if isinstance(value, list):
            return [plain(v) for v in value]
        if isinstance(value, float) and not __import__("math").isfinite(value):
            return None
        return value
    return plain(fields)

def preflight():
    x = tf.linspace(tf.constant(-.05,D),tf.constant(.05,D),320)
    source=x.numpy().tolist()
    identity=infer(source,source,0)
    shifted=infer(source,(x+tf.math.log(tf.constant(1.02,D))).numpy().tolist(),0)
    scale=infer(source,(2*x).numpy().tolist(),0)
    offset=infer((x+1000).numpy().tolist(),(x+.002+1000).numpy().tolist(),0)
    location=infer(source,(x+.002).numpy().tolist(),0)
    zero=infer([1.]*320,[1.]*320,0)
    tailx=tf.concat([x[:319],tf.constant([20.],D)],0)
    tail=infer(tailx.numpy().tolist(),tailx.numpy().tolist(),0)
    checks=dict(identity=identity["likelihood_pass"] and identity["variance_pass"],
        known_likelihood_shift=abs(shifted["estimate"][0]-float(tf.math.log(tf.constant(1.02,D)).numpy()))<1e-12 and not shifted["likelihood_pass"],
        known_variance_scaling=abs(scale["estimate"][1]-float(tf.math.log(tf.constant(4.,D)).numpy()))<1e-12,
        normalization_invariance=abs(offset["estimate"][0]-location["estimate"][0])<1e-10,
        paired_indices=identity["bootstrap_lower"]==[0.,0.] and identity["bootstrap_upper"]==[0.,0.],
        zero_variance_rejected=not zero["variance_adequate"] and zero["estimate"][1] is None,
        tail_veto=not tail["likelihood_adequate"] and not tail["variance_adequate"],
        stable_trace=statistics_kernel.experimental_get_tracing_count()==1,
        cpu_only=not tf.config.list_physical_devices("GPU"))
    if not all(checks.values()):
        raise AssertionError(checks)
    return dict(status="PASS",checks=checks,environment="tftwogpu",jit_compile=True,cpu_only=True,
                gpu_intentionally_hidden=True,numpy_used=False,bootstrap_resamples=B)
