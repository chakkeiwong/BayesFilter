"""Compiled fixed-policy exploration summaries; no posterior authority."""
from functools import lru_cache


@lru_cache(maxsize=16)
def pilot_diagnostic_program(shape, projection_count, max_lag, jit_compile=True):
    """Static window shapes bound retracing and avoid scalar or pfor loops.

    Autocorrelation is centered within each chain. Initial positive monotone
    pairs are (rho_0+rho_1), (rho_2+rho_3), ...; signed antithetic correlations
    are preserved. Finite-window estimates cannot establish stationarity.
    """
    import tensorflow as tf
    n, chains, dimension = shape
    if max_lag < 2 or max_lag % 2 or n < 4*max_lag:
        raise ValueError("even lag support and at least four observations per lag required")
    observables_count = 2*(dimension + projection_count) + 1
    signature = [tf.TensorSpec(shape, tf.float64),
                 tf.TensorSpec([dimension], tf.float64),
                 tf.TensorSpec([projection_count, dimension], tf.float64)]

    @tf.function(input_signature=signature, autograph=False, jit_compile=jit_compile)
    def summarize(samples, scale, projections):
        normalized = samples / scale
        projected = tf.einsum("ncd,pd->ncp", normalized, projections)
        linear = tf.concat((normalized, projected), -1)
        observables = tf.concat((linear, tf.square(linear),
            tf.reduce_sum(tf.square(normalized), -1, keepdims=True)), -1)
        center = tf.reduce_mean(observables, 0, keepdims=True)
        centered = observables - center
        variance = tf.reduce_mean(tf.square(centered), 0)
        # Squared machine resolution at the observed level: a numerical
        # reliability check, not a minimum exploration or convergence gate.
        resolution = tf.constant(2.220446049250313e-16, tf.float64) * tf.maximum(
            tf.ones_like(variance), tf.reduce_max(tf.abs(observables), axis=0))
        valid = tf.math.is_finite(variance) & (variance > tf.square(resolution))
        correlations = tf.TensorArray(tf.float64, max_lag+1,
            element_shape=(chains, observables_count)).write(0, tf.ones_like(variance))
        def lag_body(lag, correlations):
            covariance = tf.reduce_mean(centered[:n-lag]*centered[lag:], 0)
            rho = tf.math.divide_no_nan(covariance, variance)
            return lag+1, correlations.write(lag, rho)
        _, correlations = tf.while_loop(lambda lag, _: lag <= max_lag, lag_body,
            (tf.constant(1), correlations), parallel_iterations=1)
        rho = correlations.stack()
        pairs = rho[:max_lag:2] + rho[1:max_lag:2]
        def pair_body(i, active, previous, total):
            current = pairs[i]
            active &= tf.math.is_finite(current) & (current > 0)
            monotone = tf.minimum(previous, current)
            total += tf.where(active, monotone, tf.zeros_like(total))
            return i+1, active, monotone, total
        _, active, _, total = tf.while_loop(lambda i, *_: i < max_lag//2, pair_body,
            (tf.constant(0), valid, tf.fill(tf.shape(variance), tf.constant(float("inf"),tf.float64)),
             tf.zeros_like(variance)), parallel_iterations=1)
        iact = 2*total-1
        usable = valid & tf.math.is_finite(iact) & (iact > 0) & (~active)
        lag1 = tf.reduce_mean(tf.square(linear[1:] - linear[:-1]), (0,1))
        longer = tf.reduce_mean(tf.square(linear[max_lag:] - linear[:-max_lag]), (0,1))
        return dict(iact=iact, lag_limited=active, valid=valid, usable=usable,
            rho=rho, lag1_displacement=lag1, longer_displacement=longer,
            mean_by_chain=center[0], variance_by_chain=variance,
            finite=tf.reduce_all(tf.math.is_finite(observables)))
    return summarize


def summarize_pilot_window(samples, *, frozen_scale, max_lag, projections=(), use_xla=True):
    """Materialize diagnostic values; unusable estimates remain unavailable."""
    import tensorflow as tf
    x = tf.convert_to_tensor(samples, tf.float64)
    if x.shape.rank != 3 or not x.shape.is_fully_defined() or x.shape[-1] != len(frozen_scale):
        raise ValueError("fixed pilot window has incompatible shape")
    scale = tf.constant(frozen_scale, tf.float64)
    if not bool(tf.reduce_all(tf.math.is_finite(scale) & (scale > 0))):
        raise ValueError("frozen coordinate scales must be finite and positive")
    dimension = int(x.shape[-1])
    directions = tf.reshape(tf.constant(projections, tf.float64), [len(projections), dimension])
    if not bool(tf.reduce_all(tf.math.is_finite(directions))):
        raise ValueError("frozen projection contains nonfinite entries")
    if projections and not bool(tf.reduce_all(tf.linalg.norm(directions,axis=-1) > 0)):
        raise ValueError("frozen projections must be nonzero")
    directions = tf.math.divide_no_nan(directions,tf.linalg.norm(directions,axis=-1,keepdims=True))
    raw = pilot_diagnostic_program(tuple(x.shape),len(projections),max_lag,use_xla)(x,scale,directions)
    if not bool(raw["finite"]):
        raise ValueError("pilot observables are nonfinite")
    # All numerical reductions remain TensorFlow; only formatting follows.
    coordinate_usable = tf.reduce_all(raw["usable"][:,:dimension])
    source = (float(tf.reduce_mean(tf.math.reciprocal(
        raw["iact"][:,:dimension]))) if bool(coordinate_usable) else None)
    valid_values = tf.reduce_all(raw["valid"],axis=0).numpy().tolist()
    usable_values = tf.reduce_all(raw["usable"],axis=0).numpy().tolist()
    aggregate = tf.reduce_max(raw["iact"],axis=0).numpy().tolist()
    iact = [v if usable else None for v,usable in zip(aggregate,usable_values)]
    per_coordinate = raw["lag1_displacement"].numpy().tolist()
    linear_roles = ["coordinate"]*dimension + ["frozen_projection"]*len(projections)
    return dict(schema="bayesfilter.discarded_mams_window_diagnostics.v2",
        draws_per_chain=int(x.shape[0]), chain_count=int(x.shape[1]),
        scale_role="fixed_prepared_coordinate_scale_not_global_posterior_sd",
        observable_roles=linear_roles + [r+"_square" for r in linear_roles]+["radius_square"],
        iact_initial_positive_pairs=iact,
        iact_pairing="rho0_plus_rho1_then_positive_monotone_pairs",
        iact_by_chain=raw["iact"].numpy().tolist(),
        iact_lag_limited=tf.reduce_any(raw["lag_limited"],axis=0).numpy().tolist(),
        source_harmonic_coordinate_iact=1/source if source is not None else None,
        source_estimate_usable=bool(coordinate_usable),
        lag1_squared_displacement_by_coordinate=per_coordinate[:dimension],
        lag1_squared_displacement_by_projection=per_coordinate[dimension:],
        longer_lag=max_lag,
        longer_lag_squared_displacement_by_coordinate=raw["longer_displacement"].numpy().tolist()[:dimension],
        worst_coordinate_lag1_squared_displacement=min(per_coordinate[:dimension]),
        worst_observable_iact=max((v for v in iact if v is not None),default=None),
        constant_or_nonfinite_observable=[not v for v in valid_values],
        usable_observable_iact=usable_values,
        mean_by_chain=raw["mean_by_chain"].numpy().tolist(),
        variance_by_chain=raw["variance_by_chain"].numpy().tolist(),
        posterior_convergence_authority=False, use_xla=use_xla)


@lru_cache(maxsize=16)
def pilot_displacement_program(shape, jit_compile=True):
    """Include the first transition and rejection zeros in realized travel."""
    import tensorflow as tf
    signature = [tf.TensorSpec(shape[1:], tf.float64),
                 tf.TensorSpec(shape, tf.float64), tf.TensorSpec(shape, tf.float64),
                 tf.TensorSpec([shape[-1]], tf.float64)]

    @tf.function(input_signature=signature, autograph=False, jit_compile=jit_compile)
    def summarize(initial, samples, proposals, scale):
        before = tf.concat((initial[None], samples[:-1]), axis=0)
        return {
            "proposal_squared_displacement_by_coordinate": tf.reduce_mean(
                tf.square((proposals-before)/scale), axis=(0, 1)),
            "realized_squared_displacement_by_coordinate": tf.reduce_mean(
                tf.square((samples-before)/scale), axis=(0, 1)),
        }
    return summarize


def summarize_pilot_displacement(initial, samples, proposals, *, frozen_scale, use_xla=True):
    import tensorflow as tf
    values = pilot_displacement_program(tuple(samples.shape), use_xla)(
        initial, samples, proposals, tf.constant(frozen_scale, tf.float64))
    if not all(bool(tf.reduce_all(tf.math.is_finite(v))) for v in values.values()):
        raise ValueError("nonfinite pilot displacement")
    return {name: value.numpy().tolist() for name, value in values.items()}
