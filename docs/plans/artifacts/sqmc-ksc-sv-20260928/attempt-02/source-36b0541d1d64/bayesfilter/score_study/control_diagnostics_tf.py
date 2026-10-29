"""Compact observability from the actual canonical trace; no filter fork."""
import tensorflow as tf
from .contracts import DiagnosticFailure


def compact_control_diagnostics(trace):
    """Keep all correction diagnostics and named per-time reductions.

    This removes particle/transport histories from the returned graph result,
    while retaining the correction's scalar/vector/matrix diagnostic fields.
    """
    trace=tf.nest.map_structure(lambda *values:tf.stack(values),*trace)
    result={key:value for key,value in trace.items() if key.startswith("higher_moment_")}
    weights=trace["posterior_weights"]
    result["particle_ess_per_time"]=1/tf.reduce_sum(weights**2,axis=1)
    result["weight_sum_error_per_time"]=tf.abs(tf.reduce_sum(weights,axis=1)-1)
    transport=trace["reset_transport"]
    result["transport_row_sum_error_per_time"]=tf.reduce_max(tf.abs(tf.reduce_sum(transport,axis=2)-1),axis=1)
    result["transport_column_weight_error_per_time"]=tf.reduce_max(tf.abs(tf.reduce_mean(transport,axis=1)-weights),axis=1)
    for name in ("predicted_covariances","post_covariances","covariances_after_reset"):
        cov=trace[name]
        eigenvalues=tf.linalg.eigvalsh((cov+tf.linalg.matrix_transpose(cov))*.5)
        result[name+"_minimum_eigenvalue_per_time"]=tf.reduce_min(eigenvalues,axis=[1,2])
        result[name+"_maximum_eigenvalue_per_time"]=tf.reduce_max(eigenvalues,axis=[1,2])
    return result


def materialize_control_diagnostics(tensors):
    """Host artifact boundary; reject invalid observations without changing them."""
    values={key:value.numpy().tolist() for key,value in tensors.items()}
    finite=all(bool(tf.reduce_all(tf.math.is_finite(value)).numpy())
               for value in tensors.values() if value.dtype.is_floating)
    positive=all(bool(tf.reduce_all(value>0).numpy()) for key,value in tensors.items()
                 if key.endswith("_minimum_eigenvalue_per_time"))
    valid=bool(tf.reduce_all(tensors["higher_moment_valid"]).numpy())
    if not finite or not positive or not valid:
        raise DiagnosticFailure("canonical control diagnostics invalid",{
            "control_diagnostics":values,"finite":finite,"positive_covariances":positive,"correction_valid":valid})
    return values
