"""Symmetric finite differences of the actual Gaussian study value endpoints.

Common innovations preserve each marginal filter; they do not freeze discrete
ancestor labels. This is a separate FD estimator, never canonical LEDH score.
"""
from functools import lru_cache
import time

from .contracts import digest, seed_pair
from .runtime import configure_runtime, memory_usage


@lru_cache(maxsize=24)
def make_fd_value_kernel(proposal, d, o, N, T, controls_tuple=(),
                         dtype_name="float64", jit_compile=True):
    import tensorflow as tf
    from .gaussian_tf import make_particle_kernel, make_gaussian_kernel, parameterized_model
    dtype = tf.as_dtype(dtype_name)
    if proposal == "ledh":
        from .canonical_adapter_tf import make_canonical_kernel
        consumer = make_canonical_kernel(d, o, N, T, controls_tuple, dtype_name, jit_compile)
    elif proposal in ("kalman", "ukf"):
        consumer = make_gaussian_kernel(d, o, 6, dtype_name, jit_compile, unscented=proposal == "ukf")
    elif proposal in ("bootstrap", "adapted", "prior_sis", "adapted_sis"):
        resample = proposal in ("bootstrap", "adapted")
        consumer = make_particle_kernel(d, o, N, T, dtype_name, jit_compile,
            adapted=proposal in ("adapted", "adapted_sis"), resampling=resample)
    else:
        raise ValueError("FD value endpoint has not been verified for this proposal")

    @tf.function(input_signature=[tf.TensorSpec([6], dtype), tf.TensorSpec([T,o], dtype),
        tf.TensorSpec([N,d], dtype), tf.TensorSpec([T,N,d], dtype),
        tf.TensorSpec([T,N], dtype), tf.TensorSpec([N,d], dtype)], jit_compile=jit_compile)
    def value(theta, observations, initial, process, uniforms, design):
        if proposal == "ledh":
            return consumer(theta, tf.zeros([6], dtype), observations, initial, process, design)[0]
        if proposal in ("kalman", "ukf"):
            return consumer(observations, *parameterized_model(theta,d,o))[0]
        args = (theta, observations, initial, process)
        return consumer(*(args+(uniforms,) if resample else args))[0]
    return value


def fd_streams(row, study, direction_index, node_index):
    coupling = row["fd_coupling"]
    if coupling not in ("common_innovations", "independent_nodes"):
        raise ValueError("unknown FD coupling")
    group = row.get("coupling_group", "baseline")
    if coupling == "independent_nodes":
        group += f":fd_direction{direction_index}:node{node_index}"
    return {stream: seed_pair(master_seed=study["seed"], model=row["model"],
        dataset=row["dataset"], replicate=row["replicate"], stream=stream,
        coupling_group=group) for stream in ("initial", "process", "resampling", "reset_design")}


def evaluate_finite_difference(row, context):
    selection = None
    if row.get("role") == "claim":
        from .fd_selection import consume_fd_selection
        selection = consume_fd_selection(row,context)
    study, registry = context["study"], context["registry"]
    settings = study["settings"]
    runtime = configure_runtime(device=settings["device"], tf32=settings["tf32"],
                                jit_compile=settings["jit_compile"])
    import tensorflow as tf
    from .gaussian_tf import make_data_kernel, make_gaussian_kernel, parameterized_model
    from .finite_difference_tf import checked_nodes, stencil, make_direction_solver
    from .adapters import evaluate_gaussian
    dtype = tf.as_dtype(settings["dtype"])
    d,o,T,N = (settings[k] for k in ("dimension", "observation_dimension", "horizon", "particles"))
    kind, h = row["fd_stencil"], tf.constant(row["fd_h"], dtype)
    V = tf.constant(row["fd_directions"], dtype)
    if V.shape.rank != 2 or V.shape[0] != 6:
        raise ValueError("FD directions must be a 6 by m matrix in declared model coordinates")
    solver = make_direction_solver(6, int(V.shape[1]), settings["dtype"], settings["jit_compile"])
    _,rank,singular,amplification,valid = solver(V,tf.zeros([V.shape[1]],dtype))
    if not bool(valid.numpy()):
        raise ValueError("FD full-score direction rank veto")
    error_budget = row["fd_error_budget"]
    if float(amplification.numpy())*error_budget["directional_l2_bound"] > error_budget["score_l2_bound"]:
        raise ValueError("FD reconstruction exceeds declared propagated error budget")
    # This is an amplification budget, not evidence that directional errors
    # actually obey it. Oracle errors below measure that separate question.
    theta = tf.constant(settings["theta"],dtype)
    lower,upper = (tf.constant(row[k],dtype) for k in ("fd_lower", "fd_upper"))
    points = [checked_nodes(theta,v,h,kind,lower,upper) for v in tf.unstack(tf.transpose(V))]
    nodes,coefficients = stencil(kind,dtype)
    controls = row.get("controls",{})
    if selection is not None:
        controls = selection["selected_signature"].get("controls",{})
    with tf.device("/GPU:0" if settings["device"] == "GPU" else "/CPU:0"):
        value_kernel = make_fd_value_kernel(row["proposal"],d,o,N,T,tuple(sorted(controls.items())),settings["dtype"],settings["jit_compile"])
        data_seed = seed_pair(master_seed=study["seed"],model=row["model"],dataset=row["dataset"],
            replicate=0,stream="observations",coupling_group="common_data")
        observations = make_data_kernel(d,o,T,settings["dtype"],settings["jit_compile"])(
            tf.constant(settings["data_theta"],dtype),tf.constant(data_seed,tf.int32))
        oracle_kernel = make_gaussian_kernel(d,o,6,settings["dtype"],settings["jit_compile"])
        oracle = oracle_kernel(observations,*parameterized_model(theta,d,o))
        cache, stream_cache, seed_records = {}, {}, []
        started = time.monotonic()
        values,exact_values = [],[]
        for i,grid in enumerate(points):
            actual,exact = [],[]
            for j,point in enumerate(tf.unstack(grid)):
                seeds = fd_streams(row,study,i,j)
                key = digest(seeds)
                seed_records.append(seeds)
                if key not in stream_cache:
                    stream_cache[key] = (
                        tf.random.stateless_normal([N,d],seeds["initial"],dtype=dtype),
                        tf.random.stateless_normal([T,N,d],seeds["process"],dtype=dtype),
                        tf.random.stateless_uniform([T,N],seeds["resampling"],dtype=dtype),
                        tf.random.stateless_normal([N,d],seeds["reset_design"],dtype=dtype))
                value_key = (tuple(point.numpy().tolist()),key)
                if value_key not in cache:
                    v = value_kernel(point,observations,*stream_cache[key])
                    if not bool(tf.math.is_finite(v).numpy()):
                        raise ValueError("perturbed finite likelihood validity veto")
                    cache[value_key] = (v,oracle_kernel(observations,*parameterized_model(point,d,o))[0])
                a,e = cache[value_key]
                actual.append(a);exact.append(e)
            values.append(tf.stack(actual));exact_values.append(tf.stack(exact))
        values,exact_values = tf.stack(values),tf.stack(exact_values)
        directional = tf.linalg.matvec(values,coefficients)/h
        score,rank,singular,amplification,valid = solver(V,directional)
        if not bool(valid.numpy()): raise ValueError("FD reconstruction validity veto")
        baseline_row = dict(row, estimator="exact_gaussian" if row["proposal"] == "kalman" else "analytical_filter")
        # The FD selection already validates the full frozen controls and
        # execution scope. The analytical comparator uses exactly those controls.
        if selection is not None: baseline_row.update(role="mechanics",controls=controls)
        baseline = evaluate_gaussian(baseline_row,context)
        wall = time.monotonic()-started
        runtime.update(dtype=dtype.name,value_device=values.device,score_device=score.device,
            kernel_wall_seconds=wall,kernel_calls=len(cache)+baseline["runtime"]["kernel_calls"],
            finite_difference_value_calls=len(cache),oracle_value_calls=len(cache)+1,
            traces=value_kernel.experimental_get_tracing_count(),**memory_usage(settings["device"]))
        diagnostics = dict(baseline["diagnostics"], fd_stencil=kind,fd_h=float(h.numpy()),
            fd_coupling=row["fd_coupling"],fd_nodes=nodes.numpy().tolist(),
            fd_coefficients=coefficients.numpy().tolist(),fd_directions=V.numpy().tolist(),
            fd_values=values.numpy().tolist(),fd_exact_values=exact_values.numpy().tolist(),
            fd_directional_scores=directional.numpy().tolist(),
            fd_oracle_directional_scores=tf.linalg.matvec(tf.transpose(V),oracle[1]).numpy().tolist(),
            fd_singular_values=singular.numpy().tolist(),fd_rank=int(rank.numpy()),
            fd_error_amplification=float(amplification.numpy()),fd_error_budget=error_budget,
            fd_reconstruction_residual=tf.linalg.norm(tf.linalg.matvec(tf.transpose(V),score)-directional).numpy().item(),
            fd_analytic_score=baseline["score"],fd_analytic_value=baseline["value"],
            fd_marginal_law="original value endpoint with correct Gaussian/uniform marginals; ancestors recomputed",
            fd_streams=seed_records,canonical_score_status="separate_fd_candidate",
            score_squared_error=float(tf.reduce_sum((score-oracle[1])**2).numpy()),
            fd_cost_includes_baseline_and_oracle=True)
        diagnostics["fd_tuning_status"] = "verified_frozen_selection" if selection else "calibration_or_mechanics"
    estimator = registry.estimators[row["estimator"]]
    return dict(baseline,score=score.numpy().tolist(),derivative_target=estimator.target,
                derivative_id=estimator.derivative,runtime=runtime,diagnostics=diagnostics)
