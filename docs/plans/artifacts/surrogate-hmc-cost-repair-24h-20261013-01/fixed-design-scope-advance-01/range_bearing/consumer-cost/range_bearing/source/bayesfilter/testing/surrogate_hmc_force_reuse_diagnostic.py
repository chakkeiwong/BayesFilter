"""Paired retained-force mechanics diagnostic; no posterior or tuning admission."""
from pathlib import Path
import hashlib
import time
import tensorflow as tf
from bayesfilter.inference import neural_force_hmc as nf

FLOAT_FIELDS = ("potentials", "log_acceptance_ratio", "initial_potential",
                "final_potential", "initial_kinetic", "final_kinetic", "delta_h")
BOOL_FIELDS = ("accepted", "finite_status", "divergence", "force_fallback")
COUNT_FIELDS = ("endpoint_call_count", "force_call_count")


def paired_programs(force, target, config, *, chains=4, steps=2):
    """Use the public uncached transition as comparator for the shared sampler."""
    dtype = config.tf_dtype
    p = len(config.inverse_mass_diagonal)
    signature = [tf.TensorSpec([chains, p], dtype),
                 tf.TensorSpec([chains], dtype), tf.TensorSpec([2], tf.int32)]

    @tf.function(input_signature=signature, jit_compile=True, autograph=False)
    def cached(q, u, seed):
        chain = nf.sample_neural_force_hmc(
            q, u, force, target, config, num_warmup=0, num_results=steps, seed=seed)
        return {key: getattr(chain, key) for key in
                ("positions", *FLOAT_FIELDS, *BOOL_FIELDS, *COUNT_FIELDS, "endpoint_status")}

    @tf.function(input_signature=signature, jit_compile=True, autograph=False)
    def uncached(q, u, seed):
        fields = {"positions": tf.TensorArray(dtype, steps, element_shape=[chains, p])}
        fields.update({key: tf.TensorArray(dtype, steps, element_shape=[chains])
                       for key in FLOAT_FIELDS})
        fields.update({key: tf.TensorArray(tf.bool, steps, element_shape=[chains])
                       for key in BOOL_FIELDS})
        fields.update({key: tf.TensorArray(tf.int32, steps, element_shape=[])
                       for key in COUNT_FIELDS})
        statuses = {key: tf.TensorArray(value.dtype, steps, element_shape=value.shape)
                    for key, value in nf._default_endpoint_status(u).items()}

        def body(i, q, u, fields, statuses):
            transition = nf.neural_force_hmc_transition(
                q, u, force, target, config, tf.random.experimental.stateless_fold_in(seed, i))
            values = {key: getattr(transition.trace, key) for key in
                      (*FLOAT_FIELDS[1:], *BOOL_FIELDS, *COUNT_FIELDS)}
            values.update(positions=transition.position, potentials=transition.potential)
            return (i+1, transition.position, transition.potential,
                    {key: array.write(i, values[key]) for key, array in fields.items()},
                    {key: array.write(i, transition.trace.endpoint_status[key])
                     for key, array in statuses.items()})

        result = tf.while_loop(lambda i, *_: i < steps, body,
                               (0, q, u, fields, statuses), parallel_iterations=1)
        return dict(**{key: array.stack() for key, array in result[3].items()},
                    endpoint_status={key: array.stack() for key, array in result[4].items()})

    return dict(cached=cached, uncached=uncached)


def compare(left, right):
    """Trajectory tolerance is absolute/relative 1e-9, separate from force precision."""
    gaps = {}
    discrete_equal = True
    for key in ("positions", *FLOAT_FIELDS, *BOOL_FIELDS, "endpoint_call_count"):
        a, b = left[key], right[key]
        if a.dtype.is_floating:
            gap = tf.reduce_max(tf.abs(a-b) / tf.maximum(
                tf.ones_like(a), tf.maximum(tf.abs(a), tf.abs(b))))
            gaps[key] = float(gap)
        else:
            discrete_equal = discrete_equal and bool(tf.reduce_all(a == b))
    for key in left["endpoint_status"]:
        discrete_equal = discrete_equal and bool(tf.reduce_all(
            left["endpoint_status"][key] == right["endpoint_status"][key]))
    healthy = all(bool(tf.reduce_all(x["finite_status"])) and
                  not bool(tf.reduce_any(x["divergence"] | x["force_fallback"]))
                  for x in (left, right))
    return dict(scaled_gaps=gaps, decisions_and_status_equal=discrete_equal,
                healthy=healthy, pass_parity=healthy and discrete_equal and
                all(value <= 1.e-9 for value in gaps.values()))


def run_diagnostic(case, output):
    from bayesfilter.testing import surrogate_hmc_five_model as runtime
    components = runtime.bindings(case)
    starts = tf.constant([[v]*case.p for v in (-.15, -.05, .05, .15)], tf.float64)
    initial = components["target"].function(starts)
    tf.debugging.assert_all_finite(initial.potential, "invalid initial endpoint")
    config = nf.NeuralForceHMCConfig(.002, 1, (1.,)*case.p, dtype="float64")
    programs = paired_programs(components["force"], components["target"], config)
    result = dict(status="RUNNING", scope=case.scope(), selection_hash=case.selection_hash,
                  force_identity=components["force"].identity,
                  source_sha256=hashlib.sha256(Path(nf.__file__).read_bytes()).hexdigest(),
                  steps=2, leapfrog_steps=1, starts=starts, trials=[],
                  evidence_role="paired_mechanics_and_descriptive_cost",
                  nonclaims=["no posterior convergence", "no statistical speed ranking"])
    destination = Path(output)/"force-reuse.json"
    runtime.save(destination, result)
    # Cold and warm comparisons use the same seed within each pair. Alternate
    # warm order so initialization/cache order cannot masquerade as a speed gain.
    for trial, order in enumerate((("uncached", "cached"), ("cached", "uncached"),
                                  ("uncached", "cached"))):
        seed = tf.constant([6111111, 31+2*trial], tf.int32)
        values, seconds, archives = {}, {}, {}
        for name in order:
            begin = time.monotonic()
            values[name] = programs[name](starts, initial.potential, seed)
            tf.nest.map_structure(lambda x: x.numpy(), values[name])
            seconds[name] = time.monotonic()-begin
            archive = {}
            flattened = dict(values[name])
            statuses = flattened.pop("endpoint_status")
            flattened.update({"endpoint_status-"+key: value for key, value in statuses.items()})
            for key, tensor in flattened.items():
                path = Path(output)/f"reuse-{trial}-{name}-{key}.tensor"
                tf.io.write_file(str(path), tf.io.serialize_tensor(tensor))
                archive[key] = dict(path=str(path),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    dtype=tensor.dtype.name, shape=tensor.shape.as_list())
            archives[name] = archive
        assessed = compare(values["cached"], values["uncached"])
        counts = {name: int(tf.reduce_sum(value["force_call_count"]))
                  for name, value in values.items()}
        assessed["force_counts"] = counts
        assessed["count_pass"] = counts == {"cached": 3, "uncached": 4}
        result["trials"].append(dict(trial=trial, cold=trial==0, order=order, seed=seed,
                                    seconds=seconds, assessment=assessed, archive=archives))
        runtime.save(destination, result)
    passed = all(t["assessment"]["pass_parity"] and t["assessment"]["count_pass"]
                 for t in result["trials"])
    result.update(status="FORCE_REUSE_PARITY_PASS" if passed else "FORCE_REUSE_VETO",
                  allocator=tf.config.experimental.get_memory_info("GPU:0"),
                  posterior_sampling_complete=False)
    runtime.save(destination, result)
    return result
