"""Five-model surrogate-HMC diagnostic bindings, with independent MH endpoints.

The FP32 analytical covariance-proposal field only constructs proposals. Exact
Kalman, refined KSC integration, or independent fixed-design references define
the acceptance potential. Finite particle endpoints are explicitly approximate
posteriors; public position-field tuning is mechanics-only, not LEDH admission.
"""
from __future__ import annotations
from dataclasses import dataclass, replace, asdict
import hashlib
import json
import math
from pathlib import Path
import time

import tensorflow as tf
from bayesfilter.testing import surrogate_hmc_literature_priors as priors

from bayesfilter.highdim.covariance_proposal_tf import ProposalControls, make_filter
from bayesfilter.highdim.covariance_proposal_score_tf import make_score_filter
from bayesfilter.highdim.covariance_proposal_score_batch_tf import make_batched_score_filter
from bayesfilter.inference.neural_force_hmc import (
    FrozenPositionOnlyForce, FrozenTargetPotential, PositionOnlyForceEvaluation,
    TargetPotentialEvaluation, NeuralForceHMCConfig, sample_neural_force_hmc,
    bind_neural_force_hmc_tuning_runner,
)

N = 4032
F32, F64 = tf.float32, tf.float64
ROOT = Path(__file__).resolve().parents[2]
FOUR = ROOT / "docs/plans/artifacts/ledh-independent-validation-20261008-01"
RANGE = ROOT / "docs/plans/artifacts/ledh-range-bearing-particle-ladder-20261010-02/N4032"
# Hypotheses for an explicit benchmark posterior, NOT inferred from data.
PRIOR_SCALES = {"lgssm": [.4, .5, .5, .2], "ksc": [.5, .5],
                "predator_prey": [.35] * 6, "sir_d18": [.35] * 3,
                "range_bearing": [math.log(2.)] * 2}
CALIBRATION_SEEDS = (26101211, 26101212)
VALIDATION_SEEDS = (26101213, 26101214)
ENDPOINT_SEEDS = (6111101, 6111102, 6111103, 6111104)


def json_ready(x):
    if hasattr(x, "numpy"):
        return json_ready(x.numpy())
    if hasattr(x, "tolist"):
        return json_ready(x.tolist())
    if isinstance(x, dict):
        return {str(k): json_ready(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [json_ready(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    return x


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(data), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def hashed(data):
    return hashlib.sha256(json.dumps(json_ready(data), sort_keys=True).encode()).hexdigest()


@dataclass
class Case:
    name: str
    spec: object
    observations: object
    theta0: object
    fixed: tuple
    controls: object
    beta: object
    data_path: Path
    fixture: object = None
    precision_policy: str = "uniform"
    tuning_seed_offset: int = 0
    prior_choice: str = priors.BENCHMARK_PRIOR_ID

    @property
    def force_dtype(self):
        return F64 if self.precision_policy == "fp64_recursion_fp32_pairs" else F32

    @property
    def p(self):
        return self.spec.parameter_count

    @property
    def horizon(self):
        return int(self.observations.shape[0])

    def chart(self, q):
        if self.prior_choice == priors.PRIOR_ID:
            return priors.chart(self.name, q, self.theta0)
        if self.prior_choice != priors.BENCHMARK_PRIOR_ID:
            raise ValueError("unknown prior choice")
        scales = tf.constant(PRIOR_SCALES[self.name], q.dtype)
        center = tf.cast(self.theta0, q.dtype)
        if self.name == "predator_prey":
            theta = center * tf.exp(scales * q)
            return theta, scales * theta
        return center + scales * q, scales

    def log_prior(self, q):
        if self.prior_choice == priors.PRIOR_ID:
            return priors.log_prob_and_score(self.name, q, self.theta0)
        if self.prior_choice != priors.BENCHMARK_PRIOR_ID:
            raise ValueError("unknown prior choice")
        # Preserve the historical mechanics potential's additive constant.
        return -.5 * tf.reduce_sum(tf.square(q), axis=-1), -q

    def scope(self):
        from bayesfilter.testing.surrogate_hmc_scope_source_compatibility import scope_source_hashes
        sources = [__file__, priors.__file__,
            ROOT / "bayesfilter/highdim/covariance_proposal_shared_gaussian_tf.py",
            ROOT / "bayesfilter/testing/surrogate_hmc_scope_tuning.py",
            ROOT / "bayesfilter/highdim/analytical_jet_tf.py",
            ROOT / "bayesfilter/highdim/covariance_proposal_score_tf.py",
            ROOT / "bayesfilter/highdim/covariance_proposal_score_batch_tf.py",
            ROOT / "bayesfilter/highdim/covariance_proposal_score_moments_tf.py",
            ROOT / "docs/benchmarks/run_ledh_covariance_proposal.py",
            ROOT / "scripts/run_range_bearing_covariance_comparison.py",
            ROOT / "bayesfilter/highdim/sqmc_lgssm_tf.py",
            ROOT / "bayesfilter/highdim/sqmc_ksc_tf.py",
            ROOT / "bayesfilter/highdim/sqmc_nonlinear_tf.py",
            ROOT / "bayesfilter/highdim/covariance_proposal_beta_tf.py",
            ROOT / "bayesfilter/highdim/ledh_kalman_oracle_tf.py",
            ROOT / "bayesfilter/highdim/ledh_canonical_batch_fused_tf.py",
            ROOT / "bayesfilter/highdim/ledh_canonical_score_stages_tf.py",
            ROOT / "bayesfilter/highdim/ledh_marginal_weights_tf.py",
            ROOT / "bayesfilter/highdim/ledh_small_matrix_tf.py",
            ROOT / "bayesfilter/highdim/ledh_numerical_safety_tf.py",
            ROOT / "bayesfilter/highdim/ledh_ukf_lifecycle_tf.py",
            ROOT / "bayesfilter/highdim/transport_chunk_policy.py",
            ROOT / "bayesfilter/highdim/sqmc_campaign_tf.py",
            ROOT / "bayesfilter/ops/slogdet_tf.py",
            ROOT / "bayesfilter/testing/nonlinear_bootstrap_fisher_reference_tf.py",
            ROOT / "bayesfilter/testing/surrogate_hmc_m13_reference.py",
            ROOT / "experiments/dpf_implementation/tf_tfp/fixtures/range_bearing_tf.py",
            ROOT / "bayesfilter/testing/surrogate_hmc_m13_trajectory_reference.py",

            ROOT / "bayesfilter/highdim/covariance_proposal_tf.py",
            ROOT / "bayesfilter/highdim/covariance_proposal_transport_tf.py",
            ROOT / "bayesfilter/highdim/covariance_proposal_moments_tf.py",
            ROOT / "bayesfilter/highdim/ledh_canonical_models_tf.py",
            ROOT / "bayesfilter/highdim/models.py"]
        return dict(model=self.name, target_id=getattr(self.spec, "target_id", "m13_range_bearing"),
            horizon=self.horizon, particles=N, state_dimension=self.spec.dimension,
            parameter_dimension=self.p, dtype=self.force_dtype.name, tf32=False, jit_compile=True, force_batch_size=2,
            precision_policy=self.precision_policy, pairwise_dtype="float32",
            tuning_seed_offset=self.tuning_seed_offset,
            chunk_policy="dpf_transport_exact_divisor_cap3000_v1", chunk_size=2016,
            data_sha256=hashlib.sha256(self.data_path.read_bytes()).hexdigest(),
            prior=(dict(priors.specification(self.name), nominal=json_ready(self.theta0))
                   if self.prior_choice == priors.PRIOR_ID else
                   dict(id=priors.BENCHMARK_PRIOR_ID, coordinates="q", distribution="independent_standard_normal",
                       nominal=json_ready(self.theta0), scales=PRIOR_SCALES[self.name],
                       chart="multiplicative_exp" if self.name == "predator_prey" else "affine",
                       jacobian="prior is defined in q; no additional chart Jacobian")),
            source_sha256=scope_source_hashes({
                str(Path(s).relative_to(ROOT)): hashlib.sha256(Path(s).read_bytes()).hexdigest()
                for s in sources}))


def configure_prior(case, prior_choice):
    if prior_choice not in (priors.PRIOR_ID, priors.BENCHMARK_PRIOR_ID):
        raise ValueError("unknown prior choice: " + str(prior_choice))
    if prior_choice == priors.PRIOR_ID and case.name == "predator_prey":
        center = tf.cast(case.theta0, F64)
        if not bool(tf.reduce_all((center > tf.constant(priors.PP_LOWER, F64)) &
                                 (center < tf.constant(priors.PP_UPPER, F64)))):
            raise ValueError("predator-prey chart center must be inside the published prior box")
    return replace(case, prior_choice=prior_choice)


def design(case, seed):
    if case.name == "range_bearing":
        from scripts.run_range_bearing_covariance_comparison import inputs, specs
        raw, _ = specs(tf, F64, case.horizon)
        return inputs(tf, raw, N, seed, F32)[:3]
    from bayesfilter.highdim.sqmc_campaign_tf import random_inputs
    fixed = random_inputs("iid_dual_cap", seed, N, case.spec.dimension, case.horizon, F64,
                          jit_compile=True)
    u = tf.random.stateless_uniform([case.horizon, N, 2], [seed, 991], dtype=F64)
    return tuple(tf.cast(x, F32) for x in (*fixed[:2], u))


def load_case(name):
    from docs.benchmarks.run_ledh_covariance_proposal import spec_for
    if name == "range_bearing":
        from scripts.run_range_bearing_covariance_comparison import specs
        _, spec = specs(tf, F32)
        raw, _ = specs(tf, F64)
        path = RANGE / "data/1704-t20.json"
        saved = read(path)
        obs = tf.constant(saved["observations"], F32)
        tuning = read(RANGE / "tuning.json")["selected"]["new"]
        case = Case(name, spec, obs, tf.zeros([2], F32), (), ProposalControls(**tuning["controls"]),
                    tf.constant(tuning["beta"], F32), path, raw["fixture"])
        case.fixture = replace(case.fixture, observations=tf.cast(obs, F64),
                               observation_checksum=hashed(json_ready(obs)))
        seed = 31001
    else:
        spec = spec_for(name, lgssm_family="p44")
        path = FOUR / "inputs" / (name + "-d26100831/dataset.json")
        saved = read(path)
        if saved["target_id"] != spec.target_id or saved["data_seed"] != 26100831:
            raise ValueError("target/data identity mismatch")
        obs64 = tf.constant(saved["observations"], F64)
        if hashlib.sha256(tf.io.serialize_tensor(obs64).numpy()).hexdigest() != saved["observation_sha256"]:
            raise ValueError("observation checksum mismatch")
        tuning = read(FOUR / "frozen" / (name + "-tuning.json"))
        case = Case(name, spec, tf.cast(obs64, F32), spec.default_theta(F32), (),
                    ProposalControls(**tuning["scope"]["controls"]), tf.constant(tuning["beta"], F32), path)
        seed = 261008301
    case.fixed = design(case, seed)
    if case.horizon != (20 if name == "range_bearing" else 50):
        raise ValueError("requested horizon changed")
    return case


def bootstrap_endpoint(case, seed):
    """Independent frozen bootstrap likelihood, not an oracle or PF derivative."""
    p, horizon, d = case.p, case.horizon, case.spec.dimension
    obs = tf.cast(case.observations, F64)
    from bayesfilter.testing.nonlinear_bootstrap_fisher_reference_tf import gaussian_log_density

    @tf.function(input_signature=[tf.TensorSpec([p], F64)], jit_compile=True, autograph=False)
    def evaluate(theta):
        model, _ = case.spec.model(theta, tf.zeros_like(theta))
        factor = tf.linalg.cholesky(model.process_covariance)
        initial = case.spec.initial_mean(F64) + tf.random.stateless_normal([N, d], [seed, 0], dtype=F64)
        def body(t, states, value, valid):
            states = model.transition_mean_fn(theta, states) + tf.matmul(
                tf.random.stateless_normal([N, d], [seed, 3*t+1], dtype=F64), factor, transpose_b=True)
            if model.observation_log_density_fn is None:
                weights = gaussian_log_density(obs[t][None, :] - model.observation_fn(states),
                                               model.observation_covariance)
            else:
                weights = model.observation_log_density_fn(theta, states, obs[t])
            increment = tf.reduce_logsumexp(weights) - tf.math.log(tf.cast(N, F64))
            valid &= tf.math.is_finite(increment) & tf.reduce_all(tf.math.is_finite(states))
            probabilities = tf.nn.softmax(weights)
            cdf = tf.concat([tf.cumsum(probabilities)[:-1], tf.ones([1], F64)], 0)
            index = tf.searchsorted(cdf, tf.random.stateless_uniform([N], [seed, 3*t+2], dtype=F64), side="right")
            return t+1, tf.gather(states, index), value+increment, valid
        out = tf.while_loop(lambda t, *_: t < horizon, body,
            (0, initial, tf.constant(0., F64), tf.constant(True)), parallel_iterations=1)
        return out[2], out[3]
    return evaluate


def reference(case):
    p = case.p
    obs = tf.cast(case.observations, F64)
    if case.name == "lgssm":
        from bayesfilter.highdim.ledh_kalman_oracle_tf import kalman_filter_marginal_loglik
        def calculate(theta):
            value = kalman_filter_marginal_loglik(observations=obs, **case.spec.kalman_parameters(theta))
            return value, tf.math.is_finite(value)
        role = "exact_Kalman"
    elif case.name == "ksc":
        from bayesfilter.highdim.sqmc_ksc_tf import grid_reference
        grid = grid_reference(1201, 48., jit_compile=True)
        def calculate(theta):
            value, _ = grid(theta, obs)
            return value, tf.math.is_finite(value)
        role = "fixed_1201_node_grid_refinement_required"
    elif case.name == "range_bearing":
        from bayesfilter.testing.surrogate_hmc_m13_trajectory_reference import trajectory_importance_reference
        integration = trajectory_importance_reference(case.fixture, N, replicates=4, full_support=True)
        def calculate(theta):
            out = integration(theta, tf.constant([6111101, 9101], tf.int32))
            value = tf.reduce_logsumexp(out["log_likelihood"]) - tf.math.log(tf.constant(4., F64))
            return value, tf.reduce_all(out["valid"]) & tf.math.is_finite(value)
        role = "fixed_full_support_trajectory_importance_4x4032_approximate_posterior"
    else:
        kernels = [bootstrap_endpoint(case, seed) for seed in ENDPOINT_SEEDS]
        def calculate(theta):
            outputs = [kernel(theta) for kernel in kernels]
            value = tf.reduce_logsumexp(tf.stack([v for v, _ in outputs])) - tf.math.log(tf.constant(4., F64))
            return value, tf.reduce_all(tf.stack([ok for _, ok in outputs])) & tf.math.is_finite(value)
        role = "independent_fixed_bootstrap_4x4032_approximate_posterior"
    return tf.function(calculate, input_signature=[tf.TensorSpec([p], F64)],
                       jit_compile=True, autograph=(case.name == "lgssm")), role


def configure_precision(case, policy="uniform", tuning_seed_offset=0):
    """Bind an explicit arithmetic scope; campaign defaults remain uniform FP32."""
    from dataclasses import replace
    from bayesfilter.highdim.covariance_proposal_score_batch_tf import PRECISION_POLICIES
    if policy not in PRECISION_POLICIES or tuning_seed_offset < 0:
        raise ValueError("invalid arithmetic policy or tuning seed offset")
    dtype = F64 if policy != "uniform" else F32
    spec = case.spec
    if case.name == "range_bearing":
        from scripts.run_range_bearing_covariance_comparison import specs
        _, spec = specs(tf, dtype, case.horizon)
    return replace(case, spec=spec, observations=tf.cast(case.observations, dtype),
        theta0=tf.cast(case.theta0, dtype), fixed=tuple(tf.cast(x,dtype) for x in case.fixed),
        beta=tf.cast(case.beta,dtype), precision_policy=policy,
        tuning_seed_offset=tuning_seed_offset)


def bindings(case):
    p = case.p
    batch_size = 2
    dtype = case.force_dtype
    program = make_batched_score_filter(case.spec, N, case.horizon, case.controls, dtype,
        jit_compile=True, batch_size=batch_size, precision_policy=case.precision_policy)
    independent, reference_role = reference(case)

    @tf.function(input_signature=[tf.TensorSpec([p], F64)], jit_compile=True, autograph=False)
    def scalar_force(q):
        theta, jacobian = case.chart(tf.cast(tf.stack([q, q]), dtype))
        result = program(theta, case.observations, *case.fixed, case.beta)
        valid = result["valid"][0] & (result["steps_completed"][0] == case.horizon)
        # The public API subtracts the force in each kick: use grad potential.
        jacobian = tf.broadcast_to(jacobian, [batch_size, p])
        prior_value, prior_score = case.log_prior(q)
        grad = -prior_score - tf.cast(jacobian[0] * result["score"][0], F64)
        return grad, valid & tf.math.is_finite(prior_value) & tf.reduce_all(tf.math.is_finite(grad))

    @tf.function(input_signature=[tf.TensorSpec([None, p], F64)], jit_compile=True, autograph=False)
    def force(positions):
        count = tf.shape(positions)[0]
        groups = (count + batch_size - 1) // batch_size
        values = tf.TensorArray(F64, size=groups, element_shape=[batch_size, p])
        statuses = tf.TensorArray(tf.bool, size=groups, element_shape=[batch_size])
        def body(i, values, statuses):
            first = batch_size * i
            indices = tf.minimum(first + tf.range(batch_size), count - 1)
            q = tf.gather(positions, indices)
            theta, jacobian = case.chart(tf.cast(q, dtype))
            result = program(theta, case.observations, *case.fixed, case.beta)
            prior_value, prior_score = case.log_prior(q)
            grad = -prior_score - tf.cast(jacobian * result["score"], F64)
            valid = result["valid"] & (result["steps_completed"] == case.horizon)
            valid &= tf.math.is_finite(prior_value)
            valid &= tf.reduce_all(tf.math.is_finite(grad), axis=1)
            return i+1, values.write(i, grad), statuses.write(i, valid)
        _, values, statuses = tf.while_loop(lambda i, *_: i < groups, body,
            (0, values, statuses), parallel_iterations=1)
        return PositionOnlyForceEvaluation(
            tf.reshape(values.stack(), [-1, p])[:count],
            tf.reshape(statuses.stack(), [-1])[:count])

    @tf.function(input_signature=[tf.TensorSpec([None, p], F64)], jit_compile=True, autograph=False)
    def endpoint(positions):
        count = tf.shape(positions)[0]
        values = tf.TensorArray(F64, size=count, element_shape=[])
        statuses = tf.TensorArray(tf.bool, size=count, element_shape=[])
        def body(i, values, statuses):
            theta, _ = case.chart(positions[i])
            ll, valid = independent(theta)
            prior_value, _ = case.log_prior(positions[i])
            potential = -prior_value - ll
            valid &= tf.math.is_finite(potential)
            return i+1, values.write(i, tf.where(valid, potential, tf.constant(float("inf"), F64))), statuses.write(i, valid)
        _, values, statuses = tf.while_loop(lambda i, *_: i < count, body, (0, values, statuses),
                                           parallel_iterations=1)
        valid = statuses.stack()
        return TargetPotentialEvaluation(values.stack(), dict(value_valid=valid, force_valid=valid,
            numerical_failure=~valid, outside_declared_support=tf.zeros_like(valid)))

    scope = hashed(dict(scope=case.scope(), controls=asdict(case.controls), beta=json_ready(case.beta),
                        endpoint=reference_role, endpoint_seeds=ENDPOINT_SEEDS))
    frozen_force = FrozenPositionOnlyForce(force, scope+":analytical-"+case.precision_policy+"-"+case.force_dtype.name,
        semantics="deterministic_position_only_proposal_field")
    frozen_target = FrozenTargetPotential(endpoint, scope+":independent-reference")
    binding = bind_neural_force_hmc_tuning_runner(force=frozen_force, target=frozen_target, target_scope=scope)
    class Adapter:
        target_scope = scope
        def adapter_signature(self):
            return scope + ":" + binding.binding_hash
    return dict(force=frozen_force, target=frozen_target, binding=binding, adapter=Adapter(),
                scope=scope, scalar_force=scalar_force, independent=independent, reference_role=reference_role)



def pilot_health(chain):
    """Numerical vetoes apply to cold and warmed diagnostic trajectories."""
    finite = bool(tf.reduce_all(chain.finite_status)
                  & tf.reduce_all(tf.math.is_finite(chain.positions))
                  & tf.reduce_all(tf.math.is_finite(chain.potentials))
                  & tf.reduce_all(tf.math.is_finite(chain.delta_h)))
    fallbacks = int(tf.reduce_sum(tf.cast(chain.force_fallback, tf.int32)))
    divergences = int(tf.reduce_sum(tf.cast(chain.divergence, tf.int32)))
    return dict(finite=finite, force_fallback_count=fallbacks, divergences=divergences,
                veto=not finite or bool(fallbacks) or bool(divergences))


def pilot(case, output, *, steps=2):
    """Actual four-chain HMC mechanics/cost only; no tuning or posterior claim."""
    begin = time.monotonic()
    components = bindings(case)
    p = case.p
    starts = tf.constant([[s] * p for s in (-.15, -.05, .05, .15)], F64)
    progress = dict(status="EVALUATING_INITIAL_ENDPOINT", starts=starts,
        scope=case.scope(), posterior_sampling_complete=False)
    progress_path = Path(output)/"pilot-progress.json"
    save(progress_path, progress)
    potential = components["target"].function(starts)
    progress.update(status="EVALUATING_INITIAL_FORCE", initial_potential=potential.potential,
                    elapsed_seconds=time.monotonic()-begin)
    save(progress_path, progress)
    force = components["force"].function(starts)
    progress.update(status="INITIAL_FORCE_EVALUATED", initial_force=force.force,
        initial_force_valid=force.force_valid, elapsed_seconds=time.monotonic()-begin)
    save(progress_path, progress)
    initial_healthy = (bool(tf.reduce_all(tf.math.is_finite(potential.potential))) and
        bool(tf.reduce_all(force.force_valid)) and
        bool(tf.reduce_all(tf.math.is_finite(force.force))))
    if not initial_healthy:
        progress.update(status="PILOT_INITIAL_HEALTH_VETO", hmc_executed=False,
                        wall_seconds=time.monotonic()-begin)
        save(progress_path, progress)
        save(Path(output)/"pilot.json", progress)
        return progress
    config = NeuralForceHMCConfig(.002, 1, (1.,) * p, dtype="float64")
    @tf.function(input_signature=[tf.TensorSpec([4, p], F64), tf.TensorSpec([4], F64),
                                 tf.TensorSpec([2], tf.int32)], jit_compile=True, autograph=False)
    def run(position, initial, seed):
        return sample_neural_force_hmc(position, initial, components["force"], components["target"],
                                      config, num_warmup=0, num_results=steps, seed=seed)
    # Compilation and first execution must never enter the steady-state forecast.
    start = time.monotonic()
    first = run(starts, potential.potential, tf.constant([6111111, 29], tf.int32))
    tf.nest.map_structure(lambda x: x.numpy() if tf.is_tensor(x) else x, first)
    first_seconds = time.monotonic() - start
    cold_archive = {}
    for name in ("positions", "potentials", "accepted", "delta_h", "force_fallback", "divergence"):
        tensor = getattr(first, name)
        path = Path(output)/("cold_"+name+".tensor")
        tf.io.write_file(str(path),tf.io.serialize_tensor(tensor))
        cold_archive[name] = dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    progress.update(status="COLD_RUN_COMPLETE_WARM_RUN_PENDING",
        cold_run_health=pilot_health(first), cold_archive=cold_archive,
        first_compilation_and_execution_seconds=first_seconds,
        elapsed_seconds=time.monotonic()-begin)
    save(progress_path,progress)
    start = time.monotonic()
    chain = run(starts, potential.potential, tf.constant([6111111, 31], tf.int32))
    tf.nest.map_structure(lambda x: x.numpy() if tf.is_tensor(x) else x, chain)
    seconds = time.monotonic() - start
    cold_health = pilot_health(first)
    warm_health = pilot_health(chain)
    finite = cold_health["finite"] and warm_health["finite"]
    veto = cold_health["veto"] or warm_health["veto"]
    archive = {}
    for prefix, samples in (("cold_", first), ("", chain)):
        for name in ("positions", "potentials", "accepted", "delta_h", "force_fallback", "divergence"):
            tensor = getattr(samples, name)
            path = Path(output) / (prefix + name + ".tensor")
            tf.io.write_file(str(path), tf.io.serialize_tensor(tensor))
            archive[prefix + name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                         dtype=tensor.dtype.name, shape=tensor.shape.as_list())
    result = dict(status="PILOT_HEALTH_VETO" if veto else "PASS_HMC_MECHANICS_ONLY",
        hmc_executed=True, posterior_sampling_complete=False, steps_per_chain=steps,
        particles=N, filter_dtype=case.force_dtype.name, precision_policy=case.precision_policy, tf32=False, hmc_coordinate_dtype="float64",
        endpoint_precision="float64_independent_reference_exception",
        scope=case.scope(), endpoint=components["reference_role"], force_role="selected_controls" if getattr(case, "selection_hash", None) else "saved_controls_warm_start",
        acceptance_per_chain=tf.reduce_mean(tf.cast(chain.accepted, F64), axis=0),
        max_abs_delta_h=tf.reduce_max(tf.abs(chain.delta_h)), force_fallback_count=tf.reduce_sum(tf.cast(chain.force_fallback, tf.int32)),
        divergences=tf.reduce_sum(tf.cast(chain.divergence, tf.int32)), finite=finite,
        cold_run_health=cold_health, warmed_run_health=warm_health,
        first_compilation_and_execution_seconds=first_seconds,
        transition_wall_seconds=seconds, seconds_per_four_chain_transition=seconds/steps,
        minimum_2000_warmup_hours_at_measured_rate=seconds/steps*2000/3600,
        timing_limitation="warmed L=1 mechanics only; tuned leapfrog count and adaptation cost remain unmeasured",
        score_execution="one all-parameter recurrence with particle algebra batched over two chains",
        wall_seconds=time.monotonic()-begin, archive=archive,
        nonclaims=["no convergence or posterior agreement from short mechanics",
                   "no canonical LEDH/default admission", "particle endpoints are approximate posteriors"])
    save(Path(output)/"pilot.json", result)
    progress.update(status=result["status"], elapsed_seconds=time.monotonic()-begin)
    save(progress_path,progress)
    return result


def fresh_observations(case, seed):
    if case.name == "range_bearing":
        from bayesfilter.testing.marginal_weight_campaign import range_fixture
        return tf.cast(range_fixture(case.horizon, seed).observations, F32)
    if case.name in ("predator_prey", "sir_d18"):
        return case.spec.observations(case.horizon, seed, dtype=F32, jit_compile=True)
    return tf.cast(case.spec.simulate(tf.cast(case.theta0, F64), case.horizon, seed,
                                     jit_compile=True), F32)


def tune_filter_scope(case, output, *, smoother_reset=False, fixed_ot_five=False):
    """Fresh exact-scope calibration through the shared numerical implementation."""
    from bayesfilter.testing.surrogate_hmc_scope_tuning import tune
    return tune(case, output, smoother_reset=smoother_reset, fixed_ot_five=fixed_ot_five)


def load_selection(case, path):
    record = read(path)
    signature = record.pop("selection_hash")
    if signature != hashed(record) or record["scope"] != case.scope():
        raise ValueError("stale or mismatched exact-scope filter selection")
    if record["status"] != "FILTER_SCOPE_SELECTED_FOR_EXPLORATORY_HMC" or not record["selected"]["valid"]:
        raise ValueError("filter selection failed")
    case.controls = ProposalControls(**record["selected"]["controls"])
    case.beta = tf.constant(record["selected"]["beta"], case.force_dtype)
    case.selection_hash = signature
    return record


def make_tuning_config(case, scope, *, target_accept_prob=.7, initial_step_size=.01):
    import sys
    from bayesfilter.inference import (
        TensorFlowHMCKernelTuningConfig, VerifiedTensorFlowHMCConfig,
        FourChainMeanBandAcceptancePolicy)
    nomination = TensorFlowHMCKernelTuningConfig(
        parameter_dimension=case.p, evidence_role="candidate",
        mass_window_results=(128, 256), step_adaptation_results=256,
        verification_results=128, max_leapfrog_steps=4, initial_step_size=initial_step_size,
        budget_provenance="five-model N4032 master enforces the owner-selected wall budget",
        initial_step_size_provenance="small standardized-coordinate hypothesis, adapted on own target",
        geometry_provenance="identity coordinate-metric warm start, own-target windows with 10 percent identity shrinkage; not an assertion about prior covariance",
        target_scope=scope,
        acceptance_policy=FourChainMeanBandAcceptancePolicy(overall_band=(.5, 1.), per_chain_band=(.3, 1.)),
        target_accept_prob=target_accept_prob, verification_repair_rounds=1, step_repair_factor=2.,
        mass_shrinkage=.1, covariance_jitter=0.,
        eigenvalue_floor=64*sys.float_info.epsilon, max_condition_number=sys.float_info.epsilon**-.5,
        seed=(6111117, 61), use_xla=True, energy_error_max=1000.)
    return VerifiedTensorFlowHMCConfig(
        nomination=nomination,
        initial_chain_positions=tuple(tuple(s for _ in range(case.p)) for s in (-.75, -.25, .25, .75)),
        initialization_provenance="four dispersed coordinate starts, not prior draws; readiness must be measured on the selected posterior",
        warmup_results=2000, warmup_max_results=10000,
        verification_results=1000, verification_max_results=20000,
        fresh_seed=(6111121, 71), energy_error_max=1000.,
        chunk_results=250, check_every_results=1000)


def inference(case, output, selection_path, *, resume_manifest=None,
              target_accept_prob=None, initial_step_size=None):
    """Public-tuner nomination, adaptive warmup and cumulative retained sampling."""
    from dataclasses import replace
    from bayesfilter.inference import tune_hmc_kernel, VerifiedTensorFlowHMCConfig
    load_selection(case, selection_path)
    components = bindings(case)
    if resume_manifest is not None:
        if target_accept_prob is not None or initial_step_size is not None:
            raise ValueError("resuming a frozen kernel cannot also request retuning")
        parent = Path(resume_manifest).resolve()
        config = replace(VerifiedTensorFlowHMCConfig.from_payload(read(parent)["config"]),
                         resume_manifest=str(parent))
    else:
        config = make_tuning_config(case, components["scope"],
            target_accept_prob=.7 if target_accept_prob is None else target_accept_prob,
            initial_step_size=.01 if initial_step_size is None else initial_step_size)
    save(Path(output)/"hmc-config.json", dict(config=config.payload(), scope=case.scope(),
         endpoint=components["reference_role"], posterior_sampling_includes_public_verification_draws=True,
         redundant_validation_phase=False))
    begin = time.monotonic()
    try:
        result = tune_hmc_kernel(adapter=components["adapter"], initial_position=(0.,)*case.p,
            config=config, parameter_scales=(1.,)*case.p, runner_binding=components["binding"],
            output_dir=Path(output)/"public-tuning")
    except ValueError as exc:
        if "TensorFlow nomination failed" not in str(exc):
            raise
        record = dict(status="HMC_TUNING_REPAIR_REQUIRED", error=str(exc), continuation_veto=False,
                      posterior_sampling_complete=False, wall_seconds=time.monotonic()-begin)
        save(Path(output)/"hmc.json", record)
        return record
    manifest = read(result.artifact_manifest_path)
    record = dict(status=manifest["sequence"]["status"], manifest=result.artifact_manifest_path,
        counts=manifest["sequence"]["counts"], warmup_ready=manifest["sequence"]["warmup_ready"],
        posterior_sampling_complete=result.passed, endpoint=components["reference_role"],
        all_postwarmup_draws_are_inference_draws=True, no_extra_validation=True,
        public_authority="position_field_conditional_mechanics_only",
        posterior_agreement="not_checked", default_readiness=False,
        wall_seconds=time.monotonic()-begin)
    if manifest["verification"]:
        from bayesfilter.inference.hmc_posterior_diagnostics import compute_coordinate_diagnostics
        root = Path(result.artifact_manifest_path).parent
        def tensor(key):
            entry = manifest["verification"][key]
            path = root/entry["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError("retained tensor checksum mismatch")
            return tf.io.parse_tensor(tf.io.read_file(str(path)), tf.as_dtype(entry.get("dtype", "float64")))
        states = tensor("states")
        # These archived states are affine tuning coordinates; return to q.
        q = result.nomination.center + tf.linalg.matmul(states, result.nomination.factor, transpose_b=True)
        record.update(diagnostics=compute_coordinate_diagnostics(tf.transpose(q, [1, 0, 2])),
                      q_mean=tf.reduce_mean(q, axis=[0, 1]),
                      q_standard_deviation=tf.math.reduce_std(q, axis=[0, 1]),
                      divergence_count=tf.reduce_sum(tf.cast(tensor("divergence"), tf.int32)),
                      force_fallback_count=tf.reduce_sum(tf.cast(tensor("force_fallback"), tf.int32)),
                      max_abs_delta_h=tf.reduce_max(tf.abs(tensor("delta_h"))),
                      acceptance_per_chain=tf.reduce_mean(tf.cast(tensor("is_accepted"), F64), axis=0),
                      mean_acceptance_probability_per_chain=tf.reduce_mean(
                          tf.exp(tf.minimum(tensor("log_accept_ratio"), 0.)), axis=0))
    save(Path(output)/"hmc.json", record)
    return record


def endpoint_audit(case, output):
    """Independent likelihood/refinement checks; no posterior prior is required."""
    start = time.monotonic()
    evaluate, role = reference(case)
    theta = tf.cast(case.theta0, F64)
    value, valid = evaluate(theta)
    replay, replay_valid = evaluate(theta)
    result = dict(status="PASS_ENDPOINT_MECHANICS", endpoint=role, value=value,
                  fixed_input_replay_equal=bool(value == replay),
                  valid=bool(valid & replay_valid), particles_per_reference_design=N,
                  likelihood_only=True, exact_posterior_claim=False)
    if case.name == "lgssm":
        oracle, score = case.spec.reference_value_and_score(theta, tf.cast(case.observations, F64))
        result.update(oracle_value=oracle, oracle_score=score, oracle_difference=value-oracle)
        result["valid"] &= abs(float(value-oracle)) < 1e-8
    elif case.name == "ksc":
        from bayesfilter.highdim.sqmc_ksc_tf import grid_reference
        refined, score = grid_reference(1601, 56., jit_compile=True)(theta, tf.cast(case.observations, F64))
        result.update(refined_value=refined, refined_score=score, refinement_difference=value-refined,
                      reference_scope="nominal point only; retained posterior region remains to be checked")
        result["valid"] &= abs(float(value-refined)) < .01
    elif case.name in ("predator_prey", "sir_d18"):
        replicates = [bootstrap_endpoint(case, seed)(theta) for seed in ENDPOINT_SEEDS]
        values = tf.stack([v for v, _ in replicates])
        # Delta-method MCSE of log(mean likelihood), conditional on only four replications.
        relative = tf.exp(values-tf.reduce_max(values))
        mcse = tf.math.reduce_std(relative) * tf.sqrt(tf.constant(4./3., F64)) / 2. / tf.reduce_mean(relative)
        result.update(replicate_log_likelihoods=values, likelihood_log_mcse_delta_method=mcse,
                      uncertainty_role="explanatory; four replicates cannot certify posterior accuracy")
    if not result["valid"] or not result["fixed_input_replay_equal"]:
        result["status"] = "ENDPOINT_REPAIR_REQUIRED"
    result["wall_seconds"] = time.monotonic()-start
    save(Path(output)/"endpoint-audit.json", result)
    return result
