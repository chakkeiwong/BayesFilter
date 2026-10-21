"""Discarded fixed-geometry ChEES/SNAPER pilots for the shared candidate tuner.

This is an explicit adaptation of TFP 0.25's SNAPER primitives: diagonal metric
updates are disabled, Halton jitter is disabled, and pilot objectives nominate
settings only. Frozen candidates use independent per-chain IID duration laws;
TFP's shared pilot jitter is not misrepresented as that retained transition.
No pilot draw or objective can issue tuning or posterior authority.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
import time
from typing import Any

from .hmc_mams_trajectory_proposals import MAMSPilotConfig, propose_mams_trajectory_candidates


@dataclass(frozen=True)
class TrajectoryPilotConfig:
    criterion: str
    adaptation_steps: int
    initial_step_size: float
    target_accept_probability: float
    max_steps: int
    initial_mala_steps: int
    trajectory_learning_rate: float
    principal_component_ema_factor: int
    state_ema_factor: int
    epsilon_factors: tuple[float, ...]
    length_factors: tuple[float, ...]
    seed: tuple[int, int]

    def __post_init__(self):
        if self.criterion not in {"chees_rate", "snaper"}:
            raise ValueError("criterion must be chees_rate or snaper")
        for name in ("adaptation_steps", "max_steps", "principal_component_ema_factor", "state_ema_factor"):
            if type(getattr(self,name)) is not int or getattr(self,name)<1:
                raise ValueError(name+" must be a positive integer")
        if type(self.initial_mala_steps) is not int or not 0 <= self.initial_mala_steps <= self.adaptation_steps:
            raise ValueError("invalid initial MALA window")
        for name in ("initial_step_size", "trajectory_learning_rate"):
            value=getattr(self,name)
            if type(value) not in (int,float) or not math.isfinite(value) or value <= 0:
                raise ValueError(name+" must be finite and positive")
        if type(self.target_accept_probability) not in (int,float) or not 0 < self.target_accept_probability < 1:
            raise ValueError("invalid optimizer acceptance target")
        for name in ("epsilon_factors","length_factors"):
            values=tuple(getattr(self,name))
            if not values or len(set(values)) != len(values) or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in values):
                raise ValueError("declare distinct positive proposal factors")
            object.__setattr__(self,name,values)
        if len(self.seed)!=2 or any(type(s) is not int or not 0<=s<2**31 for s in self.seed):
            raise ValueError("seed requires two nonnegative int32 components")


def propose_trajectory_candidates(binding: Any, config: TrajectoryPilotConfig) -> dict[str, Any]:
    """Return immutable setting hypotheses; reset starts for actual qualification.

    The caller can put epsilon_by_l into HMCControllerConfig for the same bound
    geometry. The common public tuner remains the only artifact authority.
    """
    import tensorflow as tf
    import tensorflow_probability as tfp
    from tensorflow_probability.python.experimental.mcmc import snaper_hmc as source
    from tensorflow_probability.python.experimental.mcmc import gradient_based_trajectory_length_adaptation as gbtla
    from tensorflow_probability.python.math.generic import reduce_log_harmonic_mean_exp
    from .hmc_candidate_set_execution import HMCCandidateExecutionBinding
    from .hmc import _make_tfp_target_log_prob_fn

    if not isinstance(binding,HMCCandidateExecutionBinding) or not isinstance(config,TrajectoryPilotConfig):
        raise TypeError("repository execution binding and typed pilot config required")
    binding.validate()
    profile=binding.config.transition_profile
    if profile is None or (profile.family,profile.trajectory)!=("hmc","uniform_duration"):
        raise ValueError("trajectory pilots propose only explicit randomized-HMC candidates")
    if tfp.__version__ != "0.25.0":
        raise ValueError("TFP source adaptation requires a reviewed 0.25.0 implementation")
    starts=binding.initial_active_state
    if starts.shape[0]<2:
        raise ValueError("trajectory criterion requires multiple chains")
    lo,hi=binding.scope.epsilon_domain
    if not lo<=config.initial_step_size<=hi:
        raise ValueError("initial pilot epsilon is outside the declared exploration domain")

    class FixedGeometrySNAPER(source.SNAPERHamiltonianMonteCarlo):
        def _make_kernel(self, batch_shape, step, state_ema_points, state, mean, variance, principal_component):
            fixed=tf.nest.map_structure(tf.ones_like,variance)
            kernel,scale=super()._make_kernel(batch_shape,step,state_ema_points,state,mean,fixed,principal_component)
            if config.criterion=="chees_rate":
                kernel=kernel.copy(criterion_fn=gbtla.chees_rate_criterion)
            return kernel,scale

    log_prob=_make_tfp_target_log_prob_fn(binding._active_adapter,dtype=tf.float64)
    # The discarded pilot uses the installed source's shared IID jitter. Its
    # metric stays identity in these already prepared coordinates.
    inner=FixedGeometrySNAPER(log_prob,step_size=tf.constant(config.initial_step_size,tf.float64),
        num_adaptation_steps=config.adaptation_steps,num_mala_steps=config.initial_mala_steps,
        max_leapfrog_steps=config.max_steps,
        trajectory_length_adaptation_rate=config.trajectory_learning_rate,
        principal_component_ema_factor=config.principal_component_ema_factor,
        state_ema_factor=config.state_ema_factor,
        gradient_based_trajectory_length_adaptation_kwargs={
            "use_halton_sequence_jitter":False,"jitter_amount":1.,
            "use_reverse_estimator":True,"averaged_sq_grad_adaptation_rate":.5})
    kernel=tfp.mcmc.DualAveragingStepSizeAdaptation(inner,
        num_adaptation_steps=config.adaptation_steps,
        target_accept_prob=config.target_accept_probability,
        reduce_fn=reduce_log_harmonic_mean_exp)

    def trace_fn(state,kr):
        snaper=kr.inner_results
        length=snaper.inner_results
        mh=length.inner_results
        out={"epsilon":kr.new_step_size,"duration":length.averaged_max_trajectory_length,
             "metric_scale":snaper.max_ema_variance,"criterion":length.criterion,
             "log_accept_ratio":mh.log_accept_ratio,
             "position_finite":tf.reduce_all(tf.math.is_finite(state),axis=-1),
             "realized_steps":mh.proposed_results.num_leapfrog_steps}
        if binding.config.target_status_trace_policy=="per_chain_step":
            out["target_status"]=binding._active_adapter.target_status_telemetry(state)
        return out

    @tf.function(input_signature=[tf.TensorSpec(starts.shape,tf.float64),tf.TensorSpec([2],tf.int32)],
                 autograph=False,jit_compile=binding.config.use_xla)
    def execute(q,seed):
        return tfp.mcmc.sample_chain(num_results=config.adaptation_steps+1,current_state=q,
            kernel=kernel,seed=seed,trace_fn=trace_fn)

    raw=json.dumps({"purpose":"discarded_trajectory_pilot_v1","binding":binding.binding_hash,
                    "config":asdict(config)},sort_keys=True).encode()
    digest=hashlib.sha256(raw).digest()
    seed=tuple(int.from_bytes(digest[i:i+4],"big") & 0x7fffffff for i in (0,4))
    started=time.monotonic()
    samples,trace=execute(starts,tf.constant(seed,tf.int32))
    for name in ("epsilon","duration","metric_scale","criterion","log_accept_ratio"):
        if not bool(tf.reduce_all(tf.math.is_finite(trace[name]))):
            raise ValueError("invalid discarded pilot: "+name)
    if not bool(tf.reduce_all(trace["position_finite"])) or not bool(tf.reduce_all(trace["metric_scale"]==1.)):
        raise ValueError("pilot state or fixed metric contract failed")
    if "target_status" in trace:
        from .hmc_verification import target_status_telemetry_has_failure
        if target_status_telemetry_has_failure(trace["target_status"],expected_shape=(config.adaptation_steps+1,int(starts.shape[0]))):
            raise ValueError("discarded pilot target status failed")
    epsilon=float(trace["epsilon"][-1]);duration=float(trace["duration"][-1])
    if epsilon<=0 or duration<=0:
        raise ValueError("pilot did not yield a positive finite scale")
    pairs=set(); omitted=[]
    for ef in config.epsilon_factors:
        eps=epsilon*ef
        for lf in config.length_factors:
            count=max(1,math.ceil(duration*lf/eps))
            if lo<=eps<=hi and count<=config.max_steps:
                pairs.add((count,eps))
            else:
                omitted.append({"epsilon":eps,"max_steps":count,"reason":"outside_declared_candidate_support"})
    if not pairs:
        raise ValueError("no pilot hypothesis fits the declared epsilon/step support")
    sources={str(Path(inspect.getsourcefile(m)).resolve()):hashlib.sha256(Path(inspect.getsourcefile(m)).read_bytes()).hexdigest()
             for m in (source,gbtla)}
    return {"schema":"bayesfilter.discarded_trajectory_proposals.v1",
        "binding_hash":binding.binding_hash,"config":asdict(config),"derived_seed":seed,
        "tfp_version":tfp.__version__,"author_source_hashes":sources,
        "profile":"fixed_geometry_tfp_trajectory_adaptation_v1",
        "pilot_jitter":"shared_iid_uniform","frozen_jitter":"independent_per_chain_iid_uniform",
        "geometry_adaptation":False,"adaptation_draws_discarded":int(samples.shape[0]),
        "epsilon_by_l":tuple((l,tuple(e for k,e in sorted(pairs) if k==l)) for l in sorted({l for l,_ in pairs})),
        "candidates":[profile.freeze(epsilon=e,max_steps=l).payload() for l,e in sorted(pairs)],
        "omitted_candidates":omitted,"trace":tf.nest.map_structure(lambda x:x.numpy().tolist(),trace),
        "elapsed_seconds":time.monotonic()-started,"use_xla":binding.config.use_xla,
        "samples_device":samples.device,"tuning_artifact_authority":False,
        "fresh_measurement_and_verification_required":True}
