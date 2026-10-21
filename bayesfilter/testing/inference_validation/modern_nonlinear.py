"""Current nonlinear target factories for diagnostic/integration comparisons.

These factories do not load historical learned maps, tuning recipes or result
files. Truncated observations define a new explicit debugging target, not a
qualified replacement for the full model. Initial centers are data-independent
mechanics hypotheses; they are not scientific preparation defaults.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .modern_hmc_matrix import NONLINEAR_COMPARISON_TARGETS


@dataclass(frozen=True)
class NonlinearTargetFixture:
    target_id: str
    adapter: Any
    initial_state: Any
    observations: Any
    source_paths: tuple[str, ...]
    metadata: dict[str, Any]


class FullStatusComparisonAdapter:
    """Preserve the already computed numerical status instead of dropping fields.

    Several older fixtures expose only two fields in target_status_telemetry,
    although their batched evaluator computes floor/conditioning diagnostics.
    This adapter delegates to that evaluator without inventing successful bits.
    """

    def __init__(self, base):
        self.base=base
        self.parameter_dim=base.parameter_dim
        self.batch_rank_policy="rank2_required"

    def adapter_signature(self):
        import hashlib
        return hashlib.sha256((self.base.adapter_signature()+":full_numerical_status_v1").encode()).hexdigest()

    def value_score_capability(self):
        return self.base.value_score_capability()

    def log_prob_and_grad_status(self, q):
        return self.base.neutra_batch_log_prob_and_grad_status(q)

    def log_prob_and_grad(self, q):
        return self.log_prob_and_grad_status(q)[:2]

    def target_status_telemetry(self, q):
        return self.log_prob_and_grad_status(q)[2]


def make_nonlinear_comparison_target(target_id: str, *, horizon: int | None = None,
                                     jit_compile: bool = True) -> NonlinearTargetFixture:
    import hashlib
    import importlib
    import inspect
    import tensorflow as tf

    cell=next((c for c in NONLINEAR_COMPARISON_TARGETS if c.target_id==target_id),None)
    if cell is None or cell.requirement!="core":
        raise ValueError("target requires an explicit adapter/source-route repair before execution")
    if horizon is not None and (type(horizon) is not int or horizon<2):
        raise ValueError("horizon must be at least two observations")
    module=importlib.import_module(cell.module)
    if target_id=="NL-ACC":
        observations=module.model_b_observations_tf()
    elif target_id.startswith("NL-PP"):
        from bayesfilter.testing.predator_prey_ukf_neutra_target_tf import generate_frozen_predator_prey_dataset_tf
        _,observations=generate_frozen_predator_prey_dataset_tf()
    elif target_id=="NL-SIR":
        _,observations,_=module.generate_frozen_sir_dataset_tf()
    else:
        _,observations=module.generate_frozen_structural_dataset_tf()
    full_horizon=int(observations.shape[0])
    if horizon is not None:
        if horizon>full_horizon:
            raise ValueError("requested horizon exceeds this frozen dataset")
        observations=observations[:horizon]
    data_hash=hashlib.sha256(tf.io.serialize_tensor(observations).numpy()).hexdigest()
    if target_id=="NL-ACC":
        bundle=module.make_simple_nonlinear_generic_target_fixture(observations=observations,
            data_hash="sha256:"+data_hash,jit_compile=jit_compile)
        adapter=bundle.adapter
        # Four distinct starts are required by the replicated evidence profile.
        # These inherited fixture-scale offsets are mechanics hypotheses only.
        starts=bundle.initial_batch+tf.constant([-.15,-.05,.05,.15],tf.float64)[:,None]
    else:
        kwargs={"observations":observations}
        if target_id=="NL-PP-SGQF": kwargs["sparse_level"]=2
        adapter=getattr(module,cell.factory)(**kwargs)
        adapter=FullStatusComparisonAdapter(adapter)
        starts=tf.broadcast_to(tf.constant([-.15,-.05,.05,.15],tf.float64)[:,None],[4,cell.parameter_dim])
    return NonlinearTargetFixture(target_id,adapter,starts,observations,
        (str(inspect.getsourcefile(module)), str(inspect.getsourcefile(FullStatusComparisonAdapter))),
        {"target_id":target_id,"family":cell.family,"parameter_dim":cell.parameter_dim,
         "data_sha256":data_hash,"horizon":int(observations.shape[0]),"full_horizon":full_horizon,
         "initialization":"data_independent_mechanics_fixture_not_tuned_geometry",
         "historical_map_or_recipe_loaded":False,"posterior_accuracy_established":False})
