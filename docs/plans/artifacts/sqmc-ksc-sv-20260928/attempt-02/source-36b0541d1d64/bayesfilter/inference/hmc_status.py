"""Carry deterministic endpoint telemetry alongside the unchanged TFP HMC state."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bayesfilter.inference.hmc_verification import (
    TARGET_STATUS_TELEMETRY_CORE_FIELDS,
    TARGET_STATUS_TELEMETRY_FIELDS,
)


_EXTRA_KEY = "bayesfilter_target_status"


def cached_target_status(results: Any, *, proposed: bool = False) -> Mapping | None:
    extra = getattr(results, "extra", None)
    if not isinstance(extra, Mapping) or _EXTRA_KEY not in extra:
        return None
    return extra[_EXTRA_KEY]["proposed" if proposed else "accepted"]


def cache_hmc_target_status(inner_kernel, target_status_fn, *, uses_dual_averaging=False):
    """Evaluate proposal status once; select accepted status with TFP's MH mask.

    The kernel, integrator and random streams remain TFP-owned. Only its unused
    ``extra`` result field is populated. This saves accepted-state reevaluation;
    the proposal still requires its own status call. There is no mutable cache.
    """
    import tensorflow as tf
    import tensorflow_probability as tfp

    def endpoint(results):
        return results.inner_results if uses_dual_averaging else results

    def replace_endpoint(results, updated):
        return results._replace(inner_results=updated) if uses_dual_averaging else updated

    def read_status(state, shape):
        payload = target_status_fn(state)
        if not isinstance(payload, Mapping) or any(
            key not in payload for key in TARGET_STATUS_TELEMETRY_CORE_FIELDS
        ):
            raise ValueError("cached HMC target-status telemetry is incomplete")
        status = {key: tf.convert_to_tensor(payload[key])
                  for key in TARGET_STATUS_TELEMETRY_FIELDS if key in payload}
        if any(value.shape != shape for value in status.values()):
            raise ValueError("cached HMC target status must match the chain shape")
        return status

    def attach(results, accepted, proposed):
        base = endpoint(results)
        if not isinstance(base.extra, (list, tuple)) or base.extra:
            raise ValueError("cached HMC status requires an unused TFP extra field")
        return replace_endpoint(results, base._replace(extra={
            _EXTRA_KEY: {"accepted": accepted, "proposed": proposed}}))

    class CachedTargetStatus(tfp.mcmc.TransitionKernel):
        @property
        def is_calibrated(self):
            return inner_kernel.is_calibrated

        @property
        def parameters(self):
            return {"inner_kernel": inner_kernel, "target_status_fn": target_status_fn,
                    "uses_dual_averaging": uses_dual_averaging}

        def copy(self, **overrides):
            return cache_hmc_target_status(**(self.parameters | overrides))

        def bootstrap_results(self, init_state):
            results = inner_kernel.bootstrap_results(init_state)
            status = read_status(init_state, endpoint(results).is_accepted.shape)
            return attach(results, status, status)

        def one_step(self, current_state, previous_kernel_results, seed=None):
            previous = endpoint(previous_kernel_results)
            accepted = cached_target_status(previous)
            if accepted is None:
                raise ValueError("cached HMC status missing from previous kernel results")
            # The inner kernel receives its original result structure. Its MH
            # decision supplies the exact per-chain choice for cached evidence.
            clean = replace_endpoint(previous_kernel_results, previous._replace(extra=[]))
            next_state, results = inner_kernel.one_step(current_state, clean, seed=seed)
            base = endpoint(results)
            proposed = read_status(base.proposed_state, base.is_accepted.shape)
            if accepted.keys() != proposed.keys():
                raise ValueError("cached HMC target-status fields changed within a chain")
            selected = {key: tf.where(base.is_accepted, proposed[key], accepted[key])
                        for key in proposed}
            return next_state, attach(results, selected, proposed)

    return CachedTargetStatus()
