"""Exact fixed-chart and proper-bridge replica-exchange transitions.

The chart selector is a fixed categorical law, independent of the current
state.  Each injected chart kernel must already be an invariant
Metropolis-corrected kernel for the same target.  Replica swaps use the full
bridge density at both exchanged states; no pure-power shortcut is used.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import tensorflow as tf


TRANSITION_NONCLAIMS = (
    "fixed mechanics and invariance fixtures only",
    "fixed chart frequencies are not posterior mode masses",
    "swap and chart mechanics do not establish convergence or discovery",
    "no default HMC policy or posterior-validity claim",
)


class TemperedTransitionError(ValueError):
    """Raised when a transition contract is invalid."""


def _hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _rank2(value: Any, dimension: int, name: str) -> tf.Tensor:
    tensor = tf.convert_to_tensor(value, tf.float64)
    if tensor.shape.rank != 2 or tensor.shape[0] is None or tensor.shape[-1] != int(dimension):
        raise TemperedTransitionError(f"{name} must have static shape [batch,{int(dimension)}]")
    if int(tensor.shape[0]) <= 0:
        raise TemperedTransitionError(f"{name} must have a nonempty batch")
    return tensor


def _rank3(value: Any, shape: tuple[int, int, int], name: str) -> tf.Tensor:
    tensor = tf.convert_to_tensor(value, tf.float64)
    if tensor.shape.rank != 3 or tuple(tensor.shape.as_list()) != tuple(shape):
        raise TemperedTransitionError(f"{name} must have static shape {shape}")
    return tensor


@dataclass(frozen=True)
class FixedChartSelection:
    """Immutable state-independent chart-selection frequencies."""

    gamma: tuple[float, ...]
    chart_ids: tuple[str, ...]
    policy_id: str = "fixed_state_independent_chart_mixture_v1"

    def __post_init__(self) -> None:
        try:
            values = tuple(float(item) for item in self.gamma)
        except (TypeError, ValueError) as exc:
            raise TemperedTransitionError(
                "gamma must be a fixed numeric sequence; state-dependent selectors "
                "are not representable"
            ) from exc
        ids = tuple(str(item) for item in self.chart_ids)
        if not values or len(values) != len(ids):
            raise TemperedTransitionError("gamma and chart_ids must be nonempty and aligned")
        if any(not math.isfinite(item) or item <= 0.0 for item in values):
            raise TemperedTransitionError("gamma entries must be finite and strictly positive")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise TemperedTransitionError("gamma must sum to one")
        if any(not item for item in ids) or len(set(ids)) != len(ids):
            raise TemperedTransitionError("chart_ids must be nonempty and unique")
        object.__setattr__(self, "gamma", values)
        object.__setattr__(self, "chart_ids", ids)

    @property
    def chart_count(self) -> int:
        return len(self.gamma)

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.tempered.fixed_chart_selection.v1",
            "policy_id": self.policy_id,
            "gamma": list(self.gamma),
            "chart_ids": list(self.chart_ids),
            "state_independent": True,
            "nonclaims": list(TRANSITION_NONCLAIMS),
        }

    @property
    def signature(self) -> str:
        return _hash(self.payload())


class FixedChartKernelMixture:
    """Apply one of several invariant kernels with fixed categorical gamma."""

    def __init__(
        self,
        kernels: Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]],
        *,
        gamma: Sequence[float],
        chart_ids: Sequence[str] | None = None,
    ) -> None:
        values = tuple(kernels)
        if not values or any(not callable(kernel) for kernel in values):
            raise TemperedTransitionError("kernels must be a nonempty callable sequence")
        ids = tuple(chart_ids) if chart_ids is not None else tuple(
            f"chart-{index}" for index in range(len(values))
        )
        self.selection = FixedChartSelection(tuple(gamma), ids)
        if len(values) != self.selection.chart_count:
            raise TemperedTransitionError("kernel count does not match gamma")
        self.kernels = values
        self.parameter_dim: int | None = None

    def _transition_tensor(
        self, state: Any, seed: Any
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Apply the fixed categorical kernel without graph-hostile metadata.

        XLA GPU does not provide a string ``GatherV2`` kernel.  The compiled
        replica program only needs the physical state and numeric chart index;
        human-readable chart IDs stay at the eager/artifact boundary.
        """
        current = tf.convert_to_tensor(state, tf.float64)
        if (
            current.shape.rank != 2
            or current.shape[0] is None
            or current.shape[1] is None
        ):
            raise TemperedTransitionError(
                "state must have static rank-2 shape [chain,dimension]"
            )
        seed_tensor = tf.convert_to_tensor(seed, tf.int32)
        if seed_tensor.shape != (2,):
            raise TemperedTransitionError("seed must have shape [2]")
        logits = tf.math.log(tf.constant(self.selection.gamma, tf.float64))[tf.newaxis, :]
        # ``stateless_categorical`` currently returns int64 on this TensorFlow
        # build, while ``tf.switch_case`` requires an int32 branch index.
        selected = tf.cast(
            tf.squeeze(
                tf.random.stateless_categorical(logits, 1, seed_tensor),
                axis=(0, 1),
            ),
            tf.int32,
        )
        branches = {}
        for index, kernel in enumerate(self.kernels):
            branch_seed = tf.random.experimental.stateless_fold_in(
                seed_tensor, 1000 + index
            )

            def branch(kernel=kernel, branch_seed=branch_seed):
                result = tf.convert_to_tensor(kernel(current, branch_seed), tf.float64)
                if result.shape != current.shape:
                    raise TemperedTransitionError(
                        "chart kernel must preserve the state shape"
                    )
                return result

            branches[index] = branch
        next_state = tf.switch_case(selected, branch_fns=branches)
        return next_state, selected

    def transition_state(self, state: Any, seed: Any) -> tf.Tensor:
        """Return only the tensor state for compiled transition programs."""
        return self._transition_tensor(state, seed)[0]

    def transition(self, state: Any, seed: Any) -> Mapping[str, Any]:
        next_state, selected = self._transition_tensor(state, seed)
        # Keep the descriptive ID for eager diagnostics.  During tracing the
        # compiled caller consumes only tensor fields; constructing a string
        # tensor there would introduce an XLA-incompatible GatherV2 operation.
        selected_chart_id: str | None
        if tf.inside_function():
            selected_chart_id = None
        else:
            selected_chart_id = self.selection.chart_ids[int(selected.numpy())]
        return {
            "state": next_state,
            "selected_chart_index": tf.cast(selected, tf.int32),
            "selected_chart_id": selected_chart_id,
            "gamma": tf.constant(self.selection.gamma, tf.float64),
            "selection_signature": self.selection.signature,
            "state_independent_selection": True,
            "nonclaims": TRANSITION_NONCLAIMS,
        }


def build_fixed_transport_hmc_kernel(
    adapter: Any,
    *,
    state_shape: tuple[int, int],
    step_size: float,
    num_leapfrog_steps: int,
    jit_compile: bool = True,
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """Build one non-claim-bearing exact HMC mechanics transition.

    Serious sampling must use ``build_tuned_fixed_transport_hmc_kernel`` so the
    mechanics come from a verified scope-specific tuner handoff.
    """
    shape = tuple(int(item) for item in state_shape)
    if len(shape) != 2 or any(item <= 0 for item in shape):
        raise TemperedTransitionError(
            "state_shape must be a positive (chain,dimension) tuple"
        )
    if int(getattr(adapter, "parameter_dim", -1)) != shape[1]:
        raise TemperedTransitionError(
            "fixed-transport adapter dimension does not match physical state"
        )
    transport = getattr(adapter, "transport", None)
    inverse = getattr(transport, "inverse_theta_to_z_batch", None)
    if not callable(inverse):
        inverse_pair = getattr(transport, "inverse_and_forward_logdet", None)
        if callable(inverse_pair):
            inverse = lambda value: inverse_pair(value)[0]
    forward = getattr(adapter, "latent_to_position", None)
    if not callable(inverse) or not callable(forward):
        raise TemperedTransitionError(
            "physical chart kernel requires explicit inverse and forward maps"
        )
    try:
        from bayesfilter.inference.fixed_transport_hmc_mechanics_tf import (
            build_fixed_transport_one_step_transition,
        )
    except ImportError as exc:  # pragma: no cover - environment failure path
        raise TemperedTransitionError(
            "shared fixed-transport HMC mechanics are unavailable"
        ) from exc
    try:
        shared_transition = build_fixed_transport_one_step_transition(
            adapter,
            state_shape=state_shape,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            use_xla=jit_compile,
        )
    except (TypeError, ValueError) as exc:
        raise TemperedTransitionError(str(exc)) from exc

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape, tf.float64),
            tf.TensorSpec([2], tf.int32),
        ),
        jit_compile=bool(jit_compile),
        reduce_retracing=False,
    )
    def run(state: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        latent = tf.ensure_shape(
            tf.convert_to_tensor(inverse(state), tf.float64), shape
        )
        inverse_finite = tf.reduce_all(tf.math.is_finite(latent))
        safe_latent = tf.where(
            inverse_finite,
            latent,
            tf.fill(shape, tf.constant(float("nan"), tf.float64)),
        )
        next_latent = shared_transition(safe_latent, seed)[0]
        next_physical = tf.ensure_shape(
            tf.convert_to_tensor(forward(next_latent), tf.float64), shape
        )
        return tf.where(
            tf.logical_and(
                inverse_finite,
                tf.reduce_all(tf.math.is_finite(next_physical)),
            ),
            next_physical,
            tf.fill(shape, tf.constant(float("nan"), tf.float64)),
        )

    run.mechanics_role = "smoke_or_analytic_fixture_only"  # type: ignore[attr-defined]
    run.shared_mechanics = True  # type: ignore[attr-defined]

    return run


def build_tuned_fixed_transport_hmc_kernel(
    handoff: Any,
    *,
    state_shape: tuple[int, int],
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """Build one chart kernel from an immutable verified tuner handoff."""
    try:
        from bayesfilter.inference.fixed_transport_hmc_tuning_tf import (
            VerifiedFixedTransportHMCHandoff,
        )
    except ImportError as exc:  # pragma: no cover - environment failure path
        raise TemperedTransitionError("fixed-transport tuner is unavailable") from exc
    if not isinstance(handoff, VerifiedFixedTransportHMCHandoff):
        raise TemperedTransitionError(
            "claim-bearing chart kernels require VerifiedFixedTransportHMCHandoff"
        )
    payload = handoff.payload()
    adapter = handoff.transformed_adapter
    if payload.get("schema") != "bayesfilter.verified_fixed_transport_hmc_handoff.v1":
        raise TemperedTransitionError("verified tuner handoff schema mismatch")
    if payload.get("target_scope") != getattr(adapter, "target_scope", None):
        raise TemperedTransitionError("verified tuner handoff target scope mismatch")
    adapter_signature = getattr(adapter, "adapter_signature", None)
    if not callable(adapter_signature) or payload.get(
        "transformed_adapter_signature"
    ) != adapter_signature():
        raise TemperedTransitionError(
            "verified tuner handoff adapter identity mismatch"
        )
    if payload.get("fixed_transport_manifest_hash") != getattr(
        adapter, "transport_manifest_hash", None
    ):
        raise TemperedTransitionError(
            "verified tuner handoff chart identity mismatch"
        )
    kernel = build_fixed_transport_hmc_kernel(
        adapter,
        state_shape=state_shape,
        step_size=handoff.step_size,
        num_leapfrog_steps=handoff.num_leapfrog_steps,
        jit_compile=bool(payload.get("use_xla", False)),
    )
    kernel.mechanics_role = "verified_scope_specific_tuner_handoff"  # type: ignore[attr-defined]
    kernel.tuning_handoff_hash = handoff.handoff_hash  # type: ignore[attr-defined]
    return kernel


@dataclass(frozen=True)
class TransportReliabilityReceipt:
    """Round-trip and score-finiteness screen for a frozen learned map."""

    passed: bool
    tolerance: float
    component_ids: tuple[str, ...]
    self_roundtrip_max: tuple[float, ...]
    cross_roundtrip_max: tuple[float, ...]
    reference_roundtrip_max: tuple[float, ...]
    declared_roundtrip_max: tuple[float, ...]
    self_logdet_residual_max: tuple[float, ...]
    cross_logdet_residual_max: tuple[float, ...]
    reference_logdet_residual_max: tuple[float, ...]
    declared_logdet_residual_max: tuple[float, ...]
    score_finite: tuple[bool, ...]
    conditioning_proxy: tuple[float, ...]
    maximum_condition_number: float
    finite_difference_relative_step: float
    physical_score_checked: bool
    failures: tuple[str, ...]

    def payload(self) -> Mapping[str, Any]:
        payload = asdict(self)
        payload["component_ids"] = list(self.component_ids)
        for key in (
            "self_roundtrip_max",
            "cross_roundtrip_max",
            "reference_roundtrip_max",
            "declared_roundtrip_max",
            "self_logdet_residual_max",
            "cross_logdet_residual_max",
            "reference_logdet_residual_max",
            "declared_logdet_residual_max",
            "conditioning_proxy",
        ):
            payload[key] = list(payload[key])
        payload["score_finite"] = list(self.score_finite)
        payload["failures"] = list(self.failures)
        payload["schema"] = "bayesfilter.tempered.transport_reliability_receipt.v2"
        payload["nonclaims"] = list(TRANSITION_NONCLAIMS)
        return payload


def derive_reliability_tolerance(
    *,
    dimension: int,
    dtype: tf.dtypes.DType = tf.float64,
    analytic_roundoff: float = 1.0e-12,
) -> float:
    """Derive a round-trip tolerance from dtype precision and dimension."""
    if int(dimension) <= 0:
        raise TemperedTransitionError("dimension must be positive")
    # Keep the runtime path TensorFlow-only.  These are the IEEE unit-roundoff
    # values for the two dtypes admitted by the transport implementation.
    dtype = tf.as_dtype(dtype)
    if dtype == tf.float64:
        epsilon = 2.220446049250313e-16
    elif dtype == tf.float32:
        epsilon = 1.1920928955078125e-7
    else:
        raise TemperedTransitionError("reliability dtype must be float32 or float64")
    roundoff = float(analytic_roundoff)
    if not math.isfinite(roundoff) or roundoff < 0.0:
        raise TemperedTransitionError("analytic_roundoff must be finite and nonnegative")
    return max(roundoff, 64.0 * float(dimension) * epsilon)


def screen_transport_reliability(
    transports: Sequence[Any],
    *,
    component_ids: Sequence[str] | None = None,
    self_latent_bank: Any,
    cross_physical_bank: Any,
    reference_points: Any,
    declared_points: Any | None = None,
    physical_score_fn: Callable[[tf.Tensor], Any],
    maximum_condition_number: float,
    tolerance: float | None = None,
) -> TransportReliabilityReceipt:
    """Screen every chart on self, cross, reference, and declared points."""
    charts = tuple(transports)
    if not charts:
        raise TemperedTransitionError("at least one transport is required")
    dimensions = tuple(int(getattr(chart, "parameter_dim")) for chart in charts)
    if len(set(dimensions)) != 1:
        raise TemperedTransitionError("all transports must share parameter_dim")
    dimension = dimensions[0]
    ids = tuple(component_ids) if component_ids is not None else tuple(
        f"chart-{index}" for index in range(len(charts))
    )
    if len(ids) != len(charts):
        raise TemperedTransitionError("component_ids do not match transports")
    self_latent = tf.convert_to_tensor(self_latent_bank, tf.float64)
    if self_latent.shape.rank != 3 or self_latent.shape[0] != len(charts) or self_latent.shape[2] != dimension:
        raise TemperedTransitionError("self_latent_bank must have shape [component,batch,dimension]")
    cross_physical = tf.convert_to_tensor(cross_physical_bank, tf.float64)
    if cross_physical.shape.rank != 3 or cross_physical.shape[0] != len(charts) or cross_physical.shape[2] != dimension:
        raise TemperedTransitionError("cross_physical_bank must have shape [component,batch,dimension]")
    reference = _rank2(reference_points, dimension, "reference_points")
    declared = reference if declared_points is None else _rank2(declared_points, dimension, "declared_points")
    tol = derive_reliability_tolerance(dimension=dimension) if tolerance is None else float(tolerance)
    if not math.isfinite(tol) or tol <= 0.0:
        raise TemperedTransitionError("tolerance must be finite and positive")
    if not callable(physical_score_fn):
        raise TemperedTransitionError("physical_score_fn must be callable")
    max_condition = float(maximum_condition_number)
    if not math.isfinite(max_condition) or max_condition < 1.0:
        raise TemperedTransitionError(
            "maximum_condition_number must be finite and at least one"
        )
    finite_difference_relative_step = 2.220446049250313e-16 ** (1.0 / 3.0)

    self_residuals = []
    cross_residuals = []
    reference_residuals = []
    declared_residuals = []
    self_logdet = []
    cross_logdet = []
    reference_logdet = []
    declared_logdet = []
    score_ok = []
    conditioning = []
    failures: list[str] = []

    for index, chart in enumerate(charts):
        forward = getattr(chart, "forward_and_logdet", None)
        inverse = getattr(chart, "inverse_and_forward_logdet", None)
        score_fn = getattr(chart, "log_abs_det_jacobian_score_batch", None)
        if not callable(forward) or not callable(inverse):
            failures.append(f"{ids[index]}:missing_forward_or_inverse")
            self_residuals.append(float("inf"))
            cross_residuals.append(float("inf"))
            reference_residuals.append(float("inf"))
            declared_residuals.append(float("inf"))
            self_logdet.append(float("inf"))
            cross_logdet.append(float("inf"))
            reference_logdet.append(float("inf"))
            declared_logdet.append(float("inf"))
            score_ok.append(False)
            conditioning.append(float("inf"))
            continue
        latent = tf.convert_to_tensor(self_latent[index], tf.float64)
        physical, logdet = forward(latent)
        recovered, inverse_logdet = inverse(physical)
        self_error = tf.reduce_max(tf.abs(recovered - latent))
        self_ld_error = tf.reduce_max(tf.abs(logdet - inverse_logdet))
        cross_rows = tf.reshape(cross_physical, [-1, dimension])
        cross_recovered, cross_inverse_logdet = inverse(cross_rows)
        cross_forward, cross_forward_logdet = forward(cross_recovered)
        cross_error = tf.reduce_max(tf.abs(cross_forward - cross_rows))
        cross_ld_error = tf.reduce_max(tf.abs(cross_forward_logdet - cross_inverse_logdet))
        ref_recovered, ref_inverse_logdet = inverse(reference)
        ref_roundtrip, ref_forward_logdet = forward(ref_recovered)
        ref_error = tf.reduce_max(tf.abs(ref_roundtrip - reference))
        ref_ld_error = tf.reduce_max(
            tf.abs(ref_forward_logdet - ref_inverse_logdet)
        )
        declared_recovered, declared_inverse_logdet = inverse(declared)
        declared_roundtrip, declared_forward_logdet = forward(declared_recovered)
        declared_error = tf.reduce_max(tf.abs(declared_roundtrip - declared))
        declared_ld_error = tf.reduce_max(
            tf.abs(declared_forward_logdet - declared_inverse_logdet)
        )
        all_latent = tf.concat(
            (latent, cross_recovered, ref_recovered, declared_recovered), axis=0
        )
        all_physical, _ = forward(all_latent)
        pulled_score = getattr(chart, "pullback_score_batch", None)
        if callable(score_fn) and callable(pulled_score):
            raw_score = physical_score_fn(all_physical)
            if isinstance(raw_score, (tuple, list)):
                if len(raw_score) < 2:
                    raise TemperedTransitionError(
                        "physical_score_fn tuple must contain a score in position one"
                    )
                raw_score = raw_score[1]
            physical_score = tf.convert_to_tensor(raw_score, tf.float64)
            if physical_score.shape != all_physical.shape:
                raise TemperedTransitionError(
                    "physical_score_fn must return one score per physical row"
                )
            transformed_score = score_fn(all_latent) + pulled_score(
                all_latent, physical_score
            )
            finite_score = bool(
                tf.reduce_all(tf.math.is_finite(transformed_score)).numpy()
            )
        else:
            finite_score = False

        row_scale = tf.maximum(
            tf.ones([tf.shape(all_latent)[0]], tf.float64),
            tf.reduce_max(tf.abs(all_latent), axis=1),
        )
        step = (
            row_scale[:, tf.newaxis, tf.newaxis]
            * tf.constant(finite_difference_relative_step, tf.float64)
        )
        directions = tf.eye(dimension, dtype=tf.float64)[tf.newaxis, :, :]
        centers = all_latent[:, tf.newaxis, :]
        plus = tf.reshape(centers + step * directions, [-1, dimension])
        minus = tf.reshape(centers - step * directions, [-1, dimension])
        plus_physical, _ = forward(plus)
        minus_physical, _ = forward(minus)
        jacobian_input_output = tf.reshape(
            plus_physical - minus_physical,
            [tf.shape(all_latent)[0], dimension, dimension],
        ) / (2.0 * step)
        jacobian = tf.transpose(jacobian_input_output, (0, 2, 1))
        singular_values = tf.linalg.svd(jacobian, compute_uv=False)
        condition_values = singular_values[:, 0] / singular_values[:, -1]
        condition_proxy = float(tf.reduce_max(condition_values).numpy())
        values = (
            float(self_error.numpy()),
            float(cross_error.numpy()),
            float(ref_error.numpy()),
            float(declared_error.numpy()),
            float(self_ld_error.numpy()),
            float(cross_ld_error.numpy()),
            float(ref_ld_error.numpy()),
            float(declared_ld_error.numpy()),
        )
        self_residuals.append(values[0])
        cross_residuals.append(values[1])
        reference_residuals.append(values[2])
        declared_residuals.append(values[3])
        self_logdet.append(values[4])
        cross_logdet.append(values[5])
        reference_logdet.append(values[6])
        declared_logdet.append(values[7])
        score_ok.append(finite_score)
        conditioning.append(condition_proxy)
        if any(not math.isfinite(value) or value > tol for value in values):
            failures.append(f"{ids[index]}:roundtrip_or_logdet_tolerance")
        if not finite_score:
            failures.append(f"{ids[index]}:nonfinite_transformed_score")
        if not math.isfinite(condition_proxy) or condition_proxy > max_condition:
            failures.append(f"{ids[index]}:jacobian_condition_number")

    return TransportReliabilityReceipt(
        passed=not failures,
        tolerance=tol,
        component_ids=ids,
        self_roundtrip_max=tuple(self_residuals),
        cross_roundtrip_max=tuple(cross_residuals),
        reference_roundtrip_max=tuple(reference_residuals),
        declared_roundtrip_max=tuple(declared_residuals),
        self_logdet_residual_max=tuple(self_logdet),
        cross_logdet_residual_max=tuple(cross_logdet),
        reference_logdet_residual_max=tuple(reference_logdet),
        declared_logdet_residual_max=tuple(declared_logdet),
        score_finite=tuple(score_ok),
        conditioning_proxy=tuple(conditioning),
        maximum_condition_number=max_condition,
        finite_difference_relative_step=finite_difference_relative_step,
        physical_score_checked=True,
        failures=tuple(failures),
    )


@dataclass(frozen=True)
class ProperReplicaExchangeConfig:
    """Frozen temperature slots for a complete proper bridge."""

    betas: tuple[float, ...]
    bridge_signature: str
    alternating_parity: int = 0

    def __post_init__(self) -> None:
        betas = tuple(float(value) for value in self.betas)
        if len(betas) < 2 or any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in betas
        ):
            raise TemperedTransitionError("betas must be a finite ladder in [0,1]")
        if betas[0] != 0.0 or betas[-1] != 1.0:
            raise TemperedTransitionError(
                "replica-exchange ladder must start at beta=0 and end at beta=1"
            )
        # Adjacent slices intentionally have lengths n-1.
        if any(right <= left for left, right in zip(betas, betas[1:])):
            raise TemperedTransitionError("betas must be strictly increasing")
        if not str(self.bridge_signature):
            raise TemperedTransitionError("bridge_signature must be nonempty")
        parity = int(self.alternating_parity)
        if parity not in (0, 1):
            raise TemperedTransitionError("alternating_parity must be zero or one")
        object.__setattr__(self, "betas", betas)
        object.__setattr__(self, "bridge_signature", str(self.bridge_signature))
        object.__setattr__(self, "alternating_parity", parity)


def proper_swap_log_ratio(cross_values: Any, left: int, right: int) -> tf.Tensor:
    """Compute the exact adjacent-swap log ratio from complete bridge values."""
    values = tf.convert_to_tensor(cross_values, tf.float64)
    if values.shape.rank != 3 or values.shape[0] != values.shape[1]:
        raise TemperedTransitionError(
            "cross_values must have shape [target_temperature,source_temperature,chain]"
        )
    # The public shape is [target_temperature, source_temperature, chain].
    return (
        values[left, right]
        + values[right, left]
        - values[left, left]
        - values[right, right]
    )


def _gather_sources(values: tf.Tensor, sources: tf.Tensor) -> tf.Tensor:
    """Gather a per-temperature cache by its source slot.

    The first two axes are ``[temperature, chain]``.  Keeping the trailing
    axes generic lets the same operation carry scalar status fields, scores,
    and future per-row diagnostics without dropping cache entries.
    """
    rank = values.shape.rank
    if rank is None or rank < 2:
        raise TemperedTransitionError("values must have rank at least two")
    permutation = (1, 0, *range(2, rank))
    by_chain = tf.transpose(values, permutation)
    gathered = tf.gather(
        by_chain,
        tf.transpose(sources, (1, 0)),
        axis=1,
        batch_dims=1,
    )
    return tf.transpose(gathered, permutation)


def _gather_cross_sources(cross_values: tf.Tensor, sources: tf.Tensor) -> tf.Tensor:
    """Evaluate a destination-temperature cross cache on incoming sources.

    ``cross_values[target, source, chain, ...]`` contains the value of a
    source state under every destination temperature.  The returned cache is
    indexed by destination and therefore cannot be obtained by gathering the
    source state's old-temperature diagonal value.
    """
    rank = cross_values.shape.rank
    if rank is None or rank < 3:
        raise TemperedTransitionError("cross cache must have rank at least three")
    levels = cross_values.shape[0]
    if levels is None or cross_values.shape[1] != levels:
        raise TemperedTransitionError("cross cache temperature axes must be square")
    chains = sources.shape[1]
    if chains is None or cross_values.shape[2] != chains:
        raise TemperedTransitionError("cross cache chain axis does not match sources")
    # Move chain first, then gather the source index independently for each
    # destination/chain pair.  The target-temperature axis remains explicit.
    by_chain = tf.transpose(cross_values, (2, 0, 1, *range(3, rank)))
    source_by_chain = tf.transpose(sources, (1, 0))
    gathered = tf.gather(
        by_chain,
        source_by_chain,
        axis=2,
        batch_dims=1,
    )
    # gathered is [chain, target, ...]; restore [target, chain, ...].
    return tf.transpose(gathered, (1, 0, *range(2, gathered.shape.rank)))


def apply_proper_adjacent_swaps(
    state: Any,
    prior_values: Any,
    likelihood_values: Any,
    identities_at_temperature: Any,
    cross_values: Any,
    *,
    seed: Any,
    parity: int = 0,
    valid_at_temperature: Any | None = None,
    status_at_temperature: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Apply one non-overlapping adjacent swap wave using full bridge values."""
    states = tf.convert_to_tensor(state, tf.float64)
    if states.shape.rank != 3 or not states.shape.is_fully_defined():
        raise TemperedTransitionError("state must have static [temperature,chain,dimension]")
    levels, chains, dimension = (int(item) for item in states.shape)
    prior = tf.convert_to_tensor(prior_values, tf.float64)
    likelihood = tf.convert_to_tensor(likelihood_values, tf.float64)
    identities = tf.convert_to_tensor(identities_at_temperature, tf.int32)
    cross = tf.convert_to_tensor(cross_values, tf.float64)
    if prior.shape != (levels, chains) or likelihood.shape != (levels, chains):
        raise TemperedTransitionError("bridge component cache shape mismatch")
    if identities.shape != (levels, chains):
        raise TemperedTransitionError("identity shape mismatch")
    if cross.shape.rank != 3 or cross.shape[0] != levels or cross.shape[1] != levels or cross.shape[2] != chains:
        raise TemperedTransitionError("cross_values shape mismatch")
    parity_value = int(parity)
    if parity_value not in (0, 1):
        raise TemperedTransitionError("parity must be zero or one")
    seed_tensor = tf.convert_to_tensor(seed, tf.int32)
    if seed_tensor.shape != (2,):
        raise TemperedTransitionError("seed must have shape [2]")
    sources = tf.broadcast_to(tf.range(levels, dtype=tf.int32)[:, tf.newaxis], (levels, chains))
    proposed = tf.zeros((levels - 1, chains), tf.bool)
    accepted_rows = tf.zeros((levels - 1, chains), tf.bool)
    log_ratios = tf.fill((levels - 1, chains), tf.constant(float("-inf"), tf.float64))
    valid = (
        tf.ones((levels, chains), tf.bool)
        if valid_at_temperature is None
        else tf.cast(valid_at_temperature, tf.bool)
    )
    if valid.shape != (levels, chains):
        raise TemperedTransitionError("valid_at_temperature must have shape [temperature,chain]")
    status_cache: dict[str, tf.Tensor] = {}
    if status_at_temperature is not None:
        for name, value in status_at_temperature.items():
            tensor = tf.convert_to_tensor(value)
            if tensor.shape.rank is None or tensor.shape.rank < 2:
                raise TemperedTransitionError(
                    f"status field {name!r} must have leading [temperature,chain] axes"
                )
            if tensor.shape[0] != levels or tensor.shape[1] != chains:
                raise TemperedTransitionError(
                    f"status field {name!r} shape does not match [temperature,chain]"
                )
            status_cache[str(name)] = tensor
    uniform = tf.random.stateless_uniform((levels - 1, chains), seed_tensor, dtype=tf.float64)
    pairs = tuple((left, left + 1) for left in range(parity_value, levels - 1, 2))
    for left, right in pairs:
        ratio = proper_swap_log_ratio(cross, left, right)
        pair_valid = tf.logical_and(
            tf.logical_and(valid[left], valid[right]), tf.math.is_finite(ratio)
        )
        ratio = tf.where(pair_valid, ratio, tf.fill(tf.shape(ratio), tf.constant(float("-inf"), tf.float64)))
        accepted = tf.math.log(uniform[left]) < tf.minimum(tf.constant(0.0, tf.float64), ratio)
        sources = tf.tensor_scatter_nd_update(
            sources,
            [[left], [right]],
            [tf.where(accepted, tf.fill((chains,), right), sources[left]), tf.where(accepted, tf.fill((chains,), left), sources[right])],
        )
        proposed = tf.tensor_scatter_nd_update(proposed, [[left]], [tf.ones((chains,), tf.bool)])
        accepted_rows = tf.tensor_scatter_nd_update(accepted_rows, [[left]], [accepted])
        log_ratios = tf.tensor_scatter_nd_update(log_ratios, [[left]], [ratio])
    result: dict[str, tf.Tensor | Mapping[str, tf.Tensor]] = {
        "state": _gather_sources(states, sources),
        "prior_values": _gather_sources(prior, sources),
        "likelihood_values": _gather_sources(likelihood, sources),
        "identities_at_temperature": _gather_sources(identities, sources),
        "source_temperature_for_destination": sources,
        "swap_is_proposed_adjacent": proposed,
        "swap_is_accepted_adjacent": accepted_rows,
        "swap_log_accept_ratio_adjacent": log_ratios,
    }
    if status_cache:
        result["status_at_temperature"] = {
            name: _gather_sources(value, sources)
            for name, value in status_cache.items()
        }
    return result


class ProperBridgeReplicaExchange:
    """Replica exchange over a complete prior-likelihood bridge."""

    def __init__(self, bridge: Any, betas: Sequence[float]) -> None:
        if not callable(getattr(bridge, "component_terms", None)):
            raise TemperedTransitionError("bridge must expose component_terms")
        self.bridge = bridge
        self.config = ProperReplicaExchangeConfig(tuple(betas), str(bridge.signature))
        self.level_count = len(self.config.betas)
        self.parameter_dim = int(bridge.parameter_dim)

    def initial_identities(self, chain_count: int) -> tf.Tensor:
        count = int(chain_count)
        if count <= 0:
            raise TemperedTransitionError("chain_count must be positive")
        return tf.broadcast_to(
            tf.range(self.level_count, dtype=tf.int32)[:, tf.newaxis],
            (self.level_count, count),
        )

    def evaluate(self, state: Any) -> Mapping[str, Any]:
        values = tf.convert_to_tensor(state, tf.float64)
        if values.shape.rank != 3 or not values.shape.is_fully_defined():
            raise TemperedTransitionError("state must have static [temperature,chain,dimension]")
        levels, chains, dimension = (int(item) for item in values.shape)
        if levels != self.level_count or dimension != self.parameter_dim:
            raise TemperedTransitionError("state shape does not match bridge ladder")
        flattened = tf.reshape(values, [levels * chains, dimension])
        likelihood, likelihood_score, prior, prior_score, status = self.bridge.component_terms(flattened)
        likelihood = tf.reshape(likelihood, [levels, chains])
        prior = tf.reshape(prior, [levels, chains])
        likelihood_score = tf.reshape(likelihood_score, [levels, chains, dimension])
        prior_score = tf.reshape(prior_score, [levels, chains, dimension])
        status_at_temperature: dict[str, tf.Tensor] = {}
        for name, raw_value in status.items():
            tensor = tf.convert_to_tensor(raw_value)
            if tensor.shape.rank is not None and tensor.shape.rank >= 1:
                if tensor.shape[0] != levels * chains:
                    raise TemperedTransitionError(
                        f"status field {name!r} must have leading size levels*chains"
                    )
                status_at_temperature[str(name)] = tf.reshape(
                    tensor, [levels, chains, *tensor.shape.as_list()[1:]]
                )
            else:
                raise TemperedTransitionError(f"status field {name!r} has unknown rank")
        if "valid_pre_regularized_score" not in status_at_temperature:
            raise TemperedTransitionError(
                "bridge status must expose valid_pre_regularized_score"
            )
        valid = tf.cast(status_at_temperature["valid_pre_regularized_score"], tf.bool)
        if valid.shape != (levels, chains):
            raise TemperedTransitionError(
                "valid_pre_regularized_score must be one boolean per state row"
            )
        beta = tf.constant(self.config.betas, tf.float64)
        cross_values = prior[tf.newaxis, :, :] + beta[:, tf.newaxis, tf.newaxis] * likelihood[tf.newaxis, :, :]
        cross_scores = (
            prior_score[tf.newaxis, :, :, :]
            + beta[:, tf.newaxis, tf.newaxis, tf.newaxis] * likelihood_score[tf.newaxis, :, :, :]
        )
        values_at_slot = prior + beta[:, tf.newaxis] * likelihood
        scores_at_slot = prior_score + beta[:, tf.newaxis, tf.newaxis] * likelihood_score
        return {
            "prior_values": prior,
            "likelihood_values": likelihood,
            "prior_score": prior_score,
            "likelihood_score": likelihood_score,
            "values_at_temperature": values_at_slot,
            "scores_at_temperature": scores_at_slot,
            "cross_values": cross_values,
            "cross_scores": cross_scores,
            "valid_at_temperature": valid,
            "status": status_at_temperature,
            "raw_status": status,
            "status_at_temperature": status_at_temperature,
            "target_call_count": tf.constant(1, tf.int64),
        }

    def transition(
        self,
        state: Any,
        identities_at_temperature: Any,
        *,
        seed: Any,
        parity: int | None = None,
    ) -> Mapping[str, Any]:
        values = tf.convert_to_tensor(state, tf.float64)
        evaluated = self.evaluate(values)
        swaps = apply_proper_adjacent_swaps(
            values,
            evaluated["prior_values"],
            evaluated["likelihood_values"],
            identities_at_temperature,
            evaluated["cross_values"],
            seed=seed,
            parity=self.config.alternating_parity if parity is None else int(parity),
            valid_at_temperature=evaluated["valid_at_temperature"],
            status_at_temperature=evaluated["status_at_temperature"],
        )
        sources = swaps["source_temperature_for_destination"]
        post_prior = swaps["prior_values"]
        post_likelihood = swaps["likelihood_values"]
        beta = tf.constant(self.config.betas, tf.float64)
        post_values = post_prior + beta[:, tf.newaxis] * post_likelihood
        post_prior_score = _gather_sources(evaluated["prior_score"], sources)
        post_likelihood_score = _gather_sources(evaluated["likelihood_score"], sources)
        post_scores = post_prior_score + beta[:, tf.newaxis, tf.newaxis] * post_likelihood_score
        post_cross_values = (
            post_prior[tf.newaxis, :, :]
            + beta[:, tf.newaxis, tf.newaxis] * post_likelihood[tf.newaxis, :, :]
        )
        return {
            **swaps,
            "prior_score": post_prior_score,
            "likelihood_score": post_likelihood_score,
            "values_at_temperature": post_values,
            "scores_at_temperature": post_scores,
            "cross_values": post_cross_values,
            "cross_scores": (
                post_prior_score[tf.newaxis, :, :, :]
                + beta[:, tf.newaxis, tf.newaxis, tf.newaxis]
                * post_likelihood_score[tf.newaxis, :, :, :]
            ),
            "valid_at_temperature": _gather_sources(evaluated["valid_at_temperature"], sources),
            "status_at_temperature": swaps.get("status_at_temperature", {}),
            "status": swaps.get("status_at_temperature", {}),
            "cross_values_before_swap": evaluated["cross_values"],
            "target_call_count": evaluated["target_call_count"],
            "bridge_signature": self.config.bridge_signature,
        }

    def posterior_state(self, transition_result: Mapping[str, Any]) -> Mapping[str, Any]:
        """Expose only the beta-one slot as a posterior state boundary."""
        state = tf.convert_to_tensor(transition_result["state"], tf.float64)
        identities = tf.convert_to_tensor(
            transition_result["identities_at_temperature"], tf.int32
        )
        if state.shape.rank != 3 or state.shape[0] != self.level_count:
            raise TemperedTransitionError(
                "transition result does not match the bridge ladder"
            )
        if identities.shape != state.shape[:2]:
            raise TemperedTransitionError(
                "transition result replica identities do not match state"
            )
        return {
            "state": state[-1],
            "replica_identity": identities[-1],
            "beta": 1.0,
            "temperature_slot": self.level_count - 1,
            "bridge_signature": self.config.bridge_signature,
            "posterior_stream_only": True,
        }

    def posterior_stream(
        self,
        state_history: Any,
        identity_history: Any,
    ) -> Mapping[str, Any]:
        """Project a sequential history onto its beta-one stream only."""
        states = tf.convert_to_tensor(state_history, tf.float64)
        identities = tf.convert_to_tensor(identity_history, tf.int32)
        if states.shape.rank != 4 or states.shape[1] != self.level_count:
            raise TemperedTransitionError(
                "state_history must have [draw,temperature,chain,dimension]"
            )
        if identities.shape != states.shape[:3]:
            raise TemperedTransitionError(
                "identity_history must have [draw,temperature,chain]"
            )
        return {
            "samples": states[:, -1],
            "replica_identity": identities[:, -1],
            "beta": 1.0,
            "temperature_slot": self.level_count - 1,
            "bridge_signature": self.config.bridge_signature,
            "posterior_stream_only": True,
        }


@dataclass(frozen=True)
class BoundWithinTemperatureKernel:
    """Bind one exact physical-state kernel to one bridge temperature."""

    beta: float
    bridge_signature: str
    kernel_signature: str
    kernel: Callable[[tf.Tensor, tf.Tensor], Any]
    mechanics_role: str

    def __post_init__(self) -> None:
        beta = float(self.beta)
        if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
            raise TemperedTransitionError("bound kernel beta must lie in [0,1]")
        if not str(self.bridge_signature) or not str(self.kernel_signature):
            raise TemperedTransitionError(
                "bound kernel bridge and kernel signatures must be nonempty"
            )
        if not callable(self.kernel):
            raise TemperedTransitionError("bound within-temperature kernel is not callable")
        if not str(self.mechanics_role):
            raise TemperedTransitionError("bound kernel mechanics_role must be nonempty")
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "bridge_signature", str(self.bridge_signature))
        object.__setattr__(self, "kernel_signature", str(self.kernel_signature))
        object.__setattr__(self, "mechanics_role", str(self.mechanics_role))

    def transition(self, state: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        result = self.kernel(state, seed)
        if isinstance(result, Mapping):
            if "state" not in result:
                raise TemperedTransitionError(
                    "bound kernel mapping must contain its physical state"
                )
            result = result["state"]
        value = tf.convert_to_tensor(result, tf.float64)
        if value.shape != state.shape:
            raise TemperedTransitionError(
                "bound kernel must preserve [chain,dimension] shape"
            )
        return value

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.tempered.bound_within_temperature_kernel.v1",
            "beta": self.beta,
            "bridge_signature": self.bridge_signature,
            "kernel_signature": self.kernel_signature,
            "mechanics_role": self.mechanics_role,
        }


class ProperReplicaExchangeTransitionProgram:
    """Compose exact within-temperature kernels and proper adjacent swaps."""

    def __init__(
        self,
        exchange: ProperBridgeReplicaExchange,
        bindings: Sequence[BoundWithinTemperatureKernel],
        *,
        jit_compile: bool = True,
    ) -> None:
        if not isinstance(exchange, ProperBridgeReplicaExchange):
            raise TemperedTransitionError(
                "transition program requires ProperBridgeReplicaExchange"
            )
        bound = tuple(bindings)
        if len(bound) != exchange.level_count or any(
            not isinstance(item, BoundWithinTemperatureKernel) for item in bound
        ):
            raise TemperedTransitionError(
                "one bound within-temperature kernel is required per ladder level"
            )
        for beta, binding in zip(exchange.config.betas, bound, strict=True):
            if binding.beta != beta:
                raise TemperedTransitionError(
                    "bound within-temperature beta does not match the ladder"
                )
            if binding.bridge_signature != exchange.config.bridge_signature:
                raise TemperedTransitionError(
                    "bound kernel bridge signature does not match replica exchange"
                )
        self.exchange = exchange
        self.bindings = bound
        self.jit_compile = bool(jit_compile)
        self.transition_signature = _hash(
            {
                "schema": "bayesfilter.tempered.replica_exchange_transition_program.v1",
                "bridge_signature": exchange.config.bridge_signature,
                "betas": list(exchange.config.betas),
                "bindings": [item.payload() for item in bound],
                "jit_compile": self.jit_compile,
            }
        )
        self._compiled: dict[tuple[int, tuple[int, int, int]], Any] = {}

    def initial_state(
        self,
        state: Any,
        identities_at_temperature: Any | None = None,
    ) -> Mapping[str, Any]:
        values = tf.convert_to_tensor(state, tf.float64)
        if values.shape.rank != 3 or not values.shape.is_fully_defined():
            raise TemperedTransitionError(
                "initial replica state must have static [temperature,chain,dimension] shape"
            )
        levels, chains, dimension = (int(item) for item in values.shape)
        if levels != self.exchange.level_count or dimension != self.exchange.parameter_dim:
            raise TemperedTransitionError(
                "initial replica state does not match the exchange bridge"
            )
        identities = (
            self.exchange.initial_identities(chains)
            if identities_at_temperature is None
            else tf.convert_to_tensor(identities_at_temperature, tf.int32)
        )
        if identities.shape != (levels, chains):
            raise TemperedTransitionError(
                "initial identities must have [temperature,chain] shape"
            )
        return {
            "state": values,
            "identities_at_temperature": identities,
            "transition_index": tf.constant(0, tf.int64),
            "transition_signature": self.transition_signature,
        }

    def posterior_state(self, transition_state: Any) -> tf.Tensor:
        if not isinstance(transition_state, Mapping):
            raise TemperedTransitionError("transition state must be a mapping")
        values = tf.convert_to_tensor(transition_state.get("state"), tf.float64)
        if values.shape.rank != 3 or values.shape[0] != self.exchange.level_count:
            raise TemperedTransitionError(
                "transition state does not match the exchange ladder"
            )
        return values[-1]

    def _compiled_program(
        self,
        *,
        num_results: int,
        state_shape: tuple[int, int, int],
    ) -> Any:
        key = (int(num_results), tuple(state_shape))
        cached = self._compiled.get(key)
        if cached is not None:
            return cached
        levels, chains, dimension = state_shape
        if levels != self.exchange.level_count or dimension != self.exchange.parameter_dim:
            raise TemperedTransitionError(
                "compiled transition shape does not match replica exchange"
            )

        @tf.function(
            input_signature=(
                tf.TensorSpec(state_shape, tf.float64),
                tf.TensorSpec((levels, chains), tf.int32),
                tf.TensorSpec([], tf.int64),
                tf.TensorSpec([2], tf.int32),
            ),
            jit_compile=self.jit_compile,
            reduce_retracing=False,
        )
        def compiled(
            initial_state: tf.Tensor,
            initial_identities: tf.Tensor,
            initial_index: tf.Tensor,
            seed: tf.Tensor,
        ):
            cold_states = tf.TensorArray(
                tf.float64,
                size=int(num_results),
                element_shape=(chains, dimension),
                clear_after_read=False,
            )
            cold_identities = tf.TensorArray(
                tf.int32,
                size=int(num_results),
                element_shape=(chains,),
                clear_after_read=False,
            )

            def condition(index, *_args):
                return index < int(num_results)

            def body(
                index,
                current_state,
                current_identities,
                all_finite,
                all_target_valid,
                proposed_count,
                accepted_count,
                cold_state_array,
                cold_identity_array,
            ):
                step_seed = tf.random.experimental.stateless_fold_in(seed, index)
                next_levels = []
                for level, binding in enumerate(self.bindings):
                    kernel_seed = tf.random.experimental.stateless_fold_in(
                        step_seed, 1000 + level
                    )
                    next_levels.append(
                        binding.transition(current_state[level], kernel_seed)
                    )
                within_state = tf.stack(next_levels, axis=0)
                within_finite = tf.reduce_all(tf.math.is_finite(within_state))
                swap_seed = tf.random.experimental.stateless_fold_in(step_seed, 2000)
                parity = tf.cast(
                    tf.math.floormod(initial_index + tf.cast(index, tf.int64), 2),
                    tf.int32,
                )

                def apply_wave(parity_value: int):
                    swapped = self.exchange.transition(
                        within_state,
                        current_identities,
                        seed=swap_seed,
                        parity=parity_value,
                    )
                    status = swapped.get("status_at_temperature", {})
                    status_valid = tf.reduce_all(swapped["valid_at_temperature"])
                    if "status_code" in status:
                        status_valid = tf.logical_and(
                            status_valid,
                            tf.reduce_all(
                                tf.equal(
                                    tf.convert_to_tensor(status["status_code"], tf.int32),
                                    0,
                                )
                            ),
                        )
                    return (
                        tf.convert_to_tensor(swapped["state"], tf.float64),
                        tf.convert_to_tensor(
                            swapped["identities_at_temperature"], tf.int32
                        ),
                        tf.cast(status_valid, tf.bool),
                        tf.reduce_sum(
                            tf.cast(swapped["swap_is_proposed_adjacent"], tf.int64)
                        ),
                        tf.reduce_sum(
                            tf.cast(swapped["swap_is_accepted_adjacent"], tf.int64)
                        ),
                    )

                (
                    swapped_state,
                    swapped_identities,
                    step_target_valid,
                    step_proposed,
                    step_accepted,
                ) = tf.cond(
                    tf.equal(parity, 0),
                    lambda: apply_wave(0),
                    lambda: apply_wave(1),
                )
                step_finite = tf.logical_and(
                    within_finite, tf.reduce_all(tf.math.is_finite(swapped_state))
                )
                return (
                    index + 1,
                    swapped_state,
                    swapped_identities,
                    tf.logical_and(all_finite, step_finite),
                    tf.logical_and(all_target_valid, step_target_valid),
                    proposed_count + step_proposed,
                    accepted_count + step_accepted,
                    cold_state_array.write(index, swapped_state[-1]),
                    cold_identity_array.write(index, swapped_identities[-1]),
                )

            outputs = tf.while_loop(
                condition,
                body,
                loop_vars=(
                    tf.constant(0, tf.int32),
                    initial_state,
                    initial_identities,
                    tf.constant(True),
                    tf.constant(True),
                    tf.constant(0, tf.int64),
                    tf.constant(0, tf.int64),
                    cold_states,
                    cold_identities,
                ),
                parallel_iterations=1,
            )
            return (
                outputs[1],
                outputs[2],
                initial_index + tf.cast(num_results, tf.int64),
                outputs[7].stack(),
                outputs[8].stack(),
                outputs[3],
                outputs[4],
                outputs[5],
                outputs[6],
            )

        self._compiled[key] = compiled
        return compiled

    def __call__(
        self,
        transition_state: Any,
        *,
        num_results: int,
        seed: Any,
        stage: str,
    ) -> Mapping[str, Any]:
        if stage not in {"warmup", "retained"}:
            raise TemperedTransitionError("stage must be warmup or retained")
        count = int(num_results)
        if count <= 0:
            raise TemperedTransitionError("num_results must be positive")
        if not isinstance(transition_state, Mapping):
            raise TemperedTransitionError("transition state must be a mapping")
        if transition_state.get("transition_signature") != self.transition_signature:
            raise TemperedTransitionError("transition continuation signature mismatch")
        state = tf.convert_to_tensor(transition_state.get("state"), tf.float64)
        identities = tf.convert_to_tensor(
            transition_state.get("identities_at_temperature"), tf.int32
        )
        transition_index = tf.convert_to_tensor(
            transition_state.get("transition_index"), tf.int64
        )
        seed_tensor = tf.convert_to_tensor(seed, tf.int32)
        if state.shape.rank != 3 or not state.shape.is_fully_defined():
            raise TemperedTransitionError(
                "transition state must have static [temperature,chain,dimension] shape"
            )
        state_shape = tuple(int(item) for item in state.shape)
        if identities.shape != state.shape[:2] or transition_index.shape.rank != 0:
            raise TemperedTransitionError("transition continuation cache shape mismatch")
        if seed_tensor.shape != (2,):
            raise TemperedTransitionError("seed must have shape [2]")
        compiled = self._compiled_program(
            num_results=count, state_shape=state_shape
        )
        (
            final_state,
            final_identities,
            final_index,
            cold_samples,
            cold_identities,
            all_finite,
            all_target_valid,
            proposed_count,
            accepted_count,
        ) = compiled(state, identities, transition_index, seed_tensor)
        finite = bool(all_finite.numpy())
        target_valid = bool(all_target_valid.numpy())
        hard_vetoes = []
        if not finite:
            hard_vetoes.append("replica_exchange_state_nonfinite")
        if not target_valid:
            hard_vetoes.append("replica_exchange_target_status_invalid")
        return {
            "transition_signature": self.transition_signature,
            "posterior_stream_only": True,
            "posterior_temperature": 1.0,
            "posterior_samples": cold_samples,
            "posterior_replica_identities": cold_identities,
            "final_transition_state": {
                "state": final_state,
                "identities_at_temperature": final_identities,
                "transition_index": final_index,
                "transition_signature": self.transition_signature,
            },
            "health": {
                "passed": bool(finite and target_valid),
                "hard_vetoes": tuple(hard_vetoes),
                "all_states_finite": finite,
                "all_target_status_valid": target_valid,
                "swap_proposal_count": int(proposed_count.numpy()),
                "swap_accept_count": int(accepted_count.numpy()),
                "jit_compile": self.jit_compile,
                "compiled_tracing_count": int(
                    compiled.experimental_get_tracing_count()
                ),
            },
        }


__all__ = [
    "BoundWithinTemperatureKernel",
    "FixedChartKernelMixture",
    "FixedChartSelection",
    "ProperBridgeReplicaExchange",
    "ProperReplicaExchangeTransitionProgram",
    "ProperReplicaExchangeConfig",
    "TRANSITION_NONCLAIMS",
    "TemperedTransitionError",
    "TransportReliabilityReceipt",
    "apply_proper_adjacent_swaps",
    "build_fixed_transport_hmc_kernel",
    "build_tuned_fixed_transport_hmc_kernel",
    "derive_reliability_tolerance",
    "proper_swap_log_ratio",
    "screen_transport_reliability",
]
