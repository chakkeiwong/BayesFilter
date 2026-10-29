"""Shared small model contracts for BayesFilter/FilterFlow tie-outs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter import highdim
from experiments.dpf_implementation.tf_tfp.fixtures.structural_ar1_quadratic_tf import (
    build_structural_ar1_quadratic_fixture_tf,
    complete_k_tf,
    structural_observation_mean_tf,
)


tfd = tfp.distributions
DTYPE = tf.float64
PI = tf.constant(3.141592653589793, dtype=DTYPE)


@dataclass(frozen=True)
class CommonModelSpec:
    """Declarative small model fixture shared across implementation adapters."""

    model_id: str
    family: str
    parameters: Mapping[str, Any]
    theta: tf.Tensor
    x0: tf.Tensor
    x_prev: tf.Tensor
    x_next: tf.Tensor
    x_obs: tf.Tensor
    observation: tf.Tensor

    def payload(self) -> dict[str, Any]:
        payload = {
            "model_id": self.model_id,
            "family": self.family,
            "parameters": _jsonable(self.parameters),
            "theta": _tensor_to_json(self.theta),
            "x0": _tensor_to_json(self.x0),
            "x_prev": _tensor_to_json(self.x_prev),
            "x_next": _tensor_to_json(self.x_next),
            "x_obs": _tensor_to_json(self.x_obs),
            "observation": _tensor_to_json(self.observation),
            "dtype": DTYPE.name,
        }
        payload["checksum"] = stable_digest(payload)
        return payload

    def checksum(self) -> str:
        return str(self.payload()["checksum"])


@dataclass(frozen=True)
class CommonModelSpecV2:
    """Declarative v2 production fixture row.

    V2 is intentionally separate from ``CommonModelSpec`` so closed v1 artifacts
    keep their original three-row semantics.
    """

    model_id: str
    family: str
    source_surface: str
    successor_of: tuple[str, ...]
    parameters: Mapping[str, Any]
    theta: tf.Tensor
    x0: tf.Tensor
    x_prev: tf.Tensor
    x_next: tf.Tensor
    x_obs: tf.Tensor
    observation: tf.Tensor
    observations: tf.Tensor
    path_contract: Mapping[str, Any]
    fixed_ancestor_contract: Mapping[str, Any]
    gradient_contract: Mapping[str, Any]
    tolerances: Mapping[str, Any]
    phase_readiness: Mapping[str, Any]
    adapter_certification: Mapping[str, Any]
    non_claims: tuple[str, ...]

    def payload(self) -> dict[str, Any]:
        payload = {
            "model_id": self.model_id,
            "family": self.family,
            "source_surface": self.source_surface,
            "successor_of": list(self.successor_of),
            "parameters": _jsonable(self.parameters),
            "theta": _tensor_to_json(self.theta),
            "density_probes": {
                "x0": _tensor_to_json(self.x0),
                "x_prev": _tensor_to_json(self.x_prev),
                "x_next": _tensor_to_json(self.x_next),
                "x_obs": _tensor_to_json(self.x_obs),
                "observation": _tensor_to_json(self.observation),
            },
            "observations": _tensor_to_json(self.observations),
            "path_contract": _jsonable(self.path_contract),
            "fixed_ancestor_contract": _jsonable(self.fixed_ancestor_contract),
            "gradient_contract": _jsonable(self.gradient_contract),
            "tolerances": _jsonable(self.tolerances),
            "phase_readiness": _jsonable(self.phase_readiness),
            "adapter_certification": _jsonable(self.adapter_certification),
            "non_claims": list(self.non_claims),
            "dtype": DTYPE.name,
        }
        payload["checksum"] = stable_digest(payload)
        return payload

    def checksum(self) -> str:
        return str(self.payload()["checksum"])

    @property
    def state_dim(self) -> int:
        return int(self.x0.shape[1])

    @property
    def observation_dim(self) -> int:
        return int(self.observation.shape[0])


@dataclass(frozen=True)
class LGSSMTFFixture:
    name: str
    A: tf.Tensor
    C: tf.Tensor
    Q: tf.Tensor
    R: tf.Tensor
    m0: tf.Tensor
    P0: tf.Tensor
    states: tf.Tensor
    observations: tf.Tensor
    fixture_generation_seed: int
    model_checksum: str
    observation_checksum: str

    @property
    def state_dim(self) -> int:
        return int(self.A.shape[0])

    @property
    def obs_dim(self) -> int:
        return int(self.C.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.observations.shape[0])


@dataclass(frozen=True)
class StochasticVolatilityTFFixture:
    name: str
    mu: tf.Tensor
    phi: tf.Tensor
    sigma: tf.Tensor
    h0_mean: tf.Tensor
    h0_variance: tf.Tensor
    states: tf.Tensor
    observations: tf.Tensor
    fixture_generation_seed: int
    model_checksum: str
    observation_checksum: str

    @property
    def horizon(self) -> int:
        return int(self.observations.shape[0])


@dataclass(frozen=True)
class RangeBearingTFFixture:
    name: str
    A: tf.Tensor
    Q: tf.Tensor
    R: tf.Tensor
    m0: tf.Tensor
    P0: tf.Tensor
    states: tf.Tensor
    observations: tf.Tensor
    dt: float
    fixture_generation_seed: int
    model_checksum: str
    observation_checksum: str

    @property
    def state_dim(self) -> int:
        return int(self.A.shape[0])

    @property
    def obs_dim(self) -> int:
        return int(self.R.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.observations.shape[0])


@dataclass(frozen=True)
class CommonRangeBearingSSM:
    """Highdim-protocol adapter for the common range-bearing contract."""

    initial_mean: tf.Tensor
    initial_covariance: tf.Tensor
    transition_matrix: tf.Tensor
    transition_covariance: tf.Tensor
    observation_covariance: tf.Tensor

    def __post_init__(self) -> None:
        object.__setattr__(self, "initial_mean", tf.convert_to_tensor(self.initial_mean, DTYPE))
        object.__setattr__(
            self,
            "initial_covariance",
            tf.convert_to_tensor(self.initial_covariance, DTYPE),
        )
        object.__setattr__(
            self,
            "transition_matrix",
            tf.convert_to_tensor(self.transition_matrix, DTYPE),
        )
        object.__setattr__(
            self,
            "transition_covariance",
            tf.convert_to_tensor(self.transition_covariance, DTYPE),
        )
        object.__setattr__(
            self,
            "observation_covariance",
            tf.convert_to_tensor(self.observation_covariance, DTYPE),
        )

    def parameter_dim(self) -> int:
        return 0

    def state_dim(self) -> int:
        return int(self.initial_mean.shape[0])

    def observation_dim(self) -> int:
        return int(self.observation_covariance.shape[0])

    def initial_log_density(self, theta: tf.Tensor, x0: tf.Tensor) -> tf.Tensor:
        del theta
        values = _as_row_matrix(x0, self.state_dim(), "x0")
        return _mvn_log_prob(values, self.initial_mean, self.initial_covariance)

    def transition_log_density(
        self,
        theta: tf.Tensor,
        x_prev: tf.Tensor,
        x_next: tf.Tensor,
        t: int,
    ) -> tf.Tensor:
        del theta, t
        previous = _as_row_matrix(x_prev, self.state_dim(), "x_prev")
        next_values = _as_row_matrix(x_next, self.state_dim(), "x_next")
        loc = tf.linalg.matmul(previous, self.transition_matrix, transpose_b=True)
        return _mvn_log_prob(next_values, loc, self.transition_covariance)

    def observation_log_density(
        self,
        theta: tf.Tensor,
        x_t: tf.Tensor,
        y_t: tf.Tensor,
        t: int,
    ) -> tf.Tensor:
        del theta, t
        values = _as_row_matrix(x_t, self.state_dim(), "x_t")
        observation = tf.reshape(tf.convert_to_tensor(y_t, DTYPE), [self.observation_dim()])
        predicted = range_bearing_observation_tf(values)
        residual = observation_residual_tf(predicted, observation)
        return gaussian_logpdf_zero_mean_tf(residual, self.observation_covariance)

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "CommonRangeBearingSSM",
            "initial_mean": self.initial_mean,
            "initial_covariance": self.initial_covariance,
            "transition_matrix": self.transition_matrix,
            "transition_covariance": self.transition_covariance,
            "observation_covariance": self.observation_covariance,
            "dimension_convention": "state=(px,py,vx,vy), observation=(range,bearing)",
        }


@dataclass(frozen=True)
class CommonSVRichSSM:
    """AR(1) stochastic-volatility row used by v2 common tie-outs."""

    mu: tf.Tensor
    phi: tf.Tensor
    sigma: tf.Tensor
    h0_mean: tf.Tensor
    h0_variance: tf.Tensor

    def __post_init__(self) -> None:
        for name in ("mu", "phi", "sigma", "h0_mean", "h0_variance"):
            value = tf.convert_to_tensor(getattr(self, name), DTYPE)
            object.__setattr__(self, name, value)
        if not bool(
            tf.math.is_finite(self.mu).numpy()
            and tf.math.is_finite(self.phi).numpy()
            and tf.math.is_finite(self.sigma).numpy()
            and tf.math.is_finite(self.h0_mean).numpy()
            and tf.math.is_finite(self.h0_variance).numpy()
            and (self.sigma > 0.0).numpy()
            and (self.h0_variance > 0.0).numpy()
        ):
            raise ValueError("CommonSVRichSSM received invalid parameters")

    def parameter_dim(self) -> int:
        return 0

    def state_dim(self) -> int:
        return 1

    def observation_dim(self) -> int:
        return 1

    def initial_log_density(self, theta: tf.Tensor, x0: tf.Tensor) -> tf.Tensor:
        del theta
        values = _as_row_matrix(x0, self.state_dim(), "x0")
        return _normal_log_prob(values[:, 0], self.h0_mean, tf.sqrt(self.h0_variance))

    def transition_mean(self, x_prev: tf.Tensor) -> tf.Tensor:
        previous = _as_row_matrix(x_prev, self.state_dim(), "x_prev")
        return self.mu + self.phi * (previous - self.mu)

    def transition_log_density(
        self,
        theta: tf.Tensor,
        x_prev: tf.Tensor,
        x_next: tf.Tensor,
        t: int,
    ) -> tf.Tensor:
        del theta, t
        next_values = _as_row_matrix(x_next, self.state_dim(), "x_next")
        return _normal_log_prob(next_values[:, 0], self.transition_mean(x_prev)[:, 0], self.sigma)

    def observation_log_density(
        self,
        theta: tf.Tensor,
        x_t: tf.Tensor,
        y_t: tf.Tensor,
        t: int,
    ) -> tf.Tensor:
        del theta, t
        values = _as_row_matrix(x_t, self.state_dim(), "x_t")
        observation = tf.reshape(tf.convert_to_tensor(y_t, DTYPE), [self.observation_dim()])
        scale = tf.exp(0.5 * values[:, 0])
        return _normal_log_prob(tf.broadcast_to(observation[0], tf.shape(scale)), 0.0, scale)

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "CommonSVRichSSM",
            "state_equation": "h_t = mu + phi (h_{t-1} - mu) + sigma eta_t",
            "observation_equation": "y_t | h_t ~ Normal(0, exp(h_t))",
            "mu": self.mu,
            "phi": self.phi,
            "sigma": self.sigma,
            "h0_mean": self.h0_mean,
            "h0_variance": self.h0_variance,
        }


@dataclass(frozen=True)
class CommonStructuralAR1QuadraticSSM:
    """Structural AR(1) split-state row with deterministic completion.

    The stochastic density terms live on the AR(1) coordinate ``m``.  The
    completion coordinate ``k`` is carried in the state and deterministically
    recomputed on path transitions.  This is a declared singular-completion
    contract for common tie-outs, not a full Lebesgue density on ``(m,k)``.
    """

    rho: tf.Tensor
    sigma: tf.Tensor
    a: tf.Tensor
    b: tf.Tensor
    c: tf.Tensor
    d: tf.Tensor
    lam: tf.Tensor
    observation_scale: tf.Tensor
    m0_mean: tf.Tensor
    m0_variance: tf.Tensor
    k0: tf.Tensor

    def __post_init__(self) -> None:
        for name in (
            "rho",
            "sigma",
            "a",
            "b",
            "c",
            "d",
            "lam",
            "observation_scale",
            "m0_mean",
            "m0_variance",
            "k0",
        ):
            object.__setattr__(self, name, tf.convert_to_tensor(getattr(self, name), DTYPE))

    def parameter_dim(self) -> int:
        return 0

    def state_dim(self) -> int:
        return 2

    def observation_dim(self) -> int:
        return 1

    def initial_log_density(self, theta: tf.Tensor, x0: tf.Tensor) -> tf.Tensor:
        del theta
        values = _as_row_matrix(x0, self.state_dim(), "x0")
        return _normal_log_prob(values[:, 0], self.m0_mean, tf.sqrt(self.m0_variance))

    def transition_mean(self, x_prev: tf.Tensor) -> tf.Tensor:
        previous = _as_row_matrix(x_prev, self.state_dim(), "x_prev")
        next_m = self.rho * previous[:, 0]
        next_k = complete_k_tf(
            previous_k=previous[:, 1],
            previous_m=previous[:, 0],
            current_m=next_m,
            a=self.a,
            b=self.b,
            c=self.c,
            d=self.d,
        )
        return tf.stack([next_m, next_k], axis=1)

    def complete_next_state(self, x_prev: tf.Tensor, current_m: tf.Tensor) -> tf.Tensor:
        previous = _as_row_matrix(x_prev, self.state_dim(), "x_prev")
        current_m = tf.reshape(tf.cast(current_m, DTYPE), [-1])
        current_k = complete_k_tf(
            previous_k=previous[:, 1],
            previous_m=previous[:, 0],
            current_m=current_m,
            a=self.a,
            b=self.b,
            c=self.c,
            d=self.d,
        )
        return tf.stack([current_m, current_k], axis=1)

    def transition_log_density(
        self,
        theta: tf.Tensor,
        x_prev: tf.Tensor,
        x_next: tf.Tensor,
        t: int,
    ) -> tf.Tensor:
        del theta, t
        previous = _as_row_matrix(x_prev, self.state_dim(), "x_prev")
        next_values = _as_row_matrix(x_next, self.state_dim(), "x_next")
        mean_m = self.rho * previous[:, 0]
        return _normal_log_prob(next_values[:, 0], mean_m, self.sigma)

    def observation_log_density(
        self,
        theta: tf.Tensor,
        x_t: tf.Tensor,
        y_t: tf.Tensor,
        t: int,
    ) -> tf.Tensor:
        del theta, t
        values = _as_row_matrix(x_t, self.state_dim(), "x_t")
        observation = tf.reshape(tf.convert_to_tensor(y_t, DTYPE), [self.observation_dim()])
        mean = structural_observation_mean_tf(values, self.lam)
        return _normal_log_prob(tf.broadcast_to(observation[0], tf.shape(mean)), mean, self.observation_scale)

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "CommonStructuralAR1QuadraticSSM",
            "state_convention": "state=(m,k); m is stochastic, k is deterministic completion",
            "density_convention": "singular_completion_density_on_m_with_k_carried_in_state",
            "rho": self.rho,
            "sigma": self.sigma,
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "d": self.d,
            "lambda": self.lam,
            "observation_scale": self.observation_scale,
            "m0_mean": self.m0_mean,
            "m0_variance": self.m0_variance,
            "k0": self.k0,
        }


def common_model_specs() -> tuple[CommonModelSpec, ...]:
    """Return the initial common BayesFilter/FilterFlow model suite."""

    return (
        common_lgssm_spec(),
        common_sv_spec(),
        common_range_bearing_spec(),
    )


def common_model_specs_v2() -> tuple[CommonModelSpecV2, ...]:
    """Return the six-row v2 production common model suite."""

    specs = (
        _common_lgssm_v2_spec(),
        _common_sv_v2_spec(),
        _common_range_bearing_v2_spec(),
        _common_structural_ar1_v2_spec(),
        _common_spatial_sir_v2_spec(),
        _common_predator_prey_v2_spec(),
    )
    _assert_exact_v2_model_ids([spec.model_id for spec in specs])
    return specs


def common_model_suite_v2_manifest() -> dict[str, Any]:
    specs = common_model_specs_v2()
    rows = [spec.payload() for spec in specs]
    payload = {
        "artifact_id": "dpf_common_model_suite_v2_manifest_2026-06-07",
        "suite_id": "dpf_common_model_suite_v2_production",
        "version": "2026-06-07.p1",
        "status": "P1_DECLARATIVE_SPEC_MANIFEST",
        "dtype": DTYPE.name,
        "declared_v2_model_ids_exact": list(EXPECTED_V2_MODEL_IDS),
        "row_count_gate": {"required_count": len(EXPECTED_V2_MODEL_IDS), "actual_count": len(rows)},
        "rows": rows,
        "pre_run_row_classification_table": [
            {
                "model_id": spec.model_id,
                "P2_density": spec.phase_readiness["P2_density"],
                "P3_noresampling": spec.phase_readiness["P3_noresampling"],
                "P4_fixed_ancestor": spec.phase_readiness["P4_fixed_ancestor"],
                "P5_gradients": spec.phase_readiness["P5_gradients"],
                "classification_frozen_before_results": True,
                "classification_reason": spec.phase_readiness["reason"],
            }
            for spec in specs
        ],
        "primary_criterion_fields": {
            "exact_six_row_gate": [spec.model_id for spec in specs],
            "v1_api_preserved": True,
            "v2_source_api": "common_model_specs_v2",
            "old_v1_api_not_v2_source": True,
        },
        "veto_diagnostics": {
            "old_three_row_common_model_specs_used_as_v2_source": False,
            "old_2026_06_06_artifact_name_used": False,
            "student_command_executed": False,
            "localsource_filterflow_mutated": False,
            "sir_predator_prey_adapter_semantics_declared_before_execution": True,
        },
        "explanatory_only_fields": {
            "v1_model_ids": [spec.model_id for spec in common_model_specs()],
            "v1_checksums": {spec.model_id: spec.checksum() for spec in common_model_specs()},
            "adapter_note": (
                "FilterFlow-side v2 execution is expected to use CPU-only local "
                "subprocess adapters without mutating .localsource/filterflow."
            ),
        },
        "review_round": 0,
        "open_material_blockers": [],
        "repair_amendment_required": False,
        "next_allowed_action": "run P2 density tie-out after Claude P1 result PASS",
        "non_claims": [
            "P1 declares contracts only; no BF/FF value agreement claim",
            "no filter correctness proof",
            "no student implementation claim",
            "no TT/SIRT or paper-scale Zhao--Cui reproduction claim",
        ],
    }
    payload["checksum"] = stable_digest(payload)
    validate_common_model_suite_v2_manifest(payload)
    return payload


def validate_common_model_suite_v2_manifest(payload: Mapping[str, Any]) -> None:
    rows = list(payload.get("rows", []))
    ids = [row.get("model_id") for row in rows]
    _assert_exact_v2_model_ids(ids)
    if payload.get("row_count_gate", {}).get("actual_count") != len(EXPECTED_V2_MODEL_IDS):
        raise ValueError("v2 manifest row-count gate failed")
    required = {
        "primary_criterion_fields",
        "veto_diagnostics",
        "explanatory_only_fields",
        "review_round",
        "open_material_blockers",
        "repair_amendment_required",
        "next_allowed_action",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"v2 manifest missing required fields: {sorted(missing)}")
    for row in rows:
        for field in (
            "density_probes",
            "path_contract",
            "fixed_ancestor_contract",
            "gradient_contract",
            "tolerances",
            "phase_readiness",
            "adapter_certification",
            "checksum",
        ):
            if field not in row:
                raise ValueError(f"{row.get('model_id')}: missing {field}")


def common_lgssm_spec() -> CommonModelSpec:
    a = tf.constant([[0.92, 0.18], [-0.04, 0.86]], dtype=DTYPE)
    c = tf.constant([[1.0, 0.35]], dtype=DTYPE)
    q = tf.constant([[0.08, 0.015], [0.015, 0.05]], dtype=DTYPE)
    r = tf.constant([[0.18]], dtype=DTYPE)
    m0 = tf.constant([0.25, -0.15], dtype=DTYPE)
    p0 = tf.constant([[0.45, 0.04], [0.04, 0.32]], dtype=DTYPE)
    return CommonModelSpec(
        model_id="lgssm_2d_linear",
        family="linear_gaussian",
        parameters={"A": a, "C": c, "Q": q, "R": r, "m0": m0, "P0": p0},
        theta=tf.zeros([0], dtype=DTYPE),
        x0=tf.constant([[0.1, -0.2], [0.55, 0.05], [-0.35, 0.4]], dtype=DTYPE),
        x_prev=tf.constant([[0.1, -0.2], [0.55, 0.05], [-0.35, 0.4]], dtype=DTYPE),
        x_next=tf.constant([[0.0, -0.1], [0.42, 0.2], [-0.25, 0.25]], dtype=DTYPE),
        x_obs=tf.constant([[0.0, -0.1], [0.42, 0.2], [-0.25, 0.25]], dtype=DTYPE),
        observation=tf.constant([0.17], dtype=DTYPE),
    )


def common_sv_spec() -> CommonModelSpec:
    model = highdim.StochasticVolatilitySSM(sigma=1.0)
    theta = model.unconstrained_from_physical(gamma=0.6, beta=0.4)
    return CommonModelSpec(
        model_id="sv_1d_synthetic",
        family="stochastic_volatility",
        parameters={"gamma": 0.6, "beta": 0.4, "sigma": 1.0, "mu": 0.0},
        theta=theta,
        x0=tf.constant([[-0.8], [0.0], [0.9]], dtype=DTYPE),
        x_prev=tf.constant([[-1.1], [-0.2], [0.35], [1.25]], dtype=DTYPE),
        x_next=tf.constant([[-0.75], [0.08], [0.42], [0.95]], dtype=DTYPE),
        x_obs=tf.constant([[-0.9], [-0.1], [0.45], [1.3]], dtype=DTYPE),
        observation=tf.constant([0.17], dtype=DTYPE),
    )


def common_range_bearing_spec() -> CommonModelSpec:
    dt = 0.1
    a = tf.constant(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=DTYPE,
    )
    q = tf.linalg.diag(tf.constant([0.0015, 0.0015, 0.0008, 0.0008], dtype=DTYPE))
    r = tf.linalg.diag(tf.constant([0.12 * 0.12, 0.04 * 0.04], dtype=DTYPE))
    m0 = tf.constant([1.2, 0.7, 0.18, -0.06], dtype=DTYPE)
    p0 = tf.linalg.diag(tf.constant([0.04, 0.04, 0.01, 0.01], dtype=DTYPE))
    return CommonModelSpec(
        model_id="range_bearing_2d_cv",
        family="range_bearing",
        parameters={"A": a, "Q": q, "R": r, "m0": m0, "P0": p0},
        theta=tf.zeros([0], dtype=DTYPE),
        x0=tf.constant(
            [[1.10, 0.64, 0.15, -0.03], [1.35, 0.78, 0.2, -0.08], [0.95, 0.5, 0.1, 0.02]],
            dtype=DTYPE,
        ),
        x_prev=tf.constant(
            [[1.10, 0.64, 0.15, -0.03], [1.35, 0.78, 0.2, -0.08], [0.95, 0.5, 0.1, 0.02]],
            dtype=DTYPE,
        ),
        x_next=tf.constant(
            [[1.13, 0.62, 0.16, -0.04], [1.38, 0.77, 0.18, -0.07], [0.97, 0.51, 0.11, 0.00]],
            dtype=DTYPE,
        ),
        x_obs=tf.constant(
            [[1.13, 0.62, 0.16, -0.04], [1.38, 0.77, 0.18, -0.07], [0.97, 0.51, 0.11, 0.00]],
            dtype=DTYPE,
        ),
        observation=tf.constant([1.29, 0.50], dtype=DTYPE),
    )


def bayesfilter_model_for_spec(spec: CommonModelSpec):
    if spec.model_id == "lgssm_2d_linear":
        return highdim.LinearGaussianSSM(
            initial_mean=spec.parameters["m0"],
            initial_covariance=spec.parameters["P0"],
            transition_matrix=spec.parameters["A"],
            transition_covariance=spec.parameters["Q"],
            observation_matrix=spec.parameters["C"],
            observation_covariance=spec.parameters["R"],
        )
    if spec.model_id == "sv_1d_synthetic":
        return highdim.StochasticVolatilitySSM(sigma=spec.parameters["sigma"])
    if spec.model_id == "range_bearing_2d_cv":
        return CommonRangeBearingSSM(
            initial_mean=spec.parameters["m0"],
            initial_covariance=spec.parameters["P0"],
            transition_matrix=spec.parameters["A"],
            transition_covariance=spec.parameters["Q"],
            observation_covariance=spec.parameters["R"],
        )
    raise ValueError(f"unknown common model spec: {spec.model_id}")


def bayesfilter_model_for_spec_v2(spec: CommonModelSpecV2):
    if spec.model_id == "lgssm_2d_h25_rich":
        return highdim.LinearGaussianSSM(
            initial_mean=spec.parameters["m0"],
            initial_covariance=spec.parameters["P0"],
            transition_matrix=spec.parameters["A"],
            transition_covariance=spec.parameters["Q"],
            observation_matrix=spec.parameters["C"],
            observation_covariance=spec.parameters["R"],
        )
    if spec.model_id == "sv_1d_h18_rich":
        return CommonSVRichSSM(
            mu=spec.parameters["mu"],
            phi=spec.parameters["phi"],
            sigma=spec.parameters["sigma"],
            h0_mean=spec.parameters["h0_mean"],
            h0_variance=spec.parameters["h0_variance"],
        )
    if spec.model_id == "range_bearing_4d_h20_rich":
        return CommonRangeBearingSSM(
            initial_mean=spec.parameters["m0"],
            initial_covariance=spec.parameters["P0"],
            transition_matrix=spec.parameters["A"],
            transition_covariance=spec.parameters["Q"],
            observation_covariance=spec.parameters["R"],
        )
    if spec.model_id == "structural_ar1_quadratic_h16":
        return CommonStructuralAR1QuadraticSSM(
            rho=spec.parameters["rho"],
            sigma=spec.parameters["sigma"],
            a=spec.parameters["a"],
            b=spec.parameters["b"],
            c=spec.parameters["c"],
            d=spec.parameters["d"],
            lam=spec.parameters["lambda"],
            observation_scale=spec.parameters["observation_scale"],
            m0_mean=spec.parameters["m0_mean"],
            m0_variance=spec.parameters["m0_variance"],
            k0=spec.parameters["k0"],
        )
    if spec.model_id == "spatial_sir_j3_rk4":
        return highdim.SpatialSIRSSM(
            kappa=spec.parameters["kappa"],
            nu=spec.parameters["nu"],
            initial_mean=spec.parameters["initial_mean"],
            neighbor_sets=tuple(tuple(row) for row in spec.parameters["neighbor_sets"]),
            delta=spec.parameters["delta"],
            rk4_internal_step=spec.parameters["rk4_internal_step"],
            process_covariance=spec.parameters["process_covariance"],
            observation_covariance=spec.parameters["observation_covariance"],
            initial_covariance=spec.parameters["initial_covariance"],
            domain_policy=spec.parameters["domain_policy"],
        )
    if spec.model_id == "predator_prey_rk4":
        return highdim.PredatorPreySSM(
            initial_mean=spec.parameters["initial_mean"],
            delta=spec.parameters["delta"],
            rk4_internal_step=spec.parameters["rk4_internal_step"],
            process_covariance=spec.parameters["process_covariance"],
            observation_covariance=spec.parameters["observation_covariance"],
            initial_covariance=spec.parameters["initial_covariance"],
            domain_policy=spec.parameters["domain_policy"],
        )
    raise ValueError(f"unknown common model v2 spec: {spec.model_id}")


def evaluate_bayesfilter_density_v2(spec: CommonModelSpecV2) -> dict[str, Any]:
    model = bayesfilter_model_for_spec_v2(spec)
    initial = model.initial_log_density(spec.theta, spec.x0)
    transition = model.transition_log_density(spec.theta, spec.x_prev, spec.x_next, t=1)
    observation = model.observation_log_density(spec.theta, spec.x_obs, spec.observation, t=1)
    scalar = tf.reduce_sum(initial) + tf.reduce_sum(transition) + tf.reduce_sum(observation)
    return {
        "status": "executed",
        "backend": "bayesfilter_common_model_v2_adapter",
        "model_id": spec.model_id,
        "initial_log_density": _tensor_to_json(initial),
        "transition_log_density": _tensor_to_json(transition),
        "observation_log_density": _tensor_to_json(observation),
        "scalar": float(tf.cast(scalar, DTYPE).numpy()),
        "finite": bool(
            tf.reduce_all(tf.math.is_finite(initial)).numpy()
            and tf.reduce_all(tf.math.is_finite(transition)).numpy()
            and tf.reduce_all(tf.math.is_finite(observation)).numpy()
            and tf.math.is_finite(scalar).numpy()
        ),
        "model_manifest": _jsonable(model.manifest_payload()),
        "spec_checksum": spec.checksum(),
    }


def evaluate_bayesfilter_density(spec: CommonModelSpec) -> dict[str, Any]:
    model = bayesfilter_model_for_spec(spec)
    initial = model.initial_log_density(spec.theta, spec.x0)
    transition = model.transition_log_density(spec.theta, spec.x_prev, spec.x_next, t=1)
    observation = model.observation_log_density(spec.theta, spec.x_obs, spec.observation, t=1)
    scalar = tf.reduce_sum(initial) + tf.reduce_sum(transition) + tf.reduce_sum(observation)
    return {
        "status": "executed",
        "backend": "bayesfilter_common_model_adapter",
        "model_id": spec.model_id,
        "initial_log_density": _tensor_to_json(initial),
        "transition_log_density": _tensor_to_json(transition),
        "observation_log_density": _tensor_to_json(observation),
        "scalar": float(tf.cast(scalar, DTYPE).numpy()),
        "finite": bool(
            tf.reduce_all(tf.math.is_finite(initial)).numpy()
            and tf.reduce_all(tf.math.is_finite(transition)).numpy()
            and tf.reduce_all(tf.math.is_finite(observation)).numpy()
            and tf.math.is_finite(scalar).numpy()
        ),
        "model_manifest": _jsonable(model.manifest_payload()),
        "spec_checksum": spec.checksum(),
    }


EXPECTED_V2_MODEL_IDS = (
    "lgssm_2d_h25_rich",
    "sv_1d_h18_rich",
    "range_bearing_4d_h20_rich",
    "structural_ar1_quadratic_h16",
    "spatial_sir_j3_rk4",
    "predator_prey_rk4",
)


def _common_lgssm_v2_spec() -> CommonModelSpecV2:
    fixture = build_lgssm_fixture_tf(horizon=25)
    x0 = tf.constant([[0.1, -0.2], [0.55, 0.05], [-0.35, 0.4]], dtype=DTYPE)
    x_prev = x0
    x_next = tf.constant([[0.0, -0.1], [0.42, 0.2], [-0.25, 0.25]], dtype=DTYPE)
    observations = fixture.observations
    return _v2_spec(
        model_id="lgssm_2d_h25_rich",
        family="linear_gaussian",
        source_surface="build_lgssm_fixture_tf(horizon=25)",
        successor_of=("lgssm_tf.py", "lgssm_2d_linear"),
        parameters={
            "A": fixture.A,
            "C": fixture.C,
            "Q": fixture.Q,
            "R": fixture.R,
            "m0": fixture.m0,
            "P0": fixture.P0,
            "fixture_generation_seed": fixture.fixture_generation_seed,
            "full_horizon": fixture.horizon,
            "fixture_model_checksum": fixture.model_checksum,
            "fixture_observation_checksum": fixture.observation_checksum,
        },
        theta=tf.zeros([0], dtype=DTYPE),
        x0=x0,
        x_prev=x_prev,
        x_next=x_next,
        x_obs=x_next,
        observation=tf.reshape(observations[0], [-1]),
        observations=observations,
        path_initial_particles=x0,
        path_observations=tf.gather(observations, [0, 1, 2]),
        transition_innovations=tf.constant(
            [
                [[0.020, -0.030], [-0.015, 0.040], [0.035, -0.020]],
                [[-0.010, 0.015], [0.025, -0.035], [-0.030, 0.020]],
                [[0.005, 0.025], [-0.020, 0.010], [0.015, -0.030]],
            ],
            dtype=DTYPE,
        ),
        gradient_knobs=[
            {
                "name": "transition_matrix_scale",
                "initial_value": 1.0,
                "parameterization": "physical scalar multiplying transition matrix A",
                "include": True,
            },
            {
                "name": "observation_noise_scale",
                "initial_value": 1.0,
                "parameterization": "physical scalar multiplying observation covariance R",
                "include": True,
            },
        ],
        adapter_note="built-in linear Gaussian density route plus local subprocess adapter",
    )


def build_lgssm_fixture_tf(
    *,
    horizon: int = 25,
    fixture_generation_seed: int = 2026052801,
) -> LGSSMTFFixture:
    """Build the retired LGSSM fixture inline for production v2 use.

    This is intentionally behavior-preserving relative to the standalone
    `lgssm_tf.py` builder so closed v1/reference artifacts remain comparable
    while production v2 no longer imports that module.
    """

    a = tf.constant([[0.92, 0.18], [-0.04, 0.86]], dtype=DTYPE)
    c = tf.constant([[1.0, 0.35]], dtype=DTYPE)
    q = tf.constant([[0.08, 0.015], [0.015, 0.05]], dtype=DTYPE)
    r = tf.constant([[0.18]], dtype=DTYPE)
    m0 = tf.constant([0.25, -0.15], dtype=DTYPE)
    p0 = tf.constant([[0.45, 0.04], [0.04, 0.32]], dtype=DTYPE)
    states, observations = _simulate_lgssm(
        a=a,
        c=c,
        q=q,
        r=r,
        m0=m0,
        p0=p0,
        horizon=horizon,
        seed=fixture_generation_seed,
    )
    model_payload = {
        "name": "lgssm_ot_tf_tfp_smoke",
        "A": _tensor_to_json(a),
        "C": _tensor_to_json(c),
        "Q": _tensor_to_json(q),
        "R": _tensor_to_json(r),
        "m0": _tensor_to_json(m0),
        "P0": _tensor_to_json(p0),
        "horizon": int(horizon),
        "fixture_generation_seed": int(fixture_generation_seed),
        "backend": "tensorflow_tensorflow_probability",
    }
    model_checksum = stable_digest(model_payload)
    observation_checksum = stable_digest(
        {"states": _tensor_to_json(states), "observations": _tensor_to_json(observations)}
    )
    return LGSSMTFFixture(
        name="lgssm_ot_tf_tfp_smoke",
        A=a,
        C=c,
        Q=q,
        R=r,
        m0=m0,
        P0=p0,
        states=states,
        observations=observations,
        fixture_generation_seed=int(fixture_generation_seed),
        model_checksum=model_checksum,
        observation_checksum=observation_checksum,
    )


def _simulate_lgssm(
    *,
    a: tf.Tensor,
    c: tf.Tensor,
    q: tf.Tensor,
    r: tf.Tensor,
    m0: tf.Tensor,
    p0: tf.Tensor,
    horizon: int,
    seed: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    state_dist = tfd.MultivariateNormalTriL(loc=m0, scale_tril=_chol(p0))
    x0 = state_dist.sample(seed=_seed_pair(seed, 1))
    states = [x0]
    observations = []
    q_dist = tfd.MultivariateNormalTriL(
        loc=tf.zeros([tf.shape(a)[0]], dtype=DTYPE),
        scale_tril=_chol(q),
    )
    r_dist = tfd.MultivariateNormalTriL(
        loc=tf.zeros([tf.shape(c)[0]], dtype=DTYPE),
        scale_tril=_chol(r),
    )
    for time_index in range(horizon):
        mean_state = tf.linalg.matvec(a, states[-1])
        state = mean_state + q_dist.sample(seed=_seed_pair(seed, 10 + time_index))
        obs_mean = tf.linalg.matvec(c, state)
        obs = obs_mean + r_dist.sample(seed=_seed_pair(seed, 100 + time_index))
        states.append(state)
        observations.append(obs)
    return tf.stack(states, axis=0), tf.stack(observations, axis=0)


def _common_sv_v2_spec() -> CommonModelSpecV2:
    fixture = build_stochastic_volatility_fixture_tf(horizon=18)
    x0 = tf.constant([[-0.8], [0.0], [0.9]], dtype=DTYPE)
    x_prev = tf.constant([[-1.1], [-0.2], [0.35]], dtype=DTYPE)
    x_next = tf.constant([[-0.75], [0.08], [0.42]], dtype=DTYPE)
    return _v2_spec(
        model_id="sv_1d_h18_rich",
        family="stochastic_volatility_ar1",
        source_surface="build_stochastic_volatility_fixture_tf(horizon=18)",
        successor_of=("stochastic_volatility_tf.py", "sv_1d_synthetic"),
        parameters={
            "mu": fixture.mu,
            "phi": fixture.phi,
            "sigma": fixture.sigma,
            "h0_mean": fixture.h0_mean,
            "h0_variance": fixture.h0_variance,
            "fixture_generation_seed": fixture.fixture_generation_seed,
            "full_horizon": fixture.horizon,
            "fixture_model_checksum": fixture.model_checksum,
            "fixture_observation_checksum": fixture.observation_checksum,
        },
        theta=tf.zeros([0], dtype=DTYPE),
        x0=x0,
        x_prev=x_prev,
        x_next=x_next,
        x_obs=x_next,
        observation=tf.reshape(fixture.observations[0], [-1]),
        observations=tf.reshape(fixture.observations, [-1, 1]),
        path_initial_particles=x0,
        path_observations=tf.reshape(tf.gather(fixture.observations, [0, 1, 2]), [-1, 1]),
        transition_innovations=tf.constant(
            [
                [[0.10], [-0.05], [0.08]],
                [[-0.07], [0.12], [-0.04]],
                [[0.03], [-0.09], [0.11]],
            ],
            dtype=DTYPE,
        ),
        gradient_knobs=[
            {
                "name": "mu",
                "initial_value": float(fixture.mu.numpy()),
                "parameterization": "physical AR(1) level",
                "include": True,
            },
            {
                "name": "phi",
                "initial_value": float(fixture.phi.numpy()),
                "parameterization": "physical AR(1) persistence",
                "include": True,
            },
            {
                "name": "sigma",
                "initial_value": float(fixture.sigma.numpy()),
                "parameterization": "positive physical transition scale",
                "include": True,
            },
        ],
        adapter_note="rich AR(1) stochastic-volatility adapter; not the v1 synthetic gamma/beta route",
    )


def build_stochastic_volatility_fixture_tf(
    *,
    horizon: int = 18,
    fixture_generation_seed: int = 2026052901,
) -> StochasticVolatilityTFFixture:
    """Build the retired SV fixture inline for production v2 use."""

    mu = tf.constant(-0.7, dtype=DTYPE)
    phi = tf.constant(0.92, dtype=DTYPE)
    sigma = tf.constant(0.28, dtype=DTYPE)
    h0_mean = mu
    h0_variance = sigma * sigma / (1.0 - phi * phi)
    states, observations = _simulate_sv(
        mu=mu,
        phi=phi,
        sigma=sigma,
        h0_mean=h0_mean,
        h0_variance=h0_variance,
        horizon=horizon,
        seed=fixture_generation_seed,
    )
    model_payload = {
        "name": "sv_cut4_ledh_smoke",
        "mu": float(tf.cast(mu, DTYPE).numpy()),
        "phi": float(tf.cast(phi, DTYPE).numpy()),
        "sigma": float(tf.cast(sigma, DTYPE).numpy()),
        "h0_mean": float(tf.cast(h0_mean, DTYPE).numpy()),
        "h0_variance": float(tf.cast(h0_variance, DTYPE).numpy()),
        "horizon": int(horizon),
        "fixture_generation_seed": int(fixture_generation_seed),
    }
    model_checksum = stable_digest(model_payload)
    observation_checksum = stable_digest(
        {"states": _tensor_to_json(states), "observations": _tensor_to_json(observations)}
    )
    return StochasticVolatilityTFFixture(
        name="sv_cut4_ledh_smoke",
        mu=mu,
        phi=phi,
        sigma=sigma,
        h0_mean=h0_mean,
        h0_variance=h0_variance,
        states=states,
        observations=observations,
        fixture_generation_seed=fixture_generation_seed,
        model_checksum=model_checksum,
        observation_checksum=observation_checksum,
    )


def _simulate_sv(
    *,
    mu: tf.Tensor,
    phi: tf.Tensor,
    sigma: tf.Tensor,
    h0_mean: tf.Tensor,
    h0_variance: tf.Tensor,
    horizon: int,
    seed: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    h = tfd.Normal(loc=h0_mean, scale=tf.sqrt(h0_variance)).sample(seed=_seed_pair(seed, 1))
    states = []
    observations = []
    for time_index in range(horizon):
        eta = tfd.Normal(loc=tf.constant(0.0, DTYPE), scale=tf.constant(1.0, DTYPE)).sample(
            seed=_seed_pair(seed, 10 + time_index)
        )
        h = mu + phi * (h - mu) + sigma * eta
        eps = tfd.Normal(loc=tf.constant(0.0, DTYPE), scale=tf.constant(1.0, DTYPE)).sample(
            seed=_seed_pair(seed, 200 + time_index)
        )
        y = tf.exp(0.5 * h) * eps
        states.append(h)
        observations.append(y)
    return tf.stack(states, axis=0), tf.stack(observations, axis=0)


def _common_range_bearing_v2_spec() -> CommonModelSpecV2:
    fixture = build_range_bearing_fixture_tf()
    x0 = tf.constant(
        [[1.10, 0.64, 0.15, -0.03], [1.35, 0.78, 0.2, -0.08], [0.95, 0.5, 0.1, 0.02]],
        dtype=DTYPE,
    )
    x_next = tf.constant(
        [[1.13, 0.62, 0.16, -0.04], [1.38, 0.77, 0.18, -0.07], [0.97, 0.51, 0.11, 0.00]],
        dtype=DTYPE,
    )
    return _v2_spec(
        model_id="range_bearing_4d_h20_rich",
        family="range_bearing",
        source_surface="build_range_bearing_fixture_tf()",
        successor_of=("range_bearing_tf.py", "range_bearing_2d_cv"),
        parameters={
            "A": fixture.A,
            "Q": fixture.Q,
            "R": fixture.R,
            "m0": fixture.m0,
            "P0": fixture.P0,
            "dt": fixture.dt,
            "fixture_generation_seed": fixture.fixture_generation_seed,
            "full_horizon": fixture.horizon,
            "fixture_model_checksum": fixture.model_checksum,
            "fixture_observation_checksum": fixture.observation_checksum,
        },
        theta=tf.zeros([0], dtype=DTYPE),
        x0=x0,
        x_prev=x0,
        x_next=x_next,
        x_obs=x_next,
        observation=tf.reshape(fixture.observations[0], [-1]),
        observations=fixture.observations,
        path_initial_particles=x0,
        path_observations=tf.gather(fixture.observations, [0, 1, 2]),
        transition_innovations=tf.constant(
            [
                [[0.006, -0.004, 0.002, -0.001], [-0.005, 0.003, -0.001, 0.002], [0.004, 0.002, 0.001, -0.002]],
                [[-0.003, 0.005, 0.001, 0.001], [0.002, -0.006, -0.002, 0.001], [0.005, -0.001, 0.002, -0.001]],
                [[0.004, 0.001, -0.001, 0.002], [-0.002, 0.004, 0.001, -0.002], [0.003, -0.003, -0.001, 0.001]],
            ],
            dtype=DTYPE,
        ),
        gradient_knobs=[
            {
                "name": "sigma_range",
                "initial_value": 0.12,
                "parameterization": "positive physical range observation standard deviation",
                "include": True,
            },
            {
                "name": "sigma_bearing",
                "initial_value": 0.04,
                "parameterization": "positive physical bearing observation standard deviation",
                "include": True,
            },
        ],
        adapter_note="local range-bearing adapter with angle wrapping",
    )


def build_range_bearing_fixture_tf(
    name: str = "range_bearing_gaussian_tf_tfp_moderate",
) -> RangeBearingTFFixture:
    """Build the retired range-bearing fixture inline for production v2 use."""

    if name != "range_bearing_gaussian_tf_tfp_moderate":
        raise ValueError("only range_bearing_gaussian_tf_tfp_moderate is authorized")
    dt = 0.1
    a = tf.constant(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=DTYPE,
    )
    q = tf.linalg.diag(tf.constant([0.0015, 0.0015, 0.0008, 0.0008], dtype=DTYPE))
    sigma_range = tf.constant(0.12, dtype=DTYPE)
    sigma_bearing = tf.constant(0.04, dtype=DTYPE)
    r = tf.linalg.diag(tf.stack([sigma_range * sigma_range, sigma_bearing * sigma_bearing]))
    m0 = tf.constant([1.2, 0.7, 0.18, -0.06], dtype=DTYPE)
    p0 = tf.linalg.diag(tf.constant([0.04, 0.04, 0.01, 0.01], dtype=DTYPE))
    seed = 701
    states, observations = _simulate_range_bearing(
        a=a,
        q=q,
        r=r,
        m0=m0,
        p0=p0,
        horizon=20,
        seed=seed,
    )
    model_payload = {
        "name": name,
        "A": _tensor_to_json(a),
        "Q": _tensor_to_json(q),
        "R": _tensor_to_json(r),
        "m0": _tensor_to_json(m0),
        "P0": _tensor_to_json(p0),
        "horizon": 20,
        "seed": seed,
        "backend": "tensorflow_tensorflow_probability",
    }
    model_checksum = stable_digest(model_payload)
    observation_checksum = stable_digest(
        {"states": _tensor_to_json(states), "observations": _tensor_to_json(observations)}
    )
    return RangeBearingTFFixture(
        name=name,
        A=a,
        Q=q,
        R=r,
        m0=m0,
        P0=p0,
        states=states,
        observations=observations,
        dt=dt,
        fixture_generation_seed=seed,
        model_checksum=model_checksum,
        observation_checksum=observation_checksum,
    )


def _simulate_range_bearing(
    *,
    a: tf.Tensor,
    q: tf.Tensor,
    r: tf.Tensor,
    m0: tf.Tensor,
    p0: tf.Tensor,
    horizon: int,
    seed: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    state_dist = tfd.MultivariateNormalTriL(loc=m0, scale_tril=_chol(p0))
    state = state_dist.sample(seed=_seed_pair(seed, 1))
    states = [state]
    observations = []
    q_dist = tfd.MultivariateNormalTriL(
        loc=tf.zeros([tf.shape(a)[0]], dtype=DTYPE),
        scale_tril=_chol(q),
    )
    r_dist = tfd.MultivariateNormalTriL(
        loc=tf.zeros([tf.shape(r)[0]], dtype=DTYPE),
        scale_tril=_chol(r),
    )
    for time_index in range(horizon):
        state = tf.linalg.matvec(a, state) + q_dist.sample(seed=_seed_pair(seed, 30 + time_index))
        obs_mean = range_bearing_observation_tf(state)
        obs = obs_mean + r_dist.sample(seed=_seed_pair(seed, 300 + time_index))
        obs = tf.stack([obs[0], wrap_angle_tf(obs[1])], axis=0)
        states.append(state)
        observations.append(obs)
    return tf.stack(states, axis=0), tf.stack(observations, axis=0)


def _common_structural_ar1_v2_spec() -> CommonModelSpecV2:
    fixture = build_structural_ar1_quadratic_fixture_tf(horizon=16)
    x0 = tf.constant([[0.10, -0.05], [0.18, -0.05], [-0.04, -0.05]], dtype=DTYPE)
    x_prev = tf.constant([[0.10, -0.05], [0.18, 0.08], [-0.04, -0.02]], dtype=DTYPE)
    current_m = fixture.rho * x_prev[:, 0] + tf.constant([0.03, -0.04, 0.02], dtype=DTYPE)
    current_k = complete_k_tf(
        previous_k=x_prev[:, 1],
        previous_m=x_prev[:, 0],
        current_m=current_m,
        a=fixture.a,
        b=fixture.b,
        c=fixture.c,
        d=fixture.d,
    )
    x_next = tf.stack([current_m, current_k], axis=1)
    observations = tf.reshape(fixture.observations, [-1, 1])
    return _v2_spec(
        model_id="structural_ar1_quadratic_h16",
        family="structural_ar1_quadratic_completion",
        source_surface="build_structural_ar1_quadratic_fixture_tf(horizon=16)",
        successor_of=("structural_ar1_quadratic_tf.py",),
        parameters={
            "rho": fixture.rho,
            "sigma": fixture.sigma,
            "a": fixture.a,
            "b": fixture.b,
            "c": fixture.c,
            "d": fixture.d,
            "lambda": fixture.lam,
            "observation_scale": fixture.observation_scale,
            "m0_mean": fixture.m0_mean,
            "m0_variance": fixture.m0_variance,
            "k0": fixture.k0,
            "fixture_generation_seed": fixture.fixture_generation_seed,
            "full_horizon": fixture.horizon,
            "fixture_model_checksum": fixture.model_checksum,
            "fixture_observation_checksum": fixture.observation_checksum,
            "density_convention": "singular_completion_density_on_m_with_k_carried_in_state",
        },
        theta=tf.zeros([0], dtype=DTYPE),
        x0=x0,
        x_prev=x_prev,
        x_next=x_next,
        x_obs=x_next,
        observation=tf.reshape(observations[0], [-1]),
        observations=observations,
        path_initial_particles=x0,
        path_observations=tf.gather(observations, [0, 1, 2]),
        transition_innovations=tf.constant(
            [
                [[0.030, 0.0], [-0.040, 0.0], [0.020, 0.0]],
                [[-0.020, 0.0], [0.015, 0.0], [-0.010, 0.0]],
                [[0.010, 0.0], [-0.025, 0.0], [0.030, 0.0]],
            ],
            dtype=DTYPE,
        ),
        gradient_knobs=[
            {
                "name": "rho",
                "initial_value": float(fixture.rho.numpy()),
                "parameterization": "physical AR(1) persistence",
                "include": True,
            },
            {
                "name": "sigma",
                "initial_value": float(fixture.sigma.numpy()),
                "parameterization": "positive physical AR(1) scale",
                "include": True,
            },
            {
                "name": "c",
                "initial_value": float(fixture.c.numpy()),
                "parameterization": "physical quadratic completion coefficient",
                "include": True,
            },
        ],
        adapter_note="structural split adapter with deterministic k completion",
    )


def _common_spatial_sir_v2_spec() -> CommonModelSpecV2:
    model = highdim.p30_spatial_sir_fixture_model(3)
    x0 = tf.constant(
        [
            [486.0, 14.0, 487.0, 13.0, 488.0, 12.0],
            [485.6, 14.3, 487.2, 12.8, 488.1, 11.7],
            [486.2, 13.8, 486.8, 13.4, 487.7, 12.2],
        ],
        dtype=DTYPE,
    )
    x_prev = x0
    innovations = tf.constant(
        [
            [[0.02, -0.03, 0.01, -0.02, 0.03, -0.01], [-0.02, 0.02, 0.03, -0.01, -0.01, 0.02], [0.01, -0.01, -0.02, 0.02, 0.02, -0.03]],
            [[-0.01, 0.02, -0.02, 0.01, 0.02, -0.01], [0.02, -0.01, 0.01, -0.03, -0.02, 0.03], [-0.02, 0.03, 0.02, -0.01, 0.01, -0.02]],
            [[0.01, 0.02, -0.01, -0.02, 0.02, 0.01], [-0.03, 0.01, 0.02, 0.03, -0.01, -0.02], [0.02, -0.03, -0.02, 0.01, 0.03, -0.01]],
        ],
        dtype=DTYPE,
    )
    x_next = model.transition_mean(x_prev) + innovations[0]
    _, observations = model.simulate(final_time=3, seed=3703)
    return _v2_spec(
        model_id="spatial_sir_j3_rk4",
        family="spatial_sir",
        source_surface="bayesfilter.highdim.p30_spatial_sir_fixture_model(3)",
        successor_of=("bayesfilter.highdim.SpatialSIRSSM",),
        parameters={
            "kappa": model.kappa,
            "nu": model.nu,
            "initial_mean": model.initial_mean,
            "neighbor_sets": model.neighbor_sets,
            "delta": model.delta,
            "rk4_internal_step": model.rk4_internal_step,
            "rk4_substeps": int(model.manifest_payload()["rk4_substeps"]),
            "process_covariance": model.process_covariance,
            "observation_covariance": model.observation_covariance,
            "initial_covariance": model.initial_covariance,
            "domain_policy": model.domain_policy,
            "state_coordinate_convention": "state=(S_1,I_1,S_2,I_2,S_3,I_3)",
            "observation_route": "infectious-only observation (I_1,I_2,I_3)",
            "graph_semantics": "undirected chain neighbor sets with diffusion-like neighbor coupling",
        },
        theta=tf.zeros([0], dtype=DTYPE),
        x0=x0,
        x_prev=x_prev,
        x_next=x_next,
        x_obs=x_next,
        observation=tf.reshape(observations[1], [-1]),
        observations=observations[1:],
        path_initial_particles=x0,
        path_observations=observations[1:4],
        transition_innovations=innovations,
        gradient_knobs=[
            {
                "name": "sir_physical_knobs",
                "initial_value": None,
                "parameterization": "none in v2 P5",
                "include": False,
                "exclusion_reason": "no reviewed BF/FF physical-gradient contract for SIR before results",
            }
        ],
        adapter_note="no-lookup RK4 SIR adapter with infectious-only observation",
    )


def _common_predator_prey_v2_spec() -> CommonModelSpecV2:
    model = highdim.p30_predator_prey_fixture_model()
    theta = model.true_parameters()
    x0 = tf.constant([[50.0, 5.0], [51.0, 4.7], [49.4, 5.3]], dtype=DTYPE)
    x_prev = x0
    innovations = tf.constant(
        [
            [[0.20, -0.10], [-0.15, 0.08], [0.10, 0.05]],
            [[-0.08, 0.06], [0.12, -0.04], [-0.10, 0.09]],
            [[0.05, -0.07], [-0.06, 0.03], [0.08, -0.02]],
        ],
        dtype=DTYPE,
    )
    x_next = model.transition_mean(theta, x_prev) + innovations[0]
    _, observations = model.simulate(theta, final_time=3, seed=4404)
    return _v2_spec(
        model_id="predator_prey_rk4",
        family="predator_prey",
        source_surface="bayesfilter.highdim.p30_predator_prey_fixture_model()",
        successor_of=("bayesfilter.highdim.PredatorPreySSM",),
        parameters={
            "initial_mean": model.initial_mean,
            "delta": model.delta,
            "rk4_internal_step": model.rk4_internal_step,
            "rk4_substeps": int(model.manifest_payload()["rk4_substeps"]),
            "process_covariance": model.process_covariance,
            "observation_covariance": model.observation_covariance,
            "initial_covariance": model.initial_covariance,
            "domain_policy": model.domain_policy,
            "parameter_box": model.parameter_box(),
            "physical_parameterization": "theta=(r,K,a,s,u,v)",
            "state_coordinate_convention": "state=(prey,predator)",
            "observation_route": "direct noisy-state observation",
        },
        theta=theta,
        x0=x0,
        x_prev=x_prev,
        x_next=x_next,
        x_obs=x_next,
        observation=tf.reshape(observations[1], [-1]),
        observations=observations[1:],
        path_initial_particles=x0,
        path_observations=observations[1:4],
        transition_innovations=innovations,
        gradient_knobs=[
            {
                "name": "r",
                "initial_value": float(theta[0].numpy()),
                "parameterization": "physical predator-prey growth rate in theta=(r,K,a,s,u,v)",
                "include": True,
            }
        ],
        adapter_note="no-lookup RK4 predator-prey adapter with direct observation",
    )


def _v2_spec(
    *,
    model_id: str,
    family: str,
    source_surface: str,
    successor_of: tuple[str, ...],
    parameters: Mapping[str, Any],
    theta: tf.Tensor,
    x0: tf.Tensor,
    x_prev: tf.Tensor,
    x_next: tf.Tensor,
    x_obs: tf.Tensor,
    observation: tf.Tensor,
    observations: tf.Tensor,
    path_initial_particles: tf.Tensor,
    path_observations: tf.Tensor,
    transition_innovations: tf.Tensor,
    gradient_knobs: list[Mapping[str, Any]],
    adapter_note: str,
) -> CommonModelSpecV2:
    path_contract = {
        "horizon": int(path_observations.shape[0]),
        "num_particles": int(path_initial_particles.shape[0]),
        "state_dim": int(path_initial_particles.shape[1]),
        "initial_particles": path_initial_particles,
        "transition_innovations": transition_innovations,
        "observations": path_observations,
        "initial_log_weight_policy": "uniform_normalized_log_weights",
        "proposal": "bootstrap_fixed_transition_innovations",
        "scalar": "sum of per-step predictive log normalizers",
        "transition_innovation_policy": (
            "structural rows use innovation in stochastic coordinate and deterministic completion"
            if model_id == "structural_ar1_quadratic_h16"
            else "additive state innovation after declared transition mean"
        ),
    }
    fixed_ancestor_contract = {
        **path_contract,
        "resampling_policy": "fixed_ancestor_replay_before_proposal",
        "resampling_flags": [False, True, False],
        "fixed_ancestor_indices": [0, 0, 2],
        "expected_resampling_count": 1,
    }
    ready_gradient = any(bool(knob.get("include")) for knob in gradient_knobs)
    phase_readiness = {
        "P2_density": "READY_FOR_P2",
        "P3_noresampling": "READY_FOR_P3",
        "P4_fixed_ancestor": "READY_FOR_P4",
        "P5_gradients": "READY_FOR_P5" if ready_gradient else "CONTRACT_BLOCKED",
        "reason": (
            "declared no-lookup adapter semantics frozen before execution"
            if ready_gradient
            else "gradient row excluded before results because no reviewed physical-gradient contract exists"
        ),
    }
    return CommonModelSpecV2(
        model_id=model_id,
        family=family,
        source_surface=source_surface,
        successor_of=successor_of,
        parameters=parameters,
        theta=tf.convert_to_tensor(theta, DTYPE),
        x0=tf.convert_to_tensor(x0, DTYPE),
        x_prev=tf.convert_to_tensor(x_prev, DTYPE),
        x_next=tf.convert_to_tensor(x_next, DTYPE),
        x_obs=tf.convert_to_tensor(x_obs, DTYPE),
        observation=tf.convert_to_tensor(observation, DTYPE),
        observations=tf.convert_to_tensor(observations, DTYPE),
        path_contract=path_contract,
        fixed_ancestor_contract=fixed_ancestor_contract,
        gradient_contract={
            "knobs": gradient_knobs,
            "finite_difference_step": 1e-5,
            "gradient_scalar": "fixed-branch sum of predictive log normalizers",
            "random_or_discrete_ancestor_gradient": "excluded",
        },
        tolerances={
            "density_value_abs": 5e-10,
            "path_value_abs": 5e-10,
            "path_ledger_abs": 5e-10,
            "gradient_abs": 5e-8,
            "finite_difference_abs": 5e-5,
        },
        phase_readiness=phase_readiness,
        adapter_certification={
            "bayesfilter_route": "local highdim or common v2 adapter",
            "filterflow_route": "CPU-only local subprocess adapter; no .localsource/filterflow mutation",
            "equation_source": source_surface,
            "adapter_note": adapter_note,
            "certification_status": "DECLARED_BEFORE_RESULTS",
        },
        non_claims=(
            "agreement is not a filter correctness proof",
            "BayesFilter and FilterFlow are peers, not oracles",
            "no student implementation claim",
            "no TT/SIRT correctness or paper-scale reproduction claim",
        ),
    )


def _assert_exact_v2_model_ids(model_ids: list[str]) -> None:
    if tuple(model_ids) != EXPECTED_V2_MODEL_IDS:
        raise ValueError(f"v2 model-id gate failed: {model_ids}")


def stable_digest(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _mvn_log_prob(values: tf.Tensor, loc: tf.Tensor, covariance: tf.Tensor) -> tf.Tensor:
    values = tf.convert_to_tensor(values, DTYPE)
    loc = tf.convert_to_tensor(loc, DTYPE)
    covariance = tf.convert_to_tensor(covariance, DTYPE)
    distribution = tfd.MultivariateNormalTriL(loc=loc, scale_tril=tf.linalg.cholesky(covariance))
    return distribution.log_prob(values)


def _normal_log_prob(value: tf.Tensor, loc: tf.Tensor | float, scale: tf.Tensor | float) -> tf.Tensor:
    return tfd.Normal(
        loc=tf.convert_to_tensor(loc, DTYPE),
        scale=tf.convert_to_tensor(scale, DTYPE),
    ).log_prob(tf.convert_to_tensor(value, DTYPE))


def range_bearing_observation_tf(x: tf.Tensor, *, eps: float = 1e-12) -> tf.Tensor:
    x = tf.cast(x, DTYPE)
    px = x[..., 0]
    py = x[..., 1]
    rng = tf.sqrt(px * px + py * py + tf.constant(eps, dtype=DTYPE))
    bearing = tf.atan2(py, px)
    return tf.stack([rng, bearing], axis=-1)


def observation_residual_tf(predicted: tf.Tensor, observed: tf.Tensor) -> tf.Tensor:
    residual = tf.cast(observed, DTYPE) - tf.cast(predicted, DTYPE)
    first = residual[..., :1]
    second = wrap_angle_tf(residual[..., 1:2])
    return tf.concat([first, second], axis=-1)


def wrap_angle_tf(value: tf.Tensor) -> tf.Tensor:
    return tf.math.floormod(tf.cast(value, DTYPE) + PI, 2.0 * PI) - PI


def gaussian_logpdf_zero_mean_tf(residuals: tf.Tensor, covariance: tf.Tensor) -> tf.Tensor:
    residuals = tf.cast(residuals, DTYPE)
    covariance = tf.cast(covariance, DTYPE)
    chol = _chol(covariance)
    solved = tf.linalg.cholesky_solve(chol, tf.transpose(residuals))
    quad = tf.reduce_sum(tf.transpose(solved) * residuals, axis=1)
    dim = tf.cast(tf.shape(covariance)[0], DTYPE)
    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
    return -0.5 * (dim * tf.math.log(2.0 * PI) + logdet + quad)


def _chol(covariance: tf.Tensor) -> tf.Tensor:
    return tf.linalg.cholesky(tf.cast(covariance, DTYPE))


def _seed_pair(seed: int, salt: int) -> tf.Tensor:
    return tf.constant([int(seed) % 2147483647, int(salt) % 2147483647], dtype=tf.int32)


def _as_row_matrix(values: tf.Tensor, width: int, name: str) -> tf.Tensor:
    tensor = tf.convert_to_tensor(values, DTYPE)
    if tensor.shape.rank == 1:
        tensor = tensor[tf.newaxis, :]
    if tensor.shape.rank != 2 or tensor.shape[1] != width:
        raise ValueError(f"{name}: invalid shape {tensor.shape}, expected width {width}")
    if not bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy()):
        raise ValueError(f"{name}: nonfinite value")
    return tensor


def _jsonable(value: Any) -> Any:
    if isinstance(value, tf.Tensor):
        return _tensor_to_json(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _tensor_to_json(value: tf.Tensor) -> Any:
    return tf.cast(value, DTYPE).numpy().tolist()
