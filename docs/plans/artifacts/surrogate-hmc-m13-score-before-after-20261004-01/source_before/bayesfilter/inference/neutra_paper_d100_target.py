"""TensorFlow targets and exact samplers for two NeuTra paper d100 controls.

The Gaussian constants are frozen by a diagnostic-only source exporter. This
candidate/runtime module deliberately does not import NumPy or ``dsge_hmc``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import tensorflow as tf

from bayesfilter.inference.posterior_adapter import ValueScoreCapability


PAPER_D100_DIMENSION = 100
PAPER_FUNNEL_NAME = "paper_funnel"
PAPER_GAUSSIAN_NAME = "paper_ill_cond_gaussian"
_PLAN = (
    "docs/plans/"
    "bayesfilter-weighted-forward-kl-paper-d100-fresh-baseline-plan-2026-08-13.md"
)


class PaperD100TargetError(RuntimeError):
    """Raised when a frozen paper-target contract is invalid."""


@dataclass(frozen=True)
class PaperD100TargetSpec:
    """Frozen target identity and, for the Gaussian, exact matrix constants."""

    name: str
    dimension: int = PAPER_D100_DIMENSION
    mean: tuple[float, ...] = ()
    covariance: tuple[tuple[float, ...], ...] = ()
    precision: tuple[tuple[float, ...], ...] = ()
    cholesky: tuple[tuple[float, ...], ...] = ()
    constants_path: str | None = None
    constants_sha256: str | None = None
    constants_hash: str | None = None
    source_path: str | None = None
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.name not in {PAPER_FUNNEL_NAME, PAPER_GAUSSIAN_NAME}:
            raise ValueError("unsupported paper d100 target")
        if int(self.dimension) != PAPER_D100_DIMENSION:
            raise ValueError("paper targets are frozen at dimension 100")
        if self.name == PAPER_FUNNEL_NAME:
            if any((self.mean, self.covariance, self.precision, self.cholesky)):
                raise ValueError("paper funnel must not carry Gaussian matrices")
            return
        dimension = int(self.dimension)
        if len(self.mean) != dimension:
            raise ValueError("Gaussian mean width mismatch")
        for name in ("covariance", "precision", "cholesky"):
            matrix = getattr(self, name)
            if len(matrix) != dimension or any(len(row) != dimension for row in matrix):
                raise ValueError(f"Gaussian {name} shape mismatch")
        numeric = (
            *self.mean,
            *(value for matrix in (self.covariance, self.precision, self.cholesky)
              for row in matrix for value in row),
        )
        if any(not math.isfinite(float(value)) for value in numeric):
            raise ValueError("Gaussian constants must be finite")
        if any(float(self.cholesky[index][index]) <= 0.0 for index in range(dimension)):
            raise ValueError("Gaussian Cholesky diagonal must be positive")
        for name in ("constants_sha256", "constants_hash", "source_sha256"):
            value = getattr(self, name)
            if value is None or len(str(value)) != 64:
                raise ValueError(f"Gaussian {name} must be a SHA-256 digest")

    def manifest_payload(self) -> Mapping[str, Any]:
        payload = {
            "schema": "bayesfilter.neutra.paper_d100_target.v1",
            "name": self.name,
            "dimension": int(self.dimension),
            "exact_sampler": True,
            "candidate_backend": "tensorflow_float64_batch_native",
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "constants_path": self.constants_path,
            "constants_sha256": self.constants_sha256,
            "constants_hash": self.constants_hash,
        }
        if self.name == PAPER_FUNNEL_NAME:
            payload.update(
                {
                    "source_formula": (
                        "y~Normal(0,1); x_i|y~Normal(0,exp(2y)) variance"
                    ),
                    "conditional_standard_deviation": "exp(y)",
                }
            )
        else:
            payload.update(
                {
                    "source_formula": (
                        "RandomState(10) Gamma(0.8,1) precision spectrum, "
                        "QR rotation, covariance inverse spectrum"
                    ),
                    "matrix_constants_embedded_in_runtime_spec": True,
                }
            )
        return payload


def make_paper_funnel_spec() -> PaperD100TargetSpec:
    """Return the formula-bound paper funnel specification."""

    return PaperD100TargetSpec(name=PAPER_FUNNEL_NAME)


def load_paper_gaussian_spec(path: str | Path) -> PaperD100TargetSpec:
    """Load and validate a diagnostic-exported Gaussian constants artifact."""

    constants_path = Path(path).resolve()
    raw = constants_path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PaperD100TargetError("Gaussian constants JSON is unreadable") from error
    if not isinstance(payload, Mapping) or payload.get("schema") != (
        "bayesfilter.neutra.paper_d100_gaussian_source.v1"
    ):
        raise PaperD100TargetError("Gaussian source schema mismatch")
    constants = payload.get("constants")
    if not isinstance(constants, Mapping):
        raise PaperD100TargetError("Gaussian constants object is missing")
    expected_hash = str(payload.get("constants_hash", ""))
    if len(expected_hash) != 64 or _stable_hash(constants) != expected_hash:
        raise PaperD100TargetError("Gaussian semantic constants hash mismatch")
    rng = constants.get("rng")
    if not isinstance(rng, Mapping) or rng != {
        "eigenvalue_role": "precision_eigenvalues",
        "gamma_scale": 1.0,
        "gamma_shape": 0.8,
        "library": "numpy.random.RandomState",
        "seed": 10,
    }:
        raise PaperD100TargetError("Gaussian source RNG contract mismatch")
    try:
        spec = PaperD100TargetSpec(
            name=str(constants["name"]),
            dimension=int(constants["dimension"]),
            mean=tuple(float(value) for value in constants["mean"]),
            covariance=tuple(
                tuple(float(value) for value in row)
                for row in constants["covariance"]
            ),
            precision=tuple(
                tuple(float(value) for value in row)
                for row in constants["precision"]
            ),
            cholesky=tuple(
                tuple(float(value) for value in row)
                for row in constants["cholesky"]
            ),
            constants_path=constants_path.as_posix(),
            constants_sha256=hashlib.sha256(raw).hexdigest(),
            constants_hash=expected_hash,
            source_path=str(payload["source_path"]),
            source_sha256=str(payload["source_sha256"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise PaperD100TargetError("Gaussian constants payload is invalid") from error
    _validate_gaussian_tensor_identities(spec)
    return spec


def paper_d100_log_prob_batch(spec: PaperD100TargetSpec, physical: Any) -> tf.Tensor:
    """Evaluate the source-equivalent unnormalized batch log density."""

    rows = _rank2(physical, spec.dimension, "paper d100 physical rows")
    if spec.name == PAPER_FUNNEL_NAME:
        y = rows[:, 0]
        x = rows[:, 1:]
        return -(
            tf.constant(0.5, tf.float64) * tf.square(y)
            + tf.constant(0.5, tf.float64)
            * tf.exp(tf.constant(-2.0, tf.float64) * y)
            * tf.reduce_sum(tf.square(x), axis=1)
            + tf.cast(spec.dimension - 1, tf.float64) * y
        )
    delta = rows - tf.constant(spec.mean, tf.float64)[tf.newaxis, :]
    precision = tf.constant(spec.precision, tf.float64)
    return -tf.constant(0.5, tf.float64) * tf.reduce_sum(
        delta * tf.matmul(delta, precision), axis=1
    )


def paper_d100_log_prob_and_score_batch(
    spec: PaperD100TargetSpec, physical: Any
) -> tuple[tf.Tensor, tf.Tensor]:
    """Evaluate the source-equivalent target and explicit analytic score."""

    rows = _rank2(physical, spec.dimension, "paper d100 physical rows")
    value = paper_d100_log_prob_batch(spec, rows)
    if spec.name == PAPER_FUNNEL_NAME:
        y = rows[:, 0]
        x = rows[:, 1:]
        inverse_variance = tf.exp(tf.constant(-2.0, tf.float64) * y)
        y_score = (
            -y
            + inverse_variance * tf.reduce_sum(tf.square(x), axis=1)
            - tf.cast(spec.dimension - 1, tf.float64)
        )
        x_score = -inverse_variance[:, tf.newaxis] * x
        score = tf.concat((y_score[:, tf.newaxis], x_score), axis=1)
    else:
        delta = rows - tf.constant(spec.mean, tf.float64)[tf.newaxis, :]
        score = -tf.matmul(delta, tf.constant(spec.precision, tf.float64))
    tf.debugging.assert_all_finite(value, "paper d100 target value")
    tf.debugging.assert_all_finite(score, "paper d100 target score")
    return value, score


def sample_paper_d100_exact(
    spec: PaperD100TargetSpec,
    sample_count: int,
    *,
    seed: tuple[int, int],
) -> tf.Tensor:
    """Draw exact independent physical rows with a stateless TensorFlow seed."""

    if isinstance(sample_count, bool) or int(sample_count) <= 1:
        raise ValueError("sample_count must exceed one")
    seed_tensor = tf.convert_to_tensor(seed, tf.int32)
    if seed_tensor.shape != (2,):
        raise ValueError("seed must contain exactly two integers")
    if spec.name == PAPER_FUNNEL_NAME:
        seeds = tf.random.experimental.stateless_split(seed_tensor, 2)
        y = tf.random.stateless_normal(
            (int(sample_count), 1), seed=seeds[0], dtype=tf.float64
        )
        residual = tf.random.stateless_normal(
            (int(sample_count), spec.dimension - 1),
            seed=seeds[1],
            dtype=tf.float64,
        )
        rows = tf.concat((y, tf.exp(y) * residual), axis=1)
    else:
        standard = tf.random.stateless_normal(
            (int(sample_count), spec.dimension), seed=seed_tensor, dtype=tf.float64
        )
        rows = tf.constant(spec.mean, tf.float64)[tf.newaxis, :] + tf.matmul(
            standard, tf.constant(spec.cholesky, tf.float64), transpose_b=True
        )
    tf.debugging.assert_all_finite(rows, "paper d100 exact samples")
    return rows


def paper_funnel_standardized_residuals(
    spec: PaperD100TargetSpec, physical: Any
) -> tf.Tensor:
    """Return exact standard-normal residuals ``x * exp(-y)`` for the funnel."""

    if spec.name != PAPER_FUNNEL_NAME:
        raise ValueError("standardized residuals are defined only for the funnel")
    rows = _rank2(physical, spec.dimension, "paper funnel physical rows")
    residual = rows[:, 1:] * tf.exp(-rows[:, :1])
    tf.debugging.assert_all_finite(residual, "paper funnel standardized residuals")
    return residual


class PaperD100ValueScoreAdapter:
    """Graph-native batch value/score adapter for an exact paper target."""

    supports_retained_draw_batch = False
    supports_retained_flat_batch = True
    supports_retained_value_score_status = True
    target_status_invalid_rows_become_nonfinite = False

    def __init__(self, spec: PaperD100TargetSpec) -> None:
        self.spec = spec
        self.parameter_dim = int(spec.dimension)
        self.target_scope = f"weighted_neutra_paper_d100:{spec.name}"

    def log_prob(self, physical: Any) -> tf.Tensor:
        return paper_d100_log_prob_batch(self.spec, physical)

    def log_prob_and_grad(self, physical: Any) -> tuple[tf.Tensor, tf.Tensor]:
        return paper_d100_log_prob_and_score_batch(self.spec, physical)

    def log_prob_and_grad_status(
        self, physical: Any
    ) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
        value, score = self.log_prob_and_grad(physical)
        finite = tf.logical_and(
            tf.math.is_finite(value),
            tf.reduce_all(tf.math.is_finite(score), axis=1),
        )
        zeros = tf.zeros_like(value, tf.int32)
        ones = tf.ones_like(value, tf.float64)
        return value, score, {
            "status_code": tf.where(finite, zeros, tf.ones_like(zeros)),
            "valid_pre_regularized_score": finite,
            "floor_count_value": zeros,
            "min_innovation_eigenvalue": ones,
            "innovation_condition_estimate": ones,
        }

    def target_status_telemetry(self, physical: Any) -> Mapping[str, tf.Tensor]:
        return self.log_prob_and_grad_status(physical)[2]

    def adapter_signature(self) -> str:
        return _stable_hash(
            {
                "schema": "bayesfilter.neutra.paper_d100_value_score.v1",
                "target_scope": self.target_scope,
                "target": self.spec.manifest_payload(),
                "value_score_authority": "graph_native_explicit_analytic_score",
            }
        )

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
            runtime_backend="tensorflow_exact_paper_d100_value_score",
            evidence_path=_PLAN,
            target_scope=self.target_scope,
            nonclaims=(
                "target implementation and exact sampling do not establish HMC validity",
                "no objective ranking or default promotion",
            ),
        )


def _validate_gaussian_tensor_identities(spec: PaperD100TargetSpec) -> None:
    covariance = tf.constant(spec.covariance, tf.float64)
    precision = tf.constant(spec.precision, tf.float64)
    cholesky = tf.constant(spec.cholesky, tf.float64)
    identity = tf.eye(spec.dimension, dtype=tf.float64)
    tf.debugging.assert_near(covariance, tf.transpose(covariance), atol=1.0e-10)
    tf.debugging.assert_near(precision, tf.transpose(precision), atol=1.0e-10)
    tf.debugging.assert_near(
        tf.matmul(precision, covariance), identity, atol=1.0e-8
    )
    tf.debugging.assert_near(
        tf.matmul(cholesky, cholesky, transpose_b=True), covariance, atol=1.0e-10
    )


def _rank2(value: Any, dimension: int, name: str) -> tf.Tensor:
    tensor = tf.convert_to_tensor(value, tf.float64)
    if tensor.shape.rank != 2 or tensor.shape[-1] != int(dimension):
        raise ValueError(f"{name} must have shape [row, {int(dimension)}]")
    if tensor.shape[0] is None:
        raise ValueError(f"{name} row count must be static")
    tf.debugging.assert_all_finite(tensor, name)
    return tensor


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
