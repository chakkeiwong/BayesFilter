"""Build Phase 1 byte-level canonical target signatures for all six rows.

This is CPU-hidden target materialization and serialization code. It does not
run a filter, construct LEDH particle randomness, or produce leaderboard cells.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
from typing import Any, Callable, Mapping, Sequence


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-matplotlib-cache")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.ledh_forward_contract import (
    ACTUAL_SV_ROW_ID,
    FIXED_SIR_AUSTRIA_ROW_ID,
    GENERALIZED_SV_PARAMETER_ORDER,
    GENERALIZED_SV_ROW_ID,
    KSC_SV_ROW_ID,
    LGSSM_M3_T50_PARAMETER_ORDER,
    LGSSM_M3_T50_ROW_ID,
    PREDATOR_PREY_PARAMETER_ORDER,
    PREDATOR_PREY_ROW_ID,
    SIR_LOG_SCALE_PARAMETER_ORDER,
    SV_SYNTHETIC_PARAMETER_ORDER,
)
from bayesfilter.highdim.models import (
    StochasticVolatilitySSM,
    p30_predator_prey_fixture_model,
    parameterized_zhao_cui_sir_austria_model,
)
from bayesfilter.highdim.sv_mixture_cut4 import (
    exact_transformed_sv_observations,
    ksc_1998_log_chi_square_mixture,
    transformed_sv_observations,
)
from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
    _generalized_sv_prior_mean_dataset,
    _lgssm_benchmark_model,
    _lgssm_dataset,
    _predator_prey_dataset,
    _sir_dataset,
    _sv_dataset,
)


SCHEMA_VERSION = "bayesfilter.complete_highdim.canonical_targets.v1"
ENCODING_ID = "length_prefixed_header_payload_then_semantics.sha256.v1"
RUN_ID = "complete-highdim-leaderboard-local-20260712-134906"
ARTIFACT_DATE = "2026-07-12"
DEFAULT_OUTPUT = ROOT / (
    "docs/plans/artifacts/complete-highdim-leaderboard/"
    "phase1-canonical-targets-2026-07-11.json"
)
LGSSM_THETA = (0.72, 0.55, 0.35, 0.35, 0.45)
SV_THETA = (0.2533471031357997, -0.916290731874155)
SIR_THETA = (0.0, 0.0, 0.0)
SIR_TARGET_GENERATION_IDENTITY = (
    "fixed_bayesfilter_sir_observations_from_dataset_seed_81103_"
    "not_author_matlab_rng1_reproduction"
)
PREDATOR_THETA = (0.6, 114.0, 25.0, 0.3, 0.5, 0.5)
GENERALIZED_SV_THETA = (1.0824113944610982, -2.076793740349318, 0.0)
KSC_TRANSFORM_OFFSET = 1.0e-8


@dataclass(frozen=True)
class TargetField:
    name: str
    value: Any
    source_slice: str
    preprocessing: tuple[str, ...]


@dataclass(frozen=True)
class RowMaterialization:
    row_id: str
    fields: tuple[TargetField, ...]
    semantics: Mapping[str, Any]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _frame(payload: bytes) -> bytes:
    return struct.pack(">Q", len(payload)) + payload


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes(paths: Sequence[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for relative in paths:
        path = ROOT / relative
        if not path.is_file():
            raise ValueError(f"missing target source path: {relative}")
        output[relative] = _sha256_file(path)
    return output


def _numeric_payload(field: TargetField) -> tuple[np.ndarray, bytes, dict[str, Any]]:
    if not field.name or not field.source_slice or not field.preprocessing:
        raise ValueError(f"incomplete field metadata: {field.name!r}")
    raw = field.value.numpy() if tf.is_tensor(field.value) else field.value
    array = np.asarray(raw)
    if array.dtype.hasobject or array.dtype.kind in {"O", "S", "U", "V"}:
        raise TypeError(f"{field.name}: implicit, object, or string dtype is forbidden")
    if array.dtype.kind not in {"b", "i", "u", "f", "c"}:
        raise TypeError(f"{field.name}: unsupported numeric dtype {array.dtype}")
    if array.dtype.kind in {"f", "c"} and not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{field.name}: nonfinite numeric payload")
    canonical_dtype = array.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array.astype(canonical_dtype, copy=False))
    if not canonical.flags.c_contiguous:
        raise ValueError(f"{field.name}: payload is not C contiguous")
    if canonical.dtype.itemsize > 1 and canonical.dtype.byteorder not in {"<", "="}:
        raise ValueError(f"{field.name}: canonical payload is not little endian")
    payload = canonical.tobytes(order="C")
    header = {
        "field_name": field.name,
        "dtype_descriptor": canonical.dtype.str,
        "dtype_kind": canonical.dtype.kind,
        "dtype_itemsize": int(canonical.dtype.itemsize),
        "shape": [int(item) for item in canonical.shape],
        "byte_order": "little",
        "memory_order": "C",
        "source_slice": field.source_slice,
        "preprocessing": list(field.preprocessing),
    }
    return canonical, payload, header


def _encode_row(row: RowMaterialization) -> dict[str, Any]:
    ledger = [field.name for field in row.fields]
    if not ledger or len(ledger) != len(set(ledger)):
        raise ValueError(f"{row.row_id}: empty or duplicate field ledger")
    digest_input = bytearray()
    records: list[dict[str, Any]] = []
    for field in row.fields:
        _array, payload, header = _numeric_payload(field)
        header_bytes = _canonical_json_bytes(header)
        header_frame = _frame(header_bytes)
        payload_frame = _frame(payload)
        digest_input.extend(header_frame)
        digest_input.extend(payload_frame)
        records.append(
            {
                "field_name": field.name,
                "header": header,
                "header_sha256": _sha256_bytes(header_bytes),
                "payload_sha256": _sha256_bytes(payload),
                "payload_base64": base64.b64encode(payload).decode("ascii"),
                "field_frames_sha256": _sha256_bytes(header_frame + payload_frame),
            }
        )
    semantics = dict(row.semantics)
    if semantics.get("row_id") != row.row_id:
        raise ValueError(f"{row.row_id}: semantics row identity mismatch")
    if semantics.get("authoritative_ordered_field_ledger") != ledger:
        raise ValueError(f"{row.row_id}: semantics field-ledger mismatch")
    semantics_bytes = _canonical_json_bytes(semantics)
    semantics_frame = _frame(semantics_bytes)
    digest_input.extend(semantics_frame)
    return {
        "row_id": row.row_id,
        "authoritative_ordered_field_ledger": ledger,
        "fields": records,
        "semantics": semantics,
        "semantics_sha256": _sha256_bytes(semantics_bytes),
        "semantics_frame_sha256": _sha256_bytes(semantics_frame),
        "row_sha256": _sha256_bytes(bytes(digest_input)),
    }


def _field(name: str, value: Any, source_slice: str, *preprocessing: str) -> TargetField:
    return TargetField(name, value, source_slice, tuple(preprocessing))


def _common_semantics(
    *,
    row_id: str,
    field_names: Sequence[str],
    theta_names: Sequence[str],
    theta_values: Sequence[float],
    time_steps: int,
    state_dim: int,
    observation_dim: int,
    dataset_seed: int,
    initial_time_convention: str,
    transition_density: str,
    observation_density: str,
    normalization: str,
    source_paths: Sequence[str],
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    if len(theta_names) != len(theta_values):
        raise ValueError(f"{row_id}: theta name/value mismatch")
    return {
        "row_id": row_id,
        "row_scope": "main_observed_data_filtering_row",
        "authoritative_ordered_field_ledger": list(field_names),
        "time_steps": int(time_steps),
        "state_dimension": int(state_dim),
        "observation_dimension": int(observation_dim),
        "dataset_seed": int(dataset_seed),
        "initial_and_time_convention": initial_time_convention,
        "target_density_definition": {
            "transition": transition_density,
            "observation": observation_density,
        },
        "normalization_definition": normalization,
        "target_scalar": "total_observed_data_log_likelihood",
        "theta_names": list(theta_names),
        "theta_order": list(theta_names),
        "theta_values": [float(value) for value in theta_values],
        "generator_data_config_paths_sha256": _source_hashes(source_paths),
        "algorithm_specific_fields_excluded": [
            "initial_particles",
            "proposal_seed_bases",
            "transition_noise",
            "fixed_resampling_mask",
            "flow_observation_covariance",
            "transport_configuration",
        ],
        **dict(extra),
    }


def _lgssm_row() -> RowMaterialization:
    dataset = _lgssm_dataset(81100)
    model = _lgssm_benchmark_model()
    fields = (
        _field("observations", dataset["observations"][:50], "dataset observations[0:50, :]", "no value conversion", "full T=50 slice"),
        _field("evaluation_theta", tf.constant(LGSSM_THETA, tf.float64), "full ordered vector", "explicit float64 construction"),
        _field("initial_mean", model.initial_mean, "full model field", "no preprocessing"),
        _field("initial_covariance", model.initial_covariance, "full model field", "symmetric model covariance"),
        _field("transition_offset", model.transition_offset, "full model field", "default zero offset materialized by model"),
        _field("transition_matrix", model.transition_matrix, "full model field", "no preprocessing"),
        _field("transition_covariance", model.transition_covariance, "full model field", "symmetric model covariance"),
        _field("observation_offset", model.observation_offset, "full model field", "default zero offset materialized by model"),
        _field("observation_matrix", model.observation_matrix, "full model field", "no preprocessing"),
        _field("observation_covariance", model.observation_covariance, "full model field", "symmetric model covariance"),
    )
    semantics = _common_semantics(
        row_id=LGSSM_M3_T50_ROW_ID,
        field_names=[field.name for field in fields],
        theta_names=LGSSM_M3_T50_PARAMETER_ORDER,
        theta_values=LGSSM_THETA,
        time_steps=50,
        state_dim=3,
        observation_dim=3,
        dataset_seed=81100,
        initial_time_convention="x0 is drawn from the stationary diagonal Gaussian; y0 is emitted from x0; transitions begin at t=1",
        transition_density="Gaussian with diagonal phi transition, zero offset, and q_scale squared identity covariance",
        observation_density="Gaussian with fixed full observation matrix, zero offset, and r_scale squared identity covariance",
        normalization="all Gaussian normalizing constants retained; total is the sum over all 50 observed-data increments",
        source_paths=(
            "docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py",
            "scripts/filtering_value_gradient_benchmark_generate_p8_datasets.py",
            "bayesfilter/highdim/models.py",
            "bayesfilter/highdim/ledh_forward_contract.py",
        ),
        extra={
            "theta_coordinate_system": "physical_benchmark_exact_oracle",
            "data_identity": "deterministic TensorFlow generator seed 81100",
            "source_relation": "BayesFilter exact-oracle benchmark; not a Zhao-Cui MATLAB rng(0) reproduction",
        },
    )
    return RowMaterialization(LGSSM_M3_T50_ROW_ID, fields, semantics)


def _actual_sv_row() -> RowMaterialization:
    dataset = _sv_dataset(81101)
    raw = tf.convert_to_tensor(dataset["observations"], tf.float64)[:1000]
    transformed = exact_transformed_sv_observations(raw)
    model = StochasticVolatilitySSM(sigma=1.0)
    fields = (
        _field("raw_observations", raw, "dataset observations[0:1000, :]", "explicit float64 tensor view", "full T=1000 slice"),
        _field("target_observations", transformed, "raw_observations[0:1000, :]", "elementwise log(y_t^2)", "zero offset"),
        _field("evaluation_theta", tf.constant(SV_THETA, tf.float64), "full ordered vector", "explicit float64 construction"),
        _field("fixed_sigma", model.sigma, "model fixed scalar", "explicit float64 model parameter"),
    )
    semantics = _common_semantics(
        row_id=ACTUAL_SV_ROW_ID,
        field_names=[field.name for field in fields],
        theta_names=SV_SYNTHETIC_PARAMETER_ORDER,
        theta_values=SV_THETA,
        time_steps=1000,
        state_dim=1,
        observation_dim=1,
        dataset_seed=81101,
        initial_time_convention="x0 uses the stationary SV prior and the t=0 target factor is initial density times exact transformed observation density",
        transition_density="x_t | x_(t-1) is Normal(gamma*x_(t-1), sigma^2) for t>0",
        observation_density="z_t - 2*log(beta) - x_t has exact log-chi-square(1) density, where z_t=log(y_t^2)",
        normalization="exact log-chi-square and Gaussian normalizing constants retained; total is the sum over 1000 increments",
        source_paths=(
            "docs/benchmarks/benchmark_ledh_same_target_actual_sv_value.py",
            "scripts/filtering_value_gradient_benchmark_generate_p8_datasets.py",
            "bayesfilter/highdim/models.py",
            "bayesfilter/highdim/sv_mixture_cut4.py",
            "bayesfilter/highdim/ledh_forward_contract.py",
        ),
        extra={
            "theta_coordinate_system": "gamma=Phi(theta[0]), beta=exp(theta[1]), sigma=1",
            "target_transform": "exact_log_y_square",
            "target_transform_offset": 0.0,
            "proposal_surface_excluded_from_target_signature": True,
        },
    )
    return RowMaterialization(ACTUAL_SV_ROW_ID, fields, semantics)


def _ksc_sv_row() -> RowMaterialization:
    dataset = _sv_dataset(81101)
    raw = tf.convert_to_tensor(dataset["observations"], tf.float64)[:1000]
    transformed = transformed_sv_observations(raw, offset=KSC_TRANSFORM_OFFSET)
    model = StochasticVolatilitySSM(sigma=1.0)
    mixture = ksc_1998_log_chi_square_mixture()
    fields = (
        _field("raw_observations", raw, "actual-SV source observations[0:1000, :]", "explicit float64 tensor view", "full T=1000 slice"),
        _field("target_observations", transformed, "raw_observations[0:1000, :]", "elementwise log(y_t^2 + 1e-8)"),
        _field("evaluation_theta", tf.constant(SV_THETA, tf.float64), "full ordered vector", "explicit float64 construction"),
        _field("fixed_sigma", model.sigma, "model fixed scalar", "explicit float64 model parameter"),
        _field("mixture_weights", mixture.weights, "full seven-component vector", "no preprocessing"),
        _field("mixture_means", mixture.means, "full seven-component vector", "no preprocessing"),
        _field("mixture_variances", mixture.variances, "full seven-component vector", "no preprocessing"),
    )
    semantics = _common_semantics(
        row_id=KSC_SV_ROW_ID,
        field_names=[field.name for field in fields],
        theta_names=SV_SYNTHETIC_PARAMETER_ORDER,
        theta_values=SV_THETA,
        time_steps=1000,
        state_dim=1,
        observation_dim=1,
        dataset_seed=81101,
        initial_time_convention="x0 uses the stationary SV prior and the t=0 target factor is initial density times KSC mixture observation density",
        transition_density="x_t | x_(t-1) is Normal(gamma*x_(t-1), sigma^2) for t>0",
        observation_density="finite seven-component KSC Gaussian-mixture density for log(y_t^2+1e-8)-2*log(beta)-x_t",
        normalization="mixture weights and every Gaussian normalizing constant retained; total is the sum over 1000 increments",
        source_paths=(
            "docs/benchmarks/benchmark_ledh_same_target_ksc_sv_value.py",
            "scripts/filtering_value_gradient_benchmark_generate_p8_datasets.py",
            "bayesfilter/highdim/models.py",
            "bayesfilter/highdim/sv_mixture_cut4.py",
            "bayesfilter/highdim/ledh_forward_contract.py",
        ),
        extra={
            "theta_coordinate_system": "gamma=Phi(theta[0]), beta=exp(theta[1]), sigma=1",
            "source_model_row_id": ACTUAL_SV_ROW_ID,
            "target_transform": "log_y_square_plus_offset",
            "target_transform_offset": KSC_TRANSFORM_OFFSET,
            "mixture_source": mixture.source,
            "mixture_component_count": mixture.component_count,
        },
    )
    return RowMaterialization(KSC_SV_ROW_ID, fields, semantics)


def _sir_row() -> RowMaterialization:
    dataset = _sir_dataset(81103)
    parameterized = parameterized_zhao_cui_sir_austria_model()
    model = parameterized.base_model
    fields = (
        _field("observations", dataset["observations"][:20], "synthetic Austria-model observations[0:20, :]", "explicit float64 generator output", "full T=20 slice"),
        _field("evaluation_theta", tf.constant(SIR_THETA, tf.float64), "full ordered vector", "explicit float64 construction"),
        _field("base_kappa", model.kappa, "full nine-region base vector", "no preprocessing"),
        _field("base_nu", model.nu, "full nine-region base vector", "no preprocessing"),
        _field("initial_mean", model.initial_mean, "full 18-state vector", "no preprocessing"),
        _field("initial_covariance", model.initial_covariance, "full model field", "symmetric model covariance"),
        _field("process_covariance", model.process_covariance, "full model field", "symmetric model covariance"),
        _field("base_observation_covariance", model.observation_covariance, "full model field", "symmetric model covariance"),
        _field("adjacency_matrix", model._adjacency_matrix, "full nine-region matrix", "derived exactly from declared neighbor sets"),
        _field("delta", model.delta, "model scalar", "no preprocessing"),
        _field("rk4_internal_step", model.rk4_internal_step, "model scalar", "no preprocessing"),
    )
    semantics = _common_semantics(
        row_id=FIXED_SIR_AUSTRIA_ROW_ID,
        field_names=[field.name for field in fields],
        theta_names=SIR_LOG_SCALE_PARAMETER_ORDER,
        theta_values=SIR_THETA,
        time_steps=20,
        state_dim=18,
        observation_dim=9,
        dataset_seed=81103,
        initial_time_convention="x0 is drawn from the declared 18-dimensional Gaussian and y0 observes its infectious coordinates; transitions begin at t=1",
        transition_density="Gaussian around the source-shaped Austria SIR RK4 mean; kappa and nu are scaled by exp(theta[0:2])",
        observation_density="nine-dimensional Gaussian around infectious coordinates; covariance is base covariance times exp(2*theta[2])",
        normalization="all multivariate Gaussian normalizing constants retained; total is the sum over 20 observed-data increments",
        source_paths=(
            "docs/benchmarks/benchmark_ledh_same_target_fixed_sir_score.py",
            "docs/benchmarks/benchmark_p8p_parameterized_sir_gradient.py",
            "docs/benchmarks/benchmark_p8j_tf32_batched_actual_sir.py",
            "scripts/filtering_value_gradient_benchmark_generate_p8_datasets.py",
            "scripts/filtering_value_gradient_benchmark_run_p8d_numeric.py",
            "bayesfilter/highdim/models.py",
            "bayesfilter/highdim/ledh_forward_contract.py",
        ),
        extra={
            "theta_coordinate_system": "sir_log_scale_theta",
            "theta_zero_semantics": "reproduces the fixed source-shaped Austria base parameters",
            "target_generation_identity": SIR_TARGET_GENERATION_IDENTITY,
            "observation_data_status": "authoritative fixed BayesFilter observations generated by _sir_dataset(81103)",
            "author_matlab_rng1_reproduction_claimed": False,
            "source_adaptation_classification_carried_from_model": "extension_or_invention",
            "process_noise_policy": model.process_noise_policy,
            "rk4_variant": model.rk4_variant,
        },
    )
    return RowMaterialization(FIXED_SIR_AUSTRIA_ROW_ID, fields, semantics)


def _predator_prey_row() -> RowMaterialization:
    dataset = _predator_prey_dataset(81104)
    model = p30_predator_prey_fixture_model()
    fields = (
        _field("observations", dataset["observations"][:20], "dataset observations[0:20, :]", "explicit float64 generator output", "full T=20 slice"),
        _field("evaluation_theta", tf.constant(PREDATOR_THETA, tf.float64), "full ordered vector", "explicit float64 construction"),
        _field("initial_mean", model.initial_mean, "full model field", "no preprocessing"),
        _field("initial_covariance", model.initial_covariance, "full model field", "symmetric model covariance"),
        _field("process_covariance", model.process_covariance, "full model field", "symmetric model covariance"),
        _field("observation_covariance", model.observation_covariance, "full model field", "symmetric model covariance"),
        _field("delta", model.delta, "model scalar", "no preprocessing"),
        _field("rk4_internal_step", model.rk4_internal_step, "model scalar", "no preprocessing"),
    )
    semantics = _common_semantics(
        row_id=PREDATOR_PREY_ROW_ID,
        field_names=[field.name for field in fields],
        theta_names=PREDATOR_PREY_PARAMETER_ORDER,
        theta_values=PREDATOR_THETA,
        time_steps=20,
        state_dim=2,
        observation_dim=2,
        dataset_seed=81104,
        initial_time_convention="x0 is drawn from the declared Gaussian and y0 observes x0; RK4 transitions begin at t=1",
        transition_density="Gaussian around the 20-substep classical RK4 predator-prey mean at physical theta",
        observation_density="two-dimensional additive Gaussian identity-state observation density",
        normalization="all multivariate Gaussian normalizing constants retained; total is the sum over 20 observed-data increments",
        source_paths=(
            "docs/benchmarks/benchmark_ledh_same_target_predator_prey_value.py",
            "scripts/filtering_value_gradient_benchmark_generate_p8_datasets.py",
            "bayesfilter/highdim/models.py",
            "bayesfilter/highdim/ledh_forward_contract.py",
        ),
        extra={
            "theta_coordinate_system": "physical",
            "rk4_substeps": int(model._rk4_substeps),
            "domain_policy": model.domain_policy,
        },
    )
    return RowMaterialization(PREDATOR_PREY_ROW_ID, fields, semantics)


def _generalized_sv_row() -> RowMaterialization:
    dataset = _generalized_sv_prior_mean_dataset(81105)
    raw = tf.convert_to_tensor(dataset["observations"], tf.float64)[:1008]
    fields = (
        _field("raw_observations", raw, "dataset observations[0:1008, :]", "explicit float64 generator output", "full T=1008 slice"),
        _field("evaluation_theta", tf.constant(GENERALIZED_SV_THETA, tf.float64), "full ordered vector", "explicit float64 construction"),
        _field("transition_innovation_variance", tf.constant(1.0, tf.float64), "fixed source-route scalar", "explicit float64 construction"),
        _field("fixed_context", tf.constant([0.0, 0.0, 0.0, 0.0, 0.0], tf.float64), "ordered vector [phi,a,delta,inv_nu1,inv_nu2]", "infinite nu1 and nu2 encoded as zero inverse degrees of freedom"),
    )
    semantics = _common_semantics(
        row_id=GENERALIZED_SV_ROW_ID,
        field_names=[field.name for field in fields],
        theta_names=GENERALIZED_SV_PARAMETER_ORDER,
        theta_values=GENERALIZED_SV_THETA,
        time_steps=1008,
        state_dim=1,
        observation_dim=1,
        dataset_seed=81105,
        initial_time_convention="initial particles represent stationary previous states; every observation including t=0 first applies the AR(1) transition",
        transition_density="x_t is Normal(mu + gamma*(x_(t-1)-mu), 1), gamma=Phi(theta[0]), tau=exp(theta[1]), mu=theta[2]",
        observation_density="raw y_t is zero-mean Normal with log variance tau*x_t because phi=a=delta=0 and nu2 is infinite",
        normalization="all univariate Gaussian normalizing constants retained; total is the sum over 1008 post-transition observation increments",
        source_paths=(
            "docs/benchmarks/benchmark_ledh_same_target_generalized_sv_value.py",
            "scripts/filtering_value_gradient_benchmark_generate_p8_datasets.py",
            "bayesfilter/highdim/ledh_forward_contract.py",
            "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/svmodels/ftt2true.m",
        ),
        extra={
            "theta_coordinate_system": "source_route_active_transformed_prior_mean",
            "fixed_context_names": ["phi", "a", "delta", "inverse_nu1", "inverse_nu2"],
            "source_route": "zhao_cui_svmodels_prior_mean_synthetic",
            "flow_log_square_observations_excluded_as_proposal_only": True,
            "sp500_returns_used": False,
        },
    )
    return RowMaterialization(GENERALIZED_SV_ROW_ID, fields, semantics)


ROW_BUILDERS: tuple[Callable[[], RowMaterialization], ...] = (
    _lgssm_row,
    _actual_sv_row,
    _ksc_sv_row,
    _sir_row,
    _predator_prey_row,
    _generalized_sv_row,
)


def build_artifact() -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("Phase 1 canonical target materialization must hide GPU devices")
    rows = [_encode_row(builder()) for builder in ROW_BUILDERS]
    row_ids = [row["row_id"] for row in rows]
    if len(rows) != 6 or len(set(row_ids)) != 6:
        raise ValueError("canonical target artifact requires exactly six unique rows")
    return {
        "schema_version": SCHEMA_VERSION,
        "encoding_id": ENCODING_ID,
        "run_id": RUN_ID,
        "artifact_date": ARTIFACT_DATE,
        "execution_scope": "cpu_hidden_target_materialization_only",
        "cuda_visible_devices": "-1",
        "framework": {
            "tensorflow_version": tf.__version__,
            "numpy_version": np.__version__,
        },
        "canonical_encoding": {
            "header_json": "sort_keys=True,separators=(',',':'),allow_nan=False,UTF-8",
            "frame_length": "unsigned 64-bit network/big-endian",
            "numeric_payload": "C-contiguous raw bytes in canonical little-endian order without value or precision conversion",
            "frame_order": "alternating header,payload for each ordered field, then one final semantics frame",
        },
        "rows": rows,
        "summary": {
            "row_ids": row_ids,
            "row_count": len(rows),
            "all_row_digests_unique": len({row["row_sha256"] for row in rows}) == len(rows),
            "filter_executed": False,
            "leaderboard_cell_admitted": False,
        },
        "nonclaims": [
            "not evaluator correctness evidence",
            "not GPU/XLA execution evidence",
            "not Zhao-Cui source-faithfulness approval",
            "not leaderboard cell admission",
            "not ranking or scientific validity evidence",
        ],
    }


def _serialized_artifact() -> str:
    return json.dumps(build_artifact(), indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    expected = _serialized_artifact()
    if args.check:
        if not output.is_file():
            raise SystemExit(f"missing canonical target artifact: {output}")
        if output.read_text(encoding="utf-8") != expected:
            raise SystemExit("canonical target artifact drift")
        print("PASS_PHASE1_CANONICAL_TARGET_BUILDER_CHECK")
        return
    _write_atomic(output, expected)
    print(f"WROTE {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
