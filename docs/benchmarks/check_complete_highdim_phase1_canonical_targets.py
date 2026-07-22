"""Independently verify the Phase 1 canonical target-signature artifact."""

from __future__ import annotations

import argparse
import base64
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


EXPECTED_SCHEMA = "bayesfilter.complete_highdim.canonical_targets.v1"
EXPECTED_ENCODING = "length_prefixed_header_payload_then_semantics.sha256.v1"
EXPECTED_RUN_ID = "complete-highdim-leaderboard-local-20260712-134906"
SIR_TARGET_GENERATION_IDENTITY = (
    "fixed_bayesfilter_sir_observations_from_dataset_seed_81103_"
    "not_author_matlab_rng1_reproduction"
)
DEFAULT_ARTIFACT = ROOT / (
    "docs/plans/artifacts/complete-highdim-leaderboard/"
    "phase1-canonical-targets-2026-07-11.json"
)


# These ledgers are intentionally encoded independently of the builder.
EXPECTED_ROWS: Mapping[str, Mapping[str, Any]] = {
    "benchmark_lgssm_exact_oracle_m3_T50": {
        "ledger": (
            "observations",
            "evaluation_theta",
            "initial_mean",
            "initial_covariance",
            "transition_offset",
            "transition_matrix",
            "transition_covariance",
            "observation_offset",
            "observation_matrix",
            "observation_covariance",
        ),
        "theta_names": ("phi1", "phi2", "phi3", "q_scale", "r_scale"),
        "theta_values": (0.72, 0.55, 0.35, 0.35, 0.45),
        "time_steps": 50,
        "state_dimension": 3,
        "observation_dimension": 3,
        "dataset_seed": 81100,
        "sources": (
            "docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py",
            "scripts/filtering_value_gradient_benchmark_generate_p8_datasets.py",
            "bayesfilter/highdim/models.py",
            "bayesfilter/highdim/ledh_forward_contract.py",
        ),
    },
    "zhao_cui_sv_actual_nongaussian_T1000": {
        "ledger": (
            "raw_observations",
            "target_observations",
            "evaluation_theta",
            "fixed_sigma",
        ),
        "theta_names": ("gamma_unconstrained", "log_beta"),
        "theta_values": (0.2533471031357997, -0.916290731874155),
        "time_steps": 1000,
        "state_dimension": 1,
        "observation_dimension": 1,
        "dataset_seed": 81101,
        "sources": (
            "docs/benchmarks/benchmark_ledh_same_target_actual_sv_value.py",
            "scripts/filtering_value_gradient_benchmark_generate_p8_datasets.py",
            "bayesfilter/highdim/models.py",
            "bayesfilter/highdim/sv_mixture_cut4.py",
            "bayesfilter/highdim/ledh_forward_contract.py",
        ),
    },
    "zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000": {
        "ledger": (
            "raw_observations",
            "target_observations",
            "evaluation_theta",
            "fixed_sigma",
            "mixture_weights",
            "mixture_means",
            "mixture_variances",
        ),
        "theta_names": ("gamma_unconstrained", "log_beta"),
        "theta_values": (0.2533471031357997, -0.916290731874155),
        "time_steps": 1000,
        "state_dimension": 1,
        "observation_dimension": 1,
        "dataset_seed": 81101,
        "sources": (
            "docs/benchmarks/benchmark_ledh_same_target_ksc_sv_value.py",
            "scripts/filtering_value_gradient_benchmark_generate_p8_datasets.py",
            "bayesfilter/highdim/models.py",
            "bayesfilter/highdim/sv_mixture_cut4.py",
            "bayesfilter/highdim/ledh_forward_contract.py",
        ),
    },
    "zhao_cui_spatial_sir_austria_j9_T20": {
        "ledger": (
            "observations",
            "evaluation_theta",
            "base_kappa",
            "base_nu",
            "initial_mean",
            "initial_covariance",
            "process_covariance",
            "base_observation_covariance",
            "adjacency_matrix",
            "delta",
            "rk4_internal_step",
        ),
        "theta_names": (
            "log_kappa_scale",
            "log_nu_scale",
            "log_obs_noise_scale",
        ),
        "theta_values": (0.0, 0.0, 0.0),
        "time_steps": 20,
        "state_dimension": 18,
        "observation_dimension": 9,
        "dataset_seed": 81103,
        "required_semantics": {
            "target_generation_identity": SIR_TARGET_GENERATION_IDENTITY,
            "observation_data_status": (
                "authoritative fixed BayesFilter observations generated by "
                "_sir_dataset(81103)"
            ),
            "author_matlab_rng1_reproduction_claimed": False,
            "source_adaptation_classification_carried_from_model": (
                "extension_or_invention"
            ),
        },
        "sources": (
            "docs/benchmarks/benchmark_ledh_same_target_fixed_sir_score.py",
            "docs/benchmarks/benchmark_p8p_parameterized_sir_gradient.py",
            "docs/benchmarks/benchmark_p8j_tf32_batched_actual_sir.py",
            "scripts/filtering_value_gradient_benchmark_generate_p8_datasets.py",
            "scripts/filtering_value_gradient_benchmark_run_p8d_numeric.py",
            "bayesfilter/highdim/models.py",
            "bayesfilter/highdim/ledh_forward_contract.py",
        ),
    },
    "zhao_cui_predator_prey_T20": {
        "ledger": (
            "observations",
            "evaluation_theta",
            "initial_mean",
            "initial_covariance",
            "process_covariance",
            "observation_covariance",
            "delta",
            "rk4_internal_step",
        ),
        "theta_names": ("r", "K", "a", "s", "u", "v"),
        "theta_values": (0.6, 114.0, 25.0, 0.3, 0.5, 0.5),
        "time_steps": 20,
        "state_dimension": 2,
        "observation_dimension": 2,
        "dataset_seed": 81104,
        "sources": (
            "docs/benchmarks/benchmark_ledh_same_target_predator_prey_value.py",
            "scripts/filtering_value_gradient_benchmark_generate_p8_datasets.py",
            "bayesfilter/highdim/models.py",
            "bayesfilter/highdim/ledh_forward_contract.py",
        ),
    },
    "zhao_cui_generalized_sv_synthetic_from_estimated_values": {
        "ledger": (
            "raw_observations",
            "evaluation_theta",
            "transition_innovation_variance",
            "fixed_context",
        ),
        "theta_names": ("gamma_unconstrained", "log_tau", "mu"),
        "theta_values": (1.0824113944610982, -2.076793740349318, 0.0),
        "time_steps": 1008,
        "state_dimension": 1,
        "observation_dimension": 1,
        "dataset_seed": 81105,
        "sources": (
            "docs/benchmarks/benchmark_ledh_same_target_generalized_sv_value.py",
            "scripts/filtering_value_gradient_benchmark_generate_p8_datasets.py",
            "bayesfilter/highdim/ledh_forward_contract.py",
            "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/svmodels/ftt2true.m",
        ),
    },
}


FORBIDDEN_TARGET_FIELDS = frozenset(
    {
        "initial_particles",
        "proposal_seed_bases",
        "transition_noise",
        "fixed_resampling_mask",
        "flow_observation_covariance",
        "transport_configuration",
    }
)


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


def _canonical_array(value: Any) -> np.ndarray:
    raw = value.numpy() if tf.is_tensor(value) else value
    array = np.asarray(raw)
    if array.dtype.hasobject or array.dtype.kind not in {"b", "i", "u", "f", "c"}:
        raise TypeError(f"independent expected value has invalid dtype {array.dtype}")
    if array.dtype.kind in {"f", "c"} and not bool(np.all(np.isfinite(array))):
        raise ValueError("independent expected value is nonfinite")
    return np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))


def _expected_lgssm() -> Mapping[str, Any]:
    dataset = _lgssm_dataset(81100)
    model = _lgssm_benchmark_model()
    return {
        "observations": dataset["observations"][:50],
        "evaluation_theta": tf.constant([0.72, 0.55, 0.35, 0.35, 0.45], tf.float64),
        "initial_mean": model.initial_mean,
        "initial_covariance": model.initial_covariance,
        "transition_offset": model.transition_offset,
        "transition_matrix": model.transition_matrix,
        "transition_covariance": model.transition_covariance,
        "observation_offset": model.observation_offset,
        "observation_matrix": model.observation_matrix,
        "observation_covariance": model.observation_covariance,
    }


def _expected_actual_sv() -> Mapping[str, Any]:
    raw = tf.convert_to_tensor(_sv_dataset(81101)["observations"], tf.float64)[:1000]
    return {
        "raw_observations": raw,
        "target_observations": exact_transformed_sv_observations(raw),
        "evaluation_theta": tf.constant([0.2533471031357997, -0.916290731874155], tf.float64),
        "fixed_sigma": StochasticVolatilitySSM(sigma=1.0).sigma,
    }


def _expected_ksc_sv() -> Mapping[str, Any]:
    raw = tf.convert_to_tensor(_sv_dataset(81101)["observations"], tf.float64)[:1000]
    mixture = ksc_1998_log_chi_square_mixture()
    return {
        "raw_observations": raw,
        "target_observations": transformed_sv_observations(raw, offset=1.0e-8),
        "evaluation_theta": tf.constant([0.2533471031357997, -0.916290731874155], tf.float64),
        "fixed_sigma": StochasticVolatilitySSM(sigma=1.0).sigma,
        "mixture_weights": mixture.weights,
        "mixture_means": mixture.means,
        "mixture_variances": mixture.variances,
    }


def _expected_sir() -> Mapping[str, Any]:
    model = parameterized_zhao_cui_sir_austria_model().base_model
    return {
        "observations": _sir_dataset(81103)["observations"][:20],
        "evaluation_theta": tf.constant([0.0, 0.0, 0.0], tf.float64),
        "base_kappa": model.kappa,
        "base_nu": model.nu,
        "initial_mean": model.initial_mean,
        "initial_covariance": model.initial_covariance,
        "process_covariance": model.process_covariance,
        "base_observation_covariance": model.observation_covariance,
        "adjacency_matrix": model._adjacency_matrix,
        "delta": model.delta,
        "rk4_internal_step": model.rk4_internal_step,
    }


def _expected_predator_prey() -> Mapping[str, Any]:
    model = p30_predator_prey_fixture_model()
    return {
        "observations": _predator_prey_dataset(81104)["observations"][:20],
        "evaluation_theta": tf.constant([0.6, 114.0, 25.0, 0.3, 0.5, 0.5], tf.float64),
        "initial_mean": model.initial_mean,
        "initial_covariance": model.initial_covariance,
        "process_covariance": model.process_covariance,
        "observation_covariance": model.observation_covariance,
        "delta": model.delta,
        "rk4_internal_step": model.rk4_internal_step,
    }


def _expected_generalized_sv() -> Mapping[str, Any]:
    return {
        "raw_observations": tf.convert_to_tensor(
            _generalized_sv_prior_mean_dataset(81105)["observations"], tf.float64
        )[:1008],
        "evaluation_theta": tf.constant(
            [1.0824113944610982, -2.076793740349318, 0.0], tf.float64
        ),
        "transition_innovation_variance": tf.constant(1.0, tf.float64),
        "fixed_context": tf.constant([0.0, 0.0, 0.0, 0.0, 0.0], tf.float64),
    }


EXPECTED_VALUE_BUILDERS: Mapping[str, Callable[[], Mapping[str, Any]]] = {
    "benchmark_lgssm_exact_oracle_m3_T50": _expected_lgssm,
    "zhao_cui_sv_actual_nongaussian_T1000": _expected_actual_sv,
    "zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000": _expected_ksc_sv,
    "zhao_cui_spatial_sir_austria_j9_T20": _expected_sir,
    "zhao_cui_predator_prey_T20": _expected_predator_prey,
    "zhao_cui_generalized_sv_synthetic_from_estimated_values": _expected_generalized_sv,
}


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _check_sources(semantics: Mapping[str, Any], expected: Mapping[str, Any], row_id: str) -> None:
    declared = _require_dict(
        semantics.get("generator_data_config_paths_sha256"),
        f"{row_id}.generator_data_config_paths_sha256",
    )
    expected_paths = tuple(expected["sources"])
    if set(declared) != set(expected_paths) or len(declared) != len(expected_paths):
        raise ValueError(f"{row_id}: source path ledger mismatch")
    for relative in expected_paths:
        path = ROOT / relative
        if not path.is_file():
            raise ValueError(f"{row_id}: missing source path {relative}")
        if declared[relative] != _sha256_file(path):
            raise ValueError(f"{row_id}: stale source hash for {relative}")


def _check_field(
    *,
    row_id: str,
    record: Mapping[str, Any],
    field_name: str,
    expected_value: Any,
) -> tuple[bytes, bytes]:
    if record.get("field_name") != field_name:
        raise ValueError(f"{row_id}: field record order mismatch at {field_name}")
    header = _require_dict(record.get("header"), f"{row_id}.{field_name}.header")
    required_header = {
        "field_name",
        "dtype_descriptor",
        "dtype_kind",
        "dtype_itemsize",
        "shape",
        "byte_order",
        "memory_order",
        "source_slice",
        "preprocessing",
    }
    if set(header) != required_header:
        raise ValueError(f"{row_id}.{field_name}: header key mismatch")
    if header["field_name"] != field_name:
        raise ValueError(f"{row_id}.{field_name}: header field mismatch")
    if header["byte_order"] != "little" or header["memory_order"] != "C":
        raise ValueError(f"{row_id}.{field_name}: byte or memory order mismatch")
    if not isinstance(header["source_slice"], str) or not header["source_slice"]:
        raise ValueError(f"{row_id}.{field_name}: missing source slice")
    if not isinstance(header["preprocessing"], list) or not header["preprocessing"]:
        raise ValueError(f"{row_id}.{field_name}: missing preprocessing ledger")
    dtype_descriptor = header["dtype_descriptor"]
    if not isinstance(dtype_descriptor, str):
        raise ValueError(f"{row_id}.{field_name}: dtype descriptor is not a string")
    dtype = np.dtype(dtype_descriptor)
    if dtype.hasobject or dtype.kind not in {"b", "i", "u", "f", "c"}:
        raise ValueError(f"{row_id}.{field_name}: forbidden dtype")
    if dtype.kind != header["dtype_kind"] or dtype.itemsize != header["dtype_itemsize"]:
        raise ValueError(f"{row_id}.{field_name}: dtype metadata mismatch")
    if dtype.itemsize > 1 and not dtype_descriptor.startswith("<"):
        raise ValueError(f"{row_id}.{field_name}: payload dtype is not explicit little endian")
    if dtype.itemsize == 1 and dtype_descriptor[0] not in {"|", "<"}:
        raise ValueError(f"{row_id}.{field_name}: one-byte dtype descriptor is ambiguous")
    shape = tuple(header["shape"])
    if not all(isinstance(item, int) and item >= 0 for item in shape):
        raise ValueError(f"{row_id}.{field_name}: invalid shape")
    try:
        payload = base64.b64decode(record.get("payload_base64", ""), validate=True)
    except Exception as exc:
        raise ValueError(f"{row_id}.{field_name}: invalid base64 payload") from exc
    if len(payload) != int(np.prod(shape, dtype=np.int64)) * dtype.itemsize:
        raise ValueError(f"{row_id}.{field_name}: payload byte length mismatch")
    observed = np.frombuffer(payload, dtype=dtype).reshape(shape, order="C")
    if observed.dtype.kind in {"f", "c"} and not bool(np.all(np.isfinite(observed))):
        raise ValueError(f"{row_id}.{field_name}: nonfinite payload")
    expected_array = _canonical_array(expected_value)
    if observed.shape != expected_array.shape or observed.dtype != expected_array.dtype:
        raise ValueError(f"{row_id}.{field_name}: independent shape/dtype mismatch")
    if payload != expected_array.tobytes(order="C"):
        raise ValueError(f"{row_id}.{field_name}: independent payload mismatch")
    header_bytes = _canonical_json_bytes(header)
    if record.get("header_sha256") != _sha256_bytes(header_bytes):
        raise ValueError(f"{row_id}.{field_name}: header digest mismatch")
    if record.get("payload_sha256") != _sha256_bytes(payload):
        raise ValueError(f"{row_id}.{field_name}: payload digest mismatch")
    if record.get("field_frames_sha256") != _sha256_bytes(_frame(header_bytes) + _frame(payload)):
        raise ValueError(f"{row_id}.{field_name}: framed field digest mismatch")
    return header_bytes, payload


def _check_row(row: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    row_id = row.get("row_id")
    ledger = tuple(row.get("authoritative_ordered_field_ledger", ()))
    expected_ledger = tuple(expected["ledger"])
    if ledger != expected_ledger:
        raise ValueError(f"{row_id}: independent field ledger mismatch")
    if FORBIDDEN_TARGET_FIELDS.intersection(ledger):
        raise ValueError(f"{row_id}: algorithm-specific field contaminated target signature")
    records = row.get("fields")
    if not isinstance(records, list) or len(records) != len(expected_ledger):
        raise ValueError(f"{row_id}: field record count mismatch")
    expected_values = EXPECTED_VALUE_BUILDERS[row_id]()
    if tuple(expected_values) != expected_ledger:
        raise ValueError(f"{row_id}: checker value-builder ledger mismatch")
    row_input = bytearray()
    for record, field_name in zip(records, expected_ledger, strict=True):
        header_bytes, payload = _check_field(
            row_id=row_id,
            record=_require_dict(record, f"{row_id}.{field_name}"),
            field_name=field_name,
            expected_value=expected_values[field_name],
        )
        row_input.extend(_frame(header_bytes))
        row_input.extend(_frame(payload))
    semantics = _require_dict(row.get("semantics"), f"{row_id}.semantics")
    if semantics.get("row_id") != row_id:
        raise ValueError(f"{row_id}: semantics row mismatch")
    if tuple(semantics.get("authoritative_ordered_field_ledger", ())) != expected_ledger:
        raise ValueError(f"{row_id}: semantics field ledger mismatch")
    for key in ("time_steps", "state_dimension", "observation_dimension", "dataset_seed"):
        if semantics.get(key) != expected[key]:
            raise ValueError(f"{row_id}: semantics {key} mismatch")
    if tuple(semantics.get("theta_names", ())) != tuple(expected["theta_names"]):
        raise ValueError(f"{row_id}: theta name mismatch")
    if tuple(semantics.get("theta_order", ())) != tuple(expected["theta_names"]):
        raise ValueError(f"{row_id}: theta order mismatch")
    if tuple(semantics.get("theta_values", ())) != tuple(expected["theta_values"]):
        raise ValueError(f"{row_id}: theta value mismatch")
    if semantics.get("target_scalar") != "total_observed_data_log_likelihood":
        raise ValueError(f"{row_id}: target scalar mismatch")
    for key in (
        "initial_and_time_convention",
        "target_density_definition",
        "normalization_definition",
    ):
        if not semantics.get(key):
            raise ValueError(f"{row_id}: missing target semantics {key}")
    for key, value in expected.get("required_semantics", {}).items():
        if semantics.get(key) != value:
            raise ValueError(f"{row_id}: required semantics {key} mismatch")
    if set(semantics.get("algorithm_specific_fields_excluded", ())) != FORBIDDEN_TARGET_FIELDS:
        raise ValueError(f"{row_id}: algorithm-specific exclusion ledger mismatch")
    _check_sources(semantics, expected, row_id)
    semantics_bytes = _canonical_json_bytes(semantics)
    semantics_frame = _frame(semantics_bytes)
    if row.get("semantics_sha256") != _sha256_bytes(semantics_bytes):
        raise ValueError(f"{row_id}: semantics digest mismatch")
    if row.get("semantics_frame_sha256") != _sha256_bytes(semantics_frame):
        raise ValueError(f"{row_id}: semantics frame digest mismatch")
    row_input.extend(semantics_frame)
    if row.get("row_sha256") != _sha256_bytes(bytes(row_input)):
        raise ValueError(f"{row_id}: row digest mismatch")


def check_artifact(path: Path) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("Phase 1 independent checker must hide GPU devices")
    artifact = _require_dict(json.loads(path.read_text(encoding="utf-8")), "artifact")
    if artifact.get("schema_version") != EXPECTED_SCHEMA:
        raise ValueError("canonical target schema mismatch")
    if artifact.get("encoding_id") != EXPECTED_ENCODING:
        raise ValueError("canonical target encoding mismatch")
    if artifact.get("run_id") != EXPECTED_RUN_ID:
        raise ValueError("canonical target run identity mismatch")
    if artifact.get("execution_scope") != "cpu_hidden_target_materialization_only":
        raise ValueError("canonical target execution scope mismatch")
    if artifact.get("cuda_visible_devices") != "-1":
        raise ValueError("canonical target artifact did not record hidden GPU devices")
    rows = artifact.get("rows")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_ROWS):
        raise ValueError("canonical target artifact must have exactly six rows")
    observed_ids = tuple(row.get("row_id") for row in rows if isinstance(row, dict))
    if observed_ids != tuple(EXPECTED_ROWS):
        raise ValueError("canonical target row order or identity mismatch")
    for row in rows:
        row_object = _require_dict(row, "row")
        _check_row(row_object, EXPECTED_ROWS[row_object["row_id"]])
    summary = _require_dict(artifact.get("summary"), "summary")
    if summary.get("row_ids") != list(EXPECTED_ROWS) or summary.get("row_count") != 6:
        raise ValueError("canonical target summary mismatch")
    if summary.get("filter_executed") is not False:
        raise ValueError("canonical target artifact cannot report filter execution")
    if summary.get("leaderboard_cell_admitted") is not False:
        raise ValueError("canonical target artifact cannot admit a cell")
    row_digests = [row["row_sha256"] for row in rows]
    if len(set(row_digests)) != 6 or summary.get("all_row_digests_unique") is not True:
        raise ValueError("canonical target row digests must be unique")
    return artifact


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    path = args.artifact if args.artifact.is_absolute() else ROOT / args.artifact
    check_artifact(path)
    print("PASS_PHASE1_CANONICAL_TARGET_INDEPENDENT_CHECK")


if __name__ == "__main__":
    main()
