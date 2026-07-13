"""Reconstruct the missing retained Phase 1/2 triangular LGSSM artifacts.

The historical result note retained the model definition, seed, and exact
array extrema, but the three referenced JSON artifacts were never committed.
This reporting/fixture harness reconstructs the deterministic NumPy reference
stream and emits fresh hashes. It does not reassert the unavailable historical
byte or payload hashes and does not run inference, training, HMC, or a GPU.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / (
    "docs/plans/artifacts/"
    "multidim-triangular-lgssm-neutra-hmc-2026-07-08"
)
CONTRACT_PATH = ARTIFACT_ROOT / "lower_triangular_lgssm_contract_v1.json"
DATA_PATH = ARTIFACT_ROOT / (
    "lower_triangular_lgssm_synthetic_data_v1_seed20260708.json"
)
MANIFEST_PATH = ARTIFACT_ROOT / (
    "lower_triangular_lgssm_synthetic_data_v1_manifest_seed20260708.json"
)

SCHEMA = "bayesfilter.multidim_triangular_lgssm.contract.v1"
CONTRACT_ID = "lower_triangular_lgssm_v1"
TARGET_ID = "bayesfilter_multidim_lower_triangular_lgssm_neutra_hmc_v1"
SEED = 20260708
HORIZON = 256

PARAMETER_NAMES = (
    "a11_raw",
    "a22_raw",
    "a33_raw",
    "a44_raw",
    "a21_raw",
    "a31_raw",
    "a32_raw",
    "a41_raw",
    "a42_raw",
    "a43_raw",
    "log_q1",
    "log_q2",
    "log_q3",
    "log_q4",
    "log_r1",
    "log_r2",
    "log_r3",
    "log_r4",
)

TRANSITION = np.asarray(
    [
        [0.62, 0.0, 0.0, 0.0],
        [0.18, 0.48, 0.0, 0.0],
        [-0.10, 0.14, 0.30, 0.0],
        [0.06, -0.08, 0.11, 0.16],
    ],
    dtype=np.float64,
)
PROCESS_STD = np.asarray([0.30, 0.26, 0.22, 0.18], dtype=np.float64)
OBSERVATION_STD = np.asarray([0.12, 0.11, 0.10, 0.09], dtype=np.float64)

HISTORICAL_PHASE2_HASHES = {
    "data_file_sha256": (
        "sha256:d3944c6c38f40031dbdfa28d17d1ac9650740604f3202b74113502ddcac6ae01"
    ),
    "data_payload_hash": (
        "sha256:84e80352e4293f8c888142a760bd81dafa52de52145d6c769bb0ecb827f7bcb4"
    ),
    "manifest_payload_hash": (
        "sha256:42711ed8c31b6806644c630860019a99ae7fe1da33b33b809ce4187d2fd1cdb0"
    ),
}
HISTORICAL_MAX_ABS_STATE = 1.0279111407619268
HISTORICAL_MAX_ABS_OBSERVATION = 1.0796169943915201


def canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def file_hash(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def stationary_covariance() -> np.ndarray:
    process_covariance = np.diag(np.square(PROCESS_STD))
    system = np.eye(16, dtype=np.float64) - np.kron(TRANSITION, TRANSITION)
    covariance = np.linalg.solve(system, process_covariance.reshape(16)).reshape(4, 4)
    return 0.5 * (covariance + covariance.T)


def raw_truth() -> np.ndarray:
    diagonal = np.diag(TRANSITION)
    lower = TRANSITION[np.tril_indices(4, k=-1)]
    # np.tril_indices orders entries by row, matching the declared raw order.
    return np.concatenate(
        (
            np.arctanh(diagonal / 0.85),
            np.arctanh(lower / 0.35),
            np.log(PROCESS_STD),
            np.log(OBSERVATION_STD),
        )
    )


def simulate(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    state = rng.multivariate_normal(np.zeros(4, dtype=np.float64), covariance)
    states: list[np.ndarray] = []
    observations: list[np.ndarray] = []
    for time_index in range(HORIZON):
        states.append(state.copy())
        observations.append(state + OBSERVATION_STD * rng.normal(size=4))
        if time_index + 1 < HORIZON:
            state = TRANSITION @ state + PROCESS_STD * rng.normal(size=4)
    return np.asarray(states), np.asarray(observations)


def build_contract() -> dict[str, Any]:
    contract: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "target_id": TARGET_ID,
        "static_shape": {
            "state_dim": 4,
            "observation_dim": 4,
            "innovation_dim": 4,
            "parameter_dim": 18,
            "first_fixture_horizon": HORIZON,
        },
        "parameter_names": list(PARAMETER_NAMES),
        "transform": {
            "diagonal_id": "rho_max_times_tanh",
            "lower_id": "lower_scale_times_tanh",
            "positive_std_id": "exp",
            "rho_max": 0.85,
            "lower_scale": 0.35,
        },
        "model_manifest": {
            "transition_matrix": "lower_triangular_A",
            "observation_matrix": "H_identity",
            "process_covariance": "diagonal_Q",
            "observation_covariance": "diagonal_R",
            "stationary_mean": "fixed_zero",
            "initial_covariance": "stationary_initial_covariance",
            "stationary_equation": "P_inf_equals_A_P_inf_A_transpose_plus_Q",
        },
        "truth_template": {
            "diag_A": [0.62, 0.48, 0.30, 0.16],
            "lower_A": {
                "a21": 0.18,
                "a31": -0.10,
                "a32": 0.14,
                "a41": 0.06,
                "a42": -0.08,
                "a43": 0.11,
            },
            "process_std": PROCESS_STD.tolist(),
            "observation_std": OBSERVATION_STD.tolist(),
        },
        "prior": {
            "family": "independent_gaussian_on_raw_coordinates",
            "center": "raw_truth_template",
            "scale_by_block": {
                "diagonal_raw": 0.50,
                "lower_raw": 0.60,
                "log_process_std": 0.35,
                "log_observation_std": 0.35,
            },
            "log_jacobian_convention": "included_in_prior",
        },
        "filter_manifest": {
            "likelihood": "exact_linear_gaussian_kalman",
            "implementation_backend": "tensorflow_tensorflow_probability",
        },
        "reconstruction_provenance": {
            "date": "2026-07-13",
            "reason": "referenced_machine_readable_contract_absent_from_both_merge_parents",
            "source_authority": (
                "docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-"
                "phase1-model-contract-result-2026-07-08.md"
            ),
        },
        "nonclaims": [
            "not global identifiability evidence",
            "not posterior correctness evidence",
            "not HMC or NeuTra readiness evidence",
            "not product or default readiness",
        ],
    }
    contract["contract_payload_hash"] = canonical_hash(contract)
    return contract


def build_data(contract: Mapping[str, Any]) -> dict[str, Any]:
    covariance = stationary_covariance()
    states, observations = simulate(covariance)
    process_covariance = np.diag(np.square(PROCESS_STD))
    observation_covariance = np.diag(np.square(OBSERVATION_STD))
    residual = covariance - TRANSITION @ covariance @ TRANSITION.T - process_covariance
    transition_eigenvalues = np.linalg.eigvals(TRANSITION)
    covariance_eigenvalues = np.linalg.eigvalsh(covariance)
    max_abs_state = float(np.max(np.abs(states)))
    max_abs_observation = float(np.max(np.abs(observations)))

    if max_abs_state != HISTORICAL_MAX_ABS_STATE:
        raise RuntimeError("reconstructed state stream does not match retained extrema")
    if max_abs_observation != HISTORICAL_MAX_ABS_OBSERVATION:
        raise RuntimeError("reconstructed observation stream does not match retained extrema")

    data: dict[str, Any] = {
        "schema": "bayesfilter.multidim_triangular_lgssm.synthetic_data.v1",
        "contract_id": CONTRACT_ID,
        "target_id": TARGET_ID,
        "source_contract": {
            "path": str(CONTRACT_PATH.relative_to(ROOT)),
            "schema": SCHEMA,
            "contract_id": CONTRACT_ID,
            "target_id": TARGET_ID,
            "contract_payload_hash": contract["contract_payload_hash"],
        },
        "seed": SEED,
        "horizon": HORIZON,
        "generator": {
            "backend": "numpy_reference_fixture",
            "rng": "numpy.random.default_rng",
            "random_stream": (
                "stationary_initial_draw_then_each_time_observation_noise_"
                "then_next_transition_noise"
            ),
            "dtype": "float64",
        },
        "parameter_names": list(PARAMETER_NAMES),
        "raw_truth": raw_truth().tolist(),
        "constrained_truth": {
            "transition_matrix": TRANSITION.tolist(),
            "process_std": PROCESS_STD.tolist(),
            "process_covariance": process_covariance.tolist(),
            "observation_matrix": np.eye(4, dtype=np.float64).tolist(),
            "observation_std": OBSERVATION_STD.tolist(),
            "observation_covariance": observation_covariance.tolist(),
            "stationary_initial_mean": np.zeros(4, dtype=np.float64).tolist(),
            "stationary_initial_covariance": covariance.tolist(),
        },
        "states": states.tolist(),
        "observations": observations.tolist(),
        "diagnostics": {
            "state_shape": list(states.shape),
            "observation_shape": list(observations.shape),
            "transition_eigenvalues": transition_eigenvalues.tolist(),
            "transition_spectral_radius": float(np.max(np.abs(transition_eigenvalues))),
            "stationarity_margin": float(1.0 - np.max(np.abs(transition_eigenvalues))),
            "stationary_covariance_eigenvalues": covariance_eigenvalues.tolist(),
            "stationary_covariance_min_eigenvalue": float(np.min(covariance_eigenvalues)),
            "lyapunov_max_abs_residual": float(np.max(np.abs(residual))),
            "state_mean": np.mean(states, axis=0).tolist(),
            "state_std_empirical": np.std(states, axis=0).tolist(),
            "observation_mean": np.mean(observations, axis=0).tolist(),
            "observation_std_empirical": np.std(observations, axis=0).tolist(),
            "max_abs_state": max_abs_state,
            "max_abs_observation": max_abs_observation,
            "process_to_observation_std_ratio_truth": (
                PROCESS_STD / OBSERVATION_STD
            ).tolist(),
        },
        "reconstruction_validation": {
            "exact_historical_max_abs_state_match": True,
            "exact_historical_max_abs_observation_match": True,
            "historical_hashes": HISTORICAL_PHASE2_HASHES,
            "historical_hash_status": (
                "not_reasserted_original_bytes_were_absent_from_git_and_worktrees"
            ),
            "historical_empirical_std_status": (
                "not_reasserted_result_note_values_do_not_match_standard_array_std"
            ),
        },
        "metric_roles": {
            "stationarity_margin": "engineering_pass_criterion",
            "lyapunov_max_abs_residual": "engineering_veto_diagnostic",
            "stationary_covariance_min_eigenvalue": "engineering_veto_diagnostic",
            "state_std_empirical": "explanatory_only",
            "observation_std_empirical": "explanatory_only",
            "max_abs_state": "reconstruction_fingerprint",
            "max_abs_observation": "reconstruction_fingerprint",
        },
        "nonclaims": [
            "reconstruction does not revalidate the unavailable historical hashes",
            "not posterior correctness or recovery evidence",
            "not HMC or NeuTra readiness evidence",
            "not scientific or default promotion evidence",
        ],
    }
    data["data_payload_hash"] = canonical_hash(data)
    return data


def build_manifest(
    contract: Mapping[str, Any],
    data: Mapping[str, Any],
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema": "bayesfilter.multidim_triangular_lgssm.synthetic_data_manifest.v1",
        "contract_path": str(CONTRACT_PATH.relative_to(ROOT)),
        "contract_payload_hash": contract["contract_payload_hash"],
        "data_path": str(DATA_PATH.relative_to(ROOT)),
        "data_payload_hash": data["data_payload_hash"],
        "data_file_sha256": file_hash(DATA_PATH),
        "seed": SEED,
        "horizon": HORIZON,
        "state_shape": [HORIZON, 4],
        "observation_shape": [HORIZON, 4],
        "reconstruction_status": "closeout_reconstruction_from_retained_contract_and_seed",
        "historical_hashes": HISTORICAL_PHASE2_HASHES,
        "historical_hash_status": "not_reasserted",
        "nonclaims": [
            "fresh reconstruction hashes are not the missing historical hashes",
            "not inference, training, HMC, or promotion evidence",
        ],
    }
    manifest["manifest_payload_hash"] = canonical_hash(manifest)
    return manifest


def main() -> int:
    contract = build_contract()
    write_json(CONTRACT_PATH, contract)
    data = build_data(contract)
    write_json(DATA_PATH, data)
    manifest = build_manifest(contract, data)
    write_json(MANIFEST_PATH, manifest)
    print(
        json.dumps(
            {
                "contract": contract["contract_payload_hash"],
                "data_payload": data["data_payload_hash"],
                "data_file": manifest["data_file_sha256"],
                "manifest": manifest["manifest_payload_hash"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
