"""Direct target registry for the BayesFilter NeuTra end-to-end campaign.

This module contains target facts and construction only.  Sampling, tuning,
training, diagnostics, and artifact policy live in
``bayesfilter.inference.neutra_end_to_end``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RecipeSpec:
    recipe_id: str
    hidden_layers: tuple[int, ...]
    learning_rate: float
    final_learning_rate_fraction: float = 0.1
    stage_count: int = 3

    def payload(self) -> Mapping[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "stage_count": self.stage_count,
            "hidden_layers": self.hidden_layers,
            "learning_rate": self.learning_rate,
            "final_learning_rate_fraction": self.final_learning_rate_fraction,
            "activation": "elu",
            "s_max": 1.0,
            "init_scale": 0.02,
            "batch_size": 128,
            "clip_norm": 10.0,
            "optimizer": "manual_adam_linear_decay",
        }


@dataclass(frozen=True)
class CellSpec:
    cell_id: str
    parameter_dim: int
    parameter_names: tuple[str, ...]
    target_signature: str
    adapter_factory: Callable[[], Any]
    geometry_factory: Callable[[Any], tuple[Any, Any, Mapping[str, Any]]]
    physical_transform: Callable[[Any, Any], Any]
    truth_factory: Callable[[Any], Any]
    recipes: tuple[RecipeSpec, ...]
    initial_seed: tuple[int, int]
    target_description: str
    require_affine_nonworse: bool = False
    preferred_recipe_id: str | None = None
    selection_mcse_multiplier: float = 2.0
    initial_step_size: float = 0.1
    leapfrog_grid: tuple[int, ...] = (6, 10)
    screen_seeds: tuple[tuple[int, int], ...] = ()
    plan_path: str = (
        "docs/plans/bayesfilter-neutra-all-executable-models-end-to-end-python-"
        "plan-2026-07-18.md"
    )
    common_tuning_status_keys: tuple[str, ...] = (
        "status_code",
        "valid_pre_regularized_score",
        "floor_count_value",
        "min_innovation_eigenvalue",
        "innovation_condition_estimate",
    )
    common_tuning_initial_epsilon_by_l: Mapping[int, float] | None = None

    def payload(self) -> Mapping[str, Any]:
        return {
            "cell_id": self.cell_id,
            "parameter_dim": self.parameter_dim,
            "parameter_names": self.parameter_names,
            "target_signature": self.target_signature,
            "target_description": self.target_description,
            "recipes": tuple(recipe.payload() for recipe in self.recipes),
            "initial_seed": self.initial_seed,
            "require_affine_nonworse": self.require_affine_nonworse,
            "preferred_recipe_id": self.preferred_recipe_id,
            "selection_mcse_multiplier": self.selection_mcse_multiplier,
            "initial_step_size": self.initial_step_size,
            "leapfrog_grid": self.leapfrog_grid,
            "screen_seeds": self.screen_seeds or (self.initial_seed,),
            "plan_path": self.plan_path,
            "common_tuning_status_keys": self.common_tuning_status_keys,
            "common_tuning_initial_epsilon_by_l": self.common_tuning_initial_epsilon_by_l,
            "geometry_factory": getattr(self.geometry_factory, "__qualname__", str(self.geometry_factory)),
            "adapter_factory": getattr(self.adapter_factory, "__qualname__", str(self.adapter_factory)),
        }


@dataclass(frozen=True)
class BlockedCellSpec:
    cell_id: str
    state: str
    reason: str
    reentry_rung: str

    def payload(self) -> Mapping[str, Any]:
        return {
            "cell_id": self.cell_id,
            "state": self.state,
            "reason": self.reason,
            "reentry_rung": self.reentry_rung,
        }


def _read_json(relative_path: str) -> Mapping[str, Any]:
    value = json.loads((ROOT / relative_path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON mapping: {relative_path}")
    return value


def _sha256(relative_path: str) -> str:
    return hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()


def _geometry_from_json(
    relative_path: str,
    *,
    expected_sha256: str,
    nested_key: str | None = None,
):
    def factory(tf: Any) -> tuple[Any, Any, Mapping[str, Any]]:
        observed_sha256 = _sha256(relative_path)
        if observed_sha256 != expected_sha256:
            raise ValueError(f"geometry SHA-256 drift: {relative_path}")
        payload = _read_json(relative_path)
        value = payload if nested_key is None else payload[nested_key]
        center_key = "center"
        factor_key = "factor" if "factor" in value else "cholesky_factor"
        center = tf.constant(value[center_key], tf.float64)
        factor = tf.constant(value[factor_key], tf.float64)
        if center.shape.rank != 1 or factor.shape.rank != 2:
            raise ValueError(f"invalid geometry shape: {relative_path}")
        return center, factor, {
            "path": relative_path,
            "file_sha256": observed_sha256,
            "geometry_key": nested_key,
            "center_role": value.get("center_role", "historical_affine_warm_start"),
        }

    return factory


def _identity_geometry(tf: Any, dimension: int) -> tuple[Any, Any, Mapping[str, Any]]:
    return (
        tf.zeros((dimension,), tf.float64),
        tf.eye(dimension, dtype=tf.float64),
        {
            "path": None,
            "file_sha256": None,
            "center_role": "source_prior_origin",
            "factor_role": "identity",
        },
    )


def _lgssm_bundle() -> Any:
    from bayesfilter.testing.deterministic_lgssm_exact_target_tf import (
        load_deterministic_lgssm_exact_target,
    )

    bundle = load_deterministic_lgssm_exact_target(
        expected_target_signature=LGSSM_SIGNATURE
    )
    if bundle.adapter.adapter_signature() != LGSSM_ADAPTER_SIGNATURE:
        raise ValueError("LGSSM adapter signature drift")
    return bundle


def _lgssm_adapter() -> Any:
    return _lgssm_bundle().adapter


def _pp_ukf_adapter() -> Any:
    from bayesfilter.testing.predator_prey_ukf_neutra_target_tf import (
        generate_frozen_predator_prey_dataset_tf,
        make_predator_prey_ukf_neutra_adapter,
    )

    _states, observations = generate_frozen_predator_prey_dataset_tf()
    return make_predator_prey_ukf_neutra_adapter(observations=observations)


def _pp_sgqf_adapter() -> Any:
    from bayesfilter.testing.predator_prey_sgqf_neutra_target_tf import (
        generate_frozen_predator_prey_dataset_tf,
        make_predator_prey_sgqf_neutra_adapter,
    )

    _states, observations = generate_frozen_predator_prey_dataset_tf()
    return make_predator_prey_sgqf_neutra_adapter(
        sparse_level=2, observations=observations
    )


def _sir_sgqf_adapter() -> Any:
    from bayesfilter.testing.sir_filter_neutra_target_design_tf import (
        generate_frozen_sir_dataset_tf,
        make_sir_sgqf_neutra_adapter,
    )

    _states, observations, _all_states = generate_frozen_sir_dataset_tf()
    return make_sir_sgqf_neutra_adapter(observations=observations)


def _structural_ukf_adapter() -> Any:
    from bayesfilter.testing.structural_ukf_neutra_target_design_tf import (
        generate_frozen_structural_dataset_tf,
        make_structural_ukf_neutra_adapter,
    )

    _states, observations = generate_frozen_structural_dataset_tf()
    return make_structural_ukf_neutra_adapter(observations=observations)


def _sv_zc_adapter() -> Any:
    from bayesfilter.testing.zhao_cui_actual_sv_neutra_target_tf import (
        make_actual_sv_zc_neutra_adapter,
    )

    return make_actual_sv_zc_neutra_adapter()


def _lgssm_physical(tf: Any, values: Any) -> Any:
    return tf.concat(
        (
            0.85 * tf.math.tanh(values[..., :4]),
            0.35 * tf.math.tanh(values[..., 4:10]),
            tf.math.exp(values[..., 10:18]),
        ),
        axis=-1,
    )


def _identity(tf: Any, values: Any) -> Any:
    return tf.convert_to_tensor(values, tf.float64)


def _pp_physical(tf: Any, values: Any) -> Any:
    from bayesfilter.testing.predator_prey_ukf_neutra_target_tf import (
        source_chart_physical_parameters,
    )

    shape = tf.shape(values)
    flat = tf.reshape(tf.convert_to_tensor(values, tf.float64), (-1, 6))
    physical, _derivative = source_chart_physical_parameters(flat)
    return tf.reshape(physical, shape)


def _sir_physical(tf: Any, values: Any) -> Any:
    return tf.constant((0.1, 18.0, 10.0), tf.float64) * tf.math.exp(
        tf.convert_to_tensor(values, tf.float64)
    )


def _structural_physical(tf: Any, values: Any) -> Any:
    from bayesfilter.testing.structural_ukf_neutra_target_design_tf import (
        structural_source_chart,
    )

    shape = tf.shape(values)
    flat = tf.reshape(tf.convert_to_tensor(values, tf.float64), (-1, 5))
    physical, _derivative = structural_source_chart(flat)
    return tf.reshape(physical, shape)


def _sv_zc_physical(tf: Any, values: Any) -> Any:
    from bayesfilter.highdim.zhao_cui_actual_sv_batched_tt_tf import (
        source_chart_physical_parameters,
    )

    shape = tf.shape(values)
    flat = tf.reshape(tf.convert_to_tensor(values, tf.float64), (-1, 2))
    gamma, beta = source_chart_physical_parameters(flat)
    physical = tf.stack((gamma, beta), axis=-1)
    return tf.reshape(physical, shape)


def _constant_truth(tf: Any, values: tuple[float, ...]) -> Any:
    return tf.constant(values, tf.float64)


LGSSM_SIGNATURE = "bd40a828bc4916e5e09a8e6135f315ebc45c06844aed38a506d6296c2642557d"
LGSSM_ADAPTER_SIGNATURE = "1ddb0a1106488871643e79ce0a575db6871e24963332119cc8a20a436d84b872"
PP_UKF_SIGNATURE = "d3ed745b4f755582bfce46b24992e9d626e10c1409c46b0518ca8cfc673fc2f5"
PP_SGQF_SIGNATURE = "373326607b8cb06f274f03e0a523a47b24b83e35c8b37c8d264b500a6234fbac"
SIR_SGQF_SIGNATURE = "43968c975409021dcabe931081f0d1efaaae431b5b9245929a5786fe566e545d"
STR_UKF_SIGNATURE = "79cc70634f828b185fe0a01ed88a4dc15a52a4494432f87401340aa3c8199b06"
SVX_ZC_SIGNATURE = "deccdda78028706d0987322d30b9798f0f4d8b518c6773451338e83bf14d1cab"


def _truth_from_module(tf: Any, module: str, name: str) -> Any:
    target = __import__(module, fromlist=[name])
    return tf.convert_to_tensor(getattr(target, name), tf.float64)


EXECUTABLE_CELLS: tuple[CellSpec, ...] = (
    CellSpec(
        cell_id="LGSSM-EXACT",
        parameter_dim=18,
        parameter_names=(
            "a11", "a22", "a33", "a44", "a21", "a31", "a32", "a41",
            "a42", "a43", "q1", "q2", "q3", "q4", "r1", "r2", "r3", "r4",
        ),
        target_signature=LGSSM_SIGNATURE,
        adapter_factory=_lgssm_adapter,
        geometry_factory=_geometry_from_json(
            "docs/benchmarks/artifacts/multidim_lgssm_full_estimation_rerun_2026_07_13/mass.json",
            expected_sha256="54549c9156821536bc4780f0406a7716b0d3fa39a5b5900fa2893cbef2968a95",
        ),
        physical_transform=_lgssm_physical,
        truth_factory=lambda tf: _constant_truth(tf, (
            0.62, 0.48, 0.30, 0.16, 0.18, -0.10, 0.14, 0.06, -0.08,
            0.11, 0.30, 0.26, 0.22, 0.18, 0.12, 0.11, 0.10, 0.09,
        )),
        recipes=(
            RecipeSpec("source_anchor", (18, 18), 5.0e-3, 1.0),
            RecipeSpec("lower_lr", (18, 18), 1.0e-3, 1.0),
            RecipeSpec("shallow", (18, 18), 5.0e-3, 1.0, stage_count=2),
            RecipeSpec("wide", (36, 36), 5.0e-3, 1.0),
        ),
        initial_seed=(20260718, 1801),
        target_description="exact 18D deterministic triangular LGSSM posterior",
        preferred_recipe_id="source_anchor",
        selection_mcse_multiplier=1.0,
        initial_step_size=0.1,
        common_tuning_status_keys=(
            "status_code",
            "valid_pre_regularized_score",
            "floor_count_value",
            "min_innovation_eigenvalue",
            "innovation_condition_estimate",
        ),
        leapfrog_grid=(5, 10, 15),
    ),
    CellSpec(
        cell_id="PP-UKF", parameter_dim=6,
        parameter_names=("r", "K", "a", "s", "u", "v"),
        target_signature=PP_UKF_SIGNATURE, adapter_factory=_pp_ukf_adapter,
        geometry_factory=_geometry_from_json(
            "docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p4/PP-UKF/plain-hmc-affine/attempt-01-20260715T152500Z/affine_mass.json",
            expected_sha256="c0e3195dddc050654d7c2c6ab25e5e71599ecab42bc7ffb6b9f070ec4126c304",
        ),
        physical_transform=_pp_physical,
        truth_factory=lambda tf: _constant_truth(tf, (0.6, 114.0, 25.0, 0.3, 0.5, 0.5)),
        recipes=(
            RecipeSpec("source_width_lr1e3", (18, 18), 1.0e-3, 1.0),
            RecipeSpec("source_width", (18, 18), 5.0e-3, 1.0),
            RecipeSpec("wide_lr1e3", (36, 36), 1.0e-3, 1.0),
            RecipeSpec("wide", (36, 36), 5.0e-3, 1.0),
        ),
        initial_seed=(20260718, 1811),
        target_description="predator-prey principal-square-root UKF posterior",
        initial_step_size=0.2,
        common_tuning_status_keys=(
            "status_code",
            "valid_pre_regularized_score",
            "floor_count_value",
            "min_innovation_eigenvalue",
            "innovation_condition_estimate",
        ),
        common_tuning_initial_epsilon_by_l={
            3: 0.8724049589170738,
            5: 0.8426345584765329,
            9: 0.7489709357241571,
            13: 0.69086551957137,
            18: 0.6813265222611998,
            25: 0.6800917535732008,
        },
    ),
    CellSpec(
        cell_id="PP-SGQF", parameter_dim=6,
        parameter_names=("r", "K", "a", "s", "u", "v"),
        target_signature=PP_SGQF_SIGNATURE, adapter_factory=_pp_sgqf_adapter,
        geometry_factory=_geometry_from_json(
            "docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p4/PP-SGQF/laplace-geometry/attempt-01-20260715T165000Z/result.json",
            expected_sha256="b54343fdee59c3f86ffb8f8ac69ba0ea31b7a0c780a4f2eb290374df060cabc3",
            nested_key="final_geometry",
        ),
        physical_transform=_pp_physical,
        truth_factory=lambda tf: _constant_truth(tf, (0.6, 114.0, 25.0, 0.3, 0.5, 0.5)),
        recipes=(
            RecipeSpec("source_width_lr1e3", (18, 18), 1.0e-3, 1.0),
            RecipeSpec("source_width", (18, 18), 5.0e-3, 1.0),
            RecipeSpec("wide_lr1e3", (36, 36), 1.0e-3, 1.0),
            RecipeSpec("wide", (36, 36), 5.0e-3, 1.0),
        ),
        initial_seed=(20260718, 1821),
        target_description="predator-prey fixed level-2 SGQF posterior",
        initial_step_size=0.2,
        common_tuning_status_keys=(
            "status_code",
            "valid_pre_regularized_score",
            "floor_count_value",
            "min_innovation_eigenvalue",
            "innovation_condition_estimate",
        ),
    ),
    CellSpec(
        cell_id="SIR-SGQF", parameter_dim=3,
        parameter_names=("kappa", "nu", "observation_sd_scale"),
        target_signature=SIR_SGQF_SIGNATURE, adapter_factory=_sir_sgqf_adapter,
        geometry_factory=_geometry_from_json(
            "docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p6/SIR-SGQF/laplace-geometry/attempt-02/result.json",
            expected_sha256="cbe82fe175991c549ed1c7c309a03a719be372e040ea755e358589deeb2c6d67",
            nested_key="geometry",
        ),
        physical_transform=_sir_physical,
        truth_factory=lambda tf: _constant_truth(tf, (0.1, 18.0, 10.0)),
        recipes=(
            RecipeSpec("dim3", (9, 9), 1.0e-3, 1.0),
            RecipeSpec("dim3_lr5e3", (9, 9), 5.0e-3, 1.0),
            RecipeSpec("wide", (18, 18), 1.0e-3, 1.0),
            RecipeSpec("wide_lr5e3", (18, 18), 5.0e-3, 1.0),
        ),
        initial_seed=(20260718, 1831),
        target_description="parameterized Austria SIR fixed level-2 SGQF posterior",
        require_affine_nonworse=True,
        initial_step_size=0.2,
        common_tuning_status_keys=(
            "status_code",
            "valid_pre_regularized_score",
            "floor_count_value",
            "min_innovation_eigenvalue",
            "innovation_condition_estimate",
        ),
    ),
    CellSpec(
        cell_id="STR-UKF", parameter_dim=5,
        parameter_names=("rho", "sigma", "phi", "gamma", "R"),
        target_signature=STR_UKF_SIGNATURE, adapter_factory=_structural_ukf_adapter,
        geometry_factory=lambda tf: _identity_geometry(tf, 5),
        physical_transform=_structural_physical,
        truth_factory=lambda tf: _truth_from_module(
            tf, "bayesfilter.testing.structural_ukf_neutra_target_design_tf", "STRUCTURAL_TRUTH_PHYSICAL"
        ),
        recipes=(
            RecipeSpec("dim3_lr1e3", (15, 15), 1.0e-3),
            RecipeSpec("dim3", (15, 15), 5.0e-3),
            RecipeSpec("dim6_lr1e3", (30, 30), 1.0e-3),
            RecipeSpec("dim6", (30, 30), 5.0e-3),
        ),
        initial_seed=(20260718, 1841),
        target_description="structural five-parameter principal-square-root UKF posterior",
        initial_step_size=0.025,
        leapfrog_grid=(8, 12),
    ),
    CellSpec(
        cell_id="SVX-ZC",
        parameter_dim=2,
        parameter_names=("gamma_source_probit", "beta_source_probit"),
        target_signature=SVX_ZC_SIGNATURE,
        adapter_factory=_sv_zc_adapter,
        geometry_factory=lambda tf: _identity_geometry(tf, 2),
        physical_transform=_sv_zc_physical,
        truth_factory=lambda tf: _constant_truth(tf, (0.6, 0.4)),
        recipes=(
            RecipeSpec("svx_zc_narrow_lr1e3", (8, 8), 1.0e-3),
            RecipeSpec("svx_zc_narrow_lr5e3", (8, 8), 5.0e-3),
            RecipeSpec("svx_zc_wide_lr1e3", (16, 16), 1.0e-3),
            RecipeSpec("svx_zc_wide_lr5e3", (16, 16), 5.0e-3),
        ),
        initial_seed=(20260802, 1861),
        target_description=(
            "frozen T10 degree-10 rank-2 order-25 adjacent-state squared-TT "
            "actual-SV posterior"
        ),
        preferred_recipe_id="svx_zc_narrow_lr1e3",
        initial_step_size=0.1,
        leapfrog_grid=(3, 5, 9, 13, 18, 25),
        common_tuning_status_keys=(
            "status_code",
            "valid_pre_regularized_score",
            "floor_count_value",
            "min_innovation_eigenvalue",
            "innovation_condition_estimate",
        ),
    ),
)


BLOCKED_CELLS: tuple[BlockedCellSpec, ...] = (
    BlockedCellSpec("SVX-SGQF", "TARGET_BLOCKED_FILTER_ADMISSION", "no frozen SGQF level passed filter admission", "filter admission"),
    BlockedCellSpec("KSC-UKF", "TARGET_BLOCKED_FILTER_ADMISSION", "dense-reference value/score admission failed", "filter admission"),
    BlockedCellSpec(
        "PP-ZC",
        "TARGET_BLOCKED_NEUTRA_TARGET_CONTRACT",
        "sealed fixed-branch implementation passes, but no batch-native posterior adapter or frozen HMC chart/Jacobian is registered",
        "batch-native target adapter and chart admission",
    ),
    BlockedCellSpec("STR-ZC", "TARGET_BLOCKED_EXTENSION_ROUTE_NOT_DESIGNED", "extension target is absent", "extension-target design"),
    BlockedCellSpec("SIR-ZC", "TARGET_BLOCKED_MISSING_OBSERVED_DATA_SCORE_ROUTE", "observed-data parameter-score closure is absent", "observed-data score route"),
)


OWNER_EXCLUDED_CELLS: tuple[BlockedCellSpec, ...] = (
    BlockedCellSpec(
        "SIR-UKF",
        "OWNER_EXCLUDED_METHOD_NOT_APPLICABLE",
        "owner determination: UKF does not work for SIR; remove it from testing",
        "none; reentry requires a new owner direction",
    ),
)


def registry_payload() -> Mapping[str, Any]:
    return {
        "schema": "bayesfilter.neutra.all_models.registry.v1",
        "executable": tuple(spec.payload() for spec in EXECUTABLE_CELLS),
        "blocked": tuple(spec.payload() for spec in BLOCKED_CELLS),
        "owner_excluded": tuple(spec.payload() for spec in OWNER_EXCLUDED_CELLS),
        "nonclaims": (
            "registry construction is not training or HMC evidence",
            "blocked cells are not failed NeuTra candidates",
            "owner-excluded cells are not active testing or repair candidates",
        ),
    }


def validate_registry() -> Mapping[str, Any]:
    executable_ids = tuple(spec.cell_id for spec in EXECUTABLE_CELLS)
    blocked_ids = tuple(spec.cell_id for spec in BLOCKED_CELLS)
    excluded_ids = tuple(spec.cell_id for spec in OWNER_EXCLUDED_CELLS)
    all_ids = executable_ids + blocked_ids + excluded_ids
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("registry cell IDs must be unique")
    for spec in EXECUTABLE_CELLS:
        if len(spec.parameter_names) != spec.parameter_dim:
            raise ValueError(f"parameter name dimension mismatch: {spec.cell_id}")
        if len(spec.target_signature) != 64:
            raise ValueError(f"target signature malformed: {spec.cell_id}")
        if not spec.recipes:
            raise ValueError(f"no training recipes: {spec.cell_id}")
    return registry_payload()
