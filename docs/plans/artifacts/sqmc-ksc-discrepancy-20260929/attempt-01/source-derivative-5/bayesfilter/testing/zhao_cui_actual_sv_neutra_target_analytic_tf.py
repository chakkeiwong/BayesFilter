"""Analytic fixed-branch SVX-ZC posterior target for serious NeuTra/HMC use.

This adapter preserves the reviewed UKF-frozen initialization identity used by
`zhao_cui_actual_sv_neutra_target_tf.py`, but replaces the HMC-facing score
backend with the transformed-SV fixed-branch analytic TT score path.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import tensorflow as tf

import bayesfilter.highdim as highdim

from bayesfilter.highdim.filtering import FixedBranchDerivativeConfig
from bayesfilter.highdim.sv_mixture_cut4 import (
    exact_transformed_sv_independent_panel_zhaocui_tt_score,
)
from bayesfilter.highdim.zhao_cui_actual_sv_batched_tt_tf import (
    source_chart_physical_parameters,
    source_two_probit_jacobian_value_score,
    source_uniform_prior_value_score,
)
from bayesfilter.inference.posterior_adapter import ValueScoreCapability
from bayesfilter.ssm import stable_ssm_target_signature
from bayesfilter.testing.zhao_cui_actual_sv_neutra_target_tf import (
    NONCLAIMS,
    PARAMETER_NAMES,
    TARGET_SCOPE,
    _make_actual_sv_zc_neutra_adapter_on_cpu,
)


SCORE_BACKEND_ID = "analytic_fixed_branch_scalar_score_aggregation_no_autodiff_v1"
ANALYTIC_TARGET_SCOPE = TARGET_SCOPE + "-analytic"
ANALYTIC_NONCLAIMS = NONCLAIMS + (
    "serious HMC route uses transformed-SV fixed-branch analytic score, not autodiff through TT recursion",
)


def _semantic_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class ActualSVZCAnalyticNeuTraAdapter:
    """Batch-native posterior adapter with analytic transformed-SV TT score."""

    dtype = tf.float64
    parameter_dim = 2
    parameter_names = PARAMETER_NAMES
    supports_retained_flat_batch = True
    supports_retained_value_score_status = True
    score_backend_id = SCORE_BACKEND_ID
    runtime_autodiff_for_hmc = False

    def __init__(
        self,
        *,
        contract,
        raw_observations: tf.Tensor,
        initial_core_hash: str,
        adjacent_core_hash: str,
        config,
        derivative_config,
    ) -> None:
        self.contract = contract
        self.target_scope = ANALYTIC_TARGET_SCOPE
        self.raw_observations = tf.convert_to_tensor(raw_observations, tf.float64)
        self.initial_core_hash = str(initial_core_hash)
        self.adjacent_core_hash = str(adjacent_core_hash)
        self.config = config
        self.derivative_config = derivative_config
        payload = {
            "schema": "bayesfilter.testing.actual_sv_zc_analytic_neutra_adapter.v1",
            "target_signature": stable_ssm_target_signature(contract),
            "target_scope": self.target_scope,
            "parameter_names": self.parameter_names,
            "initial_core_hash": self.initial_core_hash,
            "adjacent_core_hash": self.adjacent_core_hash,
            "score_backend_id": SCORE_BACKEND_ID,
            "runtime_autodiff_for_hmc": False,
        }
        self._adapter_signature = _semantic_hash(payload)

    def adapter_signature(self) -> str:
        return self._adapter_signature

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
            runtime_backend="tensorflow_fixed_branch_transformed_sv_analytic_tt",
            evidence_path="bayesfilter/testing/zhao_cui_actual_sv_neutra_target_analytic_tf.py",
            target_scope=self.target_scope,
            nonclaims=ANALYTIC_NONCLAIMS,
        )

    def _analytic_likelihood_value_score(
        self, theta: Any
    ) -> tuple[tf.Tensor, tf.Tensor]:
        values = _rank2_theta(theta)
        gamma, beta = source_chart_physical_parameters(values)
        sigma = tf.ones_like(gamma, tf.float64)
        value_rows = []
        score_rows = []
        for row_gamma, row_beta in zip(
            tf.unstack(gamma, axis=0), tf.unstack(beta, axis=0)
        ):
            row_result = exact_transformed_sv_independent_panel_zhaocui_tt_score(
                self.raw_observations,
                gamma=tf.reshape(row_gamma, [1]),
                beta=tf.reshape(row_beta, [1]),
                sigma=tf.ones([1], tf.float64),
                config=self.config,
                derivative_config=self.derivative_config,
                fixture_id="svx_zc_neutra_analytic_fixed_branch_tt.v1",
                branch_seed_prefix="svx-zc-neutra-analytic-fixed-branch-tt",
            )
            value_rows.append(tf.reshape(row_result.log_likelihood, [1]))
            score_rows.append(tf.reshape(row_result.score, [1, 2]))
        return tf.concat(value_rows, axis=0), tf.concat(score_rows, axis=0)

    def neutra_batch_log_prob_and_grad_status(
        self, theta: Any
    ) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
        likelihood_value, likelihood_score = self._analytic_likelihood_value_score(theta)
        prior_value, prior_score = source_uniform_prior_value_score(theta)
        jacobian_value, jacobian_score = source_two_probit_jacobian_value_score(theta)
        finite = tf.logical_and(
            tf.math.is_finite(likelihood_value),
            tf.reduce_all(tf.math.is_finite(likelihood_score), axis=1),
        )
        return (
            likelihood_value + prior_value + jacobian_value,
            likelihood_score + prior_score + jacobian_score,
            {
                "status_code": tf.where(
                    finite,
                    tf.zeros(tf.shape(likelihood_value), tf.int32),
                    tf.ones(tf.shape(likelihood_value), tf.int32),
                ),
                "valid_pre_regularized_score": finite,
                "floor_count_value": tf.zeros(tf.shape(likelihood_value), tf.int32),
                "min_innovation_eigenvalue": tf.ones(
                    tf.shape(likelihood_value), tf.float64
                ),
                "innovation_condition_estimate": tf.ones(
                    tf.shape(likelihood_value), tf.float64
                ),
            },
        )

    def log_prob(self, theta: Any) -> tf.Tensor:
        value, _score, _status = self.neutra_batch_log_prob_and_grad_status(theta)
        return value

    def log_prob_and_grad(self, theta: Any) -> tuple[tf.Tensor, tf.Tensor]:
        value, score, _status = self.neutra_batch_log_prob_and_grad_status(theta)
        return value, score

    def target_status_telemetry(self, theta: Any) -> Mapping[str, tf.Tensor]:
        _value, _score, status = self.neutra_batch_log_prob_and_grad_status(theta)
        return status


def _rank2_theta(theta: Any) -> tf.Tensor:
    values = tf.convert_to_tensor(theta, tf.float64)
    if values.shape.rank == 1:
        values = values[tf.newaxis, :]
    if values.shape.rank != 2 or values.shape[-1] != 2:
        raise ValueError("SVX-ZC analytic target requires theta shape [batch, 2]")
    return values


def make_actual_sv_zc_analytic_neutra_adapter() -> ActualSVZCAnalyticNeuTraAdapter:
    """Build the serious/admissible analytic SVX-ZC adapter on CPU."""

    base = _make_actual_sv_zc_neutra_adapter_on_cpu()
    import docs.benchmarks.run_contract_e_tp_phase6_zhao_cui_comparator as comparator

    model, center, _transformed = comparator._row_inputs("actual_sv", 10)
    raw = tf.convert_to_tensor(
        comparator._sv_dataset(81101)["observations"], tf.float64
    )[:10]
    initial, adjacent, initializer = comparator._ukf_initial_cores(
        model=model,
        theta=center,
        raw_observations=raw,
        degree=10,
        order=25,
        rank=2,
        coordinate_half_width=8.0,
    )
    product_basis = highdim.ProductBasis(
        [highdim.LegendreBasis1D(highdim.BoundedInterval(-1.0, 1.0), 48)],
        highdim.MeasureConvention(
            density_measure=highdim.DensityMeasure.REFERENCE_MEASURE,
            mass_measure=highdim.MassMeasure.REFERENCE_MEASURE,
            reference_weight_name="omega",
        ),
    )
    config = highdim.FixedBranchFilterConfig(
        fit_config=highdim.FixedTTFitConfig(
            ranks=(1, 1),
            ridge=1e-12,
            max_sweeps=2,
            sweep_order=(0,),
            row_budget=512,
            column_budget=128,
            dense_matrix_byte_budget=200_000,
            normal_matrix_byte_budget=100_000,
            condition_number_warning=1e10,
            condition_number_veto=1e14,
            holdout_tolerance=5e-4,
        ),
        density_tau=0.0,
        normalizer_floor=1e-12,
        denominator_floor=1e-12,
        retained_storage_byte_budget=10_000_000,
        coordinate_maps=(
            highdim.AffineCoordinateMap(
                offset=tf.constant([0.0], dtype=tf.float64),
                matrix=tf.constant([[8.0]], dtype=tf.float64),
            ),
        ),
        measure_convention=highdim.MeasureConvention(
            density_measure=highdim.DensityMeasure.REFERENCE_MEASURE,
            mass_measure=highdim.MassMeasure.REFERENCE_MEASURE,
            reference_weight_name="omega",
        ),
        deterministic_seed="svx-zc-neutra-analytic-target-center-frozen-v1",
        product_basis=product_basis,
        initial_cores=(
            highdim.TTCore(
                tf.ones([1, product_basis.bases[0].basis_dim, 1], dtype=tf.float64)
            ),
        ),
        fit_quadrature_order=141,
    )
    return ActualSVZCAnalyticNeuTraAdapter(
        contract=base.contract,
        raw_observations=raw,
        initial_core_hash=base.initial_core_hash,
        adjacent_core_hash=base.adjacent_core_hash,
        config=config,
        derivative_config=FixedBranchDerivativeConfig(
            parameter_indices=(0, 1),
            finite_difference_h=(3e-3, 1e-3),
            solve_condition_number_veto=1e16,
        ),
    )


__all__ = [
    "ANALYTIC_TARGET_SCOPE",
    "ActualSVZCAnalyticNeuTraAdapter",
    "SCORE_BACKEND_ID",
    "make_actual_sv_zc_analytic_neutra_adapter",
]
