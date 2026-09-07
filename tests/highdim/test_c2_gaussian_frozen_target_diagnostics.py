"""CPU-only mechanics checks for the C2 frozen-target diagnostic route.

GPU devices are intentionally hidden: these small tests check target identity,
capture wiring, and serialization. They are not production-target evidence.
"""

from __future__ import annotations

import inspect
import math
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf

from bayesfilter.highdim.bases import ProductBasis
from bayesfilter.highdim.fitting import FixedTTFitter
from bayesfilter.highdim.squared_tt_engine_gaussian_tf import (
    _hermite_product_basis,
)
import bayesfilter.highdim.squared_tt_engine_gaussian_xla_tf as engine
from bayesfilter.highdim.squared_tt_engine_gaussian_xla_tf import (
    GaussianXLARetainedProposalSnapshot,
    evaluate_gaussian_xla_frozen_transition,
    gaussian_xla_frozen_snapshot_fingerprint,
    gaussian_xla_frozen_snapshot_from_parts,
    gaussian_xla_frozen_snapshot_parts,
    gaussian_xla_retained_proposal_snapshot_fingerprint,
    gaussian_xla_retained_proposal_snapshot_from_parts,
    gaussian_xla_retained_proposal_snapshot_parts,
    run_value_filter_branch_axis_gaussian_xla,
    run_value_filter_branch_axis_gaussian_xla_diagnostic,
    run_value_filter_branch_axis_gaussian_xla_retained_proposal_diagnostic,
)
from bayesfilter.highdim.squared_tt_engine_v0_tf import (
    DiscreteIndicatorBasis1D,
    EngineConfig,
)
from bayesfilter.highdim.tt import TTCore

sys.path.insert(0, os.path.dirname(__file__))
import test_c2_gaussian_engine_oracle as oracle  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "docs", "benchmarks"))
import sv_fixture_c2_20260826 as sv_fixture  # noqa: E402
import diagnose_c2_n4_frozen_target_stage2_20260828 as stage2_diag  # noqa: E402

DTYPE = tf.float64


@pytest.fixture(scope="module")
def captured_run():
    adapter, observations, _steps, model = oracle._lgssm_fixture(1, 3, 308)
    config = EngineConfig(
        basis_degree=2,
        rank=2,
        row_count=64,
        sweeps=2,
        ridge=1e-10,
        tau=1e-6,
        coordinate_half_width=3.0,
        seed=7712,
        row_design="sobol",
    )
    initial_hint, predictive_hint = oracle._exact_hint_factories(model)
    plain_value, plain_diagnostics = run_value_filter_branch_axis_gaussian_xla(
        adapter,
        observations,
        config,
        predictive_moment_hint=predictive_hint,
        initial_moment_hint=initial_hint,
    )
    initial_hint, predictive_hint = oracle._exact_hint_factories(model)
    captured_value, captured_diagnostics, snapshots = (
        run_value_filter_branch_axis_gaussian_xla_diagnostic(
            adapter,
            observations,
            config,
            predictive_moment_hint=predictive_hint,
            initial_moment_hint=initial_hint,
            capture_steps=(2,),
            run_identity="cpu-stage0-lgssm-n1-t3-seed308",
        )
    )
    initial_hint, predictive_hint = oracle._exact_hint_factories(model)
    retained_value, retained_diagnostics, retained_snapshots = (
        run_value_filter_branch_axis_gaussian_xla_retained_proposal_diagnostic(
            adapter,
            observations,
            config,
            predictive_moment_hint=predictive_hint,
            initial_moment_hint=initial_hint,
            capture_steps=(1, 2),
            run_identity="cpu-retained-lgssm-n1-t3-seed308",
        )
    )
    return {
        "adapter": adapter,
        "observations": observations,
        "model": model,
        "config": config,
        "plain_value": plain_value,
        "plain_diagnostics": plain_diagnostics,
        "captured_value": captured_value,
        "captured_diagnostics": captured_diagnostics,
        "snapshot": snapshots[2],
        "retained_value": retained_value,
        "retained_diagnostics": retained_diagnostics,
        "retained_snapshots": retained_snapshots,
    }


def _mixed_basis(snapshot) -> ProductBasis:
    current = _hermite_product_basis(snapshot.state_dim, snapshot.basis_degree)
    return ProductBasis(
        list(current.bases)
        + [DiscreteIndicatorBasis1D(snapshot.branch_count)]
        + list(current.bases),
        current.convention,
    )


def _shared_target(snapshot, adapter, rows, weights, shift_offset=0.0):
    n = snapshot.state_dim
    current = _hermite_product_basis(n, snapshot.basis_degree)
    prefix_shapes = tuple(tuple(value.shape.as_list()) for value in snapshot.prefix_values)
    return engine._assemble_transition_target(
        adapter=adapter,
        current_basis=current,
        prefix_shapes=prefix_shapes,
        branch_gram_floor=snapshot.branch_gram_floor,
        defensive_nu=snapshot.defensive_nu,
        prefix_values=snapshot.prefix_values,
        gram=snapshot.suffix_gram,
        tau_abs_prev=snapshot.tau_abs_previous,
        u_rows=rows,
        u_weights=weights,
        y=snapshot.observation,
        m_c=snapshot.joint_mean[:n],
        l_cc=snapshot.joint_chol[:n, :n],
        m_p=snapshot.joint_mean[n:],
        l_pc=snapshot.joint_chol[n:, :n],
        l_pp=snapshot.joint_chol[n:, n:],
        l_old=snapshot.old_coordinate_matrix,
        m_old=snapshot.old_coordinate_offset,
        frozen_shift=snapshot.frozen_shift + tf.constant(shift_offset, DTYPE),
    )


def test_retained_capture_does_not_change_forward_result(captured_run) -> None:
    assert math.isclose(
        float(captured_run["plain_value"].numpy()),
        float(captured_run["retained_value"].numpy()),
        rel_tol=5e-12,
        abs_tol=5e-12,
    )
    plain = captured_run["plain_diagnostics"]
    captured = captured_run["retained_diagnostics"]
    assert len(plain) == len(captured)
    for left, right in zip(plain, captured):
        assert left.keys() == right.keys()
        for key in left:
            if isinstance(left[key], bool):
                assert left[key] is right[key]
            else:
                assert math.isclose(
                    float(left[key]), float(right[key]), rel_tol=5e-12, abs_tol=5e-12
                ), key


def test_retained_capture_is_the_exact_production_call_chain(captured_run) -> None:
    tf.debugging.assert_equal(
        captured_run["retained_value"], captured_run["plain_value"]
    )
    assert captured_run["retained_diagnostics"] == captured_run["plain_diagnostics"]
    assert set(captured_run["retained_snapshots"]) == {1, 2}
    for time_index, snapshot in captured_run["retained_snapshots"].items():
        assert isinstance(snapshot, GaussianXLARetainedProposalSnapshot)
        assert snapshot.time_index == time_index
        assert snapshot.basis_identity == "hermite_retained_quadratic_form_v1"
        tf.debugging.assert_near(
            snapshot.z_complete,
            snapshot.z_h + snapshot.tau_abs,
            atol=2e-14,
            rtol=2e-14,
        )


def test_retained_snapshot_round_trip_preserves_fingerprint(captured_run) -> None:
    snapshot = captured_run["retained_snapshots"][2]
    metadata, tensors = gaussian_xla_retained_proposal_snapshot_parts(snapshot)
    encoded = {
        name: tf.io.serialize_tensor(value) for name, value in tensors.items()
    }
    decoded = {
        name: tf.io.parse_tensor(value, out_type=DTYPE)
        for name, value in encoded.items()
    }
    restored = gaussian_xla_retained_proposal_snapshot_from_parts(metadata, decoded)
    assert gaussian_xla_retained_proposal_snapshot_fingerprint(restored) == (
        gaussian_xla_retained_proposal_snapshot_fingerprint(snapshot)
    )


def test_training_target_rms_and_gram_identities(captured_run) -> None:
    snapshot = captured_run["snapshot"]
    result = evaluate_gaussian_xla_frozen_transition(
        snapshot,
        captured_run["adapter"],
        snapshot.training_rows,
        snapshot.training_weights,
    )
    assert float(result["target_branch_closure_relative_max"].numpy()) < 2e-13
    assert math.isclose(
        float(result["counting_residual"].numpy()),
        snapshot.branch_count * float(snapshot.weighted_fit_rms.numpy()) ** 2,
        rel_tol=2e-11,
        abs_tol=2e-13,
    )
    assert math.isclose(
        float(result["emitted_rms"].numpy()),
        float(snapshot.weighted_fit_rms.numpy()),
        rel_tol=2e-11,
        abs_tol=2e-13,
    )
    for key in ("z_h_direct", "z_h_factored"):
        assert math.isclose(
            float(result[key].numpy()),
            float(snapshot.z_h.numpy()),
            rel_tol=2e-12,
            abs_tol=2e-13,
        )
    assert math.isclose(
        float(result["fit_log_ratio_exact"].numpy()),
        float(
            (
                result["fit_log_ratio_qmc"]
                + result["gram_vs_qmc_log_gap"]
            ).numpy()
        ),
        rel_tol=2e-13,
        abs_tol=2e-13,
    )
    qmc_log_ratio = float(result["fit_log_ratio_qmc"].numpy())
    assert float(result["reverse_triangle_bound_valid"].numpy()) == 1.0
    assert float(result["reverse_triangle_log_lower"].numpy()) <= qmc_log_ratio
    assert qmc_log_ratio <= float(result["reverse_triangle_log_upper"].numpy())

    target = _shared_target(
        snapshot,
        captured_run["adapter"],
        snapshot.training_rows,
        snapshot.training_weights,
    )
    cores = tuple(
        TTCore(tf.reshape(value, shape))
        for value, shape in zip(snapshot.fitted_core_values, snapshot.mixed_shapes)
    )
    terminal_design = FixedTTFitter()._build_design_matrix(
        _mixed_basis(snapshot), target.expanded_rows, cores, len(cores) - 1
    )
    terminal_prediction = tf.linalg.matvec(
        terminal_design, tf.reshape(cores[-1].values, [-1])
    )
    tf.debugging.assert_near(
        result["prediction"], terminal_prediction, rtol=2e-13, atol=2e-13
    )


def test_branch_closure_on_independent_central_and_tail_rows(captured_run) -> None:
    snapshot = captured_run["snapshot"]
    rows = tf.random.stateless_normal(
        [snapshot.row_count, 2 * snapshot.state_dim],
        tf.constant([991, 17], tf.int32),
        dtype=DTYPE,
    )
    rows = tf.tensor_scatter_nd_update(
        rows,
        [[0, axis] for axis in range(2 * snapshot.state_dim)],
        tf.constant([7.0, -7.0], DTYPE),
    )
    weights = tf.fill([snapshot.row_count], tf.constant(1.0 / snapshot.row_count, DTYPE))
    result = evaluate_gaussian_xla_frozen_transition(
        snapshot, captured_run["adapter"], rows, weights
    )
    assert float(result["target_all_finite"].numpy()) == 1.0
    assert float(result["target_branch_closure_relative_max"].numpy()) < 2e-13
    assert float(result["row_u_abs_max"][0].numpy()) == 7.0


def test_frozen_shift_offset_has_derived_scaling(captured_run) -> None:
    snapshot = captured_run["snapshot"]
    base = evaluate_gaussian_xla_frozen_transition(
        snapshot,
        captured_run["adapter"],
        snapshot.training_rows,
        snapshot.training_weights,
    )
    offset = 2.25
    shifted = evaluate_gaussian_xla_frozen_transition(
        snapshot,
        captured_run["adapter"],
        snapshot.training_rows,
        snapshot.training_weights,
        shift_offset=offset,
    )
    assert math.isclose(
        float(shifted["z_t"].numpy()),
        math.exp(-offset) * float(base["z_t"].numpy()),
        rel_tol=2e-12,
        abs_tol=2e-13,
    )
    tf.debugging.assert_near(
        shifted["sqrt_target"],
        math.exp(-0.5 * offset) * base["sqrt_target"],
        rtol=2e-12,
        atol=2e-13,
    )


def test_snapshot_tensor_round_trip_preserves_identity(captured_run) -> None:
    snapshot = captured_run["snapshot"]
    metadata, tensors = gaussian_xla_frozen_snapshot_parts(snapshot)
    decoded = {
        name: tf.io.parse_tensor(tf.io.serialize_tensor(value), out_type=DTYPE)
        for name, value in tensors.items()
    }
    restored = gaussian_xla_frozen_snapshot_from_parts(metadata, decoded)
    assert gaussian_xla_frozen_snapshot_fingerprint(restored) == (
        gaussian_xla_frozen_snapshot_fingerprint(snapshot)
    )
    assert restored.run_identity == snapshot.run_identity
    assert restored.training_row_seed == snapshot.training_row_seed
    assert restored.mixed_shapes == snapshot.mixed_shapes


def test_compiled_wrappers_have_explicit_signatures_and_bounded_traces(
    captured_run,
) -> None:
    config_cache = engine._STEP_CACHE[captured_run["adapter"]][captured_run["config"]]
    assert config_cache
    for compiled, _shapes in config_cache.values():
        assert compiled.input_signature is not None
        assert compiled.experimental_get_tracing_count() == 1


def test_production_and_evaluator_are_wired_to_shared_assembler() -> None:
    transition_source = inspect.getsource(engine._run_value_filter_branch_axis_gaussian_xla)
    evaluator_source = inspect.getsource(engine._make_frozen_transition_evaluator)
    assert "_assemble_transition_target(" in transition_source
    assert "_assemble_transition_target(" in evaluator_source


def test_t0_capture_is_rejected_as_a_separate_fit(captured_run) -> None:
    initial_hint, predictive_hint = oracle._exact_hint_factories(captured_run["model"])
    with pytest.raises(ValueError, match="transition capture steps"):
        run_value_filter_branch_axis_gaussian_xla_diagnostic(
            captured_run["adapter"],
            captured_run["observations"],
            captured_run["config"],
            predictive_moment_hint=predictive_hint,
            initial_moment_hint=initial_hint,
            capture_steps=(0,),
            run_identity="invalid-t0-capture",
        )


def test_pf_per_step_uncertainty_observability_closes() -> None:
    model = sv_fixture.sv_model(1, 52)
    observations = sv_fixture.sv_simulate(model, 2, 42)
    result = sv_fixture.sv_particle_reference(
        model, observations, n_particles=256, replicates=4, seed=73
    )
    replicates = tf.constant(result["per_step_replicates"], DTYPE)
    totals = tf.constant(result["total_replicates"], DTYPE)
    covariance_of_mean = tf.constant(result["per_step_mean_covariance"], DTYPE)
    tf.debugging.assert_near(
        tf.reduce_sum(replicates, axis=1), totals, rtol=2e-13, atol=2e-13
    )
    assert math.isclose(
        result["se_total"] ** 2,
        float(tf.reduce_sum(covariance_of_mean).numpy()),
        rel_tol=2e-12,
        abs_tol=2e-13,
    )


def test_stage2_classification_requires_per_step_qmc_precision() -> None:
    half_widths = {2: 0.0012809624, 3: 0.0011314002, 4: 0.4008769259}
    fit_means = {2: 0.043976, 3: 1.524930, 4: 7.440936}
    state_means = {2: -0.001934, 3: -0.012090, 4: -0.292810}
    christoffel = {
        "summaries": {
            str(step): {
                "log_z_t": {"half_width_95": half_widths[step]},
                "e_fit": {
                    "mean": fit_means[step],
                    "half_width_95": half_widths[step],
                },
                "e_state": {
                    "mean": state_means[step],
                    "half_width_95": half_widths[step],
                },
            }
            for step in stage2_diag.CAPTURE_STEPS
        }
    }
    result = stage2_diag._classify_steps(
        christoffel, {"per_step_se": [0.0] * 5}
    )

    assert result["2"]["classification"] == "unresolved_qmc_precision"
    assert result["2"]["fit_material"] is False
    assert result["2"]["descriptive_fit_material"] is True
    assert result["3"]["classification"] == "fit_and_state_material"
    assert result["3"]["qmc_precision_pass"] is True
    assert result["4"]["classification"] == "unresolved_qmc_precision"
    assert result["4"]["fit_material"] is False
    assert result["4"]["descriptive_fit_material"] is True
