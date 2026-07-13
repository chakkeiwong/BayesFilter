from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = (
    ROOT
    / "docs/benchmarks/"
    "benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization_2026_07_09.py"
)


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization",
        HARNESS_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _AffineStandardNormalAdapter:
    target_scope = "test:affine_standard_normal"

    def __init__(self, *, center, factor):
        self.center = np.asarray(center, dtype=float)
        self.factor = np.asarray(factor, dtype=float)

    def latent_to_position(self, z):
        z_tensor = tf.convert_to_tensor(z, dtype=tf.float64)
        center = tf.constant(self.center, dtype=tf.float64)
        factor = tf.constant(self.factor, dtype=tf.float64)
        return center + tf.tensordot(z_tensor, factor, axes=[[-1], [1]])

    def log_prob_and_grad(self, z):
        z_tensor = tf.convert_to_tensor(z, dtype=tf.float64)
        return -0.5 * tf.reduce_sum(tf.square(z_tensor)), -z_tensor


def _phase2s_payload_for_orientation():
    scale = np.array([0.5, 2.0, 1.0, 0.25])
    factor_z = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.2, 1.2, 0.0, 0.0],
            [-0.1, 0.3, 0.8, 0.0],
            [0.05, -0.2, 0.4, 0.9],
        ]
    )
    return {
        "map_local_handoff": {
            "center_free_parameter_values": [0.1, 0.2, 0.3, 0.4],
            "scale": scale.tolist(),
            "factor_z": factor_z.tolist(),
            "coordinate_formula": "free = center_free_parameter_values + scale * (factor_z @ u_new)",
        },
        "initializer": {
            "geometry": {
                "diagnostics": {
                    "center_log_prob": 0.0,
                    "config": {"trust_radius": 0.6},
                }
            }
        },
    }


def _phase2w_reference_payload():
    samples = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [-1.0, -0.0, -0.0, -0.0],
            [-0.0, -2.0, -0.0, -0.0],
        ],
        dtype=float,
    )
    proposal_log_prob = -0.5 * np.sum(samples**2, axis=1) - 0.5 * 4 * np.log(2.0 * np.pi)
    target_log_prob = np.array([0.0, -5.0, -2.0, -6.0], dtype=float)
    return {
        "proposal": {
            "sample_count": 4,
            "samples": samples.tolist(),
            "proposal_log_prob": proposal_log_prob.tolist(),
        },
        "importance_reference": {
            "target_log_prob": target_log_prob.tolist(),
            "proposal_log_prob": proposal_log_prob.tolist(),
        },
    }


def _phase2x_reference_payload():
    standard = np.array(
        [
            [0.5, 0.0, 0.0, 0.0],
            [-0.5, -0.0, -0.0, -0.0],
        ],
        dtype=float,
    )
    shifted_center = np.array(
        [
            0.16900152112527375,
            0.34590014590251295,
            0.47216707577215133,
            -0.3362900480743778,
        ],
        dtype=float,
    )
    offset = np.array([1.0, -0.5, 0.25, 0.75])
    shifted = np.vstack([shifted_center + offset, shifted_center - offset])
    samples = np.vstack([standard, shifted])
    return {
        "proposal": {
            "sample_count": 4,
            "component_counts": {"standard": 2, "shifted": 2},
            "component": ["standard", "standard", "shifted", "shifted"],
            "samples": samples.tolist(),
            "proposal_log_prob": [0.0, 0.0, 0.0, 0.0],
        },
        "importance_reference": {
            "target_log_prob": [0.0, -1.0, 3.0, -2.0],
            "proposal_log_prob": [0.0, 0.0, 0.0, 0.0],
        },
    }


def test_antithetic_partner_index_handles_phase2w_and_phase2x_components() -> None:
    harness = _load_harness()
    phase2w_payload = _phase2w_reference_payload()
    phase2x_payload = _phase2x_reference_payload()

    assert harness.antithetic_partner_index("phase2w", phase2w_payload, 0) == 2
    assert harness.antithetic_partner_index("phase2w", phase2w_payload, 3) == 1
    assert harness.antithetic_partner_index("phase2x", phase2x_payload, 0) == 1
    assert harness.antithetic_partner_index("phase2x", phase2x_payload, 3) == 2


def test_build_anchor_set_includes_center_top_weights_and_partners() -> None:
    harness = _load_harness()

    anchors = harness.build_anchor_set(
        _phase2w_reference_payload(),
        _phase2x_reference_payload(),
        top_count=1,
    )

    assert anchors["built"] is True
    rows = anchors["rows"]
    assert rows[0]["anchor_id"] == "center"
    assert {row["relation"] for row in rows} == {"center", "top_weight", "antithetic_partner"}
    assert anchors["summary"]["source_counts"]["phase2w"] == 2
    assert anchors["summary"]["source_counts"]["phase2x"] == 2


def test_orientation_diagnostic_accepts_row_vector_contract_and_flags_display_ambiguity() -> None:
    harness = _load_harness()
    phase2s = _phase2s_payload_for_orientation()
    scale = np.asarray(phase2s["map_local_handoff"]["scale"], dtype=float)
    factor_z = np.asarray(phase2s["map_local_handoff"]["factor_z"], dtype=float)
    adapter = _AffineStandardNormalAdapter(
        center=phase2s["map_local_handoff"]["center_free_parameter_values"],
        factor=np.diag(scale) @ factor_z,
    )
    anchors = {
        "rows": [
            {
                "anchor_id": "a",
                "source": "phase2w",
                "source_index": 0,
                "relation": "top_weight",
                "u_new": [0.7, -1.0, 0.25, 2.0],
            }
        ]
    }

    diagnostic = harness.evaluate_orientation_diagnostic(adapter, anchors, phase2s)

    assert diagnostic["computed"] is True
    assert diagnostic["vetoes"] == []
    assert diagnostic["summary"]["artifact_bug_indicated"] is False
    assert diagnostic["summary"]["adapter_vs_row_formula_max_abs"] < 1.0e-12
    assert diagnostic["summary"]["wrong_column_vs_adapter_max_abs"] > 0.0
    assert diagnostic["summary"]["display_string_ambiguous"] is True


def test_quadratic_log_prob_and_radial_score_component_are_consistent() -> None:
    harness = _load_harness()

    u = np.array([3.0, 4.0, 0.0, 0.0])
    score = np.array([-6.0, -8.0, 1.0, 2.0])

    assert harness.quadratic_log_prob(-10.0, u) == -22.5
    assert harness.radial_score_component(u, score) == -10.0
    assert harness.radial_score_component(np.zeros(4), score) is None


def test_proposal_log_density_replay_matches_saved_phase2w_logq() -> None:
    harness = _load_harness()
    phase2w_payload = _phase2w_reference_payload()
    phase2x_payload = _phase2x_reference_payload()
    anchors = {
        "rows": [
            {
                "anchor_id": "phase2w_top",
                "source": "phase2w",
                "source_index": 0,
                "relation": "top_weight",
                "u_new": phase2w_payload["proposal"]["samples"][0],
            }
        ]
    }

    replay = harness.replay_proposal_log_densities(anchors, phase2w_payload, phase2x_payload)

    assert replay["computed"] is True
    assert replay["vetoes"] == []
    assert replay["summary"]["artifact_bug_indicated"] is False
    assert replay["summary"]["source_saved_replay_max_abs_delta"] < 1.0e-12
