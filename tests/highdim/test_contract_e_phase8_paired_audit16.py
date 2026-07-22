from __future__ import annotations

import importlib.util
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "docs/benchmarks/run_contract_e_phase8_paired_audit16.py"
SPEC = importlib.util.spec_from_file_location("contract_e_paired_audit16", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_frozen_seed_block_and_shape_are_disjoint_from_reserved_sets() -> None:
    assert audit.ESTIMATOR_SEEDS == tuple(range(81220, 81236))
    assert not set(audit.ESTIMATOR_SEEDS) & {80920}
    assert not set(audit.ESTIMATOR_SEEDS) & set(range(81020, 81025))
    assert not set(audit.ESTIMATOR_SEEDS) & set(range(81120, 81184))
    assert audit.TIME_STEPS == 2
    assert audit.NUM_PARTICLES == 128


def test_interval_uses_bessel_corrected_sample_standard_deviation() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    interval = audit._interval(values, critical=2.0)
    expected_sd = math.sqrt(5.0 / 3.0)
    assert interval["mean"] == 2.5
    assert math.isclose(interval["sample_standard_deviation"], expected_sd)
    assert math.isclose(interval["standard_error"], expected_sd / 2.0)
    assert math.isclose(interval["half_width"], expected_sd)


def test_interval_rejects_nonfinite_or_singleton_values() -> None:
    for values in ([1.0], [1.0, float("nan")]):
        try:
            audit._interval(values, critical=2.0)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid Student inputs were accepted")


def _identity(mask_hash: str) -> dict[str, object]:
    return {
        "preparation_id": "p",
        "residual_design_id": "r",
        "rng_algorithm": "philox",
        "root_seeds_in_order": list(audit.ESTIMATOR_SEEDS),
        "num_particles": 128,
        "time_steps": 2,
        "sinkhorn_steps": 20,
        "row_chunk_size": 16,
        "col_chunk_size": 16,
        "tensor_sha256": {
            "fixed_reset_mask": mask_hash,
            "initial_noise": "same-initial",
            "transition_noise": "same-transition",
            "residual_design": "same-residual",
            "observations": "same-observation",
            "prepared_ridge": "same-ridge",
            "epsilon": "same-epsilon",
            "scaling": "same-scaling",
        },
    }


def test_paired_identity_allows_only_reset_mask_to_differ() -> None:
    checks = audit._paired_preparation_identity_check(
        {"preparation_identity": _identity("active")},
        {"preparation_identity": _identity("inactive")},
    )
    assert all(checks.values())


def test_bonferroni_family_is_exactly_six_prespecified_quantities() -> None:
    assert audit.FAMILY_SIZE == 6
    assert audit.QUANTITY_NAMES == (
        "value",
        "phi1",
        "phi2",
        "phi3",
        "q_scale",
        "r_scale",
    )
