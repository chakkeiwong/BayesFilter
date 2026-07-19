from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from bayesfilter.hmc_route_contract import (
    HMC_FIXED_TRAJECTORY_STAGE,
    HMC_ROUTE_CONTRACT_VERSION,
    HMC_TOP_LEVEL_SELECTION_STAGE,
    HMC_WINDOWED_MASS_STAGE,
    LEGACY_JOINT_L_EPSILON_ALGORITHM_ID,
    LEGACY_SEGMENTED_WINDOWED_MASS_ALGORITHM_ID,
    OPERATIONAL_FIXED_TRAJECTORY_ALGORITHM_ID,
    OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID,
    UnsupportedHMCAlgorithmRoute,
    require_hmc_algorithm_route,
    resolve_hmc_algorithm_route,
    windowed_algorithm_for_selection_algorithm,
)


def test_route_contract_import_does_not_import_tensorflow() -> None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; import bayesfilter.hmc_route_contract; "
                "print(json.dumps({'tf': 'tensorflow' in sys.modules, "
                "'tfp': 'tensorflow_probability' in sys.modules}))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert json.loads(completed.stdout.strip()) == {"tf": False, "tfp": False}


@pytest.mark.parametrize(
    ("control", "value"),
    [
        ("timeout_enabled", True),
        ("heartbeat_enabled", True),
        ("output_path_enabled", True),
        ("checkpointing_enabled", True),
    ],
)
def test_execution_controls_cannot_select_algorithm(control: str, value: bool) -> None:
    baseline = resolve_hmc_algorithm_route(
        algorithm_id=OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID,
        stage=HMC_WINDOWED_MASS_STAGE,
    )
    controlled = resolve_hmc_algorithm_route(
        algorithm_id=OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID,
        stage=HMC_WINDOWED_MASS_STAGE,
        **{control: value},
    )
    assert controlled.algorithm_id == baseline.algorithm_id
    assert controlled.route_contract_version == HMC_ROUTE_CONTRACT_VERSION
    assert controlled.supported is True


def test_stage_scoped_identity_mapping_is_deterministic() -> None:
    assert windowed_algorithm_for_selection_algorithm(
        OPERATIONAL_FIXED_TRAJECTORY_ALGORITHM_ID
    ) == OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID
    assert windowed_algorithm_for_selection_algorithm(
        LEGACY_JOINT_L_EPSILON_ALGORITHM_ID
    ) == LEGACY_SEGMENTED_WINDOWED_MASS_ALGORITHM_ID


@pytest.mark.parametrize(
    ("algorithm_id", "stage"),
    [
        (OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID, HMC_WINDOWED_MASS_STAGE),
        (LEGACY_SEGMENTED_WINDOWED_MASS_ALGORITHM_ID, HMC_WINDOWED_MASS_STAGE),
        (OPERATIONAL_FIXED_TRAJECTORY_ALGORITHM_ID, HMC_FIXED_TRAJECTORY_STAGE),
        (LEGACY_JOINT_L_EPSILON_ALGORITHM_ID, HMC_TOP_LEVEL_SELECTION_STAGE),
    ],
)
def test_known_stage_route_truth_table(algorithm_id: str, stage: str) -> None:
    decision = resolve_hmc_algorithm_route(algorithm_id=algorithm_id, stage=stage)
    assert decision.supported is True
    assert decision.blocker_code is None
    assert decision.payload()["reports_posterior_convergence"] is False
    assert decision.payload()["reports_sampler_superiority"] is False


@pytest.mark.parametrize(
    ("overrides", "blocker"),
    [
        ({"algorithm_id": "unknown"}, "unknown_or_stage_incompatible_algorithm_id"),
        ({"runtime_backend": "numpy"}, "unsupported_runtime_backend"),
        ({"use_xla": True}, "operational_windowed_warmup_xla_not_validated"),
        ({"runner_identity": "injected"}, "operational_windowed_warmup_requires_default_runner"),
    ],
)
def test_unsupported_route_is_typed_and_never_falls_back(
    overrides: dict[str, object], blocker: str
) -> None:
    arguments: dict[str, object] = {
        "algorithm_id": OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID,
        "stage": HMC_WINDOWED_MASS_STAGE,
    }
    arguments.update(overrides)
    decision = resolve_hmc_algorithm_route(**arguments)
    assert decision.supported is False
    assert decision.algorithm_id == arguments["algorithm_id"]
    assert decision.blocker_code == blocker
    with pytest.raises(UnsupportedHMCAlgorithmRoute) as caught:
        require_hmc_algorithm_route(**arguments)
    assert caught.value.decision == decision


def test_legacy_route_is_explicit_and_non_authoritative() -> None:
    decision = resolve_hmc_algorithm_route(
        algorithm_id=LEGACY_SEGMENTED_WINDOWED_MASS_ALGORITHM_ID,
        stage=HMC_WINDOWED_MASS_STAGE,
        timeout_enabled=True,
        runner_identity="injected",
    )
    assert decision.supported is True
    assert decision.operational_authority is False
    assert decision.promotion_role == "non_promoting"
