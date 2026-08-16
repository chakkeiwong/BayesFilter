from __future__ import annotations

import ast
import os
from pathlib import Path


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / (
    "docs/benchmarks/run_ssl_lstm_q20_gpu_native_eigh_localization_2026_07_31.py"
)
PLAN = ROOT / (
    "docs/plans/bayesfilter-ssl-lstm-q20-gpu-native-eigh-localization-plan-2026-07-31.md"
)


def test_localization_surface_is_bounded_and_has_no_campaign_modes() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert "numpy" not in imports
    assert 'choices=("gpu",)' in source
    assert "MATERIAL_CAP_SECONDS = 1800.0" in source
    assert "for index in range(1, 4)" in source
    assert "sample_chain(" not in source
    assert "HamiltonianMonteCarlo(" not in source
    assert "TUNING_STEPS" not in source
    assert "FINAL_MAX_STEPS" not in source


def test_localization_requires_parity_before_native_trainer() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    parity = source.index('"backend_parity"')
    trainer = source.index('"construct_native_trainer"')

    assert parity < trainer
    assert 'target_program("compiled_custom_op")' in source
    assert 'target_program("tensorflow_eigh_strict")' in source
    assert "VALUE_ATOL = 1.0e-8" in source
    assert "SCORE_ATOL = 1.0e-7" in source


def test_plan_records_host_callback_root_cause_and_nonclaims() -> None:
    plan = PLAN.read_text(encoding="utf-8")

    assert "cudaDeviceSynchronize()" in plan
    assert "Eigen" in plan
    assert "GPU-native" in plan
    assert "No tuning arm, final stream, HMC" in plan


def test_current_custom_op_cuda_callback_is_classified_as_host_staged() -> None:
    source = (ROOT / "bayesfilter/ops/symmetric_sylvester_op.cc").read_text(
        encoding="utf-8"
    )

    sylvester = source[
        source.index("SymmetricSylvesterXlaImpl_Gpu") : source.index(
            "XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(\n"
            '    "SymmetricSylvesterXlaImpl", SymmetricSylvesterXlaImpl_Gpu'
        )
    ]
    principal = source[
        source.index("SymmetricPrincipalSqrtXlaImpl_Gpu") : source.index(
            "XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(\n"
            '    "SymmetricPrincipalSqrtXlaImpl", SymmetricPrincipalSqrtXlaImpl_Gpu'
        )
    ]
    for callback in (sylvester, principal):
        assert "cudaDeviceSynchronize()" in callback
        assert "cudaMemcpyDeviceToHost" in callback
        assert "cudaMemcpyHostToDevice" in callback
        assert "cudaStreamSynchronize(stream)" in callback
