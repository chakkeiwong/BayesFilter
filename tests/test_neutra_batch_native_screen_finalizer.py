from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from docs.benchmarks import finalize_lgssm_neutra_batch_native_screen as finalizer
from bayesfilter.testing import lgssm_neutra_strict_training_tf as strict_training


ROOT = Path(__file__).resolve().parents[1]
PHASE7 = ROOT / "docs/plans/artifacts/neutra-batch-native-training-2026-07-14/phase7"


def test_fresh_screen_finalizer_revalidates_existing_phase7_evidence() -> None:
    smoke_root = PHASE7 / "fresh-protocol"
    screen_root = PHASE7 / "screen-500"
    smoke = tuple(
        finalizer._result(smoke_root, "smoke", recipe)
        for recipe in finalizer.SCREEN_RECIPE_ORDER
    )
    screen = tuple(
        finalizer._result(screen_root, "screen", recipe)
        for recipe in finalizer.SCREEN_RECIPE_ORDER
    )
    finalizer._validate_rows(smoke, expected_kind="smoke", expected_steps=5)
    finalizer._validate_rows(screen, expected_kind="screen", expected_steps=500)
    finalizer._validate_cross_row_identity((*smoke, *screen))
    result = json.loads(
        (screen_root / "screen" / "result.json").read_text(encoding="utf-8")
    )
    assert result["selection"]["selected_recipe_id"] == "wide_2x_lr5e3"
    assert result["selection"]["ranking_statistically_supported"] is False


def test_finalizer_import_does_not_load_numpy_or_tensorflow() -> None:
    code = (
        "import importlib.util,json,sys; "
        "from pathlib import Path; "
        "p=Path('docs/benchmarks/finalize_lgssm_neutra_batch_native_screen.py'); "
        "s=importlib.util.spec_from_file_location('finalizer_probe',p); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "print(json.dumps({'numpy':'numpy' in sys.modules,"
        "'tensorflow':'tensorflow' in sys.modules},sort_keys=True))"
    )
    completed = subprocess.run(
        (sys.executable, "-c", code),
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
        cwd=ROOT,
    )
    assert completed.stdout.strip() == '{"numpy": false, "tensorflow": false}'


def test_finalizer_artifact_hash_is_self_consistent() -> None:
    path = PHASE7 / "screen-500" / "screen" / "result.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = finalizer._stable_artifact_hash(payload)
    assert payload["artifact_hash"] == expected
    assert payload["screen_weights_reused_by_final"] is False
    assert payload["evidence_role"] == "proxy_nomination_only_not_transport_promotion"


def test_selected_recipe_file_binds_to_result_file_hash() -> None:
    screen_root = PHASE7 / "screen-500"
    result_path = screen_root / "screen" / "result.json"
    selected = json.loads(
        (screen_root / "selected_recipe.json").read_text(encoding="utf-8")
    )
    assert selected["selection_result"]["file_sha256"] == hashlib.sha256(
        result_path.read_bytes()
    ).hexdigest()
    assert selected["screen_weights_reused"] is False


def test_strict_final_job_requires_and_validates_selected_recipe() -> None:
    path = PHASE7 / "screen-500" / "selected_recipe.json"
    recipe, seed, steps, reference = strict_training._job_spec(
        job_kind="final",
        job_id="dense_seed1201",
        selected_recipe_path=path,
    )
    assert recipe.recipe_id == "wide_2x_lr5e3"
    assert seed == strict_training.FINAL_SEEDS["dense_seed1201"]
    assert steps == 5000
    assert reference is not None
    assert reference["screen_weights_reused"] is False


def test_historical_selection_is_job_scoped_not_retroactively_invalidated() -> None:
    path = PHASE7 / "screen-500" / "selected_recipe.json"
    try:
        strict_training._job_spec(
            job_kind="final",
            job_id="dense_seed1203",
            selected_recipe_path=path,
        )
    except strict_training.StrictLGSSMNeuTraTrainingError as error:
        assert "final contract mismatch" in str(error)
    else:
        raise AssertionError("historical two-seed selection authorized seed1203")


def test_strict_final_job_rejects_missing_selected_recipe() -> None:
    try:
        strict_training._job_spec(
            job_kind="final",
            job_id="dense_seed1201",
        )
    except strict_training.StrictLGSSMNeuTraTrainingError as error:
        assert "explicit selected recipe" in str(error)
    else:
        raise AssertionError("final job accepted an implicit selected recipe")


def test_strict_final_job_rejects_tampered_selected_recipe(tmp_path: Path) -> None:
    source = PHASE7 / "screen-500" / "selected_recipe.json"
    selected = json.loads(source.read_text(encoding="utf-8"))
    selected["selected_recipe"]["recipe_id"] = "source_anchor_lr5e3"
    path = ROOT / (
        "docs/plans/artifacts/neutra-batch-native-training-2026-07-14/phase7/"
        f"screen-500/finalization-attempts/tampered-{tmp_path.name}.json"
    )
    path.write_text(json.dumps(selected), encoding="utf-8")
    try:
        try:
            strict_training._job_spec(
                job_kind="final",
                job_id="dense_seed1201",
                selected_recipe_path=path,
            )
        except strict_training.StrictLGSSMNeuTraTrainingError as error:
            assert "artifact hash mismatch" in str(error)
        else:
            raise AssertionError("final job accepted a tampered selected recipe")
    finally:
        path.unlink(missing_ok=True)
