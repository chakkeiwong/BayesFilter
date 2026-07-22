from __future__ import annotations

import ast
from pathlib import Path

from bayesfilter.testing import lgssm_new_fixture_neutra_training_f1_tf as campaign


def test_f1_recipe_ladder_and_budgets_are_frozen() -> None:
    assert campaign.RECIPE_ORDER == (
        "inherited_wide_lr5e3", "source_width_lr5e3", "wide_lower_lr1e3"
    )
    assert campaign.SCREEN_STEPS == 500
    assert campaign.FINAL_STEPS == 5000
    assert campaign.BATCH_SIZE == 128
    assert campaign.SCREEN_SEED != campaign.FINAL_SEED


def test_f1_source_has_no_numpy_or_training_python_step_loop() -> None:
    source = Path(campaign.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any(name == "numpy" or name.startswith("numpy.") for name in imported)
    assert "tf.numpy_function" not in source
    assert "tf.py_function" not in source


def test_f1_selection_prefers_lower_capacity_inside_uncertainty_set() -> None:
    assert campaign._parameter_count(campaign.RECIPES["source_width_lr5e3"]) < campaign._parameter_count(campaign.RECIPES["inherited_wide_lr5e3"])
