from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
from bayesfilter.testing.lgssm_neutra_training_tf import (
    LGSSMAffineNeuTraTrainingConfig,
    LGSSMNeuTraTrainingError,
    _historical_train_and_validate_lgssm_affine_neutra,
    train_and_validate_lgssm_affine_neutra,
)


def test_lgssm_affine_neutra_training_is_retired_before_artifact_write(tmp_path) -> None:
    artifact_dir = tmp_path / "artifacts"
    validation_path = tmp_path / "validation.json"

    with pytest.raises(LGSSMNeuTraTrainingError, match="retired migration debt"):
        train_and_validate_lgssm_affine_neutra(
            LGSSMAffineNeuTraTrainingConfig(
                seed=20260707,
                steps=4,
                batch_size=8,
                learning_rate=0.01,
                artifact_dir=artifact_dir,
                validation_path=validation_path,
            )
        )
    assert not artifact_dir.exists()
    assert not validation_path.exists()


def test_lgssm_affine_historical_body_is_non_executable(tmp_path) -> None:
    config = LGSSMAffineNeuTraTrainingConfig(
        artifact_dir=tmp_path / "artifacts",
        validation_path=tmp_path / "validation.json",
    )
    with pytest.raises(LGSSMNeuTraTrainingError, match="non-executable"):
        _historical_train_and_validate_lgssm_affine_neutra(config)

    assert not config.artifact_dir.exists()
    assert not config.validation_path.exists()
