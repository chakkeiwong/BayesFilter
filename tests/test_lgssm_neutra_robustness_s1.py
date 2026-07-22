from __future__ import annotations

import ast
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from bayesfilter.testing import lgssm_neutra_robustness_s1_tf as campaign


def test_s1_training_identity_loads_and_binds_fresh_payload() -> None:
    source, loaded, bundle, adapter = campaign._load_candidate()
    assert source["training_seed"] == (20260715, 1203)
    assert source["payload_file_sha256"] == campaign.EXPECTED_PAYLOAD_SHA256
    assert loaded.artifact_signature
    assert bundle.target_signature == campaign.reference.EXPECTED_TARGET_SIGNATURE
    assert adapter.adapter_signature()


def test_s1_route_has_no_numpy_or_local_tfp_sampler() -> None:
    path = Path(campaign.__file__)
    source = path.read_text(encoding="utf-8")
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
    assert "tfp.mcmc.sample_chain" not in source
    assert "tfp.mcmc.HamiltonianMonteCarlo" not in source


def test_s1_seed_ledger_is_disjoint() -> None:
    seeds = (
        campaign.PROBE_SEED,
        campaign.ADMISSION_WARMUP_SEED,
        campaign.ADMISSION_RETAINED_SEED,
        campaign.CONFIRMATION_WARMUP_SEED,
        campaign.CONFIRMATION_RETAINED_SEED,
    )
    assert len(set(seeds)) == len(seeds)
