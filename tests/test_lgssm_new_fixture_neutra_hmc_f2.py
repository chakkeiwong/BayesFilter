from __future__ import annotations

import ast
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from bayesfilter.testing import lgssm_new_fixture_neutra_hmc_f2_tf as campaign


def test_f2_inputs_bind_exact_payload_target_and_comparator() -> None:
    source, loaded, bundle, _adapter, comparator = campaign._load_inputs()
    assert source["payload_file_sha256"] == campaign.EXPECTED_PAYLOAD_SHA256
    assert loaded.artifact_signature
    assert bundle.target_signature == campaign.EXPECTED_TARGET_SIGNATURE
    assert comparator["artifact_hash"] == campaign.EXPECTED_COMPARATOR_ARTIFACT_HASH


def test_f2_seeds_are_disjoint_and_caps_are_in_source() -> None:
    seeds = (
        campaign.PROBE_SEED, campaign.ADMISSION_WARMUP_SEED,
        campaign.ADMISSION_RETAINED_SEED, campaign.CONFIRMATION_WARMUP_SEED,
        campaign.CONFIRMATION_RETAINED_SEED,
    )
    assert len(set(seeds)) == len(seeds)
    source = Path(campaign.__file__).read_text(encoding="utf-8")
    assert source.count("warmup_max_results=10000") == 2
    assert source.count("retained_max_results=10000") == 2


def test_f2_route_has_no_numpy_or_local_tfp_sampler() -> None:
    source = Path(campaign.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    assert not any(name == "numpy" or name.startswith("numpy.") for name in imported)
    assert "tf.numpy_function" not in source
    assert "tf.py_function" not in source
    assert "tfp.mcmc.sample_chain" not in source
    assert "tfp.mcmc.HamiltonianMonteCarlo" not in source
