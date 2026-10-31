"""Diagnostic checks for the explicitly reviewed orchestration-only migration."""

import hashlib
from pathlib import Path

from bayesfilter.testing import surrogate_hmc_scope_source_compatibility as compat


def test_only_reviewed_runtime_hash_maps_to_saved_scope_without_mutation():
    hashes = {
        compat.RUNTIME_PATH: compat.REVIEWED_RUNTIME_SHA256,
        "bayesfilter/highdim/ledh_canonical_score_tf.py": "numerical-code-hash",
    }
    normalized = compat.scope_source_hashes(hashes)
    assert normalized[compat.RUNTIME_PATH] == compat.SAVED_RUNTIME_SHA256
    assert normalized["bayesfilter/highdim/ledh_canonical_score_tf.py"] == (
        "numerical-code-hash"
    )
    assert hashes[compat.RUNTIME_PATH] == compat.REVIEWED_RUNTIME_SHA256


def test_unknown_runtime_and_unrelated_hashes_remain_strict():
    hashes = {compat.RUNTIME_PATH: "unreviewed-source", "other.py": "unchanged"}
    assert compat.scope_source_hashes(hashes) == hashes
    assert compat.scope_source_hashes({"other.py": "unchanged"}) == {
        "other.py": "unchanged"
    }


def test_provenance_preserves_actual_runtime_digest_for_unreviewed_source(tmp_path):
    runtime = tmp_path / compat.RUNTIME_PATH
    runtime.parent.mkdir(parents=True)
    runtime.write_text("# unknown runtime\n")
    actual = hashlib.sha256(runtime.read_bytes()).hexdigest()
    provenance = compat.source_migration_provenance(tmp_path)
    assert provenance["migration_id"] is None
    assert provenance["executed_runtime_sha256"] == actual
    assert provenance["tuning_scope_runtime_sha256"] == actual


def test_current_runtime_cannot_inherit_historical_tuning_identity():
    # NeuTra integration changed the runtime after the exact October 18
    # orchestration migration. Its fresh tuning must bind its actual source,
    # not silently inherit the earlier run's compatibility identity.
    root = Path(__file__).resolve().parents[1]
    actual = hashlib.sha256((root / compat.RUNTIME_PATH).read_bytes()).hexdigest()
    assert actual != compat.REVIEWED_RUNTIME_SHA256
    provenance = compat.source_migration_provenance(root)
    assert provenance["migration_id"] is None
    assert provenance["executed_runtime_sha256"] == actual
    assert provenance["tuning_scope_runtime_sha256"] == actual
