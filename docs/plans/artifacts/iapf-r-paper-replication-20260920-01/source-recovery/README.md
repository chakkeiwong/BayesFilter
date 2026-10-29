# Source snapshot repair

The original driver copied source files using basenames. Both
`docs/benchmarks/reference_iapf_paper.R` and `tests/reference_iapf_paper.R`
therefore mapped to `sources/reference_iapf_paper.R`; the test overwrote the
algorithm snapshot. Manifests recorded separate hashes before the copy, and
the experiments executed the repository source rather than those snapshots.

The overwritten algorithm versions were reconstructed by reversing the exact
subsequent diagnostic-only edits. Every recovered byte stream matches its
original pre-run SHA-256: see `recovery.json` and `verify_recovery.py`. The
historical snapshots and manifests remain untouched. This repairs source
availability without replacing their recorded identities. The current driver
preserves repository-relative paths, verifies each copy against its recorded
hash, and has a regression test for this exact basename collision.
