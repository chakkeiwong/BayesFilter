# SQMC repair evidence

The current result is [SQMC repair results](../../../benchmarks/sqmc-repair-results-20260925.md).

Each numbered attempt preserves its original manifest and logs. Attempts 01
and 02 failed an independent old TensorFlow autodiff/XLA fixture, 07 failed
collection due to an absent historical runner, and 09 is the final passing
73-test suite. CPU runs intentionally hid GPUs. GPU attempts 04 and 08
configured memory growth and recorded actual devices. 08 is the final
eight-case compatibility check. 06 contains the disjoint-seed tuning workflow.

replay-scripts/ preserves the exact scripts used from /tmp. Their original
output paths are intentionally fixed and fail if already present. For a new
authorized replay, copy the script, choose a fresh output directory, retain its
CPU/GPU and numerical settings, and use the original manifest command. Never
overwrite this evidence. The CPU runner itself prints the child exit status;
the authoritative result is manifest.exit_code, not its wrapper exit status.

source-snapshot/ contains the repaired Python files. tracked-changes.patch
preserves tracked edits against the baseline commit. completion-manifest.json
hashes repository and TensorFlow filter dependencies, records the uncommitted
state, and links the plan/result. Installed TensorFlow/TFP and environment
identities are in per-run manifests/logs.

historical-docs/ and historical-controls/ are archival only. They are not new
accuracy, tuning, ranking, or HMC evidence. The cited historical Austria raw
data and the older mechanics runner were unavailable.
