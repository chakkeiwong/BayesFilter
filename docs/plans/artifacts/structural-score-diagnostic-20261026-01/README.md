# Completed structural score diagnostics

The investigation is closed. Start with the
[final report](../../structural-score-final-report-2026-10-26.md).
`result.md` contains detailed numerical results, `source-audit.md` the
call-chain and mathematical audit, and `execution-record.json` the attempt
record and timing limitations. Raw evidence is preserved under `attempts/`.
Original metadata errors are identified separately in
`metadata-corrections.json`; historical manifests have not been rewritten.

## Reproducing the diagnostic inputs

`reproduction-inputs.tar.xz` contains the original T2/T50 observation files
and fixed particle-design files, byte for byte. The design files were absent
from Git and otherwise depended on a temporary source worktree. Their
portable archive closes that preservation gap. `reproduction-inputs.json`
records the archive and member SHA-256 checksums. The archived files were
read back and checked against those checksums before commit.

From the repository root, extract into a fresh local directory:

```bash
mkdir -p /tmp/structural-score-reproduction-inputs
tar -xJf docs/plans/artifacts/structural-score-diagnostic-20261026-01/reproduction-inputs.tar.xz -C /tmp/structural-score-reproduction-inputs
```

Pass `/tmp/structural-score-reproduction-inputs` to the diagnostic runner's
`--source-root` argument. Each launch must use a fresh `--output-root`. The
original environment and commands are recorded in `commands.txt` and the
attempt manifests. The checked environment is the local `tf-gpu` conda
environment; the repository's Sylvester custom op must be built for the
chosen TensorFlow environment before importing the structural target.

Modes are the full diagnostic (no mode flag), `--t50-only`,
`--reset-only --cpu`, and `--fp64-only --cpu`. CPU runs deliberately hide
CUDA devices; GPU runs must use the trusted/escalated execution context
required by repository policy. The full diagnostic uses FP32/N=4032; the
FP64 mode is a short-horizon diagnostic exception. Reproduction instructions
do not schedule or authorize a new research campaign.

No production implementation, controls, or HMC target changed. The remaining
T50 accuracy question and optional terminal-reset repair are documented for
a separate future investigation, not left as active work in this one.
