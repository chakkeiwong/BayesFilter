# KSC-UKF NeuTra/HMC Reset Memo

Date: 2026-08-03

## Restart Point

Resume from
`docs/plans/bayesfilter-ksc-ukf-neutra-hmc-terminal-result-2026-08-02.md`.
The repaired KSC route and its target signature are ready; do not use the old
`KSC-UKF` registry entry, which remains the historical `T=1000` single-Gaussian
route.

## First Restart Action

Relaunch the unchanged trusted GPU broad-grid command with a fresh output root:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/bayesfilter-mpl \
/home/chakwong/anaconda3/envs/tf-gpu/bin/python \
docs/benchmarks/run_neutra_ksc_gaussian_sum_ukf_end_to_end_20260802.py \
  --action broad-grid --screen-results 65 --broad-grid-seed 20260803 2881 \
  --frozen-transport \
  docs/plans/artifacts/bayesfilter-ksc-ukf-neutra-hmc-20260802/final-training-attempt01/KSC-UKF-GAUSSIAN-SUM-T20/final/segments/steps-004001-005000/frozen_transport.json \
  --frozen-transport-sha256 dbbaba3735404d9dd98b233e9419ab4fd3d82c8ac9a5922c9e47712d42e8bddb \
  --output-root docs/plans/artifacts/bayesfilter-ksc-ukf-neutra-hmc-20260802/broad-grid-attempt04
```

The previous three broad-grid attempts were blocked before process creation by
platform permission-review timeouts. They are not candidate failures and must
not be overwritten or interpreted as tuning results.

## Handoff Rules

- Require the public broad-grid artifact to report a complete viable pair set;
  preserve all viable pairs without stochastic ranking.
- Launch sequential HMC only when exactly one primary pair survives, using the
  same frozen transport and broad-grid SHA-256.
- If zero or more than one viable pair survives, record the terminal broad-grid
  result and stop before sequential HMC. This outcome is a handoff gate, not a
  research-direction rejection; the narrowing or repair step needs a new plan
  decision.
- Keep acceptance and energy as diagnostics/health evidence; they do not veto a
  candidate by point estimate alone.
- Use fresh roots and record command, commit, environment, GPU/XLA, memory
  policy, target signature, seed, wall time, and artifact hashes.

## Known Debt And Clarifications (2026-08-03 audit)

- The top-level `final-training-attempt01/.../result.json` decision is
  `TUNING_FAILED` because the superseded in-cell legacy tuner failed on
  acceptance-telemetry provenance after training completed. The training
  subtree passed (`PASS_TRAINING_HARD_GATES`, frozen/trainable parity clean);
  the failed stage is replaced by the separate broad-grid action per the
  continuation plan phase 4 and does not veto the frozen transport.
- The NeuTra route-ledger guard (`tests/test_neutra_hmc_route_policy.py`)
  fails repo-wide with about 20 unledgered qualifying routes, including
  `bayesfilter/inference/neutra_end_to_end.py`. This is pre-existing debt from
  earlier campaigns. Broad-grid tuning discards all draws and trains nothing;
  the ledger must be repaired before the claim-bearing sequential HMC result is
  accepted. The ledger discovery markers also miss the new
  broad-grid/sequential entry points and need extending.
- 2026-08-03 execution note: skeptical pre-execution audit passed. Broad-grid
  manifest provenance fixes were applied before relaunch (manifest
  `plan_path` now uses the CellSpec plan, tuner `evidence_path` is passed from
  the CellSpec, and `environment`/`device` fields were added). The launch
  command itself is unchanged.

