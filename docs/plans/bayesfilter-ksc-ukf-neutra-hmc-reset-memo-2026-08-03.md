# KSC-UKF NeuTra/HMC Reset Memo

Date: 2026-08-03 (updated after broad-grid attempt04)

## Restart Point

Resume from
`docs/plans/bayesfilter-ksc-ukf-broad-grid-result-2026-08-03.md`.
Broad-grid attempt04 completed successfully (`BROAD_GRID_TUNING_VIABLE_PAIR_SET`,
159 s, commit `efce62b5`): exactly one viable primary `(L=25, eps~0.9896)`, but
its compatible `L=24` same-epsilon coverage probe makes the unranked
primary-plus-coverage union size two, so the implemented sequential handoff
gate (exactly one union entry) is not satisfied. Sequential HMC was not
launched. The next step is an owner decision among the three options in the
result note (role-based gate amendment, follow-up grid round, or stop).

The repaired KSC route and its target signature remain ready; do not use the
old `KSC-UKF` registry entry, which remains the historical `T=1000`
single-Gaussian route.

## Completed Restart Action (2026-08-03)

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
not be overwritten or interpreted as tuning results. Attempt04 above is the
first and only execution of this command; its output root is now claim
evidence and must not be overwritten.

## Handoff Rules

- Require the public broad-grid artifact to report a complete viable pair set;
  preserve all viable pairs without stochastic ranking.
- Launch sequential HMC only when the implemented gate is satisfied: the
  unranked primary-plus-coverage union (`next_round_candidates` in the private
  tuning result) contains exactly one entry and that entry is an independently
  tuned primary. Use the same frozen transport and the private broad-grid
  result SHA-256. Note: "exactly one viable primary" is **not** sufficient
  under the implemented union semantics; a compatible coverage probe also
  occupies the union (this fired on attempt04).
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

