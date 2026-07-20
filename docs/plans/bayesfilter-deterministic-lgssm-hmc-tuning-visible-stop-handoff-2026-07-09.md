# BayesFilter Deterministic LGSSM HMC Tuning Visible Stop Handoff

Date: 2026-07-11

Status: `BLOCKED_PHASE7_PREFLIGHT_HASH_CONTRACT_MISMATCH`

Active gate: P7G private-replay refresh exact-hash validation.

## Blocking Condition

The authorized Phase 6 private-replay refresh completed successfully and
passed its own tuning gate, but did not reproduce the pinned public-kernel,
private-loop-kernel, or selected-trajectory hashes. The exact-hash mismatch is
a predeclared Phase 7 continuation veto.

The committed and refreshed private event artifacts agree on the selected HMC
mechanics fields available in both. The refreshed artifacts add current
`handoff_screen_policy` provenance to hashed stage lineage. This supports an
engineering hash-contract/baseline-migration diagnosis, but the old full
private replay was never persisted and complete payload identity is therefore
not checked. Do not replace exact identity with equal acceptance or other
descriptive evidence.

## Command Run

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-bayesfilter-phase7-refresh \
python docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py \
  --stage kernel_tuning \
  > /tmp/bayesfilter-phase7-phase6-refresh.log 2>&1
```

The command exited `0`. GPU devices were intentionally hidden; TensorFlow host
XLA compilation was observed.

## Gate State

- P7G: `BLOCK_PRIVATE_REPLAY_REFRESH_HASH_MISMATCH`.
- P7H tiny actual-target multicore XLA smoke: not executed.
- P7I serious burn-in and retained sampling: not executed.
- Phase 8 recovery: not approved and not executed.
- NeuTra training: not approved and not executed.

## Artifacts

- Plan:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-phase7-repair-and-execution-plan-2026-07-11.md`.
- Result/blocker:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase7-burnin-sampling-result-2026-07-09.md`.
- Public structured blocker:
  `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/burnin_sampling.json`.
- Refreshed public kernel artifact:
  `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning.json`.
- Ignored private replay:
  `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/private_diagnostics/kernel_tuning_replay.json`.
- Full refresh log: `/tmp/bayesfilter-phase7-phase6-refresh.log`.

## Exact Next Safe Action

Write and review a narrow baseline-migration repair that separates semantic
HMC-mechanics identity from policy/provenance content hashes. It must compare
the committed private event, refreshed private replay, fixed target and mass
identities, selected step, leapfrog count, trajectory length, adapter stack,
and replay reconstruction. It may update Phase 7 pins only after that proof and
a fresh review. Do not launch smoke, serious Phase 7, Phase 8, or NeuTra while
this handoff remains active.
