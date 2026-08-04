# Clean Reset Memo: NeuTra Four Blocked Targets

Date: 2026-07-31

## Restart Point

Resume from the reviewed plan
`docs/plans/bayesfilter-neutra-four-blocked-target-repair-and-admission-plan-2026-07-31.md`
and terminal result
`docs/plans/bayesfilter-neutra-four-blocked-target-repair-result-2026-07-31.md`.

No NeuTra training or HMC was launched for `SVX-SGQF`, `SVX-ZC`, `KSC-UKF`, or
`PP-ZC` during this repair campaign.

## Current State

- `SVX-SGQF`: terminally blocked under the current SGQF mechanism. Levels
  10, 12, 16, 20, and 24 all failed the unchanged dense-prefix admission gate;
  the error plateau is not a launch failure.
- `KSC-UKF`: new mass-preserving clustered Gaussian-sum repair passes the CPU
  T20 dense value/score screen at caps 7-256. Focused tests pass. A trusted
  GPU/XLA canary is still required before target identity or training. The
  earlier canary initialized GPU/XLA but exposed and then fixed a static
  `while_loop` requirement; the final retry was blocked before process creation
  by platform permission-review timeouts.
- `SVX-ZC`: the monograph fixed-branch candidate passes structural and
  same-scalar FD checks, but ranks 1, 2, 4, and 6 all fail the declared
  rank-saturation residual veto. It remains blocked on numerical admission.
- `PP-ZC`: fresh scope-specific tuning and sealed CPU/GPU implementation gates
  pass. The assembled route remains `extension_or_invention`, and it still has
  no registered batch-native posterior adapter or frozen HMC chart/Jacobian;
  it remains blocked on the NeuTra target contract, not source mismatch.

## First Restart Action

Run one fresh trusted GPU/XLA KSC cap-32 canary using the repaired runner:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/bayesfilter-mpl \
/home/chakwong/anaconda3/envs/tf-gpu/bin/python \
docs/benchmarks/run_neutra_ksc_gaussian_sum_ukf_admission_20260731.py \
--gpu-canary \
--output-root \
docs/plans/artifacts/bayesfilter-neutra-four-blocked-target-repair-20260731/ksc-ukf/attempt09
```

Use trusted/escalated GPU execution. The output root must be fresh. Do not
launch NeuTra training from this command.

If cap-32 GPU/CPU value and score parity passes, write a separate reviewed
scope-specific tuning/training plan and issue identity only from that plan.
If parity fails, preserve the CPU admission as diagnostic evidence and keep
`KSC-UKF` blocked for NeuTra.

## Do Not Do

- Do not overwrite attempts 01-08 or reuse their output directories.
- Do not weaken the dense reference thresholds.
- Do not relabel `extension_or_invention` as `source_faithful`.
- Do not use the CPU KSC result as GPU, NeuTra, HMC, posterior, or default-readiness evidence.
- Do not start a broad four-model run while any target lacks a valid target identity.
