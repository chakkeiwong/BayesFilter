# Corrected Parameter-Authority Phase 40 Subplan

Parent: `bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md`  
Version: `v2.2-root-group-stratified`  
Entry gate: Phase 39 identified storage-order/ancestry split defect  
Status: `PASS_V2_2_ROOT_GROUP_BOUNDARY_MEASURE_DIAGNOSTIC_REPAIR_TRIGGERED`  
Local cap: 1200 s

## Question

After replacing the order-dependent v2.1 split with a deterministic,
root-disjoint, sign-balanced split, do identity and exact-training-measure
affine NeuTra screens show smaller validation/audit discrepancies, or do large
residuals persist under a more defensible empirical partition?

## Corrected split contract

Use the existing N=256 M0 theta bank only as a fixed diagnostic source. Group
rows by the recorded terminal SMC root. Fail closed if a root crosses the
declared sign boundary. Within each sign, order root groups by

`SHA256("root_group_stratified_v1:" || root_id)`.

Use a deterministic subset-sum selection to allocate whole groups closest to
six rows per sign to audit, then six rows per sign to validation, and put every
remaining row in training. Record exact row/root counts, sign counts, root
disjointness, and complete row coverage. No root may occur in more than one
partition. The split is a reviewed hypothesis, not a guarantee of target
representativeness.

## Evidence contract

**Comparator:** identity versus train-measure affine, same N=256 M0 bank,
target signature, proposal log density, seeds, 200-step budget, and GPU/XLA
batch-native trainer.

**Hard gates:** theta `[N,4]`, target/status finite, source/protocol hash match,
positive-definite affine covariance and exact train oracle, root-disjoint and
complete split, finite transport round trips, memory growth verified, and
checkpoint receipts if requested.

**Promotion veto:** any IID, posterior, HMC, canonical LEDH, or superiority
claim. Moment/loss differences remain descriptive. A failed candidate triggers
objective/support repair; it does not invalidate the q=20 target.

## Commands

```text
TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase31_2026_08_25.py \
  --authority-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase37-support-ladder/n256 \
  --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase40-root-group-stratified-boundary/identity-trace \
  --precondition identity --split-policy root_group_stratified_v1 --steps 200 \
  --checkpoint-steps 1 5 10 25 50 100 150 200 --seed 20260825 4011

TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase31_2026_08_25.py \
  --authority-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase37-support-ladder/n256 \
  --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase40-root-group-stratified-boundary/affine-trace \
  --precondition affine --split-policy root_group_stratified_v1 --steps 200 \
  --checkpoint-steps 1 5 10 25 50 100 150 200 --seed 20260825 4011
```

The reporter may reuse the Phase 38 selection rule only after verifying the
new v2.2 plan version and split receipt; it must write a fresh result root.

The receipts and the adjacent repair refresh are preserved at:

- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase40-result-2026-08-25.md`
- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase40-repair-refresh-2026-08-25.md`

The remaining finite-support discrepancy is an entry condition for the
versioned v2.3 independent-bank phase, not a reason to relabel this phase as
whitening evidence.

## Pre-mortem, repair, and stop

Root groups can be large and force unequal row counts; record the actual counts
and do not silently split a root. Small holdouts can still have high moment
variance. A passing boundary can still have poor whitening. These are
diagnostics or repair triggers. Stop only if the target/common support is
unavailable, the root-group contract cannot be constructed, or the campaign
budget is exhausted.
