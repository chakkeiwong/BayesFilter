# Phase 12 High-Capacity NeuTra Subplan

Program: `docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-master-program-2026-08-25.md`  
Status: `PASS_HARD_GATES_ROLE_LIMITED_CAPACITY_NOT_RESOLVING_RESIDUALS`  
Budget cap: `3600 s` within the unchanged global `64800 s` cap  
Input: Phase 8 metadata-bound/audited N=300 bank  
Output root:
`docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase12`

## Objective

Test whether the residual latent moments after 300 updates are capacity-limited
for the fixed audited empirical measure. Compare the existing compact arm with
one higher-capacity, three-stage flow under the same target, split, seed, and
300-update budget. This is a representation diagnostic only.

## Audit and evidence contract

The input hash, finite-measure audit, mode axis, and split/weight alignment are
already passing. The high-capacity arm is a hypothesis, not a default. Hard
criteria remain GPU memory growth before device creation, batch-native updates,
finite target/status, exact parity, untouched audit rows, and no HMC. Validation
loss and latent moments are explanatory and cannot establish IID whitening or
rank an arm with one bank.

Run the runner with `--profile capacity --steps 300` on the same audited bank and
a fresh output root. If capacity materially reduces residuals, refresh a
frozen tuning/claim screen; if not, refresh a support/measure limitation note.
