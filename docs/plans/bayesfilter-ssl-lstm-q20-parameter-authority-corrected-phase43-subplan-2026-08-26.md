# Corrected Parameter-Authority Phase 43 Subplan

Parent: `bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md`  
Version: `v2.5-third-bank-support-diagnostic`  
Entry gate: Phase 42 hard boundary/report passed with the predeclared bank-variability repair trigger  
Status: `PASS_V2_5_THREE_BANK_REPORT_REPAIR_TRIGGERED`  
Local cap: 3600 s

## Question

Is bank A an isolated finite-support/mode draw, or does the same support
variability recur in a third independently generated N=256 theta bank?

## Frozen boundary

Keep the q=20 batch-native target in `theta in R^4`, the v2.2 N=256 M0
root-group training split, normalized training weights, proposal protocol,
identity/affine preconditions, architecture arms, 200-step optimizer budget,
and target/protocol signatures unchanged. The 60D UKF state remains internal.

Reuse banks A and B only as untouched post-training audits. Generate bank C
with seed `(20260826, 4104)` using the exact Phase 28 pilot protocol. Each arm
creates one trainer, consumes only the old 232-row training partition, and
evaluates A, B, and C only after the final update. Bank C cannot enter training,
checkpoint selection, or tuning.

The runner reconstructs each v2.4 trainer with the same arm seed and settings
and requires an exact match to the v2.4 terminal `state_hash`. A mismatch is a
harness/determinism veto; it is not silently treated as a new scientific
state. This gate makes the three-bank comparison an audit of bank variation,
not a confounding comparison of retrained transports.

## Evidence contract

**Primary hard gates.**

1. Bank C passes the `theta_R4` finite/status gates and exact target, M0, and C0
   protocol signatures.
2. Authority, A, B, and C pilot receipts and all required tensor hashes are
   distinct; no copied bank is accepted.
3. The old root-group split is complete and root-disjoint, and the affine
   training-measure oracle is exact.
4. Every arm uses one batch-native GPU/XLA trainer on old rows only, with GPU
   memory growth verified before initialization.
5. Reconstructed terminal state hashes match the v2.4 audit for all four arms.
6. A, B, and C have finite target/status receipts and bank-specific residual,
   support, and fresh-use fields.

**Promotion vetoes.** Any hash/protocol mismatch, fresh-row use during
optimization or selection, non-finite/status-invalid target, shape/measure
mismatch, state-hash mismatch, GPU policy failure, or missing bank receipt.
Material residuals and bank variability remain explanatory diagnostics and
repair triggers; they are not a continuation veto.

## Predeclared interpretation branches

| Result | Role | Next action |
|---|---|---|
| C is B-like and A remains the sole clear outlier | support-draw explanation remains viable | run a larger-N or support-envelope diagnostic before objective changes |
| C is A-like or another bank is comparably poor | recurring support/proposal variability | repair proposal/support generation or test N=512 under frozen trainer |
| all three fresh banks remain materially residual-heavy | objective/capacity remains possible, but not identified | write an objective-repair subplan only after support gate and uncertainty design |
| any hard gate fails | engineering/numerical veto | preserve the failed root, classify, repair, and rerun with a new root |

These are descriptive branches, not rankings. Three banks and one frozen state
do not establish statistical superiority, posterior correctness, or IID
whitening.

## Default and assumption audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|
| N=256 | inherited v2.4 comparison scope | particle-count noise mistaken for proposal failure | compare A/B/C support and later N=512 if needed | hypothesis |
| seed `(20260826,4104)` | fresh campaign allocation | accidental correlated or copied bank | pilot/tensor hash separation | reviewed campaign choice |
| 200 optimizer steps | v2.4 frozen comparator | undertraining mistaken for bank effect | exact state hash and finite trace | comparison hypothesis |
| old 232-row train split | v2.2 root-group contract | training-measure mismatch | split and affine oracle | frozen authority |
| state-hash equality | v2.4 audit receipt | nondeterministic reconstruction | per-arm hash gate | hard audit gate |
| residual thresholds | v2.4 descriptive comparator | proxy promoted to whitening criterion | decision table and nonclaims | explanatory only |

## Skeptical pre-execution audit

The plan survives review on 2026-08-26. It changes only the number of
post-training independent banks and adds a deterministic state-identity gate.
The target, proposal, objective, training rows, promotion vetoes, hardware
class, and campaign budget are unchanged. The artifact answers the stated
question because A, B, and C share a checked terminal trainer state. Bank A
will not be removed based on its observed outcome.

## Pre-mortem

| Failure that could mislead | Distinguishing check | Response |
|---|---|---|
| C is generated with a stale protocol | exact M0/C0 and target signatures | fail closed before GPU training |
| C leaks into training | source/tensor hashes and explicit false-use fields | hard veto and preserve root |
| state reconstruction differs | exact per-arm state hash | classify as harness/determinism repair |
| all banks share proposal bias | support ranges, mode fractions, and target/proposal ratios | do not claim coverage; design support repair |
| residuals appear improved only by old tiny holdout comparison | retain frozen-train and three-bank diagnostics | no objective promotion |

## Commands and artifacts

Generate C in:

`docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase43-third-bank-support/fresh-c-n256/`

Then run the new GPU audit into a unique root:

```text
TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase43_2026_08_26.py \
  --authority-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase37-support-ladder/n256 \
  --fresh-root-a docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase42-independent-bank-replication/fresh-a-n256 \
  --fresh-root-b docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase42-independent-bank-replication/fresh-b-n256 \
  --fresh-root-c docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase43-third-bank-support/fresh-c-n256 \
  --reference-audit-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase42-independent-bank-replication/frozen-two-bank-audit \
  --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase43-third-bank-support/frozen-three-bank-audit \
  --steps 200 --seed 20260826 4211
```

The CPU-hidden reporter consumes that audit and writes a separate report root
under `phase43-third-bank-support/report/`. Failed attempts receive fresh
output roots and an adjacent repair note. Stop only for unavailable common
support/target, an unrepaired repeated infrastructure failure, a platform
block, or exhausted campaign budget.

## Required result artifact

The result must include a run manifest, exact state-hash comparison, decision
and inference-status tables, engineering/numerical/scientific ledgers, the
strongest alternative explanation, overturning evidence, and explicit
nonclaims: no IID Gaussian whitening, posterior correctness, exhaustive mode
discovery, normalizer, HMC, canonical LEDH, superiority, or default promotion.

## Closure and repair refresh (2026-08-26)

Phase 43 passed its engineering, target/status, pilot-independence, and exact
v2.4 state-hash gates. The report classified the result as
`bank_a_isolated_outlier_descriptive`: bank A was the clear descriptive outlier,
while banks B and C were below the old validation comparator for both mean and
off-diagonal covariance in all four arms. This is not a statistical ranking and
does not establish whitening. No fresh bank was pooled, dropped, used for
training, or used for selection.

The result and repair refresh are recorded in:

- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase43-result-2026-08-26.md`
- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase43-repair-refresh-2026-08-26.md`

The predeclared next artifact is Phase 44: an independent N=512 bank evaluated
after the same frozen v2.4 trainer state. It changes no target, objective,
training rows, optimizer settings, or whitening gate. The N=512 calibration
cloud uses 128 rows instead of the N=256 pilot's 64 as a campaign hypothesis
scaled with particle count; it is calibration-only and is recorded as such.
