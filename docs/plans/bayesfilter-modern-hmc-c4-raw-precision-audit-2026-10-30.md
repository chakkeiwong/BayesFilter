# C4 raw-draw precision audit after C3

30 October 2026. Continue the authorized master from the integrated tree,
including the current remote LEDH implementation. Completed C3 evidence stays
bound to its original source; this is a saved-data diagnostic, not another HMC
run or an LEDH benchmark.

## Question and evidence contract

Does the C3 difference between internal precision and independent-reference
agreement arise from a reduction/serialization defect, marginal-scale error,
batch-length sensitivity, or ordinary finite-run error? Inspect **all seven**
candidates and all 31 quantities. The comparator is each terminal stored
lugsail report and the frozen independent reference. The implementation
criterion is independent recomputation of the same means, sample SD, plain
batch means, lugsail variance and original reference-screen arithmetic.
No failed member is removed or relabeled.

Source/hash mismatch, missing draws or quantities, nonfinite original values,
wrong draw counts, or reduction disagreement invalidates this audit and
triggers a localized repair. Original precision/reference failure is a
posterior promotion veto, not a continuation veto. Batch-scale sensitivity,
chain/half means, concentration of variance contributions, marginal/reference
SD ratios and error/MCSE ratios are explanatory only. No sample-only diagnostic
can establish that all relevant tails have been visited.

The strongest alternative explanation is ordinary Monte Carlo fluctuation
combined with repeated-look stopping. The L=25/.2 reference screen is a
one-time check of internal-stopped samples; it is not a calibrated probability
test. Neither a narrow miss nor agreement across estimators proves bias or
correctness. Preserve this distinction in the result.

## Execution and numerical choices

1. Validate the original cohort, source manifest, result counts and reference;
   require the integrated `hmc_precision.py` to match the recorded C3 source.
   Record hashes for raw tensors and result files before reading.
2. Parse the existing transformed [draw,4,31] retained tensors, checking them
   against the durable committed chunks where available. Use independent
   NumPy arithmetic only in this explicitly diagnostic script, and compare
   with the unchanged TensorFlow/XLA public estimator. No new target calls or
   sampler transitions occur.
3. Independently reconstruct original b=floor(sqrt(n)), r=3,c=.5 batch means
   and compare per-chain LRV, MCSE, mean and SD at relative 2e-9/absolute 1e-12
   roundoff tolerance. These tolerances are numerical hypotheses, checked
   against deterministic fixtures; original Boolean checks must match.
4. Explain sensitivity at floor(b/2), b and 2b only when the original
   minimum of 20 complete batches is satisfied. These scales are a symmetric
   diagnostic perturbation, not a selector. Preserve negative/unavailable
   variance and compare with plain batches and between-chain sample means.
   Report each method separately; do not take the most favorable answer.
5. Preserve normalized reference error, MCSE and independent-reference
   uncertainty separately, and report all quantities. Archive compact JSON
   and a result note, then refresh the next discriminating phase.

CPU-only exception: reading existing tensors and independent reduction checks,
`CUDA_VISIBLE_DEVICES=-1`, memory growth set before import, four intra-op/two
inter-op threads and single-thread BLAS. Public reduction uses XLA. This is
not GPU performance, posterior promotion or release evidence.

Reserve at most 900 seconds from the existing upper balance
160729.47980995933s for focused checks and the audit. This is a convenience
operational allowance for about 36 MB of retained tensors; record measured
inclusive execution once. A timeout preserves work and requires repricing,
not a statistical failure. Separate SIR funds are untouched. Output uses a
fresh `N5/c4-raw-precision-20261030-01` directory. No new random seeds.

## Skeptical pre-execution review

Pass for this limited diagnostic: every member is included; the exact C3
estimator is checked against the merged source; independent arithmetic cannot
quietly replace the historical decisions. The simple comparators are plain
batch means (isolates the lugsail extrapolation), between-chain means
(isolates within-chain batch choices), half-chain means (exposes time drift),
and the independently integrated reference (checks target agreement). These
are explanatory controls rather than optimized promotion criteria.

A clean replay would rule out the tested arithmetic/serialization defect. It
would not close the pre-existing antithetic, strong-dependence or unseen-tail
reliability gaps. Fresh fixed-count, independently seeded validation must be
specified before using an estimator or count adjustment to address them.

## Localized reader repair and retry

Attempt 01 stopped after 2.2844063700176775s because the new reader compared
active whitened checkpoint coordinates directly with transformed model
quantities. This invalidates that diagnostic comparison, not the original
sampling evidence. The repaired reader applies the original issued affine
factor and center, validates the execution hash against the original manifest,
and checks the center/covariance against the original hash-bound inputs. A
nonidentity affine regression checks orientation and prevents recurrence.
No covariance refactorization or sampler change is involved.

Retry uses fresh `N5/c4-raw-precision-20261030-02`, the compact extraction
`/tmp/modern-hmc-c3-frozen-geometry-20261030.json`, and the unchanged source,
draws and reference. Audit passes after this repair: the diagnostic now
compares the same mathematical quantities. Roundoff tolerance is needed for
independent CPU versus original GPU matrix multiplication; the historical
Boolean decisions must still match exactly. Both attempts and focused tests
consume the same 900-second reservation; none renews that reservation.
