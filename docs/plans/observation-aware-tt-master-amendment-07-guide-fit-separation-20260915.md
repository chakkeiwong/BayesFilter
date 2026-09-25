# H11 amendment: separate SGQF guide robustness from TT fitting

Date: 2026-09-15. Status: **REVIEWED AGREE; ADMITTED FOR EXECUTION**.

Independent bounded review: [proposal review](artifacts/observation-tt-h11-separation-20260915-01/review-proposal-01.txt),
which returned `VERDICT: AGREE`. The review is advisory evidence for this
trusted local campaign; the owner-authorized H11 budget and this recorded
protocol provide execution authority.

This amendment extends the completed A06 campaign under the owner-authorized
H11 continuation budget. It does not reopen A06, retune its observations, or
change the scientific target. The amendment is the governing protocol for
`run_observation_tt_h11_separation.py` and its versioned output root.

## Why this amendment is needed

A06 exposed two different failure questions. First, the signed SGQF guide
can produce a non-SPD covariance at a particular observation, which prevents
all guide-dependent methods from being evaluated. Second, when a guide exists,
the exact SGQF joint can lose information during Gaussian-to-polynomial
conversion, finite-degree/rank representation, fitting, or validation
selection. A single filtering average cannot identify these mechanisms. The
next diagnostic therefore records the guide outcome first and runs the fitting
diagnostic only on complete fresh guide paths.

The unchanged signed rule is the object under test. No clipping, covariance
ridge, level fallback, mass floor, or other numerical repair is admitted by
this amendment. Such a repair would be a separate Class C candidate requiring
its own calibration and review.

## Research intent and evidence contract

**Question.** On fresh observations from the frozen d=1 and d=4 SV model, is
the current SGQF guide valid for the whole horizon, and, conditional on a
valid guide, where does the TT approximation lose the exact SGQF joint?

**Comparator ladder.** The representation stage preserves the exact
analytic SGQF joint as the reference, and compares (i) the generic TT start,
(ii) the SGQF-converted/warm start, and (iii) the validation-selected warm or
generic fit. The fitted TT uses the frozen A06 degree/rank/sweep/L1 procedure;
there is no post-outcome capacity retuning.

**Primary diagnostic and vetoes.**

* Every fresh sequence must record all attempted SGQF levels (2, 3, 4, 5),
  scaled signed mass, covariance minimum eigenvalue and the exact rejection
  reason. Any guide failure is a hard veto on claiming guide robustness and on
  starting a downstream independent-filtering phase. It remains valid evidence
  for the guide-failure question.
* For each complete guide path, every representation target must have finite
  rows, weights, amplitudes, fit diagnostics and Hellinger scores. A failed
  fit is a candidate veto, not a reason to replace the target or skip the
  sequence.
* The validation-selected candidate is evaluated on the untouched audit panel
  produced after selection. An audit loss relative to the exact SGQF joint is
  a candidate promotion veto. Passing this screen does not establish a
  statistical ranking, filtering accuracy, posterior correctness, HMC
  readiness, or default readiness.

**Explanatory diagnostics.** Per-level negative-weight counts, mass and
eigenvalue margins; guide failure time; conversion compression; row effective
sample size and maximum importance weight; validation versus audit Hellinger
loss; selected family and L1; and elapsed time explain the mechanism. They do
not become tuning targets.

**Heuristic adversary set.** For this representation-only phase the constructed
cheap adversaries are the generic degree-3/rank-3 TT start (a no-information
baseline), the SGQF warm start (the available informed initialization), and
the exact SGQF joint (the tractable oracle for the tested target). The
transition and stationary-prior particle proposals remain the required cheap
adversaries for any later filtering phase; they are deliberately not claimed
to have been evaluated here. A relative TT comparison cannot promote a filter
without that later conditional heuristic table.

**Nonclaims.** This phase does not claim that SGQF is an exact posterior, that
TT fitting converges, that the selected TT is better than SGQF, that any method
is statistically superior, or that a candidate is ready for filtering,
production, HMC, or a changed default.

## Frozen scope and assumptions

* Model, parameters, dtype, chart convention, levels, target density and
  TensorFlow/XLA route are those used by A06 and are recorded in the manifest.
* Generate six stateless sequences for each dimension, T=20, using fresh data
  seeds `926000 + 100*d + sequence_index`; fitting seeds are
  `927000 + 10000*d + 100*sequence_index`. A06 seeds and observations are not
  read or reused.
* Use the A06 fitting controls: degree 3, rank 3, four sweeps, 128 proximal
  steps, L1 grid `{0, 1e-5, 1e-3}`, 1024 training rows, 4096 validation rows,
  and 8192 audit rows. The validation panel selects; the audit panel is
  final-only. These are frozen baseline controls, not promoted defaults.
* Use six fresh sequences because this phase is a bounded mechanism
  diagnostic, not a population estimate. A zero-failure result is only a
  continuation screen; it is not evidence that the guide is universally
  robust.

### Default and assumption audit

The SGQF levels and degree/rank controls are inherited from A06 for
comparability, not justified as universal defaults. Their failure modes are,
respectively, signed cancellation/non-SPD moments and finite-capacity or
optimizer error. The earliest diagnostics are the per-level covariance
margin/mass records and the exact validation/audit target scores. Fresh seed
streams are a convenience choice whose failure mode is accidental overlap;
the manifest records them and the script checks that their generated data are
new relative to the A06 seed formula. No cross-model transfer is used.

## Skeptical audit and pre-mortem

The exact SGQF joint is the relevant target, rather than a marginal Gaussian
or a product of marginals. The audit is untouched by selection and therefore
cannot certify generalization beyond these fresh panels. A successful command
could still mislead if the fresh sequences are unusually benign, if the
degree/rank cap hides a capacity problem, or if guide failure removes the
hardest cases; the result therefore reports failures and does not average
them away. A fit can fail because of an implementation or resource problem,
which is preserved as a distinct failure class. No proxy score is used as a
promotion criterion, and no stop rule is changed after seeing results.

## Execution, budget and stop conditions

The numerical launch has a 10,800-second wall ceiling and the amendment has a
14,400-second accounting ceiling including implementation checks, review,
launch and closeout. Every attempt charges the sole H11 ledger
`docs/plans/artifacts/observation-tt-continuation-24h-20260915-01/budget.json`;
there is no overlapping sub-ledger. Use a fresh attempt directory and never
overwrite prior output.

Run only after this amendment receives an independent bounded review and the
master/checkpoint record admission. Stop for source/data corruption, missing
required records, non-finite numerical values, or exhausted H11 budget. A
candidate-specific guide or fit veto does not erase valid remaining records;
finish the bounded diagnostic and record the veto. If every sequence and
audit screen passes, the only next action is to write a new reviewed plan for
conditional independent filtering; this amendment itself does not launch it.

## Exact command and artifacts

```text
CUDA_VISIBLE_DEVICES=GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a \
TF_FORCE_GPU_ALLOW_GROWTH=true TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 \
/home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/benchmarks/run_observation_tt_h11_separation.py \
  --output-root docs/benchmarks/artifacts/observation_tt_h11_separation_20260915/attempt-01 \
  --wall-budget-seconds 10800
```

The command is run from the repository root with the `tftwogpu` environment;
the UUID identifies the reviewed RTX 5080 device. A shell timeout may add at
most 100 seconds for orderly manifest closeout, but the script's 10,800-second
scientific wall ceiling is the budgeted stop condition.

The run manifest records Git state, source hashes, TensorFlow/GPU and memory
growth policy, seeds, controls, data, failures, and elapsed time. Per-sequence
data, guide records, fit/selection panels and the aggregate result are kept
under the unique output root. The terminal result must include a decision
table and inference-status table, explicitly distinguishing hard vetoes,
descriptive differences, unsupported rankings, and the next justified action.
