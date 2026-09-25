# Frozen R reference: complete the small dimensions

The owner requests continuous execution through bounded repairs. This is
independent work that remains useful while exact original-paper numerical
replication is blocked by missing source specifications. It completes the
same-setting first-study dimension ladder for the already developed optional
R reference. It does not resolve equation-15 conformance.

## Research intent and evidence contract

Question: does the frozen `log_quadratic`, positive-floor-power-8 reference
pass the established likelihood and validity checks also at d5, d10 and d20?
Earlier d5/d10 results used a different fitter, and the larger d20 batches used
a different floor; those results cannot fill these cells. Existing d40/d80
results with the same numerical-core hash remain separate dataset evidence.

Primary comparator: exact Kalman likelihood on each generated dataset.
The dataset-level pass criterion remains the entire 95% percentile-bootstrap
interval for mean estimated/exact likelihood lying in [0.9,1.1], with all 32
repetitions retained. This is a bounded accuracy screen, not a proof of
unbiasedness or replication of the paper's published plots. An interval
excluding one is explanatory and triggers an uncertainty diagnosis, not an
automatic change of fitter or a declaration of bias.

Constructed heuristic adversaries: bootstrap PF (10000 particles: plain
resampling baseline), fully adapted PF (5000: one-step conditional Gaussian
proposal), and sequential importance sampling (10000: no-resampling stress
baseline). Exact Kalman evaluation is the certifying oracle. Evaluate ordinary
and large-innovation time prefixes separately using the existing recorded
classification and paired bootstrap squared-error differences. Statistically
supported inferiority to any heuristic in either situation vetoes the candidate.
Costs differ; no efficiency ranking is authorized. Lack of a significant
difference is not evidence of equivalence.

Hard validity vetoes: incomplete or nonfinite repetitions, a failed fit/status
check, deficient QR rank, invalid source/data identity, missing prefix rows,
or nonpositive Gaussian-limit second-moment margin. An invalid harness/result
is a continuation veto for dependent conclusions and triggers localized repair.
An accuracy, tail or heuristic failure rejects this candidate in the tested
scope and triggers diagnosis from saved guides; it does not reject iAPF as a
research direction. No claim data may select a new setting. Filter runtime,
iteration counts, variance ratios, floor fractions and condition numbers are
explanatory unless already identified above as vetoes.

## Frozen settings and assumption audit

Core SHA-256:
`c7152fea96d917bf90218bbbbc2bb31c01b0831b439625ff922454b01d748fd0`.
Keep T100, N0=1000, k=5, tau=kappa=.5, max_iterations=20,
max_particles=16000, fit_maxit=200, `first_full_window`, `log_quadratic`,
floor power 8. CPU R is the explicitly authorized independent-reference
exception; set CUDA_VISIBLE_DEVICES=-1 and BLAS/OMP threads to one.

| Choice | Provenance and reason | Risk and earliest diagnostic | Status |
|---|---|---|---|
| Model and five dimensions | Paper first study; same generator as completed datasets | Wrong model convention; retained settings plus exact Kalman tie-out | Reconstruction baseline; not original author data |
| Log-quadratic fit | Recorded coverage/conditioning repairs; exactly represents diagonal log-quadratic targets | Different objective; fit rank, finite coefficients, explicit method label | Optional reference hypothesis, not equation 15 |
| Positive floor power 8 | Prior Gaussian-tail calibration and non-harm evaluation | Finite tails may recur elsewhere; inspect every retained guide's margin | Frozen candidate, tested afresh here |
| Controller and resource caps | Existing checked R implementation; fresh final estimate | Undefined early paper convention; recorded convention and cap/status checks | Reconstructed baseline |
| 32 repeats and 10% interval screen | Earlier reference-validation contract | Rare weights and finite bootstrap coverage; second-moment margins and raw rows retained | Bounded qualification; not 1000-repeat replication |
| Fresh seeds below | Deterministic bookkeeping, unselected datasets | Easy-dataset dependence; report each dimension separately | Convenience design with explicit limitation |

## Budget, commands and repair mechanism

Available 1333.492512594 summed worker seconds: 479.519126226 remaining after
source reconciliation plus 853.973386368 unused positive-floor validation
seconds. These are transfers of recorded allowances, not new compute. Reserve
1200 for filtering (at most five launches, maximum two simultaneous workers)
and the remaining 133.492512594 for saved-guide diagnostics/reporting.
Charge every attempt, including failed launches. Never overwrite evidence.

| Attempt | Dimension | Data seed | Replication IDs | Timeout seconds |
|---|---:|---:|---|---:|
| attempt01-d5 | 5 | 84000005 | 1001..1032 | 300 |
| attempt02-d10 | 10 | 84000010 | 1001..1032 | 350 |
| attempt03-d20 | 20 | 84000020 | 1001..1032 | 400 |

Use `run_iapf_r_replication.py --campaign small_dimension_completion
--mode replication --fit-mode log_quadratic --floor-power 8 --repeats 32`
with each row's dimension, data seed, first=1001, timeout and output path.
Method seeds remain 53000000+dimension*100000+replication*10+method_id.
Output root: `docs/plans/artifacts/iapf-r-small-dimension-completion-20260921-01`.
Snapshot all source dependencies and the plan before execution. Reuse the
checked saved-guide diagnostics and validators; retain all input/output hashes.

Proceed directly from each result to its saved-guide diagnosis and the next
eligible row. Repair an infrastructure/serialization failure and retry the
same scientific settings in a new attempt directory within the shared cap.
If a candidate fails, inspect the saved evidence before deciding whether a
fresh-calibration repair is justified; do not silently retry scientific
failures or tune on these datasets. Reviewer availability is not a gate.

After these checks, refresh the unified five-dimension evidence and inspect
the R-to-TensorFlow comparison dependency. A paper-identity blocker applies to
exact replication claims; it does not automatically invalidate independent
reference diagnostics. Stop only at a genuine unresolved dependency, spent
campaign allowance or an external/irreversible boundary. Save the exact next
action in the checkpoint.

## Skeptical pre-execution audit

PASS. This fills genuinely missing same-setting cells, uses exact conditional
comparators, preserves the source gap, and does not promote a successful smoke
or a variance proxy. No scientific setting is being tuned. The main ways to
be misled are rare importance weights, conditional data luck, incomplete output
and conflating objectives. Per-guide margins, per-dataset uncertainty, complete
row/hash checks and explicit method identities address these risks without
claiming to eliminate them. Focused harness tests precede research execution.

Closeout must include run/report manifests, source hashes, decisions, inference
status, the strongest alternative explanation, remaining budget and the next
scientifically justified action. No original-paper, canonical LEDH, KDM,
gradient, HMC, production or default-readiness conclusion follows.
