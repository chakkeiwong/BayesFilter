# Zhao--Cui original-code reference repair plan

**Date:** 2026-10-03  
**Working branch:** `sqmc-development`  
**Status:** completed; engineering repair passes, accuracy/score admission remains open

## Objective

Make the pinned Zhao--Cui author snapshot usable as a reproducible reference for the
BayesFilter nonlinear experiments without silently treating it as the current
production implementation, a same-data oracle, or an observed-data score oracle.

The immediate question is:

> Can the original Zhao--Cui route be executed on a bounded author-style
> reproduction, with a corrected observed-data likelihood statistic and an
> explicit report of whether it is aligned with the current saved-data target?

A successful bounded run will establish executable reference provenance and expose
the corrected evidence values. It will not establish LEDH correctness, HMC
readiness, production readiness, or a score implementation.

## Source and mathematical boundary

The source of record is:

- paper: `.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.pdf`
  and its local text extraction;
- pinned author snapshot:
  `third_party/audit/zhao_cui_tensor_ssm_p10/source`;
- upstream commit recorded in that snapshot's `MANIFEST.yml`:
  `80034dccb99eb1d86284a1839b4a12067d13b9da`.

The paper's Algorithm 1 (local text approximately lines 476--530) recursively
forms the unnormalised TT posterior
\[
q_t(x_t,\theta,x_{t-1}) = \widehat\pi(x_{t-1},\theta\mid y_{1:t-1}) f(x_t\mid x_{t-1},\theta)g(y_t\mid x_t,\theta),
\]
then reapproximates it and records a normalising constant. The author drivers
`eg3_sir/mainscript.m` and `eg4_predatorprey/mainscript.m` are the executable
reference routes for the SIR and predator--prey examples.

The current source snapshot is already an audit copy with documented Octave
compatibility patches. It remains immutable. Extracting code from MATLAB Live
Script `.mlx` containers into derived `.m` files is a representation
conversion. Any Octave-only patch or likelihood-statistic correction is a local
adaptation and must be labelled as such.

The source `models/full_sol.m` (lines 139--206) computes a joint
target/proposal log-weight in `smooth`. It comments out
`log(mean(exp(w)))`, silently removes nonfinite entries, and stores
`mean(w_temp)`. That quantity is the mean log weight, not the finite-sample
log marginal-likelihood estimator
\[
\log\widehat Z =
\operatorname{logsumexp}(w_1,\ldots,w_N)-\log N.
\]
The reference harness will preserve both quantities, fail closed on nonfinite
weights, and never overwrite the pinned source. Full evidence additionally
requires the proposal to cover target support; otherwise this estimates the
target integral over the proposal support. The finite-sample logarithm is biased.

The paper's SIR example (Section 6.3, local text approximately lines 2249--2410)
uses a fixed-parameter 18-dimensional state with nine observed infected
coordinates. The predator--prey example (Section 6.4, approximately lines
2411 onward) jointly learns six parameters with a two-dimensional state and its
own synthetic observations. These source targets are not assumed to be the same
as the current BayesFilter saved-data targets.

## Explicit implementation classifications

| Decision | Classification | Evidence and consequence |
|---|---|---|
| Parse `.mlx` code cells into derived `.m` files | `source_faithful` representation conversion | XML payload and cell order are preserved; input/output hashes are recorded. |
| Use the pinned snapshot's documented Octave compatibility copy | extension_or_invention compatibility adaptation (no HMC claim) | The author route is retained, but the execution environment is Octave rather than MATLAB. The patch manifest remains attached. |
| Replace `mean(w)` with stable `logmeanexp(w)` while retaining the old value | `extension_or_invention` diagnostic correction | This fixes an estimator mismatch; it is not an assertion that the author code itself implemented this correction. |
| Run fixed current observations through the author classes | only after an alignment proof | The source parameterisation, initial law, transition/observation timing, and data hash must match. Otherwise the report is `not_aligned` and no reference value is admitted. |
| Obtain a likelihood score from the original code | unavailable as source-faithful evidence | The source implements value propagation and smoothing, not an analytical observed-data score. A finite-difference score, if later added, is an explicitly labelled extension. |

## Evidence contract

**Scientific question.** Does the original Zhao--Cui route execute on its
documented SIR and predator--prey examples, and what is the corrected
observed-data evidence relative to the legacy mean-log-weight statistic?

**Comparator.** The pinned author route with the same generated author-style
observations, using the derived corrected `full_sol_reference` class. The
current TensorFlow target is a separate implementation and is compared only
after the alignment report passes.

**Primary pass criteria.**

1. Every extracted callback has a nonempty code payload, a source SHA-256, and a
   derived SHA-256.
2. The bounded Octave route completes with finite target/proposal values and
   finite corrected `logmeanexp`.
3. The report contains the legacy mean-log-weight value, corrected value,
   finite-weight fraction, and importance ESS, with the corrected value
   identified as the corrected importance estimate. The TT normalizer is
   reported separately; mean log weight is not called log likelihood.
4. A machine-readable alignment report explicitly states `aligned_at_tested_point`,
   `not_aligned`, or `not_checked` for each current saved-data target.

**Hard vetoes.**

- extraction hash or function-definition failure;
- missing source commit or compatibility-patch provenance;
- nonfinite corrected value, invalid normalised weights, or silent deletion of
  nonfinite entries;
- source/current target mismatch hidden by a caller-supplied label;
- output written outside a fresh versioned result directory;
- any claim that the source run supplies an analytical score.

**Explanatory diagnostics.** Runtime, TT rank, approximation residuals, legacy
mean-log-weight, finite-weight fraction, ESS, and the difference between legacy
and corrected values explain behaviour but do not promote a method.

**Nonclaims.** The run does not prove source-code correctness in MATLAB,
paper-scale accuracy, posterior correctness, score correctness, LEDH
superiority, HMC convergence, or default readiness. A successful author
reproduction is not a same-data comparison unless the alignment gate passes.

**Preserved artifacts.** A fresh directory under
`docs/plans/artifacts/zhao-cui-reference-repair-20261003/<run-id>/` will
contain the command/environment manifest, extraction manifest, derived source,
Octave stdout/stderr, raw `.mat` results, scalar summaries, alignment report,
literature ledger, and result note. Existing results are never overwritten.

## Work packages and bounded budget

### WP0 -- Source and literature audit (read-only)

Record paper section/equation anchors, author driver and `full_sol.m` line
anchors, the pinned commit, the compatibility patch, and the source/data
mismatches. The audit ledger records the seed paper, local source snapshot,
technical claims, support status, and unavailable forward metadata. No claim
uses a title or abstract alone.

### WP1 -- Derived-source extractor

Implement a standard-library utility that reads the XML inside `.mlx` files,
extracts code paragraphs in order, writes derived `.m` files below the fresh
artifact directory, and records input/output hashes plus the extractor version.
It must refuse empty or non-code files and must never mutate
`third_party/audit/.../source`.

### WP2 -- Corrected reference class

Generate a derived `full_sol_reference.m` from the pinned `full_sol.m`.
The derived class retains the original route and records:

- raw log weights;
- the legacy finite-entry mean log weight;
- stable `logmeanexp`;
- finite-weight fraction;
- normalised-weight ESS;
- a status flag that is invalid when any raw weight is nonfinite.

The original `full_sol.m` remains unchanged. The derived class is not imported
by `bayesfilter` runtime code.

### WP3 -- Bounded author-route execution

Run a small, fixed-seed source reproduction for both author examples, using
short horizon and low TT rank to test the call chain and the corrected statistic.
Use separate per-case timeouts within a total 15-minute CPU budget. Preserve a
failure log and classify a failure as extraction, Octave compatibility,
numerical, or source-route failure; do not turn it into evidence against the
method.

### WP4 -- Current-data alignment gate

Inspect the current saved-data manifests and compare target identity, dimensions,
state ordering, parameter chart, initial distribution, time step/integrator,
process and observation covariances, and observation SHA-256. The SIR source
driver fixes parameters (`d=0`); the current three-log-scale target can agree
at zero scales because its consumer retains the source half-step convention.
Verify that point numerically rather than inferring a mismatch from dimension.
PP requires an explicitly labeled fixed-target extension for conditioning,
parameter chart and the fourth RK stage. Any uncorrected mismatch writes
`not_aligned`. The adapter review below records the exact bounded comparison.

### WP5 -- Tests and result note

Add focused tests for `logmeanexp`, nonfinite fail-closed behaviour, XML
extraction, hash recording, and alignment rejection. Write a result note with a
decision table and stochastic-evidence status table. Commit the plan, harness,
tests, and completed artifacts on `sqmc-development`; do not merge or push
this new work unless separately requested.

## Skeptical plan audit before execution

| Risk checked | Audit finding | Mitigation |
|---|---|---|
| Wrong baseline/statistic | `mean(w)` is not `logmeanexp(w)`; comparing them as two likelihoods would be wrong | Report both, identify corrected `logmeanexp` as the importance estimate, and retain raw weights and the separate TT normalizer. |
| Proxy promoted to criterion | ESS, rank, residual, and runtime can look favourable without validating the target | Keep them explanatory; finite corrected value and provenance are the bounded pass criteria. |
| Hidden target mismatch | Source PP/SIR data generation and parameter charts differ from current saved targets | Generate author-style data separately and require a machine-readable alignment proof. |
| Nonfinite handling | Silently deleting nonfinite weights can make a failed run appear valid | Preserve all raw weights and mark the result invalid on any nonfinite value. |
| Source representation | `.mlx` is OOXML, not executable `.m` | Extract code with hashes and test the derived call chain. |
| Environment mismatch | Octave compatibility patches are not MATLAB equivalence | Record Octave version and patch manifest; call the run an Octave audit reproduction. |
| Score overclaim | Original source has no analytical observed-data score | State score status as unavailable; any finite-difference extension remains separate. |
| Artifact ambiguity | A successful smoke could be mistaken for paper-scale evidence | Use fresh versioned artifacts, explicit run scope, and nonclaims. |
| Resource overrun | TT source execution can be expensive | Short horizon, low rank, per-case timeout, 15-minute total budget, stop on repeated infrastructure failure. |

This audit passes: the plan has a correct comparator, explicit vetoes, target
alignment checks, finite compute bounds, and an artifact that answers the stated
question. Execution may begin with WP1 and WP2; WP3--WP5 use the same contract and
are stopped if a veto fires.

## Planned review and execution record

Before running the harness, append the exact review date, branch, current
commit, and audit result to this section. After execution, append the actual
commands, statuses, artifact paths, failure classification, decision table,
strongest alternative explanation, overturning evidence, and next justified
action.


### Pre-execution review (2026-10-03)

- **Branch and commit reviewed:** `sqmc-development` at
  `a925f67a19d8c5d265b8ff71bfd2c78ac1964ff4`; the working tree was clean.
- **Scope check:** WP1--WP5 are bounded to a diagnostic/reference lane. They do
  not modify the pinned source, production `bayesfilter` code, current
  likelihood defaults, or claim-bearing LEDH routes.
- **Evidence check:** the primary statistic is the corrected stable
  `logmeanexp`; legacy mean log weight, ESS, residuals, and runtime are
  explanatory. The plan carries explicit hard vetoes and nonclaims.
- **Alignment check:** source-generated author data and current saved data are
  kept as separate targets. The initial concern about SIR parameter dimension
  is superseded by the source-audit correction and executable pointwise checks
  below; dimension alone does not establish different fixed-parameter laws.
- **Budget/stop check:** extraction and unit checks precede Octave execution;
  each source case is timeout-bounded inside a 15-minute total CPU budget, and
  nonfinite or provenance failures stop promotion while preserving diagnostics.
- **Review verdict:** PASS. The plan answers the stated reference-repair
  question and is safe to execute under the repository's source-anchor and
  evidence rules.

### Source-audit correction and review amendment

Inspection of all 19 extracted callbacks found material differences omitted
from the first draft. The draft's Algorithm 1 expression has been corrected to
the paper's two-slice joint density (equation 12), rather than a full-history
recursion. The original predator--prey driver integrates a parameter prior;
it does not compute the current fixed-parameter likelihood. Its ODE uses
physical K=90+20*theta(5) and a=20+10*theta(6), with source truth
(r,K,a,s,u,v)=(0.6,100,25,1.2,0.5,0.3). Its RK fourth stage evaluates at a
half-step (predator_step.m extracted lines 35--40), while the current target
uses standard RK4. Those are actual target differences.

The original SIR fixed point can agree with the current three-log-scale model
at theta=(0,0,0). A parameter-dimension difference alone is therefore NOT a
value-alignment veto. The source-generated and saved observations still differ.
An injected-data source run may be a value comparator after executable
transition, initial-density, and observation-density parity; it supplies no
analytical score by itself.

The implementation will first execute the immutable-source mechanics check,
then implement a separately labelled fixed-target adapter where needed. This
adapter preserves the author TT solver call chain but is an
extension_or_invention for the current model target, not a faithful reproduction
of the author's predator--prey example. It will use the saved observations and
preserve the original reference output separately. The budget remains 900 CPU
wall seconds for Octave experiments, including failed attempts; syntax and
focused regression checks are routine development checks.

Review verdict: the first draft was incomplete on target alignment. The above
correction resolves that issue before source execution. No source result will
be exported as a converged oracle, and a successful short smoke only establishes
wiring and reproducibility.

### Fixed-target adapter review before execution

The actual `NonlinearSQMCSpec.model` call chain reaches
`austria_sir_canonical_model`, whose fourth RK stage already preserves the source
half-step. SIR value alignment is therefore plausible at zero log scales and
will be checked numerically. The PP adapter will fix the six physical parameters,
map them to the source ODE chart, and change only the fourth RK stage to the
current full-step convention. Both use the exact committed T=20 observations,
with their saved tensor hash verified before injection. The source classes still
perform TT fitting and conditional smoothing. The fixed-target adapter is an
`extension_or_invention`; the author reproduction remains separate.

Before solving, five state probes must agree between Octave and the actual
TensorFlow consumer for transition means, initial log densities, transition log
densities, and observation log densities (absolute error <= 1e-8). Stable Gaussian
logs must also agree with the source PDFs wherever those PDFs remain positive.
These checks test model and call-chain wiring, not TT convergence. CPU eager
TensorFlow is an explicit small independent-reference exception for these probes.

The author PP driver has bounded proposal support. For a proposal q, the mean
importance weight integrates the target over the support of q; it equals the full
evidence only if the missing target mass is zero. Correct logmeanexp does not
repair missing support. The fixed-target PP adapter therefore uses the author's
available `AlgebraicMapping(1)` basis-domain option, as the SIR driver does, so
its proposal can cover unbounded Gaussian states. This is a recorded numerical
choice, not a promoted default; rank, sampling error, and tail coverage remain
unchecked. Rank=4, one ALS pass, N=64 are intentionally cheap mechanics settings,
not tuning. ESS and discrepancy from the TT normalizer expose poor accuracy but
do not invalidate a model-wiring check.

Two T=20 fixed-target cases are bounded to 180 seconds each, with at most one
localized repair retry per case inside the original 900-second Octave budget.
No statistical ranking or oracle admission is attempted. Failure to align stops
the corresponding solve; nonfinite output stops numerical admission and is
preserved as a repair diagnostic. Review passes for this bounded reference
extension; accuracy certification and analytical score construction are separate
work beyond this repair.


### Completed execution and terminal review

WP0--WP5 completed. Final author-route mechanics runs are `author-reference-02`;
fixed-target T=20 runs are `fixed-target-02`, both under
`docs/plans/artifacts/zhao-cui-reference-repair-20261003/`. Their run manifests
preserve the actual commands, source/implementation hashes, environment and
seeds. The source tree is unchanged. The attempt ledger records failures and
repairs: 89.716 of 900 allowed Octave wall seconds were consumed.

The corrected fixed-target values are -97.335236612 for PP (ESS 51.379/64) and
-4253.054415329 for SIR (ESS 1/64). The saved independent bootstrap values are
-97.342973851 and -678.077466314. Data hashes and horizons match. Parameter
points are nominally the same; the saved PP float32 parameters differ from the
float64 probes by at most 2.384186e-8. This precision boundary is retained in
`comparison-parameter-audit.json`; no exact numerical-target identity or
statistical agreement is asserted.

Five-probe transition/log-density checks pass (max error 0 for PP and
2.328306e-10 for SIR, threshold 1e-8). This verifies the tested call chain and
model point, not global equivalence. Original SIR sampling clips susceptible
states while its density is Gaussian; the preserved source sampling convention
affects fitting, while the fixed-target comparison evaluates the Gaussian
likelihood. Original bounded PP proposal support remains a limitation of the
author smoke; the fixed-target extension uses AlgebraicMapping(1).

Terminal skeptical review: PASS for execution, provenance and statistic repair;
no value/score-oracle admission. SIR weight collapse rejects this small
configuration as an accuracy reference, not the method or harness. PP closeness
is descriptive only. Scope-specific TT/sample calibration, independent
replications and uncertainty, and a separate analytical-score construction
remain research work. The bounded mechanics plan is complete.

See `docs/benchmarks/zhao-cui-reference-repair-results-20261003.md` for the
actual numerical table, source limitations, decision and inference-status
tables, terminal red-team assessment and reproduction instructions.

Final focused validation, 2026-10-04: 11 regression tests pass; Python compilation,
extraction-only/source-immutability smoke and structured preflight-rejection
check pass. Logs are under the evidence root in `validation/`.
