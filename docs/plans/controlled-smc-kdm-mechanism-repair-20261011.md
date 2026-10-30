# Replication mechanism investigation and repair — 11 October 2026

## Question and scope

Does the remaining discrepancy come from a translation defect, ordinary
cross-backend floating-point sensitivity, or an uninformative comparison?
The target remains the pinned author's controlled-SMC and Bearings MDPF/A-MDPF
behavior. Author failures are accepted baseline outcomes. Original iAPF,
algorithm improvement, Task 3, HMC and new paper-scale claims are out of scope.

The 1% likelihood and 5% variance cSMC margins were wrong planning and remain
withdrawn. Source dispersion is a precision diagnostic, not an equivalence
margin. A nonsignificant difference cannot establish no-worse performance.

## Checked call chains and hypotheses

- cSMC: Lorenz refinement -> multivariate regression -> Gram/RHS and solver.
  The 10 October exact-input experiment already matches 101 Gram/RHS arrays.
  Ill-conditioned normal equations and Octave's conditional solver explain
  coefficient sensitivity. Q2 contains zero added TF failures in 13,120 pairs.
  Those observations support implementation/failure findings, not an identified
  stochastic equivalence margin. Do not repair intrinsic author instability.
- KDM: source full-sequence trainer -> proposal encoder -> flattened dense MLP
  -> sigmoid residual -> normalized sine/cosine angle -> KDE objective -> BPTT
  -> Adam. The runtime uses rank-three dense operations where the author
  explicitly flattens particle and batch axes. This is algebraically equivalent
  but may change rounding and parameter-gradient reductions. It is a hypothesis,
  not yet a demonstrated root cause. The already repaired Adam scalar precision
  cannot explain update-zero raw gradients. Angle conditioning can amplify tiny
  forward errors; its isolated derivative passed the prior checks.

## Execution phases and evidence contract

1. **Source-only cSMC feasibility.** Inspect only source fields of existing Q2
   evidence (exploratory, not fresh calibration): failures, source logZ variance,
   effective residual count and tail diagnostics where available. Quantify what
   is estimable without reinstating either fixed margin. If the record cannot
   justify an independent degradation bound, leave stochastic equivalence open
   and specify the missing decision; do not launch another blind full grid.
2. **KDM mechanism isolation.** Reuse the exact saved source inputs, pre-update
   parameter states and gradients at all four updates of both variants and both
   precisions. Compare the actual runtime with a diagnostic subclass that uses
   the author's flattened dense layout. Add source layer VJPs with common
   activations and incoming adjoints if the intervention does not explain the
   discrepancy. Evaluate FP64 as a reference, never silently promote it to the
   FP32 runtime. The existing 128-epsilon gradient screen is explanatory only.
   A causal claim requires a controlled intervention that removes the error,
   or measured decomposition of its upstream and local contributions.
3. **Localized repair and regression.** Promote a layout change only if source
   anchors and full-consumer checks support it without additional failures.
   Preserve diagnostic-only interventions when the evidence is inconclusive.
   Check the real full-sequence endpoint, stable tracing, GPU/XLA, dtype and
   parameter coverage. Do not demand bitwise cross-backend gradients as a new
   replication criterion.
4. **Training comparison readiness.** Audit the proposed 64-update smoke before
   a complete training launch. Each method must sample from its own current
   distribution. Shared realized source particles are valid for localization
   only. If the current shared-randomness interface cannot preserve the source
   sampling law, repair that interface before claiming a paired training smoke.
   Record the exact missing interface and next bounded implementation when the
   mechanism phase exposes more work than its budget permits. A long training
   launch needs its concrete data, RNG, stage schedule and cost specification.

Promotion criteria: correct source wiring, finite source-valid cases, no newly
introduced mismatches in the full consumer, and improvement attributable to the
tested mechanism. These are engineering repair criteria. KDM downstream NLL
0.01 nat and RMSE 1% criteria are unchanged and require separate training data.
No scientific superiority or replication-completion claim follows from this
diagnostic. The exact author is the comparator; heuristic superiority tests
are inapplicable to the user's restricted code-replication question.

Continuation vetoes: invalid input/hash, wrong baseline, corrupted output,
source-invalid input interpreted as source-valid, or exhausted phase/campaign
budget. A negative mechanism hypothesis triggers the next localization phase,
not abandonment of KDM. A confirmed defect triggers repair and focused rerun.

## Assumption audit and skeptical review

The saved trained checkpoints are diagnostic hypotheses about difficult input
states, not fresh qualification data. Exact pre-update resets remove accumulated
Adam drift but do not model autonomous training. Flattening is grounded in
`proposal_models.py:187-193`; its failure mode is a different GPU reduction
without meaningful downstream benefit. Full-gradient comparisons at common
states are its early check. The existing gradient tolerance only locates
discrepancies. Wide author tails invalidate precision assumptions, not the
author algorithm's admissibility as a replication baseline.

Pre-run skeptical audit: PASS for these bounded diagnostics. The earlier plan
mistakenly risked treating strict gradients and old percentage limits as closure
conditions. This plan keeps them diagnostic, separates fresh qualification from
reused evidence, and forbids a training claim from replayed source particles.
No unreviewed damping, clipping, precision change or altered source optimizer
is authorized by the diagnostic result.

## Budget, commands and artifacts

Existing balance: 220860.173800 CPU / 117789.917545 GPU wall seconds. This phase
has an aggregate 300 CPU / 900 GPU second cap, maximum five versioned attempts.
The original three-attempt count was extended for two localized infrastructure
retries without enlarging either compute cap: attempt 01 had an overbroad
parameter-coverage assertion; attempt 03 inherited the system C++ runtime
instead of the pinned source environment. The CPU subprocess now uses the
established source library path. Neither failure is evidence against the method. Attempt 04 also exposed JSON
configuration keys losing their integer type. The reference now loads verified
author YAML, as the existing working source harness does, and checks its JSON
representation against the saved configuration before executing.
No new compute grant. Source-only analysis uses the standard library; source
VJP export uses pinned PyTorch in isolated CPU reference mode. Runtime checks
use the tftwogpu environment, GPU UUID recorded by the supervisor, verified
memory growth, FP32 with TF32 off, and XLA; FP64 is reference only.

Execute through the existing approved prefix:

```
/home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/benchmarks/run_replication_gap_localization.py --phase mechanism --output /home/chakwong/BayesFilter/docs/plans/artifacts/controlled-smc-kdm-replication-20260928/mechanism-20261011/mechanism-01
```

The supervisor enforces timeouts, preserves logs, snapshots changed code and
inputs, writes a manifest and charges the existing ledger. Results belong in
`controlled-smc-kdm-mechanism-result-20261011.md`. After this bounded phase,
refresh the checkpoint and master with actual findings and remaining work.

## Native-training smoke: reviewed continuation after mechanism results

The layout intervention failed to remove any mismatch. All 3,840 common-input
local derivative checks passed. No runtime change is justified by those tests.
The next question is whether the repaired runtime can execute 64 autonomous
updates of each MDPF/A-MDPF variant without adding invalid outcomes. This is a
development smoke from the existing difficult trained checkpoint, not fresh
complete training or qualification.

Use identical starting parameters, zero Adam moments, the same 4,096 training
sequences in a fixed seeded permutation, B=64, N=25, T=17, source sparse labels,
truncation modulo 4, source learning rates and clipping 100. Each implementation
uses its own native initialization, resampling and proposal-noise generators.
The streams are independent across Torch and TensorFlow; no realized source
particles are injected into TF. This replaces the earlier unnecessarily strict
common-random-input smoke requirement. Paired data and starting states do not
require identical random streams. Source-native sampling and TF-native sampling
must remain visible in code and recorded in the result.

Keep four development validation batches (256 sequences) for descriptive NLL
before/after; they are not untouched qualification data. Freeze all otherwise
frozen parameters and check every gradient, parameter and optimizer state for
finiteness. Require all 64 updates to commit and one trace per stable TF
configuration. Source failure triggers classification, not a demand to repair
the author. A missing gradient, invalid source-valid TF computation, incorrect
sample law, changed input/source hash, or budget exhaustion stops the smoke.
Training/validation loss differences are explanatory only; they cannot establish
NLL/RMSE no-worse performance, convergence or a preferred implementation.

Assumptions: the existing hard checkpoint is a stress fixture, not an unbiased
training start; native independent streams preserve each model's sampling law
but sacrifice common-random-number precision; 64 updates test accumulated
optimizer execution, not full convergence; fixed batch ordering is paired
development input rather than an author's complete epoch. Source Adam and the
current TF binary64 scalar correction remain unchanged. No new numerical
protection or backend is selected.

Skeptical review: PASS for the stated smoke. Wrong-baseline, replay-law,
stale-verification and gradient-proxy risks are addressed by using pinned source
code, actual native consumer endpoints, recorded current runtime hashes and
finite/commit checks. Old verification hashes cannot certify changed runtime;
the new smoke records current call-chain execution, but does not replace the
separate three-stage transfer/controller review required before complete
training. No heuristic superiority claim is made.

Additional bounded allocation from the existing campaign: 600 CPU / 900 GPU
wall seconds, at most three versioned smoke attempts. Local infrastructure
retries stay within that aggregate cap. Output root is
`artifacts/controlled-smc-kdm-replication-20260928/native-training-20261011/`.
Exact launch:

```
/home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/benchmarks/run_replication_gap_localization.py --phase native-smoke --output /home/chakwong/BayesFilter/docs/plans/artifacts/controlled-smc-kdm-replication-20260928/native-training-20261011/native-smoke-01
```

After the smoke, preserve both checkpoints and losses. Review current full-stage
verification/transfer/controller wiring before specifying a fresh paired
complete-training pilot. Do not restart the 41-cell cSMC margin campaign or
make strict gradient matching a prerequisite for that pilot.


## Refresh current training evidence and exercise all three stages

The native smoke passed 64/64 updates for both variants in both backends.
A separate harness defect remains: the complete TF driver hardcodes the
7 October verification, whose runtime hash predates the Adam repair. The
staleness rejection is correct; bypassing it would be wrong. Make verification
an explicit CLI input, preserve its checksum and current-runtime checks, and
produce a new source-stage verification on the current code.

Evidence contract: compare initialization, weight pretraining, native/replayed
full-sequence wiring, optimizer state, and both stage transfers against the
preserved pinned-source stage fixture. Reuse its data and source outputs only
as development mechanics fixtures. Require all existing stage checks, atomic
incomplete-transfer rejection, fresh optimizer state at transfers, and bounded
one-trace graph signatures to pass. Then run two complete epochs per stage
(weight pretraining, MDPF, A-MDPF) from the existing fresh constructor/data
fixture, using the current verification and each TF model's native sampling.
Require every row once per epoch, finite committed updates, immutable validation,
source controller decisions, best-checkpoint transfers, and all three completed
stages. Neither these mechanics checks nor development losses certify final
no-worse accuracy. A source fixture mismatch is a localization/repair trigger;
invalid data, unresolved sample-law changes, or exhausted budget stops execution.

Assumptions/default audit: the old source fixture remains a valid deterministic
reference because the source pin and input hashes are unchanged; verify those
hashes. Two epochs exercise the full data and stage lifecycle but cannot
establish convergence. Training uses the source batch widths, partial batches,
N=25, sparse-label loss and truncation, learning rates, clipping, and controller
settings already traced to author code. TF and Torch native streams remain
independent. The source controller oracle and atomic-transfer checks answer
mechanics only; no heuristic-dominance or superiority claim is made.

Skeptical review: PASS for this bounded readiness phase. It corrects stale
verification without removing the guard, tests the actual full-training consumer
and transfer chain, and separates mechanics from stochastic acceptance. The
historical six-attempt/old-date training wrapper is not a new compute grant.
Do not launch through it under an exhausted old phase budget.

Allocate at most 900 additional GPU wall seconds from the existing campaign,
maximum three versioned attempts, for verification plus the two-epoch smoke.
No new source training is needed. Preserve logs, current code snapshots,
reference hashes, proof, batch/epoch records and resource charges. Execute:

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/benchmarks/run_replication_gap_localization.py --phase stage-refresh --output /home/chakwong/BayesFilter/docs/plans/artifacts/controlled-smc-kdm-replication-20260928/stage-refresh-20261011/stage-refresh-01

After a pass, specify a complete-training development pilot using matching fresh
source constructor/data inputs and the current repaired TF optimizer. Preserve
source intrinsic failures; compare final heldout NLL/RMSE with uncertainty
before deciding qualification. Do not elevate strict gradient tolerance into
acceptance or demand a repair to the author's filter.
