# Stan warm-up transplant and native comparison

Owner request, October 27, 2026: plan, thoroughly review, implement Stan-style
NUTS tuning, write its guide, install Stan and execute a comparison. This
extends the standalone benchmark exception only; public retired NUTS routes
and other running studies remain unchanged.

## Research intent and evidence contract

Question: can a TensorFlow/XLA implementation reproduce Stan's warm-up
recurrences and yield valid samples, and what does its complete warm-up plus
sampling cost relative to native CmdStan on matched targets? Comparator:
CmdStan 2.37.0, pinned release (a reproducibility choice, not a latest-version
claim). Candidate: stock TFP NUTS with source-checked Stan window scheduling,
per-chain covariance and dual averaging, reasonable-epsilon initialization,
metric-boundary restart and frozen post-warm-up state. TFP's tree algorithm
and RNG are not transplanted; identical random paths are not expected.

Engineering pass: deterministic recurrence agreement with actual Stan C++
headers, metric-coordinate derivation and numerical checks, no cross-chain
adaptation pooling, reproducibility and finite cache, enclosing XLA compilation
with stable signatures, native executable and saved source/version evidence.
These checks cannot establish posterior correctness alone.

Sampling pass: independent multi-chain assessments in original and analytic
reference coordinates; maximum rank/folded split R-hat <=1.01; minimum
bulk/tail ESS >=400; standardized MCSE <=0.05 for reference means and second
moments; agreement with known moments within max(4 MCSE,0.05 reference SD).
The cutoffs are conservative convenience screens for this finite benchmark,
not proofs of convergence or calibrated simultaneous confidence coverage.
Report all failures and nonfinite diagnostics. Precision is checked only at
the fixed terminal budget: this is a tuned fixed-budget comparison, not a
claim of fastest sequential time to precision.

Primary measured endpoint: synchronized warm-up plus retained-sampling time,
including all initial and metric-boundary step searches. Separately report
first-call/compile, process startup, diagnostics and output costs. A method
failing posterior screens cannot receive a valid efficiency ranking. Across
three complete independent repetitions, report paired ratios and a 95%
Student-t interval of log ratios (df=2); this is low-power and model-dependent,
not publication-grade superiority evidence. ESS/sec, acceptance, leapfrogs,
tree saturation, divergence counts and metric/epsilon variability explain
outcomes. Native energy/divergence flags are reporting-only under repo policy.

Continuation vetoes: incorrect target/gradient or metric orientation, invalid
retained state/cache, failed recurrence check, non-XLA numerical fallback,
invalid GPU allocation policy, corrupt artifacts or exhausted budget. Poor
mixing rejects that configured candidate, not NUTS or its tuning direction.
Localized infrastructure failures trigger repair/retry within budget.

## Implementation and comparison stages

1. Install an isolated CmdStan release under `.localresources/cmdstan/`; do not
   mutate the shared conda environment. Archive download/version hashes and
   build logs. Read the release's adaptation and service code, not only the
   earlier moving-branch audit.
2. Implement a standalone TF/TFP warm-up module. Represent the learned inverse
   metric C=A A^T with q=A z and unit-mass NUTS in z. This implies physical
   momentum p_q=A^-T p_z and kinetic energy p_q^T C p_q/2. Recompute transformed
   state and gradient cache at metric boundaries; keep physical q fixed.
   Support diagonal and dense window estimates. No learned transform is used.
3. Verify Stan schedule, Welford regularization, dual averaging and restart
   arithmetic against compiled actual Stan headers. Test step initialization,
   metric boundary continuity, frozen handoff, complete XLA graphs and chain
   independence. Write a user guide identifying exact matches and deviations.
4. Compile one native Stan model with data-selected synthetic targets. Check
   density/gradient agreement at common points. Use identical start banks and
   data for both engines. Native Stan is explicitly a non-XLA external reference.
5. Calibrate CPU/XLA versus GPU/XLA and scalar versus batched execution on a
   discarded cheap-target case. Select a practical measured layout before
   evaluation. Record peak device allocator memory and host high-water RSS.
   CPU results do not change the repository GPU default.
6. Run Gaussian32, rotated Gaussian32 (condition 1000), product banana8,
   centered funnel10, noncentered funnel10, and Gaussian regression32 using
   the existing target definitions. Dense metric for rotated Gaussian; diagonal
   for the others. Four chains, 1000 warm-up, 2000 retained, seeds
   20261027/20261028/20261029; depth10, delta0.8, epsilon initial1, jitter0,
   buffers75/25/50, DA gamma0.05/kappa0.75/t0=10. Preserve all attempts.
7. Run common TF posterior diagnostics on both engines' samples; write the
   full table, limitations, inference-status and decision tables, exact
   manifests and next justified action. Review the results against the question.

## Budget and defaults audit

New campaign, independent of the closed previous pilot. Up to 45 elapsed
minutes for download/build and 45 elapsed minutes for tests/benchmarks after
installation, at most four CPU workers and one GPU worker at a time; no paid
compute. At most 12 worker launches including repairs/calibration, each with
a timeout within the remaining budget. These are convenience resource caps;
record consumed time and do not extend them silently. Prefer bounded smoke
checks before the full set. A worker may process several target/seed cells.
Unique evidence root: `docs/benchmarks/artifacts/stan-transplant-20261027/`;
fresh numbered attempt directories, no overwrites. Source snapshots/hash list,
git commit plus dirty-state indication, command, seeds, environment, placement,
XLA/HLO, allocator, wall time, plan and result paths go in each manifest.

| Choice | Provenance/status | Risk and early check |
| --- | --- | --- |
| Stan2.37.0 / installed TF2.20 TFP0.25 | Pinned comparator / installed baseline | Version differences; inspect installed source and metadata |
| Adaptation constants and metric shrinkage | Stan release defaults | Default is not target-specific adequacy; inspect terminal metric and diagnostics |
| 1000/2000 draws, four chains, three repetitions | Bounded convenience allocation | Funnel may fail; report failure, no rescue-by-cherry-picking |
| Six targets and dense rotated case | Existing independent analytic controls | Easy targets may favor CPU; include matrix workload and device calibration |
| Initial starts from known reference law | Matched diagnostic design | Favorable starts may hide initialization problems; disclose and do not claim cold-start robustness |
| Affine implementation of metric | Derived canonical coordinate change | Transpose/inverse/cache mistakes; analytic and Stan density checks |
| TFP tree + Stan adaptation | Explicit implementation hypothesis | Transition and RNG differ; no bitwise Stan sampler equivalence claim |
| CPU workers4 / GPU1 and time caps | Local resource hypothesis | Contention; record topology/load and separate calibration |

The current deliverable isolates NUTS adaptation against Stan. The prior
fixed-HMC audit remains applicable: a fair HMC comparison must call the public
candidate-set tuner, preserve members and charge independent selection. This
campaign does not relabel the old finite-grid HMC timings as guide-compliant.

## Skeptical review before execution

**Baseline:** comparing stock identity-mass NUTS with native adapted Stan would
confound metric learning and execution. Both new arms use the same metric
family, schedule and stated tuning parameters. Initial/boundary DA order must
come from the release code.

**Scientific scope:** fixed-budget timing cannot establish time-to-precision
optimality. The primary table will display validity screens beside timing;
failed cells stay visible, and short-run ESS is never a tuning selector.

**Hidden defaults:** Stan service initialization, diagonal/dense metric meaning,
jitter, depth, CSV precision, thread counts and chain-local adaptation are
explicit. Native Stan must use 17-digit output for common diagnostics. RNG
differences preclude pathwise equivalence but not recurrence and density checks.

**Compilation:** all numerical initialization/adaptation/transition loops are
inside stable-signature XLA functions, including metric-boundary conditionals.
Host file handling and orchestration are excluded and their costs disclosed.
No fallback to eager kernels, pfor, NumPy runtime or another autodiff backend.

**Timing fairness:** Stan's printed warm-up time omits some initialization;
process wall time is therefore separately primary for complete native calls.
TF first execution includes compilation and discarded work; report this
explicitly, plus subsequent full-call time. Do not subtract a fabricated
compile estimate. Shared targets and diagnostic code are identical.

**Pre-mortem:** a successful short Gaussian run can hide wrong dense orientation
or funnel underexploration. Use actual-header arithmetic tests, rotated target
and analytic reference moments before trusting aggregate speed. A GPU compile
failure is compatibility evidence, not permission for an eager benchmark.

Review verdict: proceed with this bounded engineering/diagnostic comparison.
No current route is promoted to production, no Stan transition identity is
claimed, and no statistically unsupported sampler winner will be declared.
