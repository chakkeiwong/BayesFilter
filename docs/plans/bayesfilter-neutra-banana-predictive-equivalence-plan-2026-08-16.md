# NeuTra Banana Predictive-Equivalence Diagnostic Plan (2026-08-16)

## Research intent ledger

| Field | Predeclared statement |
|---|---|
| Main question | Do retained raw-coordinate draws from the frozen learned-transport banana HMC candidate have an output law that is detectably different from the exact analytic banana law at the tested sample size? |
| Candidate | Seed-15 root-preserving `(32,32)` dense-IAF transport, frozen identity z mass, `L=10`, step size `0.7709722545680272`, from the L=10 confirmation artifact. |
| Comparator | Independent stateless standard-normal draws transformed by the exact triangular banana map `theta_0=z_0`, `theta_1=z_1+0.35(z_0^2-1)`, and `theta_j=z_j` for `j>=2`. |
| Sample unit | One 16-dimensional banana draw. No temporal horizon is introduced. HMC draws retain four chains and their draw order. |
| Primary diagnostic | Fixed multi-bandwidth RBF MMD on a fixed per-chain subsample, with a stratified moving-block bootstrap interval. This is a distributional diagnostic, not a proof of equality. |
| Promotion criterion | None. A candidate may be called `not_detectably_different_under_screen` only when the upper descriptive MMD interval is below a calibration envelope from independent exact-vs-exact controls. |
| Hard vetoes | Missing or stale confirmation artifact, wrong candidate state/kernel, nonfinite samples, malformed chain layout, invalid memory/XLA provenance, or failed exact-vs-exact calibration integrity. |
| Explanatory diagnostics | Coordinate means/variances, banana cross-moment `E[z_0^2(theta_1-z_1)]`, MMD point estimate, bootstrap width, chain movement, and calibration envelope. These do not override a veto or establish posterior correctness. |
| Nonclaims | No universal equivalence threshold, no statistically proven equality, no HMC superiority, no transfer to SSL-LSTM, and no production/default readiness. |

## Default and assumption audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|
| Four chains x 1,024 draws/chain | Derived from the confirmation output layout and pairwise-kernel memory budget | Subsampling can miss a localized discrepancy | Repeat with a second deterministic offset and report both | Reviewed diagnostic setting |
| Moving block length 64 | Hypothesis based on the retained chain length; it preserves short-range HMC dependence while leaving 16 complete blocks | Too short or too long changes uncertainty | Compare block lengths 32 and 128 in the result note | Hypothesis, not a default |
| 256 bootstrap replicates | Convenience budget for a bounded local diagnostic | Tail quantile Monte Carlo error remains descriptive | Report bootstrap count and do not call the interval a calibrated test | Descriptive only |
| RBF bandwidths `(2.0, 4.0, 8.0)` | Revised after the pre-run shift test showed that `(0.5,1.0,2.0)` was wrong for 16-dimensional Euclidean distance: independent standard-normal draws are typically about `sqrt(32)` apart | A discrepancy can still be invisible at this finite grid | Record per-bandwidth MMD and retain the grid in the manifest | Dimension-derived diagnostic hypothesis |
| 99% upper interval | User/project convention for a conservative screen; not a literature-calibrated equivalence margin | It can be underpowered or overconservative | Exact-vs-exact calibration arm is mandatory | Screen-level only |

## Evidence contract

| Item | Predeclared value |
|---|---|
| Scientific question | Whether the frozen HMC output law differs detectably from the exact banana output law under this finite diagnostic. |
| Comparator | Exact banana simulator with independent seed family; analytic-vs-analytic controls use the same chain/block layout. |
| Primary result | Candidate MMD estimate and 99% moving-block bootstrap upper interval, compared descriptively with the 99% quantile of exact-vs-exact control upper intervals. |
| Veto diagnostics | Any invalid artifact, nonfinite draw, insufficient complete blocks, or failed exact-control reproducibility check invalidates the campaign. |
| Explanatory diagnostics | Coordinate and nonlinear moments, per-bandwidth MMD, block-length sensitivity, and runtime. |
| What will not be concluded | Passing the screen does not prove equality, convergence, multimodal coverage, or correctness outside the tested banana target. |
| Artifact | `docs/plans/artifacts/neutra-banana-predictive-equivalence-2026-08-16-r2/` with plan copy, manifest, source hashes, calibration/candidate JSON, result note, and artifact hashes. |

## Skeptical plan audit

| Risk | Disposition |
|---|---|
| Treating iid MMD theory as valid for HMC draws | Rejected: the runner preserves chain axes and uses moving-block bootstrap; the quadratic MMD itself remains descriptive. |
| Reusing SSL-LSTM horizon or feature margins | Rejected: banana uses one 16-dimensional draw and a separately recorded calibration envelope. |
| Calling a finite MMD screen an equality test | Rejected: the result status is explicitly descriptive and the nonclaim is recorded. |
| Pairwise kernel memory explosion | Controlled: fixed 1,024-draw/chain subsample and explicit memory-bounded bootstrap; no 20,000 x 20,000 kernel is formed. |
| Calibration leakage | Rejected: candidate HMC draws and analytic calibration draws use disjoint stateless seed families; calibration is not used to tune the transport or HMC. |
| False confidence from bootstrap replicates | Rejected: 256 is reported as descriptive Monte Carlo support; no formal p-value is emitted. |
| Stale HMC state | Rejected: confirmation result, kernel, state hash, and archive SHA-256 are checked and copied into the manifest. |

Audit verdict: the plan is fit for a bounded target-specific diagnostic after the result is explicitly treated as a calibrated screen with descriptive uncertainty, not as a general equality test or promotion gate.

Pre-execution repair: the first focused large-shift test invalidated the initial
`(0.5,1.0,2.0)` kernel grid because it confused per-coordinate scale with the
16-dimensional Euclidean distance used by the RBF. No campaign was run under
that grid. The revised `(2.0,4.0,8.0)` grid brackets the derived typical exact-law
distance `sqrt(2 * 16)`, and the shift test is rerun before execution.

## Execution

1. Run focused unit tests for the new sample loader, banana simulator, MMD/bootstrap statistic, and artifact validation.
2. Execute a tiny CPU smoke with the same code path and reduced counts; it is mechanics-only and not evidence.
3. Execute the GPU/XLA campaign with the frozen confirmation artifact, exact-vs-exact controls, and candidate comparison.
4. Write the result and reset memo, including decision and inference-status tables and a post-run red-team note.
