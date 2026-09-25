# iAPF and KDM public-reference comparison

Date: 2026-09-18. Owner request: create, review, and execute the comparison.
This is a new bounded diagnostic campaign, not a fourth launch of the closed
September 17 fitting campaign. The active checkout is `surrogate-hmc`.

## Question and evidence contract

Do the shared mathematical operations in BayesFilter's iAPF and KDM/IWSG
implementations agree with executable public references? Which disagreements
come from different targets, and which are implementation errors?

The primary criterion is deterministic agreement on matched targets, supported
by a third, closed-form or finite-difference calculation. A failure after
matching inputs is a repair trigger and vetoes a correctness claim for that
operation. Missing source provenance, nonfinite arithmetic, incorrect loaders,
or incomparable targets invalidate the affected comparison. Optimizer
convergence is a prerequisite for comparing fitted optima. It is not evidence
of filter quality. Raw differences on deliberately different targets explain
the implementations and do not veto continuation.

No outcome establishes a speed ranking, learned-model replication, superiority,
nonlinear score accuracy, unbiased model scores, GPU readiness, canonical LEDH
conformance, HMC readiness, or a new numerical default. The full master program
remains incomplete. This comparison precedes further fitting-control tuning.

## Sources and mathematical targets

1. KDM: official author repository
   <https://github.com/asyounis/mdpf_neurips_2023>, commit
   `b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18`, MIT license. Local checkout:
   `.localresources/code/younis-mdpf-neurips-2023`.
   Inspect the actual `KernelDensityEstimator` class, particularly `sample`
   and `log_prob`, and `KDEParticleFilter.resample_particles`'s
   `ImportanceSampling` branch. The paper is Younis and Sudderth (2023),
   Section 2.2, Section 4 equations (14)-(15), Appendix B.1, stored under
   `.localresources/papers/younis-sudderth-2023-long-range-tracking.*`.
2. iAPF: independent public repository <https://github.com/Sempreteamo/iAPF>,
   commit `a88114395f6c11075fedc653db9480791b43391a`, `iapf.R`.
   This is not verified original-author code and has no declared license.
   Keep the pinned source as a private local reference; do not vendor it into
   runtime code. Original paper: Guarniero, Johansen and Lee (2017),
   Algorithms 3-5 and equations (15)-(16), stored under
   `.localresources/papers/guarniero-johansen-lee-2017-iterated-auxiliary-particle-filter.*`.

For KDM's shared diagonal Gaussian bandwidth, map bandwidth `b` to covariance
`diag(b^2)` and its derivative to `diag(2*b*db)`. The author density uses
`wbar_i=(w_i+1e-8)/(sum(w)+N*1e-8)`. BayesFilter uses normalized weights without
that addition. Test raw discrepancies and the explicitly epsilon-aligned
density separately, including the derivative of the normalization. At the
resampling endpoint the author adds another `1e-8` to each injected unit
weight. After normalization this divides its score contribution by `1+1e-8`.
Account for that factor explicitly; do not change BayesFilter's runtime.

For iAPF let `p` be candidate Gaussian densities on the fitting cloud, `y`
the positive backward targets, `a=p'p`, `b=p'y`, and `c=y'y`. Profiling the
paper's scale gives `lambda=b/c` and `L=a-b^2/c`. BayesFilter's optional
relative-shape objective is `R=L/a=1-b^2/(a*c)`. The R source minimizes
`F=||y-p/lambda||^2=c*R/(1-R)`. Thus, where `b>0`, F and R have the same
unconstrained minima, but different gradients and optimization geometry:
`grad F=c/(1-R)^2 * grad R`. Neither is the paper's unnormalized L.
Verify this identity using the unchanged source objective and finite
differences. Source iAPF also omits the positive floor in paper equation (16),
uses adaptive ESS resampling and an exact initial Gaussian twist, and starts
its iteration counter at one. The local finite program starts at zero and
uses the published `l>k` stopping condition. Compare these differences openly.

## Executable comparisons

| Operation | Comparator and third check | Cases and acceptance |
|---|---|---|
| KDM density and analytical score | Author `log_prob` with Torch reference autodiff; direct Gaussian sum and central differences | Dimensions 1 and 2; overlapping, separated, and tiny-weight components; matched FP64 absolute/relative error <= 2e-8; finite-difference relative error <= 2e-5 |
| KDM IWSG resampling call chain | Execute author's actual `resample_particles`; feed its detached samples to BayesFilter's full-mixture IWSG kernel | Raw and epsilon-aligned weights; value and directional derivatives before/after observation-weight normalization; same FP64 tolerance |
| iAPF Gaussian twist | Original `mu_aux`, `f_aux`, `psi_t`, `psi_tilda`, `g_aux`; closed-form Gaussian product | Two-dimensional nonzero transition, unequal variances, initial/intermediate/terminal weights; <= 2e-8 absolute/relative error |
| iAPF fit | Source nested `fn`, source `Psi`/R `optim`, local bounded relative-shape fit, known Gaussian target | Shared deterministic cloud; objective identity <= 2e-8; derivative <= 2e-5; exact Gaussian fitted means/variances <= 2e-3, both solvers converged |
| iAPF iteration control | Source `Num` and actual parsed controller conditions; paper Algorithm 4 | First complete window, next window, oscillating and constant likelihood histories; exact action classification and CV <= 2e-12 |
| iAPF finite filter value | Original R `APF`, actual local `make_fitted_twist_kernel`, exact Gaussian marginal likelihood | Scalar independent transitions A=0, Q=R=H=1, three observations; Gaussian twist without floor as an explicit reference-only case; value error <= 2e-8 |

The independent-transition example makes the differing initial latent-time
conventions irrelevant, and optimal twists make the likelihood value exact.
It is a call-chain check, not a correlated or nonlinear filter replication.
The fitted-twist consumer's gradients remain derivatives of its frozen finite
program; the R source does not provide model-score ground truth.

Constructed simple adversaries: direct scalar Gaussian marginal likelihood
for the full-filter check; direct finite Gaussian-mixture sum for KDM;
closed-form Gaussian product and exact target mean/variance for twist fitting.
Evaluate each in its applicable regime, including tiny mixture weights and
unequal twist variances. These are deterministic correctness authorities, not
tuning targets or a stochastic method ranking.

## Assumptions and defaults reviewed before execution

| Choice | Provenance and reason | Failure mode and early check | Status |
|---|---|---|---|
| CPU FP64, TF XLA on | Reference arithmetic, small arrays, installed R/Torch/TF | Cannot certify GPU/TF32; record CPU hiding and actual backend | Reference exception |
| AST-load unchanged author class/method | Optional training dependencies are absent; only Normal/IWSG needed | Missing globals or altered semantics; record source hashes and executed member names; use original method body | Diagnostic loader |
| Parse only R function definitions and controller conditions | Whole script launches an unrelated large experiment | Accidental top-level execution; explicitly select definitions | Diagnostic loader |
| Base-R multivariate-normal density/sampler shims | `mvtnorm`, `FKF`, `jsonlite` absent; no package installation | Wrong covariance/RNG interpretation; closed-form checks and recorded shim; no bitwise RNG claim | Reference dependency substitution |
| R optimizer defaults unchanged | Preserve actual public reference | Unconverged returned `$par`; wrap `optim` to retain convergence and counts | Reference baseline |
| Local fit: mean bound 4, SD bounds 0.2/4, 10000 steps, 30 backtracks, tolerance 1e-7, floor ratio 1e-8 | Existing bounded solver; wide enough for the exact test Gaussian | Bound activity or insufficient steps; veto optimum comparison if unconverged/bound | Diagnostic hypothesis |
| Absent floor in matched Gaussian checks | Required to reproduce reference's target, not a safety recommendation | Loses prior-mixture component; explicitly segregate from positive-floor runtime | Reference-only hypothesis |
| Fixed data and seeds 91801-91806 | Reproducible fresh mechanics fixtures | No statistical power for ranking; forbid quality/ranking claims | Convenience fixture |

No damping, ridge, floor, optimizer, or numerical default is changed. Positive
floor behavior is reported as an intentional mathematical difference. No
cross-model tuning artifact or old LEDH result is imported.

## Budget, commands, artifacts, and stop rules

At most four harness attempts, 1800 seconds total CPU wall time including
focused regression checks, 600 seconds maximum per attempt, no GPU launches,
no model training, no package installs, and no external write/publication.
Use distinct `attemptNN` output directories. Repairs within this budget may
fix only diagnostic plumbing or a proven localized correctness defect. A
numerics/default change or broader campaign needs its own evidence contract.

Run the new diagnostic harness with
`CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 OMP_NUM_THREADS=2`:

```text
/home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/benchmarks/diagnose_younis_iapf_kdm_reference_comparison.py --output docs/plans/artifacts/younis-iapf-kdm-reference-comparison-20260918-01/attempt01
```

The harness records the command, Git commit and changes, package versions,
source hashes, source-load restrictions, CPU/XLA settings, seeds, per-check
errors and verdicts, elapsed time, and captured R output. Preserve logs,
structured results, a decision/inference table and an updated checkpoint under
`docs/plans/artifacts/younis-iapf-kdm-reference-comparison-20260918-01/`.

Stop a comparison for a corrupt/missing source, nonfinite accepted result,
unmatched target or invalid loader. Record unaffected comparisons. Candidate
disagreement or an unconverged source optimizer does not terminate unrelated
checks. Stop the campaign at its budget; report unresolved scope plainly.

## Skeptical review, completed before implementation

Self-review: PASS for this bounded diagnostic scope. The initial idea of
direct whole-method parity would be invalid: KDM has two epsilon additions;
iAPF differs in objective, floor, stopping index, initialization and resampling.
The revised checks isolate those differences and exercise real source methods
and local consumers where targets coincide. Paper Algorithm 4 supports the
local zero-based stopping boundary. No stochastic metric is promoted to a
correctness or quality criterion, and a source implementation is not treated
as infallible. Source agreement cannot rescue the failed curved-regime claims.

Pre-mortem: a wrapper could accidentally compare our own equations twice;
therefore execute pinned source bodies and preserve source/executed-member
identities. A perfectly matching Gaussian fixture could conceal nonlinear
failure; keep nonlinear quality explicitly outside the verdict. Missing
training libraries must not lead to broad environment mutation or invented
claims about running the entire author project. Review is self-review; no
independent agent or Claude review has been requested or performed.

## Attempt 1 review and bounded diagnostic amendment

Attempt 1 completed in 12.57 seconds: 128/129 checks passed. Preserve its one
failed comparison: source iAPF first-time variance 0.9959796291 versus exact
1 and local 1.0000000011, outside the predeclared 0.002 tolerance. R reports
convergence after six objective/gradient evaluations. This is a source-fit
accuracy failure, not evidence that the local Gaussian twist formula is wrong.

The inspected installed `optim` documentation says L-BFGS-B uses `factr=1e7`
and `pgtol=0` by default. The reference F objective scales with the squared
backward-target amplitude c, unlike R; an objective-reduction stopping test
can therefore stop early on an accurate-looking tiny F. Before interpreting
this as the cause, run one extra iAPF-only diagnostic keeping source bodies,
data, initialization, finite-difference step, objective, and acceptance
tolerance fixed and changing only R `optim(control=list(factr=1))` through the
existing diagnostic wrapper. Preserve the default arm and its failure. Retain
optimizer termination messages. The tighter arm is a diagnostic substitution,
not the unmodified reference or a BayesFilter default. Success requires the
same 0.002 mean/variance threshold and zero convergence code. An unsuccessful
arm leaves the cause unresolved and must not trigger a tuning search here.

Self-review of amendment: PASS. It changes a single stopping-control hypothesis
on a target whose exact solution is known; it cannot select a filter on heldout
performance. Attempt 2 uses `--methods iapf` with a new output directory.
Budget remains four attempts and 1800 CPU seconds.
