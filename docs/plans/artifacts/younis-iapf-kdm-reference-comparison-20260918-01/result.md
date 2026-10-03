# Public-reference comparison: completed with an explained source-fit failure

Date: 2026-09-18. [Reviewed plan](../../younis-iapf-kdm-reference-comparison-2026-09-18.md).

The shared KDM density, score and IWSG operations agree with the official author
code after its epsilon additions are included in the mathematical target.
iAPF's Gaussian products, backward normalizers, incremental weights and one
exact finite-filter example also agree. The public R iAPF fit misses one
predeclared accuracy threshold despite returning convergence code zero.
Changing only its optimizer stopping control resolves that discrepancy on
the same known-answer fixture. No BayesFilter runtime or default was changed.

These results establish agreement for the tested common operations. They do
not explain away the prior curved-regime score failures. The R reference is
an independent public implementation, not verified original-author code;
the KDM reference is the original authors' implementation.

## What ran

The pinned KDM source is `asyounis/mdpf_neurips_2023` at
`b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18`. The pinned iAPF source is
`Sempreteamo/iAPF/iapf.R` at `a88114395f6c11075fedc653db9480791b43391a`.
Both source files and papers were inspected before the plan was finalized.
The unchanged author KDE class and actual `resample_particles` method ran
through a restricted AST loader. The actual R `Psi`, nested objective,
`APF`, auxiliary proposal/weight functions, `Num`, and parsed controller
conditions ran without executing the original script's large top-level
experiment. Base-R Gaussian density/sampling functions replaced unavailable
R package primitives; sampling uses a Cholesky factor, so random streams are
not claimed bitwise identical to MASS. R optimizer termination codes and
messages were retained by a diagnostic wrapper.

The BayesFilter endpoints were `make_gaussian_kdm_kernel`,
`make_full_mixture_iwsg_resampling_kernel`, `bounded_density_fit`, the shared
twisted proposal/normalizer functions, `iteration_decision`, and
`make_fitted_twist_kernel`. The separate 19-test regression suite includes
actual master-consumer execution, fitted-control forwarding, independent
final streams, both KDM representations, and finite-program score checks.
This is executable component and consumer evidence; the complete public
training and outer adaptive iAPF experiments were not reproduced.

All numerical runs deliberately hid the GPU with `CUDA_VISIBLE_DEVICES=-1`.
Local repeated kernels used CPU TensorFlow/XLA FP64; author-reference
derivatives used CPU Torch. Versions: TensorFlow `2.20.0-dev0+selfbuilt`,
Torch `2.11.0+cpu`, Python `3.11.15`, R `4.1.2`. Main commit:
`6fbcf3147660c40d5d5644bbcbcc9fadbcb06aef`, branch `surrogate-hmc`.

## KDM results

Six fresh fixtures cover dimensions one and two, overlapping and separated
components, and a component weight of `1e-12`. There are 66 passing KDM
checks. Maximal absolute discrepancies after matching targets are:

| Computation | Maximum absolute discrepancy |
|---|---:|
| Log density, local versus author | 7.11e-15 |
| Analytical score versus author autodiff | 1.20e-14 |
| Analytical score versus central difference | 6.47e-10 |
| IWSG unnormalized-weight derivative | 1.11e-16 |
| Next normalized-weight derivative, author epsilon included | 9.71e-17 |

The author density is `qbar(x)=sum_i wbar_i phi_i(x)`, with
`wbar_i=(w_i+epsilon)/(1+N*epsilon)` for normalized input weights and
`epsilon=1e-8`. BayesFilter evaluates `q(x)=sum_i w_i phi_i(x)`.
The author source adds another epsilon to its unit-valued IWSG weights.
The corresponding normalized-weight derivative differs by `1/(1+epsilon)`.
Matching these definitions accounts for the measured differences; no local
analytical-derivative error was found on these cases.

Raw log-density differences are about `2e-8` for ordinary weights but reach
`9.0653` in a tail near the tiny-weight component. This is a different density,
not floating-point disagreement. It does not mean that the total distributions
are far apart: from the formula above the mixture-weight total variation is
at most `N*epsilon/(1+N*epsilon)`, while relative density errors in a rare tail
can be large. The unchanged author sampler uses the original categorical
weights, whereas its `log_prob` adds epsilon. Thus these checks establish
agreement with its regularized density derivatives; they do not establish
an exactly unbiased IWSG estimator for that sampler/density combination.

Source anchors: author `kernel_density_estimator.py:146` (sample), `:232`
(density), and `kde_particle_filter.py:738` (IWSG), `:927` (subsequent weight
normalization). Paper anchors: Younis and Sudderth (2023), Section 2.2,
Section 4 equations (14)-(15), Appendix B.1. Shared diagonal bandwidths were
compared; learned bandwidth networks and full-covariance extensions were not.

## iAPF results and mathematical explanation

The R objective is different from paper equation (15). With fitting-cloud
Gaussian values p and backward targets y, define `a=p'p`, `b=p'y`, `c=y'y`.
After profiling the scale `lambda=b/c`, the paper minimizes

```text
L = ||p-lambda*y||^2 = a-b^2/c.
```

The public R source instead minimizes

```text
F = ||y-p/lambda||^2 = c*(a*c/b^2-1).
```

Our optional relative-shape objective is `R=L/a`. Substitution gives
`F=c*R/(1-R)` and `grad(F)=c*grad(R)/(1-R)^2`. These identities agree with
the executed R objective and independent finite differences on three fixtures.
For positive overlap b, they have the same unconstrained minima. Bounds,
optimizer geometry and stopping rules can still give different finite fits.

This distinction matters to the stalled fitting problem. If Gaussian values
on a fixed cloud shrink while their relative shape stays poor, L shrinks with
their squared amplitude. An optimizer can report a tiny loss or gradient
without an accurate twist. Dividing by a removes that failure mechanism.
The R objective likewise removes the candidate-density amplitude, but still
scales with c, the squared backward-target amplitude. Its default stopping
rule can therefore stop early when backward targets are small.

That last mechanism appeared in the executed known-Gaussian example:

| First-time fit | First variance | Error from exact variance 1 |
|---|---:|---:|
| Public R `optim`, unchanged defaults | 0.9959796291 | 0.0040203709 |
| Same source, only `factr=1` | 1.0000005315 | 5.31e-7 |
| Local relative-shape fit | 1.0000000011 | 1.13e-9 |

The original R arm returns code zero and
`CONVERGENCE: REL_REDUCTION_OF_F <= FACTR*EPSMCH` after six function/gradient
evaluations. The stricter arm takes twelve and returns code zero. Across all
three times, its largest variance error is below `7.4e-7`, with the original
`0.002` accuracy threshold unchanged. Local fits take 46-49 iterations and
pass convergence, bounds and exact-Gaussian checks. These deterministic
known-answer results diagnose premature stopping in this R fixture; they are
not evidence of a general solver or filter ranking.

The R source omits the positive floor required by paper equation (16), whereas
the local fitted twist includes it. Gaussian-component comparisons deliberately
set the floor to zero as a reference-only case. The paper's zero-based
Algorithm 4 stops only when `l>k`. The R script starts at one and can stop one
iteration earlier. Executed controller checks confirm that local behavior
matches the paper at that boundary; copying the R boundary would change it.

For the scalar exact-filter case, `A=0`, `Q=R=H=1`, observations
`[0.3,-0.5,0.8]`, `N=32`, and an exact Gaussian observation twist give:

```text
Closed-form log likelihood: -4.041536370453937
Local finite-filter value:   -4.041536370453937
Public R APF value:          -4.041536370453940
```

Independent transitions make the different initial-time conventions irrelevant
in this fixture. The source supplies no derivative of the true likelihood to
validate our model score. Correlated full-filter equivalence, curved-model
quality, adaptive outer-loop equivalence, and positive-floor performance
remain unevaluated by this comparison.

Source anchors: `iapf.R:54` (initial twist), `:60` (weights), `:72`
(transition), `:102` (CV), `:114` (normalizer), `:142` (APF), `:191`
(backward fit), `:217` (objective), `:230` (optimizer), `:322` (controller).
Paper anchors: GJL (2017), Algorithms 3-5 and equations (15)-(16).

## Decision and inference

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Accept tested KDM algebra agreement | All 66 checks pass after matching epsilon semantics | No matched-target numerical veto | Full learned model and other covariance families untested | Keep exact target definitions explicit in subsequent comparisons | General model-score correctness or GPU readiness |
| Accept tested iAPF Gaussian operations | Proposal, normalizer, weights and exact value agree | No algebra/value veto | Nonlinear family and bound/floor behavior | Diagnose curved fitting separately | Curved-regime success or full reference replication |
| Reject R default fit as a high-accuracy oracle on this fixture | Variance discrepancy 0.00402 exceeds 0.002 | Accuracy veto retained | Target scaling affects termination | Use exact solution plus explicitly controlled reference precision | All R iAPF results are wrong |
| Continue master Phase 0E investigation | Reference disagreement classified; tighter R check passes | No continuation veto | Curved claim failure remains | Fresh fitting-control and family/bound/floor diagnostics | LEDH/default/HMC promotion |

| Inference status | Assessment |
|---|---|
| Hard veto screen | One source-default optimizer accuracy failure; no matched algebra/value failure |
| Statistically supported ranking | None; no stochastic ranking was attempted |
| Descriptive-only differences | Error sizes concern these deterministic fixtures; timing and iteration counts do not rank methods |
| Default readiness | Unchanged and not established by this CPU reference comparison |
| Next evidence needed | Fresh scope-specific calibration, nonlinear conditional oracle/heuristic comparisons, and powered untouched claims |

Post-run review: PASS for reporting this bounded comparison, with the failed
source-default check retained. Strongest alternative explanation for a
misleading broad conclusion is that easy Gaussian cases conceal the curved
failure. The next curved diagnostic could overturn any extrapolation from
these fixtures, so none is made. The weakest reference is the independent R
implementation: altered objective, omitted floor, stopping-index difference,
dependency substitutions, and limited source provenance prevent treating it
as an authoritative full-paper reproduction. No independent reviewer was used.

## Evidence and accounting

- [Attempt 1](attempt01/results.json): 128/129 checks pass, 12.5707 seconds.
- [Attempt 2, iAPF only](attempt02/results.json): 77/78 pass, 8.3002 seconds;
  the same original default-fit failure remains; all fifteen added checks of
  the tighter source fit pass. It deliberately still exits 1 to preserve the
  negative comparison. There are 143 passing checks out of 144 distinct checks
  across the two attempts, not 205 independent checks.
- [Regression log](regression.log) and [JUnit result](regression.xml): 19 passed,
  zero skipped/failed, 59.399 seconds; actual consumer wiring is exercised.
- R source outputs: [original run](attempt01/r-reference.csv),
  [controlled run](attempt02/r-reference.csv), and
  [optimizer messages](attempt02/r-reference.log).
- [Campaign accounting](campaign-accounting.json): 2/4 attempts, 80.2699 measured
  numerical/test seconds; 100/1800 CPU seconds conservatively charged.

Per-attempt JSON files preserve commands, seeds, source hashes, environment,
Git state, CPU/XLA settings and errors. No GPU launch, training, package
installation, publication, or runtime-default change occurred. The saved
regression command is recorded in the campaign accounting file.
