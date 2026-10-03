# Density fitting can converge by losing density mass on its own particle cloud

Five of the 16 inspected actual time-wise fits decreased density loss while
increasing shape error and decreasing sampled density energy. The clearest
case is terminal time, d2/s82/cloud_moments/initial_peak: loss falls from
0.2887 to 3.804e-9, shape error rises from 0.7579 to nearly one, and KL to the
exact message rises from 1.242 to 100.711. This fit reports convergence and
hits a bound. Its target is the observation likelihood, with no future message;
recursive contamination cannot explain this terminal failure.

The arithmetic is correct for the claimed profiled Equation-15 objective on
these checked inputs. Four real adaptive consumers reproduced their histories
exactly, including the original d5 rejection. The three completed consumers
also reproduced coefficients, values and scores exactly. Fourteen fitting calls
were captured; the last call of each supplied the 16 inspected time/cloud pairs.
All 16 isolated original fits reproduced the captured coefficients and floors
exactly. Sixty-four objective/target combinations were evaluated. Independent
base R checks of 384 parameter sets agree with native density loss, energy,
shape and both analytic gradients; the largest discrepancy is 2.50e-14 for the
shape gradient. All graphs traced once per declared signature.

## Why this happens

Let p_i be a Gaussian density evaluated at the fitting particle x_i and b_i the
positive recursive target g_t(x_i) Fpsi_(t+1)(x_i). GJL Section 5.1 Equation 15
minimizes the sum of squared differences p_i-lambda b_i. The optimal scale is
lambda=(p'b)/(b'b). Substitution gives

L = (p'p/N) [1-(p'b)^2/((p'p)(b'b))] = E S.

E is sampled density energy; S is the squared residual after normalizing out
that energy. Thus small L need not mean good shape: E can collapse while S
rises. Moving a Gaussian mean arbitrarily far from any finite fitting cloud
at fixed covariance proves that the unbounded objective has infimum zero even
without a shape match. The current solver uses finite bounds, so that argument
alone does not prove failure of a particular bounded fit. The captured events
below establish this mechanism on actual bounded trajectories. The loss omits
the fixed Gaussian normalization/cloud-scale factor exactly as the local code
does; within each cloud it preserves the objective and its minimizers. The
fixed initial_peak factor affects stopping units but does not remove E.

| case | time (1-based) | density loss | density energy | shape error | KL to exact message |
|---|---:|---:|---:|---:|---:|
| d2-s82-cloud_moments-initial_peak | 2 | 0.14195 → 4.9557e-10 | 0.32524 → 5.0045e-10 | 0.43643 → 0.99025 | 0.70095 → 22 |
| d2-s82-cloud_moments-initial_peak | 4 | 0.28865 → 3.804e-09 | 0.38085 → 3.804e-09 | 0.75791 → 0.99999 | 1.2417 → 100.71 |
| d5-s82-log_quadratic-initial_peak | 1 | 0.12255 → 1.5254e-07 | 0.21568 → 1.739e-07 | 0.56823 → 0.87716 | 4.5569 → 6.8563 |
| d5-s82-log_quadratic-initial_peak | 4 | 0.0047177 → 6.4046e-09 | 0.025271 → 9.2777e-09 | 0.18668 → 0.69033 | 0.54484 → 7.0997 |
| d10-s82-log_quadratic-native | 3 | 0.00094229 → 2.739e-08 | 0.0068189 → 1.4467e-07 | 0.13819 → 0.18933 | 4.5836 → 4.3969 |

These are selected diagnostic cases, not an estimate of failure frequency.
Correct optimization of a poor empirical objective is different from an
arithmetic defect. A converged fit is not sufficient evidence of guide quality.
The author's initialized local solver may avoid these trajectories; its unknown
choices and the paper's other model/controller differences remain unresolved.

## Remaining mechanisms

Backward target error is also real. In d10 at the first time, even the full
exact backward Gaussian has shape residual 0.5317 against the actual recursively
constructed target. Against the independently exact target its residual is
numerically zero. In d5 the corresponding discrepancy is 0.4515. A solver that
fits an incorrect approximate future target perfectly can still give a poor
true future guide. Holding the cloud fixed separates this target discrepancy
from fresh randomness or different adaptive stopping.

The existing relative_shape alternative minimizes S, a different objective
from Equation 15. It converges in 9/16 actual-target fits and 9/16 exact-target
fits under the unchanged 2000-step cap, compared with 15/16 for density_l2 in
each target group. The unchanged density convergence rate is not evidence of
quality. Relative shape repairs the terminal d2 geometry in this example
(KL 0.0523), but all four inspected d10 fits remain unconverged; their results
are rejected, not repairs. An independent bounded quasi-Newton solver is the
next small check of whether projected-gradient conditioning explains this gap.

The cloud-moment, strict QR, KL-diagonal and full exact Gaussian comparators
are explicitly recorded in each row. Their shape/objective/KL scores are
conditional explanatory diagnostics. They do not clear the prior downstream
bootstrap, constant-guide, one-step proposal and Kalman filtering vetoes.

## Decision, inference and terminal review

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Confirm actual objective escape | Replay and independent arithmetic pass; five events | No harness invalidity | Diagnostic selections; local solver choices | Preserve math and captured clouds | Every iAPF implementation fails |
| Reject relative-shape promotion | Several fits remain unconverged | 7/16 actual-target convergence vetoes | Conditioning, local minima, sampled support | Independent bounded optimizer comparison | Shape improvement is filter improvement |
| Record recursive target error | Exact Gaussian disagrees with actual target | Explanatory only | Future fitting/floor contributions | Separate from local optimizer diagnosis | Optimizer change alone fixes recursion |

| Inference status | Finding |
|---|---|
| Hard veto screen | Source/identity/replay/arithmetic pass; original d5 and alternative nonconvergence preserved |
| Statistically supported ranking | None |
| Descriptive-only differences | Per-cloud loss, shape, energy, KL and convergence outcomes |
| Default readiness | No production changes; original filtering and TF32 vetoes remain |
| Next evidence needed | Independent solver isolation, then fresh end-to-end validation of any viable repair |

Terminal skeptical review PASS for the causal diagnosis. The strongest
alternative explanation of poor nonterminal KL is recursive target contamination,
which is measured separately and cannot explain the terminal example. The
weakest evidence is selection of four small diagnostic cases and the lack of a
new downstream filter trial. A source/replay mismatch or failure of the R
objective identity would overturn the attribution; neither occurred. The result
does not establish a preferred solver, universal hyperparameters, paper-scale
replication, marginal model-score quality, canonical LEDH or HMC readiness.

Exact commands, snapshots, seeds, device settings and wall times are in the
three launch records. GPU work used FP64/XLA, TF32 disabled and verified growth
on RTX 4080 SUPER. Independent base R verification hid GPU. Production sources
and the frozen R filtering reference were unchanged. The next phase is within
the existing authorization and remaining compute budget.
