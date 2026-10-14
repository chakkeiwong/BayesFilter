# Thesis-guided iAPF reconstruction: completed diagnostic result

Date: 2026-09-27. Both requested scientific replications remain INCOMPLETE.
This bounded iAPF diagnostic tranche is COMPLETE. It has produced no eligible
fit for a fresh backward-sweep or adaptive-filter qualification run.

## What changed in our understanding

The recovered first-author thesis specifies R optim/BFGS with a diagonal
Gaussian and zero explicit penalty for all Part III state-space examples.
It also explicitly describes the vanishing-density/amplitude degeneracy and
suggests box-constrained L-BFGS-B when that problem occurs. Explicit Tikhonov
regularization belongs to its Part IV diffusion examples. These statements
correct the earlier assertion that the optimizer itself was unknown.

The original initialization, parameterization, objective scaling, stopping
controls, floor formula and executable author code remain unrecovered. The
final published article's indexed mention of regularization does not settle
those choices. Its complete final PDF could not be retrieved; the thesis is
a distinct inspected primary source, not a substitute silently called the
published article. See [source recovery](source-recovery.md), thesis Section
7.3, pages 53–55, and the local thesis text at lines 2253–2310.

## Executed work and findings

The executable master ran both phases through frozen scripts and unique output
directories. These were independent R/FP64 CPU diagnostics with GPU devices
deliberately hidden. No TensorFlow, XLA or GPU kernel was involved.

| Phase | Execution | Finding |
| --- | --- | --- |
| Calibration, attempt01 | 96 BFGS fits: 3 starts × joint/profiled amplitude × raw/scaled SSE × default/tight controls × 4 frozen t98 cases | 0/96 passed independent stationarity; 67/96 nevertheless returned optimizer code zero |
| Initial constrained diagnostic, attempt01 | Two fixed boxes on each of the same 4 cases | 0/8 passed projected gradient; residuals 0.0376–6.82 against 1e-6 |
| Solver repair, attempt02 | Same objective, coordinates, endpoints and walls; frozen positive rescaling between bounded optimizer segments; stricter stopping | 8/8 passed projected gradient, all with original-objective descent |
| Independent controls | 11 calibration controls and 16 repair controls | 27/27 passed; includes derivative, fixed-scale parity, known-Gaussian recovery, no-fire and endpoint-wiring checks |
| Fresh backward sweeps | Conditional on an eligible unbounded arm | Not launched: no arm qualified |

The repair used one positive constant throughout each optim call. It did not
minimize a parameter-dependent normalized shape loss. It therefore preserved
the constrained minimizers while changing stopping sensitivity and optimizer
trajectories. Since both scaling and stopping controls changed, these results
establish the effectiveness of the combined repair; they do not isolate either
change as the sole cause of premature termination.

Numerical completion exposed a substantive failure. All repaired endpoints had
9 or 10 of their 10 coordinates at box walls. Their squared density norms on
the fitting cloud fell by factors with log ratios between -72.06 and -440.94,
relative to the unrepaired endpoints. Shape errors increased in every case.

Here shape error means profiled SSE divided by the squared density norm on the
same cloud. It is an explanatory fitting diagnostic, not a filter-quality
criterion. The final boxes also lose to the inexpensive log-quadratic
initialization in every saved case:

| Saved seed / floor power | Log-quadratic start | Repaired radius 2 | Repaired radius 4 |
| --- | ---: | ---: | ---: |
| 9262001 / 1 | 0.001509 | 0.881262 | 0.999993 |
| 9262001 / 2 | 0.001975 | 0.210875 | 0.921686 |
| 9262002 / 1 | 0.016716 | 0.643756 | 0.996438 |
| 9262002 / 2 | 0.019704 | 0.652500 | 0.995756 |

The full conditional diagnostic table also includes weighted-moment and
cloud-moment starts. It does not evaluate any of these as filters and cannot
close the master program's downstream heuristic-dominance requirement.

## Why more accurate minimization can make this fit worse

Let p be the vector of Gaussian density values on the saved cloud and b the
positive backward-target vector, rescaled by a constant. Profiling amplitude
gives lambda=(p^T b)/(b^T b) and

L = ||p||² - (p^T b)²/||b||²
  = ||p||² [1 - (p^T b)²/(||p||² ||b||²)].

The bracketed term is the shape error reported above. The loss can decrease
because ||p||² becomes tiny even while shape agreement becomes worse. This is
an algebraic property of Eq.15, not an XLA, retracing, NumPy or Python-loop
effect. The thesis discusses the same escape and says its local BFGS procedure
found nontrivial local solutions for the reported state-space examples.

The boxes stop an unlimited escape, but these particular boxes stop it at a
bad boundary. A zero projected gradient there is a first-order necessary
condition for constrained optimality; it neither proves a local minimum nor
establishes a useful twist. These constrained endpoints are rejected for
downstream use. This does not show that all constraints or all local starts
fail, or that the published iAPF method cannot work.

The 1e-6 residual is a numerical check on our approximate-local-solution claim.
It is not a paper-specified stopping tolerance and cannot by itself decide
filter quality. Likewise an R convergence flag, small raw SSE or good initial
shape cannot establish replication. Any future finite-optimization
reconstruction must keep its source status, numerical status and downstream
filter evidence separate.

## Decision and inference

| Decision | Primary criterion | Veto / diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Engineering evidence retained | PASS: 27 controls; executable command/source wiring and frozen hashes checked | No failed validity check | Controls are finite diagnostics | Preserve both attempts and use their saved cases | Complete implementation correctness |
| Unbounded reconstruction not qualified | FAIL: 0/96 independently stationary | No candidate eligible for planned fresh sweeps | Original coordinates, initializer and stopping settings unknown | Recover those details or predeclare another source-preserving local-fit hypothesis | Absence of useful local optima; failure of iAPF |
| Solver repair works on these boxes | PASS: 8/8 projected stationary with descent | Density collapse and 9–10 active walls | Combined scaling/stopping intervention | Keep as a solver diagnostic; reject these endpoints for filtering | Constrained minima certified or source-equivalent fits |
| iAPF experiment replication | INCOMPLETE | No fresh qualifying sweep or adaptive likelihood study | Author fitting reconstruction | Resolve fitting, then fresh sweeps, adaptive likelihoods and paper-scale study | Scientific replication or TF/GPU readiness |
| MDPF replication | INCOMPLETE; unchanged by this run | Full conditional heuristic/protocol/uncertainty checks open | One NLL source trial is not the full paper protocol | Independently plan K3a conditional evaluation and K2 source reconciliation | Two-method replication complete |

| Inference status | Verdict |
| --- | --- |
| Hard veto screen | The 96 fits fail the predeclared numerical screen; the repaired boxes collapse and cannot be promoted |
| Viable candidates | No tested arm is qualified for downstream execution; untested source-preserving fitting choices remain open |
| Statistically supported ranking | None; four deliberately selected failure cases are not a random performance sample |
| Descriptive-only differences | Shape errors, density changes, runtime and all between-arm differences |
| Default readiness | Not established; these are CPU references, with no eligible TF/TFP implementation demonstrated |
| Next evidence needed | Supported fitting reconstruction, untouched complete sweeps, adaptive filtering against Kalman/classical baselines, conditional checks and uncertainty |

Terminal red-team review: the strongest alternative explanation is that our
unknown initialization/parameterization/stopping choices miss the author's
implicit local regularization. Recovered author code producing stable local
fits on comparable clouds would resolve the reconstruction gap. A failed
derivative or source-parity check on the saved cases would invalidate the
numerical diagnosis; none failed in the checks performed. The weakest evidence for any
broader generalization is the four-case selection from earlier failures.

## Artifacts, commands and budget

Executed twice through:
`/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_replication_master.py run-next`.

The first invocation ran calibration; the second dispatched the explicitly
reviewed box repair. Exact R commands, environment, commit/branch, data and
source checksums, seeds and elapsed times are in
[attempt01/manifest.json](attempt01/manifest.json) and
[attempt02/manifest.json](attempt02/manifest.json). Each attempt preserves its
own plan and source snapshots. Prior evidence was not overwritten.

The focused audit command is
`/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/audit_iapf_thesis_reconstruction.py`.
Its [evidence-audit.json](evidence-audit.json) passes 210 checks, including 191
checksum comparisons. It checks executed consumer commands against the frozen
diagnostics; it does not certify an unexecuted runtime implementation.

Calibration: 9.431763 CPU wall seconds. Repair: 0.565130 seconds. This tranche:
9.996893/600 seconds, 2/3 allowed launches. Combined authorized CPU accounting:
209.097046/1800 seconds; 1590.902954 seconds remain. This diagnostic tranche is
closed rather than spending its unused allowance on undeclared alternatives.
Zero GPU time was used. The previously recorded GPU balance is about 36.567
hours, unchanged by this work.

The next active research questions are original local-fit reconstruction
(I2b.3) and the independent MDPF conditional-baseline/protocol work (K3a/K2).
There is no unattended worker from this tranche, and no automatic paper-scale
or adaptive-filter launch is eligible from its results.
