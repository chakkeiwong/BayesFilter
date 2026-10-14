# iAPF fitting root-cause audit — 26 September 2026

Status: COMPLETE (27 September 2026). User request: trace the code and reread the paper and mathematics
to explain the actual failures. This is a diagnostic investigation, not a new
replication or a change to the default fitting objective.

## Question and evidence contract

Why do the saved Section 5.2 d=5 backward fits pass at times 100 and 99 and
fail at 98? Distinguish a source/transcription mismatch, incorrect backward
targets, incorrect derivatives or numerical scaling, ill-posed global fitting,
and solver/local-basin failure. Explain what the evidence can and cannot
identify about the authors' actual procedure.

Baseline: the exact executed source copies and four saved inputs in
`artifacts/iapf-scale-aware-reconstruction-20260926-01/attempt03`, plus the
published-method equations in the locally held accepted manuscript and the
recovered arXiv TeX sources. Follow the consumer call chain, not just isolated
function definitions. The final publisher text remains a source gap unless
recovered in this audit.

Primary criterion: a causal account supported by derivation and executable
checks on the saved failures. There is no method promotion or stochastic
ranking in this audit. Numerical equality checks use scale-aware relative
errors (1e-5 for finite differences, 1e-10 for analytic double-precision
identities), with absolute error also reported near zero. A finite local fit
requires the existing relative-gradient threshold 1e-6 and nonnegative
curvature within the existing numerical tolerance; that would only nominate
a local solver, not reproduce the paper.

Continuation vetoes: source/input hash mismatch, wrong model or data, failed
derivative/analytic-identity checks, nonfinite diagnostic arithmetic, time
budget exhaustion. A fit that escapes or reaches its iteration limit is an
explanatory result, not a continuation veto. Repair a localized diagnostic
harness failure within the stated budget and preserve the failed attempt.

Explanatory diagnostics: relative shape residual, density amplitude, lambda,
parameter displacement, Hessian/Jacobian eigenvalues, fitting-cloud coverage,
target concentration and floor contribution. No explanatory threshold silently
becomes a promotion criterion. Failed basins do not prove absence of other
finite local minima; successful basins do not prove original-author fidelity.

## Execution

1. Reinspect Equations 5–6, 10, 12–16, Algorithms 3–5, Section 5.2 and relevant
   appendix arguments. Check available source versions and original-code
   provenance. Reuse prior source ledgers; record any new source gap explicitly.
2. Trace the executed runner to particle-cloud generation, backward targets,
   initialization, profiled objective, derivative, LM step, stopping rule,
   floor, and controller. Record exact line anchors.
3. Replay the four saved failures. Verify targets independently using the
   Gaussian product/convolution formula, verify the objective/gradient and
   fixed-scale acceptance, and characterize the first escape steps.
4. Discriminate mechanisms using the same clouds: remove the floor as a
   diagnostic counterfactual; construct an exactly diagonal Gaussian target;
   inspect the full-covariance Gaussian component; compare reduced and joint
   optimization and an independent BFGS route. Preserve the original objective
   for optimizer comparisons. Do not substitute shape or log loss and call it
   Equation 15. At most 800 iterations per alternative fit; no random search.
5. Write a root-cause report, executable evidence, decision table and updated
   checkpoint/master entry. Report confirmed causes separately from unresolved
   possibilities and from historical TF/XLA execution issues.

## Defaults, assumptions and controls

| Choice | Provenance / justification | Failure mode and early check | Status |
| --- | --- | --- | --- |
| R/FP64 CPU | Existing independent reference and saved RDS inputs | Does not test TF/GPU behavior; say so | Diagnostic exception |
| Four actual failures | Two data seeds, two floor choices from prior campaign | Selection cannot estimate failure frequency; restrict conclusion to these cases | Fixed evidence |
| QR initial fit | Existing executed code | Good log fit need not be in a raw-loss local basin; inspect raw and shape derivatives | Hypothesis |
| Profile lambda | Analytic minimization of paper Eq15 | Objective equality does not imply identical optimization trajectory; compare joint/reduced steps | Checked math, trajectory hypothesis |
| Diagonal Gaussian | Paper Section 5.1 | Correlated backward target is outside this family; use exact Gaussian controls | Source restriction |
| Floors N^-1 / N^-2 | Prior reconstruction, not recovered author choices | Floor can change target shape; compute its contribution and zero-floor counterfactual | Hypotheses |
| 1e-6 stationarity | Prior scale-aware diagnostic | Coordinate units matter; retain fixed initial-SD units and report curvature | Diagnostic criterion |
| BFGS / joint LM | Independent local-solver probes | Failures cannot exclude other solvers or starts; no ranking or default selection | Hypotheses |

The cheap adversaries are exact diagonal Gaussian recovery, the exact full
Gaussian product/convolution, and direct joint least-squares arithmetic. Check
conditionally at terminal, first successful nontrivial, and failed backward
times. This is a deterministic component audit; it does not run or satisfy a
whole-filter heuristic dominance comparison.

## Budget, artifacts and skeptical review

Use at most 600 additional CPU wall seconds, within the existing 1668.338229-second remaining
CPU allowance. Zero GPU hours. New versioned output root:
`docs/plans/artifacts/iapf-root-cause-audit-20260926-01/`.
Reserve 60 seconds conservatively for preliminary trace inspections whose exact
timing was not preserved. Permit one structured launch plus one localized harness
repair inside the same 600 seconds. This replaces the initial four-launch sketch,
not the compute limit or scientific target.
Runner: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python
scripts/run_iapf_root_cause_audit.py`. Geometry identities must pass before solver
probes in that same process. The runner freezes sources, records commands,
seeds, git identities, environment, inputs, wall time and output checksums.
It hides GPUs and uses one BLAS thread before starting R. No environment changes.

Pre-run skeptical audit: PASS for this diagnostic scope. The comparison uses
actual failures and unchanged objectives, separates numerical correctness from
fit quality, allows failed candidates as evidence, and forbids converting
unknown author choices into source facts. A key pre-mortem is that an optimizer
can report convergence only because amplitudes vanish; independently recompute
the scale-aware gradient and curvature at every claimed solution. A second is
that a different objective appears to repair Eq15; record the quantity actually
computed. Source-version gaps and lack of full-filter validation remain explicit.

Review is self-review; no independent reviewer is claimed. The literature skill
is applied to technical source support and correction/source leads, not as a
claim of a comprehensive literature survey or bibliometric ranking.

Pre-execution refinement (27 September): a positive floor is already added at
T, so both t99 and t98 complete targets are mixtures. Do not mislabel t99 as
exactly Gaussian or infer that mixture mismatch alone explains the time of
failure. The same-start no-floor and exact-diagonal targets discriminate its
contribution; independent BFGS and joint-lambda LM discriminate optimizer
trajectory dependence. Both optimized quantities remain the original Eq15.
Syntax checks passed. The frozen source/input hashes must match the historical
manifest before the new numerical diagnostic starts. No stochastic ranking or
full-filter promotion follows from this deterministic saved-input experiment.

## Completion

Attempt01 completed with 48/48 checks passing and the declared controlled probes.
See artifacts/iapf-root-cause-audit-20260926-01/result.md and decision.json.
Structured run7.438382s plus60s conservative reconnaissance charge; GPU0.
The failure is raw-loss amplitude escape; no fitted adaptive pilot qualifies.
Self-review PASS for this bounded diagnostic conclusion; other local basins remain open.
