# Guided LEDH conditional Jacobian: result

The shared single-cloud guided consumer passes the planned conditional
Jacobian and analytical total-derivative checks. All **37 regression tests
pass**. No runtime algorithm was changed. The new proposition supplies a
conditional invertibility argument; MathDevMCP did not certify it.

Work was performed on `ledh-refactor-with-policy-fix` at
`ea1018de1da55ed02a29a8d60f08a6827fce8a42`, in the LEDH execution worktree.
The separate surrogate-hmc checkout was not edited. The
[plan](../../ledh-backward-guide-jacobian-validation-20261016.md) records the
pre-execution skeptical review, assumptions, criteria and budget. Exact
commands, source hashes, settings and environment are in the
[run manifest](run-manifest.json).

## Target, computed quantity and evidence

The target is the change-of-variables Jacobian with respect to the current
pre-flow draw, conditional on the incoming cloud, physical prediction anchors,
covariances, observation and parameters. The separate localization trajectory
starts at the physical prediction and is independent of that draw. Thus the
recorded product of affine matrices is the actual innovation Jacobian under
these conditions, even with nonlinear observations.

Proposition `prop:bg-flow-jacobian` in
[the manuscript source](../../../papers/ledh_younis_kdm_score/ledh_backward_guidance.tex)
derives this equality and invertibility for positive definite predicted and
observation covariances and the fixed equal-step schedule. Whitening the
predicted covariance makes each step similar to a symmetric matrix whose
eigenvalues exceed one half. Every step determinant is positive. Jacobi's
identity then gives its log-determinant parameter derivative, including the
dependence of the physical anchor and nonlinear localization trajectory.
This is a checked conditional derivation, not a statement about every
possible localization rule or execution lane.

The new endpoint test captures the flow actually invoked by the guided
consumer at both observations. A nonlinear observation law and a moving
localization trajectory prevent the check from reducing to a linear fixture.
Its full 16-by-16 finite-difference Jacobian agrees with the returned
block-diagonal affine matrix and recorded positive log determinant at
relative/absolute tolerances 2e-6/2e-8.

The second test perturbs parameters through the entire guided program.
Both observation log-determinant tangents and the final scalar directional
score agree with central differences at steps 2e-6 and 1e-6, with tolerances
2e-5/2e-7. The second observation follows a reset with nonzero state
sensitivity, so it exercises incoming-cloud dependence as well as the direct
parameter terms. These checks concern the total derivative of this finite
program. They do not establish equality with the exact model likelihood score.

The actual consumer-to-flow call chain is exercised by
`test_guided_consumer_actual_nonlinear_flow_jacobian` and
`test_curved_guided_flow_logdet_and_score_total_derivative` in
[the guide tests](../../../../tests/highdim/test_ledh_backward_guide_tf.py).
The [regression log](integration-tests.log) and
[JUnit record](integration-tests.xml) preserve the executable evidence.

## Execution and audit

Tests used TensorFlow/TFP in the tftwogpu environment with FP64 CPU reference
fixtures. GPU devices were intentionally hidden with
`CUDA_VISIBLE_DEVICES=-1`; these are explicit non-JIT reference/debug checks.
The new checks use N=8, two state dimensions and two observations. Epsilon=1
and five Sinkhorn iterations remain the owner's untuned development
comparator; 32 terminal balancing iterations remain a reference-fixture
setting. None is newly promoted or tuned by this validation.

The first focused attempt passed the total-derivative check but failed
endpoint capture because it observed symbolic tracing rather than executed
inputs. A test-only capture repair resolved that failure. Both new checks
then passed, followed by the 37-test regression. There were two TensorFlow
Probability deprecation warnings. No scientific validity guard was relaxed.

The label-scoped [MathDevMCP audit](mathdev-label-audit.json) returned
**inconclusive** for all ten obligations: seven were not encodable and three
were not extracted. None was certified or refuted. This is an independent
audit limitation, not a formal proof certificate or evidence that all
mathematical claims passed.

The installed environment lacks latexmk. The build used pdflatex and bibtex
without dependency installation. The final [64-page PDF](ledh_younis_kdm_score.pdf)
build has no reported LaTeX warnings or overfull/underfull boxes. Changed
proof pages 57 and 58 were rendered and visually inspected; both are clean.

Three pytest attempts consumed 190.17 wall seconds. Timed build commands
consumed another 3.00 seconds, for 193.17 timed seconds against the 20-minute
CPU-check allowance. Bibtex, rendering and review were not timed separately;
this aggregate is not the elapsed session duration. One test-harness repair
and one build invocation adaptation used two of three allowed local repairs.

## Decision

| Field | Result |
| --- | --- |
| Decision | Complete this bounded Jacobian-validation stage |
| Primary criterion | Actual consumer Jacobian, determinant, determinant tangent and finite-program directional score pass the declared checks |
| Veto diagnostics | No unresolved nonfinite, singularity, call-chain or numerical-disagreement veto in the final run |
| Main uncertainty | Generality beyond the conditional proof assumptions and tested fixture; formal audit remains inconclusive |
| Next justified action | Design the separate scope-specific guide-fitting and untouched target-score experiment |
| Not concluded | Exact model-score correctness, score improvement, unbiased full-filter normalizer, GPU/XLA readiness, tuned default, canonical admission or HMC readiness |

| Inference question | Status |
| --- | --- |
| Hard veto screen | Final deterministic checks pass; the initial failure was test instrumentation |
| Statistically supported ranking | Not evaluated; this is not a stochastic method comparison |
| Descriptive-only differences | Timings are recorded for budget accounting, not method ranking |
| Default-readiness | Not established; epsilon/Sinkhorn settings remain an untuned development comparator |
| Next evidence needed | Scope-specific fitting/tuning, untouched score references, conditional heuristic comparisons and uncertainty analysis |

## Post-run red-team review and next action

The strongest alternative explanation for passing finite differences is that
the small fixture misses a dependency present elsewhere. The nonlinear
observation, explicit moving anchor, full innovation Jacobian, two parameter
steps and second observation reduce that risk, but do not remove it across
models, dimensions, devices or batch lanes. If localization were changed to
depend on the sampled point, the affine proof would no longer apply.
An endpoint Jacobian disagreement, a missing total parameter term, or failure
of its stated covariance assumptions would reopen this result.

The remaining scientific question is whether a fitted guide improves score
estimation on the actual target. The next experiment needs a separate
scope-specific fitting and tuning protocol, disjoint validation and untouched
claim data, the plain LEDH comparator, constructed cheap heuristic baselines,
conditional evaluations and uncertainty. The current tests neither promote
nor reject that research direction. No such campaign, GPU run, commit, push
or branch change was performed in this continuation.

