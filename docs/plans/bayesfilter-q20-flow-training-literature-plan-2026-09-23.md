# Literature investigation of weak nonlinear NeuTra learning

Question: which documented optimization and parameterization failures can
explain the almost-affine saved q20 flows, and which source-grounded changes
are worth a controlled repair? Baseline: the exact four-parameter UKF-target
training and saved-map findings in the September 23 mathematical audit.

Inspect primary method, derivation, experimental and relevant appendix sections
for NeuTra/IAF, variational path-gradient estimators, nonlinear scalar flow
transformations, and scale/conditioning controls. Inspect author implementations
for recommendations materially affecting code. Use ResearchAssistant for
retrieval/parsing where available, with public-source fallback for unavailable
providers. Preserve tractable sources under
`.localresources/q20-flow-training-literature-20260923/` and the source/retrieval
record under `docs/plans/artifacts/q20-flow-training-literature-2026-09-23/`.

Evidence contract: a recommendation passes this investigation only when its
mechanism has an inspected equation/section or code anchor and its relation to
our implementation is explicit. Source mismatch or unavailable technical text
vetoes the affected attribution. Experiments on different targets support a
hypothesis, never local numerical defaults, superiority, convergence or HMC
readiness. Separate mathematical capability, optimization behavior, and actual
q20 benefit. No paper's hyperparameters become our default by repetition.

Bound scope to an initial core of eight papers, extending to at most twelve
if citation chaining exposes a directly relevant mechanism. These counts are
convenience limits on reading scope, not scientific thresholds. Public searches
and downloads use no paid services or new credentials. No training, target
evaluation, GPU work or campaign extension is included. Document retrieval
failures and coverage limits; do not stop useful source reading for an optional
provider. The owner request authorizes this public literature survey.

Skeptical audit: nearly affine values do not bound derivatives; nonlinear
residuals do not prove architectural impossibility; clipping is now occasional
in intervention arms; parent-to-final improvement is not late convergence.
Reverse-KL, score-matching and forward-KL objectives are distinct. Dropping the
wrong derivative term can bias a variance-reduction estimator. Spline knots,
tails and smoothness may affect HMC. A bounded scale can prevent instability
while also impeding contraction. Universal approximation does not establish
trainability, and image-density success is not target-specific VI evidence.
The survey will preserve these distinctions and nominate a small sequence of
discriminating repairs, with no implementation or experiment promotion. Audit
passes with those limits.

Completed with twelve papers, including the directly relevant 2024 fast
path-gradient follow-up. The
[result](bayesfilter-q20-flow-training-literature-results-2026-09-23.md)
and [source ledger](artifacts/q20-flow-training-literature-2026-09-23/source-ledger.json)
preserve technical anchors and coverage limits. Author code was inspected as
source only. No target evaluations, training, HMC, or GPU checks were run.
Source review added two material qualifications: exact identical sigmoid
components have only affine initial scalar directions, and the inspected
NAF implementation clips the range despite the ideal map being full-range.
The fast path-gradient recursion applies mathematically to IAF, but the
paper's optimized experiments use coupling flows; local cost remains untested.
