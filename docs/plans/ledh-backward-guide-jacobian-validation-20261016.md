# Backward-guided LEDH: conditional flow Jacobian validation

## Question and scope

Does the shared executor used by the optional frozen backward guide record the
actual change-of-variables Jacobian, including with nonlinear observations?
The earlier integration plan completed implementation and branch integration;
this bounded continuation resolves its next mathematical dependency. Work in
the clean LEDH checkout `ledh-refactor-with-policy-fix`, initially
`ea1018de1da55ed02a29a8d60f08a6827fce8a42`. Preserve the separate, dirty
`surrogate-hmc` checkout.

The mechanism under examination is the auxiliary localization trajectory. It
starts at the physical transition mean, independently of the current Gaussian
innovation, and supplies the flow coefficients. Conditional on the incoming
cloud, the resulting map of the sampled point is therefore affine. Its
parameter derivative must still include movement of that trajectory.

## Evidence contract and skeptical review

The comparator is a central finite-difference Jacobian of the actual shared
flow callable reached from the guided executor, holding its physical anchors,
covariances, observation and parameters fixed. Compare it with the accumulated
affine matrix and recorded log determinant. Independently compare the recorded
analytical parameter tangent of the determinant with finite differences through
the full guided consumer, including the later observation after a reset.
Nonlinear observation curvature must be nonzero in this check.

Pass requires agreement at predeclared FP64 finite-difference tolerances:
Jacobian and log determinant relative/absolute tolerance `2e-6`/`2e-8`;
directional determinant and scalar-score tolerance `2e-5`/`2e-7`.
Use a second finite-difference step where needed to distinguish cancellation
from a missing derivative. Non-finite results, a singular Jacobian, missed
consumer wiring, or unexplained disagreement veto the engineering claim.
Timing is explanatory only. This is a deterministic engineering check, not a
comparison of stochastic score accuracy.

Skeptical review: the old accumulated-matrix determinant check is insufficient
because it does not differentiate the nonlinear consumer's map. Using a
transported-particle localization point would invalidate the affine argument;
the executable check must exercise the separate physical anchor. The guide
must not replace that anchor. Testing only the first observation would miss
dependence through reset moments and weights. The proof must distinguish the
innovation Jacobian from the total parameter derivative. CPU/non-JIT results
cannot establish GPU/XLA readiness. These issues are addressed by endpoint
capture, nonlinear observations, both observations and full-program finite
differences. Proceed under this bounded contract.

No model-score improvement, unbiased full-filter normalizer, canonical
admission, HMC readiness, tuned default, or empirical ranking follows. The
guide coefficients remain frozen. Guide fitting and untouched target-specific
score comparisons require the separate experiment design already identified
by the integration plan; they are outside this continuation.

## Assumptions and defaults

| Choice | Provenance and justification | Failure mode and earliest check | Status |
| --- | --- | --- | --- |
| Separate auxiliary anchor | Shared executor initializes localization at the physical prediction | Sample-dependent coefficients; capture and perturb actual consumer inputs | Implementation hypothesis to verify |
| Positive definite predicted and observation covariances | Existing Gaussian/UKF route; needed for invertibility proof | Invalid solve; finite/SPD checks in focused fixture | Mathematical assumption |
| Seven equal flow steps | Existing reference fixture; also exercises moving nonlinear localization | Degenerate linear-only test; assert curved observation and anchor movement | Reference fixture, not a tuned default |
| Epsilon 1 and five Sinkhorn steps | Owner's development comparator | Poor balance; retain existing terminal balancing guard | Untuned development comparator |
| 32 terminal balance steps | Prior integration fixture repair, unchanged validity threshold | Treating a fixture repair as tuning; make no target claim | Reference-only setting |
| Fixed two-observation FP64 fixture and seeds | Existing guide tests; makes finite differences reproducible | Failing to exercise reset dependence; check both trace records | Diagnostic fixture |
| CPU, GPU hidden, non-JIT, two TF threads | Small independent finite differences and endpoint instrumentation | Confusing diagnostic with production evidence; record exception | Explicit reference/debug exception |
| No numerical algorithm changes | This task verifies the existing map | A failed check exposing a real defect | Repair only a localized defect within budget; otherwise stop and report |

## Execution and budget

1. Add executable nonlinear endpoint Jacobian and total-tangent checks.
2. Document conditional affinity, invertibility and the determinant derivative
   in proposition/proof form; request a bounded MathDevMCP audit.
3. Run the focused CPU reference suite and build/inspect the changed manuscript.
4. Record results, uncertainty, exact commands, environment, wall time and next
   justified action; update the stale integration checkpoint.

Allow 20 minutes of local CPU checks and at most three localized repair
attempts. No GPU job, guide training, new dependencies, commit, push or branch
change. Stop for a material mathematical mismatch, an incompatible shared
route, unrelated dirty overlap, or budget exhaustion. MathDevMCP formalization
failure is recorded as a limitation; it cannot replace numerical evidence.

Artifacts: `docs/plans/artifacts/ledh-backward-guide-jacobian-20261016/`.
Python: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`.
Set `CUDA_VISIBLE_DEVICES=-1`, `TF_NUM_INTRAOP_THREADS=2` and
`TF_NUM_INTEROP_THREADS=2` before import; run the guide test module with pytest.
Compile the existing main LaTeX document with `latexmk -pdf`, writing generated
files under the artifact directory. Preserve complete logs there. The installed
environment lacks latexmk; execution used installed pdflatex and bibtex instead,
without adding dependencies.

## Review status

Plan audit passed before execution: the actual consumer, nonlinear mechanism,
full parameter dependence, comparator, validity vetoes, bounded environment,
stop conditions and limits of inference are explicit. Execution is complete:
both new nonlinear endpoint checks and the full 37-test regression pass. The
64-page manuscript builds cleanly, and changed proof pages 57 and 58 were
visually inspected. No runtime algorithm was changed.

MathDevMCP's label audit is inconclusive for all ten obligations (seven not
encodable, three not extracted). The conditional derivation and executable
checks support the scoped engineering result; formal certification remains
open. See the [result](artifacts/ledh-backward-guide-jacobian-20261016/result.md)
and [run manifest](artifacts/ledh-backward-guide-jacobian-20261016/run-manifest.json).

Timed test/build commands used 193.17 seconds of the 20-minute check budget;
untimed bibliography, rendering and review are identified in the manifest.
One test-harness repair and one build invocation adaptation used two of three
allowed local repairs. The next research stage is a separate scope-specific
guide-fitting and untouched score-validation design. No score-improvement,
GPU/XLA, default, canonical or HMC claim follows from this continuation.
