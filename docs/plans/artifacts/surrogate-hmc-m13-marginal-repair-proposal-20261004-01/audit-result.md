# M13 marginal-correction proposal: mathematical audit result

Date: 2026-10-04. Decision: the proposal is ready for the requested Claude
mathematical review and a separately identified implementation experiment.
It is not an implemented fix or a promoted algorithm. The campaign goal
remains testing surrogate HMC, including its actual endpoint and posterior
sampling; improving the upstream particle score is one repair phase.

The proposal retains the current UKF/LEDH flow and replaces the single-ancestor
importance correction with the marginal predictive/proposal mixture ratio.
Its conditional integration identity and analytical finite-program derivative
are valid under the stated full-rank, invertibility, sampling-law and
differentiability assumptions. The random-label variance theorem is valid,
but does not guarantee improvement for the current stratified/frozen design.
An explicit counterexample proves why that stronger claim would be wrong.

LGSSM and KSC-SV protection is by retaining their existing post-fix program
and then testing the actual consumers against frozen outputs and independent
references. The new correction is optional and scope-selected before tracing.
This stage edits documentation and diagnostic algebra only. New algorithmic
consumer regression is not claimed to have run.

## Evidence and exact scope

Primary proposal:
../../surrogate-hmc-m13-marginal-repair-proposal-2026-10-04.tex

Rendered PDF:
surrogate-hmc-m13-marginal-repair-proposal-2026-10-04.pdf

Claude memo:
../../surrogate-hmc-m13-marginal-repair-claude-handoff-2026-10-04.md

Existing empirical evidence comes from the October 4 M13 likelihood-gap
localization and score-before/after artifacts, not a new filter experiment.
At the low-Q T=1 point the current LEDH score is the derivative of a likelihood
whose importance approximation is inaccurate. The same-design bootstrap is
closer to quadrature at all eight diagnosed points, descriptively; no
replicated stochastic ranking is established.

## Checked mathematical claims

| Claim | Evidence | Qualification |
|---|---|---|
| Marginal weights integrate the finite predictive mixture | Explicit finite sum/integral proof, MPF primary source | Conditional on the declared randomized sampling law. Not a fixed-design, log-likelihood, score or full-recursion unbiasedness theorem. |
| Random-label variance cannot increase | Conditional expectation and total variance proof; two MathDev symbolic identities | Different experiment from one child per ancestor. |
| General stratified non-deterioration is false | Two-state exact counterexample; rational mean and variance checks | New variance 555264/28826161 > 0; old variance zero. Not evidence the M13 candidate fails. |
| Mixture and weight scores include the total parameter derivative | Explicit finite chain rules; rational mixture identity | Incoming particle/weight/UKF tangents must already be correct and all reset/correction paths retained. |
| Gaussian and inverse-map derivatives are correct | Symbolic scalar identities, 2×2 Gaussian coefficient checks, independent local noncommuting-matrix algebra | Does not certify an unimplemented TensorFlow route. |
| Defensive mixture has the displayed second-moment bound | Direct pointwise inequality; checked variance decomposition | Includes (1/alpha-1) Z_N^2. No general non-harm or score-accuracy guarantee. |
| Identical legacy execution preserves values and returned scores | Proposition by function/program identity | Conditional implementation obligation; includes pre-existing errors. Consumer-level tests still required after code changes. |

## MathDevMCP evidence

Eleven original checks returned equivalent:
two-component second-moment identity; conditional-expectation identity;
mixture total derivative; scalar Gaussian tangent; defensive bound
rearrangement; counterexample variance and mean; both constant old
component weights; scalar inverse-map tangent; KSC observation tangent.

Five additional checks compared independently expressed coefficients of
the full 2×2 Gaussian total derivative (da, db, dc, dea, deb). Each returned
equivalent with “SymPy simplified lhs - rhs to zero.” The input expressions
are in gaussian-2x2-mathdev-coefficients-unreduced.json. The locally derived
expressions used symbolic differentiation on one side and the proposed
matrix formula on the other. Earlier checks of already reduced coefficient
expressions are also preserved, but are not counted again as independent
evidence.

Raw outputs: mathdev-symbolic-raw.json and mathdev-supplemental-raw.json.
The reproducible local diagnostic is verify_algebra.py. Its
local-algebra-report.json records nine passing checks, including a full
symmetric 2×2 covariance derivative, genuinely noncommuting affine maps,
composition derivatives, inverse differentiation and exact counterexample
arithmetic. It used CPU-only SymPy/Python fractions, with no TensorFlow or
CUDA import, and took approximately 0.62 seconds.

Tool limitations are part of the record:

- A nested diff/log/exp expression was misparsed; the response reported a
  mismatch plus a non-encodable backend expression. This was not a valid
  mathematical counterexample. The explicit rational derivative formulation
  passed.
- The full 2×2 expression exceeded the tool's 500-character limit and returned
  inconclusive. Splitting the linear directional derivative into its five
  coefficients produced bounded checks. The full unsplit equality also
  passed the independent local symbolic diagnostic.
- The high-level request to refute universal stratified non-deterioration
  returned inconclusive even though its routed backend found the exact
  nonzero difference. The wrapper lacked its required counterexample
  artifact. We preserve that status and supply the explicit construction in
  stratified-counterexample.json; the mathematical refutation does not rely
  on relabeling the wrapper as successful.
- Both document-rigor calls timed out at the MCP boundary after 300 seconds,
  but wrote inspectable JSON/Markdown reports first. Those reports are
  partial document audits, not proofs. No Lean certificate was generated.

## Document-audit findings and disposition

Initial report: mathdev-document-rigor.json / .md; 11 selected equation rows
out of 23, five distinct findings. The requested proposition labels were not
treated as equation rows. Narrowed revision: mathdev-revision-rigor.json / .md;
three selected rows, three findings. Both original reports remain intact.

| Finding | Action and present interpretation |
|---|---|
| eq:affine-density / matrix-domain-and-invertibility | Added explicit d-dimensional vectors, d×d matrices, symmetric positive-definite Q and nonsingular B in a parameter neighborhood. These are sufficient for the stated affine change of variables. |
| eq:inverse-tangent / matrix-domain-and-invertibility | Added explicit dimensions and local nonsingularity immediately before the equation. Also checked noncommuting 2×2 inverse differentiation independently. |
| eq:gaussian-score / matrix-domain-and-invertibility | Added explicit symmetric positive-definite Q, symmetric DQ and vector dimensions beside the derivation. Full 2×2 symbolic equality passes. |
| eq:integration-proof / formalization-and-source-role | The extractor reported an unsplit or incomplete obligation. The manual proof explicitly sums conditional expectations and cancels the mixture density. Retain the tool's formalization abstention; no claim of machine-certified integration. |
| eq:rb-proof / formalization-and-source-role | The extractor could not certify the integral/probability argument and requested human review. The proof states the conditional probabilities, conditional mean and law of total variance. The two-component algebra is separately checked. Claude is asked to audit this scope. |

The narrowed audit still reported “dimension_contract” despite the explicit
matrix and vector declarations in its inspected source. This is an unresolved
recognition limitation in that report, not a missing assumption we silently
waived. The source declarations and multivariate checks justify the manual
disposition; the tool has not certified closure. Preserve that distinction.

proposal-audited-v1.tex and proposal-audited-v2.tex preserve the exact audited
drafts. The final version additionally removes a duplicated assumption,
adds the Li–Coates citation, breaks a long defensive-mixture equation across
lines, and documents the multivariate checks. Those changes do not alter
the mathematical formulas or theorem scope. Source digests and the final
build are recorded in the run manifest.

## Skeptical implementation and default audit

| Choice | Provenance / status | Failure mode and earliest discriminating check |
|---|---|---|
| Keep canonical UKF, flow, Contract E, correction | Existing baseline, not a universal accuracy claim | Reusing the same map may leave proposal coverage poor; T=1 exact-mixture normalizer and quadrature comparison discriminate. |
| Marginal weights | MPF construction; new M13 hypothesis | May worsen stratified integration, cost N², or expose poor prior cloud; counterexample limits theory, then compare bootstrap/current LEDH/quadrature. |
| One child per ancestor with leading omega_i | Matches existing allocation; explicit sampling-law choice | Using iid-label factors here or double-weighting labels changes target; normalized toy mixtures test it. |
| Positive-definite Q and nonsingular B | Change-of-variables prerequisites | Structural singular models or invalid flow maps are unsupported; fail closed, no hidden ridge. |
| Exact tiled mixture evaluation | Mathematical proposal plus existing chunk policy | Nearest-neighbor approximation or wrong mixture denominator changes method; dense/tiled and batch/scalar equality first. |
| Optional defensive alpha | Untuned hypothesis, not an active default | Can worsen a good proposal; evaluate actual-mixture sampling and non-harm before inclusion. No value selected here. |
| Preserve existing LGSSM/KSC mode | Explicit protection mechanism | Shared-helper refactor or RNG/reduction changes break identity; compare actual consumer outputs before M13 admission. |
| FP32 GPU/XLA | Existing owner default | Cancellation, device precision and resource cost; FP64 reference plus actual GPU consumer checks and memory-growth records. No GPU run in this stage. |
| T=1 diagnostic reference | Refined quadrature from previous stage | Does not qualify full T=20 score; independently refine/replicate that reference before full-horizon decisions. |

No approximate mixture truncation, dropped derivative, weight clipping,
UKF-to-Q substitution, parameter-dependent switching, unreviewed alpha,
new tolerance, or fixed terminal HMC count is introduced.

## Decision table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Deliver mathematical proposal and Claude handoff | Conditional derivations complete; scoped algebra passes | No remaining identified mathematical contradiction under stated assumptions; document automation remains partial | Whether the actual M13 fixed design improves | Review the proposal, then implement the optional exact correction with preserved legacy consumers | No empirical fix or all-model completion |
| Retain current LGSSM/KSC behavior | Preservation theorem stated; runtime source unchanged during this stage | Future shared-path changes must pass consumer parity/reference checks | Future implementation effects have not been measured | Freeze post-fix baseline and run those checks before candidate admission | No completed new regression campaign |
| Keep marginal/defensive candidates unpromoted | No new filter run | Universal improvement claim explicitly refuted | Stratified error, full-horizon reference score and runtime | First discriminating T=1 comparison; defensive arm only if needed | No default-readiness or performance ranking |

## Inference-status table

| Item | Status |
|---|---|
| Hard veto screen | Wrong universal variance claim rejected; parser/timeout limitations retained. No candidate implementation exists to pass numerical validity or posterior vetoes. |
| Statistically supported ranking | None established in this stage. |
| Descriptive-only differences | Prior eight-point LEDH/bootstrap/reference comparisons explain the repair mechanism only. |
| Default-readiness | Not established; existing LGSSM/KSC and FP32 defaults retained. |
| Next evidence needed | Actual consumer preservation, end-to-end total-score parity, T=1 value/score accuracy, qualified T=20 score reference, fresh tuned holdout, downstream surrogate-HMC evidence. |

## Post-run red team

The strongest alternative explanation is that the flowed mixture itself has
poor coverage, so marginalization fixes label variance but not the important
integration error. A second is that the previous cloud already misses relevant
mass. Either can survive a correct mathematical derivation. Failure to improve
T=1 normalization and both score coordinates against a refined reference would
weaken this candidate. Good normalization with a bad score would also block
promotion; it would not prove the integration problem solved.

The weakest evidence is empirical: no new candidate has been run, and the
full T=20 reference score remains approximate. Local algebra cannot overcome
that gap. The proposed optional correction is viable for testing, not proven
superior.

The PDF was built twice with the installed pdflatex; the final build reports
zero warnings. Runtime-source preservation, render inspection and exact hashes
are recorded in the final manifest/checkpoint. The original diagnostic budget
balance of 1,575.90 seconds is reserved; no GPU/HMC sampling time was consumed.
