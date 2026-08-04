# GenUT Score Note: MathDevMCP Audit Ledger

Date: 2026-07-31
Audited document:
`docs/bayesfilter-genut-score-variance-problem-and-repair-note-2026-07-31.tex`
(source digest at audit time:
`aecef9d617ea3ff100c659b07ca9e0fceb48cc06b1c21dc24054b69d402f3818`)
Companions:
`docs/plans/bayesfilter-genut-score-computation-audit-result-2026-07-30.md`,
`docs/plans/bayesfilter-genut-score-estimator-options-mathematical-note-2026-07-31.md`.

## What The Document Contains

A standalone proposition--proof note with two parts.

Part I (the problem): the executed-route tangent recursion and its
accumulation bound (`prop:gsv-tangent-recursion`,
`cor` accumulation); the increment/weight tangent forms
(`prop:gsv-increment-tangent`, `prop:gsv-weight-gain`); ridge/damping-limited
stage-gain bounds for the Contract-E affine map and the damped Gauss--Newton
correction (`prop:gsv-contract-e-gain`, `prop:gsv-gn-gain`); and the
replicated cubature residual-design defect: per-axis kurtosis `= d` and
pairwise co-kurtosis `= 0` versus Gaussian `3` and `1`
(`prop:gsv-cubature-design-moments`).

Part II (the repairs): the exactly whitened fixed Gaussian residual design
(`prop:gsv-whitened-design-identities`, `prop:gsv-iid-gaussian-moments`,
`prop:gsv-delta-method`, `prop:gsv-genut-positivity`); surrogate-force HMC
with the exact executed energy
(`prop:gsv-leapfrog-volume`, `prop:gsv-leapfrog-reversible`,
`prop:gsv-mh-invariance`, corollary); Fisher-identity/forward-smoothing score
estimators without transport derivatives (`prop:gsv-fisher-identity`,
`lem` backward kernel, `prop:gsv-backward-recursion`); and the
ESS-trigger branch-discontinuity warning (`prop:gsv-ess-branch`).

Lyapunov-sign, causal-attribution, and whitening-vs-studentization gap
statements are explicitly marked as hypotheses/remarks, not theorems, inside
the document.

## Evidence Contract For This Audit

| Item | Status |
|---|---|
| Question | Do the note's propositions parse, and does any bounded backend find an algebraic mismatch? |
| Primary criterion | Zero `mismatch` obligations across all label audits; SymPy certification of every scalar obligation submitted |
| Veto | Any `mismatch`, any SymPy `not equivalent`, LaTeX build failure |
| Explanatory | Abstention taxonomy (`manual_formalization_required`, `missing_assumption`, `source_label_missing`) |
| Nonclaims | Abstentions are not certifications; MathDevMCP does not certify matrix/measure-theoretic proofs |

## Verification Ledger

### LaTeX build

```text
cd docs && latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=/tmp/bayesfilter-gsv-note \
  bayesfilter-genut-score-variance-problem-and-repair-note-2026-07-31.tex
```

Result: pass; PDF written.

### MathDevMCP doctor

`PYTHONPATH=src python -m mathdevmcp.cli doctor`: pass (SymPy/Sage/Lean
backends available; known unrelated `magic-pdf`/pydantic conflict noted by
the doctor report, not affecting these checks).

### Bounded proposition audits (`audit-derivation-v2-label`)

All 15 proposition labels were audited against the note file. **No
obligation returned `mismatch` (0 of 66 extracted obligations).** Every
obligation abstained; abstention is a diagnostic boundary of the scalar
obligation backend, not a proof and not a refutation.

| Label | Status | Substatus counts | Obligations | Mismatch |
|---|---|---|---:|---:|
| `prop:gsv-tangent-recursion` | unverified | manual 3, missing-assumption 1, source-label 1 | 5 | 0 |
| `prop:gsv-increment-tangent` | unverified | manual 6, missing-assumption 1 | 7 | 0 |
| `prop:gsv-weight-gain` | unverified | manual 4, missing-assumption 2 | 6 | 0 |
| `prop:gsv-contract-e-gain` | unverified | missing-assumption 2, manual 3 | 5 | 0 |
| `prop:gsv-gn-gain` | unverified | source-label 1, missing-assumption 2 | 3 | 0 |
| `prop:gsv-cubature-design-moments` | unverified | manual 8 | 8 | 0 |
| `prop:gsv-whitened-design-identities` | unverified | missing-assumption 7 | 7 | 0 |
| `prop:gsv-iid-gaussian-moments` | unverified | manual 10 | 10 | 0 |
| `prop:gsv-delta-method` | unverified | manual 3 | 3 | 0 |
| `prop:gsv-genut-positivity` | inconclusive | source-label 3 | 3 | 0 |
| `prop:gsv-leapfrog-volume` | inconclusive | source-label 1 | 1 | 0 |
| `prop:gsv-leapfrog-reversible` | unverified | manual 1 | 1 | 0 |
| `prop:gsv-mh-invariance` | unverified | manual 2 | 2 | 0 |
| `prop:gsv-fisher-identity` | unverified | manual 2, source-label 1, missing-assumption 2 | 5 | 0 |
| `prop:gsv-backward-recursion` | unverified | manual 1 | 1 | 0 |
| `prop:gsv-ess-branch` | unverified | missing-assumption 1, manual 1 | 2 | 0 |

(The matrix, measure-theoretic, and inequality obligations are outside the
configured scalar backend; this reproduces the abstention pattern of the
2026-07-20 and 2026-07-23 chapter audits.)

### SymPy-certified scalar obligations (`check-proof-obligation --backend sympy`)

All 19 submitted obligations returned `equivalent` ("SymPy simplified
lhs - rhs to zero"). Note: `derive-step` was found to be a lexical
symbol-overlap heuristic, not a CAS check; its outputs were discarded and
`check-proof-obligation` used instead.

Proposition arithmetic:

| Obligation | Anchors | Status |
|---|---|---|
| `(2*M*d^2)/(2*d*M) = d` (design fourth moment) | `eq:gsv-cubature-fourth` | equivalent |
| `(2*M*d)/(2*d*M) = 1` (design variance) | `eq:gsv-cubature-first-second` | equivalent |
| Delta-method kurtosis quadratic form `= 24` | `eq:gsv-delta-kurt-arithmetic` | equivalent |
| Delta-method co-kurtosis quadratic form `= 4` | `eq:gsv-delta-cokurt-arithmetic` | equivalent |
| GenUT root: `u^2+su+s^2-k = 0` at `u=(-s+sqrt(4k-3s^2))/2` | `prop:gsv-genut-positivity` | equivalent |
| `1/(sqrt(3)*2*sqrt(3)) = 1/6` (axis weight) | same | equivalent |
| `1 - d*(1/6+1/6) = 1 - d/3` (central weight) | `eq:gsv-genut-central-weight` | equivalent |
| `105 - 9 = 96` (Var of `z^4`) | `eq:gsv-iid-moments` | equivalent |
| `15 - 3 = 12` (Cov(`m2`,`m4`)) | same | equivalent |
| `9 - 1 = 8` (Var of `z^2 w^2`) | same | equivalent |
| `3 - 1 = 2` (Cov(`m22`,`m2`); Var `m2`) | same | equivalent |

Proof-step algebra (scalar shadows of the displayed derivations):

| Obligation | Anchors | Status |
|---|---|---|
| Leapfrog reversal half-kick 1: `-(p0+(e/2)F0+(e/2)F1)+(e/2)F1 = -(p0+(e/2)F0)` | `prop:gsv-leapfrog-reversible` | equivalent |
| Leapfrog reversal drift telescoping | same | equivalent |
| Leapfrog reversal half-kick 2: `-(p0+(e/2)F0)+(e/2)F0+p0 = 0` | same | equivalent |
| logsumexp tangent (2-atom shadow) | `eq:gsv-increment-tangent` | equivalent |
| Softmax weight tangent (2-atom shadow) | `prop:gsv-weight-gain` | equivalent |
| Contract-E `dA` quotient algebra (scalar shadow, `d=1`) | `eq:gsv-affine-tangent` | equivalent |
| Damped-GN coefficient tangent (scalar shadow) | `eq:gsv-gn-tangent` | equivalent |

Scalar shadows certify the displayed algebraic manipulations in one
dimension; they do not certify matrix ordering, which remains covered by the
written proofs and (for the code-side objects) by the float64
forward-accumulator parity tests of the 2026-07-30 audit.

### Focused document-rigor audit (`audit-math-document-rigor`)

A full-document rigor pass over all 30 labeled display equations exceeded the
bounded runtime (tool-runtime limitation, recorded as such, not evidence
about the mathematics).  A focused pass over eight high-value equation
labels completed with reports at
`docs/plans/bayesfilter-genut-score-note-mathdevmcp-rigor-audit-2026-07-31.md`
/ `.json` (7 of the 8 focus labels selected by the tool's equation
selector).

Outcome: **0 mismatches; 3 gaps; 2 concrete exposition-repair proposals; 0
resolved-by-context.**

| Gap | Classification | Disposition |
|---|---|---|
| `eq:gsv-cubature-fourth` "formalization-and-source-role" | diagnostic abstention (the selector could not route definition-vs-derived-claim) | no change: the display is a derived claim proved in `prop:gsv-cubature-design-moments`; abstention recorded |
| `eq:gsv-affine-tangent` "matrix-domain-and-invertibility" | concrete repair: state invertibility of the displayed inverse operand | **applied**: proposition now states explicitly that positive definiteness of `L_E L_E^T` makes `L_E` nonsingular, so `L_E^{-1}` exists |
| `eq:gsv-gn-tangent` "matrix-domain-and-invertibility" | concrete repair: same class | **applied**: proposition now states explicitly that `M_a >= delta I` with `delta > 0` is symmetric positive definite, hence nonsingular |

Both invertibility conditions were already implied by the stated hypotheses;
the patches move them adjacent to the displayed inverses as the auditor
requested.  Post-patch source digest:
`6293ecb7d5b4fd20fd8e18a7fe5b271a1a5c42595592c6826ee3f763944f355b`; LaTeX
rebuild passes.  A v2 focused re-audit of the two patched labels was run
with reports at
`docs/plans/bayesfilter-genut-score-note-mathdevmcp-rigor-audit-v2-2026-07-31.md`
/ `.json`; its outcome row is below.

| Re-audit (v2) of patched labels | Outcome |
|---|---|
| `eq:gsv-affine-tangent`, `eq:gsv-gn-tangent` | Both gaps moved from open to `partially_resolved` after the invertibility patches (the auditor's context support now spans the patched statements). The residual obligation is the gap-class template's second clause: "the displayed Neumann series additionally requires a convergence condition." Neither display contains a series — `eq:gsv-affine-tangent` is the exact identity `Adot = (Ldot_w - A Ldot_E) L_E^{-1}` and `eq:gsv-gn-tangent` is the exact identity `cdot = rho M^{-1}(rdot - Mdot M^{-1} r)`, with the exact inverses' existence now stated in the propositions. The residual is classified as a checker-template limitation (inverse/series bundled in one obligation class), not a mathematical gap; recorded and not chased further per review-proportionality policy. |

## Decision

| Decision | Primary criterion | Veto status | Meaning |
|---|---|---|---|
| Note admitted as a derivation artifact | LaTeX pass; 0/66 bounded-obligation mismatches; 19/19 SymPy scalar certifications | no veto fired | The written proofs stand with machine-checked scalar cores; matrix/measure steps rest on the manual proofs |
| No proof-certificate claim | abstentions dominate the bounded audits | n/a | MathDevMCP abstention is not certification; nothing here upgrades the note's own hypothesis/remark labels |

## Nonclaims

The audits certify scalar algebra only. They do not certify the matrix
propositions, the measure-theoretic arguments, the delta-method CLT
hypotheses, or any empirical claim (Lyapunov sign, variance attribution,
repair effectiveness). Those remain, respectively, manually proved in the
note, standard-but-unformalized, and explicitly-marked hypotheses awaiting
the planned experiments.
