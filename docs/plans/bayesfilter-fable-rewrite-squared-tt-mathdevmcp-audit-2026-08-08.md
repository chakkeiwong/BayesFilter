# Squared-TT retained-block MathDevMCP audit ledger

- **Date:** 2026-08-08
- **CLI:** /home/chakwong/anaconda3/envs/tf-gpu/bin/mathdevmcp
- **Chapter 36b SHA-256:** 3b4d81531c5ec30bfdbe62656c31234a1731c273bcba16ff744734160a663c6a
- **Chapter 37 SHA-256:** 5f25e8672034091e67255974ff1f6b577d75dd8f2fae31a5fffb8a40f3c6cb9b
- **Policy:** MathDevMCP is an audit assistant, not an oracle. Unverified and
  inconclusive are abstentions unless the report identifies a mismatch.

## Audit contract

| Item | Declaration |
|---|---|
| Question | Do the repaired retained-prefix value, derivative, coefficient, normalization, and query formulas expose any MathDevMCP mismatch or missing dimensional constraint? |
| Primary mathematical evidence | Independent direct-polynomial versus right-contraction equality in exact rational arithmetic for m=1,D=2 and m=2,D=4. |
| MathDevMCP veto | Any mismatch, refutation, nonconformable product, or unresolved missing constraint. |
| Explanatory only | manual_formalization_required and source_label_missing abstentions. |
| Nonclaim | Label audits do not prove matrix/integral identities that the bounded backend declines to formalize. |

MathDevMCP doctor returned ok. SymPy 1.14.0, Sage 9.5, Lean 4.29.1, and
LeanDojo were available. Optional LeanExplore and Pantograph integrations were
unavailable and were not needed for this bounded audit.

## Commands

Every changed derivation label was bound to its exact file and SHA-256:

    mathdevmcp audit-derivation-v2-label \
      --root docs/fable-rewrite/monograph \
      --file CHAPTER_RELATIVE_PATH \
      --source-digest CHAPTER_SHA256 \
      --backend sympy --paragraph-context --summary-only LABEL

    mathdevmcp typed-obligation-label \
      --root docs/fable-rewrite/monograph \
      --file CHAPTER_RELATIVE_PATH \
      --source-digest CHAPTER_SHA256 \
      --backend sympy --context-text EXPLICIT_SHAPE_AND_BRANCH_CONTEXT LABEL

The first batch summarizer attempted to use unavailable jq, causing the CLI to
report BrokenPipeError after its output pipe closed. That attempt produced no
mathematical result. All audits were rerun to completion with a Ruby JSON
summarizer.

## Chapter 36b label audits

| Label | Derivation audit | Obligations | Mismatch | Typed audit |
|---|---|---:|---:|---|
| eq:bf-hd-squared-tt-mass-contraction | unverified: manual formalization | 1 | 0 | conformable product explicitly satisfied |
| eq:bf-hd-squared-tt-retained-prefix | unverified: manual formalization | 1 | 0 | consistent |
| eq:bf-hd-squared-tt-right-mass-recursion | unverified/inconclusive | 9 | 0 | consistent |
| eq:bf-hd-squared-tt-defensive-retained | unverified: manual formalization | 1 | 0 | consistent |
| eq:bf-hd-squared-tt-retained-numerator-contraction | unverified: manual formalization | 1 | 0 | conformable product explicitly satisfied |
| eq:bf-hd-squared-tt-frozen-derivative-ledger | unverified: manual formalization | 5 | 0 | consistent |
| eq:bf-hd-squared-tt-dot-retained-prefix | unverified: manual formalization | 1 | 0 | consistent |
| eq:bf-hd-squared-tt-dot-right-mass-recursion | unverified: manual formalization | 1 | 0 | consistent |
| eq:bf-hd-squared-tt-dot-retained-numerator | unverified: manual formalization | 1 | 0 | conformable product explicitly satisfied |
| eq:bf-hd-squared-tt-scalar-retained-certificate | unverified: manual formalization | 1 | 0 | conformable product explicitly satisfied |
| eq:bf-hd-squared-tt-scalar-exact-certificate | unverified: manual formalization | 2 | 0 | consistent |
| eq:bf-hd-squared-tt-vector-exact-certificate | inconclusive: source-label splitting | 2 | 0 | consistent |

The source_label_missing substatus on aligned or multirow displays means the
row splitter did not extract a safe scalar proof obligation. It is not negative
evidence about the displayed equality.

## Chapter 37 label audits

| Label | Derivation audit | Obligations | Mismatch | Typed audit |
|---|---|---:|---:|---|
| eq:bf-hd-ttkr-mass-recursion | unverified: manual formalization | 7 | 0 | consistent |
| eq:bf-hd-ttkr-dot-mass-recursion | unverified/inconclusive | 18 | 0 | consistent |
| eq:bf-hd-ttkr-normalizer-block | unverified: manual formalization | 2 | 0 | consistent |
| eq:bf-hd-ttkr-dot-normalizer | unverified: manual formalization | 1 | 0 | consistent |
| eq:bf-hd-ttkr-carried-numerator | inconclusive: source-label splitting | 2 | 0 | consistent |
| eq:bf-hd-ttkr-dot-carried-filter | unverified: manual formalization | 1 | 0 | consistent |
| eq:bf-hd-ttkr-retained-product-basis | unverified/inconclusive | 2 | 0 | consistent |
| eq:bf-hd-ttkr-retained-prefix-coefficients | unverified: manual formalization | 2 | 0 | conformable product explicitly satisfied |
| eq:bf-hd-ttkr-defensive-retained-coefficients | unverified: manual formalization | 1 | 0 | consistent |
| eq:bf-hd-ttkr-retained-Q-construction | inconclusive: source-label splitting | 2 | 0 | consistent |
| eq:bf-hd-ttkr-retained-Q | unverified: label context reports missing shape | 2 | 0 | conformable product explicitly satisfied |
| eq:bf-hd-ttkr-retained-P | unverified/inconclusive | 2 | 0 | consistent |
| eq:bf-hd-ttkr-retained-query-basis | unverified: label context reports missing shape | 1 | 0 | conformable product explicitly satisfied |
| eq:bf-hd-ttkr-retained-query-rule | inconclusive: source-label splitting | 2 | 0 | consistent |

The first Chapter 37 audit exposed omitted adjacent assumptions: the dimensions
of Q_t and dot Q_t, and unit full mass of the defensive density. Those
assumptions were added and the complete audit was rerun against the new digest.
The label auditor continued to report missing shape on two transpose-bearing
rows even though the repaired source states the dimensions immediately above
and the typed route reports each product conformable with no missing constraint.
This residual is a label-context limitation, not verified algebra and not an
unresolved shape defect.

## Scalar-shadow proof obligations

The check-proof-obligation command with the SymPy backend certified:

| Obligation | Status |
|---|---|
| derivative of g m g | equivalent |
| derivative of the right-mass scalar shadow c m c | equivalent |
| frozen-scale retained derivative | equivalent |
| derivative of phi squared | equivalent |
| quotient derivative, under nonzero denominator | equivalent |
| derivative of the Q_t construction scalar shadow | equivalent |
| fixed defensive derivative after literal substitution of all frozen dots by zero | equivalent |
| scalar certificate value polynomial transcription | equivalent |
| scalar certificate derivative polynomial transcription | equivalent |

The first attempt to express frozen derivatives as assumptions such as dc=0
returned unverified because this command does not substitute equality
assumptions. The exact same obligation with frozen derivatives substituted
literally was certified. One negative-leading command initially parsed as a CLI
option and was rerun successfully with the end-of-options delimiter.

## Verdict

mathdevmcp_status: NARROW_SUPPORT_NO_MISMATCH

No MathDevMCP route reported a mathematical mismatch, refutation,
nonconformable product, or unresolved typed constraint. Scalar algebra was
certified narrowly. Matrix and integral claims remain supported by the written
derivation plus the independent exact certificate, not by a false
MathDevMCP-proof claim.
