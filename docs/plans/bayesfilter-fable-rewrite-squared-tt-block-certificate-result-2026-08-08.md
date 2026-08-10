# Squared-TT retained-block certificate execution result

- **Date:** 2026-08-08
- **Plan:** bayesfilter-fable-rewrite-squared-tt-block-certificate-execution-plan-2026-08-08.md
- **Git commit at execution:** 35ae61b984bfe2f9589d6c5f1eeae41c7ca14946
- **Execution target:** CPU-only exact arithmetic and documentation build; no GPU work was required.

## Outcome

The retained-prefix mathematical blocker is closed for the declared
fixed-branch contract. The prior Chapter 36b formula was correct only for a
scalar retained state and was wrong relative to its advertised vector-state
generalization. It now uses:

    G_m = H_1 ... H_m
    right contraction indices = D,...,m+1
    retained square = G_m M_{>m} G_m^T

It differentiates the retained prefix, suffix contraction, and quadratic form
on the same frozen branch.

## Exact certificate

Command:

    python docs/plans/artifacts/fable-rewrite-squared-tt-certificate-20260808/check_squared_tt_retained_block_certificate.py \
      --output docs/plans/artifacts/fable-rewrite-squared-tt-certificate-20260808/certificate.json

| Case | Direct value vs contraction | Direct derivative vs dotted contraction |
|---|---|---|
| m=1,D=2, ranks (1,2,1) | exact equality | exact equality |
| m=2,D=4, ranks (1,2,2,2,1) | exact equality | exact equality |

The checker uses two independent paths: direct multivariate-polynomial
expansion/integration and TT right-mass contraction. All arithmetic uses Python
Fraction; no floating tolerance is involved.

Artifact digests:

| Artifact | SHA-256 |
|---|---|
| checker | 662b916d1461753accdc46e2c3dd51c238637528225a77af1dbeb6bf11aaf782 |
| JSON result | da8c2dddde84e82a3ed20caa6d3feb9e02f79c84ab4844029138dcb4a27cb963 |

## Documentation changes

Chapter 36b now:

- distinguishes the scalar specialization from the vector retained-prefix
  formula;
- stops right contraction at m+1;
- states the complete frozen-derivative ledger;
- gives retained-prefix and mass-recursion derivatives;
- contains exact scalar and vector certificate coefficients.

Chapter 37 now:

- derives retained-prefix coefficient matrices A_t and dot A_t;
- derives Q_t and dot Q_t from A_t, M_{>m}, and the defensive retained
  coefficient matrix;
- states the defensive density's unit full reference mass;
- preserves reference-coordinate query and next-step Jacobian ownership;
- requires a separate defensive evaluator if its retained marginal is not
  exactly representable in the retained product basis.

The original scalar-only handoff was materially amended. The documentation
agent handoff records why a genuine m=2,D=4 certificate is required.

## MathDevMCP

Every added or materially changed derivation label in the repaired Chapter 36b
and Chapter 37 blocks was audited with source-bound audit-derivation-v2-label
and typed-obligation-label commands.

No mismatch was reported. Typed audits found no unresolved missing constraint.
Nine scalar-shadow or transcription obligations were certified as equivalent.
Matrix and integral labels mostly remained unverified or inconclusive because
the bounded backend requires manual formalization. These are recorded
abstentions, not proof.

See bayesfilter-fable-rewrite-squared-tt-mathdevmcp-audit-2026-08-08.md.

## Build result

Command:

    cd docs/fable-rewrite/monograph
    latexmk -pdf -interaction=nonstopmode -halt-on-error \
      -outdir=/tmp/bayesfilter-fable-squared-tt-build main.tex

Result:

| Check | Result |
|---|---|
| LaTeX exit | pass |
| PDF | 494 pages, 2,119,761 bytes |
| Undefined references | 0 |
| Undefined citations | 0 |
| Multiply defined affected labels | 0 |
| Overfull boxes | 191 |
| Underfull hboxes | 760 |
| Underfull vboxes | 9 |
| Built PDF SHA-256 | ea0b61e92a45ba0bdf834ee904e8f69a09970820e6489ba139f6d3a8915aa136 |

The two width warnings introduced by the first certificate layout were repaired
before the final build. Remaining layout warnings predate or lie outside this
bounded derivation repair.

## Decision table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Close the retained-prefix derivation blocker | pass: two exact value and derivative certificates | no veto fired | MathDevMCP abstains on general matrix/integral formalization | Treat Chapter 36b's mathematical identity as certified for the declared fixed branch | No fitting, implementation, posterior, HMC, or whole-book certification |
| Keep Chapter 37 implementation/source-route caveat | contract is internally consistent | implementation parity not checked | Actual fixed-branch route may encode measures or saved objects differently | Perform a separate code/source audit before implementation promotion | No runtime route admission |

## Inference status

| Row | Status |
|---|---|
| Hard veto screen | passed for the bounded exact derivation certificate |
| Statistically supported ranking | not applicable |
| Descriptive-only differences | not applicable |
| Default-readiness | not established |
| Next evidence needed | implementation/source parity for the Chapter 37 query and Jacobian ownership contract |

## Post-run red team

The strongest alternative explanation is that the exact fixtures certify the
algebra but not the production implementation. That explanation is correct and
is preserved as a nonclaim. A runtime route that uses a different coordinate
order, measure, defensive normalization, or parameter-dependent branch field
would overturn applicability of this certificate. The weakest evidence is
general machine formalization: MathDevMCP mostly abstains on matrix and integral
objects, so the general formula rests on the explicit derivation and the two
independent exact fixtures.
