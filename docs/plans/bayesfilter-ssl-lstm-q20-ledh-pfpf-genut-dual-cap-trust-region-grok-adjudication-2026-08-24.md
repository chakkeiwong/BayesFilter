# Adjudication of Grok and Fable reviews: LEDH-PFPF-GenUT dual-cap note

Date: 2026-08-24  
Status: `BOTH_REVIEWS_RECEIVED_WRITTEN_CLAIMS_PASS_IMPLEMENTATION_ALIGNMENT_PENDING`

## Inputs

- [Grok review](./bayesfilter-ssl-lstm-q20-ledh-pfpf-genut-dual-cap-trust-region-grok-review-2026-08-24.md)
- [Literature/solution plan](./bayesfilter-ssl-lstm-q20-ledh-pfpf-genut-dual-cap-trust-region-literature-solution-plan-2026-08-24.md)
- [Mathematical note](./bayesfilter-ssl-lstm-q20-ledh-pfpf-genut-dual-cap-trust-region-mathematical-note-2026-08-24.md)
- [MathDevMCP audit](./bayesfilter-ssl-lstm-q20-ledh-pfpf-genut-dual-cap-trust-region-mathdevmcp-audit-2026-08-24.md)
- [Fable review](./bayesfilter-ssl-lstm-q20-ledh-pfpf-genut-dual-cap-trust-region-fable-review-reply-2026-08-24.md)
- [Fable handoff](./bayesfilter-ssl-lstm-q20-ledh-pfpf-genut-dual-cap-trust-region-fable-handoff-2026-08-24.md)

## Adjudication

Grok and Fable independently give `AGREE` verdicts for both the written plan
and the written mathematical note. Their agreement is accepted for the
bounded documentary claims: the conditional/no-go argument is coherent, and
neither review found a blocking or major defect.

One sentence must not be promoted: Grok says that "the current code separates
these correctly," while also stating that the review was written-claims-only,
not an implementation-alignment review. That code-level conclusion is therefore
`unsupported` by this review. It remains a hypothesis supported only by the
local source inspection and the separate MathDevMCP code-path screen, whose
equation obligations were `not_checkable`. A dedicated implementation-alignment
audit is still required before any density or HMC claim.

Likewise, Grok's statement that "no further artifacts are required from the
two paths" is read narrowly: no further artifact was required for Grok's
bounded review. Fable's reply makes the same boundary explicit. Neither
statement cancels the density fixtures, replay checks, two-mode validation, or
downstream NeuTra/HMC gates required by the plan.

Fable reports no blocking or major findings. It records two minor derivational
clarifications (the whitening-factor identity in Proposition 2 and the
differentiation/interchange assumption after Proposition 5) and three
editorial repairs (the Neal attribution, a grammar correction, and a possible
strengthening of Proposition 3's injectivity wording). These do not change the
theorems or the no-go conclusion. They are recorded as documentation cleanup
for any later implementation-phase revision; changing the note would require
rerunning its audit and refreshing its checksums.

Both reviews inspected source anchors to evaluate written claims, but neither
is an implementation-alignment or runtime-validity certification. In
particular, Fable identifies three remaining evidence gaps: the per-proposal
density identity, the defensive-component tail second moment, and replay
metadata parity.

## Decision table

| Decision | Primary criterion | Veto status | Interpretation | Next action | Nonclaim |
|---|---|---|---|---|---|
| Accept Grok's plan verdict | Written plan satisfies its declared evidence contract and skeptical audit | No written-plan contradiction found | `AGREE` is valid for the bounded plan review | Preserve verdict and await Fable | No implementation approval |
| Accept Grok's mathematical verdict | Propositions and source boundaries agree with the note and MathDevMCP's limited result | No algebraic mismatch found; typed semantic rows remain abstentions | `AGREE` is valid for the written note | Cross-review with Fable | No proof that runtime code satisfies every hypothesis |
| Accept Fable's plan verdict | Independent bounded review finds the plan's ledger, evidence contract, and stop rules coherent | No blocking or major finding | `AGREE` is valid for the bounded plan review | Preserve verdict | No implementation approval |
| Accept Fable's mathematical verdict | Independent rederivation supports the stated propositions and source boundaries | No blocking or major finding; three minor/editorial repairs recorded | `AGREE` is valid for the written note | Apply optional cleanup only in a separately audited revision | No proof that runtime code satisfies every hypothesis |
| Current code density alignment | Review explicitly excluded implementation alignment | Veto remains active for promotion | Grok's code-correctness parenthetical is unsupported | Run a separate bounded code/equation audit | No claim that current PF-PF/reset density is exact |
| NeuTra/HMC admission | Required downstream gates are not run | Promotion blocked | Document review alone is insufficient | New scoped validation plan after reviewer adjudication | No whitening, posterior, or HMC claim |
| Documentary reviewer agreement | Grok and Fable agree on the written conditional/no-go claims | Runtime evidence remains absent | Scoped documentary consensus only | Start with the cheapest implementation-phase artifact | No scientific or default-readiness claim |

## Inference status

| Evidence class | Status |
|---|---|
| Hard veto evidence | No written mathematical contradiction; implementation alignment is still unevaluated |
| Statistically supported ranking | None; no stochastic comparison was performed |
| Descriptive differences | None promoted; Grok's verdict is documentary evidence only |
| Default-readiness | Not ready; dual-cap remains a candidate component |
| Next evidence needed | Bounded code/density fixtures, replay-mixture validation, defensive-tail check, replay-metadata parity, and independent two-mode checks |

## Disposition

Record both Grok and Fable as independent written-claims `AGREE` reviews,
retain the explicit implementation-alignment caveat, and do not modify the
runtime route. The documentary review phase is complete. Before any density,
replay, or NeuTra/HMC work, write a separate implementation-phase experiment
plan beginning with Fable's cheapest discriminating artifact: a per-proposal
density record plus an affine known-map identity test.
