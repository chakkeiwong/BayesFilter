# Phase 7 Result: Contract E LaTeX Reconciliation

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Status:
`NORMATIVE_DOCUMENTATION_RECONCILED_AND_MONOGRAPH_BUILT_SCIENTIFIC_PROMOTION_BLOCKED`

## Outcome

The two named chapters now describe the same canonical finite Contract E--Chol
program and total derivative as the Phase 1 normative specification and checked
owned TensorFlow modules. The repair covers transition-first timing, normalized
weights, population moments, mandatory row quotient, fixed realized residual
design, fixed prepared ridge, row-oriented Cholesky affine restoration,
direct-plus-transport pullback, probability/log-weight coordinate conversion,
active/inactive reset carry semantics, and historical-only raw routes.

The closeout audit found and repaired two notation defects before certification:
the Contract E paragraph had relied on earlier column-oriented moment symbols,
and the custom-gradient API used `rho` for OT settings although canonical
Contract E freezes residual amplitude `rho=1`. The canonical equations are now
self-contained in row orientation, and the OT settings bundle is distinct.

The full 387-page monograph builds successfully. Existing duplicate-label and
bibliography warnings remain as pre-existing monograph debt; no Phase 7 label
collides and Phase 7 added no citation.

## Evidence

- Full build:
  `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`, exit `0`.
- PDF: 387 pages, 1,532,934 bytes.
- PDF SHA-256:
  `aa72610b424a749ee98e93ca935cdd559388f507b0099ca92e4ffdb86bf38df6`.
- Scoped `git diff --check`: passed.
- Seven new labels: each occurs exactly once.
- Code-anchor and contradiction audit: `PASS`.
- Canonical chapter SHA-256 values and owned-source hashes are recorded in the
  Phase 7 run manifest.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close documentation reconciliation | Every required semantic item agrees with spec/code and full monograph builds | Passed | General numerical behavior is outside this phase | Freeze Phase 8 statistical design | Numerical or scientific validity |
| Treat raw routes as canonical | Forbidden | Historical-only language explicit; no fallback | Unclassified future consumers | Retain fail-closed policy | Raw target correctness |
| Start final Phase 8 multi-seed oracle run | Blocked pending margin decision and design freeze | Equivalence margin remains unresolved | Practical HMC gradient error budget | Obtain owner decision or defensible derivation before observing final results | Kalman equivalence |

## Inference Status

| Inference | Status |
| --- | --- |
| Hard veto screen | Documentation/code contradictions and LaTeX integration passed |
| Statistically supported ranking | None; no stochastic experiment was run |
| Descriptive-only differences | Build warning counts and document size |
| Default-readiness | Not established |
| Next evidence needed | A pre-result Phase 8 design with an explicit, justified Kalman-gradient margin |

## Ledgers

| Ledger | Verdict |
| --- | --- |
| Engineering correctness | Documentation and owned-code anchors agree for checked semantics |
| Numerical validity | Not evaluated in Phase 7 |
| Scientific interpretation | No Kalman, nonlinear, HMC, or leaderboard claim is supported yet |

## Warning Classification

The build reports four duplicate label names and eleven unresolved citation
occurrences. All four duplicate names were already duplicated in `HEAD`. The
unresolved keys are absent from the `HEAD` bibliography, and the Phase 7 diff
adds no citation. These warnings do not invalidate the Phase 7 chapter changes,
but this result does not claim the entire monograph is warning-free.

## Unresolved Blockers

- general row/Sinkhorn/chunk and ridge/reset/conditioning adequacy;
- full `T=50,N=10000` canonical GPU/XLA/TF32 filter feasibility;
- exact Kalman value and gradient equivalence;
- a scientifically justified Phase 8 gradient-equivalence margin;
- nonlinear Contract E correctness and state-support safety;
- production v2 registration/admission;
- HMC, leaderboard, default, and release readiness; and
- the separate post-run integrity audit.

## Phase 8 Handoff

Phase 8 planning may proceed. Material final oracle execution may not begin
until the estimand, pairing, interval, simultaneous component rule, near-zero
rule, seed/precision plan, and equivalence margin are fixed before results. The
historical `1%` criterion and `0.05*sqrt(p)` FD screen are not eligible defaults
for Kalman-gradient equivalence.

## Post-Run Red Team

The strongest alternative explanation is that prose can agree with code while
both implement an inadequate finite approximation. That is exactly why this
phase claims documentation consistency only. A checked code change that alters
the finite target, an undiscovered contradictory canonical statement, or a new
label/reference failure in the edited chapters would overturn this closeout.
