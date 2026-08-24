# P1B attempt04 Result — r*(2)=6 confirmed; n=4 cells invalidated by the plan's own flag — 2026-08-22

Plan: `bayesfilter-p1b-attempt04-plan-2026-08-21.md`.
Artifact: `docs/benchmarks/artifacts/p1b_lgssm_value_ladder_20260817/attempt04/result.json`
(+ accumulator rows). Engine: adapted triangular XLA (parity 1.3e-15),
truncation correction on, kappas (4, 3), Sobol 8192, 3 seeds.
Infrastructure notes: in-process battery died at cell 7 (LLVM section
memory; the ARM-H lesson now enforced by per-cell subprocesses);
n=2 r=10 and n=4 r=10 hit compile-time timeouts (rank-10 9-axis XLA
graphs; guard fired as designed); skip-established-rank refinement
recorded in the driver.

## r*(2) = 6 CONFIRMED under the valid design

All n=2 r=6 and r=8 cells pass with margins 3-8x (per-step 3.2e-4 ..
8.6e-4) and truncation ratios <= 0.11. The n=2 rung is closed and
robust.

## n=4: NOT valid rank evidence — the pre-declared flag fired

| cell | per-step | fit rms | truncation ratio M_out/M_in |
|---|---|---|---|
| r=6 s=42/142/242 | 0.451 / 0.466 / 0.364 | ~0.20 | 0.36 / 1.57 / 0.79 |
| r=8 s=42/142/242 | 0.340 / 0.386 / 0.365 | ~0.16 | 1.14 / 1.85 / 0.44 |
| r=10 | timeout (compile guard) | - | - |

r_star(4) = null, but the decisive reading is the TRUNCATION RATIO
column: in 5 of 6 cells the correction contributes >= 36% of the
increment and in 3 cells MORE mass lies outside the box than inside
(ratio > 1). Per the plan's pre-declared rule these cells are
flagged (map mis-sized) and are NOT rank evidence. The "rank
saturation -> degree arm" branch does NOT fire: the flag rule
supersedes it.

## Mechanism: the containment cap is dimension-lethal

The containment fixed point (design note Section 11) caps the
previous-block box at ~2 effective sigma. At n=2 that truncates
2-11% of step mass (ratios 0.03-0.11; correction handles it; 46x
margins). At n=4 the SAME cap truncates 26-65% (the tail fraction
outside a ~2-sigma box grows geometrically in dimension). The scalar
correction rescues the likelihood VALUE (0.34-0.47 instead of 2.5),
but (a) it now carries the increment with plain-MC accuracy, and
(b) the retained-SHAPE box-conditioning — declared second-order in
the design note — is FIRST-order at n=4 and compounds through the
recursion. No rank or degree can repair a retained object that
represents a minority slice of the posterior.

## Decision table

| item | status |
|---|---|
| decision | r*(2)=6 CONFIRMED; n=4 cells invalid as rank evidence (flag rule); bounded-box program classified INSUFFICIENT at n>=4 |
| primary criterion | n=2: passed (3 seeds, both ranks run); n=4: flag rule pre-empts |
| veto/flag diagnostics | truncation-ratio flag fired exactly as designed; r=10 compile timeouts recorded (resource, not scientific) |
| main uncertainty | whether shape-corrected bounded retention could rescue n=4 without the reference change (judged unlikely: three bounded-box repairs have hit the same wall in different guises) |
| next justified action | OWNER DECISION: Gaussian-reference program variant (the paper's own unbounded construction — no boxes, no chaining, no truncation) vs stopping the n>=4 line at this recorded boundary |
| not concluded | any n=4 rank statement; any claim the affine/triangular machinery is wasted (it is prerequisite for the Gaussian-reference variant too — maps ARE the preconditioner there) |

## Inference status

| row | status |
|---|---|
| hard veto screen | n=2 passed; n=4 flagged (correction dominance) |
| statistically supported ranking | none claimed |
| descriptive-only | all continuous gaps/ratios/walls |
| default-readiness | adapted engine remains an option, not a default |
| next evidence needed | owner fork; if Gaussian-reference: derivation note first (bases, mass matrices, Gram chains under Hermite/weighted reference) |

## Post-run red team

Strongest alternative explanation: kappas are simply mis-tuned and a
per-scope tuned (kappa_c, kappa_p) could push ratios under 0.5 at
n=4 — but the Section 11 fixed point argues the cap is
kappa-invariant, and the wide-kappa grid measured exactly that.
What would overturn the "bounded-box insufficient" classification:
any bounded-box configuration achieving ratio < 0.2 at n=4.
Weakest evidence: single fixture family; exact-hint moments (real
hints would be worse, strengthening, not weakening, the conclusion).
