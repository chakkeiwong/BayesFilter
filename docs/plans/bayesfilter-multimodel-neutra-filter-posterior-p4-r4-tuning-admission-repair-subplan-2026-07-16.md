# P4 R4 Repair Subplan: Predator-Prey Tuning Admission

Date: 2026-07-16

Program ID: `multimodel-neutra-filter-posterior-20260715`

Status: `COMPLETE`

## Objective And Entry Conditions

Repair the exact tuning-admission gap found by P7 for `PP-UKF` and `PP-SGQF`,
then run fresh target-bound NeuTra warm-up and retained confirmation for each
cell. The repair starts at R4 tuning admission; target identity, frozen
transport, same-target comparator, physical estimands, agreement margins,
kernel grid, leapfrog count, final thresholds, hardware class, and total P4
budget do not change.

Entry evidence:

| Cell | Typed target | Frozen transport | Comparator | Probe-source result |
| --- | --- | --- | --- | --- |
| `PP-UKF` | `036948f0faaf028d159d7b70337214f01514d732112c2d10e9f7eea1e13b8e30` | `18546c2b30a5e2236e001293f9bbfc71babed47f5592d6821cabe0972990beec` | `4c7e001b181033f4191acf5a6dd841c2dc507c4b25c015ce69817976eec345d5` | `c69117dcdc378623f054742954bf43bfa9af60a3601b0df548960369b9433375` |
| `PP-SGQF` | `8e0a9582fd30643b2e77e7615a21c0d44cc6c1827865ea52c841cc6dbfdde1ad` | `603a07c420579788e3981aa44dd67892902dc8c32da6ddf7c171918300da6811` | `015348e162d35cb062be274eb4b420ee881eb364473b5b7ce5acfdca7c0192ec` | `2fb38a2e5727ab486de8f5840c50881331cdcbd0352b97710260d5e72f4fe50e` |

The complete recursive ledgers for both probe-source roots must verify before
their six short-probe rows are reused. Reuse is ordering-only and resumes from
the earliest invalid rung; old warm-up and retained samples remain historical
and are not pooled or relabeled.

## Research Intent And Evidence Contract

| Field | Frozen repair contract |
| --- | --- |
| Question | After valid disjoint tuning admission, do the unchanged frozen P4 transports support converged, health-valid HMC whose six physical posterior means agree with their same-target comparators? |
| Baseline | the cell's admitted same-target plain-HMC retained archive |
| Candidate | exact target pulled back through the existing frozen target-specific dense IAF |
| Tuning ordering | hash-verified old short probes ordered by lowest modern R-hat, then highest minimum bulk ESS, then grid order |
| Tuning admission | first candidate passing fresh disjoint 1,000 burn-in plus 1,000 draws, modern R-hat `<=1.01`, finite health/status, and zero declared energy divergences |
| Final sampler pass | warm-up recent-window modern R-hat `<=1.05`; retained modern R-hat `<=1.01`; bulk ESS `>=1000`; tail ESS `>=400`; health/status clear |
| Agreement pass | all six Bonferroni simultaneous mean upper bounds no greater than `0.10` comparator posterior SD |
| Promotion | both tuning admission and fresh final sampler/agreement pass are required for `NEUTRA_CONFIRMED` at six-mean scope |
| Explanatory only | old and new short probes, acceptance, runtime, loss, quantiles, standard deviations, and correlations |
| Not concluded | full-distribution equivalence, filter exactness/ranking, calibration, cross-fixture robustness, production, or default readiness |

## Fresh Seed And Root Contract

| Cell | Tuning-verifier root | Warm-up root | Retained root | Fresh attempt root |
| --- | --- | --- | --- | --- |
| `PP-UKF` | `(20260716,42100)` plus grid index | `(20260716,42201)` | `(20260716,42301)` | `phase-p4/PP-UKF/neutra-confirmation/attempt-03` |
| `PP-SGQF` | `(20260716,43100)` plus grid index | `(20260716,43201)` | `(20260716,43301)` | `phase-p4/PP-SGQF/neutra-confirmation/attempt-02` |

Each verifier archive is excluded from inference. Warm-up uses 1,000-draw
chunks, minimum 2,000 and cap 10,000. Retained sampling uses 2,000-draw chunks,
minimum 4,000 and cap 10,000. Every stage has a distinct archive path and seed
domain.

## Required Checks And Artifacts

1. Rehash the old R4 result and every recursive ledger entry; verify target,
   transport, comparator, grid, leapfrog count, probe count, and old probe seed
   identity before reusing probe rows.
2. Reconstruct the repository-issued typed identity and frozen transport; run
   the compiled GPU/XLA target/status canary with memory growth.
3. Archive every fresh disjoint tuning verifier with target signature, seed,
   grid index, step, latent/model tensors, health, and modern R-hat.
4. Archive fresh warm-up/retained chunks and cumulative tensors separately;
   never load old P4 samples into the new controller.
5. Write tuning selection, progress, result, cell ledger, run manifest,
   recursive hashes, decision/inference tables, and exact nonclaims.
6. Run focused CPU-hidden tests, independently verify every new recursive
   ledger, and refresh P4/P7 terminal records from actual results.

## Repair, Handoff, And Stops

A serialization, path, reporting, or localized XLA harness defect may be
repaired in a fresh attempt under unchanged contract and remaining budget.
No tuning admission, warm-up cap, retained convergence cap, health/status veto,
supported mean disagreement, or unresolved agreement precision is a cell-local
terminal result. It does not invalidate the other cell or the research
direction.

After each cell, write a result record and update the terminal ledger. After
both cells are terminal, rerun P7 from a fresh attempt root and replace the
provisional P7 terminal artifacts with evidence-bound refreshed versions.

## Compute Budget

This repair remains inside the existing P4 phase and per-cell R4 budgets. One
fresh verifier sequence plus one fresh sequential confirmation per cell is
authorized. The two cells run sequentially on the existing GPU. No training,
comparator, target, package, environment, or hardware change is allowed.

## Skeptical Pre-Execution Audit

Decision: `PASS`.

The repair does not treat high final ESS as a substitute for tuning admission.
It does not rerun or tune on old retained results, change the margin after
inspection, select by acceptance, or reuse old warm-up/inference draws. Reusing
hash-bound short probes is justified because they are ordering-only diagnostics
and the repair begins at the earliest invalid rung. New verifier and downstream
seeds are disjoint from the program evidence checked by exact search. The
commands will answer the missing question directly: whether a fixed kernel can
first pass disjoint modern-R-hat admission and then support a fresh valid
confirmation.

Claude review is not repeated because the five-round P7 review ceiling has
been reached. The material P4 downgrade and this exact repair boundary were
already reviewed; Codex's local skeptical audit and focused tests govern the
implementation.

## Close Record

Both cells completed under the unchanged repair contract.

| Cell | Verifier modern R-hat | Warm-up / retained per chain | Final modern R-hat | Final bulk / tail ESS minimum | Six-mean agreement | Result SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `PP-UKF` | `1.0054056853` | `2,000 / 4,000` | `1.0008110775` | `27,623.60 / 13,394.13` | all passed | `d9b4f603b28acb06154ab554f41f745c5f544e2516ba4969c6b21d9e5268bacf` |
| `PP-SGQF` | `1.0013382279` | `2,000 / 4,000` | `1.0003275699` | `26,978.49 / 12,974.65` | all passed | `a77d5edf2b8129d6ff95844e9c5d4bb94b7125c9997777b517f36b830fbda9c4` |

Both verifier archives are target-bound, seed/grid matched, exactly 1,000
draws by four chains, and excluded from posterior inference. Warm-up and
retained samples use fresh disjoint seed domains and separate archives. All
recursive ledger entries reverified. The repaired harness suite passed `23`
tests; the refreshed P7 suite passed `3` tests.

P7 attempt 02 completed with `CELL_COMPLETE_WITH_BLOCKERS`: three narrow
confirmations, eight precise blockers, and no remaining tuning-admission
blocker. This repair does not establish any of the forbidden broader claims.
