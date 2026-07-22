# P7 Subplan: Cross-Cell Synthesis And Terminal Integrity Audit

Date: 2026-07-16

Program ID: `multimodel-neutra-filter-posterior-20260715`

Status: `ATTEMPT_02_COMPLETE`

## Phase Objective

Reconcile all eleven mandatory cells, verify terminal evidence and state
bindings, separate engineering, numerical/filter, sampler, and scientific
claims, audit policy and execution drift, and write a terminal program result
and reset memo. P7 attempt 02 must verify the new P4 repair evidence rather
than infer states from attempt-01 labels. It may not launch a new scientific
run or promote a cell without post-repair evidence that passes every gate.

## Inherited Entry Conditions

- P0 and P1 completed the registry and shared-harness work.
- P2-P6 each have a terminal phase result or close record.
- P4-P6 have machine-readable phase-close ledgers; P2 and P3 predate that
  close-package convention and are bound directly by their result-note hashes
  and terminal result artifacts.
- Every mandatory cell has a terminal state. No candidate-family rejection is
  claimed, so untried enhanced transport arms are not being mislabeled as
  scientific rejection.
- P6 closed with one SIR confirmation and two precise cell-local blockers; no
  program budget or shared-harness veto fired.
- P7 attempt 01 validly found two P4 tuning-admission defects and remains
  preserved as the repair trigger.
- The bounded P4 R4 repair completed under the unchanged target, transport,
  comparator, thresholds, hardware class, and campaign budget. Both cells now
  have fresh disjoint verifier, warm-up, and retained archives in new roots.

## Inherited Terminal Matrix To Audit

| Cell | Expected terminal state | Evidence scope |
| --- | --- | --- |
| `SVX-SGQF` | `TARGET_BLOCKED_FILTER_ADMISSION` | implementation/score passed; frozen numerical filter gate failed |
| `SVX-ZC` | `TARGET_BLOCKED_SOURCE_ROUTE_MISMATCH` | current wrapper is an extension/invention and no production source route exists |
| `KSC-UKF` | `TARGET_BLOCKED_FILTER_ADMISSION` | declared UKF recurrence passed engineering checks but failed dense-reference margins |
| `PP-SGQF` | `NEUTRA_CONFIRMED` | fresh disjoint verifier R-hat `1.0013382`; fresh final R-hat `1.0003276`, ESS/health and all six mean bounds passed; result SHA-256 `a77d5edf2b8129d6ff95844e9c5d4bb94b7125c9997777b517f36b830fbda9c4` |
| `PP-UKF` | `NEUTRA_CONFIRMED` | fresh disjoint verifier R-hat `1.0054057`; fresh final R-hat `1.0008111`, ESS/health and all six mean bounds passed; result SHA-256 `d9b4f603b28acb06154ab554f41f745c5f544e2516ba4969c6b21d9e5268bacf` |
| `PP-ZC` | `TARGET_BLOCKED_SOURCE_ROUTE_MISMATCH` | production-ineligible generic retained-grid route |
| `STR-UKF` | `COMPARATOR_BLOCKED_GEOMETRY` | typed target admitted; source-health and affine-mode gates blocked comparator |
| `STR-ZC` | `TARGET_BLOCKED_EXTENSION_ROUTE_NOT_DESIGNED` | extension target absent |
| `SIR-SGQF` | `NEUTRA_CONFIRMED` | three physical posterior means, one T=20 fixture |
| `SIR-UKF` | `IMPLEMENTATION_BLOCKED_GPU_SCORE_PARITY` | trusted GPU/CPU score parity exceeded frozen limit |
| `SIR-ZC` | `TARGET_BLOCKED_MISSING_OBSERVED_DATA_SCORE_ROUTE` | parameter inference extension lacks observed-data derivative closure |

The expected corrected program classification is
`CELL_COMPLETE_WITH_BLOCKERS`: three narrow mean-level confirmations and eight
target/filter/implementation/comparator/source blockers.
`ALL_CELLS_CONFIRMED` is forbidden.

## Research Intent And Evidence Contract

| Field | Frozen P7 contract |
| --- | --- |
| Question | Is the complete program record internally consistent, target-bound, reproducible from preserved artifacts, and no stronger than its cell evidence? |
| Baseline | P0 eleven-cell registry, P2-P6 terminal results, P4-P6 close packages, and raw claim-bearing artifacts |
| Primary pass | exactly eleven unique cells; every terminal state has existing hash-matched evidence; confirmed cells pass typed target/transport/comparator/archive/diagnostic checks; no substitution or unsupported claim; terminal ledgers and reset memo complete |
| Promotion criteria | none inside P7; P4 repair evidence may restore a P4 state only when independently reverified |
| Promotion vetoes | missing/duplicate cell, unsupported terminal state, signature collision, hash mismatch, wrong target/transport/comparator binding, invalid confirmed-cell convergence/agreement/archive evidence, or overclaim |
| Continuation veto | irreparable corruption or missing raw evidence required for a terminal claim |
| Repair trigger | reporting, path, hash, state-label, claim-scope, or audit-harness defect with intact raw evidence |
| Explanatory only | aggregate counts, runtimes, losses, acceptance, quantiles, standard deviations, correlations, and observed blocker frequencies |
| Not concluded | universal NeuTra success, full-distribution equivalence, filter exactness/ranking, calibration, robustness, production/default readiness, or invalidity of blocked research directions |

## Required Artifacts

1. A CPU-only standard-library audit harness and structured result under
   `docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p7/attempt-02/`.
2. Terminal eleven-cell ledger with state, target signature where issued,
   evidence path, SHA-256, earliest re-entry rung, and narrow conclusion.
3. Hash/manifest/archive/diagnostic/policy audit rows and an explicit list of
   legacy or non-material evidence gaps.
4. Terminal program result with decision and inference-status tables,
   engineering/numerical/scientific ledgers, nonclaims, and red-team note.
5. Reset memo naming the exact next optional repair lanes without presenting
   them as unfinished P7 requirements.
6. One bounded read-only review record or explicit reviewer limitation.

## Required Audit Checks

1. Load
   `docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p0/attempt-04-20260715T1658/target_registry.json`
   and require exact set equality between its eleven `cell_id` values and the
   terminal matrix. Verify every registry cell has exactly one terminal state
   and no duplicate or conflicting assignment exists. State labels may
   legitimately be shared by multiple cells.
2. Recompute every terminal evidence hash and every P4-P6 phase-result binding.
3. Reverify recursive ledgers for all three claim-bearing R4 evidence roots and
   their final training/comparator roots; reject missing or mismatched files.
   Require `PP-UKF` attempt 03 and `PP-SGQF` attempt 02 rather than their old
   pre-repair confirmation roots.
4. For each confirmed cell, verify target signature equality across
   confirmation, comparator, training, ledger, and retained archive; verify
   transport/result bindings and no cross-cell signature collision.
5. Require fixed-kernel tuning admission from a disjoint tuning verifier with
   modern R-hat `<=1.01`, finite health/status, and zero declared energy
   divergences. Short-probe acceptance or bulk ESS may order candidates but
   cannot admit one. Verify warm-up and retained archives are separate;
   verify the tuning archive is target-bound, exactly 1,000 draws by four
   chains, seed/grid matched, and excluded from inference; verify warm-up and
   retained archives are separate; warm-up is excluded; retained draws are at
   most 10,000 per chain; and
   modern R-hat is explicitly the maximum of rank and folded rank-normalized
   split R-hat. The old missing-verifier defect remains historical evidence;
   only the fresh repaired roots may support the restored active state.
6. Verify final R-hat `<=1.01`, bulk ESS `>=1000`, tail ESS `>=400`, finite
   health/status, no declared energy divergences, and prospective simultaneous
   physical-mean agreement for all three confirmed cells. Final checks alone
   do not repair tuning admission; the separate verifier check above is also
   mandatory.
7. Verify serious-run manifests record Git commit, command, Python environment,
   GPU/memory growth, XLA, TF32, dtype, target, seeds, wall time, output root,
   plan, result, and nonclaims. Classify legacy omissions rather than inventing
   fields after the run.
8. Verify P2/P3 blockers are filter/source admission failures rather than HMC
   or NeuTra evidence; verify P5 comparator blockers do not become target
   rejection; verify P6 UKF/ZC evidence was not substituted into SIR-SGQF.
   Semantically revalidate every exact blocked subtype from raw structured
   fields or anchored result text, not merely from a matching hash or generic
   `not confirmed` state. In particular:
   - `SVX-SGQF` and `KSC-UKF` must expose the failed prospective filter value
     and/or score margins while their engineering/reference gates remain
     distinct;
   - `SVX-ZC` and `PP-ZC` must bind the source-route mismatch and production-
     ineligible extension/invention classification;
   - `STR-UKF` must bind the source-kernel energy-error health failure and the
     affine geometry terminal-score failure to
     `COMPARATOR_BLOCKED_GEOMETRY`, without target rejection;
   - `STR-ZC` must bind an absent, undesigned extension target;
   - `SIR-UKF` must bind the observed trusted GPU/CPU scale-normalized score
     gap `5.97e-7` and prospective limit `1e-7` to
     `IMPLEMENTATION_BLOCKED_GPU_SCORE_PARITY`; and
   - `SIR-ZC` must bind the fixed-rate paper/source example, three-parameter
     extension classification, and absent retained-marginal/proposal-transport
     observed-data derivative closure to
     `TARGET_BLOCKED_MISSING_OBSERVED_DATA_SCORE_ROUTE`.
9. Static-scan active shared NeuTra training/batching/confirmation routes for
   `numpy`, `tf.numpy_function`, `tf.py_function`, and Python sample-axis loops.
   Reporting/reference exceptions do not invalidate algorithmic paths.
10. Verify GPU training, XLA default, memory growth, target-specific fresh seed,
    5,000-step final training, and frozen/trainable parity for all confirmed
    cells.
11. Verify Zhao-Cui terminal claims remain source-anchored and retain
    `extension_or_invention` classification where applicable.
12. Search every inherited and P7-produced claim-bearing artifact for forbidden
    broad claims or rankings and confirm that descriptive diagnostics remain
    explanatory. This includes phase results, the P7 structured audit result,
    synthesized ledgers/tables, terminal program result, review record, and
    reset memo.

## Exact Commands And Environment

P7 is a deliberate CPU-only integrity run. Set `CUDA_VISIBLE_DEVICES=-1`
before Python import. It does not initialize a numerical framework or run HMC.

```bash
CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tf-gpu/bin/python \
  docs/benchmarks/audit_multimodel_neutra_p7.py \
  --output-root docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p7/attempt-02

CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tf-gpu/bin/python -m pytest -q \
  tests/test_multimodel_neutra_p7_audit.py
```

The audit harness and dedicated P7 test module use Python standard-library
artifact parsing only. The test asserts that neither TensorFlow nor TensorFlow
Probability is imported. The harness may read and hash existing evidence and
write only its fresh P7 output root. It must not mutate prior phase artifacts
or replay any target, HMC, training, or numerical-framework path.

## Default And Assumption Audit

| Choice | Provenance | Risk | Early diagnostic | Status |
| --- | --- | --- | --- | --- |
| terminal matrix | P2-P6 phase results | stale or prose-only state | compare exact result/close hashes | frozen audit baseline, not truth by assertion |
| confirmed scope | prospective P4/P6 R4 contracts | mean pass overread as distribution pass | inspect agreement schema and nonclaims | mean-level only |
| manifest completeness | runbook serious-run template | older results may predate fields | field-level manifest audit | missing field downgrades reproducibility claim, not raw numerical evidence automatically |
| static policy scan | repository policy | token search can produce false positives/negatives | limit scan to active modules and classify each hit | engineering audit, not proof of runtime absence by itself |
| normal MCSE agreement | P4/P6 prospective contracts | does not test tails/modes | preserve claim scope and explanatory distribution diagnostics | accepted for declared mean scope only |
| one-fixture evidence | actual campaigns | no generalization | explicit terminal nonclaim | limitation, not hidden default |

## Forbidden Claims And Actions

- No threshold change, new target, HMC, training, GPU run, or scientific rerun
  inside P7. A repaired P4 state is accepted only from the already completed,
  hash-bound post-repair artifacts.
- No `ALL_CELLS_CONFIRMED`, universal NeuTra, full-distribution equivalence,
  filter ranking/exactness, calibration, robustness, or readiness claim.
- No claim that a blocked target, comparator, or implementation disproves
  NeuTra or the research direction.
- No deletion, overwrite, retroactive manifest fabrication, or cross-cell
  substitution.

## Repair And Stop Conditions

A reporting/hash/path/state-label defect with intact raw evidence may be fixed
in P7 and rerun under a fresh attempt root. A confirmed-cell raw-artifact hash
mismatch, missing retained archive, signature collision, or failed required
diagnostic downgrades that cell and requires a visible result repair. Stop with
`BLOCKED_INVALID_EVIDENCE` only if required raw evidence is corrupted or
irretrievable. Do not stop for reviewer timeout, procedural wording, or an
expected cell-local blocker.

## Handoff Conditions

Close `CELL_COMPLETE_WITH_BLOCKERS` when the eleven-cell matrix, hashes,
confirmed-cell diagnostics, policies, claims, terminal result, review record,
and reset memo pass. The reset memo must identify future repair lanes by
earliest invalid rung and required new evidence. Those lanes are follow-up
research, not incomplete execution of this program.

## Compute And Attempt Budget

At most 8 CPU wall-hours and three focused repairs for one identical P7 audit
defect. No GPU or new scientific campaign belongs in P7.

## Skeptical Pre-Execution Audit

Decision: `PASS_AFTER_REFRESH`.

The stale attempt-01 draft assumed the P4 tuning defect remained terminal. The
attempt-02 refresh binds the completed repair roots and keeps P7 a bounded
CPU-only integrity audit.
It explicitly checks the main ways the terminal result could be misleading:
wrong or weak baselines, mean-level proxy overclaim, missing stop conditions,
target/transport/comparator substitution, stale state, legacy manifest gaps,
wrong modern R-hat semantics, archive pooling, under-accounted blockers, and
commands that merely regenerate prose. The structured harness must answer those
questions from preserved artifacts and is forbidden from upgrading evidence.

The bounded advisory review ceiling was reached before the repair. Attempt 01
correctly found that the old predator-prey confirmations lacked disjoint
modern-R-hat admission. The repair resumed at that earliest invalid rung,
preserved old evidence, used fresh seeds and roots, and passed focused tests.
Codex's skeptical attempt-02 audit found and fixed the stale expected matrix,
hardcoded roots, confirmation count, re-entry rungs, claim scan, and verifier-
archive checks. Reviewer unavailability is not a continuation veto under the
program review ceiling and repository proportionality policy.
