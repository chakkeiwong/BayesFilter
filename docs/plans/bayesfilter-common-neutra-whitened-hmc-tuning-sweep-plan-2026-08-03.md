# Common NeuTra-Whitened HMC Tuning Sweep Plan

Date: 2026-08-03
Status: `REVIEWED_READY_TO_EXECUTE`

## Objective

After promoting the repaired state-continuing epsilon-repair route to the
common/default NeuTra-whitened HMC tuning procedure, run a bounded tuning-only
validation sweep on the active executable lanes and determine whether each lane
can produce a valid complete viable-set artifact.

This sweep is engineering + tuning evidence only. It does not launch sequential
HMC, does not claim posterior convergence, and does not rank targets.

## Research intent ledger

| Field | Binding decision |
| --- | --- |
| Main question | For each active executable NeuTra-whitened target lane, can the common repaired tuning procedure produce a valid complete viable-set artifact? |
| Candidate/mechanism | One common repaired state-continuing epsilon-repair broad-grid tuning function with small target-specific hooks only (status-key contract, optional warm-start epsilons). |
| Comparator | Historical target-local procedures and artifacts are used only as identity/procedure references, not as ranking baselines. |
| Primary criterion | Barrier completes without infrastructure invalidity, artifact records the common procedure id, and the result is a valid viable-set or no-viable-pair terminal artifact under the common repaired procedure. |
| Promotion veto | Target/signature mismatch, frozen transport mismatch, missing required status keys, GPU/memory policy failure, barrier invalidity, stale output root, or procedure-variant/provenance mismatch. |
| Continuation veto | Two consecutive targets fail for the same infrastructure reason in the common repaired route, or total budget exceeded. |
| Repair trigger | Localized driver/config/serialization/plan-path/metadata error without scientific scope change. |
| Explanatory diagnostics | Viable primary count, viable coverage count, next-round candidate count, per-L acceptance intervals, repair-step counts, wall time. These are descriptive only. |
| Nonclaims | No sequential/posterior/convergence claim, no ranking, no default-readiness claim, no inference about blocked/excluded lanes. |

## Sweep scope

Active lanes to run:

- `PP-UKF`
- `PP-SGQF`
- `SIR-SGQF`
- `SVX-ZC`
- `KSC-UKF-GAUSSIAN-SUM-T20`
- `LGSSM-EXACT`

Blocked / excluded lanes to record but not run:

- historical `KSC-UKF` -> blocked
- `SIR-UKF` -> owner-excluded
- `SVX-SGQF`, `PP-ZC`, `SIR-ZC`, `STR-ZC` -> blocked / out of scope

## Artifact root

`docs/plans/artifacts/bayesfilter-neutra-common-tuning-sweep-20260803/`

with subroots:

- `pp-ukf-attempt01/`
- `pp-sgqf-attempt01/`
- `sir-sgqf-attempt01/`
- `svx-zc-attempt01/`
- `ksc-gaussian-sum-attempt01/`
- `lgssm-attempt01/`

Each run must use a fresh output root and must record the exact procedure id.

## Budget and stop conditions

Treat as a serious local research campaign.

- One attempt per active lane.
- One local retry only for infrastructure/harness failure with unchanged
  target, transport, procedure, and budget.
- Stop the entire sweep if **two consecutive targets** fail for the same
  infrastructure reason in the common repaired route.
- Stop any single target if wall time exceeds **2x** its predeclared projection.

Approximate projections:

- PP-UKF: already observed ~2.7 h under the repaired route
- PP-SGQF: expect similar or somewhat lower than PP-UKF
- SIR-SGQF: moderate
- SVX-ZC: moderate/high
- KSC gaussian-sum: moderate
- LGSSM: likely low/moderate, but record if it behaves anomalously

## Execution order

1. `PP-UKF` — known repaired-route success; canary for the common path
2. `PP-SGQF` — nearby family, same tuning function on a different filter lane
3. `SIR-SGQF` — active Austria SIR lane
4. `SVX-ZC` — active actual-SV lane
5. `KSC-UKF-GAUSSIAN-SUM-T20` — motivating regression lane
6. `LGSSM-EXACT` — last, because if it still behaves strangely under the common repaired route, that is evidence about its suitability, not about infrastructure correctness

## Pre-execution skeptical audit

Before running the sweep:

- verify the repaired common procedure is now the default path and the generic
  route is no longer silently selected;
- verify target-specific status-key contracts and warm-start hooks are coming
  from shared metadata rather than benchmark-local CLI ad hoc flags;
- verify KSC special-lane metadata is bound and recorded in the same shared
  way as registry cells;
- verify emitted manifests/artifacts record the exact common procedure id and
  correct plan path;
- verify the sweep driver refuses blocked/owner-excluded lanes;
- verify the route-ledger debt is recorded as pre-existing and not worsened by
  the sweep.

If that audit fails materially, revise code/plans first and do not run.
