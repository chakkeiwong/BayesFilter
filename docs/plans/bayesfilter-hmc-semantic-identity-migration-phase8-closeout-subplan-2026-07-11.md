# Phase 8 Closeout And Boundary Handoff Subplan

Date: 2026-07-13

Status: `COMPLETE_DOCUMENTATION_CLOSEOUT_ONLY`

## Phase Objective

Close the semantic-identity migration runbook at the terminal Phase 7
diagnostic-cap result. Consolidate evidence, update the master/runbook/ledger/
stop handoff, and preserve the boundary to any future HMC repair campaign.

This phase authorizes no HMC transition, posterior-recovery evaluation, new
diagnostic experiment, retry, NeuTra work, model-file change, package change,
default-policy change, or scientific promotion claim.

## Entry Conditions

- Phase 7 terminal result exists at
  `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase7-academic-campaign-result-2026-07-13.md`.
- Attempt 1 is classified `diagnostic_cap_failure`, with exit code `1` and no
  retry permitted by the academic campaign state machine.
- The terminal result, progress, manifest, checksum manifest, attempt summary,
  and log are readable and their recorded checksums verify.
- No Phase 7 worker or controller process remains.
- No retained samples exist or are claimed because retained sampling did not
  begin.

## Required Artifacts

- this Phase 8 closeout subplan;
- Phase 7 academic campaign result;
- refreshed master program;
- refreshed visible runbook;
- refreshed visible ledger;
- refreshed visible stop handoff; and
- `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase8-closeout-result-2026-07-11.md`.

## Required Checks And Review

1. Re-run `load_attempt_history` and `verify_checksum_manifest` on the campaign
   root and attempt directory.
2. Confirm the terminal classification, exit code, burn-in cap, aggregate
   diagnostics, eight failed R-hat rows, absence of retained checks/samples,
   and Phase 8/NeuTra nonexecution fields.
3. Confirm no Phase 7 process remains.
4. Confirm historical authority files and the adopted V2 config retain their
   recorded SHA-256 values.
5. Run focused documentation consistency searches for stale active/resume or
   retry claims and `git diff --check` on the files touched by this closeout.
6. Use no additional review unless a material scientific or boundary
   inconsistency is discovered; the implementation convergence review already
   returned `VERDICT: AGREE`.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the program close consistently at the exact Phase 7 terminal result without weakening or overstating it? |
| Exact baseline | Phase 7 attempt-1 terminal result and checksum-verified artifact graph |
| Pass criterion | Every program document agrees that the fixed campaign failed R-hat at burn-in cap, retries are forbidden, retained sampling/Phase 8 runtime/NeuTra did not execute, and any repair requires a new plan |
| Veto | Checksum drift, process still running, disagreement about terminal classification or failed rows, false retained-sample claim, or any Phase 8/NeuTra execution claim |
| Explanatory only | Wall time, intermediate burn-in metrics, and historical governance artifacts |
| Not concluded | Target invalidity, HMC-direction rejection, posterior recovery, sampler ranking, production/default/GPU readiness, or a preferred repair |
| Preserving artifact | Phase 8 closeout result and refreshed program documents |

## Forbidden Claims And Actions

- Do not retry Phase 7 under the existing campaign.
- Do not reinterpret remaining wall-clock or attempt count as retry authority.
- Do not execute retained sampling, Phase 8 scientific evaluation, posterior
  recovery, NeuTra, or another sampler.
- Do not change the fixed result thresholds after observing the result.
- Do not describe the failure as target invalidity, implementation failure,
  broad HMC failure, or statistically supported inferiority.
- Do not overwrite or delete any campaign or historical authority artifact.

## Exact Handoff Conditions

Close the program when:

- the Phase 8 result records a documentation-only closeout pass;
- all refreshed documents name the terminal Phase 7 result and forbid retry;
- all evidence and boundary checks pass; and
- the next action is explicitly outside this program: a new research/repair
  plan with its own evidence contract and user direction.

There is no automatic next phase inside this runbook.

## Stop Conditions

Stop closeout and record a blocker if any required terminal artifact fails
validation, a Phase 7 process is still running, failed-row inventory differs,
retained samples unexpectedly exist, or concurrent edits make the five program
documents internally inconsistent.
