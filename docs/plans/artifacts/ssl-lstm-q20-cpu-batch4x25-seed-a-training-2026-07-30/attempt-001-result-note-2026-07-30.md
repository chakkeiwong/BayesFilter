# q=20 CPU Seed-A Resume Attempt 001 Result Note

Date: 2026-07-30
Status: `RESOURCE_STOP_INCOMPLETE_FINALIZATION`

## Decision

Attempt 001 resumed the valid seed-A step-250 joint checkpoint after the
reboot, completed checkpoints through step 1750, and stopped on the declared
cumulative campaign cap before final support and heldout audit could run.
The run is engineering-valid checkpoint evidence and an incomplete training
screen. It is not a completed promotion result and does not reject the NeuTra
research direction.

## Evidence Contract And Accounting

| Item | Status | Evidence |
| --- | --- | --- |
| Main question | Partially answered | One q=20 `(32,32)` CPU batch-native stream reached step 1750 under the reviewed topology |
| Exact baseline | Preserved | Same target, seeds, architecture, optimizer, controller, batch 100, 25 workers x 4 rows |
| Cumulative cap | Vetoed continuation | Prior charge `5587 s`; attempt-001 wall `25762.2576 s`; manifest wall `31349.2576 s`; cap `31500 s` |
| Checkpoint finite/support screen | Passed at every written checkpoint | Steps `250, 500, 750, 1000, 1250, 1500, 1750`; all checkpoint eligibility veto lists empty; support finite; round-trip max `<=4.44e-15` |
| Controller behavior | Observed | Improvements through step 1500; step 1750 reduced learning rate to `0.0002` from `0.0004` after no meaningful improvement |
| Final support | Not run | Launcher reserved finalization time but the cumulative budget stopped the stream first |
| Heldout audit | Not run | No `result.json`; summary `results` is empty by design on `RESOURCE_STOP` |
| Primary completion criterion | Not met | Neither declared plateau stop nor 2,000 updates plus final support/audit completed |

The exact terminal artifacts are:

- launch manifest: `launch-attempt-001.json`
- restart arithmetic and command: `restart-attempt-001-record.md`
- progress receipt: `seed-a/progress.json`
- latest checkpoint: `seed-a/checkpoint-1750.json`
- terminal summary: `summary-attempt-001.json` and `summary.json`

The terminal summary records `RESOURCE_STOP` with reason
`declared CPU campaign cap exhausted`; it records no candidate result because
the final audit contract was not satisfied.

## Decision Table

| Decision | Primary criterion | Veto diagnostic | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Preserve checkpoint evidence | Passed through step 1750 | No finite/support/thread/memory/artifact veto observed | Final audit was not admitted | Keep all receipts and hashes immutable | No convergence or posterior claim |
| Treat attempt as incomplete | Failed finalization requirement | Cumulative budget exhausted before final support/audit | Whether step-1500 best would pass untouched audit | Record incomplete status; do not relabel as screen-passed | No candidate rejection from missing audit |
| Do not promote seed A | Not eligible under CPU exception | GPU/XLA and independent seed evidence absent | Cross-backend and seed robustness | Require separate authorized evidence before any promotion | No HMC readiness, transport promotion, or default change |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Checkpoint-level screens passed; terminal campaign continuation veto was budget exhaustion |
| Statistically supported ranking | None; one seed and no comparator uncertainty analysis |
| Descriptive-only differences | Loss decreased at checkpoints through step 1500; step 1750 triggered LR repair; these are descriptive only |
| Default-readiness | Ineligible CPU diagnostic exception |
| Next evidence needed | A separately authorized untouched final audit or fresh scope-specific run, independent seed replication, and claim-bearing GPU/XLA training before HMC |

## Engineering And Scientific Interpretation

The resume bug discovered after reboot was an infrastructure defect in stream
identity comparison: JSON-decoded seed lists were compared directly with
in-memory tuples. The minimal canonical-JSON comparison repair was tested
(`32 passed` in the focused launcher/control suite), recorded before launch,
and bound in the attempt-001 launcher hash. It did not alter the scientific
target or training procedure.

The run therefore distinguishes:

- engineering correctness: resume, topology, finite values, checkpoint hashes,
  thread audit, and memory checks passed;
- numerical/training validity: checkpoint support and round-trip checks passed,
  but final support and heldout audit were not executed;
- scientific interpretation: unsupported beyond descriptive one-seed CPU
  checkpoint behavior.

## Post-Run Red-Team

- Strongest alternative explanation: the apparent loss improvement may reflect
  one-seed training trajectory and cannot establish heldout generalization.
- Result that would overturn this note: a valid untouched final support/audit
  artifact under the same scope, or a declared continuation repair that changes
  the budget or target; neither is present here.
- Weakest evidence: raw checkpoint loss differences and runtime. They remain
  descriptive and were not used as promotion criteria.

## Nonclaims

This artifact does not establish convergence, posterior correctness, HMC
readiness, transport promotion, statistical superiority, architecture ranking,
production readiness, scientific validity, seed robustness, or a change to the
repository GPU NeuTra default. Seed B and claim-bearing GPU/XLA training remain
separate work.
