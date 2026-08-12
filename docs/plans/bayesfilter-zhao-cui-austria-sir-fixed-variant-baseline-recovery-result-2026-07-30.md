# Zhao-Cui Austria SIR Fixed-Variant Baseline Recovery Result

Date: 2026-07-30

Status: `BLOCK_EXACT_P88_RECOVERY_EXHAUSTED`

Plan:
`docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-baseline-recovery-plan-2026-07-30.md`.

Terminal artifact:
`docs/plans/artifacts/zhao-cui-austria-sir-fixed-variant-baseline-recovery-20260730/recovery-attempt-02/result.json`.

Terminal artifact SHA-256:
`2ebf1e7ca2bdda6bfd9877011b60e4fd59029610a0a8a3f022a4bcfebbbb90d7`.

## Outcome

Lane A executed and found no admissible historical artifact for the complete
P88 fixed-TTSIRT retained program. P88 remains an exact T1 squared-TT density
artifact only.

No numerical replay ran because no candidate bound the exact frame, KR
configuration, frozen references, retained identity, observation/input
identity, and source closure. Recomputing those fields would create a new
program rather than prove the historical P88 program.

Lane B is not authorized by this result. The next boundary is an owner decision
whether to construct the newly named
`zhao_cui_austria_sir_fixed_variant_training_base_v1` baseline.

## Decision Table

| Field | Result |
|---|---|
| Decision | Stop Lane A without replay or replacement. |
| Primary criterion | Failed: no artifact bound every required identity group. |
| Density sub-gate | Still passed from Phase 0: exact P88 `phi^2 + tau*lambda` density. |
| Hard veto | Missing frame, CDF, frozen references, retained identity, input identity, and source closure. |
| Main uncertainty | An artifact outside the searched local workspace could exist; no local evidence supports it. |
| Next justified action | Owner decision on Lane B and its new baseline identity. |
| Not concluded | No complete filter, value, score, T2/T20, GPU, HMC, correctness, or production readiness. |

## Search Evidence

The terminal inventory covered:

- 42,615 workspace files, including registered Claude worktrees, ignored
  artifacts, local logs/resources, and the checked-out July 11 source snapshot;
- all 15,201 main-repository Git blobs, including 389 unreachable blobs;
- all 10,748 Git blobs in the preserved July 11 complete-source snapshot,
  including 456 unreachable blobs; and
- every ref, reflog/stash object, branch, and registered worktree exposed by
  those object databases.

One 555,637,037-byte blob exceeded the bounded in-memory scan in both Git
databases. It is the same unrelated DPF teacher-data JSON at
`experiments/dpf_implementation/reports/outputs/batched_annealed_transport_teacher_data_expanded_2026-06-20.json`.
A streaming search found none of the exact P88 or missing-identity anchors.

The exact P88 artifact appears in the current tree and the July 11 snapshot
with the same SHA-256
`ea5fc7434f328b95e3c2c53bca3e1a7bee6b35a452a81acce8230407ea11ef8e`.
Both copies have the same missing identity groups.

## False-Lead Review

The syntactic inventory nominated two Phase-0 result JSONs because each
contains the P88 identifiers and names every missing field. Manual typed-value
review rejected both:

| Candidate | Actual content | Verdict |
|---|---|---|
| First Phase-0 result | Field names in `missing_identity_fields`; source closure and observation binding are literal `absent_in_p88_artifact` markers. | Reject. |
| Terminal Phase-0 attempt 02 | Same missing-field declarations and absence markers. | Reject. |

Neither contains `coordinate_frame_mu`, a 36-by-36 frame, a CDF configuration,
reference arrays, or a retained identity. No Git blob produced a complete lead.

## Provenance Check

The exact P88 JSON exists at introducing commit
`c815edc52162779e969b2982723b2f52770fd849`, but the named fit script does not.
The script first appears at
`9bc5a658bfaac29987438a50aea4bf7e9036719f`. This independently preserves the
Phase-0 conclusion that current-code recomputation cannot prove historical
identity.

## Attempt Ledger

| Attempt | Classification | Outcome |
|---|---|---|
| 01 | Harness failure | Git batch response was read from the wrong pipe. No inventory JSON or scientific decision was produced. |
| 02 | Repaired terminal inventory | Focused regression passed; inventory completed in 19.44 seconds; manual admission rejected both false leads. |

The repair changed only subprocess plumbing. It did not change the baseline,
search scope, admission criterion, or compute budget.

## Engineering, Numerical, And Scientific Ledgers

| Ledger | Status |
|---|---|
| Engineering correctness | Inventory harness and Git batch regression pass. Workspace and both Git object databases completed. |
| Numerical validity | No new numerical computation; exact Phase-0 density evidence is unchanged. |
| Scientific interpretation | Complete historical P88 retained program is not locally recoverable. |

## Inference Status

| Field | Status |
|---|---|
| Hard veto screen | Fired: every candidate lacks required identity values. |
| Statistically supported ranking | Not applicable; no stochastic comparison ran. |
| Descriptive-only differences | Search counts and runtime are descriptive inventory facts. |
| Default readiness | No. |
| Next evidence needed | Owner-approved Lane B or an externally supplied exact historical artifact. |

## Post-Run Red Team

Strongest alternative explanation: a private artifact exists outside this
workspace. That remains possible but unsupported. Within the authorized local
scope, both reachable and unreachable Git objects, snapshots, worktrees, and
ignored artifacts were searched.

The conclusion would be overturned by an artifact that independently binds all
missing identity groups to the exact P88 cores and branch. A plausible
recomputed frame, seeds, log determinant, or matching scalar would not overturn
it.

This result rejects exact historical recovery, not the fixed-variant research
direction. The planned repair is Lane B, but changing to that new baseline
identity requires the owner decision specified in the recovery plan.
