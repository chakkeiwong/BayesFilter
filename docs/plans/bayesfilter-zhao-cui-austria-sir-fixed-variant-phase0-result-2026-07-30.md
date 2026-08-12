# Zhao-Cui Austria SIR Fixed-Variant Phase 0 Result

Date: 2026-07-30

Status: `BLOCK_FIXED_VARIANT_BASELINE_NOT_RECONSTRUCTIBLE`

Plan:
`docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-parameter-extension-master-plan-2026-07-30.md`.

Execution note:
`docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-phase0-execution-note-2026-07-30.md`.

Terminal structured artifact:
`docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-phase0-result-attempt02-2026-07-30.json`.

Artifact SHA-256:
`4cb674e902646372527c9b0847cf95c2582efab7c70a7aabff3c50c90ada916a`.

## Outcome

The exact P88 T1 squared-TT density is reconstructible. The exact P88
fixed-TTSIRT retained-object baseline is not reconstructible from the committed
artifact and bound provenance.

Phase 0 therefore reached its declared blocker and stopped. Phase 1,
parameterization, score work, T2/T20 training, GPU, and HMC are not authorized.
No replacement estimator or transport was synthesized.

## Decision Table

| Field | Result |
|---|---|
| Decision | Stop without replacement. |
| Primary criterion | Failed for complete baseline reconstruction. |
| Density sub-gate | Passed exact P88 T1 density reconstruction. |
| Transport/retained sub-gate | Blocked by missing exact identity fields. |
| T2 boundary | Not evaluated; no valid P88 retained object can be issued. |
| Phase 1 authorization | `false` |
| Main uncertainty | Whether an unlocated historical artifact contains the exact frame, transport, frozen samples, retained identity, and source closure. |
| Next justified action | Search for the missing historical identity artifact, or obtain an owner decision defining a new baseline. |
| Not concluded | No active-data value, score, T2/T20, GPU, HMC, correctness, or production readiness. |

## Exact Passes

The CPU-only TensorFlow audit verified:

- P88 file SHA-256
  `ea5fc7434f328b95e3c2c53bca3e1a7bee6b35a452a81acce8230407ea11ef8e`;
- status `P88_PHASE2_DEGREE_ORDER3_RANK4_CANDIDATE_TRAINING_BASE_COMPLETED`;
- trainer `training_base_optimizer`;
- 36 serialized float64 cores and every per-core SHA-256;
- target id `zhao_cui_sir_austria_d18` from `route_manifest`;
- order 3, eight elements, 25 functions per axis, and rank-4 interior tuple;
- reference density/mass measure;
- `tau=1e-8`, normalizer floor `1e-12`, and denominator floor `1e-12`;
- density branch hash
  `265f9a06877e9babbba22dde187487fde4b50d08d8ecb98cd26b16467b6c1f10`;
- reconstructed square normalizer `4.544027196172014e-06`; and
- reconstructed full normalizer `4.554027196172014e-06`.

Both normalizers exactly match the P88 stored values under the executed
float64 TensorFlow contractions.

## Binding Blockers

P88 does not serialize or bind:

- `coordinate_frame_mu`;
- the full `coordinate_frame_matrix`;
- the transport `KRCDFConfig`;
- frozen reference samples used to issue the retained object;
- retained branch identity;
- the July 30 observation hash; or
- a source/callable dependency closure.

Only the frame log-determinant is stored. That scalar cannot reconstruct or
identify the 36-by-36 frame matrix.

The historical P59/P69 code contains diagnostic numerical-CDF settings such as
`grid_size=5` and `bisection_steps=4`, but P88 does not bind those settings and
the transport labels them diagnostic/non-production. Selecting them now would
create a new branch, not reconstruct P88.

The artifact-introducing commit
`c815edc52162779e969b2982723b2f52770fd849` does not contain
`scripts/p86_author_lagrangep_phase5_budget_fit.py`; that script first appears
in commit `9bc5a658bfaac29987438a50aea4bf7e9036719f`. Current source files also differ
from the artifact-introducing commit. Recomputing the frame from today's code
and recorded seeds therefore cannot prove the original P88 frame identity.

A repository-wide exact-identity search found the P86 predecessor
`docs/plans/bayesfilter-highdim-zhao-cui-p86-phase6y-degree-order3-rank4-lr3e-4-l1-0-fit-2026-06-26.json`.
It has the same density branch hash, serialized-core global hash, frame
log-determinant, and seed pair. It also omits the full frame, CDF configuration,
frozen references, retained identity, and source closure. No other matching
artifact recovered those fields. The predecessor confirms density provenance
but does not repair the fixed-filter reconstruction gap.

## Mathematical Classification

| Claimed target | Quantity actually reconstructed | Relation | Verdict |
|---|---|---|---|
| P88 squared-TT T1 density | Exact `phi^2 + tau*lambda` density from serialized cores and configuration | Equal under checked TensorFlow contractions | `correct` |
| P88 T1 fixed-TTSIRT retained scalar | No exact transport/frame/reference branch can be issued | Not reconstructed | `unsupported` |
| P88 T2 previous-marginal boundary | Requires the missing T1 retained object | Not evaluated | `not checked` |
| Parameterized value and total score | Outside Phase 0 and depends on the missing baseline | Not implemented | `not checked` |

## Attempt Ledger

| Attempt | Classification | Result |
|---|---|---|
| Launch 0 | Harness failure | Direct script execution lacked repository-root import context; no artifact written. |
| Structured attempt 1 | Schema ambiguity | Scientific result correct, but one `artifact_path` key obscured input versus output path. Preserved at `docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-phase0-result-2026-07-30.json`. |
| Structured attempt 2 | Terminal result | Corrected distinct input/result paths; reproduced density parity and both blockers. |

No scientific target, method, gate, or budget changed across the harness/schema
repairs.

## Verification

Intentional CPU-only command:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp /home/chakwong/anaconda3/envs/tf-gpu/bin/python -m pytest tests/highdim/test_zhao_cui_austria_sir_fixed_variant_phase0.py -q
```

Result: `3 passed, 2 warnings in 4.70s` for the initial Phase-0 nodes. The
terminal combined regression, including the runner contract and
`tests/highdim/test_p86_downstream_author_route_wiring.py`, returned
`10 passed, 2 warnings in 5.24s`. The warnings are TensorFlow Probability
deprecation warnings.

Artifact command:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp /home/chakwong/anaconda3/envs/tf-gpu/bin/python -m scripts.run_zhao_cui_austria_sir_fixed_variant_phase0 --repository-root . --output docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-phase0-result-attempt02-2026-07-30.json
```

TensorFlow emitted CUDA plugin registration and `cuInit` diagnostics despite
the pre-import `CUDA_VISIBLE_DEVICES=-1`. This was an intentionally CPU-only
run; no GPU execution was attempted and no GPU claim is made.

## Engineering, Numerical, And Scientific Ledgers

| Ledger | Status |
|---|---|
| Engineering correctness | Loader, hash validation, density reconstruction, tamper rejection, and fail-closed blocker tests pass. |
| Numerical validity | Exact stored/reconstructed normalizer parity passes. |
| Scientific interpretation | Complete fixed-variant value/score remains unavailable; no scientific comparison is authorized. |

## Inference Status

| Inference field | Status |
|---|---|
| Hard veto screen | Complete baseline admission veto fired. Density finiteness/hash/parity vetoes did not fire. |
| Viable object | Exact P88 T1 density only. |
| Statistically supported ranking | Not applicable; no stochastic method comparison ran. |
| Descriptive-only differences | None used for a decision. |
| Default readiness | No. |
| Next evidence needed | Exact historical transport/retained/source-closure artifact, or explicit owner selection of a new baseline. |

## Post-Run Red Team

Strongest alternative explanation: the frame and transport might be
deterministically reproducible from recorded seeds and the intended fit code.
That does not close identity: P88 binds neither the full frame nor the source
dependency closure, and the artifact-introducing commit lacks the named fit
script. A newly recomputed object could be numerically plausible but cannot be
proved to be the same P88 finite program.

The conclusion would be overturned by an immutable historical artifact that
binds the missing fields and reproduces the P88 branch, or by an owner decision
that explicitly changes the target from exact P88 reconstruction to a new
fixed-variant baseline.
