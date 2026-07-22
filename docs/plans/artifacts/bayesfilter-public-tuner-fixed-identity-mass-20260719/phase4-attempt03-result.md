# Phase 4 Attempt 03 Reboot-Replacement Result

Snapshot time: `2026-07-20T06:09:31+08:00`

Decision: `TUNING_REPLAY_HASH_MISMATCH`

Attempt 03 completed deterministic public retuning and passed the tuner's
fresh fixed-kernel verification, but failed the predeclared exact replay-hash
gate. Sequential warm-up and retained sampling were therefore not authorized
and did not start. This attempt is not second-seed posterior or truth-tail
evidence.

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Close Phase 4 under the current subplan at the replay gate and do not start Phase 5 | Failed: observed public final-kernel hash `e1d61cd46e9e65cd510bad14669619c7c9854348bee6a3a659e065a26a4ce0b6` did not equal expected Attempt 01 hash `e46effed4649e4cb7c3e25343549ab4c22315269fc46ccdba7b6506c076077fc` | Replay mismatch fired before sampling; no tuner hard veto; fixed identity mass, target, transport, GPU/XLA, and memory-growth gates remained valid | The mechanics-visible kernel fields match exactly, while the public hash also binds run-specific intermediate artifact hashes; hidden private mechanics are not exposed, so neither numerical equality nor numerical drift is proved | Replace the public-summary equality check with a repository-owned identity that is sufficient to witness executable kernel mechanics, then use a fresh reviewed plan and budget before any new sampling attempt | No second-seed replication, convergence, truth-tail, posterior-validity, universal NeuTra, sampler-superiority, or default-readiness conclusion |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | `TUNING_REPLAY_HASH_MISMATCH` is supported. Sequential sampling was correctly blocked. No numerical tuner hard veto, mass mutation, nonfinite target, or device-policy failure was observed. |
| Statistically supported ranking | None. No Attempt 03 posterior samples exist, so no stochastic ranking or second-seed inference is possible. |
| Descriptive-only differences | Attempt 03 took `21594.7682` seconds and emitted different intermediate artifact hashes. Those differences are engineering diagnostics only. |
| Default readiness | Not established. Phase 4 did not complete its two-seed diagnostic. |
| Next evidence needed | A reviewed mechanics-stable replay identity, focused regression proving invariance to lineage-only changes and sensitivity to step/L/mass/target/adapter or other transition-affecting changes, then a fresh-budget second-seed run under the unchanged scientific target. |

## Claimed Target And Quantity Computed

Claimed target: a pure second sampling-seed replication using exactly the
Attempt 01 executable HMC kernel.

Quantity checked by the implemented gate: equality of
`bayesfilter.hmc_frozen_kernel_handoff.v1` public summary hashes.

The public-summary hash is not a sufficient witness of executable-kernel
identity for this question. It includes hashes of intermediate bootstrap,
windowed-mass, fixed-mass-step, and Phase 7 artifacts. Those run-instance
lineage fields changed even though every exposed executable-kernel field checked
below matched. The gate verdict is therefore correct relative to its declared
exact-hash contract, but the artifact proves neither that the executable kernel
mechanics changed nor that hidden private mechanics were identical.

Mechanics-visible comparison with Attempt 01:

| Field | Attempt 01 | Attempt 03 |
| --- | --- | --- |
| Public tuner seed | `(20260621, 8)` | `(20260621, 8)` |
| Step size | `0.7779889586003162` | `0.7779889586003162` |
| Leapfrog steps | `6` | `6` |
| Selected-step hash | `c39e59a4ec867b98594e40b2a1551fbe92eabf4f05b4826e0a0e8bdd1631a9ec` | same |
| Fixed identity mass signature | `25eb272b3f8b1e742173a12ea1ae6a07ba8a203dfdba3e6f67deebc30a7598fe` | same |
| Verification acceptance | `0.7109046801795739` | `0.7109046801795739` |
| Public final-kernel hash | `e46effed4649e4cb7c3e25343549ab4c22315269fc46ccdba7b6506c076077fc` | `e1d61cd46e9e65cd510bad14669619c7c9854348bee6a3a659e065a26a4ce0b6` |

The differing public payload fields were only:

- `bootstrap_artifact_hash`;
- `windowed_stage_artifact_hash`;
- `fixed_mass_step_stage_artifact_hash`; and
- nested `phase7_final_kernel_hash`.

This establishes that public-summary inequality is an insufficient witness of
executable-kernel inequality: the hash can differ through lineage fields while
all exposed mechanics match. It does not establish that only lineage changed,
because the public artifact intentionally omits private mechanics. This is not
permission to override the gate retroactively. The Phase 4 contract required
exact top-level equality, so sampling remains ineligible.

## Execution And Artifact Evidence

Trusted GPU/XLA preflight passed before launch:

- NVIDIA GeForce RTX 4080 SUPER visible;
- TensorFlow `2.19.1` created `GPU:0`;
- XLA compilation executed on GPU;
- `bayesfilter.tensorflow.gpu_memory_policy.v1` recorded memory growth on every
  visible physical GPU and disabled full-device preallocation;
- TF32 execution was enabled.

Terminal tuning evidence:

- tuner final status: `passed`;
- diagnostic role: `fresh_fixed_kernel_verification_passed`;
- verification acceptance: `0.7109046801795739`, inside `[0.65, 0.75]`;
- hard vetoes: none;
- fixed identity mass signature remained unchanged;
- historical repair trigger:
  `verification_acceptance_evidence_inconclusive` on outer attempt 0;
- outer attempt 1 passed dependence-aware fixed-kernel verification.

Replay-gate evidence:

- status: `MISMATCH`;
- sequential sampling authorized: `false`;
- sequential warm-up seed: `null`;
- sequential retained seed: `null`;
- no warm-up, retained, chunk, sample, or truth-tail artifact exists.

Terminal artifact SHA-256 values:

```text
c846d283fe345af26117951ea17d5576dd6711309e9186715b4dac43fcde1daa  result.json
c3c1edb598461573a58d487abe9dcc54bf4d5a900b1ec4aaac7ac5e53f441629  run_manifest.json
97cba8019a6418e842288e21970238163519fc1729f97a2726c338c5e88c1a61  tuning_replay_hash_gate.json
8885f8f73ab457491025adfbcd43ee4349264424695cdf1fbe40dbb85ee86074  tuning/hmc_kernel_tuning_result.json
```

Artifact root:
`phase4-lgssm-attempt03-reboot-replacement/LGSSM-EXACT/`.

## Negative-Result Classification

- implementation failure: not established for tuning or sampler mechanics;
  the public replay-identity test is insufficient to decide the intended
  pure-kernel identity question;
- tuning failure: no, the public tuner passed its declared verification;
- diagnostic failure: yes, the exact replay gate vetoed the run using an
  identity that includes run-instance lineage fields and omits private
  mechanics from its public payload;
- evidence against NeuTra: none from Attempt 03, because sampling did not run;
- evidence against the current Phase 4 procedure: yes, exact retuning plus an
  over-broad public-summary hash is too expensive and cannot reliably authorize
  a sampling replication when all exposed mechanics match.

The strongest alternative explanation is that hidden private mechanics differ
despite matching exposed fields. The current public artifact is explicitly
non-replayable and does not expose mass arrays or all private tuning controls,
so this possibility is not ruled out. A repaired repository-owned identity
must bind the actual replayable step size, leapfrog count, mass payload, target,
adapter, and executable settings directly, while separating those mechanics
from run-instance provenance.

## Phase Decision

The governing Phase 4 subplan is
`phase3-next-subplan.md`. Its `Budget, Repair, And Stop Conditions` section
states the operative terms:

> Budget: one serious first-seed launch plus one second-seed launch when the
> owner's marginal rule fires. A localized harness repair may replace, but not
> add to, the remaining launch. Every retry uses a fresh attempt root and
> preserves the failure, repair, focused regression, and remaining budget.
>
> Stop on a true continuation veto or exhausted budget.

Attempt 01 consumed the first-seed launch. Attempt 02 was the authorized
second-seed launch but was interrupted by reboot before admission. Attempt 03
was its one-for-one localized infrastructure replacement, not an added launch,
and reached the terminal replay veto. Under the quoted terms, no additional
launch remains in the current subplan and its budget is exhausted.

Do not launch another seed, relax the hash gate, reinterpret Attempt 03 as
posterior evidence, or start Phase 5 under the current plan. A new run requires
an identity implementation sufficient to witness executable mechanics, focused
evidence, a refreshed skeptical plan, and fresh user-authorized compute budget.
