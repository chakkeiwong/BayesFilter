# HNN-NeuTra Native Tuning Correction Plan

Status: `COMPLETE_CANARY_BUDGET_EXHAUSTED_NO_PROMOTION`.

Date: 2026-07-18.

Supervisor/executor: Codex. This plan supersedes the tuning and performance
parts of `bayesfilter-hnn-neutra-exact-gradient-comparison-repair-plan-2026-07-18.md`.
Prior posterior-validity artifacts remain historical evidence only where their
own fixed-kernel validity gates passed.

## Research Intent Ledger

| Field | Binding statement |
| --- | --- |
| Main question | With the same frozen target-specific NeuTra chart and exact Metropolis endpoint density, does HNN force substitution preserve posterior accuracy and reduce useful-sample cost relative to exact-gradient NeuTra-HMC after both arms are tuned correctly? |
| Candidate | Frozen position-only HNN force inside the reversible volume-preserving leapfrog map; exact transformed log density at both Metropolis endpoints. |
| Comparator | Exact transformed-posterior score inside the same leapfrog map and exact same endpoint density. |
| Native tuning authority | `bayesfilter.inference.fixed_transport_hmc_tuning.tune_fixed_transport_hmc_kernel`; its BayesFilter budget ladder, dual averaging, fixed-kernel screen, verification, and artifact schema. |
| Promotion criterion | Each arm independently obtains a native handoff with target acceptance `0.70`, fresh verification acceptance in `[0.65, 0.75]`, no hard health veto, and four-chain modern R-hat `max(rank-normalized split, folded rank-normalized split) <= 1.01` on at least 1000 retained draws per chain. Accuracy and cost are evaluated only after those gates pass. |
| Promotion veto | Acceptance outside `[0.65, 0.75]`; nonfinite state/value/score/log acceptance; native divergence; energy-health failure; missing or mismatched target, transport, force, coordinate, mass, or kernel identity; modern R-hat failure; unfair arm mechanics; or stale prior artifact presented as native-retuned evidence. |
| Continuation veto | Shared target/transport invalidity, native tuner API incapable of representing exact-value/HNN-score dynamics after a focused extension, trusted GPU/XLA failure after one localized repair, corrupted output, or canary budget exhaustion. Candidate rejection is not a continuation veto. |
| Repair trigger | Boundary step selection, out-of-band verification, rare energy tail, adapter sign/shape mismatch, artifact identity mismatch, or failed focused test. |
| Explanatory only | Short-screen R-hat, candidate runtime, ESS, energy quantiles, and fixed-mechanics timing. These do not choose the step size or establish superiority. |
| Forbidden conclusion | No tuned speed, seconds/ESS, break-even, superiority, or default-readiness claim from the superseded ad hoc selector or from a canary. |

## Mass And Coordinate Contract

The admitted NeuTra transport defines the HMC coordinate `z`. BayesFilter's
native fixed-transport policy is fixed identity mass in `z`; it deliberately
does not run a second windowed mass adaptation after the trained transport.
Both arms must use that same policy.

The identity artifact is constructed by
`PrecomputedMassArtifact.from_covariance(position=z0, covariance=I, ...)` and
must record:

- `covariance_source = fixed_identity_z`;
- `matrix_used_for_square_root = identity_z`;
- covariance and factor equal to identity up to the declared jitter policy;
- the transformed adapter signature, target scope, NeuTra transport hash, and
  artifact signature/hash; and
- `windowed_mass_adaptation_used = false` and
  `mass_adaptation_used = false` in the final handoff.

For the general BayesFilter mass route, a covariance `C` is represented by a
factor `L` with `L L^T = C`, and the affine program is
`theta = center + z L^T`. TFP then uses standard normal momentum in `z`. In raw
coordinates this is equivalent to inverse mass `C` (mass `C^{-1}`). The HNN
comparison does not invoke this second affine route because it would change the
coordinates in which the frozen HNN was trained.

Warm-up/tuning states may construct or diagnose a kernel but are not posterior
draws. Native final verification and retained inference use a frozen step size,
leapfrog count, mass artifact, target identity, and force identity. Warm-up
archives remain retained as diagnostic evidence and are excluded from posterior
summaries.

## Exact-Value / HNN-Score Adapter Contract

The HNN was trained on the transformed potential `U(z) = -log pi_z(z)` and
returns `grad U(z)`. BayesFilter value/score adapters return
`(log pi_z(z), grad log pi_z(z))`. The HNN tuning adapter must therefore return
the exact transformed log-density value from the admitted adapter and score
`-HNN_force(z)`. It must:

- preserve the exact target value used by the Metropolis correction;
- bind the frozen HNN identity and NeuTra transport hash in its adapter
  signature;
- expose batch-native TensorFlow value/score evaluation with no Python row loop;
- pass sign, shape, finiteness, target-value parity, and XLA tests; and
- never claim that the learned score equals the exact derivative.

## Skeptical Pre-Execution Audit

Status: `PASS_AFTER_REQUIRED_REPAIRS`.

Material defects found before execution:

1. The prior `tune_force` searched fixed grids with no adaptation and selected
   by lowest short-chain R-hat, then energy, then closeness to `0.8`. It did not
   optimize the required `0.70` target and bypassed native BayesFilter tuning.
2. The fixed-transport native config targeted `0.70` but its default pass band
   was `(0.60, 0.90)`. This must become `(0.65, 0.75)`; the repair band will be
   `(0.55, 0.85)` to match the native serious stack.
3. The earlier HNN harness hardcoded diagonal identity mechanics directly and
   emitted no native mass-artifact binding. The numerical mass choice happened
   to match fixed-NeuTra policy, but its provenance and freeze contract did not.
4. Short tuning R-hat was incorrectly used as an objective. Modern R-hat is a
   final fixed-kernel convergence veto, not a candidate ranking score.
5. The old performance and break-even claims used invalid tuning evidence and
   must be marked unsupported until native reruns complete.

The corrected artifacts will answer the research question because both arms
use the same admitted target/chart, the same native policy and compute ladder,
the same identity `z` mass, the same acceptance target/band, exact endpoint
values, fresh verification, and separately bound force identities.

## Phases And Repair/Continue Procedure

### Phase 0: Merge And Audit

Fast-forward to `origin/main`, preserve concurrent work, reconcile conflicts,
audit the native tuning/mass stack, and record this plan. Close when the worktree
has no unmerged paths and the native route is identified precisely.

### Phase 1: Native Policy And Adapter Repair

Change only the fixed-transport acceptance defaults, add the exact-value/HNN-
score adapter, wire the comparison harness to the native tuner, and disable the
ad hoc selector for research claims. Add focused tests for target/band defaults,
out-of-band rejection, modern R-hat veto, score sign/value parity, identity mass
binding, force/transport identity, and frozen-kernel handoff.

### Phase 2: CPU-Hidden Focused Verification

Run syntax, import, native tuner, HNN adapter, comparison, convergence, mass,
selection, warm-up, and verification tests. No result from this phase supports
a scientific or GPU performance claim.

### Phase 3: Trusted GPU/XLA Canary

Use the cheapest admitted cell and a fresh versioned root. Run one native exact
and one native HNN tuning canary with memory growth, GPU/XLA, target `0.70`, band
`[0.65, 0.75]`, fixed identity `z` mass, and bounded verification. The canary
budget is at most 30 GPU minutes and two attempts. A localized harness failure
may be repaired once without changing the scientific contract.

Attempt 1 completed in about six minutes.  Its exact arm's 64-step native
screens landed at `0.7806` (`L=6`) and `0.6352` (`L=10`), immediately outside
the owner band, while the learned `L=6` arm reached acceptance `0.7238` but
failed modern R-hat (`1.3590` rank-normalized, `1.0929` folded).  This is a
native warm-up-budget repair trigger, not authority for a hand-selected step.
Attempt 2 therefore extends the unchanged dual-averaging ladder to
`(16, 32, 64, 128, 256)` and preserves every promotion/veto criterion.

### Phase 4: Serious Rerun Decision

Write a canary result and drift audit. If both native handoffs pass, refresh the
four-cell serious campaign with versioned outputs and an explicit remaining GPU
budget before launch. If a candidate fails scientifically, preserve the result
and continue to the planned repair/next cell unless a true continuation veto
fired.

Terminal decision: do not launch the four-cell serious rerun.  Both allowed
PP--UKF canary attempts completed, but no exact/HNN pair passed the full native
acceptance, energy, and modern-R-hat contract.  The result and next justified
repair question are recorded in
`bayesfilter-hnn-neutra-native-tuning-correction-result-2026-07-18.md`.

At the end of every phase: run required checks; write or update the phase result;
refresh the next phase; audit baseline, defaults, artifacts, feasibility, and
boundaries; repair locally and continue when no real blocker remains.

## Planned Artifacts And Nonclaims

- Plan: this file.
- Merge/audit result:
  `bayesfilter-hnn-neutra-native-tuning-correction-result-2026-07-18.md`.
- Canary root:
  `docs/plans/artifacts/hnn-neutra-native-tuning-correction-20260718/`.
- Prior artifacts are preserved, never overwritten.
- Until Phase 3 passes, tuned runtime, seconds/ESS, and break-even remain
  `UNSUPPORTED_PENDING_NATIVE_RETUNING`.
