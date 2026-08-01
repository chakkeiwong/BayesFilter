# HNN-NeuTra Native Tuning Correction Result

Date: 2026-07-18.

Decision: `NATIVE_TUNING_CORRECT_CURRENT_PP_UKF_CANDIDATES_NOT_PROMOTED`.

Plan:
`docs/plans/bayesfilter-hnn-neutra-native-tuning-correction-plan-2026-07-18.md`.

## Direct Answer

The previous HNN--NeuTra comparison was invalid as a tuned performance
experiment.  It used an ad hoc fixed grid, no dual-averaging warm-up, different
candidate seeds, short-chain R-hat as a selector, and an unbound hard-coded
identity diagonal mass.  Its tuned runtime, seconds/ESS, speed, break-even, and
performance-pass claims are now marked
`UNSUPPORTED_PENDING_NATIVE_RETUNING`.

The replacement uses BayesFilter's native
`tune_fixed_transport_hmc_kernel`, native fixed-mass dual averaging, target
acceptance `0.70`, fresh verification band `[0.65, 0.75]`, exact endpoint
Metropolis energy for the HNN arm, and the required maximum of rank-normalized
split and folded rank-normalized split R-hat.  Acceptance tuning now works: in
attempt 2 the HNN `L=10` candidate verified at `0.74223` and the exact `L=6`
candidate at `0.73545`.  Neither candidate passed the full validity gate.

## Mass-Matrix Audit

The correct fixed-NeuTra policy is identity mass in trained transport
coordinates `z`.  BayesFilter constructs one repository-owned
`PrecomputedMassArtifact` with position `z0`, covariance `I`, factor `I`,
`covariance_source=fixed_identity_z`, and
`matrix_used_for_square_root=identity_z`.  Both arms used the same artifact
signature:

`9e32548127ea9972a1fbf8e19de22791ed1972be1cc414431db8ca1b63cd0a38`.

No windowed mass adaptation ran.  This is intentional: NeuTra already defines
the trained geometry, and a second affine mass chart would change the
coordinates in which the frozen HNN force was trained.  In the general native
mass route, covariance `C` is factored as `L L^T=C` and the row-vector chart is
`theta=center+z L^T`; identity momentum in `z` corresponds to inverse mass `C`
in raw coordinates, or raw-coordinate mass `C^{-1}`.

The final handoff also binds target scope, base/transported adapter signatures,
transport manifest hash, mass signature, and proposal-dynamics identity.  The
HNN runner applies the native affine force pullback and computes untruncated
`log_accept_ratio=-delta_h` from exact current/proposed endpoint potentials.

## Canary Evidence

Both attempts used the trusted RTX 4080 SUPER, TensorFlow/TFP, GPU memory
growth, XLA JIT, float64, four chains, and fresh output roots.  The attempt-2
ladder was the unchanged native method with an extended adaptation schedule
`(16,32,64,128,256)`, not a hand-selected step.

Promotion required fresh mean Metropolis acceptance probability in
`[0.65,0.75]`, maximum rank-normalized/folded split R-hat at most `1.01`, and
maximum absolute endpoint energy error at most `1000`, in addition to finite
states/values and identity checks.

| Attempt | Arm / candidate | Native fixed verification acceptance | Binary acceptance | Rank R-hat | Folded R-hat | Max abs energy error | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | HNN, `L=6`, eps `1.03308` | 0.72377 | 0.72925 | 1.35902 | 1.09286 | 53.80 | R-hat veto |
| 1 | exact | no in-band native candidate | N/A | N/A | N/A | N/A | ladder exhausted |
| 2 | HNN, `L=10`, eps `0.93417` | 0.74223 | 0.74275 | 1.00286 | 1.02372 | 34.44 | folded R-hat veto |
| 2 | exact, `L=6`, eps `1.00845` | 0.73545 | 0.74100 | 1.64387 | 1.09801 | 3898.23 | rank/folded R-hat and energy veto |

Attempt 2 also screened exact `L=10` through 256 adaptation steps, but its
final fixed screen acceptance was `0.78484`, outside the owner band.  It was
not promoted or used to rank performance.

Artifacts:

- attempt 1 result SHA-256:
  `6841942ce94a544269de857c6f771511468828005b0157d26c326594df43fe7e`;
- attempt 2 result SHA-256:
  `9d6c634d76cef488701377ac6e297b709935bdb11a353ee2d0116c8b8a89f6cb`;
- attempt 1 wall time: `366.65 s`;
- attempt 2 wall time: `972.57 s`;
- output roots:
  `docs/plans/artifacts/hnn-neutra-native-tuning-correction-20260718/pp-ukf-canary-attempt-01/`
  and `pp-ukf-canary-attempt-02/`.

Total GPU wall time was about 22.3 minutes, within the 30-minute/two-attempt
campaign budget.

## Verification And Review

The merge fast-forwarded local `main` to `origin/main` commit
`fbc3b6e9aaf882b8275bfb94aaa2ff43cc4c5a98`, reapplied the shared dirty
worktree, and resolved all conflicts.  Recovery stash
`pre-native-hmc-tuning-audit-20260718-shared-lanes` remains preserved.

Focused CPU-hidden native-stack verification passed `137` tests after stale
fixtures were corrected to make their `log_accept_ratio` agree with claimed
acceptance.  A narrower final HNN/native suite passed `41` tests.  Production
acceptance semantics were not weakened to satisfy old fixtures.
The terminal native-stack run passed `334` tests with one expected skip.

Claude's bounded plan review returned `VERDICT: AGREE`.  Its first
implementation review returned `REVISE` because delegated dual averaging,
endpoint energy, mass identity, and four-chain initialization were not visible
from the one reviewed file.  The local audit then inspected those exact native
implementations, removed the reduced canary ladder, bound proposal-force
identity, asserted cross-arm mass-signature equality, and added contract tests.

## Decision Table

| Decision field | Result |
| --- | --- |
| Primary criterion | failed: no exact/HNN pair passed all native gates |
| Acceptance tuning | passed mechanically for HNN `L=10` and exact `L=6` at target `0.70` / band `[0.65,0.75]` |
| Hard vetoes | HNN folded R-hat; exact rank/folded R-hat and energy tail |
| Main uncertainty | whether longer target-specific warm-up, different trajectory lengths, or improved HNN training can repair convergence without energy failure |
| Next justified action | design a separate PP--UKF candidate-repair plan; do not launch the four-cell performance campaign |
| Not concluded | no HNN accuracy, speed, seconds/ESS, break-even, superiority, or default-readiness claim |

## Inference Status

| Field | Status |
| --- | --- |
| Hard veto screen | failed for every attempt-2 candidate |
| Viable candidates | none promoted; HNN `L=10` is an acceptance-valid repair candidate only |
| Statistically supported ranking | none |
| Descriptive-only differences | all timing, acceptance, R-hat, and energy values except their stated hard-screen roles |
| Default readiness | false |
| Next evidence needed | a separately reviewed, target-specific convergence/energy repair with adequate warm-up and no performance interpretation until both arms pass |

## Post-Run Red Team

The strongest alternative explanation is inadequate target-specific warm-up
or trajectory selection rather than failure of HNN force substitution itself.
That explanation does not rescue the current candidates: modern R-hat and the
exact energy tail are promotion vetoes.  Conversely, the run does not reject
the research direction because the observed vetoes did not invalidate the
target, transport, mass identity, endpoint Metropolis computation, or native
tuner.  It rejects only the current PP--UKF candidate kernels under the
declared budget.

The weakest engineering point is repeated XLA retracing in the injected HNN
runner across tuning configurations.  This raises tuning cost but does not
change acceptance, endpoint energy, mass coordinates, or R-hat.  A reusable
dynamic runner is an optimization task, not a reason to reinterpret the failed
scientific gates.
