# Phase 8 Subplan: Same-Program FD Gate Reclassification

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Status: `CLOSED_FORMAL_UNSUPPORTED_HEURISTIC_INCONCLUSIVE_INVALID_ENDPOINT`

## Phase Objective

Replace the unsupported claim of a rigorous finite-difference error-bound
certificate with an executable seven-step same-program implementation screen
that matches the owner's stated purpose. The screen is heuristic only. It may
corroborate or flag the checked manual score, but it cannot prove a derivative,
define a confidence interval, or substitute for Kalman-gradient agreement.

## Entry Conditions

- The canonical Phase 5 fixture and one value-and-score callable are frozen.
- Manual JVP versus TensorFlow forward autodiff is already zero ULP on that
  fixture; matrix identities and local JVP/VJP tests provide the primary
  engineering derivative evidence.
- The Phase 1 seven-step ladder, common-randomness, representable-endpoint, and
  branch-identity definitions are frozen.
- Phase 1 left endpoint-value and score absolute-error bounds null. No checked
  TensorFlow/XLA kernel theorem supplies those bounds.
- No target-prefix, `T=10`, GPU, or primary-shape value/gradient runs are
  authorized here.

## Scientific Classification

Claimed target: a rigorous interval-certified FD proof of the compiled
callable's derivative.

Actually available: central finite differences of the same finite scalar at
seven representable step sizes, compared descriptively to the checked manual
score under the owner-directed relative screen.

Verdict: the rigorous certificate is `unsupported` because callable-specific
absolute forward-error bounds are absent. The seven-step screen is
`heuristic only`; it is useful as an independent finite-scalar wiring check but
must not be promoted to a mathematical error certificate.

This formal-certificate reclassification is unconditional and does not depend
on the heuristic outcome. The separate heuristic outcome is exactly one of
`PASSED`, `FAILED`, or `INCONCLUSIVE`.

## Frozen Screen

For each of the five physical coordinates, use float64

```text
h0_j = cbrt(machine_epsilon_float64) * max(1, abs(theta_j))
multipliers = [8,4,2,1,0.5,0.25,0.125]
```

Construct plus/minus endpoints through float64, record the actual positive and
negative steps, and require exact symmetry and noncollapse. Every endpoint is
evaluated through the same canonical value-and-score concrete function with
identical prepared inputs. Center and endpoint calls must return finite values
and scores and the exact same branch hash/chart masks.

For valid step `k`, compute

```text
D_jk = (f(theta+h_jk e_j)-f(theta-h_jk e_j))/(2*h_actual_jk)
relative_error_jk = abs(score_j-D_jk)/abs(D_jk).
```

An observed FD denominator is eligible for this heuristic relative-error
calculation only if every `abs(D_jk)` is strictly greater than the diagnostic
central-difference cancellation floor

```text
cancellation_floor_jk =
  machine_epsilon_float64 *
  (abs(f_plus)+abs(f_minus)) / (2*h_actual_jk).
```

The floor is a diagnostic scale, not a rigorous bound. Eligibility does not
establish that the mathematical derivative is nonzero or not near zero. If any
`abs(D_jk) <= cancellation_floor_jk`, that coordinate is
`INCONCLUSIVE_NEAR_ZERO`; no absolute pass boundary is invented. Otherwise the
owner screen is

```text
relative_error_jk <= 0.05*sqrt(5)
```

for all seven valid steps. Requiring every step avoids choosing a favorable
step after seeing the result. Also report consecutive differences and ordinary
second-order Richardson estimates descriptively; do not convert them into
confidence/error intervals.

Outcome classification is deterministic:

- any endpoint collapse, unequal actual plus/minus step, nonfinite, chart
  invalidity, or branch mismatch gives
  `FD_HEURISTIC_INCONCLUSIVE_INVALID_ENDPOINT` and vetoes a pass;
- any denominator ineligible by the diagnostic floor gives
  `FD_HEURISTIC_INCONCLUSIVE_NEAR_ZERO`;
- otherwise, all 35 relative errors at or below `0.05*sqrt(5)` gives
  `SEVEN_STEP_FD_HEURISTIC_SCREEN_PASSED`;
- otherwise the completed valid screen gives
  `SEVEN_STEP_FD_HEURISTIC_SCREEN_FAILED`.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does the checked canonical score agree with all seven same-callable central differences under the owner FD-only heuristic while endpoint/branch identity remains exact? |
| Baseline | Phase 5 v2 fixture/callable and zero-ULP manual-JVP/forward-autodiff evidence |
| Pass criterion | all 35 endpoint pairs valid, finite, branch-identical, non-near-zero by the diagnostic floor, and each relative error passes `0.05*sqrt(5)` |
| Hard vetoes | changed prepared input; separate value graph; endpoint collapse/asymmetry; nonfinite; chart/branch mismatch; missing per-step result; each veto blocks a pass and classifies the screen inconclusive-invalid-endpoint |
| Inconclusive | diagnostic denominator ineligible or any invalid endpoint; no absolute tolerance may rescue it |
| Explanatory only | Richardson estimates, consecutive differences, actual relative-error magnitudes |
| Not concluded | rigorous derivative proof, confidence coverage, Kalman equivalence, target-shape FD, HMC/default/leaderboard readiness |

## Skeptical Plan Audit

Decision: `PASS_FOR_HEURISTIC_RECLASSIFICATION_ONLY`.

- The plan removes an unsupported formal claim instead of fabricating kernel
  error bounds.
- It uses the owner's threshold only for its stated FD purpose.
- All steps must pass; there is no post-result step selection.
- The diagnostic cancellation floor determines only when relative error is
  uninterpretable; it is not an absolute pass tolerance.
- Forward autodiff equality remains the primary checked derivative evidence;
  FD remains structurally independent corroboration of the finite scalar.

## Required Artifacts And Checks

- dedicated CPU-hidden harness and tests;
- one structured JSON with all 35 endpoint pairs, branch hashes, actual steps,
  FD estimates, cancellation floors, relative errors, and nonclaims;
- exact output root
  `docs/plans/logs/contract-e-canonical-gradient-migration-2026-07-13/phase8/fd-reclassification-attempt1/`;
- CPU-hidden prefix
  `CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 MPLCONFIGDIR=/tmp`;
- `300`-second timeout, one planned attempt, and one localized harness retry cap;
- result, manifest, focused checks, and review record.

The exact baseline is the fixture
`docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase5-tiny-fixture-freeze-v2-2026-07-14.json`
with SHA-256
`f6b6e2895208d7cd5cba0f57b05d4de7fb0de79e50ba62b7e6c70b06879942f4`.
The pre-instrumentation Phase 5 v2 certificate is
`docs/plans/logs/contract-e-canonical-gradient-migration-2026-07-13/phase5/cpu-xla-same-callable-certificate-v2.json`
with SHA-256
`20ec133bc5aee47f5daf3dc54d4c3593189202b1c305640cbd30b5e33b4ca709`.
Expected center identity is objective hex `-0x1.55564a66d9848p+2`, score hex
`[-0x1.c993b9119c770p-2,-0x1.cad12b05cc707p-3,
0x1.cc0ca41e05574p-5,-0x1.c6af389364ccfp+1,
-0x1.2fce89f3bf0cap+2]`, and branch hash
`bf25ece12ff85525620fdc1284abab76a35a54c28a4f998b89bbabd56aa005d7`.
The eight prepared-input hashes are exactly those recorded in that certificate
and must be reproduced in the new artifact. The current source closure is
canonical `33f37f6bfd156b82b3f66334545ce5c16ddb94a59040a2d434a36cec06ad8f0b`,
streaming `f98e46ee7a80588ba3f9b2a242121786dc9b3999a8f6c2943c22899fdcbc04df`,
reset `5a226b53f4a881a1b66cee00902dcd007c82de3c01e3440101c111c5095ee023`,
and annealed transport
`137a6ce58c2d6708d58a6714ab725be7a7497f3ea82e901de4dec376b0b11479`.

The callable is `tf.function` float64 with one input signature `[5]`,
`jit_compile=True`, one concrete function, and the same callable returns value
and score at center and all endpoints. Record Python/TensorFlow/Git, logical
devices, JIT, source/prepared hashes, command, wall time, and output path. The
attempt-1 result path is `<root>/attempt1/result.json`; a localized harness retry
must preserve attempt 1, use unchanged semantics, refuse overwrite, and write
only `<root>/attempt2/result.json`.

## Forbidden Claims And Actions

- Do not call this a rigorous error bound, confidence interval, or proof.
- Do not choose a subset of steps or alter the ladder after output.
- Do not apply `0.05*sqrt(p)` to Kalman agreement, ridge/transport adequacy, or
  any other numerical/scientific decision.
- Do not run target-prefix, `T=10`, GPU, primary shape, HMC, nonlinear,
  leaderboard, release, or integrity work.

## Exact Handoff Conditions

Reclassify the formal-error-bound certificate unconditionally as unsupported.
Then record the separate heuristic outcome using the exact four statuses above.
A pass closes only the FD heuristic implementation-screen requirement. A valid
completed screen above threshold is a failure; a denominator-ineligible or
invalid-endpoint screen is inconclusive. No outcome clears the owner numerical-
design or primary statistical blockers.

## Stop Conditions

Stop for scalar/source drift, endpoint/branch invalidity, nonfinite output,
near-zero inconclusiveness, campaign-clock exhaustion, or a material review
finding. A serialization/harness defect is a localized repair trigger.
