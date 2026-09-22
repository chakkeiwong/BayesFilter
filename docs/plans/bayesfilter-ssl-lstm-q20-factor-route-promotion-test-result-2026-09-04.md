# q=20 Factor-Route Promotion Test Result

Date: 2026-09-04  
Plan: `docs/plans/bayesfilter-ssl-lstm-q20-factor-route-promotion-test-plan-2026-09-04.md`  
Parent: `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md`  
Status: `DEFAULT_ELIGIBLE_PENDING_FRESH_TUNING`  
Decision: factor route remains opt-in; strict route remains the default numerical authority.

## Executive result

The factor-basis route passed the bounded implementation and mechanics test.
It agrees with the strict route on the tested q=20 value/score/status program,
paired HMC transitions, and a short four-chain run.  The repaired identity
boundary also rejects a stale strict tuning handoff when it is presented to a
factor adapter.

This is sufficient to call the factor route a viable mechanics candidate.  It
is not sufficient to promote it as the repository default or to issue a
posterior claim: the route still needs a fresh factor-specific scope tuning
artifact and an untouched run under that identity, followed by the parent
program's downstream gates.

## Repair performed before the campaign

`BatchNativeSSLLSTMComplexityPosteriorTarget.adapter_signature()` already
included `principal_sqrt_backend`.  `GaussianLikelihoodBridge` and
`FixedBetaBridgeAdapter` did not propagate that identity, so their signatures
could collide across strict and factor constructions.  The bridge now records
`component_target_adapter_signature` in its source facts and signature payload;
the fixed-beta adapter carries the same field.  A missing generic component
signature is represented explicitly as unbound rather than fabricated.

The eigensystem benchmark also received an explicit
`--candidate-backend` selector.  This repaired an evidence-harness error in
which the intentionally rejected raw-cache arm forced an overall failure even
when the factor arm passed.  The all-candidate diagnostic behavior is retained
for historical comparison; promotion status is determined only by the declared
factor candidate.

The standalone P4 payload is deliberately incomplete and is used only to test
stale cross-backend rejection.  Positive construction of a valid measured-grid
handoff was checked separately with the repository tuner regression; it is not
inferred from the synthetic P4 payload.

## Attempt ledger

| Phase/attempt | Command role | Outcome | Classification |
|---|---|---|---|
| P0 focused tests | bridge, tempered ensemble, fixed-transport step-cap tests | 28 passed in 11.90 s | identity repair verified |
| P1 first CPU all-candidate artifact | old benchmark invocation | overall `FAIL_DIAGNOSTIC_PARITY`; factor itself passed, raw cache failed | harness/evidence classification; preserved at `p1-cpu/center.json` |
| P1 repaired CPU | center and varied batches, factor required | both `PASS_DIAGNOSTIC_PARITY` | numerical pass |
| P1 repaired GPU0/XLA | center and varied batches, factor required; authoritative rerun with `CUDA_VISIBLE_DEVICES=0` | both `PASS_DIAGNOSTIC_PARITY` | numerical pass |
| P2 GPU0/XLA | two beta levels, two fresh seeds, four-chain one-result runs | `PASS_MECHANICS_PARITY` | transition pass |
| P3 GPU0/XLA | two beta levels, two fresh seeds, four chains, eight retained results after four burn-in steps | `PASS_MECHANICS_PARITY` | short-chain sensitivity pass |
| P4 identity admission | stale strict tuning payload presented to factor handoff constructor; authoritative manifest rerun with elapsed/device fields | `PASS_IDENTITY_REJECTION` | stale cross-backend rejection pass |
| measured-grid handoff regression | valid same-scope handoff and stale scope/transport rejection tests | 4 passed in 9.67 s | positive handoff path and rejection pass |
| final focused regression | q20 bridge, phase9a runner, eigensystem tests | 63 passed in 32.87 s | engineering pass |

The complete fresh artifacts are under
`docs/plans/artifacts/ssl-lstm-q20-factor-route-promotion-2026-09-04/`.
The authoritative GPU parity manifests are `p1-gpu-attempt3/`, together with
`p2-gpu-attempt2/`, `p3-gpu-attempt2/`, and `p4-identity-attempt3/`.  The
earlier `p1-gpu-attempt2/` run is retained as a diagnostic because it exposed
both GPUs rather than the plan-bound GPU0 selection; it is not the final P1
provenance receipt.  The initial failed all-candidate P1 artifact is also
retained and is not used as a factor verdict.

## Numerical evidence

The strict route is `tensorflow_eigh_strict`; the candidate is
`tensorflow_eigh_strict_factor_cached`.  Both use q=20, float64,
TensorFlow/TFP, XLA, GPU0 with `TF_FORCE_GPU_ALLOW_GROWTH=true`, and the same
static batch shapes.

On the authoritative GPU0 center batch, the factor candidate had maximum absolute
score difference `9.910960940828772e-12` and value difference `0`, matching row
classes and status codes.  On the varied row-offset batch, the maximum absolute
score difference was `1.1580092440510725e-10` and the value difference was `0`,
again with matching classes and statuses.  The factor graph exposed four
`SelfAdjointEigV2` nodes versus six
for strict; this is a descriptive performance observation, not a promotion
criterion.

The CPU-hidden center and varied runs also passed the same declared tolerances
(`value rtol=1e-10`, `score rtol=1e-9`, `atol=1e-10`).  The near-degenerate
covariance/derivative fixtures passed the focused eigensystem suite (46 tests
in the standalone suite; 63 in the final combined run).

## HMC mechanics evidence

The paired GPU transition artifact compares the same state bank, beta, step
size, leapfrog count, and stateless seeds.  Both beta `0.5` and beta `1.0`
cases passed for both fresh seeds.  The largest observed paired state
difference was about `2.61e-15`; energy-error maxima were below `5e-7`, well
under the frozen hard bound `abs(delta_h) <= 1.0`.  Acceptance bits, target
status telemetry, target values, scores, and trace counts matched.

The short-chain artifact used four chains, four burn-in transitions, and eight
retained mechanics transitions for each case.  The largest paired state
difference was about `1.87e-14`; all reported status rows were valid and all
required values were finite.  The installed TFP kernel exposes no native
divergence flag; both artifacts record
`native_divergence_status=unavailable`.  This is not interpreted as zero
divergences.  The declared finite energy-error bound is the available hard
screen in this environment.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Kernel candidate | P0/P1 identity and value/score/status parity | no identity or parity veto | stress coverage is finite, not exhaustive over all q=20 states | keep factor available for scoped diagnostics | no posterior correctness or scaling |
| Mechanics candidate | P2 paired transition parity | no nonfinite/status/energy veto; native divergence unavailable and recorded | only fixed small epsilon/L and short runs | run fresh factor-specific tuning | no HMC convergence |
| Default readiness | P0--P4 plus untouched scope-specific tuning required by parent policy | not yet satisfied | no factor tuning artifact or six-scope claim run exists | execute the follow-on factor tuning plan | no default promotion |
| Scientific interpretation | downstream posterior/whitening gates | not assessed | short diagnostic chains and no uncertainty analysis | retain strict authority and diagnostic labels | no whitening, mode discovery, superiority, or production claim |

## Inference-status table

| Evidence class | Status | Interpretation |
|---|---|---|
| Hard veto screen | passed for the tested mechanics scope | no crash, nonfinite value/score, status mismatch, identity collision, trace retracing, or allocator-policy violation was observed |
| Statistically supported ranking | not available | no ranking of strict versus factor is supported; there are too few stochastic replications and no uncertainty model |
| Descriptive-only differences | factor used fewer eigensystem nodes and was faster in the benchmark | useful for nomination only; timings and acceptance do not establish superiority |
| Default readiness | not ready | fresh factor tuning and untouched replay are still required |
| Next evidence needed | defined | exact-scope factor tuning artifact, stale-artifact rejection in the real replay loader, and an untouched downstream run under the factor identity |

## Post-run red-team

The strongest alternative explanation is that the tested state bank and small
step size avoid the spectral regions where factor-basis reuse can accumulate
roundoff.  The varied and near-degenerate fixtures reduce that risk but do not
cover the full posterior trajectory.  A transition mismatch, status-class
change, or score failure in a fresh factor-tuned scope would overturn the
mechanics promotion; a successful short chain would not overturn the need for
that scope-specific evidence.  The weakest evidence is downstream scientific
behavior: no claim-bearing replay, convergence assessment, or mode/whitening
diagnostic was run here.

## Post-execution skeptical audit

The closeout was re-audited after the final retries.  The only material
provenance finding was that the first repaired P1 GPU receipt exposed both
devices; the benchmark was repaired to enforce GPU0 and the attempt-3 receipts
were rerun and hashed.  The P4 synthetic payload was checked for an opposite
failure mode: it is incomplete by design and therefore cannot establish
positive acceptance.  The plan now says so plainly, and the valid measured-grid
handoff path is covered by four repository tuner tests.  No target, data,
bridge, tolerance, seed role, or campaign cap was changed after observing the
factor result.  Timings and acceptance remain descriptive, and no downstream
posterior or whitening claim was upgraded.  The audit verdict is
`PASS_CLOSEOUT_PROVENANCE_AND_CLAIM_BOUNDARY`.

## Reproducibility and provenance

All repaired serious artifacts record the exact command, plan path, Git
revision, worktree-status hash, Python environment, device visibility,
memory-growth policy, XLA/TF32 settings, seeds, source paths/hashes, and wall
time.  The campaign used a fresh versioned output root and never overwrote the
earlier attempts.  The authoritative P1 GPU artifacts were launched with
`CUDA_VISIBLE_DEVICES=0`; all GPU artifacts were launched with
`TF_FORCE_GPU_ALLOW_GROWTH=true` before TensorFlow import.

The authoritative receipt hashes are:

| Receipt | SHA-256 |
|---|---|
| `p1-gpu-attempt3/center.json` | `70069526139edbde9b0c808b6847720a62eb84568eb66553b927d97e6f45dd6b` |
| `p1-gpu-attempt3/varied.json` | `7439186f03fcff0f2f932b1b42b7fdc8db081555460f9fd63e622ded3e9449cc` |
| `p2-gpu-attempt2/transition.json` | `a8460bb4a2e36b8a4e69488d5cb6df56955d933f1ddde3198dd1c0a816a63ca2` |
| `p3-gpu-attempt2/chain.json` | `f9b16eaf5521a20e4fa1ecb3593d8edfc398828659097459b30f38ec3ef2a30b` |
| `p4-identity-attempt3/admission.json` | `f03719aee99d3fd1caa3f7cf2a06178513260c97fc016cb4652f87d90e97c474` |
| `p1-cpu-attempt3/center.json` | `0f43bdb3bf530ad8ad5abbec7e5cc015fea1359fff9f09ad72e35004002b04a4` |
| `p1-cpu-attempt3/varied.json` | `61229415ed7adda31e9bd41a36d10a5df3ecdeced9fef58228789b17224af49c` |

## Final decision and next plan

The factor route is `DEFAULT_ELIGIBLE_PENDING_FRESH_TUNING`, not promoted.  The
strict route remains the default and the raw-covariance cache remains vetoed.
The refreshed next step is
`docs/plans/bayesfilter-ssl-lstm-q20-factor-route-fresh-tuning-admission-plan-2026-09-04.md`.
It must bind the repaired backend identity into a fresh factor tuning scope,
run a canary before any full replay, and require the parent program's
untouched-run gates before a default edit.
