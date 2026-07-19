# SSL-LSTM NeuTra Phase 6 Transformed-HMC Tuning Plan

Date: 2026-07-16

Status: `PHASE6_COMPLETE_IDENTITY_MASS_KERNELS_FROZEN_AFTER_H_REPAIR`

## Objective And Entry Conditions

Determine whether exact HMC in the immutable G/H transport coordinates can
produce a viable frozen kernel from all four inverse-mapped original starts.
This phase tunes sampler mechanics only; it does not retain posterior evidence
or support a convergence, posterior-correctness, predictive, superiority, or
default claim.

Entry conditions pass for planning:

- Phase 5 decision `PHASE5_EXACT_TRANSFORMED_TARGET_PASSED`;
- immutable G/H payload hashes remain
  `6e147d5b...dedc354` and `ed0e4260...c9120fb`;
- locked target semantic SHA-256 remains `549efdf2...18f719e`; and
- the four original starts and their G/H inverse coordinates are preserved.

The user's 2026-07-16 instruction to create, review, and execute this plan
authorizes the implementation, focused checks, and bounded Stage A timing/
mechanics canary below. The larger Stage B ladder remains closed until the
canary receipt is reviewed and a measured cumulative cap is appended here.

## Research Intent And Evidence Contract

| Field | Prospective contract |
| --- | --- |
| Main question | Does identity-mass HMC in both exact G/H `z` charts admit a finite, moving, per-chain viable kernel after a target-specific step/trajectory search? |
| Exact baseline | Identity mass in each NeuTra `z` chart, not ordinary-HMC or historical DSGE tuning settings |
| Candidate mechanism | G and H frozen nonlinear charts; no transport retraining or selection by HMC outcome |
| Primary pass | G and H each produce a predeclared kernel that passes short mechanics/scale screens and a fresh longer confirmation with all four chains moving, finite telemetry, per-chain acceptance inside the declared viability screen, and no exposed native divergence |
| Promotion veto | Target/payload/source drift, nonfinite state/value/log-acceptance, any unmoved chain, invalid mass semantics, positive native divergence when exposed, missing per-chain telemetry, seed overlap, or failed longer confirmation |
| Repair trigger | All-chain high acceptance including `1.0`, all-chain low acceptance, acceptable mechanics but no scale bracket, or identity-mass cross-chain inefficiency |
| Continuation veto | Broken Phase 5 exactness, HMC implementation invariant failure, corrupt artifact, GPU/XLA route unavailable, or resource cap exhausted |
| Explanatory only | Continuous acceptance, energy/log-acceptance summaries, runtime, movement distance, and G/H differences among viable candidates |
| Nonclaims | No retained-sample admission, convergence, posterior correctness, support completeness, predictive validity, ranking, or default readiness |

## Skeptical Pre-Execution Audit

Status: `PASSED_FOR_STAGE_A_ONLY_2026_07_16`.

- Wrong baseline is prohibited: start with identity `z` mass because that is
  the NeuTra design claim. Do not import ordinary-HMC mass, step, or trajectory.
- Proxy promotion is prohibited: acceptance alone cannot freeze a kernel;
  movement, finiteness, per-chain telemetry, and fresh confirmation are gates.
- Unfair comparison is prohibited: run the same prospective ladder and budgets
  for G and H; do not stop after whichever candidate looks favorable.
- Hidden assumption: the current
  `bayesfilter/inference/fixed_transport_hmc_tuning.py` implementation supports
  identity mass only. It does not implement the optional diagonal/dense repair.
- Environment mismatch: serious execution must use physical GPU 1, TensorFlow /
  TFP `float64`, and whole-chain XLA. The existing tuner defaults to non-XLA,
  so the future runner must explicitly set and verify XLA authority.
- Capability mismatch: the locked target intentionally advertises target-only
  XLA capability (`xla_hmc_ready=false`). The Phase 6 runner therefore needs a
  narrow HMC-scoped batch adapter whose authority is bound to the passing Phase
  5 GPU/XLA receipt. Do not mutate or globally promote the locked target's
  capability metadata.
- Resource uncertainty is handled prospectively: Stage A has an exact command,
  seed ledger, fresh namespace, `2,400`-second in-run cap, and `2,700`-second
  external timeout. Stage B cannot launch until Stage A supplies first-call and
  warm-call timings for both G and H and the measured cap is appended.
- Baseline audit conclusion: a dedicated Phase 6 runner around the existing
  `FullChainHMCConfig`/reusable full-chain primitive is required. The generic
  fixed-transport tuner is not admissible because it broadcasts one start to
  all chains and selects on aggregate acceptance. The shared HMC primitive
  itself is admissible because standard trace policy exposes draw-by-chain
  acceptance, log-acceptance, target values, samples, and native divergence
  only when TFP actually supplies it.

Focused native review disposition, 2026-07-16: `PASS_AFTER_REPAIR`.

- The first review rejected the draft runner because it treated the four A0
  affine latent starts as target parameters. The runner now reconstructs
  `theta = center + z_A0 @ factor.T` from the exact A0 target lock before each
  frozen transport inverse, and a regression test binds the Phase 5 G radii.
- The first review also rejected largest-step selection because it did not
  target acceptance `0.70`. Scale selection now minimizes the worst per-chain
  deviation from `0.70` with a tolerance-stable larger-step tie-break.
- Follow-up review repaired movement accounting to include the initial-to-first
  transition, permits both symmetric scale expansions when G/H point in
  opposite directions, reuses the compiled scale runner, verifies runner/HMC
  sources against the canary before Stage B, and avoids relabeling every failed
  identity ladder as proven bad geometry.
- Final focused checks: script/test compilation passed, `10` focused CPU-hidden
  tests passed, and `git diff --check` passed. CPU-hidden tests are engineering
  evidence only; no HMC transition was run by them.

## Implementation And Test Work Before HMC

1. Add an SSL-LSTM HMC-scoped batch bridge identical in value/score semantics
   to the Phase 5 bridge. It may advertise XLA HMC authority only within the
   Phase 6 scope and only by binding the exact Phase 5 GPU receipt, source
   hashes, target signature, and G/H payload hash. Leave the locked target's
   global capability metadata unchanged.
2. Bind each exact loaded artifact through `FixedTransportValueScoreAdapter` and
   revalidate source/payload signatures at runner start.
3. Add a mechanics-only CUDA XLA canary that starts from all four preserved `z`
   rows and records native divergence status exactly as exposed or unavailable.
4. Identity mass means momentum covariance `I`, precision `I`, kinetic energy
   `0.5 * p^T p`, and no Cholesky transform. Test this analytic fixture plus
   per-chain acceptance shape, no aggregate masking, seed separation, fixed
   candidate order, scale expansion, and no-overwrite output.
5. Review the final exact command and runner before launching any HMC call.

## Prospective Sequential Ladder

1. Identity-mass mechanics/timing canary: for each of G/H, run two calls with
   `4` results, `2` burn-in steps, `epsilon=0.01`, and `L=2`. The first call
   measures compile plus execute; the second measures warm execute. Both must
   have finite samples/target/log-acceptance, all four chains moving, four
   distinct preserved starts, and zero native divergences when available.
2. Target-specific step-scale pilot: use `16` results and `8` burn-in steps at
   fixed `L=4` for the ordered initial grid `0.05, 0.10, 0.20, 0.40`. If either
   chart has every arm above the pilot band, run `0.80, 1.60` for both charts;
   if either has every arm below the band, run `0.025, 0.0125` for both charts.
   Do not expand after a mixed no-pass outcome. A pilot arm is viable only when
   every chain has acceptance in `[0.50, 0.90]`, movement rate at least `0.25`,
   RMS jump distance at least `0.05`, finite telemetry, and no positive exposed
   native divergence. Select the viable arm minimizing the maximum absolute
   per-chain deviation from target acceptance `0.70`, breaking exact ties in
   favor of the larger step. This is a predeclared tuning target, not a claim
   of statistical superiority. Acceptance `1.0` is a high-side repair trigger,
   never a pass.
3. Conditional trajectory grid: at each chart's selected step size, run
   `L=2,4,8,16` with the same short-run sizes and fresh seeds. Apply the same
   viability gate and select the first viable arm in fixed priority
   `L=8,4,16,2`. This changes `L` at fixed `epsilon`; it does not tune only
   `epsilon * L`.
4. Fresh longer confirmation: use `64` results, `32` burn-in steps, and seeds
   disjoint from mechanics and tuning. Every chain must have acceptance in
   `[0.55, 0.85]`, movement rate at least `0.50`, RMS jump distance at least
   `0.05`, finite samples/target/log-acceptance, and no positive exposed native
   divergence.
5. Freeze a kernel only if every hard gate passes. If multiple candidates pass,
   choose only by a predeclared rule; do not claim statistical superiority.
6. If identity mass fails only an explicitly declared geometry repair trigger,
   stop and draft/review a separate diagonal/dense mass implementation plan.
   Do not improvise mass adaptation inside this run.

## Required Artifacts

- exact runner and focused tests;
- source/payload/target binding manifest;
- inverse-start receipt;
- identity-mass semantic and analytic fixture receipt;
- mechanics, scale-pilot, trajectory-grid, and confirmation JSON;
- complete command/device/JIT/TF32/seed/budget ledger;
- immutable selected-kernel manifest or blocker result; and
- Phase 7 retained-admission plan only after a valid frozen kernel exists.

## Seed Ledger

| Role | G second seed word | H second seed word |
| --- | ---: | ---: |
| Canary first/warm | `6101`, `6102` | `6201`, `6202` |
| Scale pilot candidates | `6300 + candidate index` | `6400 + candidate index` |
| Trajectory candidates | `6500 + candidate index` | `6600 + candidate index` |
| Fresh confirmation | `6701` | `6801` |

Every seed is paired with first word `20260716`. Candidate indices follow the
prospective order in this plan. No seed may move between roles after outcomes.

## Exact Stage A Command And Resource Stop

Environment: conda `tfgpu`, physical GPU 1 exposed as logical GPU 0,
TensorFlow/TFP `float64`, TF32 enabled, whole-chain XLA enabled.

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-phase6-pyc CUDA_CACHE_PATH=/tmp/bayesfilter-phase6-cuda timeout 2700 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_neutra_phase6_transformed_hmc_tuning_2026_07_16.py --stage canary --output docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-6-trial0-gh/canary.json --wall-cap-seconds 2400
```

The script checks the cap before and after each HMC call and refuses overwrite.
The external timeout is the cancellation backstop for an in-flight compile or
transition call. Stage A is mechanics/timing evidence only and cannot nominate
or freeze a kernel.

## Stage B Budget Freeze

Status: `PASSED_AND_FROZEN_2026_07_16`.

Stage A authoritative receipt:

- path:
  `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-6-trial0-gh/canary.json`;
- SHA-256:
  `175edddb56907cc5880cae02643e28ac35b33cff1b945775d5bc1b8b911d9c6c`;
- decision: `PHASE6_STAGE_A_CANARY_PASSED_BUDGET_FREEZE_REQUIRED`;
- total wall time: `328.7869` seconds;
- G first compile/execute and warm execute: `161.1006` and `0.3669`
  seconds; and
- H first compile/execute and warm execute: `163.5704` and `0.3530`
  seconds.

Both charts had finite samples, accepted/proposed target values,
log-acceptance ratios/corrections, and all four chains moving on both calls.
TFP exposed no native divergence boolean; this remains
`unavailable_not_zero`. Acceptance was `1.0` on the deliberately conservative
canary and is a Stage B high-side expansion trigger, not a kernel pass.

Worst-case Stage B projection:

- four graph builds (G/H pilot plus G/H confirmation), conservatively charged
  at `4 * 163.5704 = 654.2816` seconds;
- maximum initial plus both-direction expansion, trajectory, and confirmation
  work is `6,048` transition-leapfrog units;
- the slower canary warm rate was `0.3669 / 12 = 0.03058` seconds per unit,
  projecting at most about `184.9` warm-execution seconds; and
- projected total is about `839.2` seconds before process, load, serialization,
  and runtime variability.

The Stage B cumulative cap is `1,800` seconds, more than twice the measured
projection. The exact command is:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-phase6-pyc CUDA_CACHE_PATH=/tmp/bayesfilter-phase6-cuda timeout 2100 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_neutra_phase6_transformed_hmc_tuning_2026_07_16.py --stage ladder --canary-receipt docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-6-trial0-gh/canary.json --canary-sha256 175edddb56907cc5880cae02643e28ac35b33cff1b945775d5bc1b8b911d9c6c --output docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-6-trial0-gh/ladder.json --wall-cap-seconds 1800
```

The in-run cap is checked between every HMC call; the external timeout is the
backstop for an in-flight compile/call. Stage B must stop without partial
kernel freeze if either chart lacks a passing fresh confirmation.

Native review disposition: `PASS_FOR_STAGE_B`. Receipt binding, device/XLA,
source hashes, four reconstructed A0 starts, G/H symmetry, finite proposal
health, per-chain vetoes, target-`0.70` scale selection, deterministic
trajectory priority, disjoint seeds, no-overwrite behavior, and all-or-nothing
G/H freeze were inspected. The focused suite passed `12` tests; the exact
shared dynamic-leapfrog subset passed `4` tests. A broader shared-HMC subset
had `24` passes and four unrelated stale exact-trace-key assertions because
the current trace now also exposes proposed-target/correction health fields;
this lane did not change shared runtime or those tests.

### Stage B Artifact Failure And R2 Repair

The first Stage B HMC process completed its calls but failed before writing
`ladder.json`: shared explanatory HMC telemetry intentionally contains IEEE
`NaN` when a summary is unavailable, while the Phase 6 serializer correctly
used strict JSON with `allow_nan=False`. The core Phase 6 gates separately
checked actual samples, accepted/proposed target values, log-acceptance ratios/
corrections, movement, acceptance, and native divergence status. However, the
in-memory results were lost and are not admissible evidence. No ladder receipt,
kernel freeze, candidate decision, or scientific result exists from that run.

Classification: `ARTIFACT_SERIALIZATION_FAILURE`, not target, HMC, transport,
candidate, geometry, or scientific-direction failure.

The repair converts only nonfinite explanatory scalars to explicit JSON strings
`"NaN"`, `"Infinity"`, or `"-Infinity"` before strict serialization. It does
not change HMC, seeds, gates, candidate order, or selection. A regression test
parses the output with nonfinite JSON constants forbidden. Post-repair checks:
script/test compilation passed, `15` focused plus shared dynamic-runner tests
passed, and `git diff --check` passed.

Because the runner hash changed, the old canary cannot authorize the repaired
ladder. Run a fresh source-bound `r2` canary under the same `2,400`-second cap
and `2,700`-second timeout, then run a fresh `r2` ladder only if it passes. The
fresh namespaces are:

- `.../phase-6-trial0-gh/canary-r2.json`; and
- `.../phase-6-trial0-gh/ladder-r2.json`.

The `r2` canary command is the Stage A command above with output changed to
`canary-r2.json`. After it passes, bind its exact SHA-256 into the otherwise
unchanged Stage B command, change output to `ladder-r2.json`, and retain the
same `1,800`-second in-run cap and `2,100`-second external timeout. The first
failed Stage B attempt does not relax any threshold or authorize seed changes.

### R2 Ladder Result And H Confirmation Repair

The strict `r2` ladder receipt is
`docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-6-trial0-gh/ladder-r2.json`,
SHA-256
`6065d862f7dd6aeaea5db57a10f7d4a06be7292a93ffac4e4320e689f7533c51`.
It completed in `794.0802 / 1800` seconds with decision
`PHASE6_IDENTITY_MASS_CONFIRMATION_FAILED`.

- Both charts required the prospectively declared high-side scale expansion.
- `epsilon=0.8` was the only viable scale for both charts; `1.6` was too low
  acceptance with insufficient per-chain movement.
- G selected `L=4` and passed its fresh confirmation with per-chain acceptance
  `[0.78125, 0.6875, 0.734375, 0.71875]`.
- H selected `L=8`; all finite/movement/divergence gates passed, but fresh
  acceptance `[0.546875, 0.609375, 0.671875, 0.53125]` missed the lower `0.55`
  bound in two chains by one and two accepted decisions.
- H's prospectively tested adjacent `L=4` rung was viable in the trajectory
  grid. Thus the result rejects the selected H confirmation kernel, not the
  target, harness, identity mass, NeuTra direction, or scientific idea.

One bounded one-change repair was authorized and executed: H only, fixed
`epsilon=0.8`, adjacent `L=4`, `64` results, `32` burn-in, fresh seed
`(20260716, 6901)`, unchanged confirmation gates, no mass adaptation, no new
candidate search, and no threshold relaxation. Shortening the trajectory is
the direct response to an acceptance-low-only failure. G is not rerun; its
passing confirmation remains bound to the ladder receipt. No G or H kernel is
frozen unless this repair passes.

The H confirmation compile/execute cost was `172.9166` seconds. The executed
repair cap was `600` seconds with a `900`-second external timeout:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-phase6-pyc CUDA_CACHE_PATH=/tmp/bayesfilter-phase6-cuda timeout 900 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_neutra_phase6_transformed_hmc_tuning_2026_07_16.py --stage h-confirmation-repair --ladder-receipt docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-6-trial0-gh/ladder-r2.json --ladder-sha256 6065d862f7dd6aeaea5db57a10f7d4a06be7292a93ffac4e4320e689f7533c51 --output docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-6-trial0-gh/h-confirmation-repair.json --wall-cap-seconds 600
```

Review disposition: `PASS_AFTER_PROVENANCE_REPAIR`. The repair runner verifies
the exact ladder hash, predecessor runner hash, unchanged HMC/target/adapter/
loader sources, G pass, H acceptance-low-only failure, and prospectively viable
H `L=4` rung. Its new source hash is recorded in the repair receipt rather than
incorrectly equated to the predecessor runner hash.

The single repair authority is exhausted. Its passing result is recorded in
the final closeout below; no further Phase 6 candidate or seed search remains
authorized.

## Stop And Handoff

Stop immediately for any continuation veto. A large-step or long-trajectory
candidate-local failure is not a continuation veto; execute the next
predeclared candidate. A conservative-canary failure, source/receipt drift,
corrupt telemetry, unavailable GPU/XLA route, exhausted cap, or lack of a
prospectively permitted expansion is a stop. Do not delete chains, replace
starts, reuse tuning samples, relax diagnostics, or treat unavailable
divergence telemetry as zero divergences.

Phase 7 may receive only a source-bound preflight-passing transport, exact
target signature, identity or separately reviewed mass convention, frozen step
size and trajectory length, final tuning state, and independent retained seeds.

## Final Closeout

Phase 6 closes with decision
`PHASE6_IDENTITY_MASS_KERNELS_FROZEN_AFTER_H_REPAIR`. The authoritative final
receipt is
`docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-6-trial0-gh/h-confirmation-repair.json`,
SHA-256
`dc340ab2570032a85062d0ec9cd8c9e020c41a133ec9d11b78982502ff08b9b2`.

- G kernel: identity mass, `epsilon=0.8`, `L=4`, trajectory length `3.2`.
- H kernel: identity mass, `epsilon=0.8`, `L=4`, trajectory length `3.2`.
- H repair acceptance was `[0.609375, 0.625, 0.625, 0.6875]`; all four
  movement rates were at least `0.609375`; all core telemetry was finite; and
  native divergence remained unavailable, not zero.
- These are tuning-kernel handoff artifacts only. No tuning sample is retained
  evidence, and Phase 6 does not establish convergence, posterior correctness,
  predictive validity, transport superiority, or default readiness.

Detailed interpretation and manifests are in
`docs/plans/bayesfilter-ssl-lstm-neutra-phase-6-transformed-hmc-tuning-result-2026-07-16.md`.
