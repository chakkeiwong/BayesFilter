# SSL-LSTM q=20 physical replica-travel repair reset memo (2026-08-11)

## State

Update, 2026-08-11 09:35 +08:00: the authorized exact `24x1` continuation is
complete and the ratio-0.50 candidate is rejected at the transition-1000 warm-up
cap.  The terminal artifact status is `RESUMED_MATERIAL_WARMUP_NOT_READY`.

- The chain resumed exactly from the verified transition-500 checkpoint with the
  unchanged master seed, stateless transition index, ladder, steps, `L=8`, and
  four chains.
- It completed 500 additional transitions in `5,344.60 s`; all 1,000 total draws
  are discarded warm-up and there are zero retained draws.
- No hard gate failed.  All 50 continuation manifests and 550 continuation tensor
  receipts independently verify; together `r8+r11` contain 100 verified manifests
  and 1,200 verified tensors.  Every continuation cache residual is exactly zero.
- The final 300-draw window failed modern R-hat at `1.141610 > 1.05`.  Travel
  passed with per-chain returns `[6,4,5,5]`, and hot forgetting passed with local
  sign changes `[1,6,2,3]`.
- Earlier new-window R-hat values were `1.133129`, `1.056483`, `1.159304`, and
  `1.073534` at transitions 600--900.  The nonmonotone path does not support the
  claim that more unchanged warm-up is steadily repairing the candidate.
- No posterior archive, posterior summary, predictive validation, exhaustive-mode,
  superiority, or default-readiness action is allowed from this run.

Binding terminal artifact:

`docs/plans/artifacts/ssl-lstm-q20-physical-replica-travel-repair-2026-08-10/r11-material-24x1-resumed/material.json`

SHA-256: `0fbec0c372008d406953908a30b6aa66a27d843781a93dba5ae52cd98235c66b`

The earlier numerical-invalidity verdict is retracted.  The bounded
numerical-materiality canary passed, but `12x2` failed as a performance repair.

- SMC receipt provenance was repaired and verified without target reevaluation.
- Monolithic exact HMC is valid but unaffordable: `235.085 s` per cached
  four-thread transition.
- Distributed exact HMC and swaps were implemented in TensorFlow and passed
  analytic/TFP mechanics parity, invalid-path self-rejection, and focused tests.
- The `24 workers x 1 row` four-chain checkpoint passed all hard validity,
  communication, and acceptance screens.  One round trip was observed
  descriptively in 25 transitions, but the run projected to `21,607.97 s` under
  the frozen 50% margin versus the `20,000 s` cap.
- The final `12 workers x 2 rows` performance repair failed exact terminal cache
  parity.  A changed-pair diagnosis showed the same row's value changes by up to
  `3.70e-7` and score by up to `4.66e-6` when only its batch companion changes.
- The inherited `1e-9` value and `1e-8` score cache tolerances were not calibrated
  to this target and were incorrectly elevated into a continuation veto.  The
  measured differences are about `1e-8` relative and plausibly ordinary XLA
  floating-point/eigensolver variation.  Do not reject `12x2` from this evidence
  alone.
- The identical-randomness `r7` canary found identical path validity, HMC
  accept/reject decisions, and swap decisions for contiguous and shifted `12x2`
  pairings versus `24x1`.  Maximum log-acceptance perturbation was `7.52e-8` and
  proposal-state perturbation `1.40e-8`.  Treat the rounding differences as
  immaterial for this canary.
- `12x2` is not a useful speed repair: checkpoint-equivalent cost was `26.008 s`
  versus `11.373 s` for the same-run `24x1` reference.  Its conservative material
  projection was `50,715.80 s`.
- No material HMC, posterior archive, predictive validation, or NeuTra action was
  launched.  GPU 0 was not used; all final work was CPU-only with GPU hidden.

Terminal result:

`docs/plans/bayesfilter-ssl-lstm-q20-physical-replica-travel-repair-result-2026-08-10.md`

## Binding artifacts

| Artifact | SHA-256 |
|---|---|
| SMC receipt recovery | `3aea988e7b27381a6b62e7a2d452db8251b9bd7d8b9f5e68ad08fcbe711b6d97` |
| Monolithic timing | `d4a0be4b4ac0a8fe5d4daf1a4a3bfb1425f774e393231a2f987aa5fe248ed4ed` |
| Passed distributed canary `r3` | `bfcbb5840622e761e052b5dfe398c6ae194570765294a4f1d159091b1569d471` |
| Four-chain checkpoint `r4` | `8276947db5785786567c5194b469c0938907820faf8d1bafd0265b1d4f87adab` |
| Failed topology canary `r5` | `08e9d29fee2af56aeadc3622f01a6f97487384c4446e01f16fc00dedb2ecb3ac` |
| Cache-pair diagnosis | `1a29bd118fb75481aa86dde0dd6a3353d4f7b729b6e9c6cf0bf55ac2e5774363` |
| Materiality reporting failure `r6` | `408645656995f123f334a4c92e1c8eb779cd9dd2633540a89e3029b0cd93caa9` |
| Materiality result `r7` | `5de1e5d217abd9ae293aff81356955c799ed6328e6a66670b019220f6d27aad2` |

## Failure classification

| Question | Answer |
|---|---|
| Harness invalid? | The `r5` failure was not merely a harness bug; a separate immutable-state diagnosis reproduced pair-group dependence. |
| Exact physical target invalid? | Not established.  The one-row route passes; the batch-two execution is invalid for the row-independent claim. |
| Distributed HMC mechanics invalid? | No evidence of that.  Analytic/TFP parity and the `24x1` exact-target gates passed. |
| Current topology repair failed? | Numerically no for this canary; as a performance repair yes, because `12x2` is more than twice as slow. |
| Material campaign affordable? | `12x2` is not.  `24x1` has a raw minimum-run estimate below `20,000 s` but misses the predeclared 50%-margin admission screen. |
| Research direction rejected? | No.  This is a target-topology failure plus a budget stop, not a replica-exchange or scientific-model rejection. |

## Verification already run

```bash
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_ssl_lstm_q20_physical_12x2_cache_parity_diagnosis.py \
  tests/test_ssl_lstm_q20_physical_distributed_replica_12x2_canary.py \
  tests/test_distributed_replica_exchange_tf.py
```

Observed: `8 passed` with two TensorFlow Probability deprecation warnings.
Both new runners also passed `py_compile`, and the touched files passed
`git diff --check`.

## Next justified action

The old budget-policy question is resolved by the completed eight-hour campaign
authorization and is no longer the blocker.  Do not continue the unchanged
ratio-0.50 candidate beyond transition 1000 and do not use any warm-up draw as a
posterior draw.  The next smallest discriminating step is target-free analysis of
the verified cold trace: separate between-chain mode occupancy from within-mode
location/scale disagreement.  Only after that decomposition should a new sampler
hypothesis be reviewed.  A mode-occupancy failure points toward a global
tempering/swap or initialization repair; within-mode disagreement points toward a
local mass/trajectory repair.  The decomposition is explanatory only and cannot
promote the posterior.

Historical next-action text below is superseded.

## Final repair handoff

The target-free trace diagnosis and bounded dense-mass repair are complete.

- Terminal failed-window occupancy fractions were
  `[0.333,0.727,0.660,0.447]`; sign R-hat was `1.2325`.
- Source-center-residual R-hat was `1.1299`; between-chain sum squares were
  `0.165` occupancy versus `3.778` residual.  Treat the cause as mixed, with
  larger observed residual disagreement, not as a proved causal percentage.
- The shared distributed exact HMC helper now has optional fixed dense mass with
  identity as its unchanged default.  Focused mechanics/TFP tests pass.
- Dense mean-local-precision mass with `step=0.35, L=8` passed the 100-transition
  viability screen: zero invalid paths, all acceptance means in band, all adjacent
  pairs communicated, hot changes `[3,1,3,7]`.
- `step=0.70, L=8` failed acceptance and hot forgetting.  `step=0.35, L=4`
  failed hot forgetting for chain 1 despite valid acceptance and travel
  `[2,2,2,5]`.  Do not promote either arm.
- The viable `0.35/L8` arm costs `11.081 s/transition`; a fresh minimum
  300-warm-up plus 1,000-retained campaign requires about `14,405 s` before
  startup/finalization and did not fit the remaining absolute deadline.
- No fresh claim run, retained draw, posterior archive, or predictive validation
  was launched.  GPU 0 was never used.

Next campaign starting point: fresh balanced sign-separated chains, ratio-0.50
ladder, fixed mean mapped-local-precision mass, cold step `0.35` scaled by
`1/sqrt(beta)`, `L=8`, four chains, `24x1` CPU/XLA, and the same warm-up,
R-hat/ESS, travel, forgetting, acceptance, numerical, and nonclaim gates.  The
candidate is viable, not accepted.  A new campaign needs at least the measured
`14,405 s` plus explicit startup/finalization margin; do not continue the old
warm-up or reuse tuning draws.

Do not run a fresh `12x2` checkpoint; it cannot repair cost.  The only viable
measured route is `24x1`.  Its historical checkpoint-equivalent cost `11.081 s`
gives a raw 1,300-transition estimate of about `14,405 s`, below the `20,000 s`
hard wall cap, but its prospectively frozen 50%-margin admission projection is
`21,607.97 s`.  Whether to replace that prelaunch margin screen with the hard
runtime cap is a human budget-policy decision, not a numerical-validity issue.
Until that decision, preserve the accepted SMC result solely as a two-known-region
mass authority and preserve all posterior, predictive, and exhaustive-mode
nonclaims.

## Resilient execution

Continue using detached transient user services with explicit wall caps, absolute
append-only logs, unique output roots, atomic progress, and short artifact polls.
The final canary and diagnosis both survived independently of the client stream.
