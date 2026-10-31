# Surrogate-HMC repair continuation, 2026-10-12

The question is whether the shared analytical implementation computes every
parameter score in one time recursion, at FP32 / N4032, accurately and cheaply
enough for surrogate HMC. The additional 36,000 GPU wall seconds include
compilation, rejected candidates, and failures. This note is in progress.

The covariance-guided force remains optional, without canonical Contract-E LEDH
admission. Corrected HMC uses an independent Metropolis endpoint. The benchmark
normal prior remains a mechanics fixture, not an approved inferential prior.

## Completed evidence

LGSSM shared-map values and scores were bit-identical to the paired baseline.
Warmed calls took 16.20/16.44 seconds versus 42.34/42.97 seconds; allocator peaks
were 560 versus 603 MB. These are descriptive timings, not a statistical speed
ranking. Evidence: artifacts/surrogate-hmc-lgssm-shared-map-20261012-01/.

SIR T50/N4032 FP64 parameter-zero derivative was -67.7706117257 versus
fine-width Richardson -67.7706362498. Gap 0.00002452 plus width uncertainty
0.01842 is below the declared 1% tolerance 0.6777. This checks one direction,
not the complete score. Evidence:
artifacts/surrogate-hmc-sir-shared-map-fp64-t50-20261012-01/.

Centered normalized moments and barycenters passed 73 focused CPU regressions
and all five small-filter derivative checks. Full-scope precision is running
under artifacts/surrogate-hmc-centered-primal-precision-20261012-01/.
FP64 is a same-filter precision reference, not a model oracle.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Limit |
|---|---|---|---|---|---|
| Continue repair | Shared recurrence and small checks pass | Full precision incomplete | SIR sensitivity | Finish precision, repair calibration, fresh tuning, HMC mechanics | Master incomplete |
| Keep FP32 | Owner's 1% final-output criterion | Reject failing controls | Cross-data robustness | Two fresh validation datasets per candidate | No automatic FP64 default change |

| Inference status | Finding |
|---|---|
| Hard veto screen | SIR FP32 score discrepancy unresolved |
| Statistically supported ranking | None |
| Descriptive differences | LGSSM timing reduction and saved-input numerical errors |
| Default readiness | FP32 owner-selected; controls require own-scope validation |
| Next evidence needed | Fresh controls, precision, endpoint checks, HMC mechanics and cost |

Alternative explanation: a speed gain may come from changed numerical work;
paired values/scores and source snapshots address that locally. Weakest evidence:
full SIR precision and no converged posterior run. An older calibration call
chain must be replaced before tuning so calibration and HMC use the same filter.

Authoritative budget:
artifacts/surrogate-hmc-repair-continuation-20261012-01/budget.json.
See the active checkpoint and master for commands and next action.

The centered-primal precision phase finished: LGSSM, KSC, predator-prey and range bearing pass 1%. SIR FP64 timed out at800s; comparison unresolved, not a demonstrated numerical failure. Shared Gaussian candidate passed19 dense checks; broader regression remains pending.

## Completed precision and pre-tuning review

The next paired phase completed at all five full scopes, including SIR FP64.
Evidence: artifacts/surrogate-hmc-shared-gaussian-precision-20261012-01/precision.json.
Maximum scaled score differences: LGSSM0.00002018, KSC0.00023688,
predator0.00131408, SIR0.44503549, range0.00275933.
Four pass the1% screen. SIR fails despite likelihood error only0.00115;
the smoother-reset family is its next planned repair. The earlier timeout
is now resolved and must not be described as the current obstacle.

Descriptive warmed B2 FP32 seconds, direct/matrix: LG16.78/29.63,
KSC4.90/5.85, predator16.00/11.69, SIR49.95/23.25, range8.51/10.54.
These single measurements nominate frozen execution controls; they do not
establish a statistical speed ranking. Fresh tuning checks precision again
on two new validation datasets before permitting mechanics.

The136 numerical/call-chain regressions,47 master tests, and all five tiny
filter derivative checks under both direct and matrix contractions passed.
Master tests caught a misplaced scope-setting block; it was repaired before
execution. Tuning retains full health diagnostics and exact/refined references
where available. Both mechanics trajectories are checked.

Pre-run skeptical audit: own-scope source/data identities prevent stale
tuning reuse; independent MH endpoints remain unchanged; nomination metrics
cannot certify inference. Remaining GPU allocation29870.08s supports bounded
tuning caps22300s plus mechanics caps6000s. Independent scope failures trigger
repair and do not terminate the other scopes. Full posterior inference has
not been run. The benchmark prior and public tuner authority remain unresolved
for promotion.
