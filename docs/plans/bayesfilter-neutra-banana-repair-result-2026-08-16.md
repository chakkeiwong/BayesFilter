# NeuTra Banana Budget Repair Result (2026-08-16)

## Outcome

The corrected terminal campaign completed in `331.31 s` on GPU 0 with
float64, XLA, TF32 disabled, and TensorFlow memory growth verified before
device initialization. The terminal artifact root is:

`docs/plans/artifacts/neutra-banana-repair-2026-08-16-r3/`

The target-specific 6,000-update root-preserving banana transport passed the
fresh proposal-law replication gate on all three seeds. It did not produce an
HMC-valid candidate: warm-up passed, but the retained phase failed its
predeclared convergence/health gate and the retained exact-law screen failed.
No banana HMC result is admitted.

The prior `r2` root is preserved as non-promotable debugging evidence. It
scaled the learning-rate phase boundaries with the longer budget, so it
confounded budget and schedule. The terminal `r3` runner fixes the original
3,000-update phase boundaries at updates 1,800 and 2,550; only the final
low-rate phase is extended.

## Evidence Contract

| Item | Value |
|---|---|
| Plan | `docs/plans/bayesfilter-neutra-banana-repair-plan-2026-08-16.md` |
| Terminal artifact root | `docs/plans/artifacts/neutra-banana-repair-2026-08-16-r3/` |
| Target | 16-dimensional analytic banana, curvature `0.35`, unit Jacobian |
| Transport | Dense IAF `(32,32)`, ELU, root-preserving reverse permutation |
| Training | Batch-native reverse KL, batch `4096`, peak `LR=5e-4`, fixed 3,000-update schedule horizon |
| Arms | Baseline 3,000 updates; extended 6,000 updates |
| Fresh seeds | `13,14,15` for both arms |
| Proposal audit | 131,072 stateless draws per cell, unchanged 99.9% exact-law mean/second-moment/adjacent-cross-moment screens |
| HMC | Four chains, shared sequential controller, `L=20`, step size `0.8446953`, no NUTS, `L=1` forbidden |
| HMC warm-up | 2,000 per chain, max recent-window R-hat `1.00863` under `1.05` |
| HMC retained | Failed after first 500-draw retained chunk; hard veto `retained_chunk_health_failed` |
| Integrity | 25 terminal artifacts; all SHA-256 hashes passed |
| Git commit recorded | `3030d86df9cb00346df82c7c19f015c09c7c6e1f` |

## Training Results

| Arm | Seed 13 | Seed 14 | Seed 15 | Gate |
|---|---:|---:|---:|---:|
| 3,000 updates | Pass | Pass | Pass | `3/3` |
| 6,000 updates | Pass | Pass | Pass | `3/3` |

The extended arm had proposal importance-ESS fractions `0.99719`, `0.99673`,
and `0.99763` for seeds 13, 14, and 15. These are descriptive diagnostics,
not evidence of superiority. Its selected losses were `8.014976`, `7.993247`,
and `7.990998`; the corresponding selected checkpoints were updates `6000`,
`6000`, and `5250`. All proposal-law screens passed for every extended seed.

This repairs the specific training repeatability failure seen in terminal `r6`
under seeds `10,11,12`: under the new fresh seeds and fixed schedule, the
6,000-update protocol is viable as a target-specific proposal-training
candidate. It does not establish that 6,000 updates is required, optimal, or a
general default.

## HMC Results

The controller selected `L=20` with step size `0.84469532597`. Warm-up reached
readiness at 2,000 transitions per chain with maximum rank-normalized split or
folded R-hat `1.00862695`. States, target values, scores, log-acceptance
values, movement, and the declared energy-error diagnostic were finite/passed;
native TFP divergence telemetry remains unavailable and is not interpreted as
zero divergences.

The first retained 500-draw chunk failed the retained full-convergence health
gate: its maximum R-hat was `1.03027`, above the retained threshold `1.01`,
although bulk and tail ESS were already above 1,800. The controller correctly
rejected the kernel rather than treating acceptance or ESS alone as sufficient.

The retained exact-law screen also failed independently: coordinate mean index
5 was `0.08315` with standard error `0.02215`, a standardized discrepancy of
`3.75`. Coordinate second moments and adjacent cross moments passed in the
available retained chunk. Because retained convergence and exact-law agreement
both failed, there is no valid banana posterior/HMC result.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| 6,000-update proposal training | 3/3 fresh exact-law proposal audits | Passed | Three new seeds are bounded evidence | Retain as a viable banana training arm for further HMC-specific repair | Universal budget or optimality |
| Banana sequential HMC | Retained R-hat/health and post-HMC exact-law screens | Vetoed | Whether kernel tuning, initial states, or transport geometry causes retained drift | Diagnose HMC failure with a fresh target-specific kernel/initial-state plan before rerunning | HMC impossibility or posterior correctness |
| `r2` comparison | Clean budget isolation | Invalidated by schedule confound | It remains useful only for debugging | Preserve, do not promote | Evidence that more updates alone caused `r2` behavior |
| SSL-LSTM transfer | Target-specific adapter and evidence | Not authorized | No SSL-LSTM evidence in this campaign | Do not transfer settings | SSL-LSTM readiness |

## Inference Status

| Evidence class | Status |
|---|---|
| Hard veto screen | Proposal training passed for the extended arm; HMC retained convergence and retained exact-law screens failed |
| Statistically supported ranking | None; no superiority or budget ranking is supported |
| Descriptive-only differences | Loss, ESS fraction, ratio SD, runtime, acceptance, and leapfrog count |
| Default-readiness | Not supported |
| Next evidence needed | A reviewed banana HMC repair isolating kernel/initial-state/transport-geometry effects; proposal training need not be repeated unless that plan changes the transport |

## Red-Team Note

The strongest alternative explanation for the HMC failure is not insufficient
training: all three extended proposals pass exact-law audits, while the
sequential controller fails after entering HMC. The remaining candidates are
the fixed identity mass and tuned leapfrog kernel, the initial-state bank, or
transport geometry under HMC. A longer retained run cannot repair a failed
first-chunk R-hat gate without a predeclared controller change.

The weakest evidence is any comparison of `r2` versus `r3`, because `r2` used
different learning-rate phase timing and is explicitly non-promotable.
