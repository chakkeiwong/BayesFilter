# NeuTra reverse-funnel architecture and tuning campaign (2026-08-15)

## Research intent ledger

| Field | Predeclared statement |
|---|---|
| Main question | Does the reverse-funnel failure come from insufficient representational capacity, from the generic full-reversal ordering, or from failure to tune optimization for each architecture? |
| Candidate mechanisms | Compare a one-stage exact-compatible map, a three-stage root-preserving map, the existing three-stage full-reversal map, and a wider root-preserving map. |
| Baseline | Existing three-stage `(100,100)` IAF, full coordinate reversal, first-stage unbounded conditional-scale path, stage caps `(4,0.5,0.5)`. |
| Primary promotion criterion | On an untouched iid proposal audit, all predeclared exact-funnel moment, conditional-residual, and two tail screens pass their separate 99.9% intervals. This nominates a candidate only; it does not establish HMC validity. |
| Hard vetoes | Non-finite loss/gradient/state, failed forward/inverse/logdet mechanics, stale or overwritten artifact, missing tuning manifest, or GPU/XLA/memory-growth policy failure. |
| Continuation veto | None from a failed candidate alone. Stop only for invalid target, broken harness, corrupted replay, or unavailable required runtime. |
| Repair trigger | If all tuned architectures fail while mechanics and exact-map fitting pass, run the smallest root-preserving staged warm-start experiment; do not enlarge the network again without evidence. |
| Explanatory diagnostics | Reverse-KL loss, gradient norm, clipping frequency, per-stage scale saturation, root variance/kurtosis, and importance-weight ESS. They cannot promote a candidate. |
| Must not conclude | No claim about SSL-LSTM posterior correctness, universal NeuTra reliability, HMC convergence, or a new default follows from this campaign. |

## Architecture arms

| Arm | Stages | Hidden layers | Between-stage permutation | Scale path | Status |
|---|---:|---|---|---|---|
| `one_stage_exact` | 1 | `(100,100)` | none | unbounded linear conditional scale | exact-compatible diagnostic |
| `three_root_preserving` | 3 | `(100,100)` | keep root coordinate 0, reverse only coordinates 1:100 | unbounded linear scale in stage 0, bounded residual stages | primary repair hypothesis |
| `three_full_reverse` | 3 | `(100,100)` | reverse all coordinates | same as root-preserving arm | matched baseline |
| `three_root_wide` | 3 | `(200,200)` | root-preserving | same as root-preserving arm | capacity hypothesis |

The exact funnel is

```text
y = z[0],  x[i] = exp(y) z[i],  i=1,...,99.
```

The one-stage arm tests whether the implemented architecture can learn a known
map. The root-preserving arms prevent later stages from making the root scale
or shift depend on the child cloud. The full-reversal arm preserves the generic
paper-style comparator. Width is changed only in the final arm, so a width gain
cannot be confused with the ordering repair in the first two three-stage arms.

## Tuning protocol

Each arm receives its own calibration run and schedule. No learning rate is
transferred as a promoted default. The calibration grid is:

| Parameter | Values | Provenance |
|---|---|---|
| Peak learning rate | `2e-4`, `5e-4`, `1e-3`, `2e-3` | target-specific hypothesis; bounded by prior stable canaries |
| Schedule | constant; piecewise with multipliers `1`, `0.1`, `0.01` over `[0,60%)`, `[60%,85%)`, `[85%,100%]` of the update budget | standard optimization hypotheses, not paper defaults |
| Calibration updates | 1,000 | budgeted nomination screen, not convergence |
| Training batch | 4,096 | inherited from NeuTra paper and prior BayesFilter runs; held fixed |
| Seeds | fixed initialization and calibration seed bank; two confirmation seeds only after nomination | matched comparison first, replication second |

The cross-product contains eight tuning candidates per architecture, 32 total.
Calibration selects the lowest held-out reverse-KL on a disjoint 65,536-row
latent validation cloud among finite candidates. Proposal-law diagnostics remain
explanatory during calibration. The selected schedule is frozen before the
5,000-update confirmation arm, which is checkpointed by held-out reverse-KL and
audited only on a fresh proposal cloud. If all eight schedules for an arm are
non-finite, that arm is implementation-vetoed; if they are finite but fail the
proposal audit, the arm is rejected without ranking it against other failed
arms.

## Evidence contract and pre-mortem

The exact target and replay manifest are fixed. Calibration data, selection
data, and proposal-audit data use disjoint stateless seeds. Every run records
the git commit, command, environment, device, memory-growth verification, XLA
status, architecture, schedule, seeds, wall time, and artifact hashes.

Pre-mortem:

1. A lower validation loss may still suppress tails. Countermeasure: proposal
   law, not loss, is the primary nomination screen.
2. Wider networks may win because of a different optimization scale rather
   than capacity. Countermeasure: independent schedule tuning and fixed batch,
   seed, and update budget.
3. A permutation implementation may pass forward tests but break inverse or
   score transport. Countermeasure: round-trip, logdet, and autodiff parity
   tests for both permutation policies before training.
4. A concurrent GPU process may distort timing or memory. Countermeasure:
   record trusted device provenance and treat wall time as descriptive only.
5. A failed arm may reflect insufficient updates rather than architecture.
   Countermeasure: retain the confirmation checkpoint curve and trigger the
   staged warm-start repair rather than claiming capacity failure.

## Execution and stop conditions

1. Add explicit permutation policy and dynamic stage/width controls while
   preserving full-reversal defaults.
2. Run mechanics tests and explicitly construct the exact funnel map inside
   every architecture. This proves representational inclusion without adding a
   separate regression objective.
3. Run the four-arm, eight-schedule 1,000-update calibration grid.
4. Confirm at most one nominated schedule per arm for 5,000 updates.
5. Run the untouched proposal audit and write a decision table and inference-
   status table. No HMC is launched by this plan.

## Artifact

Results are written below
`docs/plans/artifacts/neutra-reverse-funnel-architecture-tuning-2026-08-15/`.
The plan, calibration manifest, confirmation manifests, result notes, and
reset memo are part of the evidence package.

## Post-calibration repair amendment

The initial campaign completed all 32 calibration cells and eight confirmations
without an engineering veto, but every confirmation failed the exact proposal
law. Because all four architectures explicitly contain the exact map, this is a
repair trigger, not a capacity verdict or a continuation veto.

The next smallest discriminator is a staged one-stage experiment:

1. freeze the MADE weights, shifts, and root affine map at identity;
2. train only the first-stage unbounded autoregressive scale matrix from zero;
3. verify that reverse KL learns the exact first-row coefficients and passes the
   untouched proposal law;
4. use that state to initialize joint one-stage training, tuning the joint peak
   learning rate over `2e-4`, `5e-4`, and `1e-3` with the same piecewise
   schedule; and
5. reject the joint route if it moves an exact-law warm start outside the
   untouched exact-law intervals.

The restricted fit is a mathematical diagnostic, not a deployable procedure
for unknown targets. Its purpose is to distinguish failure of the reverse-KL
objective or gradients from failure caused by joint parameterization and
co-adaptation. The exact proposal audit remains the primary criterion; training
loss, coefficient error, and gradient norms remain explanatory.
