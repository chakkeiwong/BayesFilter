# NeuTra HMC Robustness Phase F1 Subplan

Date: 2026-07-15  
Status: `EXECUTED_AND_CLOSED`

## Objective And Entry Conditions

Run a target-specific three-recipe screen on the admitted new fixture and train
one fresh 5,000-step GPU/XLA dense-IAF candidate. F0's exact target, geometry,
mass, and plain-HMC comparator passed.

## Training Geometry And Recipe Audit

The fixed affine factor is the new fixture's reviewed mass factor. The training
center is not the truth-centered comparator center: it is
`mass_center + 0.25 * prior_scale * alternating_sign`, frozen before training.
This is an initialization-stress hypothesis, not a reviewed default. It may
cause failure for initialization reasons; the three-arm screen is the early
diagnostic.

Screen arms, each 500 steps on common training and held-out seeds:

- `inherited_wide_lr5e3`: 3 stages, hidden `(36,36)`, LR `5e-3`;
- `source_width_lr5e3`: 3 stages, hidden `(18,18)`, LR `5e-3`;
- `wide_lower_lr1e3`: 3 stages, hidden `(36,36)`, LR `1e-3`.

All use batch 128, ELU, `s_max=1`, init scale `0.02`, clip norm 10, and constant
LR to match the existing recipe contract. These are hypotheses, not target-
independent defaults. A valid candidate is nominated by lowest common-heldout
reverse-KL mean; differences are descriptive only and do not establish a
statistical ranking. If candidates are within two paired MCSE, choose the lower
parameter-count arm, then lower LR, then declared order.

## Evidence Contract

The question is whether at least one target-specific recipe yields an
engineering-valid frozen candidate for F2. Every job must be batched,
TensorFlow-only, one compiled GPU/XLA `tf_while_loop`, memory-growth enabled,
freshly initialized, finite/status-valid, and exactly reload/score-parity
checked. Screen loss/heldout objective nominates only. Promotion of NeuTra on
the new fixture requires F2 downstream HMC, not F1.

Required artifacts: geometry ledger, three screen results with common held-out
rows, selection ledger with uncertainty/nonclaims, one fresh 5,000-step result,
frozen payload and hashes, GPU manifest, parity/closure checks, F1 close record,
and F2 subplan.

## Budget, Stop, And Handoff

GPU budget is three 500-step screens plus one 5,000-step selected job, at most
60 minutes aggregate. No screen weights may be reused. One fresh-directory
retry is allowed only for localized infrastructure failure. A candidate
numeric/status failure rejects that arm; continue the screen. Stop F1 if all
arms fail, the exact target/batch binding is invalid, trusted GPU is unavailable,
or budget is exhausted. At least one engineering-valid frozen 5,000-step
candidate hands off to F2.

Skeptical audit verdict: `PASS`. The target and factor are new-fixture bound,
the inherited recipe is only one arm, proxy metrics cannot promote, the offset
center risk is explicit, and F2 remains the downstream scientific gate.
