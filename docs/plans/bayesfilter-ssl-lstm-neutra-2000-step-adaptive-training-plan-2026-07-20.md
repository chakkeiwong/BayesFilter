# SSL-LSTM NeuTra 2,000-Step Adaptive Training Amendment

Date: 2026-07-20  
Tier: 2 material research engineering  
Status: `AUDITED_FOR_IMPLEMENTATION`

## Research Intent Ledger

| Role | Contract |
| --- | --- |
| Main question | Can the q-general SSL-LSTM NeuTra final-training protocol reduce its maximum from 5,000 to 2,000 optimizer steps while using heldout evidence to repair and stop plateaus at 250-step boundaries? |
| Exact baseline | The current q-general runner: batch size 480, validation batch 64, paired heldout-loss upper-bound improvement test, best trainer/Adam checkpoint restoration, 50% LR repair, six Optuna trials with rungs 50/100/200/400, two final streams, and at most one triggered confirmation stream. |
| Candidate mechanism | Set the final/confirmation cap to 2,000, validate every 250 steps, repair after the first 250-step cycle without meaningful improvement, then allow two additional 250-step cycles before stopping. |
| Promotion criterion | Focused deterministic tests prove the exact `best -> +250 repair -> +500 continue -> +750 stop` sequence, improvement reset, resume equivalence, configuration binding, and runner contract. |
| Promotion veto | Wrong boundary, failure to restore/set LR in the runner, resume mismatch, stale 5,000-step runner cap, changed Optuna/search/batch contract, import/compile failure, or focused-test failure. |
| Continuation veto | A controller invariant cannot represent the requested sequence without corrupting existing checkpoint semantics, or concurrent edits make the scoped files internally inconsistent. |
| Repair trigger | A focused failure triggers a same-file controller or runner repair and rerun. |
| Explanatory only | Historical 5,000-step projections and prior q=1 training outcomes; they do not validate the amended q-general schedule. |
| Nonclaims | No NeuTra-quality, posterior-correctness, HMC-convergence, optimal-patience, optimal-cap, runtime, or scientific-validity claim. No material GPU training or HMC is authorized here. |

## Exact Schedule

Let `b` be the most recent support-eligible best validation step. Meaningful
improvement retains the existing paired one-sided upper-confidence-bound rule;
it is not exact scalar equality or an arbitrary loss tolerance.

1. At `b + 250`, if there is no meaningful improvement, restore the best
   transport and Adam state and set `LR <- 0.5 * LR`.
2. At `b + 500`, if there is still no meaningful improvement, continue for one
   more validation cycle.
3. At `b + 750`, if there is still no meaningful improvement, stop with
   `plateau_after_lr_repair`.
4. Any meaningful improvement becomes the new best and resets this sequence.
5. Stop at 2,000 steps even if the current plateau sequence is incomplete.

## Default And Assumption Audit

| Choice | Provenance/status | Justification | Failure mode and early diagnostic |
| --- | --- | --- | --- |
| 2,000-step cap | Owner-requested hypothesis | Reduces the prior 5,000-step maximum while preserving eight validation opportunities. | May truncate a still-improving run; terminal reason and best step remain recorded, and later material evidence must assess this. |
| 250-step validation cycle | Owner-requested reviewed default for this runner | Gives four checks per 1,000 steps and matches the requested repair cadence. | Checks may be too sparse or noisy; deterministic boundary tests verify mechanics, while material heldout histories must assess adequacy. |
| Two post-repair cycles | Owner-requested reviewed default for this runner | Implements the literal request: two further checks after the repair before stopping. | Off-by-one stopping at +500; exact transition and resume tests veto it. |
| 50% LR factor | Existing reviewed training policy | Already implemented and explicitly requested. | Trainer/controller LR can diverge; source-contract test and checkpoint path inspection cover restoration and setting. |
| Paired heldout UCB | Existing statistical rule | Distinguishes statistically supported loss reduction from scalar noise on the fixed heldout batch. | A small validation batch can lack power; batch size remains 64 and no claim of optimality is made. |
| Batch 480 and validation batch 64 | Inherited baseline, unchanged | The request does not authorize retuning batch sizes. | Could affect optimization/noise; constants are asserted unchanged and remain future tuning hypotheses. |
| Optuna rungs/trials | Inherited nomination protocol, unchanged | The request concerns final adaptive training, not candidate search. | Short-rung nomination may not predict 2,000-step behavior; it remains nomination-only and is not promoted by this change. |

## Skeptical Pre-Execution Audit

- Wrong baseline: passed. The change targets the q-general final/confirmation
  runner, not the earlier completed scalar G/H experiment.
- Proxy promotion: passed. Heldout loss controls training checkpoints only; it
  does not establish transport or posterior correctness.
- Missing stop: repaired prospectively. The old controller stopped after only
  one post-repair cycle when patience equaled the validation interval. The new
  configuration makes the number of post-repair cycles explicit.
- Hidden assumption: recorded. The 2,000/250/two-cycle numbers are
  owner-requested hypotheses, not statistically optimized constants.
- Environment mismatch: focused checks deliberately hide GPUs; no performance
  or material-training inference will be drawn from them.
- Artifact adequacy: focused tests directly answer controller timing, resume,
  and runner-configuration questions. A material training result would require
  a separate authorized GPU run and result record.
- Unfair comparison: not applicable. No stochastic candidate ranking or method
  comparison is performed.

Audit decision: `PASS_FOR_FOCUSED_IMPLEMENTATION_AND_CPU_HIDDEN_TESTS`.

## Implementation And Checks

- Add an explicit post-repair no-improvement cycle count to
  `NeuTraPlateauConfig`, defaulting to one to preserve current callers.
- Bind the field through the existing config manifest and checkpoint hash.
- Set the q-general runner to 2,000 maximum steps, 250-step validation and
  patience, a 0.5 LR factor, and two post-repair cycles.
- Preserve best trainer/Adam restoration before applying the lower LR.
- Add focused controller and runner tests, compile touched Python files, run a
  q=20 contract smoke with GPUs hidden, and run `git diff --check` on lane files.

Result artifact:
`docs/plans/bayesfilter-ssl-lstm-neutra-2000-step-adaptive-training-result-2026-07-20.md`.

