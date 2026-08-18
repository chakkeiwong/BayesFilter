# NeuTra reverse-funnel capacity repair plan (2026-08-14)

Status: `COMPLETE_CANDIDATE_REJECTED_NO_HMC`

## Research intent ledger

| Item | Contract |
|---|---|
| Main question | Is the reverse-KL funnel failure caused materially by the transport's shared `s_max=1` conditional log-scale ceiling? |
| Mechanism under test | The exact funnel map needs `log scale(x_i|y)=y`, while the current stage scale is bounded to `[-1,1]`. Give the first autoregressive stage enough scale capacity while keeping later stages tightly bounded. |
| Matched baseline | Fresh reverse-KL training with stage caps `(1,1,1)`, Gaussian base, `(100,100)` ELU network, three stages, batch 4096, constant LR `1e-3`, 5000 updates, and the same initialization/latent seeds as both repair arms. |
| Repair arms | `(3,0.5,0.5)` and `(4,0.5,0.5)`. The smaller passing arm is preferred. |
| Promotion criterion | An untouched iid proposal audit passes separate 99.9% intervals for exact funnel `E[y]=0`, `E[y^2]=1`, both `P(y<-2)=P(y>2)=Phi(-2)`, standardized residual mean `0`, and standardized residual second moment `1`. |
| Promotion vetoes | Nonfinite update/value/gradient, artifact/hash mismatch, XLA or memory-growth failure, cap saturation inconsistent with the declared arm, or any proposal-law interval failure. |
| Continuation vetoes | Broken target/transport math, corrupted state, invalid proposal diagnostic, or both cap arms becoming nonfinite under the bounded training budget. Candidate failure alone is not a continuation veto. |
| Repair trigger | If neither cap arm passes proposal law but remains numerically finite, add a zero-initialized masked linear skip to the scale head and repeat only the smallest discriminating arm. |
| Downstream validation | HMC only for the smallest cap arm passing the proposal gate. Retune fixed-length HMC, `L>=2`, and retain at least 4000 draws per chain before exact funnel-law adjudication. |
| Must not be concluded | A pass on the exact funnel does not establish SSL-LSTM posterior correctness, universal reverse-KL reliability, Student-t superiority, or a new default. |

## Mathematical target

For `z=(z_0,z_1,...,z_99) ~ N(0,I)`, the exact funnel transport can be
written

```text
y = z_0
x_i = exp(z_0) z_i,  i=1,...,99.
```

Thus one autoregressive stage can represent the essential geometry when the
first coordinate is `y`: output coordinate `i>0` may condition on `z_0`, and
its correct conditional log-scale is `z_0`. A hard cap of one truncates that
relation for `|z_0|>1`. The repair increases the first-stage cap but keeps later
residual stages at `0.5`; cumulative declared maxima are four and five, not
twelve as in an unstable `(4,4,4)` design.

## Evidence contract

- Exact comparator: `y~N(0,1)` and `r_i=x_i exp(-y)~N(0,1)`.
- Proposal audit count: `131072` iid latent draws. This is a convenience count
  giving roughly 2982 expected observations in each `|y|>2` tail under the
  exact law; it is not inherited as a universal default.
- Interval level: `99.9%`, retained from the reviewed d100 repair calibration.
  Means and second moments use iid influence-function standard errors; tail
  probabilities use Wilson score intervals.
- HMC comparator: the existing reverse-funnel baseline and exact funnel law.
  HMC is not used to select between proposal-failing arms.
- Result artifact: fresh roots under
  `docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/paper-d100/reverse-funnel-capacity-r1/`.

## Default and assumption audit

| Choice | Provenance/status | Justification | Failure mode | Early diagnostic |
|---|---|---|---|---|
| Gaussian base | Exact funnel construction; reviewed baseline | The exact funnel is a deterministic transform of iid Gaussian variables | Blaming the base hides a transport defect | Exact-map derivation and proposal audit |
| Stage caps `(1,1,1)`, `(3,.5,.5)`, `(4,.5,.5)` | Baseline plus derived capacity hypotheses | Isolates the first-stage ceiling while bounding residual stages | Cap still insufficient or excessive | Per-stage scale saturation and finite-value summaries |
| Width `(100,100)`, three stages, ELU | Existing d100 baseline; held fixed | Avoids confounding scale capacity with width/depth | Other capacity may still matter | Proposal audit; skip repair only if cap ladder fails |
| Constant LR `1e-3` | Conservative arm from prior canaries; matched convenience choice | Avoids the historical `1e-2` clipping regime and keeps arms comparable | Too slow at 5000 updates | 250-update canary, late heldout slope, gradient/clipping trace |
| 5000 updates | Historical baseline budget; hypothesis, not convergence fact | Fast reverse updates make a matched full baseline inexpensive | Loss still improving at boundary | checkpoint curve and proposal audit |
| One shared seed bank | Matched capacity test | Removes seed variation from the first discriminator | Result may be seed-specific | no robustness claim; replicate only after one arm passes |
| HMC 4000 retained/chain | Derived from tail-event precision | About 364 expected observations per tail across 16000 retained draws | Serial dependence leaves tail MCSE large | chain-aware exact-law intervals and ESS |

## Pre-mortem

- Misleading pass: proposal central moments pass while conditional residuals are
  wrong. Countermeasure: include residual mean/second moment and importance
  ratio diagnostics.
- Implementation failure mistaken for scientific failure: stage caps are loaded
  in the wrong order. Countermeasure: configuration roundtrip tests and direct
  stage-bound tests.
- Tuning failure mistaken for capacity failure: all arms use an unstable LR.
  Countermeasure: 250-update finite canary for each arm before the 5000-update
  run; constant `1e-3` and clipping telemetry.
- HMC false rejection from too few tail events: require 4000 retained draws per
  chain and report separate 99.9% diagnostics without an omnibus p-value.

## Skeptical plan audit

| Audit question | Finding |
|---|---|
| Wrong baseline? | Repaired. The historical `(1,1,1)` result used a different LR schedule and seed; a fresh matched baseline is included. |
| Proxy promoted to criterion? | No. Training loss and saturation are explanatory. Exact proposal-law intervals select the HMC candidate; exact HMC-law intervals decide downstream validity. |
| Missing stop conditions? | No. Nonfinite mechanics and invalid artifacts stop; a finite candidate failure triggers the next declared repair. |
| Unfair comparison? | Arms share architecture, optimizer, update count, batch, initialization, and latent batches. Only stage caps differ. |
| Hidden environment mismatch? | Serious training/HMC use GPU 1, float64, TF32 off, XLA, and verified memory growth. |
| Could commands succeed without answering the question? | The proposal audit directly measures whether reverse-KL learned the exact funnel marginal and conditional law before HMC. |

Audit verdict: `PASS_FOR_EXECUTION`.

## Cap-ladder result and skip-repair audit

All three matched 5,000-update arms remained finite, but none passed the exact
proposal-law gate. Increasing the first-stage cap from one to four improved the
proposal `y` second moment from `0.7496` to `0.8658` and the two tail estimates
from `0.00193/0.00491` to `0.00808/0.01252`; the exact tail probability is
`0.02275`. The cap-four arm had negligible cap saturation, so another cap-only
increase is not supported. This is candidate failure, not a continuation veto,
and it fires the predeclared masked-linear skip repair.

The skip enters the first-stage pre-`tanh` scale logit, remains bounded by the
stage cap, uses the strict mask `M[j,i]=1[j<i]`, and starts at zero so the initial
transport is unchanged. Its manual pullback must add
`scale_logit_cotangent @ (W_skip * M)^T` for both mapped-score and logdet-score
paths. A focused autodiff parity test is required before the GPU canary.

Skip-repair skeptical audit: `PASS_FOR_EXECUTION`. The cap-four no-skip arm is
the exact matched comparator; proposal-law diagnostics remain the promotion
criterion; training loss remains explanatory; nonfinite mechanics, score
parity failure, invalid state, or GPU/XLA/memory-policy failure remain hard
vetoes. A finite proposal-gate failure does not justify HMC.

### Post-run correction

The first implemented skip entered the pre-`tanh` scale logit. Its 5,000-update
arm was finite and descriptively improved the cap-four comparator, but failed
the exact proposal gate: `E_q[y^2]=0.8746`, lower/upper tail probabilities were
`0.0100/0.0135`, and the exact value is `0.02275`. More importantly, the
post-run mathematical audit found that this path remained bounded by the hard
scale cap. It therefore did not answer the capacity question and is classified
as a rejected pre-cap diagnostic, not evidence against a linear scale path.

The corrected repair is

```text
scale_log(z) = z (W_linear * M) + s_max tanh(scale_mlp(z) / s_max),
M[j,i] = 1[j<i].
```

This permits the exact funnel coefficient `W_linear[0,i]=1` for every `i>0`
while the nonlinear residual remains bounded. The linear matrix is initialized
to zero, so the initial transport is unchanged. Stability relies on the
existing gradient clipping, finite-update rejection, 250-update canary, and
fresh artifact boundary rather than an arbitrary coefficient cap that would
reintroduce the representational defect.

Corrected-repair skeptical audit: `PASS_FOR_EXECUTION`. An analytic test must
first set the derived exact coefficient and verify `y=z_0`,
`x_i=exp(z_0)z_i`, and the exact log determinant. Manual mapped-score and
logdet-score paths must independently match autodiff. The matched cap-four
pre-cap/no-skip runs remain descriptive comparators; exact proposal law remains
the only HMC nomination criterion.

## Terminal decision

The corrected 5,000-update arm was finite and selected update 5,000, but failed
the exact proposal-law gate. It produced `E_q[y^2]=0.8939`, lower/upper tail
probabilities `0.01305/0.01546` versus exact `0.02275`, and a nonzero mean
`0.02259`; the standardized conditional residual mean and second moment passed.
No HMC was run.

The capacity hypothesis is only partly supported. The original cap is wrong
relative to exact-funnel representability, and the additive route is proven in
focused tests to represent the exact map and its log determinant. The remaining
failure is therefore not evidence of insufficient representational capacity in
that corrected route. Under the tested constant-LR joint-training protocol,
reverse-KL optimization did not learn the exact route. The next discriminating
experiment is a separately reviewed optimizer decomposition: train the linear
scale path alone first, then compare frozen-linear residual training with joint
fine-tuning. More generic width/depth or cap increases are not justified by this
campaign.

Result note:
`docs/plans/bayesfilter-neutra-reverse-funnel-capacity-repair-result-2026-08-14.md`.

## Execution stages

1. Add backward-compatible stage-specific caps to `WeightedNeuTraConfig` and
   test configuration, forward/inverse, logdet, and explicit score parity.
2. Add a capacity runner with 250-update canary and 5000-update matched arms.
3. Generate independent proposal audits and nominate the smallest passing arm.
4. If an arm passes, run retuned fixed-length HMC with at least 4000 retained
   draws per chain and exact funnel diagnostics. If neither passes but mechanics
   remain valid, implement and test the declared masked-linear scale skip.
5. Preserve commands, device provenance, seeds, wall times, results, and hashes
   in the result note and artifact roots.
