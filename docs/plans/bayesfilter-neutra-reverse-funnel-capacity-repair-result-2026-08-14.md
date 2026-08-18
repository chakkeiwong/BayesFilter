# NeuTra reverse-funnel capacity repair result (2026-08-14)

Status: `COMPLETE_CANDIDATE_REJECTED_NO_HMC`

## Outcome

The original three-stage dense IAF was wrong relative to exact-funnel
representability when every conditional log scale was hard-bounded by
`s_max=1`. Raising the first-stage cap improved the proposal but did not pass
the exact proposal-law gate. A first attempted linear skip was also bounded
because it entered before `tanh`; that arm is retained as a rejected diagnostic.

The corrected route adds a zero-initialized strictly autoregressive linear term
after the bounded nonlinear residual:

```text
scale_log(z) = z (W_linear * M) + s_max tanh(scale_mlp(z) / s_max),
M[j,i] = 1[j<i].
```

This route can exactly represent the target construction with
`W_linear[0,i]=1` for `i>0`. Focused tests verified that construction, inverse
and log-determinant roundtrips, frozen-state reconstruction, and manual
mapped-score/logdet-score parity with autodiff.

The corrected matched 5,000-update reverse-KL arm remained finite but failed
the independent 131,072-draw proposal-law audit. No HMC was run.

## Claimed target and computed quantity

The target is the normalized 100-dimensional funnel

```text
y ~ Normal(0,1),
x_i | y ~ Normal(0, exp(2y)), i=1,...,99.
```

The runtime target omits only the normalizing constant. For any invertible
transport from a standard Gaussian base, the expected training objective obeys

```text
E[-log p_unnormalized(T(z)) - log |det J_T(z)|] - 50
    = KL(q_T || p).
```

Thus the exact population optimum is `50`. The selected checkpoint used a fixed
65,536-row selection cloud; the independent 65,536-row audit estimated `50.0682`
for the corrected arm. This aggregate reverse-KL estimate is explanatory, not a
proposal-equivalence criterion. Reverse KL can assign little cost to missing
tail mass, and the 99 conditional coordinates can dominate the scalar `y`
diagnostic. The exact proposal-law intervals remain the decision authority.

## Matched results

| Arm | Selected update | Audit objective | `E_q[y^2]` | Tail `<-2` | Tail `>2` | Importance ESS fraction | Proposal gate |
|---|---:|---:|---:|---:|---:|---:|---|
| Cap `(1,1,1)` | 4750 | 50.10570 | 0.74962 | 0.00193 | 0.00491 | 0.18878 | Fail |
| Cap `(3,.5,.5)` | 4500 | 50.07204 | 0.83701 | 0.00587 | 0.00977 | 0.00764 | Fail |
| Cap `(4,.5,.5)` | 4500 | 50.06073 | 0.86578 | 0.00808 | 0.01252 | 0.01456 | Fail |
| Pre-cap linear diagnostic | 5000 | 50.07775 | 0.87457 | 0.01002 | 0.01346 | 0.67415 | Fail |
| Additive unbounded linear | 5000 | 50.06824 | 0.89391 | 0.01305 | 0.01546 | 0.79534 | Fail |
| Exact law | N/A | 50 population | 1.00000 | 0.02275 | 0.02275 | 1 population | Pass target |

These are matched-seed descriptive results, not a statistically supported
ranking. Importance ESS is also descriptive: the cap-one arm's apparently high
ESS did not repair its severe tail failure, illustrating why it is not the
promotion criterion.

## Corrected-arm proposal audit

| Diagnostic | Estimate | 99.9% interval | Exact | Role | Status |
|---|---:|---:|---:|---|---|
| `E[y]` | 0.022590 | [0.013999, 0.031181] | 0 | Promotion criterion | Fail |
| `E[y^2]` | 0.893914 | [0.883348, 0.904480] | 1 | Promotion criterion | Fail |
| `P(y<-2)` | 0.013054 | [0.012062, 0.014126] | 0.022750 | Promotion criterion | Fail |
| `P(y>2)` | 0.015457 | [0.014375, 0.016619] | 0.022750 | Promotion criterion | Fail |
| Residual mean | -0.000100 | [-0.001012, 0.000812] | 0 | Promotion criterion | Pass |
| Residual second moment | 1.000113 | [0.998821, 1.001405] | 1 | Promotion criterion | Pass |

The learned first-row additive coefficients had mean `0.2450`, range
`[0.2039, 0.5019]`; coefficients equal to one give the exact map only when the
later stages remain identity. They grew from mean `0.0848` at update 250, while
the selected loss was still improving at update 5,000. The August 15 root-cause
trace showed that the three stages had co-adapted: forcing this block alone to
one was uphill, while a coordinated root/conditional correction was downhill.
See
`docs/plans/bayesfilter-neutra-reverse-funnel-root-cause-diagnosis-2026-08-15.md`.

## Decision table

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Reject all tested candidates for HMC nomination | Exact proposal law failed | No nonfinite, XLA, memory-growth, state, or hash veto fired | One matched seed/protocol; no optimizer decomposition | Test linear-only warm-up, frozen-linear residual training, then joint fine-tuning under a new plan | NeuTra direction rejected; reverse KL universally invalid; SSL-LSTM result decided |

## Inference status

| Item | Status |
|---|---|
| Hard veto screen | No mechanics/invalidity hard veto fired; the exact proposal-law promotion veto fired for every arm |
| Statistically supported ranking | None; no multi-seed uncertainty analysis was run |
| Descriptive-only differences | Wider caps and both linear paths improved selected diagnostics, but no arm passed |
| Default readiness | No candidate/default/HMC readiness |
| Next evidence needed | Target-specific optimizer decomposition with the exact-capable additive route and multi-seed confirmation only after one protocol passes proposal law |

## Failure classification

- Implementation failure: not supported for the corrected mechanics; exact-map
  construction and manual/autodiff score tests passed.
- Target failure: not supported; the TensorFlow target and exact sampler match
  the checked funnel equations.
- Capacity failure: supported for the historical hard-capped route, not
  supported for the corrected additive route.
- Tuning/optimization failure: supported for the tested joint constant-LR
  protocol; the full-reversal middle stage distorted the root marginal and the
  three stages co-adapted before the fixed budget ended.
- Evidence against reverse-KL NeuTra generally: unsupported. Only this target,
  architecture family, optimizer protocol, and seed bank were tested.

## Post-run red team

Strongest alternative explanation: 5,000 updates at constant `1e-3` may be
under-budgeted for the additive coefficients, since their mean continued to
grow and the fixed selection objective improved at the terminal checkpoint.
However, simply extending the same joint run is not the smallest discriminating
test: nonlinear paths can absorb conditional fit while slowing the exact linear
coordinate. Linear-only warm-up directly tests that mechanism.

A result that would overturn the current candidate rejection is a fresh
corrected-route training protocol that passes every untouched proposal-law
interval. A result that would weaken the optimizer-interference hypothesis is a
linear-only arm that stays finite but fails to learn coefficients near one or
fails proposal law.

The weakest evidence is comparative ranking: all continuous arm differences
are descriptive because this campaign used a shared single seed bank. The hard
candidate rejection does not depend on that ranking; every arm separately
failed its exact-law intervals.

## Run manifest

| Field | Value |
|---|---|
| Git commit | `3030d86df9cb00346df82c7c19f015c09c7c6e1f` |
| Dirty-source core SHA-256 | `850a7b4d765643c683e92c1f783184c10cb98593af314482a59cc91631325669` |
| Dirty-source runner SHA-256 | `d5f3f6482bbc20d99e72f692f3e45b28556714eac6c47fa889c6ac96704246cc` |
| Environment | `/home/ubuntu/anaconda3/envs/tfgpu`, TensorFlow `2.20.0` |
| Hardware | GPU 1, NVIDIA GeForce RTX 4080 SUPER; GPU 0 not used |
| GPU policy | Memory growth verified before logical initialization; peak TensorFlow allocator bytes `1,980,168,960` |
| Numeric mode | TensorFlow float64, TF32 off, XLA JIT on |
| Training | Batch 4096, 5,000 updates, Adam LR `1e-3`, clipping norm 10 |
| Seeds | Initialization `20260814,70001`; selection `70002`; training root `70003`; audit `70004`; proposal `71001` |
| Data | Exact funnel replay `paper-d100/funnel-replay-r1`; untouched replay hashes checked |
| Corrected full wall time | 66.198 s |
| Whole capacity campaign GPU wall time | 394.491 s across ten archived canary/full arms |
| Plan | `docs/plans/bayesfilter-neutra-reverse-funnel-capacity-repair-plan-2026-08-14.md` |
| Result | This file |
| Corrected artifacts | `docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/paper-d100/reverse-funnel-capacity-r3/` |

Corrected full command:

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_neutra_reverse_funnel_capacity_2026_08_14.py \
  --output-root docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/paper-d100/reverse-funnel-capacity-r3/full-cap4-unbounded-linear \
  --replay-root docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/paper-d100/funnel-replay-r1 \
  --device 1 --stage-s-max 4,0.5,0.5 \
  --first-stage-unbounded-scale-linear \
  --updates 5000 --batch-size 4096 --checkpoint-every 250 \
  --learning-rate 1e-3 --proposal-audit-count 131072
```

Artifact hashes were verified after the run. The corrected full hash ledger is
`reverse-funnel-capacity-r3/full-cap4-unbounded-linear/artifact_hashes.json`
with SHA-256
`4f6a6806d7962098325ad572ad57251a71350c07c04baa62a269b133593709d2`.

## Verification

Broadened CPU/XLA verification across transport mechanics, the exact funnel
target authority, runner contract, and HMC loader compatibility: `43 passed`.
Python compilation and `git diff --check` passed. The test suite intentionally
hid GPUs.
