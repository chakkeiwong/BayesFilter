# Corrected Parameter-Authority Phase 42 Subplan

Parent: `bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md`  
Version: `v2.4-independent-bank-replication`  
Entry gate: Phase 41 independent-bank report passed with a finite-holdout repair trigger  
Status: `PASS_V2_4_TWO_BANK_REPLICATION_REPORT_REPAIR_TRIGGERED`
Local cap: 3000 s

## Question

Does the Phase 41 residual reduction reproduce across two new independent
theta banks when both are evaluated by one frozen trained transport state?

## Frozen boundary

Keep the q=20 batch-native target in `theta in R^4`, the v2.2 N=256 M0
root-group training split, normalized train weights, proposal protocol,
identity/affine preconditions, architecture arms, 200-step optimizer budget,
and target/protocol signatures unchanged. The 60D UKF state remains internal.

Generate two new N=256 C0/M0 pilots with seeds `(20260826, 4102)` and
`(20260826, 4103)`. These are reviewed campaign hypotheses, not defaults.
Neither bank may enter training, checkpoint selection, or tuning. A single
trainer instance per arm must consume only the frozen old training rows, then
evaluate both fresh banks after the final update. This isolates bank-to-bank
variation from repeated retraining variation.

## Evidence contract

**Primary hard gates.**

1. Both fresh pilots pass `theta_R4` status/finite gates and match the exact
   target signature, M0 protocol hash, and C0 protocol hash.
2. Both fresh roots and all tensor hashes differ from the old authority and
   from each other.
3. The old root-group split is complete and root-disjoint; the affine oracle
   is exact on the old training measure.
4. Each arm trains once on old rows only, uses batch size greater than one,
   verifies GPU memory growth before TensorFlow initialization and XLA, and
   evaluates both fresh banks only after the final optimizer update.
5. The result records bank-specific residuals, support summaries, hashes,
   fresh-use flags, and a predeclared replication interpretation.

**Promotion vetoes.** Any fresh-bank use during optimization/selection,
signature/hash mismatch, non-finite target/status, shape or measure mismatch,
GPU policy failure, or missing bank-specific receipt. Poor whitening or
bank-to-bank variation is a repair trigger, not a continuation veto.

## Interpretation branches

| Result | Role | Next action |
|---|---|---|
| Both banks show lower residuals than old tiny validation, with modest bank variation | explanatory repair evidence | design a larger/representative support bank before objective changes |
| One improves and one does not, or variation is large | explanatory uncertainty | add a third bank or enlarge particles before changing objective |
| Both resemble old holdouts and residuals persist | explanatory repair evidence | investigate objective/capacity/weighting mismatch |
| Any hard gate fails | veto | repair pilot/runner and rerun in a new root |

These branches are not candidate rankings. With two banks and one frozen state,
continuous differences remain descriptive; no superiority or statistical
generalization claim is allowed.

## Skeptical pre-mortem and default audit

| Risk | Early diagnostic | Handling |
|---|---|---|
| Two fresh seeds share the same proposal bias | compare support ranges, mode fractions, and root counts | label replication as limited; do not claim coverage |
| Fresh rows leak into training | separate tensor hashes and explicit false-use flags | fail closed |
| A single trainer is accidentally recreated per bank | record one state lineage and arm seed per result | fail closed; do not interpret |
| GPU/XLA compilation changes state or dtype | memory-growth/device/XLA manifest and finite gates | preserve failed root and repair |
| Tiny old validation remains an unfair comparator | retain it only descriptively; compare both fresh banks to frozen train measure too | no checkpoint selection |

Inherited N=256, 200-step, epsilon, scale, and architecture values are fixed
comparison hypotheses, not promoted defaults. The fresh seeds are measured
campaign choices.

## Pre-execution skeptical audit

The plan passes the required audit on 2026-08-26. The target, comparator,
promotion/veto roles, stop conditions, stale-version risks, environment
requirements, and artifact/question alignment were checked. The phase changes
only the evidence boundary (two independent post-training audits); it does not
change the scientific target, objective, or default policy. A passing boundary
will not be interpreted as whitening or posterior correctness.

## Pilot preflight receipt

Both fresh pilots passed the hard theta/status gates before the shared-state
audit. Their measured M0 diagnostics are deliberately retained:

| Bank | seed | terminal ESS fraction | weighted negative-mode fraction | negative/positive terminal roots |
|---|---|---:|---:|---:|
| A | `(20260826, 4102)` | `0.801812` | `0.756588` | `59 / 44` |
| B | `(20260826, 4103)` | `0.946687` | `0.517590` | `72 / 56` |

Bank A's mode imbalance and lower ESS are explanatory diagnostics and a
repair trigger, not a hard pilot veto: all target/status, measure, protocol,
and finite gates pass, and removing it would create outcome-dependent bank
selection. The shared-state result must report both banks and may not pool or
drop A.

## Commands and artifacts

Generate the two fresh banks with the existing CPU-hidden pilot runner into
unique roots under:

`docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase42-independent-bank-replication/`

Then run:

```text
TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase42_2026_08_26.py \
  --authority-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase37-support-ladder/n256 \
  --fresh-root-a docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase42-independent-bank-replication/fresh-a-n256 \
  --fresh-root-b docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase42-independent-bank-replication/fresh-b-n256 \
  --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase42-independent-bank-replication/frozen-two-bank-audit \
  --steps 200 --seed 20260826 4211
```

Every failed attempt gets a new output root and an adjacent repair note. The
phase stops only for unavailable common support/target, an unrepaired harness
failure, or exhausted campaign budget.

## Result closure

The shared GPU/XLA audit and read-only report passed their hard engineering and
target/status gates. The result branch is
`bank_to_bank_variability_repair_triggered`: bank A is a clear support/mode
outlier, while bank B reproduces the Phase 41 descriptive residual reduction.
This is not a whitening or statistical-ranking result. Both banks remain in
the evidence set; neither was pooled, dropped, used for training, or used for
selection. The detailed result is
`docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase42-result-2026-08-26.md`.

The inter-phase repair is a third independent N=256 bank under the unchanged
target and objective. It is implemented by the v2.5 subplan and must retain
the v2.4 state-hash identity check before any bank-specific interpretation.
