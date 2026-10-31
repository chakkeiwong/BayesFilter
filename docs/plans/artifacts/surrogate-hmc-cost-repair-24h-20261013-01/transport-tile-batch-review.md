# Exact-tile scheduling probe

The diagnostic-only candidate computes the same row/column tile equations with
block axes batched. CPU FP64 parity and analytical score checks passed. GPU/XLA
primitive parity is finite with maximum scaled errors recorded in the JSON.
Warm candidate/baseline ratios are **0.9816 KSC** and **0.9454 LGSSM**. This is a
small descriptive reduction (about 1.8% and 5.5%) below the predeclared 10%
nomination threshold. The candidate is therefore not promoted to a full-filter
source change; the working source and selected tuning remain unchanged.

The probe does not establish whole-filter cost, statistical speed ranking,
posterior correctness or LEDH admission. It does show the earlier arithmetic
cache candidates were the wrong place to seek cost: exact tile scheduling has a
small benefit, but not enough to pay for a full retune under this evidence.
