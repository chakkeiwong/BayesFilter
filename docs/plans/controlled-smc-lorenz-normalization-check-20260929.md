# Lorenz source normalization check

Authorized scope: the active controlled-SMC/KDM replication master, C1 source
inspection. This is a CPU-only independent author-code reference check, not a
TensorFlow implementation or an empirical Lorenz replication.

## Question and evidence contract

The published Gaussian likelihood includes the constant omitted by the author
Lorenz drivers. Does the actual author filtering consumer preserve trajectories
and normalized weights when that constant is restored, while shifting the final
log normalizer by `-H*p/2*log(2*pi*observation_variance)`? The derivation and exact
paper/source anchors are in
[the source audit](../reference/controlled-smc-remaining-source-audit-20260929.md).

Execute the unchanged Lorenz `cSMC_Resample` with the analytic observation/APF
policy, I=0. Compare three arms with identical seeds: original source convention;
normalized observations with the original policy; normalized observations with
the matching policy constant. Use d=8, all 101 supplied observations, forcing
4.8801, initial variance .01 and transition variance .001, RK4 step .01 and
observation interval .1. The two observation variances are 1e-4 and 1e-2. The
second variance is an algebra check on the same data, not the Table 1 design.
Use N128 and seeds 20260929 and 20260930: twelve inexpensive forwards.

Pass requires all paired trajectories to agree within 1e-9 absolute plus 1e-9
relative tolerance, exact ancestry, agreeing ESS, normalized particle weights,
finite outputs and final log-normalizer differences agreeing with the formula.
These are engineering and numerical identities, not stochastic rankings. A
failed identity vetoes the proposed normalization convention until diagnosed;
it does not reject controlled SMC. Runtime and minimum ESS explain failures
only. Preserve all original forwards in an Octave archive and each error in JSON.

Do not compare partial cumulative normalizers naively: changing the policy
constant can move the future potential's constant into an earlier increment.
The required identity is for the final complete target. Do not infer that a
refitted policy is numerically invariant; fitting is deliberately absent here.
The paper-scale adaptation, APF cost matrix, likelihood curves and TF/GPU/XLA
port remain unfinished even if these identities pass.

## Defaults, audit and budget

Source settings and functions are pinned at
2696f6d611aae172d10102cb7bfbfd314bb6b88f. N128 and two deterministic seeds are
small identity fixtures, not paper repetition counts. I=0 isolates the
normalization effect without conflating it with regression conditioning. The
same independent Octave/statistics environment already executed the neuroscience
source. CPU devices only, CUDA_VISIBLE_DEVICES=-1; no framework import, package
mutation, NumPy runtime or default-policy change.

Skeptical audit: PASS for this identity question. Wrong Gaussian constants,
changing data across arms, mismatched seeds, raw weight comparisons and partial
normalizers could give misleading conclusions; the fixture fixes those choices
and checks final targets and normalized weights. An unchanged trajectory alone
cannot establish a correct likelihood; the derived final shift is indispensable.
No heuristic performance comparison is appropriate to this deterministic
constant-restoration identity.

Run phase `c1_lorenz_normalization` through scripts/run_controlled_smc_kdm.py,
with 40 CPU seconds per attempt, 80 seconds aggregate and at most two attempts,
inside the existing CPU300-second allocation and inherited CPU balance. Queue
after the K diagnostics to preserve serialized execution. The existing total
attempt limit of eighteen permits one initial attempt; any repair requires an
explicitly recorded reassessment of that limit, not new compute authorization.
Preserve unique attempt directories under the existing campaign root.
