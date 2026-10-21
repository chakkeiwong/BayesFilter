# Compact proposal covariance repair, 2026-10-12

Question: does evaluating shared proposal covariances before broadcasting remove
redundant work while preserving the same finite filter and all its derivatives?
This is the next shared implementation repair in the authorized master campaign,
motivated by component-cost-result.md. At most3000 GPU seconds/eight attempts,
within the remaining36000-second cumulative ledger; no budget extension.
First complete frozen-selection SIR/range mechanics before touching numerical
dependencies. Preserve the current source as the paired baseline.

For component b and ancestor j, the covariance is C_bj=A_bj Q A_bj^T.
Its total derivative is
dC_bj=(dA_bj) Q A_bj^T+A_bj (dQ) A_bj^T+A_bj Q (dA_bj)^T.
The prior component has A=I, the guided component has one common A, and the
flow component also has one common A when the model protocol explicitly says
the observation Jacobian is constant. Otherwise retain its N distinct A_j.
Compute C and dC at these natural extents (1,1,1 or N), then broadcast to
the existing return shapes. Broadcasting duplicates identical numbers and
their derivatives. Positive-definiteness/relative-margin checks commute with
this duplication: the all-valid reduction and minimum over repeated identical
eigenvalue margins are unchanged in real arithmetic. Keep all checks, thresholds,
matrix products, model callbacks, observation law, flow/reset iterations,
epsilon, particles and fixed designs. No model-name branches or new protection.

Comparator: the protected pre-repair source at identical theta,data,design,
controls,dtype and precision policy. Independent test reconstructs the full
broadcast covariance and its all-P product-rule derivative from returned maps;
test both shared and particle-specific flow cases,FP32/FP64,all five models.
Tiny full-filter regressions and failure-isolation checks precede GPU runs.
Primary engineering screen: same validity/full-step status, finite outputs,
all value/score components within1% of max(1,abs baseline); preserve raw gaps.
New invalid rows or precision failures veto this candidate, not the campaign.
FP64 arithmetic alone is not an oracle for true model likelihood/score.

XLA stable signatures, no pfor/autodiff score, memory growth and exact K2016
remain mandatory. Record first compilation separately, three warmed paired
calls when affordable, allocator peak, traces, source hashes, environment,
data/design identity and cumulative charged time. Timings without uncertainty
are descriptive and cannot establish superiority. Comparators are identical
implementations of the same method; Kalman/BPF/Gaussian baselines remain for
later method/HMC claims and are not replaced by this engineering test.

Default/assumption audit: owner N4032/FP32/TF32off retained. Mixed arithmetic
only in the already reviewed SIR/range optional scopes. Baseline controls are
frozen comparison inputs, not new-source tuning authority. Callback-constant
Jacobian property is conservative and tested against full particle assembly.
Compiler elimination of duplicated work is an open hypothesis; measure it.
Failure mode: changed floating-point grouping can amplify score differences.
Early diagnostic: full broadcast covariance/derivative parity, then full-horizon
paired comparisons. Source edits invalidate future tuning identity; any later
HMC under new source needs fresh scope tuning, never a stamped old selection.

Skeptical audit: the repair addresses repeated matrices, not a different
filter; it cannot claim full posterior readiness. Pre-mortem: staged cost could
overstate the gain, equality of covariances could miss a later recursive score
failure, and compiler changes could increase memory. Full-horizon comparisons
and resource records discriminate these. Candidate rejection preserves the old
source and triggers the next shared-cost investigation within budget; stop only
for the campaign budget or an actual validity/assumption failure. Save a result
decision/inference table and refresh the checkpoint after each substantive run.


Range mechanics follow-up (reviewed before GPU comparison): include the four
prescribed starts in the same protected/candidate graphs, two B2 calls per arm.
Record every row's value, score, completed steps and trace. This explanatory
check answers whether initial-region invalidity predates this covariance repair;
a new invalid row is a repair-candidate veto. Existing invalidity remains an HMC
promotion veto and triggers localization, not rejection of a parity-preserving
cost change. Inputs use saved controls as comparison inputs, not renewed tuning.
The master retains the same total/subphase budgets. No narrowing of starts.

Recovery clarification 2026-10-13: LG and range comparisons passed. Range
initial-region checks used baseline FP32-rounded mixture weights cast to FP64
and Python offsets converted through tf.cast; the failed pilot used FP64
mixture weights and tf.constant starts. These checks establish nearby-region
health and covariance parity, not an exact replay of that pilot. Preserve
this distinction and replay the actual force consumer with exact inputs.
KSC, predator and SIR comparisons remain ahead of that localization.
