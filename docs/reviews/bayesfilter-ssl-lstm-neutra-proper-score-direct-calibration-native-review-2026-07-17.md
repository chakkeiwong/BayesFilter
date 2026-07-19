# Proper-Score Direct Calibration Native Review

Date: 2026-07-17

Scope:

- `docs/plans/bayesfilter-ssl-lstm-neutra-proper-score-direct-calibration-plan-2026-07-17.md`
- `bayesfilter/inference/predictive_equivalence.py` additive dual-loss and batched APIs
- `docs/benchmarks/run_ssl_lstm_neutra_proper_score_direct_calibration_2026_07_17.py`
- focused tests for this lane

Verdict: `AGREE_GPU_SMOKE_ONLY`

## Findings And Repairs

| Finding | Severity | Repair |
| --- | --- | --- |
| One average loss cannot distinguish persistent negligible mean `0.05` from local material variance ratio `1.25` | Material design flaw | Made average and horizonwise losses co-primary over one joint region; thresholds derive prospectively from loss-scale anchors |
| Existing trust-region lower solver assumed positive-definite loss, but horizon losses are rank-two positive semidefinite | Material numerical flaw | Added null-space pseudoinverse handling and scalar/batched parity tests |
| A scalar Python loop over 11 losses and every replication would make the direct experiment impractical | Material feasibility flaw | Added TensorFlow/XLA batched HAC and exact batched trust-region surfaces; statistical procedure is unchanged |
| Initial 160-replication contract could certify a 5% false/invalid target only with zero events | Material evidence-design weakness | Froze 256 replications before outcomes; under the simultaneous allocation up to two events can remain certifiable |
| Invalid condition numbers or KKT values could leak infinity into strict JSON | Artifact-validity flaw | Report only finite descriptive maxima or `null`; invalid probability remains co-primary |
| Plan wording implied MMD values would be computed | Boundary mismatch | Clarified that MMD is omitted and cannot affect the primary gate |
| Batched covariance API did not explicitly reject asymmetric input | API hardening | Added a scale-aware symmetry guard and focused failure test |

## Audit

The threshold is the arithmetic midpoint between the worst negligible and
nearest material anchor on the declared proper-score scale. It is not selected
from calibration outcomes. The average criterion covers persistent effects;
the horizon criterion covers local effects. Both extrema use the same
20-dimensional estimate, covariance, and `chi2_20(0.95)` radius.

For independent four-chain arms with long-run covariance matrices `S_L` and
`S_R`, the runner concatenates four `+2 I_L` influence chains and four `-2 I_R`
chains. The batched HAC spectral average is `2(S_L+S_R)` and its pooled-mean
division by `8N` yields `(S_L+S_R)/(4N)`, exactly the covariance of the
difference of two four-chain means.

The truth sign matches the estimator `left - right`: added right mean and log-
variance shifts appear negatively. Persistent and local truth-vector tests
cover this mapping.

The operating evidence uses 11 required families, four claims, and at most two
looks. One-sided exact Clopper--Pearson bounds use tail alpha `0.05/88`, so the
reported family/rung claims have simultaneous coverage at least 95% without
independence assumptions. Invalid rows count as uncovered and cannot contribute
to required decisions; invalid probability is separately required below 5%.

MMD, HMC, retained archives, and G/H confirmation are absent from the primary
runner. A passing smoke is mechanics-only. A passing material result applies
only to the declared controlled laws.

## Checks Before Verdict

- `34 passed` in the initial focused pair before the final symmetry test.
- End-to-end CPU-hidden non-XLA probe: shapes `[2,4,64,2,10]`, both HAC rows
  admissible, both 11-loss rows admissible, all upper bounds finite.
- Python compilation passed.
- Scoped diff whitespace check passed.

The final focused suite, source hashes, GPU residency, XLA compilation, and
immutable smoke receipt must be checked after this review. Material execution
is not authorized by this verdict until the smoke passes and the plan is
updated with its exact receipt hash and frozen command.

## Smoke Repair Addendum

The first trusted smoke failed before receipt creation because
`tf.vectorized_map` around per-replication stateless generation lowered to the
variant operation `TensorListReserve`, unsupported by XLA GPU. This is an
implementation/XLA failure, not evidence against the statistical design.

The generator was repaired to draw one dense stateless Philox tensor per
`(rung, family, batch, arm)` and apply one time-axis `tf.scan` over dense
numeric tensors. Folded arm domains remain independent, and all tensor elements
in a stateless-normal call are independent. The repair passed `35` focused
tests, Python compilation, diff hygiene, and a CPU-hidden XLA-host generator
probe with shape `[2,4,64,2,10]`, finite output, and one trace.

Verdict remains `AGREE_GPU_SMOKE_ONLY` for one repaired smoke attempt.

The repaired GPU smoke then reached the batched HAC/trust-region compilation
but exhausted its bounded outer time without a receipt. Inspection localized
the compile blow-up to Python-unrolled 64-step bracketing and 96-step bisection
loops. These loops were replaced by fixed-count XLA `tf.while_loop`s; iteration
counts, exact equations, tolerances, and statistical criteria did not change.
Scalar/batched parity still passes. A CPU-hidden XLA-host probe for the exact
`[2,20]` estimates, `[2,20,20]` covariances, and 11 loss matrices compiled and
ran promptly, with both rows admissible and maximum KKT residual below
`1.4e-19`.

Final verdict: `AGREE_ONE_FINAL_GPU_SMOKE_ATTEMPT`. Further failure is a stop
for a separately planned execution-surface repair, not authority to raise the
budget or weaken XLA.
