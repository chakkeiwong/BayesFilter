# SSL-LSTM NeuTra Phase 4 Bounded Training Plan

Date: 2026-07-14

Status: `AUTHORIZED_TIER2_SEQUENTIAL_EXECUTION`

Owner authorization recorded 2026-07-14: the proposed sequential ladder,
independent A/B seeds, sequential stopping, and at most one trusted GPU-hour are
authorized.

## Objective And Entry Conditions

Train and freeze viable SSL-LSTM transports under a sequential ladder, using
independent training replications and diagnostics that can reject inadequate
support without treating loss as posterior evidence. Phases 0-3 have passed.

## Research Intent And Evidence Contract

| Field | Prospective contract |
| --- | --- |
| Question | Can the correct trainer produce at least two immutable candidates with finite, stable, sufficiently broad mapped support to justify exact transformed-target preflight? |
| Exact target | Locked four-coordinate SVD-UKF target, semantic SHA-256 `549efdf2aa5d9534226cb29c3678489d92766f92e6140901355eac33618f719e` |
| Baseline ladder | Untrained identity; trained diagonal affine; plain one-stage dense IAF; enhanced multi-stage/mixing only after the plain arm is evaluated and a named repair trigger fires |
| Primary pass | At least two independent frozen candidates pass engineering, heldout-loss stability, mapped-support, roundtrip, score, and artifact gates |
| Hard vetoes | Target/signature drift, nonfinite value/score/update, corrupt state, reload mismatch, invalid host finite receipt, GPU/XLA fallback, or data/seed overlap |
| Promotion vetoes | Missing original-start-neighborhood or moderate-shell coverage, material A/B gate instability, scale saturation, validation degradation, or failed frozen score/roundtrip checks |
| Repair triggers | Affine residual curvature; plain IAF support gap; saturation; validation instability; one failed seed with one viable seed |
| Explanatory only | Training/validation loss, gradients, parameter norms, runtime, mapped quantiles, local Hessians, and continuous coverage distances |
| Nonclaims | No posterior correctness, complete mode/tail coverage, HMC readiness, superiority, predictive equivalence, or readiness |
| Result | `docs/plans/bayesfilter-ssl-lstm-neutra-phase-4-bounded-training-result-2026-07-14.md` |

## Skeptical Pre-Mortem

The run could pass training loss while collapsing onto one reverse-KL mode; the
original four starts and frozen prior/shell/tail probes distinguish this. It
could fail because learning rate or initialization is poor rather than because
NeuTra is unsuitable; the affine control, two independent seeds, heldout noise,
and one predeclared learning-rate repair distinguish tuning from direction
failure. It could appear reproducible because A/B share randomness; role-coded
seed and artifact checks forbid overlap. It could pass graph asserts that XLA
ignored; every recorded step/checkpoint uses host-synchronized finite checks.

The post-authorization audit found and repaired one proxy error before runtime:
draws from the broad parameter prior and far-tail points are not necessarily
posterior-mass regions. They are finiteness, saturation, and repair diagnostics,
not required high-density transport coverage. Original-start neighborhoods and
moderate shells retain the prospective coverage veto. Audit status:
`PASS_FOR_AUTHORIZED_PHASE_4_EXECUTION`.

## Sequential Candidate Ladder

All serious runs use trusted GPU/XLA, `float64`, TF32 enabled, soft placement
disabled, and the Phase 3 trainer. Training roles are A/B `2101`, `2102`;
heldout validation roles are `2201`, `2202`.

1. Affine control A/B: `500` steps, batch `64`, learning rate `1e-3`.
2. Plain dense IAF A/B: one stage, hidden `(8,8)`, `tanh`, `s_max=1`, `2000`
   steps, batch `64`, learning rate `1e-3`.
3. One repair rung, only if both plain seeds fail for named optimization rather
   than target/support invalidity: same topology with learning rate `3e-4`, up
   to `2000` additional fresh-start steps using repair roles `2110`, `2111`.
4. Enhanced multi-stage/mixing is not authorized by this plan. If the plain arm
   is finite/stable but leaves a material ridge/support gap, stop with an
   architecture-repair result and draft a separate amendment.

Checkpoint every `100` steps and host-synchronize loss, target value, logdet,
gradient norms, variables, and optimizer state. Preserve every failed state.
Validation occurs every `250` steps on fixed independent base noise and cannot
feed optimizer updates.

## Candidate Gates

- exact target/config/seed/source bindings and non-overwriting artifacts;
- all host-synchronized values finite;
- exact resume replay at an early checkpoint;
- frozen payload reload and Phase 2 score/roundtrip checks;
- historical A4 affine center/factor and starts are converted exactly to free
  coordinates; each of the four start neighborhoods contains its center and
  `+/-0.10` in every historical latent coordinate;
- moderate shell probes are historical latent `+/-2.0` coordinate directions;
  far-tail probes are `+/-4.0` directions;
- every original-neighborhood and moderate-shell target point must invert to a
  finite standard-normal base radius `<=4.30` (approximately the 99.9% radial
  screen in four dimensions);
- `16` actual-prior probes use seed `[20260714,3301]`, locked prior center, and
  standard deviation `4.0`; their inverse radii are explanatory, while
  nonfinite transport/target/score behavior is a repair or validity veto;
- far-tail inverse radii are explanatory; roundtrip, score, logdet, and
  saturation finiteness remain mandatory;
- log-scale saturation fraction and mapped nonfinite fraction below
  prospectively encoded hard caps: for dense IAF, at most `0.05` of heldout
  scale logs may satisfy `abs(scale_log) >= 0.95*s_max`; affine log scales must
  remain finite with `abs(raw_scale) <= log(10)`;
- fixed heldout batches have `64` rows. Final minus initial paired per-sample
  loss must have one-sided 95% upper bound below zero; this is a trainer gate,
  not candidate promotion by loss;
- roundtrip maximum absolute residual must be `<=1e-9` and every transformed
  score must be finite;
- A and B are compared with uncertainty-aware heldout estimates; continuous
  differences remain descriptive unless their pass/fail gate outcomes differ.

## Resource Request And Sequential Stop

Requested cap: **1 trusted GPU-hour total**, including compilation, affine A/B,
plain IAF A/B, and at most one declared learning-rate repair pair. The measured
Phase 3 steady step was about `0.048s` at batch `4`; the one-hour cap is
conservative for larger batches, validation, checkpoints, and compilation.

Stop immediately for a hard veto. Stop after affine if target/runtime validity
fails. Stop after plain A/B if at least two candidates pass; do not spend the
repair budget merely because one descriptive metric could improve. Run the
repair rung only for a preclassified optimization failure. Do not borrow the
historical A4 HMC budget; this is a separate Phase 4 training budget.

## Authorization Boundary

The owner authorization permits implementation and execution of this bounded
runner. It does not authorize HMC, forecasting, enhanced multi-stage topology,
or use of the historical A4 HMC budget.
