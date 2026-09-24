# Automatic NeuTra post-training assessment

Question: can each q20 training endpoint automatically report numerical health,
learning status, clipping, scale saturation and transformed-score geometry, and
give its consumer an honest next action? The current repair runner checks loss
and export parity but ends without transformed-score checks. The main master
already owns fresh fixed-transport HMC tuning and posterior assessment.

## Evidence contract and implementation

Add a shared Python assessment to ordinary training rungs and repair endpoints.
Compute `grad_z log pi_z(z) + z` using the existing batch-native pullback
diagnostic, including the log-Jacobian score. Use one separately seeded batch
of the existing `validation.reliability_rows` size (32 for the campaign), a
stable TensorFlow signature and the configured XLA setting. Cache by immutable
map, target, beta, source scope, seed, row count and check policy. This costs one
target batch per assessed map, never one per optimizer update. Preserve a
cached result on resume and respect the existing worker budget before a probe.

Numerical non-finiteness, invalid target status and export-parity failure veto
that map's HMC trial. Missing checks cannot mean passed. Clipping on more than
half the observed updates is the existing occasional-guard contradiction;
bounded-tanh scale slope below 0.1 is the existing saturation alert. Both are
repair triggers, not evidence that NeuTra is invalid. The slope is
`d[s_max*tanh(a/s_max)]/da = 1 - (s/s_max)^2`; 0.1 means at least tenfold
attenuation and remains an explanatory threshold, not a promotion cutoff.
Loss improvement and observed plateaus retain their existing descriptive roles.
No finite residual cutoff is invented: Gaussianity on base draws is neither
necessary nor sufficient for useful HMC or global mode coverage.

Each export must carry the assessment and exact checkpoint identity. Serious
map selection and tuning must require a matching, completed numerical check.
Retain the existing public `tune_fixed_transport_hmc_kernel` route with identity
latent mass. Record fresh HMC tuning and posterior validation as required next
work; a training report cannot mark these passed. Repair-master results should
aggregate per-map next actions rather than leave an instruction to inspect
training manually. Existing completed artifacts remain unchanged.

## Verification and skeptical review before implementation

Primary engineering criterion: actual ordinary and repair producers emit the
assessment and actual HMC consumers reject missing, stale or invalid checks.
Independent analytic fixtures cover an exact Gaussian pullback, an incorrectly
scaled map, a missing log-Jacobian term, a saturated layer and invalid target
status. Verify graph/XLA compatibility, cache reuse, budget interruption and
unchanged optimizer/RNG state. Existing affected training and admission tests
must pass. Tests deliberately hide GPUs and make no q20 training-quality claim.

Baseline is the current checked-in/workspace call chain, not a historical q20
result. Preserve unrelated dirty changes on main, including the separate
posterior-checker work. CPU engineering tests have a 10-minute wall ceiling per
invocation; this is an operational timeout, not a scientific setting. Output:
`docs/plans/artifacts/q20-post-training-checks-2026-09-23/`. No new campaign or
GPU diagnostic run is authorized by this implementation plan; existing balances
remain 40.126 campaign hours and 8.080 diagnostic minutes until new work is
actually charged.

Audit: requiring perfect whitening or a loss plateau would be a wrong gate;
report both descriptively and leave downstream validity to HMC. Checking every
update would repeat the previous validation-cost mistake; check only assessed
endpoints and cache. Cached geometry must not be applied to a different map or
temperature. A failed candidate triggers repair, not abandonment of NeuTra.
Missing GPU validation limits performance claims but does not block these
CPU/XLA engineering regressions. This revised plan passes the skeptical audit.

## Implementation and terminal review

Implemented in `bayesfilter/inference/neutra_post_training.py`, called through
`FrozenLossCache.post_training_probe` from both `_evaluate_rung` and
`run_repair_arm`. Ordinary and repair exports carry `assessment.post_training`.
`choose_maps` and `validate_training_export` require that report for serious
HMC trials and check its checkpoint/map/beta identity and numerical validity.
The normal master retains its existing automatic fresh-tuning and sampling
stages. The bounded repair master reports per-map next actions; completing that
bounded run does not launch a new research phase or establish HMC readiness.
The repair watcher now renders numerical checks, residuals, repair signals and
next actions in its completion summary. Missing older reports display as not
checked. Resuming an older current endpoint regenerates its assessment without
an optimizer update; historical reports and genuine plateau comparisons remain
preserved.

The probe uses the existing exact batch pullback and log-Jacobian score. A
non-XLA finite-value assertion can fire before it returns: this is now recorded
as a numerical failure, while unrelated TensorFlow errors still propagate.
Budget exhaustion before the geometry batch pauses assessment; it cannot
produce an eligible export. Probes restore an immutable map, preserve optimizer
and RNG state, and cache results with checksums. One probe batch is additional
to existing loss/parity checks; q20 GPU compilation cost is not yet measured.

Verification records are in the artifact directory above:

- `check-01.xml`: 27 passed, two failures. One fixture omitted its required
  `log_prob` interface; the other exposed the early non-finite assertion above.
  Both were repaired without weakening numerical checks.
- `check-02.xml`: 78 passed, one failure in the existing full estimation test.
  Training and tuning completed for plain NeuTra and the ensemble. Posterior
  assessment rejected both because `positive_theta_2` has unavailable tail ESS;
  its R-hat and precision checks passed. This reproduces the separately
  recorded workspace checker issue. `integration-limit.json` preserves the
  exact observations. No posterior gate was relaxed.
- `check-03.xml`: **79 passed, one explicitly deselected**, 135.90 seconds.
  The deselected test is exactly the preceding documented integration failure.
  This final run includes the additional unchanged-endpoint reassessment test.
  CPU/XLA fixtures cover exact Gaussian geometry, wrong scaling, the nonlinear
  log-Jacobian term, invalid scores/status, saturation, cache reuse/corruption,
  budgets, immutable training state, both producers and both HMC consumers.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Use automatic post-training reports on new q20 endpoints | Focused producer, consumer and numerical regressions pass | Missing/stale/invalid reports cannot admit serious HMC trials | Target-specific GPU compilation cost and sparse-bank coverage | Run the check on retained research maps in the next funded phase | Existing q20 maps are well trained |
| Keep end-to-end estimation incomplete | Full integration reached training, tuning and sampling | Posterior information check still fails on binary tail ESS | Correct discrete-estimand information policy | Resolve the independent checker issue before posterior promotion | Full pipeline validation or an HMC estimate |

Terminal skeptical review: a finite, small residual under base draws can miss
posterior modes; a finite large residual can still permit useful HMC. Neither
is a promotion cutoff here. Clipping and scale alerts cannot change the map or
reject NeuTra as a direction. Additional checks run only at assessed endpoints,
and their cached values cannot silently certify posterior convergence. Source
changes are preserved in the workspace; no old research artifact was rewritten,
no GPU research work was launched, and no campaign balance was spent.

Active checkpoint update: automatic endpoint checks are implemented and focused
tests pass. A later funded GPU diagnostic has now checked the four saved maps
and their shared parent; see [the geometry results](bayesfilter-q20-post-training-geometry-results-2026-09-23.md).
All five probes were finite and valid, but training adequacy and posterior
readiness remain unestablished. Fresh fixed-transport HMC tuning and posterior
validation remain next; the workspace binary-tail-ESS issue remains open.
Do not relaunch the completed training repair. See `manifest.json` beside the
engineering test reports for their exact commands and environment; the geometry
result records the newer budget balances and diagnostic manifest.
