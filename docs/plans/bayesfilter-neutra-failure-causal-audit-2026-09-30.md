# Causal audit of the completed NeuTra warm-start campaign

Completed findings and derivations:
`bayesfilter-neutra-failure-causal-findings-2026-09-30.md`. Saved-map diagnostics
used 35.08 GPU process-seconds and 59.11 CPU core-seconds, charged to the shared
ledger. No training or numerical-policy changes were made. The next justified
work is the ordered set of targeted repairs in that findings document.

The owner requested tracing the code and mathematics to explain the failures.
This is a diagnostic continuation of the authorized campaign, with no change to
training policy, architecture, target, posterior criteria or previous results.

Question: which failures are caused by the sample generator, the learned
transport or its optimization, initialization, the validation procedure, or
bounded HMC tuning? Trace the executed call chain and saved source snapshots,
then use the smallest numerical or analytic checks that distinguish those
mechanisms. A source-level hypothesis alone is not an established cause.

| Item | Diagnostic contract |
|---|---|
| Comparators | The recorded successful Gaussian and warped cases; the actual failed mixture/funnel attempts; exact analytic benchmark moments and derivatives |
| Pass criterion | A failure explanation must reproduce the recorded decision and follow from checked code plus a derivation or targeted numerical evidence |
| Continuation veto | Mismatched source/target, corrupt input, nonfinite diagnostic arithmetic without localization, or diagnostic allocation exhausted |
| Repair trigger | A demonstrated wrong computation, misleading admission decision, inappropriate initialization or insufficient exploration |
| Explanatory only | Loss curves, clipping, empirical curvature and score residuals, finite-difference checks and descriptive cross-case differences |
| Not concluded | Global impossibility of IAF, rejection of warm starts, method superiority, production readiness, or all-mode guarantees |

Work order: reconstruct failure-specific decisions; trace objectives, map
parameterization and gradients; derive Gaussian-plateau and funnel proposal
geometry where supported; inspect all failed HMC verification records and start
states; run focused diagnostics only for unresolved mechanisms. Document direct
findings separately from hypotheses and prescribe the next smallest repairs.

Use standard-library artifact inspection and explicitly diagnostic CPU/TF
reference checks first. If framework execution is needed, CPU diagnostics hide
GPUs; GPU diagnostics use trusted GPU 1 only if free, with verified memory
growth. The diagnostic reservation is at most 900 GPU process-seconds and
3,600 CPU core-seconds, charged against the existing allowance. These are
convenience ceilings for bounded investigation, not scientific thresholds;
their purpose is to avoid launching another training campaign while explaining
the completed one. Saved-map checks use recorded seeds and exact maps.

Artifacts: `docs/plans/artifacts/neutra-warm-start-master-2026-09-29/campaign-r1/causal-audit-20260930-r1/`.
Record actual commands, source hashes, CPU/GPU settings and consumed resources
for numerical checks. Preserve earlier evidence. No package or environment
changes, new training runs or source-default changes are part of this audit.

Skeptical review: an empty HMC verified set is not automatically posterior
disagreement; a failed warm-map shape screen is not automatically a wrong
teacher; tiny-step acceptance is not mixing; and matching the Gaussian forward
loss is not proof that every flow parameter has zero gradient. The audit must
inspect those distinct quantities. The benchmark's exact oracle is a reference
for diagnosis, not a replacement teacher or an unknown-target deployment claim.
The plan passes for this bounded explanatory scope.

The targeted numerical check restores three existing maps: one ordinary-mixture
plateau, the lowest recorded ordinary-mixture development loss (a descriptive
selection, not a statistical winner), and funnel/Gabrié seed 11. It checks
forward-loss directional derivatives at displacement 1e-5, affine residuals,
and the funnel's exact recorded HMC starts versus learned-map and oracle draws.
The displacement is an FP64 finite-difference hypothesis; a disagreement
requires a step-size ladder before attributing a derivative defect. No saved
map or optimizer is updated. The executed map/target sources must match saved
manifest hashes.

For initialization diagnosis only, compare 64 independent three-leapfrog
proposals at the campaign's smallest attempted epsilon .0625, under each start
bank. This is explicitly a debugging exception to posterior chain execution;
it is neither a tuner nor an admissible posterior run. The same frozen map,
target, momenta seed and proposal rule are held fixed. Oracle starts localize
initialization effects only. Four-state curvature checks use native nested
GradientTapes with coordinate sums, no pfor. The script has a 300-second wall
and 600-core-second CPU cap inside the diagnostic reservation; GPU 1 was checked
free. Full commands and memory policy are captured in its manifest.
