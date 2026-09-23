# M28 supplied residual-scale result

M28 tested the reviewed supplied-map question: does exact noncentering plus a
bounded residual conditional scale avoid the specific tail stiffness left by
the older partial fixtures, while the public candidate-set tuner retains every
verified pair and posterior precision is assessed in model coordinates?

The answer for this analytic development fixture is positive. The exact and
residual maps both completed their declared searches and all eight predeclared
posterior members passed the existing health, readiness and lugsail mean-MCSE
checks. This is evidence for these four synthetic fits. It does not establish
coverage, statistical superiority, a universal partial-whitening rule, learned
transport quality, or a new default.

## Design and controls

The exact control uses `v=3 z0` and `x_i=exp(v/2) z_i`. The residual fixture
uses the same model law and starts with
`delta(v)=0.5*tanh(v/6)`, `x_i=exp(v/2+delta(v)) z_i`. Its child-coordinate
curvature is in `[exp(-1), exp(1)]`; the full Hessian remains unbounded through
terms proportional to the squared child coordinates. The map is a frozen
analytic payload reconstructed by the supported dense-IAF codec. No map training
was performed.

Four GPU/XLA cells used seeds `2026092381` and `2026092382`, the broad
`L=(3,5,9,13,18,25)` search, per-L pilots, one refinement round, fresh
verification and `shortest_verified_l` selection of two distinct lengths before
posterior sampling. Each design declared 10,000 minimum and window warmup,
30,000 warmup maximum, 30,000 retained minimum, 60,000 retained maximum,
5,000 retained chunks, lugsail mean MCSE, and a 60,000-count budget with a
written reason. The automatic design check compared every resolved JSON field
with the plan table before launch.

| Map | Seed | Search candidates | Verified retained | Selected members | Retained draws | Posterior result |
| --- | ---: | ---: | ---: | --- | --- | --- |
| exact | 2026092381 | 30 | 3 | L=5, 13 | 30k, 30k | both passed |
| exact | 2026092382 | 44 | 6 | L=5, 13 | 35k, 30k | both passed |
| residual | 2026092381 | 81 | 14 | L=3, 5 | 35k, 30k | both passed |
| residual | 2026092382 | 78 | 13 | L=3, 5 | 30k, 30k | both passed |

All 36 verified candidates remain in the four pipeline records. The remaining
verified members are explicitly marked `unassessed_by_design`; they were not
silently dropped or allowed to change the selected slots. Fixed-count comparator
arms were assessed for every selected member. No posterior member was chosen
after inspecting posterior output.

The terminal audit independently checked candidate-result/checkpoint hashes,
verification receipts and seed lineage, target and transport scope identity,
inverse-mapped model starts, GPU/XLA placement and memory growth, tensor bundle
checksums, model-coordinate reconstruction from latent checkpoint chunks,
warmup exclusion, count settings and normal process exits. The first completed
cell's reconstruction audit also passed separately before the final audit.

The observed maximum retained modern R-hat values were approximately 1.00026;
these are posterior diagnostics only. They did not qualify tuning members,
repair epsilon, rank candidates or determine retention. Mean MCSE values for
the selected model quantities were below the declared 0.05 screen in every
member. Those finite-window diagnostics are operational evidence, not anytime
coverage or a proof of global exploration.

## Tests and execution accounting

The frozen source regression passed 36 tests covering the residual algebra,
Jacobian and score, directional finite differences, bounded conditional
curvature versus unbounded full Hessian, zero-amplitude recovery, starts,
scope mismatch, public tuning/replay, failed-first-member continuation,
candidate retention and count-table drift. The documentation contract suite
passed 15 tests. An earlier curvature-test defect and two setup failures are
preserved in the attempt records; they were repaired before numerical launch.

The official LaTeX chapter now states why bounded applied log-scale is
insufficient, derives the residual fixture and records the remaining Hessian
qualification. The reference guide has the corresponding fixture boundary.
The book rebuilt successfully after a bounded-path retry; bibliography and
undefined-reference checks passed, and pages 418–419 were visually inspected.

Metered M28 charges are 869.37 CPU and 2,439.85 GPU worker-seconds, including
tests, audits, setup failures and the document build. The phase allowance was
1,800 CPU and 4,800 GPU seconds, leaving 930.63 CPU and 2,360.15 GPU
seconds inside M28. Against the opening campaign ledger, 75,647.83 CPU and
79,820.09 GPU worker-seconds remain. The terminal reconciliation is
`artifacts/hmc-repair-master-2026-09-16/m28-r1/reconciliation-terminal.json`.
The published [audit summary](artifacts/hmc-repair-master-2026-09-16/m28-r1/terminal-audit-summary.json)
includes the full local audit's checksum; raw tensors and repeated diagnostic
reports remain preserved locally.

## Decision and remaining gaps

| Decision | Primary criterion | Veto status | Uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close M28 engineering cell | Correct transformed target/score, starts, receipts, retention, archives, count match and normal exits | No shared veto; all four cells passed | Two seeds per map and one synthetic law | Move to child-process profiling and cost attribution | No coverage, ranking, default or learned-map claim |
| Retain exact and residual fixtures | All eight selected posterior screens passed | No health, finite-state or precision veto | Model-coordinate means can remain difficult in other regimes | Reuse only as declared diagnostics | No general geometry guarantee |
| Preserve broader campaign | M28 does not answer global modes, full-fit power or MacroFinance integration | Those gaps remain open | Confirmation inventory remains under-budgeted at measured prices | Execute M29 only under its refreshed plan | No claim that the campaign is complete |

| Inference status | Evidence and limit |
| --- | --- |
| Hard veto screen | All eight selected members passed the declared numerical health checks; native TFP divergence telemetry is unavailable, not certified zero |
| Statistically supported ranking | None; two seeds per map do not justify ranking viable kernels or maps |
| Descriptive-only differences | Candidate counts, acceptance, retained counts, precision estimates and runtime |
| Default readiness | No default change; this is one analytic fixture and a development allocation |
| Next evidence needed | Adequate independent fits for coverage, global exploration checks and target-specific validation for learned maps or consumer targets |

The strongest alternative explanation for apparently adequate child-mean
precision is that these finite samples missed rare large model-coordinate
values. An independent adequately replicated coverage experiment could overturn
that interpretation. The weakest evidence is therefore statistical calibration,
not the checked algebra or receipt/retention mechanics. This phase supplies no
ranking between exact and residual whitening.

The next executable phase is
[M29](bayesfilter-hmc-post-m28-next-phase-2026-09-23.md). It repairs the
profiling boundary so isolated numerical children, rather than a waiting
coordinator, are profiled before any maintenance or affordability decision.
