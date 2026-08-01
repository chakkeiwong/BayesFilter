# P5 Result: Chapter 18b Structural NeuTra Cells

Date: 2026-07-16

Program ID: `multimodel-neutra-filter-posterior-20260715`

Decision: `P5_COMPLETE_TWO_PRECISE_BLOCKERS_CONTINUE_P6`

## Outcome

P5 has two honest terminal cell states:

| Cell | Terminal state | Binding evidence |
| --- | --- | --- |
| `STR-UKF` | `COMPARATOR_BLOCKED_GEOMETRY` | typed target `e8d78a8ee12245fee2e6c4c739d9dc03d672e8dd9a96bfbd492b426a72e1c665`; source HMC result SHA-256 `5c01b1f9b36e1c23b4b09b22c13b98dded8f2200d180f98eb1c2b4706bacec46`; affine geometry result SHA-256 `0ee609c414673d3dc3f797aa135ae2de349cec5be39df18241b6de584a5f12d9` |
| `STR-ZC` | `TARGET_BLOCKED_EXTENSION_ROUTE_NOT_DESIGNED` | the cell is `extension_or_invention` by definition; no graph-native target, observed-data value/score route, identity, HMC, or training artifact exists |

No structural NeuTra transport was trained. P5 does not support a structural
NeuTra confirmation claim.

## Admitted Structural Evidence

The target-design rung admitted the graph-native T=100 structural UKF target:

- all 33 T=100 design information rows had rank five;
- minimum eigenvalue was at least `0.26344` and maximum condition number at
  most `1694.10`;
- all 4,096 prior-predictive trajectories were valid through T=200;
- central-FD information derivative gaps were at most about `2.73e-5`;
- the artificial innovation negative control changed the innovation variance
  and likelihood as predicted and remained ineligible for target identity.

The independently recomposed typed target then passed source decomposition,
total score, negative substitutions, CPU/GPU XLA parity, and recursive hash
checks. The CPU/GPU value gap was `5.68434e-14`; the score gap was
`2.09610e-13`.

These are engineering and target-identity results. They do not establish HMC
sampleability, filter exactness, or NeuTra quality.

## Comparator Attempts

| Attempt | Classification | Result |
| --- | --- | --- |
| source HMC attempt 01 | `HARNESS_GPU_MEMORY_POLICY_IMPORT_ORDER` | no HMC; import order repaired |
| source HMC attempt 02 | `SAMPLER_SOURCE_KERNEL_HEALTH_FAILURE` | selected step `0.04`; first 1,000-draw sequential warm-up had one energy-error divergence; zero retained draws |
| affine geometry attempt 01 | `COMPARATOR_BLOCKED_GEOMETRY` | terminal score `0.0050403880` exceeded `0.0001`; no affine HMC launched |

The source attempt did not hit its R-hat cap. It failed the health gate first.
Its warm-up is retained as tuning-only evidence and never enters inference.
The affine diagnostic had stable positive-definite curvature and exact wrapper
checks, but the plan correctly stopped at its independent mode gate.

## Integrity, Checks, And Budget

- Target-design result SHA-256:
  `214c6ba1e79d6589978b233a75015457ea08888e06d26d84203098d2736c4103`.
- R1B CPU result SHA-256:
  `73fd7a10fd89999993b2b88b636774df489e984e1c589cb3efff57ce2d3ea83d`.
- R1B GPU result SHA-256:
  `f36f9197c56b2bc88276b234c6aa0e25ea992220511272d88cece86918a910f3`.
- Source HMC recursive ledger SHA-256:
  `2b1ad3878a020f0ad26dbec7b5a008649a2a341f27bc303166ff8ba52002deed`.
- Affine geometry recursive ledger SHA-256:
  `4ac945f4dcaa86d8551ebcf820297195998607ff8a859b5270cd552189059ada`.
- Focused structural and affine-repair regression: `12 passed`.
- Recorded completed run-manifest wall time across P5: `2462.8041` seconds
  (`0.6841` hours), including CPU and GPU-labeled attempts; this is a broad
  upper accounting sum, not a pure GPU utilization claim.
- Several failed pre-result harness attempts have no trustworthy full wall-time
  manifest. Even conservatively, no P5 phase-budget veto fired.

Claude review of private workspace content was platform-blocked. The limitation
is advisory and recorded. Local skeptical review caught and repaired the
material source-failure misclassification before the affine attempt.

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| close P5 with two blockers and continue P6 | structural target/identity admitted; mandatory comparator not admitted | source health veto and affine mode gate supported; STR-ZC target absent | whether a future larger mode-locator repair would admit STR-UKF HMC | refresh P6 from actual SIR observed-data target gaps | structural NeuTra success, HMC convergence, Zhao-Cui structural reproduction, filter exactness, calibration, readiness |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | source fixed kernel and affine geometry repair each failed their frozen gate; STR-ZC lacks a target |
| Statistically supported ranking | none |
| Descriptive-only differences | source probes, acceptance, mode trajectory, Hessian spectrum, runtimes |
| Default readiness | not established |
| Next evidence needed | prospectively reviewed P5 comparator re-entry; independently, P6 full observed-data target admission |

## Engineering, Numerical, And Scientific Ledgers

| Ledger | Result |
| --- | --- |
| Engineering correctness | graph-native model, deterministic identity, negative-control distinction, typed identity, batch/XLA paths, CPU/GPU replay, and scoped tests passed |
| Numerical/sampler validity | target evaluations remained finite/status-valid, but source HMC had one energy-error divergence and affine geometry failed its mode gate; no retained comparator exists |
| Scientific interpretation | the declared structural UKF posterior is implemented and identified for one fixture; NeuTra and HMC success are unsupported; STR-ZC remains an undesigned extension |

## Post-Run Red Team And Drift Audit

The strongest alternative explanation for the `STR-UKF` blocker is an
under-budgeted eight-iteration mode locator, not an invalid posterior. That is
why the result is `COMPARATOR_BLOCKED_GEOMETRY`, not target or research-direction
rejection. The weakest evidence is posterior sampleability. Execution review
also corrected the tempting but wrong claim that source warm-up exhausted the
R-hat cap: it stopped on an energy-error veto after one chunk. No failed warm-up
was pooled, no proxy admitted a comparator, and no P5 evidence was used to issue
a P6 target.

