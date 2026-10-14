# Lorenz refinement validation result — 30 September 2026

The scoped Lorenz implementation agrees with independent reference calculations
when each fit receives identical inputs. Attempt72 passed70 checks across all
eight executed refinements (forcing2.5:4,4.8801:1,8.5:3). The largest coefficient
error divided by its original tolerance was0.045633. The maximum represented
quadratic discrepancy on a fitting cloud was1.429e-8.

Attempt70 remains FAIL under its original independent-trajectory coefficient
criterion:14 of144 checks failed. All120 non-refitted-coefficient checks passed,
including forward particles, weights, likelihoods, ESS, exact ancestry,
stopping, all four APF counts and closed-Gaussian APF identities. Attempt71
showed that separately rounded inputs explain the first-refinement discrepancy:
cloud differences were below8.0e-15 and incremental-log-weight differences below
7.46e-10. Fitting narrow clouds in raw coordinates amplifies those differences.
No tolerance, numerical solver, objective or accepted output was changed.

The revised contract at72 was written before execution. It combines
identical-input fit parity with70's original forward/controller checks.
Its PASS does not retroactively change70 or establish exact coefficient
agreement on separate trajectories. The Gaussian policy and APF identities
follow published sections4.3/6.2, equations42–44, and the pinned source ranges
listed in the [plan](controlled-smc-lorenz-refinement-validation-20260930.md).
The runtime remains the centered-QR equivalent-objective implementation,
not a replay of the author's unstable normal-equation coefficients.

| Decision | Primary criterion | Vetoes | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Proceed to a bounded Figure6 cost pilot | All8 identical-input fits pass unchanged coefficient tolerance; forward/APF/stopping checks pass | No invalid precision, rank, ancestry, nonfinite value or unexpected trace in this scope | New seeds/forcing points may expose other numerical failures | Measure repeated execution on three source-grid points before budgeting4000 pairs | Figure6, Table1 or whole-method replication |
| Preserve70 as failed |14 independent-trajectory coefficient checks failed | Blocks that original replay claim | Sensitivity beyond these clouds remains unevaluated | Retain the failure and72's separately named contract | A generic solver defect or failure of controlled-SMC theory |

| Inference status | Assessment |
|---|---|
| Hard veto screen | Passed under72's revised modular contract;70 original FAIL retained |
| Statistically supported ranking | None; these are deterministic comparisons on frozen random designs |
| Descriptive-only differences | Run times and numerical residual magnitudes |
| Default-readiness | Not established; scoped FP64 reference lane |
| Next evidence needed | Fresh-seed validity and cost pilot, followed by an explicitly contracted empirical matrix |

All numerical loops are native TensorFlow graphs with stable signatures and XLA.
Controller, forward and fit tracing counts were1. The independent NumPy/SciPy
code is deliberately CPU-only and cannot be imported into the runtime lane.
GPU memory growth was verified; TF32 was disabled for this FP64 source-reference
scope; attempt70 peak TensorFlow allocation was460,066,048 bytes.

Run evidence is under
artifacts/controlled-smc-kdm-replication-20260928/attempt69_c1_lorenz_refinement_reference,
attempt70_c2_lorenz_refinement_validation,
attempt71_c1_lorenz_refinement_diagnosis and
attempt72_c1_lorenz_refinement_matched_validation. Each contains its command,
source/input hashes, environment, seeds, hardware, wall time and result in
manifest.json, with full logs preserved. Charges:69 CPU13.002931s,
70 GPU76.881032s,71 CPU3.999600s,72 CPU3.673528s.
Cumulative campaign charges after72:CPU304.451723s,GPU13967.413987s.
Inherited remaining:CPU448.715213s,GPU116983.886364s (32.4955GPU hours).

Terminal skeptical review: PASS for this limited conclusion. The strongest
alternative explanation is agreement on shared implementation mistakes;
independent SVD algebra, author replay and closed-Gaussian checks reduce but do
not eliminate it. A fresh-seed rank/precision failure or failed conditional
reference check would overturn the wider-execution decision. Eight fits on one
dataset cannot establish robustness. Neither controlled-SMC nor KDM empirical
replication is complete.
