# SVX-ZC Value Validation, NeuTra, And HMC Terminal Result

Date: 2026-08-02

## Outcome

SVX-ZC completed the planned value, batch-native NeuTra, statistical HMC
tuning, and shared sequential HMC campaign for the frozen T=10 finite target.

- Value capacity passed for degree 10, rank 2, quadrature order 25.
- Center-frozen CPU-built UKF initializer identity passed all local value cells.
- Batch-native TensorFlow value/score/status and CPU XLA tests passed.
- One selected 5,000-step batch-128 GPU/XLA NeuTra training passed all hard
  target/status and frozen/trainable parity gates.
- Statistical broad-grid tuning produced one viable unranked pair: `L=25`,
  `epsilon=0.8434292653387`.
- Shared sequential NeuTra HMC passed warm-up readiness, retained R-hat,
  bulk/tail ESS, finite/status/movement health, and the one-seed truth-tail gate.

This establishes a valid HMC run for this frozen approximate SVX-ZC target and
scope. It does not establish exact filtering, exact score, source faithfulness,
cross-scope capacity, sampler superiority, or default/production readiness.

## Claimed And Computed Quantities

| Item | Claimed target | Quantity computed | Verdict |
| --- | --- | --- | --- |
| Value | Finite SVX-ZC fixed-adjacent squared-TT likelihood at T=10 | Degree-10/rank-2/order-25 value program with center-frozen UKF cores | Correct for the frozen finite target; not exact likelihood |
| Score | Diagnostic derivative of the same finite value program | TensorFlow derivative through the batch-native fixed finite program | Same-program derivative passed FD; capacity convergence not claimed |
| Posterior | Uniform physical prior on gamma,beta in `(0.1,0.9)` plus full source-chart Jacobian | NeuTra/HMC target with repository signature `deccdda78028706d0987322d30b9798f0f4d8b518c6773451338e83bf14d1cab` | Target identity admitted |
| HMC | Fixed-identity HMC in frozen NeuTra coordinates | Shared sequential controller, four chains, L=25, epsilon as above | Convergence and declared health gates passed |

## Value And Score Evidence

Value artifacts:

- `docs/plans/artifacts/bayesfilter-svx-zc-value-validation-neutra-hmc-20260802/value-attempt01/result.json`
- `docs/plans/artifacts/bayesfilter-svx-zc-value-validation-neutra-hmc-20260802/frozen-initializer-value-attempt01/result.json`

The first run completed 35/35 cells and returned `SELF_CONVERGED_VALUE`. The
frozen-initializer run completed all 16 cells and returned
`SELF_CONVERGED_FIXED_INITIALIZER_VALUE`.

The separate score diagnostic preserved component signs `[-,+]` at the center
and all four validation points. Center-relative cosine similarities were
`1.0`, `0.999416`, `0.999341`, `0.987775`, and `0.978692`. These are
explanatory diagnostics only; they did not veto value promotion.

## NeuTra Training

The four-recipe screen completed in 443.2 seconds. All recipes passed hard
training/status/parity gates. Because the screen used one training seed, no
statistical ranking is supported. The protocol selected the smallest viable
architecture, `svx_zc_narrow_lr1e3`, as a deterministic representative.

The final training then completed 5,000 GPU/XLA updates at batch 128. All 1,024
held-out rows were valid, with zero floor events. Frozen/trainable parity gaps
were zero for the transport and pullback, zero for logdet, and
`5.55e-17` for the logdet score. Final held-out reverse-KL values remain
descriptive only.

Frozen transport:

- Path: `docs/plans/artifacts/bayesfilter-svx-zc-value-validation-neutra-hmc-20260802/neutra-hmc-attempt01/SVX-ZC/final/segments/steps-004001-005000/frozen_transport.json`
- SHA-256: `c816de3d7101444bdeead2e9d43b0ca49de8d426ebd650c0efcf73068d9decff`

## HMC Tuning

The first legacy point-estimate tuner exhausted five repairs without any hard
target or numerical veto. Its six 32-draw mean acceptance probabilities were
`0.946716`, `0.941609`, `0.448993`, `0.831063`, `0.608396`, and `0.629035`.
Requiring one small-sample point estimate to lie exactly inside `[0.65,0.75]`
was wrong relative to the active statistical acceptance policy.

The replacement generic broad-grid route independently tuned epsilon for
`L=(3,5,9,13,18,25)`, used three replications over four chains, classified
replication means with the frozen 90% t interval, and added nonrecursive
same-epsilon neighbor coverage. Exactly one pair survived:

| Role | L | Epsilon | Grand mean acceptance | 90% working interval | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Independently tuned primary | 25 | 0.8434292653 | 0.787289 | [0.739104, 0.835475] | Provisionally viable |
| Same-epsilon coverage | 24 | 0.8434292653 | 0.788204 | [0.776924, 0.799483] | Needs higher epsilon; parent unaffected |

The broad-grid evidence is discarded tuning evidence. Survival nominated the
only kernel for sequential testing; it did not establish convergence.

## Sequential HMC Results

Terminal artifact:

- Path: `docs/plans/artifacts/bayesfilter-svx-zc-value-validation-neutra-hmc-20260802/sequential-hmc-attempt01/SVX-ZC/result.json`
- SHA-256: `00d1e47bc328bc1b8c802c29ca1acf9ebbdfabdcdaa00e27d3cc078e7fe9e9fe`
- Policy: `bayesfilter_neutra_sequential_hmc_v1`
- GPU/XLA: RTX 4080 SUPER, TensorFlow 2.19.1, XLA enabled, memory growth verified
- Warm-up: 2,015 transitions per chain, excluded from posterior estimates
- Retained: 2,080 draws per chain, four chains, 8,320 pooled physical draws
- Hard vetoes: none
- Native divergence telemetry: not exposed by TFP HamiltonianMonteCarlo

The manifest records base commit `fb9a0679adb7c731ff2ac42551f39bdcc15222a1`.
The repository was a concurrent multi-agent dirty worktree, and the run harness
did not preserve an exact launch-time `git status` snapshot. This is a
provenance limitation. The target, frozen transport, broad-grid result, and
terminal result are independently hash-bound, but the artifact must not be
represented as a clean-commit reproduction until this lane is committed and a
fresh reproduction is run from that commit.

Warm-up passed on the latest 1,000-transition window with maximum modern R-hat
`1.01319`, below the `1.05` readiness threshold.

| Physical parameter | Posterior mean | Posterior SD | Median | 95% interval | R-hat | Bulk ESS | Tail ESS | Truth-tail p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gamma | 0.44419 | 0.21712 | 0.42497 | [0.11516, 0.85798] | 1.00878 | 8215.36 | 2343.80 | 0.54284 |
| beta | 0.52182 | 0.16318 | 0.50331 | [0.24816, 0.85519] | 1.00163 | 7530.28 | 2817.79 | 0.49910 |

The run artifact used source-chart labels for these two physical transformed
diagnostics. The numerical values, R-hat/ESS computations, and truth comparison
were on physical gamma and beta; the active registry labels are corrected.

Acceptance was descriptive only. The final 65-draw retained chunk had overall
acceptance `0.78077`, with per-chain rates `0.84615`, `0.80000`, `0.66154`, and
`0.81538`. All chains moved, all sampled states and target values were finite,
and all target-status rows were valid. Finite log-acceptance values below
`-1000` occurred and are explanatory alerts, not energy-error or divergence
vetoes under the active policy.

The shared status schema requires a field named `min_innovation_eigenvalue`.
SVX-ZC has no Kalman innovation covariance: in this adapter that compatibility
alias contains the minimum TT normalizer, also emitted truthfully as
`minimum_normalizer`. It was used only as a finite/positive health diagnostic;
no innovation-eigenvalue claim is made.

## Decision Tables

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Mark frozen SVX-ZC value/NeuTra/HMC campaign complete | Value stability, valid training, unique viable kernel, sequential convergence, and truth-tail all passed | No declared hard veto | One dataset seed, approximate filter target, no exact posterior comparator | Move to the next model; do not rerun SVX-ZC by default | Exact filtering, exact score, broad validity, superiority, default readiness |

| Inference status | Result |
| --- | --- |
| Hard veto screen | Passed: finite states/targets/log acceptance, valid status, movement, R-hat, ESS; native divergence unavailable |
| Statistically supported ranking | None; the broad grid yielded only one viable pair, not a ranked winner |
| Descriptive-only differences | Recipe losses, runtime, acceptance rates, posterior means/intervals, finite extreme log-accept alerts |
| Default readiness | Not established; this is one frozen T=10 target scope |
| Next evidence needed | New target-specific plan and evidence for another model or scope; broader SVX-ZC claims would require new data/scope and comparator design |

## Post-Run Red Team

The strongest alternative explanation is that the approximate finite TT target
is internally stable and easy to sample but differs materially from the exact
observed-data posterior. The campaign does not contain an exact-likelihood
posterior comparator, so it cannot rule that out. A new same-data exact or
higher-authority comparator could overturn posterior-validity interpretations,
but it would not overturn the checked finite-target value, training, or HMC
mechanics. The weakest evidence is scientific posterior accuracy beyond the
frozen approximate target; the strongest evidence is engineering/numerical
validity and convergence for that exact frozen program.
