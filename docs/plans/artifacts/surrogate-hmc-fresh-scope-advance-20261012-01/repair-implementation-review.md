# Shared surrogate-HMC implementation review, 2026-10-13

The checked engineering repairs are retained. They do not yet complete the
five-model posterior experiment or establish that this force is economical.
The active research question is testing surrogate HMC: the finite filter
supplies a proposal force and a separately specified endpoint supplies the
Metropolis energy. Improving the filter alone is not the final outcome.

## Call-chain findings

| Claim-bearing consumer | Checked path | Finding and executable evidence |
|---|---|---|
| Four-chain force | surrogate_hmc_five_model.bindings -> make_batched_score_filter -> shared proposal maps, Gaussian contractions and transport | Stable XLA signatures; leading B2 particle algebra; all-P Jet tangent axis in one observation tf.while_loop. Initial/static row and parameter callbacks are traced Python loops, not Python execution per observation. Full-horizon all-five parity and consumer checks are saved. |
| Proposal covariance | build_maps in covariance_proposal_score_tf | Compute distinct AQA' and checked Cholesky before ancestor broadcasting. Prior/guide extent1; flow extent1 orN according to the callback's constant-Jacobian contract. No model-name special case. Compact parity checks values, scores, particles and tangents exactly for all five. |
| Reused HMC force | sample_neural_force_hmc and public TFP kernel -> _neural_force_hmc_transition -> _neural_force_proposal | Cache accompanies the retained state, selecting the proposal force on acceptance and old force on rejection. Tests cover acceptance, rejection, fallback, divergence, changed field/bootstrap and public TFP consumer. Full LG replay preserves states/energies/decisions. |
| Covariance validity | scalar/reference and batched score call covariance_restoration_tolerance | One shared guard, N-dependent FP64 accumulation allowance, finite covariance required; FP32 remains1%. Exact original failing four-chain range replay passes with identical inputs and unchanged healthy forces. Wiring test observes both consumers. |
| Tuning consumer | fresh scope tuning -> shared maps/density -> exact-scope selection loader -> bindings | Fresh disjoint calibration/validation, same-program FP64 precision veto and frozen controls. Hash/scope mismatches fail. CPU call-chain tests and actual current-source GPU tuning/mechanics cover the executing path. |
| Master continuation | performance-repair advance -> bounded tuning -> conditional mechanics | Fresh LG/KSC/SIR/range advances completed with preserved phase results. Predator-prey reached its time cap after two valid saved candidates, with no final selection or dependent mechanics. All 128 focused CPU master tests pass, including separate enforcement and reporting of attempt, campaign-time and phase-time bounds. |

For parameter coordinate j, the compact covariance derivative still contains
(d_j A) Q A' + A (d_j Q) A' + A Q (d_j A)'.
Broadcasting identical covariances after this calculation leaves the
mathematical quantity unchanged. Exact full-horizon parity checks the
implementation on the declared fixtures; it is not a theorem for all inputs.

For S transitions and L leapfrog steps, the old sampler evaluates a position
force S(L+1) times. Retaining the exact force at the current state changes this
to1+SL. The invariant is cache=F(current position), with the same validity and
fallback marker. Accept/reject row selection preserves it. A new field, chart
or chain bootstrap must recompute the cache. The retained-state tests cover
those boundaries. This removes repeated evaluations without changing the
endpoint potential, acceptance law or random stream.

The FP64 covariance allowance is a conservative accumulation allowance based
on N and dimension, not a complete backward-error theorem for centering,
factorization and matrix products. It changes only validity classification.
The material-error and nonfinite rejection tests still pass. It must not be
described as proving all covariance restoration numerics safe.

## Remaining cost and correctness limits

The single recursion removes repeated primal work across parameter directions.
It does not eliminate the all-pairs marginal-mixture and transport mathematics.
At N4032, those kernels still perform quadratic particle work over exact K2016
tiles. The first-observation diagnostic assigns much of LG/KSC/predator cost to
the reset and SIR cost to both proposal construction and densities. That is a
local diagnostic, not a whole-horizon profiler decomposition. No Python time
loop, pfor fallback, or eager scientific step is introduced by these repairs.

Existing precision checks compare the candidate and FP64 versions of the same
finite program. That checks arithmetic, not agreement with a true likelihood
or score. Exact Kalman diagnostics for LG and refined grid diagnostics for KSC
must remain separate. There is no independent score oracle in the current
SIR/predator/range tuning report. Short acceptance cannot fill that gap.

The optional covariance-proposal path is not the repository's canonical
Contract-E LEDH route. Some selected controls disable the correction. This
review therefore issues no canonical LEDH conformance or admission claim.

FP32 remains the owner default. The current SIR/range scopes failed uniform
FP32 score checks and use explicitly labeled optional FP64 recursion with FP32
pair kernels. The 1% allowance uses max(1,abs(reference component)); close to
zero it is a mixed absolute/relative scale, not a literal relative percentage.

The endpoint defines the actual fixed target. Kalman is exact for LG; KSC grid,
fixed bootstrap for SIR/predator and fixed trajectory importance for range are
approximations. Fixed bootstrap resampling can make the approximate target
nonsmooth. The benchmark normal prior and chart are documented mechanics
choices; posterior-region accuracy and substantive-prior adequacy remain open.

## Full inference route audit

The public registry and docs/reference/hmc-tuning-interface.md were inspected.
The position-only force binding has conditional mechanics authority, not
ordinary exact-score artifact authority. The typed local
VerifiedTensorFlowHMCConfig dispatch is preserved explicitly in
hmc_tuning_dispatch.py and leads to its sequential controller. The five-model
consumer labels that authority and records posterior agreement as not checked.

A tiny two-step pilot is not this controller and is not HMC tuning. Its L1,
epsilon0.002, acceptance and runtime are mechanics diagnostics. The source
contains an inference entry point with adaptive warmup and cumulative retained
sampling, but it has not been executed for these current five-model scopes.
This review checks its dispatch wiring in source, not numerical behavior of
that unexecuted full route. Full posterior acceptance additionally requires the
declared convergence, health and independent-reference evidence.

## Review disposition

The earlier CPU checks report 56 passes in the guard suite and 114 passes in
the later harness/consumer suite; these overlap and are not 170 unique tests.
Logs are precision-guard-cpu.log and precision-guard-harness-cpu.log.
Paired GPU evidence is in compact-covariance-result.md, force-reuse-result.md,
precision-guard-result.md and their linked JSON/archives. Fresh results are
assembled in repair-evidence-summary.json with exact source snapshots.

A reporting defect at the internal attempt cap is repaired: campaign_remaining
now reports actual remaining seconds independently of attempt slots. Both
entry points still enforce both limits, and phase timeouts are distinguished
from scientific candidate rejection. The final focused CPU suite passed all
128 tests; budget-reporting-result.md records coverage and the exact test log.
This repair changes orchestration and reporting, not the numerical method.

The final ledger charges 35,988.991 of 36,000 authorized worker-wall seconds,
with no active reservation. The remaining 11.009 seconds cannot fund a valid
next GPU phase. Completed predator candidate files are preserved, but the
unfinished third candidate prevents final tuning selection. The terminal
next-phase-plan.json records the remaining work as part of this same master.

No stochastic speed ranking is statistically supported. All observed short-run
acceptance, energy and runtime differences are descriptive. The practical
baseline/heuristic comparisons remain unexecuted, so the method cannot be
promoted on utility or efficiency. The strongest alternative explanation of
healthy mechanics is simply the very short, small-step run; longer tuned chains
could still fail or prove unaffordable.
