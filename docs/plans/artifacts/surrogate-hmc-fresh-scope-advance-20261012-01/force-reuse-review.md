# Retained-position force reuse: conditional repair

This is a performance repair within the five-model surrogate-HMC master, not a
change of likelihood, proposal field, particle count, or inference criterion.
Execute only after the active five-scope sequence and the same-input consumer
cost diagnostic. Keep the current numerical and HMC sources frozen until then.

The inspected call chain is `pilot -> sample_neural_force_hmc ->
neural_force_hmc_transition -> neural_force_proposal`. The public tuning lane
uses `NeuralForceTransitionKernel.one_step` and the same transition. Both
currently reevaluate the initial position-only force on every transition. A
repair limited to the pilot would leave the public tuning consumer unrepaired.

Let the frozen, sanitized position-only field be F(q), with fallback indicator
b(q). Store C_t=(F(q_t),b(q_t)). The proposal uses C_t for the first half kick and
computes C* at its terminal position q*. The Metropolis decision A uses the same
endpoint potential and the same stateless random seed as before. Update each
chain by q_(t+1)=where(A,q*,q_t) and C_(t+1)=where(A,C*,C_t). Induction gives
C_t=(F(q_t),b(q_t)) for every transition. Every kick, rejection, fallback flag,
and endpoint decision is therefore unchanged in exact arithmetic. A divergent
or endpoint-invalid proposal must retain the original cache when rejected.
Changing the frozen field or coordinate map invalidates the cache; bootstrap
must rebuild it. The cache must never be shared across chains or sampler scopes.

For S transitions with L leapfrog steps, physical force evaluations decrease
from S(L+1) to 1+SL, including cache initialization. The maximum asymptotic saving
is one half at L=1; this cannot by itself turn a 100-hour minimum warmup into a
ten-hour run. That limitation is part of the decision, not a reason to omit a
valid low-cost repair.

Evidence contract: compare cached and uncached execution with identical field,
target, starts, momentum seeds, step size, and mass. Require equality of accepted
states and decision diagnostics on deterministic CPU fixtures covering mixed
accept/reject decisions, multiple L, invalid fields, invalid endpoints, and the
TFP tuning consumer. Require correct physical evaluation accounting. On the
actual GPU field, retain all health checks and report numerical differences and
warmed time separately from compilation. Any changed fallback/rejection rule,
stale-coordinate cache, missing public-consumer wiring, or unexplained trajectory
discrepancy blocks promotion. Timing is descriptive; no posterior convergence or
statistical speed ranking follows.

The default and assumption audit is explicit: field determinism is required by
`FrozenPositionOnlyForce`, not assumed for arbitrary mutable neural networks;
cache lifetime is one frozen coordinate/field scope; unchanged seeds make paired
mechanics comparisons meaningful; L=1 is the existing pilot, not a tuned HMC
choice. A cache is an exact reuse mechanism, not a numerical protection.

Skeptical audit: the mechanism does not address quadratic particle work or XLA
layout costs, and a pilot-only implementation would fail the call-chain rule.
Do not implement before confirming the measured consumer cost and remaining
budget. If worthwhile, bound GPU verification to 3600 additional worker seconds
within the existing 36000-second ledger; CPU regression is separate. Preserve
prior results and write a fresh versioned comparison. A performance failure
leaves the uncached baseline available and does not reject surrogate HMC.

## Preparation amendment

The audited force-count reduction justifies isolated CPU preparation while the
existing GPU sequence runs. Candidate source lives under
/tmp/surrogate-force-cache-candidate-20261012 and is not imported by the campaign.
Repository numerical/HMC source remains frozen until the sequence ends. This
preparation changes no scope and consumes no GPU budget. Promotion still needs
the same-input consumer diagnostic, cached/uncached parity, and downstream
public-consumer regression. Existing single-transition APIs keep their outputs;
shared chain and TFP consumers must both use the same cached implementation.

## Cost decision and installation

The consumer diagnostic reached its1500s cap in the optional native B4 lane.
The completed same-input measurements put grouped B4 force cost at134-149s,
versus0.38-0.40s for the endpoint. B2 output projection and force wrappers
showed similar65-73s costs. This supports repairing the repeated force call.
The incomplete native-B4 result supplies no batch-size promotion evidence.
The cache preserves the existing B2 grouping.19 isolated CPU tests passed;
repository installation also preserves bootstrap fallback diagnostics.
Full public-consumer regressions and the predeclared GPU paired comparison
remain required before calling this repair verified. Baseline source is
neural_force_hmc_before_reuse.py in this directory.
