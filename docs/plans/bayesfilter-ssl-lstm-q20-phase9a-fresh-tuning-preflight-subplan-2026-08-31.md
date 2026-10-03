# Phase 9A q=20 fresh-map tuning and replica-exchange preflight

Date: 2026-08-31  
Status: `CLOSED_PHASE9A_CONTINUATION_VETO_CHART1_BETA0`

Parent program:
`docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-implementation-plan-2026-08-28.md`

Phase 8 freeze:
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c5-freeze-subplan-2026-08-31.md`

## Research question and boundary

This preflight asks whether the frozen Phase 8 protocol can be bound to the
repository's claim-bearing HMC mechanics at q=20.  It constructs fresh K=2
compact-high charts, follows the frozen L3=(0,.5,1) pure-continuation lineage,
and obtains one independent fixed-transport tuning receipt for every
`(beta, chart)` pair.  It then binds the receipts to the proper bridge and
executes one short augmented replica-exchange chunk through the shared
sequential-transition interface.

The preflight is an implementation and numerical-boundary experiment.  It is
not a posterior run and cannot establish whitening, mode discovery, mixing,
convergence, HMC readiness, posterior correctness, sampler superiority, or
high-dimensional scaling.  Calibration checkpoints are not read or used as
warm starts.  All charts are rebuilt from fresh stateless seeds in this phase.

## Evidence contract

| Item | Predeclared rule and role |
|---|---|
| Scientific question | Can fresh Phase 9 q=20 charts and per-level tuner handoffs be composed into the exact proper-bridge transition without scope, identity, status, or resource failures? |
| Exact target | q=20 SSL-LSTM Gaussian-prior/likelihood bridge, target signature `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278`, strict `tensorflow_eigh_strict` backend |
| Frozen candidate | K=2, `(16,16)` tanh, two stages, learning rate `1e-3`, L3 `(0,.5,1)`, pure continuation, fixed uniform gamma `(0.5,0.5)` |
| Comparator in this phase | No scientific sampler comparator.  The identity/physical route remains a later Phase 9B comparator; this phase tests only candidate binding and shared mechanics. |
| Primary pass criterion | Both fresh charts pass preflight/reliability, all six scope-specific tuners emit a durable passed handoff, and one four-chain replica-exchange chunk passes the shared transition health checks. |
| Hard vetoes | Target/backend mismatch; invalid or nonfinite target status; stale or mixed scope/hash; failed chart inverse/logdet/score reliability; missing tuner handoff; nonfinite state; transition signature mismatch; failed all-chain movement; GPU memory-growth/XLA policy failure; output collision; material wall or allocator cap. |
| Explanatory diagnostics | Reverse-KL loss, pullback density/score residuals, acceptance, swap count/rate, target-call counts, compile/reuse counts, and timing.  These do not promote a chart or sampler. |
| Artifact | Versioned attempt directory with run manifest, fresh-chart checkpoints, six tuning manifests, handoff payloads, transition summary, and a result note. |
| Nonclaims | No whitening/IID-Gaussian claim, no mode-discovery claim, no posterior or convergence claim, no ranking or superiority claim, no default-readiness claim, no scaling claim. |

## Default and assumption audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Promotion status |
|---|---|---|---|---|---|
| GPU 0, memory growth, TF32, XLA | Repository owner directives in `AGENTS.md` and Phase 8 receipts | Matches the active BayesFilter execution target and strict q=20 route | Allocation or compiler failure could masquerade as numerical failure | Verify policy before TensorFlow logical-device creation and record device telemetry | Reviewed repository default |
| K=2 compact-high | C5 metadata-only freeze | Parsimonious frozen Phase 9 hypothesis | A short calibration tie-break may not predict posterior performance | Fresh chart reliability and later downstream gates | Confirmation hypothesis, not default promotion |
| L3 and pure continuation | C5 freeze | Lower-cost ladder and no established branching benefit | Insufficient temperature overlap or lineage collapse | Per-level finite status, swap ratios, identity travel in later phase | Frozen protocol hypothesis |
| Batch size 32 for training | Phase 8 strict calibration | Preserves batch-native training contract while limiting memory | Too-small stochastic batch may give unstable updates | Finite update and gradient telemetry | Phase-specific reviewed value |
| Four HMC chains | Shared sequential policy | Minimum required by modern split/folded R-hat controller | Resource cost or identical starts can hide non-movement | Explicit four-row varied z bank and movement veto | Hard controller requirement |
| Initial tuner budget `(4,)`, 4 tune draws, 8 screen draws, 8 verification draws | New Phase 9A diagnostic hypothesis; chosen only to bound the first localization launch | Cheapest complete handoff exercise before long tuning | Too short for convergence or robust step selection | Label all tuning outputs mechanics-only and require longer Phase 9B retuning | Preflight-only convenience, never confirmation evidence |
| Initial epsilon `0.01`, `L=5`, first cap `0.25` | New bounded preflight hypothesis | Keeps the first localization launch small | A cap hit can falsely look like a bad transformed target | Inspect tuner acceptance and repair trigger | Superseded preflight hypothesis; retained in attempt-02 evidence |
| First repair cap `1.0`, ladder budgets `(4,4,4)` | Derived from attempt-02: acceptance `0.984226` at cap `0.25`, with multiplicative enlargement requested; three rounds permit `0.25 -> 0.5 -> 1.0` | Tests the declared repair direction while preserving target, maps, seeds, and acceptance bands | A short ladder can still fail to identify a robust step; an overly large step can create numerical vetoes | Per-round finite/status, acceptance, energy proxy, movement, and cap telemetry | Superseded preflight hypothesis; retained in attempt-03/04 evidence |
| Final cap repair `2.0`, ladder budgets `(4,4,4)` | Derived from attempt-04 chart-1 beta-0: repaired initial step `1.256879` exceeded cap `1.0` after a finite `0.999032` screen acceptance; rounded bound `2.0` is the final declared enlargement | Allows the observed tuner repair to be evaluated without changing the target, maps, seeds, or acceptance bands | A second cap failure or numerical veto means this preflight cannot bind all scopes under its budget | Per-round cap, finite/status, acceptance, energy proxy, and movement telemetry | Final Phase 9A repair hypothesis, never a confirmation default |
| Fresh seeds | Phase 9 reserved seed domain, with new Phase 9A subdomain | Prevents calibration/confirmation contamination | Seed collision or accidental reuse | Machine-check uniqueness ledger and manifest hashes | Hard provenance requirement |

## Procedure

### A0. Static and policy checks

1. Verify C5 status, target signature, bridge properness receipt, and source
   hashes.  Reject any calibration checkpoint input.
2. Run `py_compile`, `git diff --check`, the focused tempered-transition and
   fixed-transport tests, and a route scan for row-mapping/pfor tokens.
3. Launch only with `TF_FORCE_GPU_ALLOW_GROWTH=true`, one visible GPU, and the
   repository memory-policy helper before TensorFlow device initialization.

### A1. Fresh chart construction

For each of two chart IDs, instantiate a new `(16,16)` two-stage
`WeightedDenseIAFTransport` with a distinct initialization seed.  Use the
fixed pre-optimizer Gaussian bank at beta zero, then construct immutable
preflight receipts at beta `.5` and `1`.  Train with fresh IID standard-normal
batches while preserving the pure-continuation parent chain:

```text
beta=0.0: 2 diagnostic updates
beta=0.5: 2 diagnostic updates
beta=1.0: 2 diagnostic updates
```

The update count is a preflight bound, not a claim-bearing training protocol.
Persist a checkpoint after each level and verify the parent hash, tensor hash,
and replay on a fresh latent bank.  Run the full learned-map reliability
screen on both charts after the beta-one endpoint.  A failed chart is a
candidate/implementation repair trigger, not evidence against the bridge
mathematics.

### A2. Scope-specific mechanics tuning

For each chart and each beta in `(0.0, 0.5, 1.0)`, bind a
`FixedBetaBridgeAdapter` and the corresponding frozen chart.  Invoke the active
`tune_fixed_transport_hmc_kernel` route with a distinct target scope and
output directory.  Use four varied latent initial states, identity z-mass,
XLA, and the preflight-only budgets in the audit table.  Build a
`VerifiedFixedTransportHMCHandoff` only from the durable passed tuning result;
do not stamp or copy a handoff between beta levels or charts.

The preflight records whether each scope passes the mechanics handoff screen.
Its short acceptance and energy diagnostics are descriptive and are not used
to claim convergence.  If a scope fails only because the bounded epsilon
candidate is outside the acceptance band, use the tuner-reported repair
direction in one fresh attempt without changing the target or seed domains.

### A3. One shared-controller transition

Build one verified fixed-transport kernel per `(beta, chart)`, wrap the two
chart kernels at each level in the fixed state-independent uniform chart
mixture, and bind the three levels to
`ProperReplicaExchangeTransitionProgram`.  Start four chains from a finite,
varied augmented state.  Execute one retained chunk through
`run_sequential_exact_transition` using a four-chain-compatible diagnostic
configuration with warmup and retained checks disabled only because this is a
mechanics preflight; the result must still satisfy the controller's finite,
signature, beta-one-stream, final-state, and all-chain-movement checks.

The preflight must report target calls, swap proposals/acceptances, chart
identity labels, compiled tracing count, and memory telemetry.  It must not
write a posterior estimate or consume the reserved Phase 9B confirmation
streams.  The controller schedule uses the smallest legal four-chain,
four-draw warmup and retained chunks.  Its permissive short-run R-hat bound and
finite-only retained diagnostic are recorded as mechanics diagnostics; they do
not assert convergence or authorize a posterior estimate.  Finite state,
signature, beta-one-stream, final-state handoff, and all-chain-movement checks
remain hard preflight checks.

## Budgets and stop rules

The material wall cap is 1,800 seconds and the allocator cap is 4 GiB on the
visible device.  The attempt budget is one initial localization launch, one
harness repair retry, one declared epsilon-cap repair retry, one complete
six-scope launch with that repair, and one final complete launch using the
evidence-derived cap repair above.  Attempt 02 showed a finite, mobile chain
but hit the initial cap (`0.984226` mean acceptance at `0.25`).  Attempt 03
passed its single beta-zero scope with selected epsilon `0.810010`, acceptance
`0.859967`, one trace per reusable HMC graph, and `1402670592` bytes (about
1.31 GiB) peak allocation.
Attempt 04 then passed all three chart-0 scopes but chart 1 beta 0 requested
`1.256879` after a finite `0.999032` screen and exceeded cap `1.0`; this is the
sole basis for the final cap `2.0` repair.  No target, map, seed domain,
acceptance band, or scientific claim changes.  Every retry uses a fresh attempt
directory and preserves failed output.  Any scope failure under cap `2.0`, any
memory/target/transition veto, or the material budget cap is a Phase 9A
continuation veto and must be recorded rather than silently widened.

Stop immediately for target/math mismatch, corrupted or missing source/seed
provenance, invalid bridge properness, nonfinite target values or scores,
failed learned-map inverse/logdet reliability, absent handoff, memory-growth
noncompliance, or a second failure of the same localized repair.  A short
acceptance-band failure is a repair trigger; it is not a continuation veto
until the declared retry is exhausted.  A complete A3 preflight pass opens a
new, separately audited Phase 9B tuning/confirmation subplan; it does not
authorize confirmation by itself.

## Fresh seed and output identities

The following seeds are reserved exclusively for this Phase 9A preflight:

| Role | Seed root |
|---|---|
| Chart initialization | `(20260831, 73001)`, `(20260831, 73002)` |
| Chart preflight | `(20260831, 73101)`, `(20260831, 73102)` |
| Chart training | `(20260831, 73201)`, `(20260831, 73202)` |
| Tuning scope bases | `(20260831, 73301)` through `(20260831, 73306)` |
| Augmented transition | `(20260831, 73401)` |

The historical first localization launch used:

`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/phase9a-fresh-tuning-preflight/attempt-01/`

The first complete six-scope launch used:

`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/phase9a-fresh-tuning-preflight/attempt-04/`

The final cap-repair launch must use:

`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/phase9a-fresh-tuning-preflight/attempt-05/`

No existing directory may be overwritten.

## Skeptical pre-execution audit

| Risk | Finding and disposition |
|---|---|
| Calibration checkpoint silently promoted | Closed: A1 rebuilds every chart from fresh seeds and records no calibration checkpoint input. |
| Tuner scope reused across levels | Closed: scope includes bridge beta, chart manifest hash, adapter signature, and distinct output/seed domains. |
| Fixed chart mixture confused with a mixture density | Closed: each chart kernel is invariant for its own fixed target; gamma is state-independent and only selects a kernel. |
| Short tuning treated as convergence | Closed: all A2 receipts are mechanics-only; A3 is one-chunk preflight. |
| Improper beta bridge | Closed by the existing q=20 properness receipt and endpoint status check. |
| Row-mapped or pfor target path | Closed by static scan and the existing batch-native target contract. |
| Final state disconnected from returned sample | Closed by the shared transition controller's final-state equality check. |
| Hidden resource failure | Closed by predeclared wall/allocator caps and device/memory manifest fields. |
| Phase 9B contamination | Closed by fresh seed roots and an output root disjoint from reserved confirmation roots. |
| Candidate failure mistaken for direction failure | Closed: classify as chart, tuning, numerical, infrastructure, or evidence insufficiency and refresh the next subplan. |

Audit verdict before launch: `PASS_PHASE9A_BOUNDED_PREFLIGHT`.

## Exit and refresh

The result note must include a decision table and an inference-status table with
hard veto evidence, statistically supported ranking (expected `none`),
descriptive-only diagnostics, default readiness, and next evidence required.
It must include a post-run red-team note naming the strongest alternative
explanation and the evidence that would overturn the preflight decision.

After A3, refresh the parent implementation and execution records with the
actual command, commit/dirty state, environment, seeds, timing, resource
telemetry, failure classification, and the exact Phase 9B assumptions.  A
preflight pass opens Phase 9B; a preflight failure preserves the artifact and
opens only the smallest documented repair.

## Execution closeout, 2026-08-31

Attempts 01-05 are preserved under the Phase 9A artifact root.  The runner
repair in attempt 01 fixed a missing import.  Attempt 02 exposed the initial
epsilon cap, and attempt 03 validated the first cap repair on a localized
chart-0/beta-0 scope.  Full attempt 04 passed all three chart-0 scopes but
failed chart-1/beta-0 when its requested repair exceeded cap 1.0.  Full attempt
05 repeated that same scope failure under the final cap 2.0: the tuner reached
acceptance 0.998950 at epsilon 0.628978, 0.939618 at epsilon 1.205189, and then
requested 2.410379, above the cap.  The tuner emitted
`tune_initial_step_size_exceeds_configured_cap`,
`verification_acceptance_outside_pass_band`, and `no_viable_candidate`.

The continuation veto therefore fired.  A1 and the completed portions of A2
passed their finite/replay/reliability checks; all six handoffs were not
obtained, so A3 was not run and Phase 9B did not open.  The terminal
interpretation and decision tables are in
`docs/plans/bayesfilter-ssl-lstm-q20-phase9a-fresh-tuning-preflight-result-2026-08-31.md`.
Any further work requires a separately audited chart-1/beta-0 repair subplan;
the cap must not be widened silently.
