# Austria GenUT NeuTra Zero-Force Smoke Result

Date: 2026-08-04

> **Superseded 2026-08-19** by
> `docs/plans/bayesfilter-austria-genut-neutra-root-cause-execution-result-2026-08-18.md`
> for the endpoint-reproducibility and score-route claims: the endpoint
> program analyzed here was replaced by the shared-primal repair, and the
> cross-process value irreproducibility recorded here belongs to the
> pre-repair source. The behavioral score-free proposal observation is
> preserved smoke evidence only and has not been re-evaluated on the repaired
> route.

Plan:
`docs/plans/bayesfilter-austria-genut-neutra-value-surrogate-strategy-2026-08-03.md`

Claim artifact:
`docs/benchmarks/artifacts/genut_austria_neutra_zero_force_smoke_20260804/attempt04/result.json`

Status: `BEHAVIORAL_PROPOSAL_VIABLE_ENDPOINT_REPLAY_BLOCKED`

## Direct answer

**Yes, the score-free proposal idea works behaviorally in this smoke.**  A
Gaussian force \(g(z)=z\) in the transferred SIR-SGQF NeuTra chart constructed
useful Austria GenUT proposals at step size `0.1` and `L=10`.  The GenUT score
was not supplied to any leapfrog kick.  Every proposal was corrected with the
finite GenUT posterior value and both endpoint kinetic energies.

**No, this is not yet a complete usable GenUT HMC result.**  The current
endpoint still calls `finite_value_score`, so it computes and discards the
unstable score and establishes no speed saving.  More importantly, the same
fixed seed and source produced different endpoint values and validity decisions
in separate GPU processes.  A reproducible, tangent-free value-only endpoint
remains the blocker.

## Claimed and computed quantities

| Item | Classification |
|---|---|
| Claimed target | Posterior defined by the frozen `N=1008`, seed-`140000`, current-tuned finite Austria GenUT scalar, the independent `Normal(0,0.5^2)` log-scale prior, and the frozen transport Jacobian |
| Endpoint actually computed | The above scalar value returned by `finite_value_score`; its score was also computed but discarded |
| Proposal force actually computed | \(g(z)=z\), the gradient of the standard-normal NeuTra-coordinate potential |
| Equality of force and GenUT score | Different; no equality or approximation claim |
| Correctness basis checked | Symmetric position-only proposal, full endpoint potential-plus-kinetic correction, reversibility, and finite-proposal energy reconstruction |
| Remaining mismatch | The endpoint is not value-only or batch-native, and separate-process replay is not deterministic |

## Confirmatory cell

The confirmatory artifact used four chains, 32 transitions per chain, step
size `0.1`, `L=10`, GPU/XLA, FP32/TF32 GenUT arithmetic, and FP64 HMC
mechanics.

| Diagnostic | Result | Predeclared screen | Status |
|---|---:|---:|---|
| Pooled acceptance | `0.6171875` | `>=0.5` | pass |
| Per-chain acceptance | `0.625, 0.65625, 0.625, 0.5625` | every chain `>=0.25` | pass |
| Nonzero accepted moves | `20, 21, 20, 18` | every chain `>=4` | pass |
| Normalized accepted ESJD | `0.5389069` | `>=0.01` | pass |
| Finite occupied potentials | all | all | pass |
| Finite proposed endpoints | `127/128` | nonfinite proposal may reject | explanatory |
| Energy reconstruction | maximum error `0.0` on `127` finite proposals | exact | pass |
| Proposal reversibility | position `2.22e-16`, momentum `1.11e-16` | each `<=1e-12` | pass |
| Same-process initial replay | eight repeats, maximum difference `0.0` | exact | pass |
| Endpoint calls | one new endpoint batch per transition | exact | pass |
| Force calls | `L+1=11` per transition | exact | pass |

The one nonfinite proposed endpoint was assigned potential `+infinity` and
rejected.  It never became an occupied state.  Energy reconstruction is defined
on finite proposals; the attempt-3 harness mistakenly evaluated
`abs(infinity-infinity)` and reported `NaN`.  Attempt 4 repaired only that
diagnostic and repeated the nominated `0.1` cell.

## Replay blocker

Separate GPU processes did not reproduce the same endpoint program despite
identical fixed root seed, source, controls, transport, XLA, FP32, and TF32
settings.

| Replay observation | Attempt 3 | Attempt 4 |
|---|---:|---:|
| Center transformed potential | `696.1349729054` | `696.1866086476` |
| Full-scale four-state initialization | all finite | one invalid |
| Largest finite offset scale | `1.0` | `0.5` |
| Step-`0.1` pooled acceptance | `0.625` | `0.6171875` |
| Step-`0.1` normalized ESJD | `0.4915` | `0.5389` |

The center potential difference is about `0.05164`.  This is not Monte Carlo
uncertainty: both processes use the same fixed particle noise.  It is numerical
or execution-order non-reproducibility of the finite GPU program.  Within each
process, the eight initial repeats were bitwise equal.

Therefore the behavioral conclusion is robust enough to nominate the proposal
mechanism, but a multi-process chain ensemble cannot yet be claimed to target
one common finite posterior.  This replay problem also explains why the
initialization validity differed between attempts.

## Attempt ledger

| Attempt | Outcome | Classification | Repair |
|---|---|---|---|
| 1 | Failed before target evaluation | GPU memory-policy/import-order infrastructure defect | Configure memory growth before TensorFlow-bearing repository imports |
| 2 | Stopped before HMC | At least one dispersed initial endpoint invalid | Fixed offset-scale diagnostic ladder; target and HMC gates unchanged |
| 3 | Completed all three cells | `0.1` met every behavioral screen, but `inf-inf` made the reporting gate `NaN` | Restrict exact energy reconstruction to finite proposed endpoints |
| 4 | Confirmed `0.1` cell | Behavioral viability passed; cross-process target replay differed | Stop campaign and implement Stage 0 value-only endpoint |

No retry changed the GenUT target, Gaussian force, leapfrog count, nominated
step size, acceptance gates, movement gates, or HMC seed.

## Decision

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain Gaussian-force NeuTra as an Austria GenUT strategy | Passed behavioral viability at `0.1/10` | Kernel invariants passed; cross-process endpoint replay failed | Whether a tangent-free value endpoint is reproducible and faster | Implement Stage 0 true value-only endpoint, require batch-native evaluation and replay, then rerun the fuller canary | No convergence, posterior agreement, speedup, target-specific chart validity, score correctness, or default readiness |

## Inference status

| Item | Status |
|---|---|
| Hard veto screen | Proposal/kernel screen passed; common-target cross-process replay veto failed |
| Viable candidate | Transferred-chart Gaussian force at `0.1/10` is behaviorally viable |
| Statistically supported ranking | None; one fixed-noise short smoke |
| Descriptive-only differences | Acceptance, ESJD, runtime, finite-proposal fraction, and step-size differences |
| Default readiness | Not established |
| Next evidence needed | Genuine tangent-free, batch-native GenUT value endpoint with same-process and cross-process replay, followed by longer matched validation |

## Negative-result classification

- **Implementation failure repaired:** import order and `infinity-infinity`
  reporting.
- **Numerical validity failure remains:** separate-process endpoint values and
  program-valid branches differ.
- **Evidence for the candidate:** the Gaussian force produces accepted,
  nontrivial movement without using the returned score.
- **Evidence not obtained:** computational saving, posterior convergence,
  target agreement, or score-free NeuTra training.

## Post-run red team

The strongest alternative explanation for the good movement is that the short
chain remains near the initialization and has not explored difficult GenUT
regions.  The strongest reason not to run longer now is more fundamental: two
processes do not demonstrably share the same finite endpoint target.  A longer
chain cannot repair that.  The result would be overturned if a reproducible
value-only endpoint caused acceptance or movement to collapse under the same
chart and mechanics.
