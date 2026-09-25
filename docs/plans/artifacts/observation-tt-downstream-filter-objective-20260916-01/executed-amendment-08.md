# Amendment A08: downstream Zhao-Cui filtering objective

Date: 2026-09-16  
Status: owner-authorized execution amendment  
Parent: [observation-aware TT master program](observation-aware-tt-repair-complete-program-20260913.md)  
Checkpoint: [active checkpoint](observation-aware-tt-active-checkpoint.md)

## Reason for amendment

A07 correctly found six validation-selected TT audit losses against the exact
SGQF joint (one d=1 case and five d=4 cases). That comparison answers a
representation question: whether the selected finite TT reproduces the
one-step SGQF joint on an untouched panel. It is a useful repair diagnostic,
but it is not the endpoint of the Zhao-Cui filtering study. The old master
worded that diagnostic as a continuation veto and therefore stopped before
measuring the quantity for which the TT is needed.

This amendment repairs the objective while preserving A07 as historical
evidence. SGQF fit loss remains a promotion/repair diagnostic for the fitting
route. It is not a continuation veto for measuring the recursive filter. The
candidate must now be evaluated by its complete downstream filtering update
on the same observations, with independent reference and simple conditional
heuristics.

## Research question

Within the frozen A06/A07 scope, does the validation-selected,
SGQF-initialized TT produce a finite and useful Zhao-Cui filtering recursion?
The claim-bearing object is the sequence of retained filtering distributions
and their normalizers, not the TT coefficient error alone. At each time step
we evaluate

\[
  \widehat p_t^{\mathrm{TT}}(x),\qquad
  \widehat Z_t^{\mathrm{TT}},\qquad
  \widehat m_t^{\mathrm{TT}},\qquad
  \operatorname{ESS}_t,
\]

against an independent reference. The SGQF joint is retained as a one-step
Gaussian-joint comparator and guide. It is not assumed to be the exact
non-Gaussian filtering law.

## Frozen scope and comparator ladder

The rerun uses the existing independent-sequence fixture and does not retune
on the filtering observations: dimensions d=1 and d=4, horizon T=20, 12
sequences per dimension, 512 particles, four stateless particle seeds per
method/sequence, degree 3, rank 3, four sweeps, proximal iterations 128,
defensive fraction 1e-5, and the existing validation-selected L1 grid. The
GPU TensorFlow/XLA route and memory-growth policy remain unchanged.

The fixed comparator ladder is:

| Method | Role |
| --- | --- |
| transition | cheap dynamic proposal heuristic |
| stationary_prior | no-observation prior heuristic |
| sgqf_gaussian | Gaussian SGQF marginal proposal |
| sgqf_joint | full SGQF joint conditional comparator |
| tt_predictive | fitted TT predictive route |
| tt_guided | fitted TT guided route |
| tt_pair_block | paired-block TT route |
| tt_sgqf_safeguard | validation-selected SGQF-initialized candidate |

The first four methods are the constructed heuristic/adversary set. They are
evaluated conditionally in the predeclared observation regimes `near_zero`,
`ordinary`, `large`, and `all`; this is a sanity check and promotion veto,
never a tuning target.

## Evidence contract

Primary downstream evidence is the stationary-variance-normalized particle
mean MSE by dimension and observation regime, paired by sequence and particle
replicate. Supporting downstream fields are log-evidence bias, per-step ESS
and ESS/N, maximum normalized particle weight, resampling rate, unique
ancestor count, and minimum/maximum log correction. The result must preserve
per-step values and aggregate summaries so weight collapse cannot be hidden by
an average over time.

Hard validity/continuation vetoes are a crash, non-finite state/covariance/
normalizer/weight, invalid reference, CDF bracket failure, missing particle
ESS diagnostics, or exhaustion of the authorized budget. The d=4 sequence-8
signed-SGQF guide failure remains an explicit coverage exclusion; d=4
population claims are forbidden unless the reported denominator is the 11
complete sequences. A finite candidate with an observed loss to a heuristic
or to SGQF remains a promotion veto or repair trigger, but it does not stop
this downstream measurement.

Four replicates and one campaign are descriptive evidence. They can establish
finite execution and expose weight collapse, but they cannot establish a
population ranking or superiority. The paired bootstrap already used by A06
is retained for uncertainty intervals; absent an interval supporting a
ranking, the result language is “viable under the screen” and “descriptively
different.”

The result must not claim SGQF equality, posterior correctness, HMC readiness,
default readiness, scalable production performance, or Zhao-Cui source
faithfulness. Those are separate evidence questions.

## Default and assumption audit

The frozen particle count, seeds, target fixture, and method ladder come from
A06 and are retained for comparability. Their failure mode is Monte Carlo
noise and insufficient tail coverage; paired replicates and reference checks
are the earliest diagnostics. The SGQF-initialized selection comes from A07
validation panels. Its failure mode is selection/generalization loss on the
filtering sequences; untouched sequence filtering is the earliest diagnostic.
The 1e-5 defensive fraction is the previously tested candidate safeguard,
not a claim that it is optimal; non-finite correction, mass, and ESS checks are
the safety diagnostics. The d=4 guide exclusion is inherited from the checked
signed-covariance failure and is recorded rather than imputed.

## Skeptical audit and change control

The pre-run skeptical audit passed after one material correction: the old
SGQF-versus-TT audit loss was a proxy/representation diagnostic that had been
silently promoted to a continuation veto. The amended primary criterion now
measures the downstream filtering object and includes the independent
reference, conditional heuristic adversaries, validity vetoes, uncertainty
limits, and a versioned artifact. No data, method family, hardware class,
privacy boundary, or total campaign authorization changes. The owner’s
request to repair the master and rerun the Zhao-Cui comparison authorizes this
continuation under the existing 24-hour ledger.

## Execution and artifacts

The runner is `docs/benchmarks/run_observation_tt_independent_filtering.py`.
The exact wrapper is
`docs/plans/artifacts/observation-tt-downstream-filter-objective-20260916-01/run_campaign.sh`.
Smoke and full attempts use separate directories under
`docs/benchmarks/artifacts/observation_tt_downstream_filter_objective_20260916/`;
the command logs, manifest, JSON result, and terminal result note are kept
under the same versioned plan artifact root. The run is bounded by the
remaining H11 budget recorded in
`artifacts/observation-tt-continuation-24h-20260915-01/budget.json`.

