# Amendment A08: downstream Zhao-Cui filtering objective

Date: 2026-09-16  
Status: executed; measurement interpretation clarified at closeout  
Parent: [observation-aware TT master program](observation-aware-tt-repair-complete-program-20260913.md)  
Checkpoint: [active checkpoint](observation-aware-tt-active-checkpoint.md)

## Reason for amendment

A07 correctly found six validation-selected TT audit losses against the exact
SGQF joint (one d=1 case and five d=4 cases). That comparison answers a
representation question: whether the selected finite TT reproduces the
one-step SGQF joint on an untouched panel. It is a useful repair diagnostic,
but it is not the endpoint of the Zhao-Cui filtering study. The old master
worded that diagnostic as a continuation veto and therefore prevented further
downstream evaluation despite the already promising A06 filter measurements.

This amendment repairs the objective while preserving A07 as historical
evidence. SGQF fit loss remains a promotion/repair diagnostic for the fitting
route. It is not a continuation veto for measuring the recursive filter. The
candidate must now be evaluated by its complete downstream filtering update
on the same observations, with independent reference and simple conditional
heuristics.

## Research question

Within the frozen A06/A07 scope, does the validation-selected,
SGQF-initialized TT provide a useful conditional proposal for the corrected
Zhao-Cui filtering computation? The runner calls the exact-importance-corrected
particle filter. It measures particle filtering means and evidence increments,
not the mean or Gram normalizer of the uncorrected retained TT. At each step it
evaluates

\[
  \widehat \pi_t^{\mathrm{SMC}}=\sum_i w_t^{(i)}\delta_{x_t^{(i)}},\qquad
  \widehat Z_t^{\mathrm{SMC}},\qquad
  \widehat m_t^{\mathrm{SMC}},\qquad
  \operatorname{ESS}_t,
\]

against an independent reference. The SGQF joint is retained as a one-step
Gaussian-joint comparator and guide. It is not assumed to be the exact
non-Gaussian filtering law. The selected policy includes SGQF fallback as well
as generic-start and SGQF-start TT. Report those counts; do not describe that
policy as a TT-only recursion. The pair-TT arm is a separate TT comparator.

## Frozen scope and comparator ladder

The rerun reuses A06's existing independent-sequence fixture and exact seeds;
it is an instrumented reproducibility run, not new independent confirmation.
It does not retune on the filtering outcomes: dimensions d=1 and d=4, horizon T=20, 12
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
are the earliest diagnostics. The SGQF initializer and fixed selection rule
come from A04–A06; each time step uses separate training and validation rows.
Its failure mode is selection/generalization loss. A07 tests fresh panels,
whereas this exact-seed A06 replay checks downstream reproducibility.
The 1e-5 defensive fraction is the previously tested candidate safeguard,
not a claim that it is optimal; non-finite correction, mass, and ESS checks are
the safety diagnostics. ESS has no newly imposed numerical pass threshold;
it measures proposal efficiency and reveals collapse. The d=4 guide exclusion is inherited from the checked
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

## Execution closeout

`attempt-smoke-04` and `attempt-full-01` completed on the RTX 5080 with
TensorFlow/XLA and verified memory growth. Full wall time was 2023.805 seconds.
All 24 references passed. The known d=4 sequence-8 guide failure recurred;
every completed method's stored particle values were finite and no recorded
CDF bracket check failed. MSE tables and bootstrap intervals exactly reproduce
A06. The new report adds actual particle ESS, weight, resampling, ancestor
and correction statistics and checks summaries against the raw records.

The two initial UUID attempts failed at device discovery; numeric selector 0
passed a smoke on the RTX 4080 SUPER. Selector 1 was then verified to select
the RTX 5080, and used for the comparable smoke/full run. These infrastructure
attempts are preserved and are not TT failures.

The pre-execution amendment is preserved at
`artifacts/observation-tt-downstream-filter-objective-20260916-01/executed-amendment-08.md`.
This closeout corrects the distinction between retained-TT and corrected-SMC
outputs and makes seed reuse explicit; it changes no run control or metric.
See [result](observation-aware-tt-downstream-filter-objective-20260916-result.md)
for the numerical comparison, interpretation, limitations and next action.
