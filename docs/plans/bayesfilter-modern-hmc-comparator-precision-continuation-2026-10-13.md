# Continue the classical comparator and uncertainty repair

13 October 2026. Continue the existing master and compute authorization after
the full SGQF MAMS run delivered all four verified members. The terminal result
is `bayesfilter-modern-hmc-priced-mams-terminal-2026-10-13.md`. No NUTS work is
included. No new numerical defaults or comparative ranking are authorized by
this phase.

## Questions, evidence and next decision

The GPU question is whether the already frozen eight-root fixed-HMC cohort
delivers under the same target, starts, qualification and posterior rules as
MAMS. It is the next missing classical baseline, not a new search selected from
posterior outcomes. Use `priced-comparison-01/designs-02/fixed_hmc.json` and the
unchanged `full-model-source-e68cdf1c4` numerical snapshot. Preserve all eight
roots and eight possible repairs, with the existing fixed-count repair law.
Keep the original 32/64/128-trial measurement and independent verification,
all verified members, 18 quantities, warm-up and retained count limits,
precision estimator and finite-target reference unchanged. This is still one
seed on one finite SGQF target. Even complete delivery cannot rank methods or
establish general posterior coverage. Failed members remain in the denominator.

The CPU question is whether an explicit variance-estimator safeguard addresses
the observed antithetic downward correction and unavailable lugsail estimates
without silently changing the scientific target. Evaluate the optional local
extension `lugsail_max_batch_means`: for each chain and quantity compute the
existing plain batch estimate B and lugsail estimate L, then use max(B,L).
Pool the independent-chain estimates using the existing formula. Finite
negative L may be superseded only by valid positive B; report raw L, B and
every activation. Nonfinite estimates remain unavailable. This is a local
extension, not a formula attributed to Vats--Flegal and not a universal
conservative variance estimator. It cannot be below the existing plain or
lugsail estimates on common valid inputs. If both estimators consistently
estimate the same long-run variance, their maximum is consistent by continuity;
that assumption does not establish finite-sample or stopped-interval coverage.

Implement it as an explicit optional precision policy; retain the existing
defaults and frozen GPU rules. Test the real TensorFlow/XLA arithmetic with
independent batch oracles, positive and negative dependence, batch remainders,
per-chain cancellation, nonfinite values and policy serialization. Use fresh
stationary Gaussian and Student-t(5) paths with the previous three correlations,
four chains and all declared quantities. Compare the same paths and endpoints
under the original estimator and safeguard, at fixed n=10000 and first passing
looks 1000,...,10000. Then reassess saved MAMS draws under both policies, without
changing historical outcomes. The original estimator is the exact comparator;
plain batch means and autocorrelation remain explanatory comparators.

| Diagnostic | Role | Action |
| --- | --- | --- |
| Changed numerical source/data/design, corrupted trace, noncompliant GPU setup | Continuation veto for affected work | Preserve evidence, repair locally, recheck before resuming |
| Public acceptance/health verification | Tuning membership | Retain every verified member, independent of posterior diagnostics |
| R-hat, ESS, MCSE and reference agreement | Posterior criteria only | Preserve pass/failure for every member; no tuning repair from these |
| Envelope inequality, finite-input validity, CPU numerical oracle parity | Engineering pass/fail | Fix implementation before calibration |
| Fresh fixed/stopped interval coverage and availability | Calibration diagnostics | Report uncertainty; no default promotion from point estimates |
| Runtime forecasts, stopping fractions and saved-run sensitivity | Explanatory only | No ranking or retroactive relabeling |
| Actual exhausted allocation | Resource stop | Preserve all unfinished identities and charge actual work |

## Budget, outputs and launch

The incoming phase balance was 133819.624951 seconds. Keep the complete
3600-second local-check reserve held (including the prior known 1981.566793
seconds). Charge terminal pricing 6851.941203 and terminal MAMS 40403.448742
seconds. This leaves **82964.235006 seconds** after the local reserve. Reserve
an additional **3600 CPU-worker seconds** for the new calibration and saved-draw
sensitivity; allocate at most **79364.235006 seconds (22.05 hours)** to the full
fixed-HMC cohort. These are spending ceilings derived from the prior ledger,
not completion forecasts. Charge failed attempts and supervisor overhead;
release unused allocations only on terminal settlement. No duplicate worker
or additional method may spend the same reservation.

The same-root no-repair timing scenarios from the existing price calls give
about 17.94 hours at minimum posterior counts and 40.69 hours at 2000+10000
counts for fixed HMC. They repeat cold-call costs and exclude some orchestration
overhead. Both are hypotheses, not measured bounds or continuation vetoes.
MAMS actually took 11.22 hours; that does not predict fixed-HMC convergence.
Qualification can need additional looks or repairs and exceed this allocation.

Use fresh `priced-comparison-01/comparator-01/` for the controller and its
fixed-HMC child, `precision-repair-01/` for CPU diagnostics, and numbered retry
directories. Extend the existing continuation script to accept a checked
terminal child settlement and explicit method. Its default MAMS behavior stays
compatible. Save the exact command, environment, hashes and device probe before
launch; numerical work remains in the immutable source. Trusted GPU access,
memory growth, TF32/XLA, and the existing idle/shared-device recovery policy
remain required. Deliberately hide GPUs for CPU diagnostics.

The CPU reference design uses seed 73113 (a fresh convenience-chosen stream,
distinct from prior seed 72110) and 128 independent replications per marginal
and correlation. This preserves the prior study's bounded scope; it is not
powered to prove small differences in coverage. Use paired records and report
binomial uncertainty for availability/miss rates and paired uncertainty for
continuous differences. A local snapshot assembly differs from the immutable
GPU source only in the reviewed precision implementation, with exact hashes.
Do not migrate the active GPU run to that assembly.

## Skeptical pre-execution review

The plan preserves a substantive missing baseline rather than pricing only a
convenient short trajectory. It includes complete root/repair denominators,
fresh verification and unchanged posterior rules. Its actual allocation is
positive but does not fund the broad twenty-cell or replicated comparison;
partial outcomes cannot close N5. Source isolation protects the comparator
from unrelated shared-worktree changes and the experimental precision option.

The estimator safeguard is deliberately limited: a maximum can add upward
bias, delay stopping and still underestimate true variance. It does not fix
warm-up, missing modes, infinite moments or optional-stopping coverage. The
earliest useful check is arithmetic and finite-covariance behavior, followed
by fresh calibration and existing HMC sensitivity. A change in passing rates
alone cannot promote it. These limits and the original comparison make this a
testable repair instead of a threshold relaxation. Proceed with implementation
and bounded execution; preserve any failed hypothesis and refresh the next
phase from terminal costs and results.

Implementation check: 48 focused tests pass, including actual settled-cost
accounting, price-reservation release, held CPU reserves, mutated or unfinished
settlements, all eight comparator roots, oversized forecasts, and immutable
child launch with both complete and resource-stopped outcomes. The actual
terminal ledger reconciles to 79364.235006 seconds for the comparator. No
statistical rule or numerical snapshot changed. The bounded launch command is:

```sh
systemd-run --user --unit=bayesfilter-modern-fixed-comparison-20261013-01 \
  --property=Type=exec --working-directory=/home/ubuntu/python/BayesFilter \
  --setenv=TF_FORCE_GPU_ALLOW_GROWTH=true \
  --setenv=TF_NUM_INTEROP_THREADS=2 --setenv=TF_NUM_INTRAOP_THREADS=2 \
  --setenv=OMP_NUM_THREADS=2 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  scripts/continue_modern_hmc_priced_comparison.py \
  --campaign docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/priced-comparison-01 \
  --output docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/priced-comparison-01/comparator-01 \
  --deployment-audit docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/preconfirmation-01/consumer-checks-01/deployment-terminal-audit-02.json \
  --method fixed_hmc \
  --completed-mams docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/priced-comparison-01/continuation-02 \
  --cpu-reservation-seconds 3600 \
  --plan docs/plans/bayesfilter-modern-hmc-comparator-precision-continuation-2026-10-13.md
```

## Precision implementation audit before execution

The public call chain is `HMCPosteriorAssessmentPolicy.precision` to
`assess_posterior`, `precision_report`, and `mean_precision`. The optional method
will use the existing compiled batch calculation, followed by a compiled,
explicitly finite-checked elementwise maximum. It must retain the original
per-chain estimates and boolean activation mask in the public report. A
nonfinite component cannot be hidden by TensorFlow maximum semantics; a
nonpositive plain estimate cannot grant precision. No target, chain state,
acceptance rule, or quantile estimator changes.

The calibration gets an explicit repair profile; the historical default method
inventory and continuation records keep their existing meaning. Pair the two
policies within each replication. At fixed length, report paired changes in
estimated variance and interval misses. At first passing looks, distinguish
different stopping endpoints, nonstoppers, and unavailable intervals; do not
apply the fixed-length analytic variance as the conditional stopped variance.
Use descriptive paired means and approximate 95% Student-t intervals for their
Monte Carlo uncertainty, and exact binomial intervals for availability/miss
counts. These pointwise diagnostics do not correct for multiple comparisons or
establish coverage. Constants and sample sizes retain the provenance above.

Skeptical verdict: proceed within the held reservations. The substantive risk
is added upward bias and delayed stopping, not just unavailable estimates. The
fresh calibration and archived-draw sensitivity expose both, while no default
promotion or running-GPU policy change is permitted. Oracle and public-policy
wiring tests must pass first. The active comparator has initialized GPU/XLA
but has not yet emitted its first qualification checkpoint at this audit.

The isolated CPU assembly passed 76 focused tests in 7.451938 seconds including
startup, charged to the held local-check reserve. It differs from the GPU
snapshot in exactly `bayesfilter/inference/hmc_precision.py`; the assembly
records both hashes. The checks exercise the public posterior-policy call chain,
independent batch oracles, both signs of dependence, remainders, per-chain
negative components, nonfinite values, JSON booleans, unchanged original
estimators, quantile controls and calibration continuation accounting.

Within the new 3600-second CPU reservation, cap the calibration subprocesses
and supervision at 3000 seconds in at most two attempts, with no individual
attempt above the inherited 1800-second cap. Reserve up to 500 seconds for
archived-MAMS sensitivity and 100 for final audit/overhead. This split is a
budgeting hypothesis: the original calibration took 1940.794223 seconds, so
3000 provides measured-cost headroom but does not guarantee completion. Keep
unfinished replication counts explicit if exhausted; never reduce the declared
128 replications to obtain a favorable finished prefix.

Add one bounded GPU/XLA arithmetic diagnostic for the new optional helper,
charged to the original local-check reserve (known charges 1993.574721 of
3600 seconds before this check). The comparator is the tested CPU/XLA
arithmetic on identical arrays, with positive/negative dependence, batch
remainders and explicit nonfinite components. Require identical activation
and availability masks and float64 agreement within relative 1e-10/absolute
1e-12, a diagnostic rounding allowance rather than a posterior threshold.
Verify actual GPU tensor placement, growth before initialization, source
hashes and XLA. A 180-second supervisor ceiling is a convenience bound for
this small engineering check, not a tuning timeout or performance benchmark.
Sharing the existing hardware class is allowed when memory is available;
no scientific target or active comparator rule changes. Failure would trigger
localization, not estimator promotion. This closes a concrete backend gap
without expanding the research campaign or drawing statistical conclusions.

The GPU/XLA diagnostic passed nine CPU/GPU cases plus explicit nonfinite
and activation controls, with verified memory growth and 3,035,648 peak
allocator bytes. Its supervisor cost was 7.430873 seconds from the local
reserve. This is arithmetic parity, not statistical or performance evidence.

During execution review, correct paired binary-miss uncertainty before
interpretation: a Student-t interval can collapse to zero when every observed
pair ties. Retain the running driver's raw records and pre-repair summaries;
recompute paired-miss intervals in the terminal audit from the original pair
identities. For D in {-1,0,+1}, the risk difference is p(+1)-p(-1). Give each
discordance probability a 97.5% Clopper-Pearson interval; the union bound gives
at least 95% simultaneous coverage for their rectangle, and subtracting its
opposite endpoints gives a conservative paired-difference interval. The
within-cell independent replications justify each binomial calculation; there
is no simultaneous claim across cells. Keep approximate Student-t intervals
for continuous changes, with their limitations. This repairs reporting only,
with no new simulation, changed stopping rule or new calibration seed. Mark
the old binary intervals as superseded in the terminal analysis.

The terminal interpretation must also separate existence of the mean MCSE
from existence of variance for a calibration statistic. Under Student-t(5),
E[X^4] is finite but E[X^8] is not. Thus the squared observable X^2 has finite
variance, while a batch-variance estimate for that observable is quartic in X
and need not have finite variance across replications. Do not report the
paired Student-t interval for changes in that estimated variance as valid
uncertainty. Retain its paired mean as descriptive and mark the interval
unsupported by the required moment assumption. The conclusion follows from
the t(5) density tail proportional to |x|^-6: the integral for E|X|^k has
tail power k-6 and converges only when k<5. Binary miss/availability indicators
are bounded and do not have this moment problem. Paired comparisons on
common valid outputs are conditional on that availability/stopping subset.


## Terminal CPU result and live comparator

The CPU study is complete: 128 replications in all six conditions (768
records), with all 704 carried records unchanged and 5163 method/endpoints
audited. The optional maximum implements the declared computation; no
coverage ranking or new default follows. The full interpretation is in
[the terminal result](bayesfilter-mcse-safeguard-result-2026-10-13.md).
Seventy-eight combined regressions passed, followed by 25 reporting tests
covering the added heavy-tail moment qualification. Nine GPU/XLA parity cases
and explicit nonfinite controls passed; saved MAMS first passing looks remain
unchanged. Both the API guide and included monograph chapter are updated.

Actual CPU-study cost including complete supervisor timing is 2019.235247
seconds. Settlement releases 1580.764753 seconds, held for the next refreshed
phase. The original 3600-second local reserve remains held with known charges
2011.917873 seconds, covering unmetered short inspections as well. The active
fixed-HMC allowance remains 79364.235006 seconds. Its checkpoint has recorded
two measurement observations and continued native receipts; no final
qualification or posterior outcome is claimed. Next execute the comparator's
existing remainder, audit its terminal evidence and costs, then refresh the
remaining N5/N8 scope from the actual result. Do not treat these precision
controls as completion of the master or permission to launch unfunded cells.
