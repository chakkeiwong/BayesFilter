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
