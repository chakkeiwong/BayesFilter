# Prospective audit of the batch-correlation veto

The completed C01 exact-force control failed some uncertainty checks despite
healthy parameter mixing. This diagnostic asks whether the existing absolute
batch-lag threshold itself frequently rejects independent exact Gaussian draws.
It does not resample HMC, reinterpret a closed attempt, change a threshold or
promote a candidate. The goal remains a trustworthy test of surrogate HMC.

## Evidence contract and audit

Generate 512 independent replications of four chains of 2,000 independent
standard three-dimensional Gaussian draws. Evaluate the actual C01 fifteen
functions and actual `_batch_statistics` implementation at the nested
1,000/2,000 looks. The primary statistic is the fraction of replications with
any function failing the existing absolute largest-batch lag limit 0.3 at
either look. Report a Wilson 95% binomial interval. A lower endpoint above
0.05 identifies excessive false veto under the prospective 5% diagnostic
tolerance; an upper endpoint below 0.05 passes this narrow screen; otherwise
the calibration is unresolved. This tolerance is a diagnostic design choice,
not an inherited promotion criterion or authority to change the HMC protocol.

Also generate 128 replications of stationary Gaussian AR(1) draws with rho=.99
using independent innovations and an exact stationary initial state. This is
an explanatory dependence control, not a required power target. Report results
separately by generator and look; preserve negative/positive lag exceedances,
per-function rates, MCSE sensitivity and Gaussian-mean MCSE relative to known
finite-sample truth. The diagnostic does not calculate full R-hat/ESS or claim
calibration of the complete equivalence procedure.

The three elementary uncertainty comparators are constructed from the target:
IID theoretical MCSE 1/sqrt(4n); naive pooled sample-standard-deviation/sqrt(4n),
which ignores serial dependence; and exact AR(1) mean variance
`[1 + 2 sum_{k=1}^{n-1}(1-k/n) rho^k]/(4n)`. The last expression follows by
summing every pairwise covariance rho^|i-j| in each chain and averaging four
independent chains. Compare conditionally on rho and n. These are diagnostic
references, not a heuristic-dominance or sampler-efficiency test.

Ten largest batches per chain are retained at both looks because that is the
existing protocol. Consequently, increasing n does not increase the number
of observations used to estimate that lag. Within-chain centering also creates
negative finite-sample dependence. For independent Gaussian batch means, their
centered covariance has diagonal sigma²(1-1/m) and off-diagonal -sigma²/m.
This motivates measuring negative as well as positive exceedances; no
conclusion about the observed rate is made before execution.

| Choice | Provenance/status | Reason | Failure mode and early check |
|---|---|---|---|
| Same fifteen functions/two looks/0.3 threshold | Frozen C01 baseline | Isolate the actual diagnostic | Wrong ordering/scaling: call the existing function builder |
| 512 IID replications | New bounded precision choice | Worst-case binomial standard error about .022 | Unresolved interval: report it without expanding the run |
| 128 rho=.99 replications | Explanatory stress hypothesis | Strong serial dependence with a known Gaussian marginal | Wrong initialization: stationary first innovation and exact recurrence |
| Seeds [20260920, 90000+i] and [20260920, 100000+i] | Fresh deterministic seeds | Independent of HMC data and each other | Duplicates: fixed disjoint integer ranges |
| Four chains, FP64, CPU/XLA | Small diagnostic exception | No DCTR/GPU workload is needed | GPU leakage: hide devices before TensorFlow import |
| Wilson 95% interval and 5% tolerance | Diagnostic hypotheses | Quantify finite replication uncertainty | Approximate interval, limited to IID replication indicators |

Pre-mortem: even a completed run could be misleading if it reproduced a
different statistic, pooled correlated replications or treated known marginal
truth as convergence evidence. The script calls the production batch-statistic
graph, checks its first Gaussian mean against an independent standard-library
formula, records seeds and source snapshots, and limits conclusions to the
diagnostic. All data are generated independently of closed posterior draws.
No numerical guard, HMC method, target, prior, margin or tuning default changes.
Skeptical audit passes for this diagnostic scope; it does not pass any future
change to an inference rule.

## Execution, budget and stop conditions

One complete diagnostic, at most one infrastructure retry, with an aggregate
300 CPU-worker-second cap charged to P3 analysis; no GPU allocation. Stop on
nonfinite values, failed independent formula parity, source mismatch, resource
cap or corrupted report. Preserve partial progress; do not increase replication
counts after seeing a result. Current campaign charges are 8,276.791 GPU seconds
and 8,677.214 CPU-worker seconds, within unchanged 63/15-hour caps.

```bash
CUDA_VISIBLE_DEVICES=-1 BAYESFILTER_PRELOAD_CUSTOM_OP=0 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/diagnose_surrogate_hmc_mcse.py --output docs/plans/artifacts/surrogate-hmc-all-models-20260920-mcse-diagnostic-01
```

The fresh output root preserves source, per-replication diagnostics, a manifest,
accounting and report. The result note will be
`docs/plans/surrogate-hmc-mcse-diagnostic-result-2026-09-20.md`. A false-veto
finding triggers prospective redesign/calibration before another inferential
attempt; it never authorizes additional looks at closed attempts.
