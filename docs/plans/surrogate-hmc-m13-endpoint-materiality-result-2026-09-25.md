# M13 endpoint materiality result — 2026-09-25

The same-data endpoint precision check passed for all 24 saved surrogate-HMC
proposal pairs. Including the pointwise approximate three paired-MCSE envelope,
the maximum acceptance-probability change was 0.00002688 (0.002688 percentage
points) at 4096 particles and 0.00001696 at 16384 particles. The declared local
screen was 0.01, or one percentage point. FP32 casting changed acceptance by at
most 0.000001784 (0.0001784 percentage points). These are immaterial for this
local screen; no numerical-default change is justified.

The independent endpoint evaluates the full wrapped range/bearing likelihood
on the same seed1704 observations and adds the exact Gaussian parameter prior.
It is a finite importance approximation to the likelihood, held deterministic
inside HMC. All 43 positions and seven particle/seed configurations were valid.
The fresh reference pooled 24 replicate likelihood estimates at each particle
count. Common random numbers across positions allow the paired uncertainty to
retain cancellation of shared likelihood error. At 4096 particles the largest
trajectory log-likelihood MCSE was 0.003608, while the largest paired-difference
MCSE was 0.000006295. At 16384 these were 0.001555 and 0.000003027. This explains
why individual likelihood offsets need not materially affect HMC acceptance.

Across nominal/stress positions the largest 16384-particle log-likelihood MCSE
was 0.002709 and the largest offset from the fixed seed was 0.004507. Across all
301 evaluations, the minimum importance ESS was 815.6 (at N4096), and the
largest normalized weight was 0.020575. No invalid-output guard fired.
The saved-loop versus replay log-ratio discrepancy was 0.000001866, retained
as an explanatory warning. FP32/TF32/GPU/XLA remains the force/HMC default;
FP64 here is an independent reference exception.

| Decision | Primary criterion | Veto diagnostic | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Local acceptance precision passes | All 24 envelopes below 0.01 | All 301 reference evaluations valid; identical data and source hashes | Short small-step trajectory, pointwise approximate MCSE | Same-data posterior moment and numerical-resolution comparison | Full endpoint admission, convergence, or posterior accuracy |

| Inference status | Finding |
|---|---|
| Hard veto screen | No invalid values, changed sources or data mismatch |
| Statistically supported ranking | None; this is a precision check, not a sampler ranking |
| Descriptive-only differences | Likelihood offsets, ESS, maximum weights, observed probability changes |
| Default-readiness | No new default; owner-approved FP32 preserved |
| Next evidence needed | Same-data posterior reference, scoped force controls, public tuning and chains |

Strongest alternative explanation: these unusually short trajectories stay in
a narrow healthy region. The broader nominal/stress probes help, but they do
not cover posterior mass or tails. Approximate delta-method MCSEs do not prove
coverage. A materially shifted posterior or failed same-data wider-domain
check would overturn any broader interpretation; no such interpretation is
made here.

Command:
```
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --stage closure --closure-stages m13-endpoint-materiality --output docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-endpoint-materiality-01
```
Plan: surrogate-hmc-m13-endpoint-materiality-2026-09-25.md. Complete raw
replicates, observations checksum, source snapshots, GPU memory-growth settings,
commands and receipts are under that output root. Main evidence:
`m13-endpoint-materiality/report.json`. The master ran 29 focused regressions
and verified unchanged sources. Charge56.814990 CPU /46.568207 GPU seconds.
Cumulative46995.935090292 CPU /36210.598950647 GPU; remaining7004.064909708 CPU /
190589.401049353 GPU. Global54000 CPU /226800 GPU unchanged. This completed
substage is part of the existing master, which remains incomplete.
