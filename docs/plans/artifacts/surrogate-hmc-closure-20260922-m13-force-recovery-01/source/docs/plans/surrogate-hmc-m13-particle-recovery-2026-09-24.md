# M13 particle-resolution repair within the surrogate-HMC master

Goal: test full canonical analytical DCTR forces in surrogate HMC with a
separately declared posterior endpoint. This diagnostic addresses a force
prerequisite; it is part of the existing repair roadmap. It changes no model,
prior, observation law, hardware class or total campaign budget. LG1 remains
closed and Claude review deferred.

## Evidence contract and intent

Question: does increasing the number of particles remove the concentration
and FP32 cloud collapse observed at N256? Compare N1024 and, only if that still
fails, N4096 against the preserved controls-12 N256 result on the same nine
nominal points, calibration seed1703/design partition6/T20 and fixed controls.
Use the shared canonical target, full corrections and analytical derivatives,
FP32/TF32/XLA, verified memory growth and the repository chunk rule
(N1024->K1024; N4096->K2048). The weighted-moment arithmetic candidate remains
in place; no covariance ridge, prior or control changes are introduced here.

Primary diagnostic criterion: a candidate has all nine nominal rows valid.
This only nominates a particle resolution for a NEW scope-specific tuning
protocol with fresh calibration/validation streams. No control selection,
posterior endpoint, HMC convergence or efficiency claim follows. Failed rows
are repair evidence, not rejection of the research direction. Unflagged
nonfinite values, wrong inputs/settings/source/device, missing diagnostic
fields or exhausted reservation terminate this attempt. Preserve partial arms.
Likelihood, force and moment differences, ESS, covariance and correction flags
are explanatory. Saved-run FP32 differences remain warnings. Healthy and failed
N256 rows are distinct situations; report every row separately.

The N256 FP32, exact-input FP64 and repaired FP32 saved evaluations are arithmetic
comparators, not independently correct posterior baselines. The independent
wrapped-density posterior reference remains unqualified. A heuristic-dominance
verdict for surrogate HMC is therefore NOT_EVALUATED in this diagnostic.

## Defaults, assumptions and skeptical audit

N1024 and4096 are a geometric resolution ladder (factor4), a hypothesis motivated
by ESS1 and roundoff-scale spread; neither is an accepted default. Designs use
the same repository seed/partition, but may not be nested across N: report prefix
equality rather than assume a paired replication. One design cannot establish
statistical superiority. Fixed controls are the prior M13 bounded candidate,
a warm start only. Different N creates a new tuning scope. Inspect the complete
consumer call and preserve its manifests; a primitive success is insufficient.

Pre-mortem: the run could succeed by a lucky design or still fail because the
posterior concentration needs a better proposal. Nomination triggers independent
fresh-scope tuning, not admission. Do not tune on untouched claim seed3701.
No kernel edits are proposed in this pass. The skeptical audit passes: same
scientific target and data; explicit conditional comparisons; bounded adaptive
ladder; no proxy-to-posterior promotion; executable input and source guards.

## Budget and execution

Reserve1200CPU/3600GPU seconds from the existing global balance8650.311038CPU /
191511.712946GPU seconds, as a distinct m13-force-recovery subpass. Existing
m13-controls receipts remain charged under their original cap. No total ceiling
changes: CPU54000, GPU226800. This allocation may cover localized retries and
fresh control preparation; each launch must fit the remaining subpass balance.
First launch: review60CPU/90wall; worker500CPU/900wall; at most two particle arms.
Stop after the first all-valid arm and record its diagnostic-only status.

```text
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --stage closure --closure-stages m13-particle-recovery --output docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-force-recovery-01
```

Save source hashes, data binding, complete row diagnostics, CPU/GPU accounting,
commands, seeds, device/memory configuration and a result note under that root.
