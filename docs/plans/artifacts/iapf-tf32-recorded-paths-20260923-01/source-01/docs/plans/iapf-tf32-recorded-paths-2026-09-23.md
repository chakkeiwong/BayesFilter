# Phase20: fresh d2 precision comparison with complete contemporaneous traces

Phase19 exposed a reproducibility limitation:21 selected d2 TF32 likelihoods
could not be reproduced exactly in two fresh compilations; matched HLO
operations selected different cuBLAS/Triton backends. Repeating the original
warm-up did not restore them. FP64/FP32 and all36 other TF32 cases reproduced.
The old d2 paths cannot be reconstructed by assumption. Preserve that result.

The authorized repair is to record every actual path during a new comparison.
This is a harness/observability extension, not a change to the numerical filter,
precision policy, scientific target, or original phase18 result.

## Contract and reviewed design

Question: at the three fixed phase18 d2 datasets, does the shared TF32 consumer
meet the same2% mean-likelihood precision budget against FP64 and ordinary
FP32, and are influential differences localized by contemporaneous traces?
Use all four existing frozen arms: random-center/floor stress, BPF,
current-observation Gaussian and exact-future Gaussian. These are the existing
three heuristic adversaries; the stress arm is not a trained candidate.
Conditional situations are the three datasets and four guide types at
d=o=2,N64,T5. No inference transfers to other dimensions/horizons.

Use1024 fresh streams/cell/mode, adding10000000 to the original particle-seed
bases; these do not overlap phase18 or each other. Keep exact common FP32
inputs and23-bit uniform lattice. Run each precision in its own GPU process
using the same consumer and original independent R fixture checks. Save every
traced value/score/cloud/ancestor/CDF/mixture record, compressed losslessly.
Require every trace to be bitwise neutral against the untraced output in the
same compiled process. Retain optimized HLO and backend settings. Do not
require a different later compilation to reproduce these original paths.

Primary precision criterion: simultaneous approximate99% paired Student
intervals, Bonferroni over24 contrasts, entirely within[-.02,.02]. A largest
centered squared contribution above20% vetoes equivalence. Use1024 newly
generated streams only; no pooling with the phase18 selection data. Student
approximations remain a limitation, especially for rare changed paths.
These criteria can establish only conditional mean precision for these actual
compiled executions, not full distributional equivalence or compiler-wide
reproducibility. No ranking of guide quality is established by descriptive MSE.

Engineering/continuation vetoes: nonfinite values, missing/corrupt records,
failed independent FP64/FP32 controls, changed source/input identity, missing
GPU/XLA/growth evidence, any trace neutrality failure, or budget exhaustion.
Localized harness faults trigger bounded repair. Candidate precision failures
remain promotion vetoes, not reasons to abandon the research direction.

Explanatory outputs: all conditional means/RMSE/MCSE and heuristic comparisons;
first changed decisions and cloud errors for the three most influential
streams per contrast; minimum effective particle count. Selection for that
post-run explanation never changes the precision criterion or its sample.
No original-author Eq15, paper-replication, adaptive-stopping, marginal-score,
KDM, canonical LEDH, HMC or numerical-default claims.

## Default/assumption audit and skeptical review

| Choice | Provenance/status | Failure risk | Early check |
|---|---|---|---|
| d2 restriction | All original replay failures were d2 | Wrong transfer to d5/long horizons | Explicit scope; retain phase18 failures elsewhere |
|1024 streams | Bounded fresh precision diagnostic | Tail uncertainty may persist | Same20% influence veto; no forced pass |
| Existing2% margin | Frozen engineering criterion | Mistaken scientific/default promotion | Limits above; no relaxed acceptance |
| Same frozen guide/data inputs | Phase18 question | Data/CSV mismatch | Ordinary hashes and exact FP32 loader regression |
| No backend or damping change | Preserve shared numerical computation | Compiler chooses a new backend | Actual HLO saved; inference conditional on those executions |
| All contemporaneous traces | Pure observability remedy | Changed fusion or excess disk/time | Bitwise neutrality per call; free-disk check; bounded worker |

Skeptical audit: PASS. Repeating selected cases until the old answer returns
would be invalid. Fresh disjoint streams with their actual traces answer a
new, explicitly bounded precision question while preserving the original
replay veto. The same numerical runtime and R/Kalman controls prevent the
observability repair from becoming a different algorithm. These runs may
remain underpowered; that is an allowed result. Self-review only.

## Execution and budget

Artifact root `docs/plans/artifacts/iapf-tf32-recorded-paths-20260923-01/`.
Use the isolated LEDH worktree, commit89065bc6 plus snapshotted diagnostic
harness changes. Read the settled phase19 ledger for exact opening campaign
balances (about18.029 CPU hours and47.189 GPU hours, no wall expiry).
Phase caps:1800 GPU process seconds,120 CPU reporting/preflight seconds,
six attempts; reserve at most600 seconds per GPU worker. Record and charge
failed attempts. Owner continuation authorization remains in force.

GPU command: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python
docs/benchmarks/diagnose_iapf_tf32_recorded_paths.py --execute`, escalated.
Use RTX4080 UUID GPU-68251639-fe82-8f81-3ccc-2953c32e805b, growth verified before
initialization, XLA on, BLAS/OMP1, TF intra/inter2, no compiler flag overrides.
CPU reporting hides GPU and uses R or standard-library Python only.
Freeze source/inputs, preserve seed/command/environment/compiler manifests,
run R summary and terminal trace/hash checks, then update master/checkpoint.
