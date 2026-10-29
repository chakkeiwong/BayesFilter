# Surrogate-HMC: additional 48-hour execution allocation

Authorized by the owner's 2026-10-18 instruction, “you have another 48 hours.
continue the execution.” This extends the existing five-model test; it does not
add a validation campaign. The executable continuation is
`scripts/run_surrogate_hmc_48h_extension.py`.

## Research intent and evidence contract

The question is whether the unfinished predator–prey, range-bearing and SIR
surrogate-HMC runs can meet the already specified adaptive sampling and health
rules within the additional allocation. Frozen LEDH supplies the surrogate
force; the Metropolis potential uses the independent-reference likelihood and
the model-specific literature prior. These references are exact Kalman for
LGSSM, the 1,201-point integration grid for KSC SV, fixed bootstrap filtering
with four replicates of 4,032 particles for predator–prey and SIR, and fixed
full-support trajectory importance sampling with four replicates of 4,032
particles for range bearing. The stochastic references define deterministic
finite-design potentials after their random designs are frozen. Passing the
sampler rule does not establish agreement with the exact-model posterior;
posterior/reference agreement remains not checked by this extension.

The comparators are the saved runs under
`artifacts/surrogate-hmc-60h-continuation-20261016-01/`. Predator–prey and SIR
resume their healthy time-limited states, preserving the kernel, random state,
warmup history and retained history. Range bearing starts a fresh HMC attempt:
the previous attempt failed the energy-error health guard and cannot be resumed
as valid inference. Its target acceptance is 0.95 and initial step size 0.005;
the existing adaptation then operates normally. The energy threshold is not
relaxed. Saved LGSSM and KSC results are preserved, with no additional chains.

| Diagnostic | Role and interpretation |
|---|---|
| Warmup readiness | Recent 1,000 transitions, maximum modern R-hat <=1.05 after at least 2,000 warmup transitions per chain; at most 10,000 |
| Retained sampling rule | Cumulative maximum rank-normalized split/folded R-hat <=1.01, assessed on the existing schedule; at most 20,000 retained draws per chain |
| Finite values, movement, target/force status, energy error | Existing health guards veto acceptance of a candidate and further sampling from its invalid state; a fresh bounded repair can remain justified |
| Broken target identity, corrupted or incompatible resume evidence | Continuation veto; repair the implementation/evidence before another launch |
| Worker allocation or global deadline exhausted | Incomplete result, not convergence or failure of the research direction |
| Acceptance, ESS, runtime and intermediate R-hat | Explanatory diagnostics; acceptance does not establish convergence |
| Posterior/reference agreement | Not checked by this extension; no exact-posterior or superiority claim |

There are four chains. Warmup is retained for diagnostics and excluded from
posterior estimates. There is no independent validation or confirmation chain.
The frozen settings remain N=4032, exact chunk K=2016, FP32 with TF32 disabled,
GPU/XLA, epsilon=1 and five OT iterations, literature model priors, and T=50
except range bearing T=20 with 16 flow steps. SIR retains its documented FP64
recurrence/FP32-pair exception. Existing prepared selections must match their
original model scope; this extension performs no cross-model tuning transfer.

The cheap reference set is the saved exact Kalman LGSSM reference, the saved
KSC integration reference and the existing model-specific filtering references.
They contextualize earlier likelihood/score evidence; this allocation runs no
new comparison against them. Conditional situations here are healthy retained
sampling (predator–prey), unfinished warmup (SIR), and a previously vetoed HMC
trajectory (range bearing). This continuation cannot establish heuristic
dominance, filter accuracy, posterior agreement or a statistically supported
ranking between methods.

## Allocation, commands and stop conditions

Use a new ledger with **172,800 worker seconds**, an absolute deadline 48 hours
after launch, and at most six attempts. Old budget is not carried into this
ledger. Run sequentially: predator–prey up to eight hours, fresh range bearing
up to eight hours, then SIR using all remaining time. Unused earlier allocations
flow to SIR. Stop each candidate on its recorded sampling rule, retained/warmup
maximum, health guard or worker budget. A candidate rejection does not block the
next queued model. Stop the supervisor for incompatible evidence or unexpected
infrastructure failure, recording the error and actual consumed time. Local
repairs may use the remaining same-contract allocation and attempt limit.

Environment: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`, RTX5080
`GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a`. Run GPU commands escalated. Workers
must establish and record memory growth before GPU initialization. Never use
the ABI-incompatible `tf-gpu` environment.

```text
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_48h_extension.py check
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_48h_extension.py launch
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_48h_extension.py status
```

Fresh output:
`docs/plans/artifacts/surrogate-hmc-48h-extension-20261018-01/run/`.
The launch manifest records commit, command, environment, GPU inventory, source
digests, prepared selections, deadline and ledger. Child manifests preserve
data identity, seeds, device policy and numerical settings. `state.json` and
`next-phase-plan.json` refresh during and between phases. Each model retains its
full master log, HMC artifacts and time charge. `result.md` records the terminal
decision and limits. Prior evidence is never overwritten.

## Default/assumption audit and pre-mortem

| Choice | Provenance/status | Failure mode and early check |
|---|---|---|
| N4032, FP32, priors, range flow16 | Owner choices; frozen target settings, not new accuracy evidence | Scope or dtype drift: prepared-selection validation and worker identity checks |
| Resume PP/SIR | Saved healthy time-limited states; reviewed repair hypothesis | Lost adaptation/history or changed kernel: focused resume equivalence and numerical-source-drift tests |
| Range acceptance .95 / step .005 | Conservative fresh HMC hypothesis, not a promoted default | Energy errors can persist or mixing can slow: unchanged chunk health checks and adaptive stopping |
| 8h/8h/remainder | Allocation choice within owner budget | SIR may consume all time before readiness: preserve chunks and report incomplete rather than promote |
| Existing R-hat/health thresholds | Previously specified sampling rules | A short retained history can pass without exact-posterior agreement: preserve the explicit nonclaim |
| Source compatibility migration | Non-numerical unfinished-warmup resume repair | Accidentally admitting numerical changes: source closure/drift denial tests; no numerical kernel changes |

The run could look successful while sampling the wrong target if scope or resume
checks were bypassed. Those checks remain active. It could fail because SIR is
slow, HMC mixes poorly or a range trajectory crosses the energy guard; those
outcomes distinguish an unfinished or rejected candidate from an invalid target.
The earliest discriminating evidence is a resumed chunk with intact history,
target identity and health diagnostics, not acceptance alone.

Skeptical audit passed before launch: the healthy unfinished-warmup repair,
history-preserving continuation, no-retuning resume path, conservative fresh-HMC
flags and source-drift denial have 214 passing focused CPU-only tests (two
dependency warnings; 43.95 seconds). Full log:
`artifacts/surrogate-hmc-48h-extension-20261018-01/focused-tests.log`.
The numerical kernel is unchanged. Baselines, scope, no-extra-validation rule,
candidate versus continuation vetoes, finite compute, and unmeasured posterior
agreement are explicit. The wrapper's command, allocation, automatic queue,
failure propagation and source-compatibility checks also passed: 11 tests in
2.41 seconds, CPU-only, recorded in
`artifacts/surrogate-hmc-48h-extension-20261018-01/new-focused-tests.log`.
The exact continuation preflight passed for all three unfinished models.
A stopped range candidate is not silently reused.

At termination, report a decision table and inference-status table. Separate
engineering correctness, recorded sampler validity and scientific interpretation.
No stochastic ranking is supported by this extension. Its strongest alternative
explanation is insufficient sampling or unfavorable geometry rather than a
filtering-method defect; posterior/reference evidence would be required for the
stronger conclusion.
