# q20 Gaussian-derived step-size canary

Owner request: test whether the six analytically derived initial epsilons are
sensible starting proposals on the current frozen NeuTra map. This is a bounded
development diagnostic within the existing campaign, not a posterior run.

## Question and evidence contract

Use q20/T30, four parameters, beta=1, float64, frozen map
`direct-w16-lr0.0005-r0-beta1-u512`, identity latent mass, exact transformed
value/score, and the qualified TensorFlow/TFP GPU/XLA runner in
`/tmp/BayesFilter-q20-recovery-20260922-r2`. No training or adaptation occurs.

The hypotheses are the smallest stable exact stationary N(0,I4) roots for
acceptance .7, recorded in
`artifacts/q20-gaussian-initial-epsilon-2026-09-22/result.json`:

| L | epsilon |
| --- | --- |
| 3 | 1.2595670000640444 |
| 5 | 1.3053652851902746 |
| 9 | 1.3474553766269795 |
| 13 | 1.2069251634800130 |
| 18 | 1.1997862812031914 |
| 25 | 1.2113847601121580 |

Each pair gets four transitions for each of four batched chains, from each of
two frozen banks: the original physical starts transformed by this map, and
four fresh standard-normal latent draws (the map's proposal distribution, not
posterior draws). The latter costs trivial CPU generation and is explicitly a
four-row diagnostic exception to multicore dataset generation. At each bank,
epsilon .062002709114199195 with L=3 is a contemporaneous harness control.
The original bank's saved prior results remain contextual evidence only.
Every pair resets to the same bank. Deterministic label-derived seeds share
random streams across settings within a bank; no ranking or uncertainty test
is inferred from this pairing. Save actual seeds and all traces.

Primary decision: is a proposed step immediately unsuitable at these starts?
Nonfinite proposals/scores, invalid proposed target status and all-chain or
per-chain rejection immobility veto nomination for that bank. Four transitions
are only a cheap rejection screen: zero movement alone has weak stochastic
evidence unless accompanied by near-zero Metropolis probabilities. Finite
transitions and movement permit longer measurement, not acceptance calibration.
Acceptance probabilities, signed log ratios, state/proposal displacement and
latent score residuals are explanatory. No new 0.7 acceptance band is used.
No posterior convergence, whitening, method ranking, optimum or qualified
kernel can result from this canary. Do not compute R-hat or ESS as a decision
statistic on four transitions.

Invalid starting banks are reported and not repaired by selecting favorable
draws. Shared harness/source/retained-state corruption stops the diagnostic.
Candidate-local numerical rejection does not stop the other declared pairs.
An unexpected execution exception stops for inspection. Resource loss, source
drift or the external deadline stops further work. No automatic numerical
retries, replacement steps or production default changes are authorized by
this plan. A failed canary motivates training/geometry/start diagnosis or
smaller measured proposals, not rejection of NeuTra.

## Budget, implementation and pre-run review

One worker, fourteen fixed-pair calls at most (224 proposals), one GPU with four
batched chains. Use the existing dynamic-L reusable compiled runner to share
compilation. Python schedules calls and writes diagnostics; transitions and
leapfrog iterations remain inside TensorFlow/XLA. Require incremental memory
growth before initialization and record device, TF32, dtype and source hashes.

The process cap is 1,200 seconds, an explicit 20-minute diagnostic allocation,
not a measured completion guarantee. It fits the current 1,722.618708 seconds
remaining diagnostics and 156,512.467747 seconds remaining campaign. The
inherited five-second termination grace is included in the cap. Charge actual
worker wall time to both balances using the existing Campaign supervisor,
including import, compilation and failed work. One canary attempt is permitted;
any bounded harness repair must preserve and debit the same allocation.

Historical width16 beta1 timings were 4.7312 seconds/transition at L3 and
27.5579 at L25, with resource history unknown. Linear interpolation gives about
721 seconds for the fourteen calls, excluding compilation/initialization.
This is only a cost hypothesis; the hard cap controls overspend. A partial run
must identify every unmeasured pair.

The map export predates the r2 host-only pricing repair. Check that its source
differences are exactly the two already audited master program/stages files;
load the unchanged frozen tensors with target/hash checks. Attach the current
r2 beta1 qualification. Preserve the map's original provenance. Do not edit the
numerical snapshot or import unrelated dirty main code.

Skeptical audit: the important risk is confusing a correct Gaussian formula
with evidence that this trained map is Gaussian, especially at prior starts.
The two banks expose that distinction without calling map-generated draws
posterior draws. The contemporaneous small-step control detects a broken test
setup. Four transitions cannot qualify acceptance or convergence; conclusions
are limited accordingly. The target, map, starts and random streams are
explicit; no weak comparator or efficiency ranking is substituted. Trace and
status checks, a deadline and preserved failures answer the stated question.
The plan passes this bounded pre-run review.

Run with trusted GPU permissions:

```text
/home/ubuntu/anaconda3/envs/tfgpu/bin/python /home/ubuntu/python/BayesFilter/docs/plans/artifacts/q20-gaussian-epsilon-canary-2026-09-22/run_canary.py
```

The script has supervisor and worker modes. New evidence goes to the next
versioned campaign-05 attempt directory, with a pointer and budget receipt in
`artifacts/q20-gaussian-epsilon-canary-2026-09-22/`. Existing results remain
preserved. The terminal note will separate engineering health, numerical
rejection and scientific interpretation, including a decision table and the
remaining allowance.

Setup repair: attempt 00003 stopped before any HMC transition because the new
canary supplied raw model status to a public telemetry validator; the raw
conditioning fields do not have that normalized schema. The repair uses the
adapter's existing `target_status_telemetry` method and retains raw status
separately. No numerical model/runner code changes. The existing qualified
public runner already uses this normalization. Preserve the failed script,
failure and budget receipts in `setup-failure-01`; 19.510748 seconds remains
charged. The next fresh attempt is limited by the same cumulative 1,200-second
stage budget, leaving at most 1,180.489252 seconds for this canary. This localized
harness repair changes none of the scientific criteria.
