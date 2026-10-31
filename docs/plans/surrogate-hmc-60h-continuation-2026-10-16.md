# Surrogate HMC: authorized 60-hour continuation

The owner approved the additional 60 GPU-worker hours on 2026-10-16. This is
the unfinished inference phase of the five-model master, not a new research
direction. LGSSM and KSC already met their recorded cumulative sampling rule;
preserve those results. The initial order was range bearing, predator–prey,
then SIR. The first range attempt has finished; PP is running and SIR follows.
The owner-authorized force-fallback repair restores range bearing after those
two phases, within its unused allocation. See
`surrogate-hmc-force-fallback-repair-2026-10-16.md`.
The prior 24-hour ledger remains historical and is not overwritten.

## Question and evidence contract

Can the fixed analytical LEDH proposal force, with the independently evaluated
Metropolis endpoint and literature prior, support healthy adaptive HMC on each
saved model? Keep N4032/K2016, GPU/XLA, FP32 with TF32 disabled, epsilon1/five
OT iterations, and the existing SIR FP64-recurrence/FP32-pairwise exception.
Range bearing uses the owner's 16 fixed flow steps and saved T20 data; the
other two use T50. All three selections have passed their own offline tuning
and mechanics. Recheck their source/data/prior identities before inference.

The comparator is the declared independent endpoint for the same saved data
and prior, not another filter's likelihood or another dataset. The primary
sampling criterion is cumulative modern R-hat <=1.01 after adaptive warmup.
Warmup is at least 2000 transitions, checked on the latest 1000 at <=1.05,
with a maximum of 10000 per chain. Retained sampling begins at 1000, extends
cumulatively, and has a maximum of 20000 per chain. There are four chains;
all retained draws are inference draws, with no extra terminal validation chain.

Numerical health checks (finite states/targets, endpoint status, divergence,
energy, movement and final-state consistency) can reject a candidate. For the
bound frozen deterministic position-only force runner, a force fallback is a
warning with recorded counts and rates, not a sampling veto. The kernel's
existing zero extension preserves the proposal's reversible, volume-preserving
construction; MH independently evaluates the declared endpoint density. Other
runner semantics retain their existing force-fallback veto. A failed health check
triggers inspection/repair and does not cancel independent model phases.
Corrupt evidence, changed target/data/source, an invalid controller, or exhausted
compute is a continuation veto. Acceptance, ESS, runtime, short mechanics and
two-transition timing estimates are explanatory; they do not replace R-hat or
establish posterior correctness. Record bulk/tail ESS and MCSE where provided.
Independent posterior agreement remains a separate unevaluated question.

This continuation does not rank methods, promote a numerical default or certify
the original nonlinear likelihood. The public position-field tuner has
conditional mechanics authority; a completed sampling status does not upgrade
that authority. The endpoint approximation and posterior coverage remain limits.
Heuristic adversaries for any later efficiency/accuracy comparison are direct
endpoint HMC (same target), a random-walk Metropolis kernel (same target, no force),
and bootstrap filtering (likelihood reference), evaluated separately near the
posterior centre, in its tails and near numerical boundaries. They are not run
or claimed defeated by this continuation, which evaluates sampler completion.

## Budget, artifacts and execution

| Model | Maximum worker time | Prepared evidence |
|---|---:|---|
| Range bearing | 8 hours | range-bearing-flow16-20261016-01/preparation |
| Predator–prey | 16 hours | cost-repair-24h-20261013-01/literature-prior-other-four-advance-01 |
| SIR | 36 hours | same four-model preparation |

The new ledger authorizes 216000 seconds in total, with at most six attempts
(three initial runs and up to three localized repairs within the same cap).
The queue launches one worker at a time, records elapsed worker time including
compilation, and applies the existing deadline/timeout controls. Unused time
remains available for a justified repair; it is not permission to change the
scientific target. The original range attempt used 6087.6444 worker seconds.
The specified repaired range retry can use at most the unused 22712.3556
seconds of its eight-hour allocation, further limited by the shared remaining
budget and attempt count. No blanket retry of numerical vetoes is authorized.
Each attempt has a fresh directory. No saved sequential state exists for these
three failed/unfinished runs, so reuse preparation and restart inference.

Output: `docs/plans/artifacts/surrogate-hmc-60h-continuation-20261016-01/`.
Preserve `manifest.json`, `budget.json`, `state.json`, `next-phase-plan.json`,
per-model master/worker logs, the public tuning archives and `hmc.json`.
The supervisor refreshes progress every 30 seconds and after each phase;
it continues to the next model after a valid candidate rejection or timeout.
`result.md` records terminal decisions, uncertainties and remaining work.
Use the existing `tftwogpu` environment and RTX5080 UUID
`GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a`; memory growth is mandatory.
Selection and worker manifests preserve data/source hashes and random seeds.

Exact launch (trusted/escalated GPU permissions):

```text
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_60h_continuation.py launch
```

The launcher starts a detached supervisor so closing the editor does not stop
the queue. Status uses the same command with `status` instead of `launch`.
No additional command approval is required by the scientific plan; platform
permissions still apply. The already-approved environment prefix is used.

The original supervisor is already running; do not repeat `launch`. Queue the
repaired range attempt once with the same executable and
`enqueue-range-bearing`. Its detached supervisor waits for the original queue
and finalized ledger before launching the existing `continue-advance` master.
The queue record is `queued-range-bearing.json`; fresh outputs are under
`range-bearing-retry-01/`. Waiting does not consume worker budget or write the
active ledger. New worker source manifests bind the repaired controller; the
already running PP process retains the code it loaded at startup.

## Assumption audit and pre-mortem

| Choice and provenance | Justification/status | Failure mode and early check |
|---|---|---|
| Existing prepared selections | Own-scope frozen settings, not cross-model transfer | Stale source/data/prior; repository selection validator must pass |
| Range 16 steps, owner choice | Local covariance repair candidate | Resonance can recur elsewhere; first long warmup health checks can veto |
| Public position-field tuner and frozen nomination config | Existing master hypothesis; no inference change | Poor geometry or expensive leapfrog choice; inspect nomination and chunk timing |
| Four dispersed starts, existing seeds | Existing reproducible sampling protocol | Incomplete exploration; cumulative diagnostics do not prove coverage |
| 8/16/36-hour allocations | Owner-approved planning caps | Two-transition L=1 pilots are noisy; report actual compilation/chunk cost |
| FP32 and explicit SIR exception | Existing reviewed scope and owner preference | Numerical guards remain active; no tolerance relaxation |

Pre-run skeptical audit: preparation/source validation passed for all three
models on 2026-10-16. The previous range attempt had only 17 minutes and ended
during compilation, not a long-run numerical rejection. The continuation
provides full inference allocations, preserves adaptive sampling, and cannot
mistake preparation success or a timeout for posterior success. The call chain
remains supervisor -> five-model `continue-advance` -> `stage_worker` ->
`surrogate_hmc_five_model.inference` -> public `tune_hmc_kernel` with its
repository runner binding and shared sequential controller. No numerical
implementation is changed. Focused queue and existing handoff tests are required
before launch. No new scientific default or learned architecture is introduced.

Post-run review must distinguish a numerical candidate failure, infrastructure
failure, budget stop and completed sampling. The strongest alternative to any
positive result is incomplete posterior exploration or endpoint approximation;
R-hat alone cannot rule either out. Record decision and inference-status tables
in the terminal note, without a superiority claim.

Implementation audit before launch: the queue calls the existing master with
one model at a time and a separate new 216000-second ledger; each phase is below
the master's 48-hour per-invocation bound. It never changes a force, endpoint,
selection or tuning configuration. Seven queue tests cover exact budget/order,
continuation after three kinds of candidate rejection, exhaustion and corrupt
controller/source handling. Together with the existing master, handoff and
flow16 checks, 45 CPU-only tests passed (two dependency deprecation warnings)
in 16.47 seconds. Log:
`artifacts/surrogate-hmc-60h-continuation-checks-20261016-01/tests.log`.
All three real prepared selections also passed the read-only handoff validator.
